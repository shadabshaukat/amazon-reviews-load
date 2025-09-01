import os
import sys
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# FastAPI App: Google-like homepage with unified search (light theme)
# - Product search uses PostgreSQL FTS on metadata.meta_fts (GIN indexed)
# - Reviews search uses PostgreSQL FTS on user_reviews.fts (GIN indexed)
# - Fuzzy fallback for products using pg_trgm on metadata.title (only on page 1)
# - Optional filters: min_rating (reviews), verified_only (reviews)
# - Product modal: click product or parent_asin link to view product details popup
# - Pagination: page-based (default 20 per page), Prev/Next controls like Google
# - AI Summary: Summarize user reviews per product using OCI Generative AI (RAG-style)
# -----------------------------------------------------------------------------

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_CONFIG = lambda: {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432)),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
    "dbname": os.getenv("PGDATABASE", "postgres"),
}

MIN_POOL = int(os.getenv("APP_DB_MIN_POOL", "1"))
MAX_POOL = int(os.getenv("APP_DB_MAX_POOL", "10"))

app = FastAPI(title="Amazon Reviews Search", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_pool: Optional[ThreadedConnectionPool] = None


def _init_pool():
    global _pool
    if _pool is None:
        cfg = DB_CONFIG()
        _pool = ThreadedConnectionPool(
            MIN_POOL, MAX_POOL,
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            dbname=cfg["dbname"],
            cursor_factory=RealDictCursor,
        )
        logging.info("PostgreSQL connection pool initialized.")


@contextmanager
def get_conn():
    global _pool
    if _pool is None:
        _init_pool()
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)


@app.on_event("startup")
def on_startup():
    _init_pool()


@app.on_event("shutdown")
def on_shutdown():
    global _pool
    if _pool:
        _pool.closeall()
        logging.info("PostgreSQL connection pool closed.")


@app.get("/health", tags=["ops"])
def health():
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return {"status": "ok"}
    except Exception as e:
        logging.exception("Health check failed")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


# -----------------------------------------------------------------------------
# HTML Frontend (single file, light theme, Google-like) with Pagination controls
# -----------------------------------------------------------------------------
HOMEPAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>SEARCH</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #f8fafc;       /* light background */
      --fg: #111827;       /* foreground text */
      --muted: #6b7280;    /* muted text */
      --accent: #2563eb;   /* blue */
      --card: #ffffff;     /* cards */
      --border: #e5e7eb;   /* border */
      --ring: rgba(37, 99, 235, 0.15);
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; background: var(--bg); color: var(--fg); font-family: system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji"; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .container { min-height: 100%; display: flex; flex-direction: column; }
    main { flex: 1; display: flex; align-items: center; justify-content: center; padding: 24px; }
    .search-card { width: 100%; max-width: 900px; background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 28px; box-shadow: 0 10px 28px rgba(0,0,0,0.08); }
    .brand { text-align: center; font-size: 82px; font-weight: 500; letter-spacing: 6px; margin-bottom: 18px; color: var(--fg); line-height: 1; font-family: 'Product Sans','Google Sans','Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
    .search-row { display: flex; gap: 10px; align-items: center; margin: 0 auto; }
    .search-input { flex: 1; padding: 14px 16px; font-size: 18px; border-radius: 999px; border: 1px solid var(--border); background: #fff; color: var(--fg); outline: none; }
    .search-input:focus { border-color: var(--accent); box-shadow: 0 0 0 4px var(--ring); }
    .search-btn { padding: 12px 18px; border-radius: 999px; border: 1px solid var(--border); background: #f3f4f6; color: var(--fg); cursor: pointer; font-weight: 600; }
    .search-btn:hover { background: #e5e7eb; }
    .filters { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 14px; color: var(--muted); }
    .select, .number { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--border); background: #fff; color: var(--fg); }
    .results { margin-top: 24px; display: grid; gap: 18px; }
    .section-title { font-size: 14px; color: var(--muted); margin: 16px 4px 4px; text-transform: uppercase; letter-spacing: 1px; }
    .card { background: #fff; border: 1px solid var(--border); border-radius: 12px; padding: 16px; display: grid; gap: 8px; }
    .title { font-size: 18px; font-weight: 700; color: var(--fg); }
    .subtitle { font-size: 14px; color: var(--muted); }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f3f4f6; color: var(--muted); font-size: 12px; margin-right: 6px; border: 1px solid var(--border); }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .rank { color: var(--accent); font-weight: 700; }
    .grid { display: grid; grid-template-columns: 120px 1fr; gap: 14px; align-items: start; }
    .thumb { width: 120px; height: 120px; object-fit: contain; border: 1px solid var(--border); border-radius: 8px; background: #fff; }
    .actions { display: flex; gap: 8px; }
    .link-btn { padding: 6px 10px; border-radius: 8px; border: 1px solid var(--border); background: #f9fafb; color: var(--fg); font-size: 12px; font-weight: 600; text-decoration: none; cursor: pointer; }
    .link-btn:hover { background: #eef2f7; }
    /* Summarize with AI button - eye-catching */
    .link-btn.summ-btn {
      background: linear-gradient(135deg, #2563eb, #0ea5e9);
      color: #fff;
      border: 1px solid rgba(37, 99, 235, 0.5);
      box-shadow: 0 6px 14px rgba(37, 99, 235, 0.25);
    }
    .link-btn.summ-btn:hover {
      filter: brightness(1.05);
      box-shadow: 0 8px 18px rgba(37, 99, 235, 0.35);
    }
    /* Summary box - visually appealing */
    .summ-box {
      display: none;
      margin-top: 12px;
      background: linear-gradient(180deg, #f0f7ff, #ffffff);
      border: 1px solid rgba(37, 99, 235, 0.25);
      border-left: 4px solid #2563eb;
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.06);
      padding: 14px;
    }
    #summaryText {
      white-space: pre-line;
      font-size: 15px;
      line-height: 1.55;
      color: #0f172a;
    }
    /* Key themes chips */
    .themes {
      display: none;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }
    .chip {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid;
    }
    .chip.positive {
      background: #dcfce7;
      border-color: #16a34a;
      color: #14532d;
    }
    .chip.negative {
      background: #fee2e2;
      border-color: #ef4444;
      color: #7f1d1d;
    }
    footer { text-align: center; font-size: 12px; color: var(--muted); padding: 16px; }

    .empty { text-align: center; color: var(--muted); padding: 16px; }

    /* Modal */
    .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.35); display: none; align-items: center; justify-content: center; z-index: 50; }
    .modal { width: min(720px, 96vw); max-height: 90vh; overflow: auto; background: #fff; color: var(--fg); border: 1px solid var(--border); border-radius: 14px; box-shadow: 0 20px 40px rgba(0,0,0,0.2); }
    .modal-header { display: flex; align-items: center; justify-content: space-between; padding: 14px 16px; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: #fff; }
    .modal-content { padding: 16px; display: grid; gap: 10px; }
    .close { border: 1px solid var(--border); background: #f3f4f6; border-radius: 8px; padding: 6px 10px; cursor: pointer; }
    .modal-grid { display: grid; grid-template-columns: 180px 1fr; gap: 16px; }
    .modal-thumb { width: 180px; height: 180px; object-fit: contain; border: 1px solid var(--border); border-radius: 8px; background: #fff; }
    .list { margin: 0; padding-left: 20px; }

    /* Pager */
    .pager { display: none; gap: 12px; align-items: center; justify-content: center; margin: 12px 0 6px; }
    .pager .btn { padding: 8px 12px; border-radius: 10px; border: 1px solid var(--border); background: #f3f4f6; color: var(--fg); font-weight: 600; cursor: pointer; }
    .pager .btn:hover { background: #e5e7eb; }
    .pager .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .pager .label { font-size: 14px; color: var(--muted); }
  </style>
</head>
<body>
  <div class="container">
    <main>
      <div class="search-card">
        <div class="brand">
          <a id="brandLink" href="#" style="text-decoration:none; color:inherit;">
            <span style="color:#4285F4">S</span><span style="color:#DB4437">E</span><span style="color:#F4B400">A</span><span style="color:#4285F4">R</span><span style="color:#0F9D58">C</span><span style="color:#DB4437">H</span>
          </a>
        </div>
        <form id="searchForm">
          <div class="search-row">
            <input id="q" class="search-input" type="text" placeholder="Search products and reviews..." autocomplete="off" />
            <button class="search-btn" type="submit">Search</button>
          </div>
          <div class="filters">
            <select id="type" class="select">
              <option value="all" selected>All</option>
              <option value="products">Products</option>
              <option value="reviews">Reviews</option>
            </select>
            <input id="limit" class="number" type="number" min="1" max="100" value="20" title="Limit" />
            <input id="minRating" class="number" type="number" min="0" max="5" step="0.1" placeholder="Min Review Rating" title="Min review rating" />
            <label style="display:flex; align-items:center; gap:8px; color: var(--muted);">
              <input id="verifiedOnly" type="checkbox" /> Verified only
            </label>
          </div>
        </form>
        <div id="results" class="results"></div>
        <div id="pager" class="pager">
          <button id="prevBtn" class="btn" type="button">Prev</button>
          <span id="pageLabel" class="label">Page 1</span>
          <button id="nextBtn" class="btn" type="button">Next</button>
        </div>
      </div>
    </main>
    <footer>Build with ❤️ by Shadab Mohammad and Powered by PostgreSQL Full-Text Search + FastAPI</footer>
  </div>

  <!-- Modal for product details -->
  <div id="modalOverlay" class="modal-overlay" role="dialog" aria-modal="true">
    <div class="modal">
      <div class="modal-header">
        <div style="font-weight:700">Product Details</div>
        <button id="modalClose" class="close" type="button">Close</button>
      </div>
      <div id="modalBody" class="modal-content"></div>
    </div>
  </div>

  <script>
    const form = document.getElementById('searchForm');
    const results = document.getElementById('results');
    const overlay = document.getElementById('modalOverlay');
    const modalBody = document.getElementById('modalBody');
    const modalClose = document.getElementById('modalClose');
    const pagerDiv = document.getElementById('pager');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const pageLabel = document.getElementById('pageLabel');
    const brandLink = document.getElementById('brandLink');

    let currentPage = 1;

    function goHome() {
      try { history.pushState({}, '', '/'); } catch (e) {}
      const qEl = document.getElementById('q');
      const typeEl = document.getElementById('type');
      const limitEl = document.getElementById('limit');
      const minRatingEl = document.getElementById('minRating');
      const verifiedEl = document.getElementById('verifiedOnly');

      if (qEl) qEl.value = '';
      if (typeEl) typeEl.value = 'all';
      if (limitEl) limitEl.value = '20';
      if (minRatingEl) minRatingEl.value = '';
      if (verifiedEl) verifiedEl.checked = false;

      currentPage = 1;
      results.innerHTML = '';
      pagerDiv.style.display = 'none';
      if (qEl) qEl.focus();
    }

    function esc(str) {
      return String(str == null ? '' : str).replace(/[&<>"']/g, s => ({'&':'&','<':'<','>':'>','"':'"',"'":'&#39;'}[s]));
    }

    function pickImageUrl(p) {
      // Prefer image_url if present, else try first images entry
      if (p.image_url) return p.image_url;
      const imgs = p.images || [];
      if (Array.isArray(imgs) && imgs.length) {
        const x = imgs[0] || {};
        return x.thumb || x.large || '';
      }
      return '';
    }

    function productCard(p) {
      const img = pickImageUrl(p);
      return `
        <div class="card">
          <div class="grid">
            <div>
              ${img ? `<img src="${esc(img)}" alt="Product" class="thumb" />` : `<div class="thumb" style="display:flex;align-items:center;justify-content:center;color:#9ca3af;">No Image</div>`}
            </div>
            <div>
              <div class="title"><a href="#" class="product-link" data-asin="${esc(p.parent_asin)}">${esc(p.title || 'Untitled')}</a></div>
              <div class="subtitle">ASIN: ${esc(p.parent_asin)} · ${esc(p.main_category || '—')} · ${esc(p.store || '—')}</div>
              <div class="row" style="margin-top:6px;">
                <span class="badge">Price: ${p.price == null ? '—' : '$' + esc(p.price)}</span>
                <span class="badge">Avg Rating: ${p.average_rating == null ? '—' : esc(p.average_rating)}</span>
                <span class="badge">Ratings: ${p.rating_number == null ? '—' : esc(p.rating_number)}</span>
                <span class="badge">Rank: <span class="rank">${p.rank?.toFixed ? p.rank.toFixed(3) : esc(p.rank)}</span></span>
              </div>
              <div class="actions" style="margin-top:10px;">
                <a href="#" class="link-btn product-link" data-asin="${esc(p.parent_asin)}">View details</a>
              </div>
            </div>
          </div>
        </div>
      `;
    }

    function renderProducts(items) {
      if (!items || !items.length) return '<div class="empty">No product results.</div>';
      return items.map(productCard).join('');
    }

    function renderReviews(items) {
      if (!items || !items.length) return '<div class="empty">No review results.</div>';
      return items.map(r => `
        <div class="card">
          <div class="title">${esc(r.review_title || r.title || '(no title)')}</div>
          <div class="subtitle">Product: <a href="#" class="product-link" data-asin="${esc(r.parent_asin)}">${esc(r.parent_asin)}</a> · ASIN: ${esc(r.asin || '—')} · ${esc(r.ts || '—')}</div>
          <div>${esc((r.review_text || '').slice(0, 320))}${(r.review_text || '').length > 320 ? '…' : ''}</div>
          <div class="row" style="margin-top:6px;">
            <span class="badge">Rating: ${r.rating == null ? '—' : esc(r.rating)}</span>
            <span class="badge">Helpful: ${r.helpful_vote == null ? '—' : esc(r.helpful_vote)}</span>
            <span class="badge">Verified: ${r.verified_purchase ? 'Yes' : 'No'}</span>
            <span class="badge">Rank: <span class="rank">${r.rank?.toFixed ? r.rank.toFixed(3) : esc(r.rank)}</span></span>
          </div>
        </div>
      `).join('');
    }

    function render(data, type) {
      results.innerHTML = '';
      const blocks = [];
      if (data.suggestion_applied && data.used_query) {
        blocks.push(`<div class="section-title">Showing results for "${esc(data.used_query)}"${data.original_query ? ' (searched for "' + esc(data.original_query) + '")' : ''}.</div>`);
      }
      if (type === 'products' || type === 'all') {
        blocks.push(`<div class="section-title">Products</div>`);
        blocks.push(renderProducts(data.products || []));
      }
      if (type === 'reviews' || type === 'all') {
        blocks.push(`<div class="section-title">Reviews</div>`);
        blocks.push(renderReviews(data.reviews || []));
      }
      results.innerHTML = blocks.join('');
      updatePager(type, data);
    }

    function updatePager(type, data) {
      // determine if there is a next page based on response
      const page = data.page || 1;
      const hasMoreProducts = !!data.has_more_products;
      const hasMoreReviews = !!data.has_more_reviews;

      let hasMore;
      if (type === 'products') {
        hasMore = hasMoreProducts;
      } else if (type === 'reviews') {
        hasMore = hasMoreReviews;
      } else {
        hasMore = hasMoreProducts || hasMoreReviews;
      }

      // Show pager if either has previous or has next
      if (page > 1 || hasMore) {
        pagerDiv.style.display = 'flex';
      } else {
        pagerDiv.style.display = 'none';
      }

      prevBtn.disabled = page <= 1;
      nextBtn.disabled = !hasMore;
      pageLabel.textContent = 'Page ' + page;
    }

    function collectInputs() {
      const q = document.getElementById('q').value.trim();
      const type = document.getElementById('type').value;
      const limit = parseInt(document.getElementById('limit').value || '20', 10);
      const minRating = document.getElementById('minRating').value;
      const verifiedOnly = document.getElementById('verifiedOnly').checked;
      return { q, type, limit, minRating, verifiedOnly };
    }

    async function fetchAndRender() {
      const { q, type, limit, minRating, verifiedOnly } = collectInputs();

      if (!q) {
        results.innerHTML = '<div class="empty">Enter a query to search.</div>';
        pagerDiv.style.display = 'none';
        return;
      }

      const params = new URLSearchParams({ q, type, limit: String(limit), page: String(currentPage) });
      if (minRating) params.set('min_rating', String(minRating));
      if (verifiedOnly) params.set('verified_only', 'true');

      results.innerHTML = '<div class="empty">Searching…</div>';

      try {
        const res = await fetch('/api/search?' + params.toString());
        if (!res.ok) {
          const txt = await res.text();
          results.innerHTML = '<div class="empty">Error: ' + esc(txt) + '</div>';
          pagerDiv.style.display = 'none';
          return;
        }
        const data = await res.json();
        render(data, type);
      } catch (e) {
        results.innerHTML = '<div class="empty">Network error.</div>';
        pagerDiv.style.display = 'none';
      }
    }

    async function doSearch(ev) {
      ev.preventDefault();
      currentPage = 1;
      await fetchAndRender();
    }

    async function goPrev() {
      if (currentPage > 1) {
        currentPage -= 1;
        await fetchAndRender();
      }
    }

    async function goNext() {
      currentPage += 1;
      await fetchAndRender();
    }

    async function fetchProduct(asin) {
      try {
        const res = await fetch('/api/product/' + encodeURIComponent(asin));
        if (!res.ok) {
          throw new Error('Product not found');
        }
        return await res.json();
      } catch (e) {
        return null;
      }
    }

    function featureList(features) {
      if (!Array.isArray(features) || features.length === 0) return '';
      const top = features.slice(0, 6).map(f => `<li>${esc(typeof f === 'string' ? f : JSON.stringify(f))}</li>`).join('');
      return `<ul class="list">${top}</ul>`;
    }

    function productModalHTML(p) {
      const img = pickImageUrl(p);
      return `
        <div class="modal-grid">
          <div>
            ${img ? `<img src="${esc(img)}" alt="Product" class="modal-thumb" />` : `<div class="modal-thumb" style="display:flex;align-items:center;justify-content:center;color:#9ca3af;">No Image</div>`}
          </div>
          <div>
            <div class="title">${esc(p.title || 'Untitled')}</div>
            <div class="subtitle" style="margin-top:4px;">ASIN: ${esc(p.parent_asin)} · ${esc(p.main_category || '—')} · ${esc(p.store || '—')}</div>
            <div class="row" style="margin-top:8px;">
              <span class="badge">Price: ${p.price == null ? '—' : '$' + esc(p.price)}</span>
              <span class="badge">Avg Rating: ${p.average_rating == null ? '—' : esc(p.average_rating)}</span>
              <span class="badge">Ratings: ${p.rating_number == null ? '—' : esc(p.rating_number)}</span>
            </div>
            <div style="margin-top:10px;">
              ${featureList(p.features)}
            </div>
            <div class="actions" style="margin-top:10px;">
                <a href="#" class="link-btn summ-btn" data-asin="${esc(p.parent_asin)}">Summarize with AI</a>
            </div>
            <div id="summaryBox" class="card summ-box">
              <div class="subtitle" style="margin-bottom:6px;">Customers say …</div>
              <div id="summaryText"></div>
              <div class="subtitle" style="margin-top:10px;"></div>
              <div id="summaryThemes" class="themes"></div>
            </div>
          </div>
        </div>
      `;
    }

    async function openProductModal(asin) {
      const data = await fetchProduct(asin);
      if (!data) {
        modalBody.innerHTML = '<div class="empty">Product not found.</div>';
      } else {
        modalBody.innerHTML = productModalHTML(data);
      }
      overlay.style.display = 'flex';
    }

    function closeModal() {
      overlay.style.display = 'none';
    }

    // Summarize handler (delegation)
    document.addEventListener('click', async (e) => {
      const a = e.target.closest('a.summ-btn');
      if (a) {
        e.preventDefault();
        const asin = a.getAttribute('data-asin');
        const box = document.getElementById('summaryBox');
        const text = document.getElementById('summaryText');
        if (box && text) {
          box.style.display = 'block';
          text.textContent = 'Summarizing with AI…';
          const themesEl = document.getElementById('summaryThemes');
          if (themesEl) { themesEl.innerHTML = ''; themesEl.style.display = 'none'; }
          try {
            const res = await fetch('/api/summarize/' + encodeURIComponent(asin));
            const data = await res.json();
            if (!res.ok) throw new Error(data && data.error ? data.error : 'Failed');
            text.textContent = data.summary || 'No summary available.';
            const themesEl2 = document.getElementById('summaryThemes');
            if (themesEl2) {
              if (Array.isArray(data.aspects) && data.aspects.length) {
                const chips = data.aspects.slice(0, 8).map(a => {
                  const label = esc(a.label || '');
                  const sentiment = (a.sentiment || 'positive').toLowerCase();
                  const chipClass = sentiment === 'negative' ? 'chip negative' : 'chip positive';
                  const emoji = sentiment === 'negative' ? '' : '✅ ';
                  return `<span class="${chipClass}">${emoji}${label}</span>`;
                });
                themesEl2.innerHTML = chips.join(' ');
                themesEl2.style.display = 'flex';
              } else {
                themesEl2.innerHTML = '';
                themesEl2.style.display = 'none';
              }
            }
          } catch (err) {
            text.textContent = 'Error generating summary.';
          }
        }
      }
    });

    form.addEventListener('submit', doSearch);
    modalClose.addEventListener('click', closeModal);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) closeModal(); });

    // Event delegation for product links (in both Products and Reviews sections)
    document.addEventListener('click', (e) => {
      const a = e.target.closest('a.product-link');
      if (a) {
        e.preventDefault();
        const asin = a.getAttribute('data-asin');
        if (asin) openProductModal(asin);
      }
    });

    prevBtn.addEventListener('click', goPrev);
    nextBtn.addEventListener('click', goNext);
    if (brandLink) {
      brandLink.addEventListener('click', (e) => { e.preventDefault(); goHome(); });
    }

    document.getElementById('q').focus();
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, tags=["ui"])
def homepage():
    return HTMLResponse(HOMEPAGE_HTML)


# -----------------------------------------------------------------------------
# API Helpers
# -----------------------------------------------------------------------------
def _rows_to_list(cur) -> List[Dict[str, Any]]:
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def _paginate_slice(items: List[Dict[str, Any]], limit: int) -> Tuple[List[Dict[str, Any]], bool]:
    # items length is at most limit+1
    has_more = len(items) > limit
    return items[:limit], has_more


def _suggest_token(conn, token: str) -> Optional[str]:
    """
    Suggest a corrected token using pg_trgm similarity against a small,
    index-backed candidate set from metadata.title or user_reviews.title.
    Uses the trigram index on metadata.title for performance.
    """
    if not token or len(token) < 4:
        return None
    with conn.cursor() as cur:
        # Try candidates from metadata.title first (uses idx_metadata_title_trgm)
        sql_meta = """
            WITH candidates AS (
                SELECT title
                FROM metadata
                WHERE title %% %s
                ORDER BY similarity(title, %s) DESC
                LIMIT 50
            ),
            words AS (
                SELECT lower(regexp_split_to_table(title, '\\W+')) AS w
                FROM candidates
            )
            SELECT w
            FROM words
            WHERE length(w) >= 3
            ORDER BY similarity(w, %s) DESC
            LIMIT 1
        """
        cur.execute(sql_meta, (token, token, token))
        row = cur.fetchone()
        if row and row.get("w"):
            return row["w"]

        # Fallback: source from user_reviews.title
        sql_rev = """
            WITH candidates AS (
                SELECT title
                FROM user_reviews
                WHERE title %% %s
                ORDER BY similarity(title, %s) DESC
                LIMIT 50
            ),
            words AS (
                SELECT lower(regexp_split_to_table(title, '\\W+')) AS w
                FROM candidates
            )
            SELECT w
            FROM words
            WHERE length(w) >= 3
            ORDER BY similarity(w, %s) DESC
            LIMIT 1
        """
        cur.execute(sql_rev, (token, token, token))
        row = cur.fetchone()
        if row and row.get("w"):
            return row["w"]
    return None


def _autocorrect_query(conn, q: str) -> Tuple[str, bool]:
    """
    Build a corrected query by suggesting replacements for likely misspelled tokens.
    Only alphanumeric tokens with length >= 4 are considered.
    """
    if not q:
        return q, False

    parts = re.split(r"(\W+)", q)  # keep delimiters
    changed = False
    out: List[str] = []
    for part in parts:
        if re.fullmatch(r"[A-Za-z0-9]+", part or "") and len(part) >= 4:
            sug = _suggest_token(conn, part.lower())
            if sug and sug != part.lower():
                out.append(sug)
                changed = True
            else:
                out.append(part)
        else:
            out.append(part)
    corrected = "".join(out)
    return corrected, changed


def _product_search(conn, q: str, limit: int, offset: int) -> Tuple[List[Dict[str, Any]], bool]:
    fetch_limit = limit + 1
    with conn.cursor() as cur:
        # Primary: FTS on metadata.meta_fts
        sql = """
            SELECT
                parent_asin,
                title,
                main_category,
                store,
                price,
                average_rating,
                rating_number,
                COALESCE((images->0->>'thumb'), (images->0->>'large')) AS image_url,
                ts_rank_cd(meta_fts, plainto_tsquery('english', %s)) AS rank
            FROM metadata
            WHERE meta_fts @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s OFFSET %s
        """
        cur.execute(sql, (q, q, fetch_limit, offset))
        products, has_more = _paginate_slice(_rows_to_list(cur), limit)

        # Fallback: if no FTS results on first page only, try fuzzy title match using pg_trgm
        if not products and offset == 0:
            sql_fuzzy = """
                SELECT
                    parent_asin,
                    title,
                    main_category,
                    store,
                    price,
                    average_rating,
                    rating_number,
                    COALESCE((images->0->>'thumb'), (images->0->>'large')) AS image_url,
                    similarity(title, %s) AS rank
                FROM metadata
                WHERE title ILIKE '%%' || %s || '%%'
                ORDER BY similarity(title, %s) DESC
                LIMIT %s OFFSET %s
            """
            cur.execute(sql_fuzzy, (q, q, q, fetch_limit, offset))
            products, has_more = _paginate_slice(_rows_to_list(cur), limit)

        return products, has_more


def _review_search(
    conn,
    q: str,
    limit: int,
    offset: int,
    min_rating: Optional[float],
    verified_only: Optional[bool]
) -> Tuple[List[Dict[str, Any]], bool]:
    fetch_limit = limit + 1
    with conn.cursor() as cur:
        where = ["r.fts @@ plainto_tsquery('english', %s)"]
        params: List[Any] = [q, q]  # for rank and match

        if min_rating is not None:
            where.append("r.rating >= %s")
            params.append(min_rating)
        if verified_only:
            where.append("r.verified_purchase = TRUE")

        where_sql = " AND ".join(where)
        sql = f"""
            SELECT
                r.review_id,
                r.parent_asin,
                r.asin,
                r.title AS review_title,
                r.review_text,
                r.rating,
                r.helpful_vote,
                r.verified_purchase,
                r.ts,
                ts_rank_cd(r.fts, plainto_tsquery('english', %s)) AS rank
            FROM user_reviews r
            WHERE {where_sql}
            ORDER BY rank DESC
            LIMIT %s OFFSET %s
        """
        params.extend([fetch_limit, offset])
        cur.execute(sql, params)
        reviews, has_more = _paginate_slice(_rows_to_list(cur), limit)
        return reviews, has_more


# -----------------------------------------------------------------------------
# AI Summarization Utilities (RAG-ish over review embeddings)
# -----------------------------------------------------------------------------
def _parse_vector_cell(val: Any) -> Optional[List[float]]:
    if val is None:
        return None
    if isinstance(val, list):
        try:
            return [float(x) for x in val]
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(",")]
        out = []
        for p in parts:
            if not p:
                continue
            try:
                out.append(float(p))
            except Exception:
                return None
        return out
    return None


def _compute_centroid(vectors: List[List[float]]) -> Optional[List[float]]:
    if not vectors:
        return None
    dim = len(vectors[0])
    acc = [0.0] * dim
    n = 0
    for v in vectors:
        if v is None or len(v) != dim:
            continue
        for i in range(dim):
            acc[i] += float(v[i])
        n += 1
    if n == 0:
        return None
    return [x / n for x in acc]


def _vector_to_sql_literal(vec: List[float]) -> str:
    # Converts a Python list to pgvector literal: "[v1,v2,...]"
    return "[" + ", ".join(f"{float(x):.6f}" for x in vec) + "]"


def _get_centroid_for_parent(conn, parent_asin: str, sample_limit: int = 1000) -> Optional[List[float]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT embedding
            FROM user_reviews
            WHERE parent_asin = %s
              AND embedding IS NOT NULL
              AND review_text IS NOT NULL
            LIMIT %s
            """,
            (parent_asin, sample_limit),
        )
        rows = cur.fetchall()
    vecs: List[List[float]] = []
    for row in rows:
        emb = row.get("embedding")
        vec = _parse_vector_cell(emb)
        if vec:
            vecs.append(vec)
    return _compute_centroid(vecs)


def _select_similar_reviews(conn, parent_asin: str, query_vec_sql: str, candidate_limit: int = 200) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        sql = """
            SELECT
                review_id,
                review_text,
                rating,
                helpful_vote,
                verified_purchase,
                ts,
                (embedding <=> %s::vector) AS dist
            FROM user_reviews
            WHERE parent_asin = %s
              AND review_text IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        cur.execute(sql, (query_vec_sql, parent_asin, query_vec_sql, candidate_limit))
        return _rows_to_list(cur)


def _choose_evidence(cands: List[Dict[str, Any]], top_k: int = 40) -> List[Dict[str, Any]]:
    # Simple rank by (similarity + helpfulness + verified); dist is lower=better
    for r in cands:
        dist = float(r.get("dist") or 0.0)
        sim = 1.0 - dist
        helpful = float(r.get("helpful_vote") or 0.0)
        verified = 1.0 if r.get("verified_purchase") else 0.0
        r["_score"] = sim + 0.15 * (helpful if helpful > 0 else 0.0) + 0.2 * verified
    cands.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    picked: List[Dict[str, Any]] = []
    seen_snips = set()
    for r in cands:
        text = (r.get("review_text") or "").strip()
        if not text:
            continue
        snip = text[:140].lower()
        if snip in seen_snips:
            continue
        seen_snips.add(snip)
        picked.append(r)
        if len(picked) >= top_k:
            break
    return picked


def _build_summary_prompt(parent_asin: str, title: Optional[str], evid: List[Dict[str, Any]]) -> str:
    header = f"Summarize the following customer reviews for product: {title or ''} (ASIN: {parent_asin})."
    rules = (
        "Write 4-5 sentences. Start with the exact phrase: \"Customers say ...\".\n"
        "Summarize common themes on quality, performance, reliability, ease of use, value, and any frequent issues.\n"
        "Use only the provided reviews. Do not include opinions or facts not present in the reviews.\n"
        "Be balanced and concise. Avoid personal data. Do not include URLs."
    )
    lines = []
    for r in evid:
        rating = r.get("rating")
        helpful = r.get("helpful_vote")
        verified = "Verified" if r.get("verified_purchase") else ""
        txt = (r.get("review_text") or "").replace("\n", " ").strip()
        if len(txt) > 600:
            txt = txt[:600] + "…"
        prefix = []
        if rating is not None:
            prefix.append(f"Rating {rating:.1f}")
        if helpful:
            prefix.append(f"Helpful {helpful}")
        if verified:
            prefix.append(verified)
        meta = " | ".join(prefix)
        if meta:
            lines.append(f"- [{meta}] {txt}")
        else:
            lines.append(f"- {txt}")
    evidence = "\n".join(lines[:80])  # hard cap
    prompt = f"{header}\n\n{rules}\n\nReviews:\n{evidence}\n\nSummary:"
    return prompt


def _extract_key_themes(summary: str) -> List[Dict[str, str]]:
    """
    Derive key positive/negative themes from an LLM summary using lightweight heuristics.
    Returns a list of {label, sentiment} where sentiment is 'positive' or 'negative'.
    """
    if not summary:
        return []

    text = summary.lower()

    # Aspect lexicon with simple keyword triggers
    aspects = {
        "Quality": ["quality", "build quality", "well-made", "construction", "craftsmanship", "finish", "materials"],
        "Effectiveness": ["effective", "effectiveness", "works", "working", "does the job", "helped", "improved", "improvement"],
        "Performance": ["performance", "speed", "fast", "quick", "snappy", "lag", "slow"],
        "Reliability": ["reliable", "reliability", "durable", "stopped working", "broke", "fails", "failure", "lasted"],
        "Ease of use": ["easy to use", "ease of use", "user-friendly", "setup", "install", "instructions", "hard to use"],
        "Value": ["value", "worth", "price", "affordable", "expensive", "overpriced", "cheap"],
        "Design": ["design", "look", "style", "compact", "foldable"],
        "Size": ["size", "small", "big", "heavy", "lightweight", "weight"],
        "Battery": ["battery", "battery life", "charge", "charging"],
        "Comfort": ["comfortable", "comfort", "fit"],
        "Sound": ["sound", "audio", "volume", "noise"],
        "Connectivity": ["connectivity", "connect", "bluetooth", "wi-fi", "wifi"],
        "Packaging": ["packaging", "package", "box", "sealed"],
        "Taste": ["taste", "flavor", "fishy", "smell", "odor", "burp", "burps"],
    }

    positive_cues = [
        "good", "great", "excellent", "love", "like", "works", "fast", "quick", "reliable", "effective",
        "no issues", "easy", "well", "worth", "value", "compact", "foldable", "improved", "happy",
        "satisfied", "recommend", "awesome", "perfect"
    ]
    negative_cues = [
        "bad", "poor", "terrible", "slow", "problem", "issue", "issues", "broke", "broken", "stopped",
        "doesn't", "didn't", "hard", "difficult", "expensive", "overpriced", "cheap", "flimsy",
        "unreliable", "return", "refund", "disappoint", "waste", "faulty", "defective", "noisy",
        "fishy", "smell", "burp", "burps", "too big", "too small", "heavy", "hot", "overheat", "overheats"
    ]

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    score: Dict[str, Dict[str, int]] = {label: {"pos": 0, "neg": 0} for label in aspects}
    total_pos = 0
    total_neg = 0

    def contains_any(s: str, terms: List[str]) -> bool:
        for t in terms:
            if re.search(r'\b' + re.escape(t) + r'\b', s):
                return True
        return False

    for sent in sentences:
        if not sent.strip():
            continue
        has_pos = contains_any(sent, positive_cues)
        has_neg = contains_any(sent, negative_cues)
        total_pos += 1 if has_pos else 0
        total_neg += 1 if has_neg else 0

        for label, kwds in aspects.items():
            if contains_any(sent, kwds):
                if has_pos:
                    score[label]["pos"] += 1
                if has_neg:
                    score[label]["neg"] += 1

    # Select top aspects by evidence counts
    ranked = sorted(
        score.items(),
        key=lambda kv: (max(kv[1]["pos"], kv[1]["neg"]), kv[1]["pos"] - kv[1]["neg"]),
        reverse=True
    )

    results: List[Dict[str, str]] = []
    for label, sc in ranked:
        evidence = sc["pos"] + sc["neg"]
        if evidence <= 0:
            continue
        sentiment = "positive" if sc["pos"] >= sc["neg"] else "negative"
        results.append({"label": label, "sentiment": sentiment})
        if len(results) >= 6:
            break

    if not results:
        overall_sent = "positive" if total_pos >= total_neg else "negative"
        results = [{"label": "Overall", "sentiment": overall_sent}]

    return results


def _oci_generate_summary(prompt: str) -> str:
    """
    Use OCI Generative AI Inference Chat API with SDK model classes,
    matching the working implementation in auslegalsearchv3 (GenericChatRequest).
    """
    USER = os.getenv("OCI_USER_OCID")
    TENANCY = os.getenv("OCI_TENANCY_OCID")
    FINGERPRINT = os.getenv("OCI_KEY_FINGERPRINT")
    KEY_FILE = os.getenv("OCI_KEY_FILE")
    REGION = os.getenv("OCI_REGION", "us-chicago-1")
    MODEL_OCID = os.getenv("OCI_GENAI_MODEL_OCID")
    COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_OCID")

    if not all([USER, TENANCY, FINGERPRINT, KEY_FILE, REGION, MODEL_OCID, COMPARTMENT_ID]):
        raise RuntimeError("OCI Generative AI environment variables are not fully configured.")

    try:
        import oci  # noqa: F401
        from oci.generative_ai_inference import GenerativeAiInferenceClient
        from oci.generative_ai_inference.models import (
            ChatDetails,
            GenericChatRequest,
            Message,
            TextContent,
            OnDemandServingMode,
            BaseChatRequest,
        )
    except Exception as e:
        raise RuntimeError("OCI Python SDK is required and must include generative_ai_inference Chat models. Install/upgrade with: pip install --upgrade oci") from e

    # Build OCI config (no explicit signer; client picks up from config)
    oci_config = {
        "user": USER,
        "tenancy": TENANCY,
        "fingerprint": FINGERPRINT,
        "key_file": KEY_FILE,
        "region": REGION,
    }
    client = GenerativeAiInferenceClient(oci_config)

    # Construct Chat request using SDK models (Generic API format)
    serving_mode = OnDemandServingMode(model_id=MODEL_OCID)

    txt_content = TextContent()
    txt_content.text = prompt

    user_msg = Message()
    user_msg.role = "USER"
    user_msg.content = [txt_content]

    chat_req = GenericChatRequest()
    chat_req.api_format = BaseChatRequest.API_FORMAT_GENERIC
    chat_req.messages = [user_msg]
    chat_req.max_tokens = 320
    chat_req.temperature = 0.3
    chat_req.frequency_penalty = 0.0
    chat_req.presence_penalty = 0.0
    chat_req.top_p = 0.9

    chat_details = ChatDetails()
    chat_details.serving_mode = serving_mode
    chat_details.chat_request = chat_req
    chat_details.compartment_id = COMPARTMENT_ID

    resp = client.chat(chat_details)
    data = getattr(resp, "data", None)

    text = None
    if data is not None:
        # Some SDKs wrap the choices under data.chat_response
        target = getattr(data, "chat_response", None)
        if target is None and isinstance(data, dict):
            target = data.get("chat_response")
        if target is None:
            target = data

        # Preferred: choices[0].message content in chat responses
        choices = getattr(target, "choices", None)
        if choices is None and isinstance(target, dict):
            choices = target.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            message = getattr(first, "message", None) or (first.get("message") if isinstance(first, dict) else None)
            if message is not None:
                # Handle content as a list of parts (SDK objects/dicts/strings)
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                if isinstance(content, list):
                    parts: List[str] = []
                    for c in content:
                        t = None
                        if hasattr(c, "text"):
                            t = getattr(c, "text", None)
                        elif isinstance(c, dict):
                            t = c.get("text")
                        elif isinstance(c, str):
                            t = c
                        if t:
                            parts.append(str(t))
                    if parts:
                        text = "\n".join(parts).strip()
                # Handle direct string content or message.text
                if not text:
                    mc = getattr(message, "content", None)
                    if isinstance(mc, str):
                        text = mc
                if not text:
                    text = getattr(message, "text", None) or (message.get("text") if isinstance(message, dict) else None)

        # Fallbacks on top-level/alternate fields
        if not text:
            text = getattr(target, "text", None) or getattr(target, "output_text", None) or getattr(target, "output", None)
            if not text and isinstance(target, dict):
                text = target.get("text") or target.get("output_text") or target.get("output")

        if not text:
            msg = getattr(target, "message", None) or (target.get("message") if isinstance(target, dict) else None)
            if msg is not None:
                c = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                if isinstance(c, list):
                    parts: List[str] = []
                    for ci in c:
                        t = getattr(ci, "text", None) if hasattr(ci, "text") else (ci.get("text") if isinstance(ci, dict) else (ci if isinstance(ci, str) else None))
                        if t:
                            parts.append(str(t))
                    if parts:
                        text = "\n".join(parts).strip()
                if not text and isinstance(c, str):
                    text = c
                if not text:
                    text = getattr(msg, "text", None) or (msg.get("text") if isinstance(msg, dict) else None)

    if not text:
        # Last resort: stringify response data to avoid a hard failure and aid diagnostics,
        # but try to avoid dumping the entire API response. Keep it concise.
        text = "No textual content returned by the model."

    return str(text).strip()


# -----------------------------------------------------------------------------
# Search API (with autocorrect)
# -----------------------------------------------------------------------------
@app.get("/api/search", tags=["api"])
def api_search(
    q: str = Query(..., min_length=1, description="Search query"),
    type: str = Query("all", pattern="^(all|products|reviews)$"),
    limit: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1),
    min_rating: Optional[float] = Query(None, ge=0.0, le=5.0, description="Filter reviews by minimum rating"),
    verified_only: Optional[bool] = Query(False, description="Limit reviews to verified purchases"),
):
    try:
        offset = (page - 1) * limit
        with get_conn() as conn:
            result: Dict[str, Any] = {"page": page, "limit": limit}
            if type in ("all", "products"):
                products, more_p = _product_search(conn, q, limit, offset)
                result["products"] = products
                result["has_more_products"] = more_p
            if type in ("all", "reviews"):
                reviews, more_r = _review_search(conn, q, limit, offset, min_rating, verified_only)
                result["reviews"] = reviews
                result["has_more_reviews"] = more_r
            # Autocorrect: if no results for requested type(s), attempt correction and rerun
            def _no_results(res: Dict[str, Any], t: str) -> bool:
                if t == "products":
                    return not res.get("products")
                if t == "reviews":
                    return not res.get("reviews")
                return not res.get("products") and not res.get("reviews")

            if _no_results(result, type):
                corrected_q, changed = _autocorrect_query(conn, q)
                if changed and corrected_q.strip().lower() != q.strip().lower():
                    used_q = corrected_q
                    if type in ("all", "products"):
                        products, more_p = _product_search(conn, used_q, limit, offset)
                        result["products"] = products
                        result["has_more_products"] = more_p
                    if type in ("all", "reviews"):
                        reviews, more_r = _review_search(conn, used_q, limit, offset, min_rating, verified_only)
                        result["reviews"] = reviews
                        result["has_more_reviews"] = more_r
                    result["original_query"] = q
                    result["used_query"] = used_q
                    result["suggestion_applied"] = True

            return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        logging.exception("Search failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/product/{parent_asin}", tags=["api"])
def api_product(parent_asin: str):
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    parent_asin,
                    title,
                    main_category,
                    store,
                    price,
                    average_rating,
                    rating_number,
                    images,
                    features,
                    description,
                    categories,
                    details
                FROM metadata
                WHERE parent_asin = %s
                """,
                (parent_asin,)
            )
            row = cur.fetchone()
            if not row:
                return JSONResponse(status_code=404, content={"error": "Not found"})
            return JSONResponse(content=jsonable_encoder(dict(row)))
    except Exception as e:
        logging.exception("Fetch product failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/summarize/{parent_asin}", tags=["api"])
def api_summarize(parent_asin: str):
    """
    Summarize user reviews for a given parent_asin using vector similarity retrieval + OCI Generative AI.
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Fetch product title for better prompt
            cur.execute("SELECT title FROM metadata WHERE parent_asin = %s", (parent_asin,))
            prod = cur.fetchone()
            title = prod.get("title") if prod else None

        with get_conn() as conn:
            centroid = _get_centroid_for_parent(conn, parent_asin, sample_limit=1000)
            if not centroid:
                return JSONResponse(status_code=404, content={"error": "No reviews with embeddings for this product."})
            vec_sql = _vector_to_sql_literal(centroid)
            candidates = _select_similar_reviews(conn, parent_asin, vec_sql, candidate_limit=200)
            evidence = _choose_evidence(candidates, top_k=40)
            if not evidence:
                return JSONResponse(status_code=404, content={"error": "No suitable reviews found for summarization."})
            prompt = _build_summary_prompt(parent_asin, title, evidence)
            summary = _oci_generate_summary(prompt)
            # Ensure it starts with "Customers say ..."
            s = summary.strip()
            if not s.lower().startswith("customers say"):
                s = "Customers say " + s.lstrip(". ").rstrip()
            aspects = _extract_key_themes(s)
            return JSONResponse(content=jsonable_encoder({"summary": s, "aspects": aspects}))
    except Exception as e:
        logging.exception("Summarization failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# -----------------------------------------------------------------------------
# Local Dev Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Run with: python search_app.py
    # Or: uvicorn search_app:app --host 0.0.0.0 --port 8000 --reload
    try:
        import uvicorn
    except ImportError:
        print("Missing dependency 'uvicorn'. Install with:\n  pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)

    uvicorn.run("search_app:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=True)
