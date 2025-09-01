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

app = FastAPI(title="Amazon Reviews Search", version="1.2.0")
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
            orig_q = q
            used_q = q
            suggestion_applied = False

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
                    suggestion_applied = True
                    if type in ("all", "products"):
                        products, more_p = _product_search(conn, used_q, limit, offset)
                        result["products"] = products
                        result["has_more_products"] = more_p
                    if type in ("all", "reviews"):
                        reviews, more_r = _review_search(conn, used_q, limit, offset, min_rating, verified_only)
                        result["reviews"] = reviews
                        result["has_more_reviews"] = more_r
                    result["original_query"] = orig_q
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
