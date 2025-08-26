import os
import sys
import argparse
import json
import time
import logging
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from typing import List

import torch
from sentence_transformers import SentenceTransformer

# Load .env for DB connection inside worker (custom .env path supported)
def setup_db_env(dotenv_path=None):
    dotenv_path = dotenv_path or ".env"
    load_dotenv(dotenv_path, override=True)

DB_CONFIG = {
    'host': os.getenv("PGHOST", "localhost"),
    'port': int(os.getenv("PGPORT", 5432)),
    'user': os.getenv("PGUSER", "postgres"),
    'password': os.getenv("PGPASSWORD", ""),
    'dbname': os.getenv("PGDATABASE", "postgres"),
}

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM = 768

REVIEW_FIELDS = [
    "asin", "user_id", "rating", "title", "text", "images",
    "parent_asin", "timestamp", "helpful_vote", "verified_purchase"
]
INSERT_FIELDS = [
    "asin", "user_id", "rating", "title", "review_text", "images", "parent_asin",
    "ts", "helpful_vote", "verified_purchase", "embedding"
]

def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

def parse_jsonl(fp, max_records=None):
    count = 0
    for line in fp:
        if line.strip():
            try:
                yield json.loads(line)
                count += 1
                if max_records is not None and count >= max_records:
                    break
            except Exception as e:
                logging.warning(f"Skipping malformed JSON line: {e}")

def batch_embed_texts(model, texts: List[str], batch_size: int):
    if not texts:
        return []
    # Use torch.no_grad to reduce VRAM use
    with torch.no_grad():
        embs = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    # Pad/truncate, cast floats
    result = []
    for emb in embs:
        if len(emb) < EMBED_DIM:
            vec = list(emb) + [0.0] * (EMBED_DIM - len(emb))
        elif len(emb) > EMBED_DIM:
            vec = list(emb[:EMBED_DIM])
        else:
            vec = list(emb)
        result.append([float(x) for x in vec])
    return result

def transform_review_json(rj: dict):
    row = {}
    for k in REVIEW_FIELDS:
        v = rj.get(k)
        row[k] = v
    ts = row.get("timestamp")
    if ts:
        try:
            row["ts"] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(int(ts) // 1000))
        except Exception:
            row["ts"] = None
    else:
        row["ts"] = None
    row["review_text"] = row.pop("text", None)
    if row.get("images") is not None:
        row["images"] = json.dumps(row["images"])
    else:
        row["images"] = None
    row.pop("timestamp", None)
    return row

def load_valid_parent_asins(metadata_path):
    valid_parent_asins = set()
    with open(metadata_path, "r", encoding="utf-8") as fp:
        for obj in parse_jsonl(fp):
            pa = obj.get("parent_asin")
            if pa:
                valid_parent_asins.add(pa)
    logging.info(f"Loaded {len(valid_parent_asins)} parent_asins from metadata for FK validation.")
    return valid_parent_asins

def insert_reviews_with_embedding(cur, rows: List[dict], texts: List[str], model):
    emb = batch_embed_texts(model, texts, batch_size=len(texts))
    assert len(emb) == len(rows)
    values = []
    for i, row in enumerate(rows):
        values.append([
            row.get("asin"),
            row.get("user_id"),
            row.get("rating"),
            row.get("title"),
            row.get("review_text"),
            row.get("images"),
            row.get("parent_asin"),
            row.get("ts"),
            row.get("helpful_vote"),
            row.get("verified_purchase"),
            emb[i]
        ])
    stmt = f"""
        INSERT INTO user_reviews
        ({", ".join(INSERT_FIELDS)})
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    execute_values(cur, stmt.replace("%s", "%s"), values, page_size=len(rows))

def worker_main():
    parser = argparse.ArgumentParser(description="Single-GPU loader worker for assigned chunk")
    parser.add_argument('--reviews', required=True, help="Chunk reviews file")
    parser.add_argument('--metadata', required=True, help="Full metadata file (for FK checking)")
    parser.add_argument('--batch-size', type=int, default=128, help="Embedding + Insert batch size")
    parser.add_argument('--skip-missing-metadata', action='store_true', help="Skip reviews not in metadata")
    parser.add_argument('--dotenv', default=".env", help="Env file for DB creds")
    args = parser.parse_args()

    setup_db_env(args.dotenv)

    # Set up GPU context
    assigned_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    logging.basicConfig(format=f'[GPU-{assigned_gpu}][%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    valid_parent_asins = load_valid_parent_asins(args.metadata) if args.skip_missing_metadata else None

    total_count = 0

    with get_db_conn() as conn, conn.cursor() as cur, open(args.reviews, "r", encoding="utf-8") as fp:
        batch_rows, batch_texts = [], []
        for rj in tqdm(parse_jsonl(fp), desc=f"Worker Reviews (GPU {assigned_gpu})", unit="rec"):
            row = transform_review_json(rj)
            if not row["parent_asin"]:
                logging.warning(f"Skipping: missing parent_asin: {rj}")
                continue
            if valid_parent_asins and row["parent_asin"] not in valid_parent_asins:
                logging.warning(f"Skipping: parent_asin={row['parent_asin']} not in metadata")
                continue
            batch_rows.append(row)
            batch_texts.append(row.get("review_text") or "")
            if len(batch_rows) >= args.batch_size:
                try:
                    insert_reviews_with_embedding(cur, batch_rows, batch_texts, model)
                    conn.commit()
                except Exception as e:
                    logging.error(f"Failed to insert batch: {e}")
                    conn.rollback()
                total_count += len(batch_rows)
                batch_rows, batch_texts = [], []
                torch.cuda.empty_cache()
        if batch_rows:
            try:
                insert_reviews_with_embedding(cur, batch_rows, batch_texts, model)
                conn.commit()
            except Exception as e:
                logging.error(f"Failed to insert batch: {e}")
                conn.rollback()
            total_count += len(batch_rows)
            torch.cuda.empty_cache()
    logging.info(f"Worker finished. Total user reviews inserted: {total_count}")

if __name__ == "__main__":
    worker_main()
