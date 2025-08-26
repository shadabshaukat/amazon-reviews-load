import os
import sys
import json
import time
import argparse
import logging
from tqdm import tqdm
from typing import List, Dict, Any, Optional

import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# For vector embeddings using SentenceTransformers
from sentence_transformers import SentenceTransformer

# Load config from .env (for DB connection)
load_dotenv()

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO,
)

DB_CONFIG = {
    'host': os.getenv("PGHOST", "localhost"),
    'port': int(os.getenv("PGPORT", 5432)),
    'user': os.getenv("PGUSER", "postgres"),
    'password': os.getenv("PGPASSWORD", ""),
    'dbname': os.getenv("PGDATABASE", "postgres"),
}

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM = 768  # For HuggingFace model

METADATA_FIELDS = [
    "parent_asin", "main_category", "title", "average_rating", "rating_number", "features", "description",
    "price", "images", "videos", "store", "categories", "details", "bought_together"
]
REVIEW_FIELDS = [
    "asin", "user_id", "rating", "title", "text", "images",
    "parent_asin", "timestamp", "helpful_vote", "verified_purchase"
]

def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

def ensure_tables():
    schema_sql_path = "postgres_schema_amazon_reviews.sql"
    with open(schema_sql_path, "r") as f:
        schema_sql = f.read()
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
            for stmt in statements:
                try:
                    cur.execute(stmt)
                except Exception as e:
                    logging.warning(f"Skipping statement due to error: {e}\n{stmt}")
        conn.commit()

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

def _clean_float(val):
    try:
        if val is None:
            return None
        # Accept both numerics and numeric strings
        if isinstance(val, (float, int)):
            return float(val)
        sval = str(val).strip()
        if sval in ("", "—", "-", "NA", "N/A"):
            return None
        return float(sval)
    except Exception:
        return None

def _clean_int(val):
    try:
        if val is None:
            return None
        # Accept both numerics and numeric strings
        if isinstance(val, int):
            return val
        sval = str(val).strip()
        if sval in ("", "—", "-", "NA", "N/A"):
            return None
        return int(float(sval))
    except Exception:
        return None

def load_metadata(metadata_path: str, test_max=None):
    logging.info(f"Loading metadata from {metadata_path} ...")
    with get_db_conn() as conn, conn.cursor() as cur:
        with open(metadata_path, "r", encoding="utf-8") as fp:
            batch = []
            for meta in tqdm(parse_jsonl(fp, max_records=test_max), desc="Metadata", unit="rec"):
                row = {k: meta.get(k) for k in METADATA_FIELDS}
                if not row['parent_asin']:
                    logging.warning(f"Skipping record with missing parent_asin: {meta}")
                    continue
                # Robustly clean up floats/ints for postgres insertion
                row["average_rating"] = _clean_float(row.get("average_rating"))
                row["price"] = _clean_float(row.get("price"))
                row["rating_number"] = _clean_int(row.get("rating_number"))
                for key in ["features", "description", "images", "videos",
                            "categories", "details", "bought_together"]:
                    if row[key] is not None:
                        row[key] = json.dumps(row[key])
                batch.append(row)
                if len(batch) == 500 and not test_max:
                    insert_metadata_batch(cur, batch)
                    conn.commit()
                    batch.clear()
            if batch:
                insert_metadata_batch(cur, batch)
                conn.commit()
    logging.info("Metadata loading complete.")

def insert_metadata_batch(cur, records: List[dict]):
    keys = METADATA_FIELDS
    values = [[row.get(k) for k in keys] for row in records]
    placeholders = ", ".join(["%s"] * len(keys))
    stmt = f"""
    INSERT INTO metadata ({", ".join(keys)})
    VALUES ({placeholders})
    ON CONFLICT (parent_asin) DO UPDATE SET
      {", ".join(f"{k}=EXCLUDED.{k}" for k in keys if k != "parent_asin")}
    """
    execute_batch(cur, stmt, values, page_size=500)

def batch_embed_texts(model, texts: List[str]) -> List[List[float]]:
    """Embed texts using SentenceTransformer; pad/truncate to EMBED_DIM if needed."""
    if not texts:
        return []
    embs = model.encode(texts, batch_size=32, show_progress_bar=False)
    result = []
    for emb in embs:
        # ensure casting to Python floats
        if len(emb) < EMBED_DIM:
            vec = list(emb) + [0.0] * (EMBED_DIM - len(emb))
        elif len(emb) > EMBED_DIM:
            vec = list(emb[:EMBED_DIM])
        else:
            vec = list(emb)
        # The critical fix: convert all elements to standard Python float
        result.append([float(x) for x in vec])
    return result

def transform_review_json(rj: dict) -> dict:
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

def load_user_reviews(reviews_path: str, batch_size: int = 128, test_max=None, skip_missing_metadata=False):
    logging.info(f"Loading user reviews from {reviews_path} ...")
    review_rows = []
    review_texts = []
    count = 0
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    valid_parent_asins = None

    with get_db_conn() as conn, conn.cursor() as cur:
        if skip_missing_metadata:
            # fetch all valid parent_asin from metadata table
            cur.execute("SELECT parent_asin FROM metadata")
            valid_parent_asins = set(r[0] for r in cur.fetchall())
            logging.info(f"Loaded {len(valid_parent_asins)} parent_asins from metadata for FK validation")

        with open(reviews_path, "r", encoding="utf-8") as fp:
            for rj in tqdm(parse_jsonl(fp, max_records=test_max), desc="User Reviews", unit="rec"):
                row = transform_review_json(rj)
                if not row["parent_asin"]:
                    logging.warning(f"Skipping user review: missing parent_asin: {rj}")
                    continue
                if skip_missing_metadata and row["parent_asin"] not in valid_parent_asins:
                    logging.warning(f"Skipping user review {row.get('asin') or ''}: parent_asin={row['parent_asin']} not in metadata")
                    continue
                review_rows.append(row)
                review_texts.append(row.get("review_text") or "")
                if len(review_rows) == batch_size and not test_max:
                    insert_reviews_with_embedding(cur, review_rows, review_texts, model)
                    conn.commit()
                    count += len(review_rows)
                    review_rows, review_texts = [], []
            if review_rows:
                insert_reviews_with_embedding(cur, review_rows, review_texts, model)
                conn.commit()
                count += len(review_rows)
    logging.info(f"User reviews loading complete. Total inserted: {count}")

def insert_reviews_with_embedding(cur, rows: List[dict], texts: List[str], model):
    emb = batch_embed_texts(model, texts)
    assert len(emb) == len(rows)
    keys = [
        "asin", "user_id", "rating", "title", "review_text", "images", "parent_asin",
        "ts", "helpful_vote", "verified_purchase", "embedding"
    ]
    values = []
    for i, row in enumerate(rows):
        embedding = emb[i]
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
            embedding
        ])
    stmt = f"""
    INSERT INTO user_reviews
    ({", ".join(keys)})
    VALUES %s
    ON CONFLICT DO NOTHING
    """
    from psycopg2.extras import execute_values
    execute_values(cur, stmt.replace("%s", "%s"), values, page_size=128)

def main():
    parser = argparse.ArgumentParser(description=(
        "Amazon Reviews Loader for Postgres with Vector & FTS\n"
        "Uses SentenceTransformers with HuggingFace nomic-embed-text-v1.5 (768-dim) for review embeddings.\n"
        "Metadata table is keyed on parent_asin, and user reviews link via parent_asin foreign key.\n"
        "FTS (tsvector) is GENERATED/STORED and updated automatically by Postgres."
    ))
    parser.add_argument('--metadata', '-m', required=True, help="Path to metadata.jsonl")
    parser.add_argument('--reviews', '-r', required=True, help="Path to user_reviews.jsonl")
    parser.add_argument('--test', action="store_true", help="Run only a small sample load (2-3 records per file) for dry run validation.")
    parser.add_argument('--skip-missing-metadata', action="store_true", help="Skip user reviews referencing parent_asin not present in metadata (instead of failing with FK error).")
    args = parser.parse_args()

    ensure_tables()
    if args.test:
        load_metadata(args.metadata, test_max=3)
        load_user_reviews(args.reviews, batch_size=2, test_max=2, skip_missing_metadata=args.skip_missing_metadata)
        logging.info("Sample test run completed (2-3 records per file loaded).")
    else:
        load_metadata(args.metadata)
        load_user_reviews(args.reviews, skip_missing_metadata=args.skip_missing_metadata)
        logging.info("All data loaded.")

if __name__ == "__main__":
    main()
