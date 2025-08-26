import os
import sys
import json
import time
import argparse
import logging
from tqdm import tqdm
from typing import List, Dict
import math
import subprocess

import psycopg2
from psycopg2.extras import execute_batch, execute_values
from dotenv import load_dotenv

import torch
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---

DB_CONFIG = lambda: {
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
INSERT_REVIEW_FIELDS = [
    "asin", "user_id", "rating", "title", "review_text", "images", "parent_asin",
    "ts", "helpful_vote", "verified_purchase", "embedding"
]

# --- UTILS AND DBA ---

def get_db_conn():
    return psycopg2.connect(**DB_CONFIG())

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

def batch_embed_texts(model, texts: List[str], batch_size: int = 32):
    if not texts: return []
    with torch.no_grad():
        embs = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
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
    emb = batch_embed_texts(model, texts, batch_size=min(32, len(texts)))
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
        ({", ".join(INSERT_REVIEW_FIELDS)})
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    execute_values(cur, stmt.replace("%s", "%s"), values, page_size=len(rows))

# --- MULTI-GPU LOGIC ---

def detect_gpus():
    return torch.cuda.device_count()

def split_jsonl(input_path, num_chunks, output_prefix='reviews_chunk_'):
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    lines_per_chunk = math.ceil(total_lines / num_chunks)
    out_files = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i in range(num_chunks):
            chunk_path = f"{output_prefix}{i}.jsonl"
            out_files.append(chunk_path)
            with open(chunk_path, 'w', encoding='utf-8') as cf:
                for _ in range(lines_per_chunk):
                    line = f.readline()
                    if not line:
                        break
                    cf.write(line)
    return out_files

def multi_gpu_review_loader(args, review_path, meta_path, batch_size=128, skip_missing_metadata=False):
    num_gpus = detect_gpus()
    if not num_gpus:
        logging.error("No GPUs detected for --multi-gpu mode.")
        sys.exit(1)
    logging.info(f"Detected {num_gpus} GPUs for parallel loading.")
    chunk_files = split_jsonl(review_path, num_gpus)
    logging.info(f"Split '{review_path}' into {len(chunk_files)} chunks.")

    # Save DB env to child procs
    db_env = {k: os.environ[k] for k in os.environ if k.startswith("PG") or k in {"PATH"}}
    worker_procs = []
    for gpu_idx, chunk_file in enumerate(chunk_files):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        env['WORKER_ID'] = str(gpu_idx)
        # Pass .env DB variables
        env.update(db_env)
        cmd = [
            sys.executable, sys.argv[0],  # Launch this script in worker mode
            '--_gpu-worker',
            '--reviews', chunk_file,
            '--metadata', meta_path,
            '--batch-size', str(batch_size)
        ]
        if skip_missing_metadata:
            cmd.append('--skip-missing-metadata')
        p = subprocess.Popen(cmd, env=env)
        worker_procs.append(p)
    for p in worker_procs:
        p.wait()
    # optionally, cleanup chunk files here

def single_gpu_or_cpu_review_loader(reviews_path: str, metadata_path: str, batch_size: int = 128, skip_missing_metadata=False):
    logging.info(f"Loading user reviews from {reviews_path} ...")
    review_rows = []
    review_texts = []
    count = 0
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    valid_parent_asins = load_valid_parent_asins(metadata_path) if skip_missing_metadata else None

    with get_db_conn() as conn, conn.cursor() as cur, open(reviews_path, "r", encoding="utf-8") as fp:
        for rj in tqdm(parse_jsonl(fp), desc="User Reviews", unit="rec"):
            row = transform_review_json(rj)
            if not row["parent_asin"]:
                logging.warning(f"Skipping user review: missing parent_asin: {rj}")
                continue
            if skip_missing_metadata and row["parent_asin"] not in valid_parent_asins:
                logging.warning(f"Skipping user review {row.get('asin') or ''}: parent_asin={row['parent_asin']} not in metadata")
                continue
            review_rows.append(row)
            review_texts.append(row.get("review_text") or "")
            if len(review_rows) == batch_size:
                try:
                    insert_reviews_with_embedding(cur, review_rows, review_texts, model)
                    conn.commit()
                except Exception as e:
                    logging.error(f"Failed to insert batch: {e}")
                    conn.rollback()
                count += len(review_rows)
                review_rows, review_texts = [], []
                torch.cuda.empty_cache()
        if review_rows:
            try:
                insert_reviews_with_embedding(cur, review_rows, review_texts, model)
                conn.commit()
            except Exception as e:
                logging.error(f"Failed to insert batch: {e}")
                conn.rollback()
            count += len(review_rows)
            torch.cuda.empty_cache()
    logging.info(f"User reviews loading complete. Total inserted: {count}")

# --- ENTRY POINT ---

def main():
    parser = argparse.ArgumentParser(description=(
        "Amazon Reviews Loader (GPU+CPU parallel aware) for Postgres + Vector + FTS\n"
        "By default uses CPU or a single GPU for embedding.\n"
        "Optionally, inject --multi-gpu flag to batch/shard reviews across all GPUs for ultra-fast load."
    ))
    parser.add_argument('--metadata', '-m', required=True, help="Path to metadata.jsonl")
    parser.add_argument('--reviews', '-r', required=True, help="Path to user_reviews.jsonl")
    parser.add_argument('--test', action="store_true", help="Run only a small sample load for dry run validation.")
    parser.add_argument('--skip-missing-metadata', action="store_true", help="Skip user reviews referencing parent_asin not present in metadata (instead of failing with FK error).")
    parser.add_argument('--batch-size', type=int, default=128, help="Embedding batch size")
    parser.add_argument('--multi-gpu', action="store_true", help="Use all detected GPUs for parallel embedding/review load (recommended for very large files)")
    # Hidden flag for internal worker process
    parser.add_argument('--_gpu-worker', action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    load_dotenv(override=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    ensure_tables()

    if args.test:
        load_metadata(args.metadata, test_max=3)
        if args.multi_gpu:
            logging.warning("--multi-gpu ignored in test mode (only single process loads small sample).")
        single_gpu_or_cpu_review_loader(args.reviews, args.metadata, batch_size=2, skip_missing_metadata=args.skip_missing_metadata)
        logging.info("Sample test run completed (2-3 records per file loaded).")
    elif args._gpu_worker:
        # Internal forked process for multi-GPU mode. Each worker does its chunk!
        single_gpu_or_cpu_review_loader(args.reviews, args.metadata, batch_size=args.batch_size, skip_missing_metadata=args.skip_missing_metadata)
    else:
        load_metadata(args.metadata)
        if args.multi_gpu:
            multi_gpu_review_loader(args, args.reviews, args.metadata, batch_size=args.batch_size, skip_missing_metadata=args.skip_missing_metadata)
        else:
            single_gpu_or_cpu_review_loader(args.reviews, args.metadata, batch_size=args.batch_size, skip_missing_metadata=args.skip_missing_metadata)
        logging.info("All data loaded.")

if __name__ == "__main__":
    main()
