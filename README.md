# Amazon Reviews Loader: Postgres + Vector Embeddings + FTS

This project provides a scalable, robust pipeline to efficiently load Amazon Reviews JSONL datasets and their metadata into a Postgres 16+ database with:
- Full Text Search (FTS) via `tsvector`
- Fast semantic similarity search via pgvector (`vector` type)
- Hybrid search and flexible ranking enabled

**Vector embeddings** are created for review texts using the HuggingFace `nomic-ai/nomic-embed-text-v1.5` model through the SentenceTransformers Python library. The loader now supports *parallel embedding and ingest* using all GPUs on your machine for massive speedups.

---

## Features

- Batch ingest large-scale metadata and user review files in JSONL format
- Automatic generation of 768-dimensional vector embeddings for each review
- Schema-aligned tsvector (FTS) generation for all inserts/updates (GENERATED ALWAYS)
- Simple, robust, CLI-driven loader with progress bars
- **Multi-GPU support:** Parallel sharding and batched DB insert for reviews for maximal speed
- Dry-run/test mode for easy validation on small samples

---

## Project Layout

```
amazon_reviews_loader/
├── amazon_reviews_loader.py   # Main loader script (metadata + reviews + multi-gpu)
├── requirements_amazon_reviews_loader.txt
├── postgres_schema_amazon_reviews.sql    # Table, index, extension schema
├── README.md
├── meta_Cell_Phones_and_Accessories.jsonl   # Example metadata JSONL file
├── Cell_Phones_and_Accessories.jsonl        # Example user reviews JSONL file
└── .env (your DB credentials -- not checked in)
```

---

## Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/shadabshaukat/amazon-reviews-loader.git
cd amazon-reviews-loader/
```

### 2. Download Dataset

Download Dataset from Hugging Face : https://huggingface.co/datasets/shadabshaukat/amazon-cell-phones-accessories-user-reviews-metadata-may96-sep23

```bash
gzip -d Cell_Phones_and_Accessories.jsonl.gz
gzip -d meta_Cell_Phones_and_Accessories.jsonl.gz
```

### 3. Install Requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements_amazon_reviews_loader.txt
```

### 4. Set Up Postgres

- Ensure Postgres 16+ is running.
- Enable required extensions as a superuser (do this once per database):
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE EXTENSION IF NOT EXISTS pg_trgm;
  ```
- Apply the schema:
  ```bash
  psql -U youruser -d yourdb -f postgres_schema_amazon_reviews.sql
  ```

### 5. Configure Database Connection

Create a `.env` file in your project directory with:
```env
PGHOST=localhost
PGPORT=5432
PGDATABASE=your_db_name
PGUSER=your_user
PGPASSWORD=your_password
```

---

## Usage

### Standard Usage: Single-GPU/CPU Loader

**Strict mode (FK errors if metadata missing):**
```bash
python3 amazon_reviews_loader.py --metadata meta_Cell_Phones_and_Accessories.jsonl --reviews Cell_Phones_and_Accessories.jsonl
```

**Resilient mode (skip reviews missing parent metadata):**
```bash
python3 amazon_reviews_loader.py --metadata meta_Cell_Phones_and_Accessories.jsonl --reviews Cell_Phones_and_Accessories.jsonl --skip-missing-metadata
```

### Multi-GPU Accelerated Ingest (Recommended for large files)

**NEW: Ingest using all detected GPUs in parallel:**

```bash
python3 amazon_reviews_loader.py --metadata meta_Cell_Phones_and_Accessories.jsonl --reviews Cell_Phones_and_Accessories.jsonl --multi-gpu
```

- This will automatically detect all available CUDA GPUs, split your reviews file into N chunks (N = GPU count), and launch one loader process per GPU.
- Each process loads, embeds, and inserts its chunk in isolation—no lock contention or overlap.
- Can achieve massive throughput, linearly scaling with number of GPUs.

**Flags can be combined as before:**
```bash
python3 amazon_reviews_loader.py --metadata meta_Cell_Phones_and_Accessories.jsonl --reviews Cell_Phones_and_Accessories.jsonl --multi-gpu --skip-missing-metadata
```

### Test Mode (Dry Run with All Options)

```bash
python3 amazon_reviews_loader.py --metadata meta_Cell_Phones_and_Accessories.jsonl --reviews Cell_Phones_and_Accessories.jsonl --skip-missing-metadata --test
```

⚠️ In `--test` mode, only a few records per file are loaded for quick validation. `--multi-gpu` is ignored in test mode for simplicity.

---

## Multi-GPU Details

- **Automatic sharding:** The file is split into as many chunks as GPUs.
- **Fault tolerance:** Each chunk is processed independently, with its own database session.
- **No DB lock conflicts:** Each process writes disjoint primary-keys; commits batching-atomic.
- **GPU memory safe:** After each batch, torch cache is emptied so VRAM cannot overflow.
- **No extra scripts:** All logic is controlled via `amazon_reviews_loader.py`—no need for separate orchestrators.

---

## Schema Notes, Customization, Example Queries

*(As before. Section unchanged: see FTS, vector, schema, query examples, and CLI embedding conversion.)*

---

## Troubleshooting

- For OOM errors, reduce `--batch-size` and/or free up GPU VRAM.
- To maximize speed, use `--multi-gpu` if multiple CUDA devices available.
- All advanced search, hybrid, and vector capabilities remain unchanged.

---

## License

MIT. Copyright (c) Amazon Reviews Loader authors.

---
