# Amazon Reviews Loader: Postgres + Vector Embeddings + FTS

This project provides a scalable, robust pipeline to efficiently load Amazon Reviews JSONL datasets and their metadata into a Postgres 16+ database with:
- Full Text Search (FTS) via `tsvector`
- Fast semantic similarity search via pgvector (`vector` type)
- Hybrid search and flexible ranking enabled

**Vector embeddings** are created for review texts using the HuggingFace `nomic-ai/nomic-embed-text-v1.5` model through the SentenceTransformers Python library.

---

## Features

- Batch ingest large-scale metadata and user review files in JSONL format
- Automatic generation of 768-dimensional vector embeddings for each review
- Schema-aligned tsvector (FTS) generation for all inserts/updates (GENERATED ALWAYS)
- Simple, robust, CLI-driven loader with progress bars
- Dry-run/test mode for easy validation on small samples

---

## Project Layout

```
amazon_reviews_loader/
├── amazon_reviews_loader.py   # Main loader script (metadata + reviews)
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
git clone https://github.com/shadabshaukat/amazon-reviews-load.git

cd amazon-reviews-load/
```

### 2. Download Dataset

Hugging Face : https://huggingface.co/datasets/shadabshaukat/amazon-cell-phones-accessories-user-reviews-metadata-may96-sep23

```bash
curl -X GET \
     "https://datasets-server.huggingface.co/first-rows?dataset=shadabshaukat%2Famazon-cell-phones-accessories-user-reviews-metadata-may96-sep23&config=default&split=train"
```

```bash
gzip -d Cell_Phones_and_Accessories.jsonl.gz
gzip -d meta_Cell_Phones_and_Accessories.jsonl.gz
```

### 3. Install Requirements

```bash
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

### Loading Only Metadata (without reviews) - Load First

```bash
python3 amazon_reviews_loader.py --metadata meta_Cell_Phones_and_Accessories.jsonl --reviews /dev/null
```
This will only load metadata into Postgres. No user reviews will be loaded.

### Loading Only User Reviews (without metadata) - Load Next

```bash
python3 amazon_reviews_loader.py --metadata /dev/null --reviews Cell_Phones_and_Accessories.jsonl
```
This will only load user review data. Make sure the referenced parent_asin values already exist in the metadata table.

### Loading Both Metadata and User Reviews (Recommended - Single Command)

To run a **full batch load** of both metadata and user reviews in one step, use:

```bash
python3 amazon_reviews_loader.py --metadata meta_Cell_Phones_and_Accessories.jsonl --reviews Cell_Phones_and_Accessories.jsonl
```
This command will ingest all records into your database with FTS and vector search enabled.

### Test Mode (Dry Run: 2-3 records per file)

```bash
python3 amazon_reviews_loader.py --metadata meta_Cell_Phones_and_Accessories.jsonl --reviews Cell_Phones_and_Accessories.jsonl --test
```

---

## Customization

- To change to a different SentenceTransformers model, edit the `EMBED_MODEL` variable at the top of `amazon_reviews_loader.py` and set `EMBED_DIM` to the new vector dimension, updating your schema if needed.
- To increase batch size for faster loads, increase the `batch_size` parameter in `load_user_reviews`.
- For loading other product categories, simply provide the corresponding .jsonl files.

---

## Schema Notes

- The `user_reviews.embedding` column uses the `vector(768)` type for rapid semantic and hybrid search.
- The `fts` column is `GENERATED ALWAYS AS` - you never need to set it from code, and it's always kept up to date.
- GIN and IVFFLAT indices are created for fast search.

---

## Example Query: Semantic and FTS Hybrid

```sql
SELECT *, (embedding <#> '[0.1, ...]') AS vector_score
FROM user_reviews
WHERE fts @@ plainto_tsquery('english', 'excellent fit case')
ORDER BY vector_score ASC
LIMIT 10;
```

---

## Troubleshooting

- Review files must be in JSONL format (one object per line).
- For extremely large files, always test with the `--test` flag first.
- If you see "psycopg2.errors.UndefinedObject: type vector does not exist", make sure you ran the vector extension enable step as a superuser.

---

## License

MIT. Copyright (c) Amazon Reviews Loader authors.

---
