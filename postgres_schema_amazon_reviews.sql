-- ---------------------------------------------------------------------------
-- Optional: Cleanup (drop objects for full redeployment, in dependency order)
-- Un-comment the following DROP statements to remove all objects before recreate
-- ---------------------------------------------------------------------------
-- DROP INDEX IF EXISTS idx_user_reviews_embedding;
-- DROP INDEX IF EXISTS idx_user_reviews_fts;
-- DROP INDEX IF EXISTS idx_metadata_fts;
-- DROP INDEX IF EXISTS idx_metadata_title_trgm;
-- DROP INDEX IF EXISTS idx_metadata_store_trgm;
-- DROP INDEX IF EXISTS idx_metadata_categories_gin;
-- DROP INDEX IF EXISTS idx_metadata_details_gin;
-- DROP TABLE IF EXISTS user_reviews CASCADE;
-- DROP TABLE IF EXISTS metadata CASCADE;
-- DROP EXTENSION IF EXISTS vector CASCADE;
-- DROP EXTENSION IF EXISTS pg_trgm CASCADE;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- METADATA TABLE
CREATE TABLE IF NOT EXISTS metadata (
    parent_asin TEXT PRIMARY KEY,
    main_category TEXT,
    title TEXT,
    average_rating FLOAT,
    rating_number INTEGER,
    features JSONB,
    description JSONB,
    price FLOAT,
    images JSONB,
    videos JSONB,
    store TEXT,
    categories JSONB,
    details JSONB,
    bought_together JSONB
);

-- Enhance METADATA with Full Text Search
-- Create a generated tsvector that indexes key textual fields and JSONB content.
-- This mirrors user_reviews.fts and enables fast product/attribute search.
ALTER TABLE metadata
ADD COLUMN IF NOT EXISTS meta_fts tsvector
GENERATED ALWAYS AS (
  to_tsvector(
    'english',
      coalesce(title, '') || ' '
    || coalesce(main_category, '') || ' '
    || coalesce(store, '') || ' '
    || coalesce(features::text, '') || ' '
    || coalesce(description::text, '') || ' '
    || coalesce(categories::text, '') || ' '
    || coalesce(details::text, '')
  )
) STORED;

-- Indexes to accelerate FTS and common fuzzy lookups on metadata
CREATE INDEX IF NOT EXISTS idx_metadata_fts ON metadata USING GIN (meta_fts);
-- Trigram indexes for ILIKE/fuzzy matches on key string columns
CREATE INDEX IF NOT EXISTS idx_metadata_title_trgm ON metadata USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_metadata_store_trgm ON metadata USING GIN (store gin_trgm_ops);
-- Optional: JSONB GIN indexes for structured lookups (presence/containment queries)
CREATE INDEX IF NOT EXISTS idx_metadata_categories_gin ON metadata USING GIN (categories);
CREATE INDEX IF NOT EXISTS idx_metadata_details_gin ON metadata USING GIN (details);

-- USER REVIEWS TABLE with Vector Embeddings and Full Text Search
CREATE TABLE IF NOT EXISTS user_reviews (
    review_id SERIAL PRIMARY KEY,
    asin TEXT,
    user_id TEXT,
    rating FLOAT,
    title TEXT,
    review_text TEXT,
    images JSONB,
    parent_asin TEXT REFERENCES metadata(parent_asin) ON DELETE CASCADE,
    ts TIMESTAMP,
    helpful_vote INTEGER,
    verified_purchase BOOL,
    embedding vector(768),
    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(review_text,''))) STORED
);

-- Index for fast Full Text Search on user reviews
CREATE INDEX IF NOT EXISTS idx_user_reviews_fts ON user_reviews USING GIN (fts);

-- Index for fast vector similarity search on review embeddings
CREATE INDEX IF NOT EXISTS idx_user_reviews_embedding ON user_reviews USING ivfflat (embedding vector_cosine_ops);

-- For very large review corpora and frequent typos, consider adding an index for review titles
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_reviews_title_trgm ON user_reviews USING GIN (title gin_trgm_ops);

-- [INFO]
-- - Using SentenceTransformers with HuggingFace model "nomic-ai/nomic-embed-text-v1.5" (768-dim).
-- - Metadata is keyed on parent_asin. In user_reviews, parent_asin is a foreign key to metadata(parent_asin).
-- - FTS columns (user_reviews.fts and metadata.meta_fts) are GENERATED ALWAYS STORED, so any insert/update will maintain them automatically.
-- - Trigram indexes (pg_trgm) support fast ILIKE/fuzzy searches on metadata.title and metadata.store.
-- - Optional JSONB GIN indexes aid structured queries on categories/details (e.g., containment).
--
-- Example FTS usage on metadata:
--   SELECT parent_asin, title, ts_rank_cd(meta_fts, plainto_tsquery('english', 'wireless earbuds')) AS rank
--   FROM metadata
--   WHERE meta_fts @@ plainto_tsquery('english', 'wireless earbuds')
--   ORDER BY rank DESC
--   LIMIT 20;
