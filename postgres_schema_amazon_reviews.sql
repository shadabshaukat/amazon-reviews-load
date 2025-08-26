-- ---------------------------------------------------------------------------
-- Optional: Cleanup (drop objects for full redeployment, in dependency order)
-- Un-comment the following DROP statements to remove all objects before recreate
-- ---------------------------------------------------------------------------
-- DROP INDEX IF EXISTS idx_user_reviews_embedding;
-- DROP INDEX IF EXISTS idx_user_reviews_fts;
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

-- Index for fast Full Text Search
CREATE INDEX IF NOT EXISTS idx_user_reviews_fts ON user_reviews USING GIN (fts);

-- Index for fast vector similarity search
CREATE INDEX IF NOT EXISTS idx_user_reviews_embedding ON user_reviews USING ivfflat (embedding vector_cosine_ops);

-- [INFO] Using SentenceTransformers with HuggingFace model "nomic-ai/nomic-embed-text-v1.5" (768-dim).
-- Note: Metadata is keyed on parent_asin. In user_reviews, parent_asin is a foreign key to metadata(parent_asin).
-- Adjusted embedding vector column to vector(768). If you use a different model, update schema accordingly.
-- FTS column is now a GENERATED ALWAYS STORED column, so any insert/update will always create/update the FTS data automatically.
