-- Example Postgres vector similarity search queries
-- Assumes pgvector extension and user_reviews.embedding vector(768)

-- 1. Top-K Cosine Similarity
SELECT review_id, parent_asin, user_id, rating, title, review_text, embedding,
  1 - (embedding <=> $1::vector) AS cosine_similarity
FROM user_reviews
ORDER BY embedding <=> $1::vector
LIMIT 10;

-- 2. Top-K L2 (Euclidean)
SELECT review_id, parent_asin, user_id, rating, title, review_text, embedding,
  embedding <-> $1::vector AS l2_distance
FROM user_reviews
ORDER BY embedding <-> $1::vector
LIMIT 10;

-- 3. Top-K Inner Product
SELECT review_id, parent_asin, user_id, rating, title, review_text, embedding,
  -(embedding <#> $1::vector) AS inner_product
FROM user_reviews
ORDER BY embedding <#> $1::vector
LIMIT 10;

-- 4. Filtered Similarity Search (with product/category filter)
SELECT review_id, parent_asin, user_id, rating, title, review_text, embedding,
  1 - (embedding <=> $1::vector) AS cosine_similarity
FROM user_reviews
WHERE parent_asin = $2
ORDER BY embedding <=> $1::vector
LIMIT 10;

-- 5. Hybrid Search: FTS + Vector Similarity
SELECT review_id, parent_asin, user_id, rating, title, review_text, embedding,
  1 - (embedding <=> $1::vector) AS cosine_similarity,
  ts_rank_cd(fts, plainto_tsquery('english', $2)) AS fts_rank
FROM user_reviews
WHERE fts @@ plainto_tsquery('english', $2)
ORDER BY (embedding <=> $1::vector) + (0.5 * ts_rank_cd(fts, plainto_tsquery('english', $2))) ASC
LIMIT 10;

-- 6. ANN (approximate) similarity search requires ivfflat/graph index on embedding
-- Use the same queries as above, and Postgres will use index if created

-- 7. Batched/Multiple query embeddings (for semantic search with several vectors)
-- Returns all reviews within top N of any of the vectors in array
SELECT review_id, parent_asin, user_id, rating, title, review_text, embedding,
  embedding <-> v AS l2_distance
FROM user_reviews, unnest(ARRAY[$1::vector, $2::vector, $3::vector]) as q(v)
ORDER BY embedding <-> v
LIMIT 10;

-- 8. Range search (return all reviews within threshold distance)
SELECT review_id, parent_asin, user_id, rating, title, review_text, embedding
FROM user_reviews
WHERE embedding <=> $1::vector < 0.2   -- for cosine, threshold tuned as needed
ORDER BY embedding <=> $1::vector
LIMIT 100;

-- Placeholders:
--   $1::vector = your query embedding (e.g., ARRAY[...]::vector)
--   $2, $3, ... as additional query parameters
