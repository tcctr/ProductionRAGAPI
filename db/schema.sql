-- pgvector schema for the PostgreSQL docs RAG corpus.
-- Idempotent: safe to re-run. Also auto-applied on first container start
-- via /docker-entrypoint-initdb.d.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per parsed page (mirrors data/parsed/docs.jsonl).
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    version       SMALLINT    NOT NULL,
    doc_type      TEXT        NOT NULL CHECK (doc_type IN ('sql_command', 'functions', 'indexes_perf')),
    section_title TEXT        NOT NULL,
    page          TEXT        NOT NULL,
    url           TEXT        NOT NULL,
    text          TEXT        NOT NULL,
    content_hash  TEXT        NOT NULL,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (version, page)
);

-- One row per retrievable chunk. version/doc_type are denormalized so that
-- filtered vector search doesn't need a join.
CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,               -- "<version>:<page>#<chunk_index>"
    document_id     BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version         SMALLINT    NOT NULL,
    doc_type        TEXT        NOT NULL,
    page            TEXT        NOT NULL,
    url             TEXT        NOT NULL,           -- page URL + #anchor of the chunk's section
    section_title   TEXT        NOT NULL,
    heading_path    TEXT[]      NOT NULL DEFAULT '{}',
    chunk_index     INT         NOT NULL,
    content         TEXT        NOT NULL,           -- exactly what gets embedded (breadcrumb + body)
    token_count     INT         NOT NULL,
    content_hash    TEXT        NOT NULL,           -- lets re-ingest skip unchanged chunks
    embedding       vector(768),                    -- NULL until embedded
    embedding_model TEXT,
    -- Lexical side of hybrid search. 'english' stems words ("returning" -> return) and
    -- splits identifiers at underscores (jsonb_path_query -> jsonb, path, queri).
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin      ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_version_type ON chunks (version, doc_type);

-- Inverted index for BM25 keyword search (app/search.py): one row per (word, chunk) with how
-- often the chunk uses the word (tf) and the chunk's length in words (after stop-word
-- removal), so a query reads only its own words' rows instead of unpacking every matching
-- chunk's tsv. Refreshed after ingesting (embed_ingest.refresh_bm25_stats); until then new
-- chunks are found by vector search only.
CREATE MATERIALIZED VIEW IF NOT EXISTS chunk_terms AS
    SELECT u.lexeme AS word, ch.id AS chunk_id,
           coalesce(array_length(u.positions, 1), 1) AS tf,
           sum(coalesce(array_length(u.positions, 1), 1)) OVER (PARTITION BY ch.id) AS length
    FROM chunks ch, unnest(ch.tsv) AS u;
CREATE INDEX IF NOT EXISTS chunk_terms_word ON chunk_terms (word);

-- Number of chunks and their average length, for BM25's IDF and length normalization.
-- Built from chunk_terms, so refresh that first.
CREATE MATERIALIZED VIEW IF NOT EXISTS corpus_stats AS
    SELECT count(*) AS n_chunks, avg(length)::float8 AS avg_length
    FROM (SELECT DISTINCT chunk_id, length FROM chunk_terms) AS per_chunk;

-- API keys (app/auth.py, manage_keys.py). Only the SHA-256 hash of a key is stored: the key is
-- shown once when created, and a request's key is hashed and looked up here. Keys are long
-- random strings, so a fast hash is safe (slow hashes like bcrypt are for guessable passwords).
CREATE TABLE IF NOT EXISTS api_keys (
    id         BIGSERIAL PRIMARY KEY,
    name       TEXT        NOT NULL UNIQUE,          -- who holds it, e.g. "demo-frontend"
    key_hash   TEXT        NOT NULL UNIQUE,
    scopes     TEXT[]      NOT NULL CHECK (cardinality(scopes) > 0 AND scopes <@ ARRAY['query', 'ingest']),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ                           -- set instead of deleting, so the history stays
);
