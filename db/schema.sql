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
-- Per-key limits that replace app/ratelimit.py's defaults, by scope, e.g.
-- {"query": {"per_minute": 60, "burst": 20}}. Set with `manage_keys.py limit`.
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS rate_limits JSONB NOT NULL DEFAULT '{}';

-- Token buckets for rate limiting (app/ratelimit.py): one per key and scope, created on the
-- key's first request. `tokens` is the count at `updated_at`; refills are computed on the next request.
CREATE TABLE IF NOT EXISTS rate_buckets (
    key_id     BIGINT      NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    scope      TEXT        NOT NULL,
    tokens     FLOAT8      NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (key_id, scope)
);

-- /query responses by request (app/cache.py): a repeated question is answered from here instead of
-- the reranker and the LLM. Emptied whenever the chunks change (/ingest, embed_ingest.py); entries
-- older than CACHE_TTL_DAYS are ignored and deleted.
CREATE TABLE IF NOT EXISTS answer_cache (
    key        TEXT        PRIMARY KEY,               -- SHA-256 of the request and the pipeline settings
    response   JSONB       NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS answer_cache_created_at ON answer_cache (created_at);

-- One row per /query request with a known API key (app/observability.py), errors and cache hits
-- included: what was asked, how it went and where the time went. Timings are ms per stage, e.g.
-- {"auth": 3.1, "embed": 14.2, "search": 21.0, "rerank": 812.4, "llm": 4105.3}; a cache hit has only
-- auth and cache_get. key_name is text, not a reference, so the history outlives deleted keys.
CREATE TABLE IF NOT EXISTS query_log (
    id                BIGSERIAL   PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    request_id        TEXT        NOT NULL,           -- also in the log lines and the X-Request-ID header
    key_name          TEXT        NOT NULL,
    status            INT         NOT NULL,
    error             TEXT,                           -- short cause for 4xx/5xx, e.g. "reranker unavailable"
    question          TEXT,                           -- NULL when rejected before the body was read (403, 429)
    params            JSONB,                          -- k, version, doc_type, compare_versions, generate
    compared          BOOLEAN,                        -- compare mode actually used (per-version search)
    cached            BOOLEAN,
    n_chunks          INT,
    timings           JSONB       NOT NULL,
    total_ms          FLOAT8      NOT NULL,
    prompt_tokens     INT,
    completion_tokens INT
);
CREATE INDEX IF NOT EXISTS query_log_created_at ON query_log (created_at);
-- A/B experiment and the key's variant in it (app/variants.py); NULL without an experiment and for
-- requests rejected before the variant is assigned (403, 429).
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS experiment TEXT;
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS variant TEXT;
CREATE INDEX IF NOT EXISTS query_log_experiment ON query_log (experiment, variant) WHERE experiment IS NOT NULL;
