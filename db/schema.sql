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
    -- Lexical side of hybrid search. 'english' stems prose; identifiers like
    -- jsonb_path_query survive as single lexemes.
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin      ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_version_type ON chunks (version, doc_type);
