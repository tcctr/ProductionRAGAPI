#!/usr/bin/env python3
"""Load parsed pages and chunks into Postgres and embed the chunks.

Usage:
    docker compose up -d --wait
    llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --embedding --port 8081
    python embed_ingest.py

Idempotent: pages and chunks are upserted, chunks whose text changed lose
their old vector, chunks that no longer exist are deleted, and only chunks
without a vector from EMBED_MODEL are sent to the embedding server. An
interrupted run resumes where it stopped (each batch is committed).

Settings (environment variables):
    DATABASE_URL  default postgresql://rag:rag@localhost:5433/rag
    EMBED_URL     default http://localhost:8081/v1/embeddings
    EMBED_MODEL   default nomic-embed-text-v1.5.Q8_0 (stored per chunk)
"""
import argparse
import hashlib
import json
import os
import time
import psycopg
import requests
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8081/v1/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text-v1.5.Q8_0")
EMBED_DIM = 768

# nomic-embed-text task prefixes
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed(texts: list[str], retries: int = 3) -> list[list[float]]:
    """Embed texts (already prefixed) via the OpenAI-compatible endpoint."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(EMBED_URL, json={"input": texts, "model": EMBED_MODEL}, timeout=120)
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vectors = [d["embedding"] for d in data]
            if len(vectors) != len(texts) or any(len(v) != EMBED_DIM for v in vectors):
                raise ValueError(f"expected {len(texts)} vectors of {EMBED_DIM} dims")
            return vectors
        except (requests.RequestException, ValueError) as e:
            if attempt == retries:
                raise
            print(f"  embed failed ({e}), retry {attempt}/{retries - 1}")
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def to_pgvector(v: list[float]) -> str:
    """pgvector accepts a '[x,y,...]' text literal cast to ::vector."""
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def upsert_documents(conn: psycopg.Connection, docs: list[dict], prune: bool = True) -> dict[tuple[int, str], int]:
    """Insert/update pages and return (version, page) -> id for them.

    With prune, pages not in docs are deleted (docs is the whole corpus); /ingest
    passes prune=False to add or update single pages.
    """
    rows = [
        (int(d["version"]), d["doc_type"], d["section_title"], d["page"], d["url"], d["text"],
         hashlib.sha256(d["text"].encode()).hexdigest())
        for d in docs
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO documents (version, doc_type, section_title, page, url, text, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (version, page) DO UPDATE SET
                doc_type = EXCLUDED.doc_type,
                section_title = EXCLUDED.section_title,
                url = EXCLUDED.url,
                text = EXCLUDED.text,
                content_hash = EXCLUDED.content_hash,
                ingested_at = CASE WHEN documents.content_hash = EXCLUDED.content_hash
                                   THEN documents.ingested_at ELSE now() END
            """,
            rows,
        )
        keys = [f"{v}:{p}" for v, _, _, p, *_ in rows]
        removed = 0
        if prune:
            cur.execute("DELETE FROM documents WHERE version || ':' || page <> ALL(%s)", (keys,))
            removed = cur.rowcount
        cur.execute("SELECT version, page, id FROM documents WHERE version || ':' || page = ANY(%s)", (keys,))
        ids = {(v, p): i for v, p, i in cur.fetchall()}
    print(f"documents: {len(rows)} upserted, {removed} removed")
    return ids


def upsert_chunks(conn: psycopg.Connection, chunks: list[dict], doc_ids: dict[tuple[int, str], int]) -> int:
    """Insert/update chunks of the pages in doc_ids and delete their stale chunks; returns #deleted.

    A chunk whose text changed loses its vector so it gets re-embedded. Chunks of
    pages outside doc_ids are untouched (pruned pages lose theirs via ON DELETE CASCADE).
    """
    rows = [
        (c["id"], doc_ids[(int(c["version"]), c["page"])], int(c["version"]), c["doc_type"], c["page"],
         c["url"], c["section_title"], c["heading_path"], c["chunk_index"], c["content"],
         c["token_count"], c["content_hash"])
        for c in chunks
    ]
    with conn.cursor() as cur:
        # Drop stale chunks first so a page that shrank can't collide on (document_id, chunk_index).
        cur.execute(
            "DELETE FROM chunks WHERE document_id = ANY(%s) AND id <> ALL(%s)",
            (list(doc_ids.values()), [r[0] for r in rows]),
        )
        removed = cur.rowcount
        cur.executemany(
            """
            INSERT INTO chunks (id, document_id, version, doc_type, page, url, section_title,
                                heading_path, chunk_index, content, token_count, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                doc_type = EXCLUDED.doc_type,
                url = EXCLUDED.url,
                section_title = EXCLUDED.section_title,
                heading_path = EXCLUDED.heading_path,
                content = EXCLUDED.content,
                token_count = EXCLUDED.token_count,
                content_hash = EXCLUDED.content_hash,
                embedding = CASE WHEN chunks.content_hash = EXCLUDED.content_hash
                                 THEN chunks.embedding END,
                embedding_model = CASE WHEN chunks.content_hash = EXCLUDED.content_hash
                                       THEN chunks.embedding_model END
            """,
            rows,
        )
    print(f"chunks: {len(rows)} upserted, {removed} removed")
    return removed


def embed_pending(conn: psycopg.Connection, batch_size: int, ids: list[str] | None = None) -> int:
    """Embed every chunk (or every chunk in ids) lacking a vector from EMBED_MODEL, committing after each batch."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, content FROM chunks "
            "WHERE (embedding IS NULL OR embedding_model IS DISTINCT FROM %s) "
            "AND (%s::text[] IS NULL OR id = ANY(%s)) ORDER BY id",
            (EMBED_MODEL, ids, ids),
        )
        pending = cur.fetchall()
    if not pending:
        print("embeddings: all chunks up to date")
        return 0

    print(f"embeddings: {len(pending)} chunks to embed (batch size {batch_size})")
    start = time.monotonic()
    done = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        vectors = embed([DOC_PREFIX + content for _, content in batch])
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET embedding = %s::vector, embedding_model = %s WHERE id = %s",
                [(to_pgvector(v), EMBED_MODEL, cid) for (cid, _), v in zip(batch, vectors)],
            )
        conn.commit()
        done += len(batch)
        if done == len(pending) or (i // batch_size) % 10 == 0:
            rate = done / (time.monotonic() - start)
            eta = (len(pending) - done) / rate
            print(f"  {done}/{len(pending)}  {rate:.1f} chunks/s  eta {eta:.0f}s")
    return done


def test_search(conn: psycopg.Connection, question: str, k: int = 5) -> None:
    """Embed a question and print the nearest chunks, to sanity-check retrieval end to end."""
    qvec = to_pgvector(embed([QUERY_PREFIX + question])[0])
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, 1 - (embedding <=> %s::vector) AS similarity, heading_path
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, k),
        )
        print(f"\ntest search: {question!r}")
        for cid, sim, path in cur.fetchall():
            print(f"  {sim:.3f}  {cid}  {' > '.join(path)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, default=Path("data/parsed/docs.jsonl"))
    ap.add_argument("--chunks", type=Path, default=Path("data/chunks/chunks.jsonl"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--test-query", default="how do I create an index without locking the table")
    args = ap.parse_args()

    docs, chunks = read_jsonl(args.docs), read_jsonl(args.chunks)
    with psycopg.connect(DATABASE_URL) as conn:
        # Pages and chunks change together in one transaction.
        doc_ids = upsert_documents(conn, docs)
        upsert_chunks(conn, chunks, doc_ids)
        conn.commit()

        embed_pending(conn, args.batch_size)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*), count(embedding) FROM chunks")
            total, embedded = cur.fetchone()
            cur.execute("ANALYZE chunks")
        conn.commit()
        print(f"\nchunks in db: {total}, with embeddings: {embedded}")

        if args.test_query and embedded:
            test_search(conn, args.test_query)


if __name__ == "__main__":
    main()
