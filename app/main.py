"""RAG API over the PostgreSQL docs.

Usage:
    uvicorn app.main:app --reload        # needs ragapi-db, the embedding server and the LLM server
    open http://localhost:8000/docs      # interactive API docs

Endpoints are plain `def`s: FastAPI runs them in a thread pool, so the blocking
psycopg and requests calls (including the seconds-long LLM call) don't stall other requests.
"""
from collections.abc import Iterator
from contextlib import asynccontextmanager

import psycopg
import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from psycopg_pool import ConnectionPool, PoolTimeout

from app import generate as llm
from app.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.search import search
from chunk_docs import MAX_TOKENS, TARGET_TOKENS, TokenCounter, chunk_record
from embed_ingest import (DATABASE_URL, EMBED_URL, QUERY_PREFIX, embed, embed_pending, to_pgvector,
                          upsert_chunks, upsert_documents)


def health_url(api_url: str) -> str:
    """llama-server's /health lives at the root of the server that serves /v1/..."""
    return api_url.split("/v1/")[0] + "/health"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A pool keeps a few connections open and lends one to each request, instead of
    # paying for a new connection every time. check= replaces connections that died
    # (e.g. after a database restart) before handing them out.
    with ConnectionPool(DATABASE_URL, min_size=1, max_size=10,
                        check=ConnectionPool.check_connection) as pool:
        app.state.pool = pool
        app.state.count = TokenCounter()  # loaded once; /ingest chunks with it
        yield


app = FastAPI(title="PostgreSQL Docs RAG API", lifespan=lifespan)


def get_conn(request: Request) -> Iterator[psycopg.Connection]:
    """Lend the request a pooled connection; commits on success, rolls back on error."""
    with request.app.state.pool.connection() as conn:
        yield conn


def embedding_unavailable(e: Exception) -> HTTPException:
    return HTTPException(503, f"embedding server unavailable: {e}")


def llm_unavailable(e: Exception) -> HTTPException:
    return HTTPException(503, f"LLM server unavailable: {e}")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request):
    try:
        # One attempt: a user is waiting, so fail fast instead of embed()'s backoff retries.
        qvec = to_pgvector(embed([QUERY_PREFIX + req.question], retries=1)[0])
    except (requests.RequestException, ValueError) as e:
        raise embedding_unavailable(e)
    # Not get_conn, which holds the connection until the response is sent: give it back to the
    # pool before the LLM call, so slow answers don't use up connections that searches need.
    with request.app.state.pool.connection() as conn:
        chunks = search(conn, qvec, req.k, req.version, req.doc_type)
    answer = None
    if req.generate and chunks:
        try:
            answer = llm.generate(req.question, chunks)
        except (requests.RequestException, ValueError) as e:
            raise llm_unavailable(e)
    return {"question": req.question, "answer": answer, "chunks": chunks}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, request: Request, conn: psycopg.Connection = Depends(get_conn)):
    """Add or update one page: chunk it, upsert it, and embed its new or changed chunks.

    Re-sending an unchanged page is a no-op. If embedding fails, the chunks are stored
    without vectors (and skipped by /query) and re-sending the page resumes.
    """
    rec = req.model_dump()
    rec["version"] = str(req.version)  # chunk_docs works on docs.jsonl records, where it is a string
    chunks = chunk_record(rec, request.app.state.count, TARGET_TOKENS, MAX_TOKENS)

    doc_ids = upsert_documents(conn, [rec], prune=False)
    deleted = upsert_chunks(conn, chunks, doc_ids)
    conn.commit()
    try:
        embedded = embed_pending(conn, batch_size=32, ids=[c["id"] for c in chunks])
    except (requests.RequestException, ValueError) as e:
        raise embedding_unavailable(e)
    return {"id": f"{req.version}:{req.page}", "chunks_total": len(chunks),
            "chunks_embedded": embedded, "chunks_deleted": deleted}


@app.get("/health")
def health(request: Request):
    """200 if the database, the embedding server and the LLM server all respond, else 503."""
    status = {}
    try:
        # Not get_conn: when the database is down, fail in 2s instead of the pool's 30s default.
        with request.app.state.pool.connection(timeout=2) as conn:
            conn.execute("SELECT 1")
        status["database"] = "ok"
    except (psycopg.Error, PoolTimeout) as e:
        status["database"] = f"error: {e}"
    for name, url in [("embeddings", EMBED_URL), ("llm", llm.LLM_URL)]:
        try:
            requests.get(health_url(url), timeout=2).raise_for_status()
            status[name] = "ok"
        except requests.RequestException as e:
            status[name] = f"error: {e}"
    if any(v != "ok" for v in status.values()):
        raise HTTPException(503, status)
    return status
