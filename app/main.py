"""RAG API over the PostgreSQL docs.

Usage:
    uvicorn app.main:app --reload        # needs ragapi-db, the embedding server and the LLM server
    open http://localhost:8000/docs      # interactive API docs

Endpoints are plain `def`s: FastAPI runs them in a thread pool, so the blocking
psycopg and requests calls (including the seconds-long LLM call) don't stall other requests.

/query and /ingest need an API key with the matching scope and take a token from the key's
rate limit (app/auth.py, app/ratelimit.py, manage_keys.py); /health is open, for monitoring.

Every request is timed by stage and logged with a request ID; /query requests also go to
the query_log table (app/observability.py).
"""
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager

import psycopg
import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from psycopg_pool import ConnectionPool, PoolTimeout

from app import cache
from app import generate as llm
from app import observability
from app import rerank as reranker
from app.auth import require_scope
from app.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.observability import note, timed
from app.versions import retrieve
from chunk_docs import MAX_TOKENS, TARGET_TOKENS, TokenCounter, chunk_record
from embed_ingest import (DATABASE_URL, EMBED_URL, QUERY_PREFIX, clear_answer_cache, embed, embed_pending,
                          refresh_bm25_stats, to_pgvector, upsert_chunks, upsert_documents)


observability.setup_logging()
log = logging.getLogger(__name__)

# Answers being generated or waiting for the LLM. The LLM server answers one request at a time
# and queues the rest, so without a cap a burst of queries would wait there for minutes;
# beyond the cap, /query answers 503 at once and the client retries later.
LLM_MAX_PENDING = int(os.getenv("LLM_MAX_PENDING", "4"))
llm_slots = threading.BoundedSemaphore(LLM_MAX_PENDING)


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
app.middleware("http")(observability.middleware)


def get_conn(request: Request) -> Iterator[psycopg.Connection]:
    """Lend the request a pooled connection; commits on success, rolls back on error."""
    with request.app.state.pool.connection() as conn:
        yield conn


def unavailable(cause: str, e: Exception) -> HTTPException:
    """503 for a server the request needs; the short cause goes to query_log, the details to the log."""
    note(error=cause)
    log.warning("%s: %s", cause, e)
    return HTTPException(503, f"{cause}: {e}")


def embedding_unavailable(e: Exception) -> HTTPException:
    return unavailable("embedding server unavailable", e)


def llm_unavailable(e: Exception) -> HTTPException:
    return unavailable("LLM server unavailable", e)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_scope("query"))])
def query(req: QueryRequest, request: Request):
    # A repeated request is answered from the cache (app/cache.py); it still took a rate-limit token.
    note(question=req.question, params=req.model_dump(exclude={"question"}))
    key = cache.cache_key(req)
    with timed("cache_get"), request.app.state.pool.connection() as conn:
        hit = cache.get(conn, key)
    if hit is not None:
        note(cached=True, chunks=len(hit["chunks"]))
        return hit | {"question": req.question, "cached": True}
    note(cached=False)

    try:
        # One attempt: a user is waiting, so fail fast instead of embed()'s backoff retries.
        with timed("embed"):
            qvec = to_pgvector(embed([QUERY_PREFIX + req.question], retries=1)[0])
    except (requests.RequestException, ValueError) as e:
        raise embedding_unavailable(e)
    # Not get_conn, which holds the connection until the response is sent: give it back to the
    # pool before the LLM call, so slow answers don't use up connections that searches need.
    with request.app.state.pool.connection() as conn:
        try:
            chunks, presence = retrieve(conn, req.question, qvec, req.k, req.version, req.doc_type,
                                        req.compare_versions)
        except (requests.RequestException, ValueError) as e:
            raise unavailable("reranker unavailable", e)
    note(compared=presence is not None, chunks=len(chunks))
    answer = None
    if req.generate and chunks:
        if not llm_slots.acquire(blocking=False):
            note(error="LLM busy")
            raise HTTPException(503, f"LLM busy: {LLM_MAX_PENDING} answers already pending, retry shortly",
                                headers={"Retry-After": "5"})
        try:
            with timed("llm"):
                answer = llm.generate(req.question, chunks, presence)
        except (requests.RequestException, ValueError) as e:
            raise llm_unavailable(e)
        finally:
            llm_slots.release()
    response = {"question": req.question, "answer": answer, "chunks": chunks, "version_presence": presence}
    # Errors raised above are never cached; neither is an empty result, which an /ingest may fill.
    if chunks:
        try:
            with timed("cache_put"), request.app.state.pool.connection() as conn:
                cache.put(conn, key, response)
        except psycopg.Error as e:
            log.warning("query: caching the response failed: %s", e)  # the answer is still good
    return response | {"cached": False}


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_scope("ingest"))])
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
    refresh_bm25_stats(conn)
    # Cached answers may cite deleted chunks; new and changed ones count once embedded (below).
    # An unchanged page clears nothing.
    if deleted:
        clear_answer_cache(conn)
    conn.commit()
    try:
        embedded = embed_pending(conn, batch_size=32, ids=[c["id"] for c in chunks])
    except (requests.RequestException, ValueError) as e:
        raise embedding_unavailable(e)
    if embedded:
        clear_answer_cache(conn)  # committed by get_conn
    note(page=f"{req.version}:{req.page}", chunks=len(chunks), embedded=embedded, deleted=deleted)
    return {"id": f"{req.version}:{req.page}", "chunks_total": len(chunks),
            "chunks_embedded": embedded, "chunks_deleted": deleted}


@app.get("/health")
def health(request: Request):
    """200 if the database and the embedding, reranker and LLM servers all respond, else 503.

    Open to anyone, so failures say only "error"; the details (addresses, messages) go to the log.
    """
    status = {}
    try:
        # Not get_conn: when the database is down, fail in 2s instead of the pool's 30s default.
        with request.app.state.pool.connection(timeout=2) as conn:
            conn.execute("SELECT 1")
        status["database"] = "ok"
    except (psycopg.Error, PoolTimeout) as e:
        log.warning("health: database: %s", e)
        status["database"] = "error"
    for name, url in [("embeddings", EMBED_URL), ("reranker", reranker.RERANK_URL), ("llm", llm.LLM_URL)]:
        try:
            requests.get(health_url(url), timeout=2).raise_for_status()
            status[name] = "ok"
        except requests.RequestException as e:
            log.warning("health: %s: %s", name, e)
            status[name] = "error"
    if any(v != "ok" for v in status.values()):
        raise HTTPException(503, status)
    return status
