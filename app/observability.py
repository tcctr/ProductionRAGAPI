"""Request observability: per-stage timings, a request ID on every log line, and the query log.

Each request gets a record (a dict) in a context variable, created by the middleware:
  - timed("rerank") adds the stage's milliseconds to it (summed when a stage runs more than once,
    as in compare mode, which searches and reranks each version);
  - note(key=value) adds facts: the API key's name, its A/B variant, cached, the LLM's token counts,
    an error cause.
When the request ends, the middleware reports the record three ways:
  - one log line: `INFO app.request [<id>] /query 200 key=demo cached=false ... embed=14 rerank=812 total=4961`;
  - response headers X-Request-ID and Server-Timing (browser devtools draw the stages as a timeline);
  - for /query requests with a known API key, a row in query_log (db/schema.sql), for SQL afterwards.

The record lives in a ContextVar because the endpoints are plain defs that FastAPI runs in a worker
thread, which gets a copy of the request's context. The copy still points at the same dict, so code
in the thread only mutates it (never set()s the variable) and the middleware sees what it wrote.
Outside a request (eval scripts calling search() in-process) there is no record and these do nothing.

query_log keeps QUERY_LOG_RETENTION_DAYS of rows: after an insert, older ones are deleted, at most
once an hour per process (the delete runs before the response is sent, so not on every request).
Longer than the cache's 7 days because A/B experiments (ab_report.py) are read from this table.

Settings (environment variables):
    LOG_LEVEL                 default INFO; /health requests that succeed are logged at DEBUG (monitors poll it)
    QUERY_LOG_RETENTION_DAYS  default 30; 0 keeps every row
"""
import logging
import os
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import psycopg
from fastapi import Request
from psycopg.types.json import Jsonb
from starlette.concurrency import run_in_threadpool

log = logging.getLogger("app.request")

QUERY_LOG_RETENTION_DAYS = float(os.getenv("QUERY_LOG_RETENTION_DAYS", "30"))
PRUNE_EVERY_S = 3600
_last_prune = 0.0  # time.monotonic() of this process's last delete of old query_log rows

_record: ContextVar[dict | None] = ContextVar("request_record", default=None)

# A client's own X-Request-ID is reused (to follow a request across services) if it looks like an ID.
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


@contextmanager
def timed(stage: str) -> Iterator[None]:
    """Add the time spent in the block to the current request's timings, in ms (errors included)."""
    record = _record.get()
    start = time.perf_counter()
    try:
        yield
    finally:
        if record is not None:
            timings = record["timings"]
            timings[stage] = timings.get(stage, 0.0) + (time.perf_counter() - start) * 1000


def note(**fields) -> None:
    """Add facts about the current request (shown in its log line, stored in query_log)."""
    record = _record.get()
    if record is not None:
        record["fields"].update(fields)


def current_request_id() -> str | None:
    record = _record.get()
    return record["id"] if record else None


class RequestIdFilter(logging.Filter):
    """Puts the current request's ID on every log record ("-" outside a request)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id() or "-"
        return True


def setup_logging() -> None:
    """Log the app's messages to stderr with their request ID. uvicorn configures only its own
    loggers, so without a root handler Python would drop every INFO line."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"))
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)  # a line per HTTP call (Hugging Face Hub, TestClient)


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.0f}"
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value)
    return f'"{text}"' if " " in text else text


def server_timing(timings: dict[str, float]) -> str:
    return ", ".join(f"{stage};dur={ms:.1f}" for stage, ms in timings.items())


INSERT_SQL = """
INSERT INTO query_log (request_id, key_name, status, error, question, params, compared, cached,
                       n_chunks, timings, total_ms, prompt_tokens, completion_tokens, experiment, variant)
VALUES (%(id)s, %(key)s, %(status)s, %(error)s, %(question)s, %(params)s, %(compared)s, %(cached)s,
        %(chunks)s, %(timings)s, %(total)s, %(prompt_tokens)s, %(completion_tokens)s, %(experiment)s, %(variant)s)
"""


def write_query_log(pool, record: dict, status: int) -> None:
    """One query_log row, then expired rows (see prune_query_log()); a failure is logged, never
    raised (the response is already built)."""
    global _last_prune
    fields = record["fields"]
    timings = {stage: round(ms, 1) for stage, ms in record["timings"].items() if stage != "total"}
    try:
        with pool.connection() as conn:
            conn.execute(INSERT_SQL, {
                "id": record["id"], "key": fields.get("key"), "status": status, "error": fields.get("error"),
                "question": fields.get("question"), "params": Jsonb(fields.get("params")),
                "compared": fields.get("compared"), "cached": fields.get("cached"),
                "chunks": fields.get("chunks"), "timings": Jsonb(timings),
                "total": round(record["timings"]["total"], 1),
                "prompt_tokens": fields.get("prompt_tokens"), "completion_tokens": fields.get("completion_tokens"),
                "experiment": fields.get("experiment"), "variant": fields.get("variant"),
            })
            # Not thread-safe by design: two threads pruning at once only delete the same rows twice.
            if QUERY_LOG_RETENTION_DAYS > 0 and time.monotonic() - _last_prune >= PRUNE_EVERY_S:
                _last_prune = time.monotonic()
                prune_query_log(conn)
    except psycopg.Error as e:
        log.warning("query_log insert failed: %s", e)


def prune_query_log(conn: psycopg.Connection) -> int:
    """Delete query_log rows older than QUERY_LOG_RETENTION_DAYS; returns how many."""
    cur = conn.execute("DELETE FROM query_log WHERE created_at < now() - interval '1 day' * %s",
                       (QUERY_LOG_RETENTION_DAYS,))
    if cur.rowcount:
        log.info("query_log: deleted %d rows older than %g days", cur.rowcount, QUERY_LOG_RETENTION_DAYS)
    return cur.rowcount


# Stored in query_log but kept out of the log line (long, and logs get copied around more freely).
TABLE_ONLY = {"question", "params"}


async def middleware(request: Request, call_next):
    sent_id = request.headers.get("x-request-id", "")
    record = {"id": sent_id if REQUEST_ID.fullmatch(sent_id) else uuid.uuid4().hex,
              "timings": {}, "fields": {}}
    _record.set(record)  # this task's context; the endpoint's thread gets a copy pointing at the same dict
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500  # the error middleware outside this one turns it into a 500 response
        raise
    finally:
        record["timings"]["total"] = (time.perf_counter() - start) * 1000
        fields = {k: v for k, v in record["fields"].items() if k not in TABLE_ONLY and v is not None}
        line = " ".join([request.url.path, str(status)] + [f"{k}={fmt(v)}" for k, v in fields.items()]
                        + [f"{stage}={fmt(ms)}" for stage, ms in record["timings"].items()])
        quiet = request.url.path == "/health" and status == 200
        log.log(logging.DEBUG if quiet else logging.INFO, "%s", line)
        # A row needs a known key: unauthenticated requests (401) are logged above but not stored,
        # so anyone with the URL can't fill the table.
        if request.url.path == "/query" and "key" in record["fields"]:
            await run_in_threadpool(write_query_log, request.app.state.pool, record, status)
    response.headers["X-Request-ID"] = record["id"]
    response.headers["Server-Timing"] = server_timing(record["timings"])
    return response
