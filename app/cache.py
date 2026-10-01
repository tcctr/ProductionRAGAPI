"""Response cache for /query: a repeated question is answered from Postgres in milliseconds
instead of going through the reranker (~0.8 s) and the LLM (~5 s, one answer at a time).

Exact match only. A question is normalized by whitespace alone, not case: question_terms()
reads ALL-CAPS runs as SQL keywords and searches them case-sensitively, so "AT LOCAL" and
"at local" can get different answers. Similar-but-different questions are deliberately not
matched (a "semantic cache"): "Which version added X?" questions about different features embed
almost identically, and similarity can't even tell out-of-scope questions apart here.

The key also covers the settings that shape the response (embedding model, search and rerank
constants, LLM server and model, prompt text), so changing one makes old entries stop matching.
The chunks changing is handled by emptying the table (embed_ingest.clear_answer_cache()).
Other code changes to the pipeline aren't in the key: after one, empty the table by hand
(`DELETE FROM answer_cache`).

Settings (environment variables):
    CACHE_TTL_DAYS  entries older than this are ignored and deleted, default 7
"""
import hashlib
import json
import logging
import os

import psycopg
from psycopg.types.json import Jsonb

import embed_ingest
from app import generate, search
from app.models import QueryRequest

log = logging.getLogger(__name__)

CACHE_TTL_DAYS = float(os.getenv("CACHE_TTL_DAYS", "7"))


def normalize(question: str) -> str:
    """Strip the question and collapse runs of whitespace; case is kept (see the module docstring)."""
    return " ".join(question.split())


def cache_key(req: QueryRequest) -> str:
    """SHA-256 of the request and the pipeline settings. Read at call time, so env and test overrides count."""
    request = req.model_dump() | {"question": normalize(req.question)}
    settings = {
        "embed_model": embed_ingest.EMBED_MODEL,
        "search": [search.HYBRID_POOL, search.RRF_K, search.RERANK_POOL, search.EF_SEARCH_MIN,
                   search.BM25_K1, search.BM25_B],
        # The answer only matters when one is generated: generate=false hits whatever the LLM is.
        "llm": [generate.LLM_URL, generate.LLM_MODEL, generate.TEMPERATURE, generate.MAX_ANSWER_TOKENS,
                generate.SYSTEM_PROMPT, generate.VERSION_RULES] if req.generate else None,
    }
    blob = json.dumps({"request": request, "settings": settings}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def get(conn: psycopg.Connection, key: str) -> dict | None:
    """The cached response for key, unless it's missing or older than CACHE_TTL_DAYS."""
    row = conn.execute(
        "SELECT response FROM answer_cache WHERE key = %s AND created_at > now() - interval '1 day' * %s",
        (key, CACHE_TTL_DAYS),
    ).fetchone()
    return row[0] if row else None


def put(conn: psycopg.Connection, key: str, response: dict) -> None:
    """Store a response, replacing one stored by a concurrent miss, and delete expired entries."""
    conn.execute(
        """
        INSERT INTO answer_cache (key, response) VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET response = EXCLUDED.response, created_at = now()
        """,
        (key, Jsonb(response)),
    )
    conn.execute("DELETE FROM answer_cache WHERE created_at <= now() - interval '1 day' * %s",
                 (CACHE_TTL_DAYS,))
