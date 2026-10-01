"""Per-key rate limits: a token bucket per API key and scope, kept in Postgres.

A bucket holds up to `burst` tokens and refills at `per_minute` tokens per minute; each request
takes one. So a key can send `burst` requests at once, then `per_minute` per minute on average.
A bucket is stored as (tokens, updated_at) and the refill since updated_at is computed by the
same statement that takes the token. Postgres locks the row for that update, so two concurrent
requests (from any number of uvicorn workers) can't both take the last token.
"""
import math
from dataclasses import dataclass

import psycopg

DEFAULT_LIMITS = {"query": {"per_minute": 10, "burst": 10}, "ingest": {"per_minute": 20, "burst": 20}}

# Tokens in the bucket now: the stored count plus the refill since updated_at, capped at burst.
# clock_timestamp(), not now(): now() is when the transaction started, which can be before the
# updated_at written by a request this one waited for (on the row lock).
REFILLED = "least(%(burst)s, b.tokens + extract(epoch FROM clock_timestamp() - b.updated_at) * %(per_minute)s / 60.0)"

# A key's first request creates a full bucket minus its token. Later requests take a token only
# if one is there; otherwise the WHERE fails, nothing is updated and no row is returned.
TAKE_SQL = f"""
INSERT INTO rate_buckets AS b (key_id, scope, tokens, updated_at)
VALUES (%(key_id)s, %(scope)s, %(burst)s - 1, clock_timestamp())
ON CONFLICT (key_id, scope) DO UPDATE
    SET tokens = {REFILLED} - 1, updated_at = clock_timestamp()
    WHERE {REFILLED} >= 1
RETURNING tokens
"""

PEEK_SQL = f"SELECT {REFILLED} FROM rate_buckets b WHERE key_id = %(key_id)s AND scope = %(scope)s"


@dataclass
class Taken:
    burst: int
    remaining: int    # whole tokens left after this request
    retry_after: int  # 0 if the request is allowed, else seconds until a token is back


def limits_for(scope: str, overrides: dict) -> dict:
    """The scope's defaults, replaced by the key's own limits where it has them (api_keys.rate_limits)."""
    return DEFAULT_LIMITS[scope] | overrides.get(scope, {})


def take(conn: psycopg.Connection, key_id: int, scope: str, limits: dict) -> Taken:
    """Take one token from the key's bucket for `scope`, if there is one."""
    params = {"key_id": key_id, "scope": scope, **limits}
    row = conn.execute(TAKE_SQL, params).fetchone()
    if row is not None:
        return Taken(limits["burst"], math.floor(row[0]), 0)
    tokens = conn.execute(PEEK_SQL, params).fetchone()[0]
    wait = (1 - tokens) * 60 / limits["per_minute"]
    return Taken(limits["burst"], 0, max(1, math.ceil(wait)))
