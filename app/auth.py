"""API keys: clients send `Authorization: Bearer <key>`; only the key's SHA-256 hash is stored.

A key is checked by hashing it and looking the hash up, so the server never needs the key
itself, and a leaked api_keys table holds nothing that can be sent as a key. Keys are created,
listed, limited and revoked with manage_keys.py. Each valid request also takes a token from the
key's rate limit (app/ratelimit.py).
"""
import hashlib
import secrets

import psycopg
from fastapi import HTTPException, Request, Response, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import ratelimit

SCOPES = ("query", "ingest")
PREFIX = "rag_"  # makes keys recognizable, e.g. by secret scanners when one is committed by mistake

# auto_error=False: a missing header comes back as None, so the 401 below names the problem.
bearer = HTTPBearer(auto_error=False, description="API key created with manage_keys.py")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(conn: psycopg.Connection, name: str, scopes: list[str]) -> str:
    """Store a new key's hash and return the key, which can't be recovered later."""
    key = PREFIX + secrets.token_urlsafe(32)  # 32 random bytes from the OS's secure generator
    conn.execute("INSERT INTO api_keys (name, key_hash, scopes) VALUES (%s, %s, %s)",
                 (name, hash_key(key), scopes))
    return key


def revoke_key(conn: psycopg.Connection, name: str) -> bool:
    """True if an active key with this name was revoked."""
    cur = conn.execute("UPDATE api_keys SET revoked_at = now() WHERE name = %s AND revoked_at IS NULL", (name,))
    return cur.rowcount == 1


def set_limits(conn: psycopg.Connection, name: str, scope: str, per_minute: float, burst: int) -> bool:
    """Replace the default rate limit for one scope of an active key; True if the key exists."""
    cur = conn.execute("UPDATE api_keys SET rate_limits = rate_limits || jsonb_build_object(%s::text, "
                       "jsonb_build_object('per_minute', %s::float8, 'burst', %s::int)) "
                       "WHERE name = %s AND revoked_at IS NULL", (scope, per_minute, burst, name))
    return cur.rowcount == 1


def unauthorized(detail: str) -> HTTPException:
    # WWW-Authenticate tells the client which kind of credentials to send (HTTP's rule for 401s).
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


def require_scope(scope: str):
    """Dependency for an endpoint that needs a key with `scope`, within the key's rate limit;
    returns the key's name."""
    assert scope in SCOPES

    def check(request: Request, response: Response,
              creds: HTTPAuthorizationCredentials | None = Security(bearer)) -> str:
        if creds is None:
            raise unauthorized("missing API key: send 'Authorization: Bearer <key>'")
        with request.app.state.pool.connection() as conn:
            row = conn.execute("SELECT id, name, scopes, rate_limits FROM api_keys "
                               "WHERE key_hash = %s AND revoked_at IS NULL",
                               (hash_key(creds.credentials),)).fetchone()
            if row is None:
                raise unauthorized("invalid or revoked API key")
            key_id, name, scopes, overrides = row
            if scope not in scopes:
                raise HTTPException(403, f"API key '{name}' lacks the '{scope}' scope")
            taken = ratelimit.take(conn, key_id, scope, ratelimit.limits_for(scope, overrides))
        # Lets clients pace themselves; `response`'s headers are copied onto the endpoint's response.
        headers = {"RateLimit-Limit": str(taken.burst), "RateLimit-Remaining": str(taken.remaining)}
        if taken.retry_after:
            raise HTTPException(429, f"rate limit exceeded for '{name}' ({scope}): retry in {taken.retry_after} s",
                                headers=headers | {"Retry-After": str(taken.retry_after)})
        response.headers.update(headers)
        return name

    return check
