#!/usr/bin/env python3
"""Create, list, limit and revoke API keys for /query and /ingest.

Usage:
    python manage_keys.py create demo-frontend --scopes query
    python manage_keys.py create admin --scopes query ingest
    python manage_keys.py list
    python manage_keys.py limit demo-frontend query --per-minute 30 --burst 15
    python manage_keys.py revoke demo-frontend

`create` prints the key once: only its hash is stored, so it can't be shown again. Hand it to
the client privately; they send it as `Authorization: Bearer <key>`. A lost or leaked key is
revoked and replaced with a new one.

Rate limits (app/ratelimit.py): by default a key can send 10 queries at once, then 10 per
minute, and 20 ingests at once, then 20 per minute. `limit` changes one key's limit for a scope.

Settings (environment variables):
    DATABASE_URL  default postgresql://rag:rag@localhost:5433/rag
"""
import argparse
import sys

import psycopg

from app.auth import SCOPES, create_key, revoke_key, set_limits
from app.ratelimit import limits_for
from embed_ingest import DATABASE_URL


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create a key and print it once")
    create.add_argument("name", help="who holds the key, e.g. demo-frontend")
    create.add_argument("--scopes", nargs="+", choices=SCOPES, default=["query"])
    sub.add_parser("list", help="list keys (names, scopes and limits, never the keys)")
    limit = sub.add_parser("limit", help="set a key's rate limit for one scope")
    limit.add_argument("name")
    limit.add_argument("scope", choices=SCOPES)
    limit.add_argument("--per-minute", type=float, required=True, help="refill rate (average requests per minute)")
    limit.add_argument("--burst", type=int, required=True, help="bucket size (requests allowed at once)")
    revoke = sub.add_parser("revoke", help="revoke a key by name")
    revoke.add_argument("name")
    args = ap.parse_args()

    with psycopg.connect(DATABASE_URL) as conn:
        if args.command == "create":
            try:
                key = create_key(conn, args.name, sorted(set(args.scopes)))
            except psycopg.errors.UniqueViolation:
                sys.exit(f"a key named {args.name!r} already exists (names stay taken after revoking)")
            print(key)
            print(f"\nKey {args.name!r} with scopes {', '.join(sorted(set(args.scopes)))}. "
                  "Store it now (e.g. in an environment variable): it can't be shown again.", file=sys.stderr)
        elif args.command == "list":
            rows = conn.execute("SELECT name, scopes, rate_limits, created_at, revoked_at FROM api_keys "
                                "ORDER BY created_at").fetchall()
            for name, scopes, overrides, created, revoked in rows:
                status = f"revoked {revoked:%Y-%m-%d %H:%M}" if revoked else "active"
                limits = ", ".join(f"{s} {l['burst']} + {l['per_minute']:g}/min"
                                   for s in scopes for l in [limits_for(s, overrides)])
                print(f"{name:24} {status:22} created {created:%Y-%m-%d %H:%M}  {limits}")
            if not rows:
                print("no keys")
        elif args.command == "limit":
            if args.per_minute <= 0 or args.burst < 1:
                sys.exit("--per-minute must be > 0 and --burst >= 1")
            if not set_limits(conn, args.name, args.scope, args.per_minute, args.burst):
                sys.exit(f"no active key named {args.name!r}")
            print(f"{args.name!r} {args.scope}: {args.burst} at once, then {args.per_minute:g} per minute")
        elif not revoke_key(conn, args.name):
            sys.exit(f"no active key named {args.name!r}")
        else:
            print(f"revoked {args.name!r}")


if __name__ == "__main__":
    main()
