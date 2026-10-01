#!/usr/bin/env python3
"""Create, list and revoke API keys for /query and /ingest.

Usage:
    python manage_keys.py create demo-frontend --scopes query
    python manage_keys.py create admin --scopes query ingest
    python manage_keys.py list
    python manage_keys.py revoke demo-frontend

`create` prints the key once: only its hash is stored, so it can't be shown again. Hand it to
the client privately; they send it as `Authorization: Bearer <key>`. A lost or leaked key is
revoked and replaced with a new one.

Settings (environment variables):
    DATABASE_URL  default postgresql://rag:rag@localhost:5433/rag
"""
import argparse
import sys

import psycopg

from app.auth import SCOPES, create_key, revoke_key
from embed_ingest import DATABASE_URL


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create a key and print it once")
    create.add_argument("name", help="who holds the key, e.g. demo-frontend")
    create.add_argument("--scopes", nargs="+", choices=SCOPES, default=["query"])
    sub.add_parser("list", help="list keys (names and scopes, never the keys)")
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
            rows = conn.execute("SELECT name, scopes, created_at, revoked_at FROM api_keys ORDER BY created_at").fetchall()
            for name, scopes, created, revoked in rows:
                status = f"revoked {revoked:%Y-%m-%d %H:%M}" if revoked else "active"
                print(f"{name:24} {','.join(scopes):14} created {created:%Y-%m-%d %H:%M}  {status}")
            if not rows:
                print("no keys")
        elif not revoke_key(conn, args.name):
            sys.exit(f"no active key named {args.name!r}")
        else:
            print(f"revoked {args.name!r}")


if __name__ == "__main__":
    main()
