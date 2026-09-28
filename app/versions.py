"""Version-comparison questions ("Which version added X?"): per-version retrieval and a term check.

A single search over all versions returns whatever scores highest, often nothing from the
version that lacks the feature, and a handful of chunks can't show that a version lacks it
anyway. For these questions:
  - each version is searched separately, so every version is represented in the context;
  - identifiers from the question (`casefold()`, JSON_TABLE, AT LOCAL) are looked up by exact
    text in every chunk of every version, so the LLM can say "not in the 16 docs" as a fact.
"""
import math
import re
from typing import get_args

import psycopg

from app.models import Version
from app.search import body, search

VERSIONS: tuple[int, ...] = get_args(Version)

# "Which (PostgreSQL) version(s) ...", "Since which version ...", "In which versions ...",
# "When was/were ... added/introduced".
VERSION_QUESTION = re.compile(
    r"\b(?:which|what)\s+(?:postgresql\s+|postgres\s+)?versions?\b"
    r"|\bwhen\s+(?:was|were|did)\b.*\b(?:add|added|introduced|introduce|appear|appeared)\b",
    re.IGNORECASE)

# Term shapes, most specific first. Lowercase identifiers must look like code (a call, an
# underscore or a digit) unless a noun like "function" follows, so ordinary words aren't looked up.
BACKTICKED = re.compile(r"`([^`]+)`")
CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\)")
CODE_WORD = re.compile(r"\b([A-Za-z]+_[A-Za-z0-9_]+|[A-Za-z]+[0-9][A-Za-z0-9]*)\b")
# Runs of ALL-CAPS words: SQL keywords or phrases like "AT LOCAL", "SET EXPRESSION".
KEYWORDS = re.compile(r"\b[A-Z][A-Z_]+(?:\s+[A-Z][A-Z_]+)*\b")
KIND = r"(?:functions?|operators?|aggregates?|options?|parameters?|clauses?)"
BEFORE_KIND = re.compile(rf"\b([a-z][a-z0-9_]{{2,}})(?:\s+and\s+([a-z][a-z0-9_]{{2,}}))?\s+{KIND}\b")
NOT_TERMS = {"the", "new", "window", "aggregate", "json", "jsonb", "text", "array", "which", "what",
             "that", "this", "these", "those", "some", "any", "all", "postgresql", "version", "versions",
             "uuids", "uuid"}
# Abbreviations that read as ALL-CAPS words but aren't SQL.
NOT_KEYWORDS = {"PG", "SQL", "I", "A", "UUID", "UUIDS", "JSON"}


def is_version_question(question: str) -> bool:
    return bool(VERSION_QUESTION.search(question))


def question_terms(question: str) -> list[str]:
    """Identifiers and SQL keyword phrases in the question, in order, without duplicates."""
    found: list[str] = []
    found += BACKTICKED.findall(question)
    found += CALL.findall(question)
    found += CODE_WORD.findall(question)
    found += [k for k in KEYWORDS.findall(question) if k not in NOT_KEYWORDS and len(k) > 1]
    for m in BEFORE_KIND.finditer(question):
        found += [t for t in m.groups() if t]
    terms: list[str] = []
    for t in (t.strip() for t in found):
        if t and t.lower() not in NOT_TERMS and t not in terms:
            terms.append(t)
    return terms


def term_pattern(term: str) -> tuple[str, str]:
    """Postgres regex operator and pattern matching term as a whole word.

    ALL-CAPS keywords match case-sensitively, so "AT LOCAL" doesn't match "look at local files";
    identifiers match in any case, since questions write json_table and JSON_TABLE alike.
    """
    op = "~" if term.isupper() else "~*"
    return op, r"\m" + re.escape(term).replace(r"\ ", r"\s+") + r"\M"


def term_presence(conn: psycopg.Connection, terms: list[str]) -> dict[str, list[int]]:
    """For each term, the versions with at least one chunk containing it (an exact text scan of
    every chunk, not just the retrieved ones)."""
    presence = {}
    for term in terms:
        op, pattern = term_pattern(term)
        rows = conn.execute(f"SELECT DISTINCT version FROM chunks WHERE content {op} %s ORDER BY version",
                            (pattern,)).fetchall()
        presence[term] = [int(v) for (v,) in rows]
    return presence


def search_per_version(conn: psycopg.Connection, qvec: str, k: int,
                       doc_type: str | None = None) -> list[dict]:
    """ceil(k / #versions) chunks from each version, oldest version first; chunks whose text is
    identical across versions are merged into the first, with every version in "versions"."""
    per_version = math.ceil(k / len(VERSIONS))
    results: list[dict] = []
    seen: dict[str, dict] = {}
    for v in VERSIONS:
        for c in search(conn, qvec, per_version, v, doc_type, dedup=False):
            key = body(c["content"])
            if key in seen:
                seen[key]["versions"].append(v)
                continue
            seen[key] = c
            results.append(c)
    return results


def retrieve(conn: psycopg.Connection, question: str, qvec: str, k: int, version: int | None = None,
             doc_type: str | None = None, compare_versions: bool | None = None,
             ) -> tuple[list[dict], dict[str, list[int]] | None]:
    """The chunks /query answers from, and the term presence when comparing versions (else None).

    compare_versions None means detect it from the question; a question limited to one
    version is never compared.
    """
    if compare_versions is None:
        compare_versions = version is None and is_version_question(question)
    if not compare_versions or version is not None:
        return search(conn, qvec, k, version, doc_type), None
    chunks = search_per_version(conn, qvec, k, doc_type)
    return chunks, term_presence(conn, question_terms(question))
