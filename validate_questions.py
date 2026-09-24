#!/usr/bin/env python3
"""Check every eval question's labels against the parsed corpus.

Usage:
    python validate_questions.py

Each question in data/eval/questions.jsonl carries:
  expected_pages  ["18:sql-merge.html", ...]  pages that answer it (empty = unanswerable)
  evidence        {"sql-merge.html": "quote"}  quote that must appear on every
                                               expected version of that page
  absent          {"16:sql-merge.html": "quote"}  quote that must NOT appear there
                                               (proves a version lacks a feature)

Exits non-zero if any label is wrong, so labels can't silently drift from the docs.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED = {"id", "question", "category", "doc_type", "version", "expected_pages",
            "evidence", "absent", "reference_answer"}
CATEGORIES = {"direct", "paraphrase", "identifier", "version_specific", "version_diff",
              "cross_page", "unanswerable"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    ap.add_argument("--docs", type=Path, default=Path("data/parsed/docs.jsonl"))
    args = ap.parse_args()

    docs = {}
    for line in args.docs.open(encoding="utf-8"):
        d = json.loads(line)
        docs[f"{d['version']}:{d['page']}"] = d["text"]
    questions = [json.loads(line) for line in args.questions.open(encoding="utf-8")]

    errors: list[str] = []
    ids = Counter(q.get("id") for q in questions)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]

    for q in questions:
        qid = q.get("id", "?")
        if missing := REQUIRED - q.keys():
            errors.append(f"{qid}: missing fields {sorted(missing)}")
            continue
        if q["category"] not in CATEGORIES:
            errors.append(f"{qid}: unknown category {q['category']!r}")
        if (q["category"] == "unanswerable") != (not q["expected_pages"]):
            errors.append(f"{qid}: unanswerable questions (and only those) have no expected_pages")

        for key in q["expected_pages"]:
            if key not in docs:
                errors.append(f"{qid}: expected page {key} not in corpus")
                continue
            page = key.split(":", 1)[1]
            quote = q["evidence"].get(page)
            if quote is None:
                errors.append(f"{qid}: no evidence for {page}")
            elif quote not in docs[key]:
                errors.append(f"{qid}: evidence {quote!r} not found in {key}")
        for page in q["evidence"]:
            if not any(k.endswith(":" + page) for k in q["expected_pages"]):
                errors.append(f"{qid}: evidence for {page} but page not expected")
        for key, quote in q["absent"].items():
            if key not in docs:
                errors.append(f"{qid}: absent page {key} not in corpus")
            elif quote in docs[key]:
                errors.append(f"{qid}: {quote!r} should be absent from {key} but is present")
        if q["version"] is not None and any(not k.startswith(f"{q['version']}:") for k in q["expected_pages"]):
            errors.append(f"{qid}: asks about version {q['version']} but expects other versions")

    by_cat = Counter(q["category"] for q in questions)
    by_type = Counter(q["doc_type"] for q in questions)
    print(f"{len(questions)} questions")
    print("  by category:", dict(sorted(by_cat.items())))
    print("  by doc_type:", dict(sorted(by_type.items())))
    if errors:
        print(f"\n{len(errors)} problems:")
        for e in errors:
            print("  " + e)
        sys.exit(1)
    print("all labels verified against the corpus")


if __name__ == "__main__":
    main()
