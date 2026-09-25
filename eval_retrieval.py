#!/usr/bin/env python3
"""Measure retrieval quality against the eval question set.

Usage:
    python eval_retrieval.py                  # plain vector search (baseline)
    python eval_retrieval.py --filter-version # restrict to the question's version
    python eval_retrieval.py --dedup          # merge identical sections across versions, like /query

For each answerable question, embeds it as a search query, takes the top-k
chunks from pgvector, and checks whether any comes from an expected page
("18:sql-merge.html"). Labels are page-level so they survive re-chunking.
Search is app/search.py, the same code /query runs. With --dedup, a merged
result counts for every version in its "versions".

Metrics:
  hit@k   share of questions with an expected page in the top k
  MRR     mean of 1/rank of the first expected page (0 if not in top k)
  cover   cross_page only: share of the distinct expected pages found in top k

Unanswerable questions are not scored; their top similarity is reported next
to the answerable ones' as input for a future "I don't know" threshold.
Each run is saved to data/eval/results/<timestamp>.json.
"""
import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import psycopg

from app.search import search
from embed_ingest import DATABASE_URL, EMBED_MODEL, QUERY_PREFIX, embed, to_pgvector

KS = (1, 5, 10)


def score(results: list[dict]) -> dict:
    n = len(results)
    if not n:
        return {"n": 0}
    out = {"n": n}
    for k in KS:
        out[f"hit@{k}"] = sum(r["rank"] is not None and r["rank"] <= k for r in results) / n
    out["mrr"] = sum(1 / r["rank"] for r in results if r["rank"]) / n
    return out


def fmt(m: dict) -> str:
    if not m["n"]:
        return "n=0"
    return (f"n={m['n']:3}  " + "  ".join(f"hit@{k}={m[f'hit@{k}']:.2f}" for k in KS)
            + f"  MRR={m['mrr']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    ap.add_argument("--k", type=int, default=max(KS))
    ap.add_argument("--filter-version", action="store_true",
                    help="restrict search to the question's version when it names one")
    ap.add_argument("--dedup", action="store_true",
                    help="without a version filter, merge chunks identical across versions (as /query does)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/eval/results"))
    ap.add_argument("--show-misses", action="store_true", help="print questions with no hit in top k")
    args = ap.parse_args()

    questions = [json.loads(line) for line in args.questions.open(encoding="utf-8")]
    vectors = embed([QUERY_PREFIX + q["question"] for q in questions])

    results, unanswerable = [], []
    with psycopg.connect(DATABASE_URL) as conn:
        for q, vec in zip(questions, vectors):
            version = q["version"] if args.filter_version else None
            top = search(conn, to_pgvector(vec), args.k, version, dedup=args.dedup)
            if q["category"] == "unanswerable":
                unanswerable.append({"id": q["id"], "top_similarity": top[0]["similarity"]})
                continue
            expected = set(q["expected_pages"])
            keys = [f"{c['version']}:{c['page']}" for c in top]
            # Every version:page each result stands for (several when dedup merged copies).
            covered = [{f"{v}:{c['page']}" for v in c["versions"]} for c in top]
            rank = next((i + 1 for i, ks in enumerate(covered) if ks & expected), None)
            wanted_pages = {key.split(":", 1)[1] for key in expected}
            found_pages = {key.split(":", 1)[1] for ks in covered for key in ks & expected}
            results.append({
                "id": q["id"], "category": q["category"], "doc_type": q["doc_type"],
                "rank": rank, "top_similarity": top[0]["similarity"],
                "coverage": len(found_pages) / len(wanted_pages),
                "retrieved": keys,
            })

    overall = score(results)
    by_cat, by_type = defaultdict(list), defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
        by_type[r["doc_type"]].append(r)

    mode = ("version filter" if args.filter_version else "no filter") + (", dedup" if args.dedup else "")
    print(f"retrieval eval: {len(results)} answerable questions, top {args.k}, {mode}\n")
    print(f"{'overall':18} {fmt(overall)}")
    print("\nby category")
    for cat, rs in sorted(by_cat.items()):
        print(f"  {cat:16} {fmt(score(rs))}")
    print("\nby doc_type")
    for dt, rs in sorted(by_type.items()):
        print(f"  {dt:16} {fmt(score(rs))}")
    cross = [r["coverage"] for r in by_cat.get("cross_page", [])]
    if cross:
        print(f"\ncross_page coverage (share of needed pages in top {args.k}): {mean(cross):.2f}")
    answerable_sims = [r["top_similarity"] for r in results]
    unanswerable_sims = [u["top_similarity"] for u in unanswerable]
    if unanswerable_sims:
        print(f"top-1 similarity  answerable: median {median(answerable_sims):.3f} min {min(answerable_sims):.3f}"
              f"  |  unanswerable: median {median(unanswerable_sims):.3f} max {max(unanswerable_sims):.3f}")

    if args.show_misses:
        misses = {r["id"] for r in results if r["rank"] is None}
        print(f"\nmisses ({len(misses)}):")
        for q in questions:
            if q["id"] in misses:
                r = next(r for r in results if r["id"] == q["id"])
                print(f"  {q['id']} [{q['category']}] {q['question']}\n      got: {', '.join(r['retrieved'][:3])}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{stamp}.json"
    path.write_text(json.dumps({
        "timestamp": stamp, "git_commit": commit, "embed_model": EMBED_MODEL,
        "k": args.k, "filter_version": args.filter_version, "dedup": args.dedup,
        "overall": overall,
        "by_category": {c: score(rs) for c, rs in by_cat.items()},
        "by_doc_type": {d: score(rs) for d, rs in by_type.items()},
        "cross_page_coverage": mean(cross) if cross else None,
        "questions": results, "unanswerable": unanswerable,
    }, indent=2) + "\n")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
