#!/usr/bin/env python3
"""Paired A/B comparison of two saved eval runs, question by question.

Usage:
    python compare_runs.py data/eval/results/A.json data/eval/results/B.json
    python compare_runs.py data/eval/judgments/A.json data/eval/judgments/B.json
    python compare_runs.py --a A1.json A2.json --b B1.json B2.json   # repeat runs, averaged per question
    python compare_runs.py A.json B.json --category version_diff     # only the questions a change can affect

Works on eval_retrieval.py results or judge_answers.py judgments (both sides the same kind).
Answerable questions present on both sides are paired; each gets a score per side:
  retrieval  chunk-level reciprocal rank (1/rank, 0 if not in top k); binary: chunk hit@5
  answers    correct 1, partial 0.5, otherwise 0; binary: correct
With repeat runs, a question's score is its mean over that side's runs (binary: majority, ties count
as a miss), which damps the sampling/judge noise of single answer runs.

Statistics:
  bootstrap  resample the paired questions with replacement (10,000 times) and take the middle 95%
             of mean(B) - mean(A): if that interval contains 0, the difference could be noise
  McNemar    exact two-sided test on the binary metric: only questions where A and B disagree count,
             and p is the chance of a split at least that lopsided if each were a coin flip
"""
import argparse
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import mean

ANSWER_SCORE = {"correct": 1.0, "partial": 0.5}
RESAMPLES = 10_000


def kind(run: dict) -> str:
    return "answers" if "answers_file" in run else "retrieval"


def per_question(run: dict) -> dict[str, dict]:
    """Answerable question id -> {"category", "score", "hit", "label"} for one run."""
    out = {}
    for q in run["questions"]:
        if q["category"] == "unanswerable":
            continue
        if kind(run) == "retrieval":
            rank = q["chunk_rank"]
            out[q["id"]] = {"category": q["category"], "score": 1 / rank if rank else 0.0,
                            "hit": rank is not None and rank <= 5, "label": f"rank {rank or '-'}"}
        elif "verdict" in q:  # skipped questions (LLM down) have no verdict
            out[q["id"]] = {"category": q["category"], "score": ANSWER_SCORE.get(q["verdict"], 0.0),
                            "hit": q["verdict"] == "correct", "label": q["verdict"]}
    return out


def combine(runs: list[dict]) -> dict[str, dict]:
    """Average each question over repeat runs; only questions graded in every run are kept."""
    tables = [per_question(r) for r in runs]
    ids = set.intersection(*(set(t) for t in tables))
    return {i: {"category": tables[0][i]["category"],
                "score": mean(t[i]["score"] for t in tables),
                "hit": sum(t[i]["hit"] for t in tables) > len(tables) / 2,
                "label": "/".join(t[i]["label"] for t in tables)} for i in ids}


def bootstrap(diffs: list[float], seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(RESAMPLES))
    return means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def mcnemar(b_wins: int, a_wins: int) -> float:
    """Exact two-sided McNemar p-value: binomial test of the discordant pairs against p = 0.5."""
    n = b_wins + a_wins
    if not n:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b_wins, a_wins) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def describe(run: dict, path: Path) -> str:
    if kind(run) == "answers":
        return f"{path.name}  ({run['name']}, commit {run['git_commit']})"
    mode = (("filter" if run["filter_version"] else "no filter") + (", dedup" if run["dedup"] else "")
            + (", hybrid" if run["hybrid"] else ", vector only") + f", rerank {run.get('rerank_pool') or 0}"
            + (f", compare versions (rerank {run['compare_rerank_pool']})" if run.get("compare_versions") else "")
            + f", k {run['k']}")
    return f"{path.name}  ({mode}, commit {run['git_commit']})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", nargs="*", type=Path, help="A.json B.json (one run per side)")
    ap.add_argument("--a", nargs="+", type=Path, default=[], help="side A run(s)")
    ap.add_argument("--b", nargs="+", type=Path, default=[], help="side B run(s)")
    ap.add_argument("--category", nargs="+", help="only questions of these categories")
    args = ap.parse_args()
    if args.pair:
        if len(args.pair) != 2 or args.a or args.b:
            ap.error("give exactly two files, or --a/--b")
        args.a, args.b = [args.pair[0]], [args.pair[1]]
    if not args.a or not args.b:
        ap.error("need runs for both sides")

    sides = {name: [(p, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
             for name, paths in (("A", args.a), ("B", args.b))}
    kinds = {kind(run) for runs in sides.values() for _, run in runs}
    if len(kinds) > 1:
        ap.error("mixing retrieval results and answer judgments")
    answers = kinds == {"answers"}
    for name, runs in sides.items():
        for path, run in runs:
            print(f"{name}: {describe(run, path)}")

    a, b = (combine([run for _, run in sides[s]]) for s in ("A", "B"))
    ids = sorted(set(a) & set(b))
    if dropped := sorted(set(a) ^ set(b)):
        print(f"not on both sides, skipped: {', '.join(dropped)}")
    if args.category:
        ids = [i for i in ids if a[i]["category"] in args.category]
        print(f"categories: {', '.join(args.category)}")
    if not ids:
        ap.error("no question is graded on both sides")

    score_name = "answer score (correct 1, partial 0.5)" if answers else "chunk MRR"
    hit_name = "correct" if answers else "chunk hit@5"
    diffs = [b[i]["score"] - a[i]["score"] for i in ids]
    low, high = bootstrap(diffs)
    mean_a, mean_b = mean(a[i]["score"] for i in ids), mean(b[i]["score"] for i in ids)
    print(f"\n{len(ids)} paired questions")
    print(f"{score_name:38} A {mean_a:.3f}  B {mean_b:.3f}  B-A {mean_b - mean_a:+.3f}"
          f"  95% CI [{low:+.3f}, {high:+.3f}]" + ("  (contains 0)" if low <= 0 <= high else ""))
    b_wins = sum(b[i]["hit"] and not a[i]["hit"] for i in ids)
    a_wins = sum(a[i]["hit"] and not b[i]["hit"] for i in ids)
    hits_a, hits_b = sum(a[i]["hit"] for i in ids), sum(b[i]["hit"] for i in ids)
    print(f"{hit_name:38} A {hits_a / len(ids):.3f}  B {hits_b / len(ids):.3f}"
          f"  only B {b_wins}, only A {a_wins}  McNemar p={mcnemar(b_wins, a_wins):.3f}")

    print("\nby category (mean score)")
    by_cat = defaultdict(list)
    for i in ids:
        by_cat[a[i]["category"]].append(i)
    for cat, qs in sorted(by_cat.items()):
        ca, cb = mean(a[i]["score"] for i in qs), mean(b[i]["score"] for i in qs)
        print(f"  {cat:16} n={len(qs):3}  A {ca:.3f}  B {cb:.3f}  B-A {cb - ca:+.3f}")

    better = [i for i in ids if b[i]["score"] > a[i]["score"]]
    worse = [i for i in ids if b[i]["score"] < a[i]["score"]]
    for title, qs in ((f"better in B ({len(better)})", better), (f"worse in B ({len(worse)})", worse)):
        print(f"\n{title}:")
        for i in sorted(qs, key=lambda i: abs(b[i]["score"] - a[i]["score"]), reverse=True):
            print(f"  {i} [{a[i]['category']}] {a[i]['label']} -> {b[i]['label']}")


if __name__ == "__main__":
    main()
