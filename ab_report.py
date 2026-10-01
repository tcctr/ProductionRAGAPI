#!/usr/bin/env python3
"""Compare the variants of an A/B experiment (app/variants.py) from query_log.

Usage:
    python ab_report.py                         # the most recent experiment in query_log
    python ab_report.py --experiment compare-pool --control control --since 2026-10-01

Per variant: requests, error rate (5xx: a server down, the LLM busy), cache hits, then latency of
fresh answers (status 200, not cached) split by plain and compare-mode requests: median and p95 of
the total and median per stage. Cache hits answer in milliseconds whatever the variant, so they're
left out of latency. Each variant's median total is compared with the control's, with a 95%
bootstrap interval (each variant's requests resampled separately: different requests went to each).

Latency and errors only: answer quality needs labels, so check it offline (eval_retrieval.py,
generate_answers.py + judge_answers.py, then compare_runs.py).
"""
import argparse
import random
from statistics import median

import psycopg

from embed_ingest import DATABASE_URL

RESAMPLES = 10_000
STAGES = ("embed", "search", "rerank", "terms", "llm")


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def median_diff_interval(a: list[float], b: list[float], seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of median(b) - median(a), resampling each side on its own."""
    rng = random.Random(seed)
    diffs = sorted(median(rng.choices(b, k=len(b))) - median(rng.choices(a, k=len(a)))
                   for _ in range(RESAMPLES))
    return diffs[int(0.025 * RESAMPLES)], diffs[int(0.975 * RESAMPLES) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", help="experiment name (default: the most recent one logged)")
    ap.add_argument("--control", default="control", help="variant the others are compared with")
    ap.add_argument("--since", help="only requests from this time on, e.g. 2026-10-01 or '2026-10-01 14:00'")
    args = ap.parse_args()

    with psycopg.connect(DATABASE_URL) as conn:
        experiment = args.experiment
        if experiment is None:
            row = conn.execute("SELECT experiment FROM query_log WHERE experiment IS NOT NULL "
                               "ORDER BY id DESC LIMIT 1").fetchone()
            if row is None:
                raise SystemExit("no experiment in query_log yet (set AB_EXPERIMENT and send some queries)")
            experiment = row[0]
        rows = conn.execute(
            """
            SELECT variant, status, cached, compared, total_ms, timings FROM query_log
            WHERE experiment = %s AND (%s::timestamptz IS NULL OR created_at >= %s::timestamptz)
            """, (experiment, args.since, args.since)).fetchall()
    if not rows:
        raise SystemExit(f"no requests logged for experiment {experiment!r}")

    names = sorted({r[0] for r in rows}, key=lambda v: (v != args.control, v))
    print(f"experiment {experiment}: {len(rows)} requests" + (f" since {args.since}" if args.since else ""))
    print(f"\n{'variant':16} {'requests':>8} {'5xx':>6} {'cached':>7}")
    for v in names:
        mine = [r for r in rows if r[0] == v]
        errors = sum(r[1] >= 500 for r in mine)
        cached = sum(bool(r[2]) for r in mine)
        print(f"{v:16} {len(mine):8} {errors / len(mine):6.1%} {cached / len(mine):7.1%}")

    for label, compared in (("plain requests", False), ("compare-mode requests", True)):
        fresh = {v: [r for r in rows if r[0] == v and r[1] == 200 and r[2] is False and r[3] is compared]
                 for v in names}
        if not any(fresh.values()):
            continue
        print(f"\n{label} (fresh 200s), ms")
        print(f"{'variant':16} {'n':>4} {'median':>8} {'p95':>8}  " + "  ".join(f"{s:>7}" for s in STAGES))
        for v, rs in fresh.items():
            if not rs:
                print(f"{v:16} {0:4}")
                continue
            totals = [r[4] for r in rs]
            stage_medians = []
            for s in STAGES:
                values = [r[5][s] for r in rs if s in r[5]]
                stage_medians.append(f"{median(values):7.0f}" if values else f"{'-':>7}")
            print(f"{v:16} {len(rs):4} {median(totals):8.0f} {p95(totals):8.0f}  " + "  ".join(stage_medians))
        control = [r[4] for r in fresh.get(args.control, [])]
        for v in names:
            other = [r[4] for r in fresh[v]]
            if v == args.control or not control or not other:
                continue
            low, high = median_diff_interval(control, other)
            note = "  (contains 0)" if low <= 0 <= high else ""
            small = "  (few requests: read with care)" if min(len(control), len(other)) < 20 else ""
            print(f"  {v} - {args.control}: median {median(other) - median(control):+.0f} ms,"
                  f" 95% CI [{low:+.0f}, {high:+.0f}]{note}{small}")


if __name__ == "__main__":
    main()
