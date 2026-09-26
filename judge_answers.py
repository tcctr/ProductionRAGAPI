#!/usr/bin/env python3
"""Grade saved answers (from generate_answers.py) with an LLM judge plus free checks.

Usage:
    python judge_answers.py data/eval/answers/<name>-<timestamp>.json \\
        --judge-url http://192.168.3.195:8095/v1/chat/completions
    python judge_answers.py <answers.json> --limit 5      # quick try
    python judge_answers.py <answers.json> --ids q060 q093 # re-grade specific questions

Free checks (no LLM):
  citations   every [n] must point to one of the chunks the LLM saw
  refusal     answer opens with "the excerpts do not cover/contain ..." (checked against the judge)

LLM judge, per question: sees the question, the reference answer, the numbered excerpts
and the answer, and returns small labels constrained to a JSON schema:
  claims          each factual claim (code included), supported or unsupported by the excerpts
  contradictions  statements that conflict with the reference answer
  coverage        how much of the reference's key points the answer states: full/partial/none
  refused         whether the answer declines to answer from the excerpts

The verdict is derived from those labels by fixed rules (see verdict()), not asked for.
Each run is saved to data/eval/judgments/<answers file name>-<timestamp>.json.
"""
import argparse
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

from app import generate as llm

VERDICTS = ("correct", "partial", "incorrect", "refused")

JUDGE_PROMPT = """\
You grade answers written by a documentation assistant for PostgreSQL. The assistant was told to \
use only the numbered excerpts below, cite them like [1], and say so when they do not answer the question.

Grade the answer:
1. claims: split the answer into its factual claims, including claims made by SQL examples (syntax, \
option names, defaults). Mark each "supported" if the excerpts state it (in any excerpt, cited or not), \
otherwise "unsupported". Statements that only say what the excerpts do or do not cover are not claims.
2. contradictions: statements in the answer that conflict with the reference answer. Empty if none.
3. coverage: how many of the reference answer's key points the answer states: "full", "partial" or "none".
4. refused: true if the answer declines to answer the question because the excerpts do not cover it.

Be strict: a claim that goes beyond the excerpts is unsupported even if it is true."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {"claim": {"type": "string"},
                           "support": {"type": "string", "enum": ["supported", "unsupported"]}},
            "required": ["claim", "support"]}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "coverage": {"type": "string", "enum": ["full", "partial", "none"]},
        "refused": {"type": "boolean"},
    },
    "required": ["claims", "contradictions", "coverage", "refused"],
}

# [3], [2][3] and [1, 2] all count; each number must be 1..len(chunks).
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Refusals the system prompt asks for: "The provided documentation excerpts do not cover ...".
REFUSAL = re.compile(r"\b(?:do|does)(?: not|n't) (?:cover|contain|include|provide|mention|describe|"
                     r"address|explain|specify|state|answer)\b|\bnot covered\b", re.IGNORECASE)


def citations(answer: str) -> list[int]:
    return [int(n) for group in CITATION.findall(answer) for n in group.split(",")]


def looks_like_refusal(answer: str) -> bool:
    """Only the first sentence: a full answer may say later that a detail isn't covered."""
    return bool(REFUSAL.search(re.split(r"(?<=[.!?])\s", answer, maxsplit=1)[0]))


def judge(judge_url: str, question: dict, rec: dict) -> dict:
    sources = "\n\n---\n\n".join(llm.format_source(n, c) for n, c in enumerate(rec["chunks"], start=1))
    user = (f"Question: {question['question']}\n\nReference answer: {question['reference_answer']}\n\n"
            f"Excerpts:\n\n{sources}\n\n---\n\nAnswer to grade:\n{rec['answer']}")
    resp = requests.post(judge_url, timeout=llm.LLM_TIMEOUT, json={
        "model": llm.LLM_MODEL,
        "messages": [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 2048,
        # llama-server compiles the schema into a grammar, so the reply is always this JSON shape.
        "response_format": {"type": "json_schema", "json_schema": {"name": "grade", "schema": JUDGE_SCHEMA}},
        "chat_template_kwargs": {"enable_thinking": False},
    })
    resp.raise_for_status()
    try:
        return json.loads(resp.json()["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"unexpected judge response: {resp.text[:200]}") from e


def verdict(category: str, grade: dict) -> str:
    unsupported = any(c["support"] == "unsupported" for c in grade["claims"])
    if category == "unanswerable":
        # Declining is the right answer, but not if it then answers anyway from memory.
        return "correct" if grade["refused"] and not unsupported else "incorrect"
    if grade["refused"]:
        return "refused"
    if grade["contradictions"] or grade["coverage"] == "none":
        return "incorrect"
    return "correct" if grade["coverage"] == "full" else "partial"


def summarize(rows: list[dict]) -> dict:
    graded = [r for r in rows if "verdict" in r]
    out = {"n": len(rows), "graded": len(graded)}
    if not graded:
        return out
    for v in VERDICTS:
        out[v] = sum(r["verdict"] == v for r in graded) / len(graded)
    claims = [r for r in graded if r["claims"]]
    # Faithfulness: share of an answer's claims the excerpts support, averaged over answers with claims.
    out["faithfulness"] = (sum(r["supported"] / r["claims"] for r in claims) / len(claims)) if claims else None
    out["fully_faithful"] = sum(r["supported"] == r["claims"] for r in graded) / len(graded)
    out["bad_citations"] = sum(bool(r["bad_citations"]) for r in graded)
    return out


def fmt(m: dict) -> str:
    if not m.get("graded"):
        return f"n={m['n']:3}  not graded"
    faith = f"{m['faithfulness']:.2f}" if m["faithfulness"] is not None else "  - "
    return (f"n={m['n']:3}  " + "  ".join(f"{v}={m[v]:.2f}" for v in VERDICTS)
            + f"  faithful={faith} all={m['fully_faithful']:.2f}  badcite={m['bad_citations']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", type=Path, help="a file saved by generate_answers.py")
    ap.add_argument("--judge-url", default=llm.LLM_URL, help="OpenAI-compatible chat endpoint (default: LLM_URL)")
    ap.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    ap.add_argument("--limit", type=int, help="only the first N answers (for a quick try)")
    ap.add_argument("--ids", nargs="+", help="only these question ids, e.g. q060 q093")
    ap.add_argument("--out-dir", type=Path, default=Path("data/eval/judgments"))
    args = ap.parse_args()

    run = json.loads(args.answers.read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in map(json.loads, args.questions.open(encoding="utf-8"))}
    records = [r for r in run["answers"] if not args.ids or r["id"] in args.ids][:args.limit]

    rows = []
    for i, rec in enumerate(records, start=1):
        row = {"id": rec["id"], "category": rec["category"]}
        if "error" in rec:
            row["skipped"] = "no answer: " + rec["error"]
        else:
            cited = citations(rec["answer"])
            row["bad_citations"] = sorted({n for n in cited if not 1 <= n <= len(rec["chunks"])})
            row["cited"] = len(cited)
            row["refusal_phrase"] = looks_like_refusal(rec["answer"])
            try:
                grade = judge(args.judge_url, questions[rec["id"]], rec)
            except (requests.RequestException, ValueError) as e:
                row["skipped"] = "judge failed: " + str(e)
            else:
                row.update(grade=grade, verdict=verdict(rec["category"], grade), claims=len(grade["claims"]),
                           supported=sum(c["support"] == "supported" for c in grade["claims"]))
        rows.append(row)
        status = row.get("verdict") or "SKIPPED"
        detail = f"  {row['supported']}/{row['claims']} supported" if "verdict" in row else ""
        print(f"[{i:3}/{len(records)}] {rec['id']} {rec['category']:16} {status:9}{detail}")

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    overall = summarize(answerable)

    print(f"\n{run['name']} ({run.get('served_model') or 'unknown model'}), judged by {args.judge_url}\n")
    print(f"{'answerable':18} {fmt(overall)}")
    print("\nby category")
    for cat, rs in sorted(by_cat.items()):
        print(f"  {cat:16} {fmt(summarize(rs))}")
    # The refusal regex is free but brittle; show where it and the judge disagree.
    disagree = [r["id"] for r in rows if "verdict" in r and r["refusal_phrase"] != r["grade"]["refused"]]
    print(f"\nrefusal phrase vs judge disagree on: {', '.join(disagree) or 'none'}")
    skipped = [r["id"] for r in rows if "skipped" in r]
    if skipped:
        print(f"skipped: {', '.join(skipped)}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.answers.stem}-judged-{stamp}.json"
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": stamp, "git_commit": commit, "answers_file": str(args.answers),
            "name": run["name"], "served_model": run.get("served_model"), "judge_url": args.judge_url,
            "overall_answerable": overall,
            "by_category": {c: summarize(rs) for c, rs in by_cat.items()},
            "questions": rows,
        }, indent=2, ensure_ascii=False) + "\n")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
