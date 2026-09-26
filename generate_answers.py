#!/usr/bin/env python3
"""Answer every eval question with one LLM and save the answers for grading.

Usage:
    python generate_answers.py --name qwen3.6-35b-a3b --llm-url http://192.168.3.195:8095/v1/chat/completions
    python generate_answers.py --name qwen3.5-9b      --llm-url http://localhost:8082/v1/chat/completions

Runs the same steps as /query (embed, search with the question's version filter and
cross-version dedup, generate), calling app/search.py and app/generate.py directly so
no API server is needed. Saves the question, the chunks the LLM saw (numbered like its
citations) and the answer, so judge_answers.py can grade without regenerating.

A question whose LLM call fails is saved with "error" instead of an answer, and the
run goes on. Each run is saved to data/eval/answers/<name>-<timestamp>.json.
"""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import requests

from app import generate as llm
from app.search import search
from embed_ingest import DATABASE_URL, EMBED_MODEL, QUERY_PREFIX, embed, to_pgvector

# Chunk fields worth keeping: what the LLM saw, plus what grading needs to match pages.
CHUNK_FIELDS = ("id", "version", "versions", "page", "url", "similarity", "content")


def served_model(llm_url: str) -> str | None:
    """The model file llama-server reports, so a run records what actually answered."""
    try:
        resp = requests.get(llm_url.split("/v1/")[0] + "/v1/models", timeout=5)
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="label for this model, used in the file name")
    ap.add_argument("--llm-url", default=llm.LLM_URL, help="OpenAI-compatible chat endpoint (default: LLM_URL)")
    ap.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    ap.add_argument("--k", type=int, default=5, help="chunks per answer (/query's default)")
    ap.add_argument("--limit", type=int, help="only the first N questions (for a quick try)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/eval/answers"))
    args = ap.parse_args()

    llm.LLM_URL = args.llm_url  # generate() reads the module setting
    model = served_model(args.llm_url)
    print(f"model: {model or 'unknown'} at {args.llm_url}")

    questions = [json.loads(line) for line in args.questions.open(encoding="utf-8")][:args.limit]
    vectors = embed([QUERY_PREFIX + q["question"] for q in questions])

    records, errors, total_s = [], 0, 0.0
    with psycopg.connect(DATABASE_URL) as conn:
        for i, (q, vec) in enumerate(zip(questions, vectors), start=1):
            chunks = search(conn, to_pgvector(vec), args.k, q["version"])
            rec = {"id": q["id"], "category": q["category"], "question": q["question"],
                   "chunks": [{f: c[f] for f in CHUNK_FIELDS} for c in chunks]}
            start = time.perf_counter()
            try:
                rec["answer"] = llm.generate(q["question"], chunks)
            except (requests.RequestException, ValueError) as e:
                rec["error"] = str(e)
                errors += 1
            rec["seconds"] = round(time.perf_counter() - start, 2)
            total_s += rec["seconds"]
            records.append(rec)
            status = "ERROR" if "error" in rec else f"{len(rec['answer'])} chars"
            print(f"[{i:3}/{len(questions)}] {q['id']} {rec['seconds']:5.1f}s  {status}")

    answered = len(records) - errors
    print(f"\n{answered}/{len(records)} answered, {errors} errors, "
          f"mean {total_s / len(records):.1f}s per question")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.name}-{stamp}.json"
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": stamp, "git_commit": commit, "name": args.name, "served_model": model,
            "llm_url": args.llm_url, "temperature": llm.TEMPERATURE, "embed_model": EMBED_MODEL,
            "k": args.k, "filter_version": True, "dedup": True,
            "answered": answered, "errors": errors,
            "answers": records,
        }, indent=2, ensure_ascii=False) + "\n")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
