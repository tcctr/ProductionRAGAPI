"""Reranking search candidates with a cross-encoder served by llama.cpp (llama-server --reranking).

Search scores the question and each chunk separately (vectors computed in advance, shared
words). A cross-encoder reads the question and one chunk together and returns one relevance
score, which is more accurate but too slow for the whole corpus, so it only reorders the
candidates hybrid search found.
"""
import os

import requests

RERANK_URL = os.environ.get("RERANK_URL", "http://localhost:8083/v1/rerank")
RERANK_TIMEOUT = float(os.environ.get("RERANK_TIMEOUT", "30"))


def rerank(question: str, chunks: list[dict]) -> list[dict]:
    """The chunks sorted by relevance to the question, best first. Sets "rerank" on each chunk:
    the model's raw score (a logit, can be negative; only the order matters)."""
    if not chunks:
        return chunks
    resp = requests.post(RERANK_URL, timeout=RERANK_TIMEOUT,
                         json={"query": question, "documents": [c["content"] for c in chunks]})
    resp.raise_for_status()
    results = resp.json()["results"]
    if sorted(r["index"] for r in results) != list(range(len(chunks))):
        raise ValueError(f"expected {len(chunks)} rerank scores, got {len(results)}")
    for r in results:
        chunks[r["index"]]["rerank"] = float(r["relevance_score"])
    return sorted(chunks, key=lambda c: c["rerank"], reverse=True)
