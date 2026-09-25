"""API tests against the real local database and embedding server (see README setup)."""
import psycopg
import pytest
from fastapi.testclient import TestClient

import embed_ingest
from app.main import app

TEST_PAGE = "ragapi-test-page.html"


def page_text(sections: int) -> str:
    """A made-up page, long enough to need several chunks when sections > ~3."""
    parts = ["# 99.1. Zorblax Functions [#](#FUNCTIONS-ZORBLAX)",
             "The zorblax_frobnicate function reticulates splines inside a quantum teapot."]
    for i in range(sections):
        parts.append(f"## 99.1.{i}. Zorblax Variant {i} [#](#ZORBLAX-{i})")
        parts.append(" ".join(f"Variant {i} step {j} frobnicates the teapot spline number {j}." for j in range(40)))
    return "\n\n".join(parts)


def page(sections: int) -> dict:
    return {"version": 18, "doc_type": "functions", "section_title": "Zorblax Functions",
            "page": TEST_PAGE, "url": f"https://example.com/{TEST_PAGE}", "text": page_text(sections)}


def delete_test_page() -> None:
    with psycopg.connect(embed_ingest.DATABASE_URL) as conn:
        conn.execute("DELETE FROM documents WHERE page = %s", (TEST_PAGE,))


@pytest.fixture(scope="module")
def client():
    delete_test_page()
    with TestClient(app) as c:  # `with` runs the lifespan: opens the pool, loads the tokenizer
        yield c
    delete_test_page()


def query(client, **body):
    resp = client.post("/query", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()["chunks"]


def test_health(client):
    assert client.get("/health").json() == {"database": "ok", "embeddings": "ok"}


@pytest.mark.parametrize("body", [
    {"question": ""},
    {"question": "x", "version": 15},
    {"question": "x", "k": 0},
    {"question": "x", "k": 51},
    {"question": "x", "doc_type": "tutorial"},
])
def test_query_rejects_invalid_input(client, body):
    assert client.post("/query", json=body).status_code == 422


def test_query_version_filter(client):
    chunks = query(client, question="how do I create an index without locking the table", version=17, k=10)
    assert len(chunks) == 10
    assert all(c["version"] == 17 and c["versions"] == [17] for c in chunks)
    sims = [c["similarity"] for c in chunks]
    assert sims == sorted(sims, reverse=True)


def test_query_doc_type_filter(client):
    chunks = query(client, question="round a number", version=16, doc_type="functions", k=20)
    assert len(chunks) == 20
    assert all(c["doc_type"] == "functions" and c["version"] == 16 for c in chunks)


def test_query_dedup_merges_identical_sections(client):
    chunks = query(client, question="how do I create an index without locking the table", k=10)
    bodies = [c["content"].split("\n\n", 1)[1] for c in chunks]
    assert len(bodies) == len(set(bodies)) == 10
    assert any(len(c["versions"]) > 1 for c in chunks)


def test_ingest_lifecycle(client):
    resp = client.post("/ingest", json=page(sections=6))
    assert resp.status_code == 200, resp.text
    first = resp.json()
    assert first["id"] == f"18:{TEST_PAGE}"
    assert first["chunks_total"] > 1
    assert first["chunks_embedded"] == first["chunks_total"]

    top = query(client, question="Zorblax Functions reticulate splines in a teapot", version=18, k=1)[0]
    assert top["page"] == TEST_PAGE

    # Unchanged page: nothing to embed.
    again = client.post("/ingest", json=page(sections=6)).json()
    assert again["chunks_embedded"] == 0 and again["chunks_deleted"] == 0

    # Shrunk page: fewer chunks, the stale ones are deleted.
    shrunk = client.post("/ingest", json=page(sections=0)).json()
    assert shrunk["chunks_total"] == 1
    assert shrunk["chunks_deleted"] == first["chunks_total"] - 1


def test_ingest_rejects_bad_page_name(client):
    body = page(sections=0) | {"page": "../etc/passwd"}
    assert client.post("/ingest", json=body).status_code == 422


def test_embedding_server_down_returns_503(client, monkeypatch):
    monkeypatch.setattr(embed_ingest, "EMBED_URL", "http://localhost:1/v1/embeddings")
    resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 503
    assert "embedding server unavailable" in resp.json()["detail"]
