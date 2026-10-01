"""API tests against the real local database, embedding server and LLM server (see README setup)."""
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from fastapi.testclient import TestClient

import embed_ingest
from app import cache, generate, ratelimit, rerank
from app import main as api
from app.auth import create_key, revoke_key, set_limits
from app.main import app
from app.versions import is_version_question, question_terms

TEST_PAGE = "ragapi-test-page.html"
TEST_KEYS = {"pytest-admin": ["ingest", "query"], "pytest-query": ["query"], "pytest-revoked": ["query"],
             "pytest-limited": ["query"]}


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
        embed_ingest.refresh_bm25_stats(conn)
        embed_ingest.clear_answer_cache(conn)


@pytest.fixture(autouse=True)
def empty_cache():
    """Each test starts with an empty /query cache: the database is shared between test runs,
    and a response cached earlier would skip the failure a test sets up (e.g. a server down)."""
    with psycopg.connect(embed_ingest.DATABASE_URL) as conn:
        embed_ingest.clear_answer_cache(conn)


def delete_test_keys() -> None:
    with psycopg.connect(embed_ingest.DATABASE_URL) as conn:
        conn.execute("DELETE FROM api_keys WHERE name = ANY(%s)", (list(TEST_KEYS),))


@pytest.fixture(scope="module")
def keys() -> dict[str, str]:
    """Test keys by name; pytest-revoked is revoked right after it's created. pytest-admin, which
    the client sends by default, has limits the suite never reaches; pytest-limited allows 2 queries."""
    delete_test_keys()
    with psycopg.connect(embed_ingest.DATABASE_URL) as conn:
        created = {name: create_key(conn, name, scopes) for name, scopes in TEST_KEYS.items()}
        revoke_key(conn, "pytest-revoked")
        for scope in ("query", "ingest"):
            set_limits(conn, "pytest-admin", scope, per_minute=1000, burst=1000)
        set_limits(conn, "pytest-limited", "query", per_minute=1, burst=2)
    yield created
    delete_test_keys()


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture(scope="module")
def client(keys):
    delete_test_page()
    # `with` runs the lifespan: opens the pool, loads the tokenizer. Requests send the admin key
    # unless a test passes its own headers.
    with TestClient(app, headers=bearer(keys["pytest-admin"])) as c:
        yield c
    delete_test_page()


def query(client, **body):
    """Chunks only: retrieval tests skip the LLM, which would add seconds per call."""
    resp = client.post("/query", json={"generate": False} | body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["answer"] is None
    return resp.json()["chunks"]


def test_health(client):
    assert client.get("/health").json() == {"database": "ok", "embeddings": "ok", "reranker": "ok", "llm": "ok"}


@pytest.mark.parametrize("headers", [
    {"Authorization": ""},  # overrides the client's default key: no key at all
    bearer("rag_not-a-real-key"),
    {"Authorization": "Basic dXNlcjpwYXNz"},  # not a Bearer key
])
def test_query_rejects_missing_or_unknown_key(client, headers):
    resp = client.post("/query", json={"question": "x", "generate": False}, headers=headers)
    assert resp.status_code == 401, resp.text
    assert resp.headers["WWW-Authenticate"] == "Bearer"


def test_revoked_key_is_rejected(client, keys):
    resp = client.post("/query", json={"question": "x", "generate": False}, headers=bearer(keys["pytest-revoked"]))
    assert resp.status_code == 401
    assert resp.json()["detail"] == "invalid or revoked API key"


def test_scopes(client, keys):
    headers = bearer(keys["pytest-query"])
    resp = client.post("/query", json={"question": "round a number", "generate": False, "k": 1}, headers=headers)
    assert resp.status_code == 200, resp.text
    # Rejected before the endpoint runs: nothing is ingested.
    resp = client.post("/ingest", json=page(sections=0), headers=headers)
    assert resp.status_code == 403
    assert "lacks the 'ingest' scope" in resp.json()["detail"]


def test_rate_limit(client, keys):
    def ask():
        return client.post("/query", json={"question": "round a number", "generate": False, "k": 1},
                           headers=bearer(keys["pytest-limited"]))

    for remaining in (1, 0):  # burst 2: two requests at once are fine
        resp = ask()
        assert resp.status_code == 200, resp.text
        assert resp.headers["RateLimit-Limit"] == "2" and resp.headers["RateLimit-Remaining"] == str(remaining)
    resp = ask()
    assert resp.status_code == 429
    assert 1 <= int(resp.headers["Retry-After"]) <= 60  # 1 per minute: the next token is < 60 s away
    assert resp.headers["RateLimit-Remaining"] == "0"

    # A minute later the bucket has refilled one token (moving updated_at back stands in for waiting).
    with psycopg.connect(embed_ingest.DATABASE_URL) as conn:
        conn.execute("UPDATE rate_buckets SET updated_at = updated_at - interval '60 seconds' "
                     "WHERE key_id = (SELECT id FROM api_keys WHERE name = 'pytest-limited')")
    resp = ask()
    assert resp.status_code == 200, resp.text
    assert resp.headers["RateLimit-Remaining"] == "0"


def test_rate_limit_holds_under_concurrency(keys):
    """20 simultaneous requests on a bucket of 5 (separate connections, like separate workers): the
    row lock lets exactly 5 through."""
    with psycopg.connect(embed_ingest.DATABASE_URL) as conn:
        key_id = conn.execute("SELECT id FROM api_keys WHERE name = 'pytest-query'").fetchone()[0]
        conn.execute("DELETE FROM rate_buckets WHERE key_id = %s", (key_id,))
    start = threading.Barrier(20)

    def take(_):
        with psycopg.connect(embed_ingest.DATABASE_URL) as conn:
            start.wait()
            return ratelimit.take(conn, key_id, "query", {"per_minute": 0.001, "burst": 5}).retry_after == 0

    with ThreadPoolExecutor(20) as pool:
        assert sum(pool.map(take, range(20))) == 5


def test_llm_busy_returns_503(client, monkeypatch):
    full = threading.BoundedSemaphore(1)
    full.acquire()  # every slot taken by other requests
    monkeypatch.setattr(api, "llm_slots", full)
    resp = client.post("/query", json={"question": "round a number", "k": 1})
    assert resp.status_code == 503
    assert resp.json()["detail"].startswith("LLM busy")
    assert resp.headers["Retry-After"] == "5"


def test_health_hides_error_details(client, monkeypatch):
    monkeypatch.setattr(rerank, "RERANK_URL", "http://localhost:1/v1/rerank")
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["detail"]["reranker"] == "error"


def test_health_needs_no_key(client):
    assert client.get("/health", headers={"Authorization": ""}).status_code in (200, 503)


@pytest.mark.parametrize("body", [
    {"question": ""},
    {"question": "x", "version": 15},
    {"question": "x", "k": 0},
    {"question": "x", "k": 51},
    {"question": "x", "doc_type": "tutorial"},
    {"question": "x", "generate": "maybe"},
    {"question": "x", "version": 17, "compare_versions": True},
])
def test_query_rejects_invalid_input(client, body):
    assert client.post("/query", json=body).status_code == 422


def test_query_version_filter(client):
    chunks = query(client, question="how do I create an index without locking the table", version=17, k=10)
    assert len(chunks) == 10
    assert all(c["version"] == 17 and c["versions"] == [17] for c in chunks)
    scores = [c["rerank"] for c in chunks]
    assert scores == sorted(scores, reverse=True)


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

    # /ingest refreshed the keyword index: a made-up identifier, which vector search alone can't
    # match (~0.48 similarity), is found by its rare words (zorblax, frobnic).
    top = query(client, question="What does zorblax_frobnicate do?", k=1)[0]
    assert top["page"] == TEST_PAGE

    # Unchanged page: nothing to embed, and the cached responses stay.
    again = client.post("/ingest", json=page(sections=6)).json()
    assert again["chunks_embedded"] == 0 and again["chunks_deleted"] == 0
    zorblax = {"question": "What does zorblax_frobnicate do?", "k": 1, "generate": False}
    assert client.post("/query", json=zorblax).json()["cached"]

    # Shrunk page: fewer chunks, the stale ones are deleted, and so are the cached responses
    # (the one above cited a chunk that may be gone).
    shrunk = client.post("/ingest", json=page(sections=0)).json()
    assert shrunk["chunks_total"] == 1
    assert shrunk["chunks_deleted"] == first["chunks_total"] - 1
    assert not client.post("/query", json=zorblax).json()["cached"]


def test_ingest_rejects_bad_page_name(client):
    body = page(sections=0) | {"page": "../etc/passwd"}
    assert client.post("/ingest", json=body).status_code == 422


def test_embedding_server_down_returns_503(client, monkeypatch):
    monkeypatch.setattr(embed_ingest, "EMBED_URL", "http://localhost:1/v1/embeddings")
    resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 503
    assert "embedding server unavailable" in resp.json()["detail"]


def test_reranker_down_returns_503(client, monkeypatch):
    monkeypatch.setattr(rerank, "RERANK_URL", "http://localhost:1/v1/rerank")
    resp = client.post("/query", json={"question": "anything", "generate": False})
    assert resp.status_code == 503
    assert "reranker unavailable" in resp.json()["detail"]


def test_repeated_query_is_cached(client):
    body = {"question": "How do I round a number?", "k": 3, "generate": False}
    first = client.post("/query", json=body).json()
    assert first["cached"] is False
    # Same question up to whitespace: answered from the cache, echoing the question as asked.
    spaced = body | {"question": "  How do I   round a number? "}
    second = client.post("/query", json=spaced).json()
    assert second["cached"] is True
    assert second["question"] == spaced["question"]
    assert second | {"question": body["question"], "cached": False} == first
    # Different options are a different request.
    assert client.post("/query", json=body | {"k": 4}).json()["cached"] is False


def test_cached_answer_skips_the_pipeline(client, monkeypatch):
    calls = []
    monkeypatch.setattr(api.llm, "generate", lambda q, chunks, presence: calls.append(q) or "Use round() [1].")
    body = {"question": "How do I round a number?", "k": 1}
    first = client.post("/query", json=body)
    assert first.status_code == 200 and calls == [body["question"]]
    # With the embedding server gone, only a cache hit can still answer.
    monkeypatch.setattr(embed_ingest, "EMBED_URL", "http://localhost:1/v1/embeddings")
    second = client.post("/query", json=body)
    assert second.status_code == 200, second.text
    assert second.json()["cached"] and second.json()["answer"] == "Use round() [1]."
    assert len(calls) == 1


def test_cache_key():
    def key(**body):
        return cache.cache_key(api.QueryRequest(**({"question": "Which version added AT LOCAL?"} | body)))
    assert key() == key(question=" Which  version added\tAT LOCAL? ")
    # Case is kept: ALL-CAPS words are looked up as SQL keywords (question_terms).
    assert key() != key(question="which version added at local?")
    assert key() != key(version=17) != key(generate=False)


def test_version_question_detection_and_terms():
    assert is_version_question("Since which version does EXPLAIN support the SERIALIZE option?")
    assert is_version_question("When were the uuidv4() and uuidv7() functions added?")
    assert not is_version_question("In PostgreSQL 17, how can COPY skip rows with malformed data?")
    assert question_terms("Which version added the casefold() function?") == ["casefold"]
    assert question_terms("Which version added the AT LOCAL operator?") == ["AT LOCAL"]
    assert question_terms("Which version added the gamma and lgamma functions?") == ["gamma", "lgamma"]


def test_query_compares_versions(client):
    resp = client.post("/query", json={"question": "Which version added the casefold() function?",
                                       "generate": False})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The exact text search sees every chunk: casefold is only documented in 18.
    assert body["version_presence"] == {"casefold": [18]}
    # Every version is searched separately, so each one is represented.
    assert {v for c in body["chunks"] for v in c["versions"]} == {16, 17, 18}


def test_query_without_version_question_skips_comparison(client):
    resp = client.post("/query", json={"question": "What does the casefold() function do?", "generate": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["version_presence"] is None


def test_term_search_goes_into_prompt():
    chunk = {"version": 18, "versions": [18], "content": "PostgreSQL 18 > 9.4. String Functions\n\ncasefold"}
    system, user = (m["content"] for m in generate.build_messages("q", [chunk], {"casefold": [18]}))
    assert "- `casefold`: in 18; not in 16, 17" in user
    assert "term search" in system
    assert "term search" not in generate.build_messages("q", [chunk])[0]["content"]


def test_query_answers_with_citations(client):
    # A version question: JSON_TABLE is in the 17 and 18 docs, not in 16. The first sentence must
    # name 17 alone; a later "not in 16" or "also in 18" is fine.
    resp = client.post("/query", json={"question": "Which PostgreSQL version introduced JSON_TABLE?"})
    assert resp.status_code == 200, resp.text
    answer = resp.json()["answer"]
    first = re.split(r"(?<=[.!?])\s", answer, maxsplit=1)[0]
    assert "17" in first and "18" not in first and "16" not in first, answer
    assert re.search(r"\[\d+\]", answer), answer


def test_merged_chunk_lists_all_versions_in_prompt():
    chunk = {"version": 18, "versions": [16, 17, 18],
             "content": "PostgreSQL 18 > CREATE INDEX\n\nCREATE INDEX — define a new index"}
    user = generate.build_messages("q", [chunk])[1]["content"]
    assert "[1] PostgreSQL 16, 17, 18 > CREATE INDEX" in user


def test_llm_server_down_returns_503(client, monkeypatch):
    monkeypatch.setattr(generate, "LLM_URL", "http://localhost:1/v1/chat/completions")
    resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 503
    assert "LLM server unavailable" in resp.json()["detail"]
