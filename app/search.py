"""Vector search over chunks, shared by the API and eval_retrieval.py."""
import psycopg
from psycopg.rows import dict_row

# Without a version filter, fetch this many times k so k distinct results remain after merging
# the copies of a section that is identical in 16, 17 and 18.
DEDUP_OVERFETCH = 3


def body(content: str) -> str:
    """Chunk text without its breadcrumb line ("PostgreSQL 18 > ..."), the only part that names the version."""
    return content.split("\n\n", 1)[-1]


def search(conn: psycopg.Connection, qvec: str, k: int, version: int | None = None,
           doc_type: str | None = None, dedup: bool = True) -> list[dict]:
    """Top-k chunks by cosine similarity to qvec (a pgvector literal), best first.

    With dedup and no version filter, chunks whose text is identical across versions
    are merged into the best-scoring one, and its "versions" lists every version seen.
    """
    dedup = dedup and version is None
    fetch_k = k * DEDUP_OVERFETCH if dedup else k
    where, params = ["embedding IS NOT NULL"], {"q": qvec, "k": fetch_k}
    if version is not None:
        where.append("version = %(version)s")
        params["version"] = version
    if doc_type is not None:
        where.append("doc_type = %(doc_type)s")
        params["doc_type"] = doc_type

    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        # HNSW returns at most ef_search rows (default 40), and filters are applied after the
        # index scan; iterative_scan keeps scanning until LIMIT rows pass the filters.
        # set_config(..., true) scopes both settings to this transaction.
        cur.execute("SELECT set_config('hnsw.ef_search', %s, true), "
                    "set_config('hnsw.iterative_scan', 'strict_order', true)",
                    (str(max(40, fetch_k)),))
        cur.execute(
            f"""
            SELECT id, version, doc_type, page, section_title, heading_path, url, content,
                   1 - (embedding <=> %(q)s::vector) AS similarity
            FROM chunks
            WHERE {' AND '.join(where)}
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(k)s
            """,
            params,
        )
        rows = cur.fetchall()

    results: list[dict] = []
    seen: dict[str, dict] = {}
    for r in rows:
        r["similarity"] = float(r["similarity"])
        r["versions"] = [r["version"]]
        if dedup:
            key = body(r["content"])
            if key in seen:
                seen[key]["versions"].append(r["version"])
                continue
            seen[key] = r
        results.append(r)
    for r in results:
        r["versions"].sort()
    return results[:k]
