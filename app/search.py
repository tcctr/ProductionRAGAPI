"""Hybrid (vector + BM25 keyword) search over chunks, reranked, shared by the API and eval_retrieval.py."""
import psycopg
from psycopg.rows import dict_row

from app.observability import timed
from app.rerank import rerank

# Without a version filter, fetch this many times k so k distinct results remain after merging
# the copies of a section that is identical in 16, 17 and 18.
DEDUP_OVERFETCH = 3

# Candidates HNSW keeps while searching (pgvector's default is 40). Measured against exact search
# on the eval questions: at 40, 6 of 100 got the wrong top result and ~8% of the true top 15
# were missed (e.g. "What does the ABORT command do?" never reached sql-abort.html); 200 found
# all of them, for ~5 ms more per search.
EF_SEARCH_MIN = 200

# Hybrid search: candidates taken from each ranked list (vector, keyword) before fusing them.
HYBRID_POOL = 50
# Reciprocal rank fusion constant from Cormack et al. (2009): a chunk scores 1/(RRF_K + rank) per
# list it is in, so a larger constant flattens the gap between the top ranks.
RRF_K = 60

# Hybrid results the reranker reorders. Chunk-level MRR with /query's settings: 0.700 without
# reranking, 0.766 with 20, 0.784 with 50; the reranker takes ~40 ms per chunk, so 20 gets most
# of the gain for ~0.8 s.
RERANK_POOL = 20

# BM25 parameters (the usual defaults): K1 caps how much repeating a word helps, B how much
# a chunk longer than average is penalized (0 = not at all, 1 = fully by length).
BM25_K1 = 1.2
BM25_B = 0.75

# Keyword ranking by BM25, from the chunk_terms inverted index (db/schema.sql, refreshed after
# ingesting). terms: the question's words, stemmed and without stop words, plus its whole
# identifiers and ALL-CAPS word pairs (ident_terms(): json_table, 'at local'), each weighted by its
# IDF, higher the fewer chunks contain it. Words in more than half the chunks
# ("postgresql" is in every breadcrumb) weigh almost nothing and would make every chunk a
# candidate, so they're dropped. A chunk scores, per term it contains,
# idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avg_length)), tf = the term's count in it.
# terms is computed first (MATERIALIZED) and its words passed as an array, so the planner looks
# them up in the chunk_terms index instead of merge-joining the whole index (it can't estimate
# how many words a question has).
BM25_SQL = f"""
    WITH question AS (
        SELECT unnest(tsvector_to_array(to_tsvector('english', %(text)s))) AS word
        UNION
        SELECT ident_terms(%(text)s)
    ), terms AS MATERIALIZED (
        SELECT ct.word, ln(1 + (c.n_chunks - count(*) + 0.5) / (count(*) + 0.5)) AS idf
        FROM question JOIN chunk_terms ct USING (word), corpus_stats c
        GROUP BY ct.word, c.n_chunks
        HAVING count(*) <= c.n_chunks / 2
    ), scores AS (
        SELECT ct.chunk_id,
               sum(t.idf * ct.tf * ({BM25_K1} + 1)
                   / (ct.tf + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * ct.length / c.avg_length))) AS bm25
        FROM chunk_terms ct JOIN terms t USING (word), corpus_stats c
        WHERE ct.word = ANY (ARRAY(SELECT word FROM terms))
        GROUP BY ct.chunk_id
    )
    SELECT {{columns}}, bm25
    FROM scores JOIN chunks ON chunks.id = scores.chunk_id
    WHERE {{where}}
    ORDER BY bm25 DESC, id
    LIMIT %(k)s
"""


def body(content: str) -> str:
    """Chunk text without its breadcrumb line ("PostgreSQL 18 > ..."), the only part that names the version."""
    return content.split("\n\n", 1)[-1]


def search(conn: psycopg.Connection, qvec: str, k: int, version: int | None = None,
           doc_type: str | None = None, dedup: bool = True, query_text: str | None = None,
           rerank_pool: int = RERANK_POOL) -> list[dict]:
    """Top-k chunks for a question, best first.

    With query_text (the question; /query always passes it), hybrid search: the top HYBRID_POOL
    chunks by cosine similarity to qvec (a pgvector literal) and by keyword (BM25) score are
    fused by reciprocal rank fusion into "rrf", then reranked (below). Without it, vector
    search alone, ordered by "similarity" (the cosine similarity, set either way).

    With dedup and no version filter, chunks whose text is identical across versions
    are merged into the best-scoring one, and its "versions" lists every version seen.

    With query_text, the top rerank_pool results by rrf (after merging, so each text is scored
    once) are reordered by the reranker (app/rerank.py), which sets "rerank" on each, and the top
    k of those returned; rerank_pool=0 skips it (the results stay ordered by rrf).
    """
    dedup = dedup and version is None
    rerank_pool = rerank_pool if query_text is not None else 0
    wanted = max(k, rerank_pool)
    fetch_k = wanted * DEDUP_OVERFETCH if dedup else wanted
    if query_text is not None:
        fetch_k = max(fetch_k, HYBRID_POOL)
    where, params = ["embedding IS NOT NULL"], {"q": qvec, "k": fetch_k, "text": query_text}
    if version is not None:
        where.append("version = %(version)s")
        params["version"] = version
    if doc_type is not None:
        where.append("doc_type = %(doc_type)s")
        params["doc_type"] = doc_type

    columns = ("id, version, doc_type, page, section_title, heading_path, url, content, "
               "1 - (embedding <=> %(q)s::vector) AS similarity")
    with timed("search"), conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        # HNSW returns at most ef_search rows and applies filters after the index scan;
        # iterative_scan keeps scanning until LIMIT rows pass the filters.
        # JIT off: the planner overestimates the BM25 query's cost (~280M) and compiled it to
        # machine code for ~240 ms before running it in a few ms.
        # set_config(..., true) scopes these settings to this transaction.
        cur.execute("SELECT set_config('hnsw.ef_search', %s, true), "
                    "set_config('hnsw.iterative_scan', 'strict_order', true), "
                    "set_config('jit', 'off', true)",
                    (str(max(EF_SEARCH_MIN, fetch_k)),))
        cur.execute(
            f"""
            SELECT {columns}
            FROM chunks
            WHERE {' AND '.join(where)}
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(k)s
            """,
            params,
        )
        rows = cur.fetchall()
        if query_text is not None:
            cur.execute(BM25_SQL.format(columns=columns, where=" AND ".join(where)), params)
            rows = fuse([rows, cur.fetchall()])[:fetch_k]

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
    if rerank_pool:
        with timed("rerank"):
            results = rerank(query_text, results[:rerank_pool])
    return results[:k]


def fuse(ranked_lists: list[list[dict]]) -> list[dict]:
    """Reciprocal rank fusion: rows from several best-first lists, ordered by the sum of
    1/(RRF_K + rank) over the lists each row is in (rank 1 = best). Sets "rrf" on each row."""
    fused: dict[str, dict] = {}
    for rows in ranked_lists:
        for rank, r in enumerate(rows, 1):
            row = fused.setdefault(r["id"], {"rrf": 0.0})
            row.update(r)  # the same chunk from another list adds its own score (bm25)
            row["rrf"] += 1 / (RRF_K + rank)
    return sorted(fused.values(), key=lambda r: r["rrf"], reverse=True)
