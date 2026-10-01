# ProductionRAGAPI

A retrieval-augmented generation (RAG) API over the official PostgreSQL documentation (versions 16, 17 and 18), built to production standards: hybrid search, reranking, auth, rate limiting, caching, evals, monitoring and A/B testing.

The corpus covers three sections of the docs: the SQL command reference, functions and operators, and indexes and performance tips.

## How it works

```
postgresql.org ──fetch──▶ raw HTML ──parse──▶ docs.jsonl (704 pages)
                                                   │ chunk
                                                   ▼
                                          chunks.jsonl (~4.3k chunks)
                                                   │ embed (nomic-embed-text-v1.5)
                                                   ▼
                                      Postgres + pgvector (HNSW + BM25 word index)
                                                   │
question ──embed + keywords──▶ hybrid search ──────┘──▶ rerank top 20 ──▶ prompt with top 5 ──▶ LLM ──▶ cited answer
   └─ asked before (same question and options)? ──▶ response cache in Postgres, ~6 ms instead of ~6 s
```

## Status

| Component | Status |
|---|---|
| Docs crawler and markdown parser | Done |
| Header-aware chunking | Done |
| pgvector schema | Done |
| Embedding and ingestion | Done |
| Evaluation question set and retrieval eval | Done |
| FastAPI `/ingest` and `/query` | Done |
| Answer generation with a local LLM | Done |
| Answer evaluation (LLM judge) | Done |
| Hybrid search (vector + BM25) | Done |
| Reranking (cross-encoder) | Done |
| Auth (API keys with scopes) and rate limiting | Done |
| Caching | Done |
| Observability (per-stage timings, request IDs, query log) | Done |
| A/B testing (paired offline comparison, per-key variants online) | Done |

## Design decisions

**Chunking.** Pages that fit in one chunk stay whole (253 of 704, mostly short SQL command pages). Longer pages are split on markdown headers and packed to about 450 tokens, breaking between sections where possible. Blocks that are still too large are split by line: tables repeat their header row and caption in every piece, and code blocks are re-opened so each piece stays valid markdown.

**Breadcrumbs.** Every chunk starts with a line like `PostgreSQL 18 > CREATE INDEX > Parameters`. A chunk from the middle of a page would otherwise not say which command or which version it belongs to, and version confusion is the main risk with this corpus. The same text is shown to the LLM, so it also knows where each passage comes from.

**Exact token budget.** Chunks are measured with the embedding model's own tokenizer, including the `search_document: ` task prefix and special tokens, and capped at 512 tokens: the default batch size of llama.cpp's embedding server. The Hugging Face tokenizer collapses any word over 100 characters (such as a hex hash) into a single `[UNK]` token while llama.cpp splits it, which undercounted three chunks by up to 84 tokens; the limit is raised so counts match llama.cpp's `/tokenize` exactly for every chunk.

**Incremental ingestion.** Pages and chunks are upserted by stable IDs (`18:sql-createindex.html#3`). A chunk whose content hash changed loses its vector; only chunks without a vector from the current model are embedded, in batches committed one at a time, so an interrupted run resumes and a repeated run does nothing. Embedding the full corpus takes about 70 seconds on a laptop.

**Task prefixes.** nomic-embed-text is trained with instructions: documents are embedded as `search_document: …` and questions as `search_query: …`. Omitting them measurably hurts retrieval.

**Cross-version dedup.** Most sections are identical in 16, 17 and 18, so a plain top-5 often returned the same passage three times. Without a version filter, `/query` fetches 3×k candidates and merges chunks whose text (minus the breadcrumb) is identical into the best-scoring one, listing every version it applies to in `versions`. This raises hit@5 from 0.87 to 0.93 and paraphrase hit@5 from 0.64 to 0.82. Sections that changed between versions stay separate, so version differences remain visible.

**HNSW recall.** HNSW is an approximate index: it follows links between similar vectors and keeps only `hnsw.ef_search` candidates (default 40) while it searches, so it can miss the true best match. Compared with an exact scan on the eval questions, `ef_search = 40` gave 6 of 100 questions the wrong top result and missed about 8% of the true top 15; "What does the ABORT command do?" never reached the ABORT page and returned `CREATE POLICY` passages instead. At 200 every result matched the exact scan, for about 5 ms more per search, and the eval's MRR went from 0.759 to 0.808. The scan also applies `WHERE` filters after the index, so each query raises `ef_search` to at least 200 or the number of rows it fetches, and enables pgvector 0.8's `iterative_scan` to keep scanning until enough rows pass the filters; both settings are scoped to the query's transaction.

**Sync endpoints and a connection pool.** Endpoints are plain functions that FastAPI runs in a thread pool, which keeps the blocking psycopg and embedding calls simple; the embedding server, not Python, is the bottleneck. A `psycopg_pool` pool reuses database connections across requests.

**Local answer generation.** Answers come from a local model (Qwen3.6-35B-A3B, Q4 GGUF) behind llama.cpp's OpenAI-compatible chat endpoint, so any compatible server can replace it through `LLM_URL`. It is a mixture-of-experts model: 35B parameters, but only about 3B are used per token, which gives large-model answers at small-model speed (about 4 seconds per answer, ~3k prompt tokens). The retrieved chunks are numbered in the prompt in the order `/query` returns them, so a citation `[2]` in the answer points at `chunks[1]` in the response. The system prompt allows only facts from the excerpts, asks for the version each statement applies to, and asks the model to say when the excerpts don't cover the question; a chunk merged across versions gets all of them in its breadcrumb (`PostgreSQL 16, 17, 18 > ...`). Thinking mode is off and the temperature is 0.2: the answer is already in the excerpts. The database connection goes back to the pool before the LLM call, so slow answers don't hold connections that searches need.

**Version-comparison questions.** "Which version added X?" is hard for plain retrieval: the docs never say "added in 17", one search over all versions often returns nothing from the version that lacks X, and a few chunks can't show that a version lacks it anyway. When the question asks which version (detected by a regex that matches all 14 such eval questions and none of the other 86, or forced with `compare_versions`), `/query` searches each version separately (2 chunks each for k = 5) and runs an exact text search for the question's identifiers (`casefold()`, `JSON_TABLE`, `AT LOCAL`) over every chunk of every version. The result goes above the excerpts as a term search (`casefold`: in 18; not in 16, 17) with rules to state it as fact, to call the first covered version that documents something the one that added it, and not to take a term's presence as the feature's (MERGE and RETURNING both exist in 16; MERGE ... RETURNING doesn't).

**Hybrid search.** Vector search matches meaning but blurs exact names: `jsonb_set` and `jsonb_insert` embed almost alike, and a made-up identifier like `zorblax_frobnicate` scored only 0.48 against its own page. So `/query` also ranks chunks by keyword with BM25 and merges the two top-50 lists by reciprocal rank fusion: each list adds 1/(60 + rank) to a chunk's score, so a chunk both lists rank well wins, and scores on different scales are never compared. Postgres's own ranking (`ts_rank_cd`) came first and made retrieval worse (chunk-level MRR 0.585 vs 0.665). It has no notion of how rare a word is, so in "What is a BRIN index good for?" `index` (in 784 chunks) counted as much as `brin` (in 32), and its keyword list alone had the answering chunk in the top 10 for only 48% of questions. BM25 weights each word by its rarity (IDF), gives diminishing returns for repeats and normalizes by chunk length. Postgres has no built-in BM25, so a materialized view `chunk_terms` serves as an inverted index: one row per word and chunk with the word's count and the chunk's length (267k rows, 22 MB), refreshed in about 0.3 s after every ingest, so a query reads only its own words' rows. Words in more than half the chunks (`postgresql`, in every breadcrumb) are skipped. The keyword query takes a few ms; the first version took 220 ms, almost all of it JIT compilation triggered by a badly overestimated plan cost, so search transactions turn JIT off. With `/query`'s settings, chunk-level hit@1 rises from 0.55 to 0.60 and MRR from 0.665 to 0.700 (21 questions rank better, 12 worse), most for paraphrase and version_specific questions. The losses are mostly identifiers the English tokenizer splits into common words (`JSON_TABLE` → `json`, `tabl`; the `AT` of `AT LOCAL` is a stop word) and cross-page questions.

**Reranking.** Both search methods score the question and a chunk separately: a chunk's vector is computed at ingest time, before any question exists, and BM25 only counts shared words. So "How can I lock the rows I read …?" ranked five `LOCK TABLE` passages (full of "lock" and "transaction") above the `SELECT ... FOR UPDATE` clause that answers it, at rank 7. A reranker is a cross-encoder: a small model that reads the question and one chunk together and returns one relevance score, so it can tell that `LOCK TABLE` "deals only with table-level locks" and doesn't answer a question about rows. It is too slow to score the whole corpus at query time, since nothing can be computed in advance, so it reorders only the top 20 hybrid results (after cross-version merging, so each text is scored once), and the top k of those are returned. The model is [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) (568M parameters, Q8 GGUF) behind llama.cpp's `/v1/rerank` endpoint, about 40 ms per chunk on an M4 Max. With `/query`'s settings, chunk-level hit@1 rises from 0.61 to 0.71 and MRR from 0.708 to 0.785 (21 questions rank better, 11 worse, mostly by one place), most for version_diff (MRR 0.651 → 0.875), version_specific (0.776 → 0.856) and paraphrase (0.516 → 0.595); the `FOR UPDATE` passage is now first. Reranking 50 candidates instead of 20 raised MRR only a little more (0.784 vs 0.766 on the earlier labels) for 2.5 s instead of 0.8 s per search. The reranker can only reorder what hybrid search finds: when the answering chunk isn't among the 20 (COALESCE for "show a default value instead of NULL"), it doesn't help. Comparing versions reranks each version's candidates separately. The search keeps its database connection during the reranker call (~0.8 s).

**API keys.** `/query` and `/ingest` need an `Authorization: Bearer <key>` header; `/health` stays open for monitoring. Without keys, anyone who could reach the API could write pages into the corpus that the LLM would then quote as documentation, or keep the single-slot LLM busy. Keys are created with `manage_keys.py`, which prints a key once: the database stores only its SHA-256 hash, and a request's key is hashed and looked up, so the server never needs the key itself and a leaked `api_keys` table holds nothing that works as a key. A fast hash is enough because keys are 32 random bytes; slow hashes like bcrypt protect guessable passwords. Each key has scopes (`query`, `ingest`), so a key given to a front end can't ingest. A revoked key is marked, not deleted, and stops working on the next request. A missing, unknown or revoked key gets 401, a key without the endpoint's scope 403, both before the endpoint runs. Keys travel in plain text in each request, so a deployment needs HTTPS (usually a reverse proxy in front of uvicorn).

**Rate limiting.** Each key has a token bucket per scope: it holds up to `burst` tokens (10 for queries, 20 for ingests by default), refills at `per_minute` (10 and 20) and each request takes one, so a client can send a short burst and then a steady rate. Simpler fixed windows ("30 per minute, reset on the minute") let a client send twice the limit across a window boundary. Buckets live in Postgres rather than in process memory, so they survive restarts and stay correct with several uvicorn workers; a request costs one upsert that computes the refill since the last request and takes a token only if one is there, and the row lock lets exactly 5 of 20 simultaneous requests through a bucket of 5 (tested). Responses carry `RateLimit-Limit` and `RateLimit-Remaining`; a request over the limit gets 429 with `Retry-After`. A rate limit doesn't bound how many answers wait for the LLM at once across keys, and the LLM server answers one at a time, so `/query` also lets at most `LLM_MAX_PENDING` (4) answers be pending and returns 503 with `Retry-After` beyond that instead of queueing for minutes. `/health` needs no key, so it reports only `ok` or `error` per service; the details, which include internal addresses, go to the server log.

**Caching.** A `/query` answer takes ~6 s, almost all of it the reranker (~0.8 s) and the LLM, which answers one request at a time. Responses are cached in Postgres by an exact key: the question with its whitespace collapsed, the request options, and the settings that shape the answer (embedding model, search and rerank constants, LLM server and model, prompt text), so changing one of those makes old entries stop matching. A repeated question takes ~6 ms instead of ~6 s (measured through uvicorn, with and without an answer) and never reaches the LLM. Case is kept in the key: ALL-CAPS words are searched as SQL keywords, so "AT LOCAL" and "at local" can get different answers. There is no semantic cache (reusing the answer of a *similar* question): "Which version added X?" questions about different features embed almost identically, and cosine similarity already failed to tell out-of-scope questions apart, so a similarity threshold would serve confidently wrong answers. The cache is emptied whenever the searchable chunks change (deleted by `/ingest` or `embed_ingest.py`, or newly embedded), and entries expire after `CACHE_TTL_DAYS` (7). Errors and empty results are never cached, a cache hit still takes a rate-limit token, and responses carry `cached: true/false`. Postgres rather than an in-process dict or Redis: it survives restarts, is shared by uvicorn workers, and adds no service. The evals call the pipeline directly and never see the cache.

**Observability.** Every request gets an ID and a record of how long each stage took: `auth` (key check and rate limit), `cache_get`, `embed`, `search` (the SQL), `rerank`, `terms` (the version comparison's text scan), `llm` and `cache_put`. The record is reported three ways. A log line per request (`/query 200 key=demo cached=false compared=true chunks=4 prompt_tokens=1796 ... embed=17 search=60 rerank=2109 llm=2561 total=4783`), with the request ID on every log line written during the request. The `X-Request-ID` and `Server-Timing` response headers: browser devtools draw `Server-Timing` as a timeline, and a client's own `X-Request-ID` is reused so a request can be followed across services. And a `query_log` table, one row per authenticated `/query` (errors and cache hits included) with the question, options, status, short error cause, stage timings and the LLM's token counts, so latency questions become SQL (`percentile_cont(0.95) WITHIN GROUP (ORDER BY (timings->>'rerank')::float)`). The first measurement already showed something: for a "which version" question, reranking takes 2.1 s, as long as the LLM, because each version's 20 candidates are scored separately. The stages are timed with a context variable holding the current request's record: the pipeline code calls `timed("rerank")` without passing anything through its functions, and outside a request (the eval scripts) it does nothing. Unauthenticated requests are logged but not stored, so anyone who can reach the API can't fill the table. Postgres and logs rather than Prometheus or tracing (OpenTelemetry, Langfuse): one user and one server don't need another service yet, and the table is where A/B test results will go.

**A/B testing.** Two halves, because the two kinds of evidence come from different places. Answer quality needs labels, which only the eval set has, so a change is first measured offline: `compare_runs.py` pairs two runs question by question and reports the difference with a bootstrap interval and McNemar's test. Pairing matters with 92 questions: the same question answered by both configurations cancels out how hard it is. Latency and errors under real use come from live traffic: with `AB_EXPERIMENT` set, each API key is assigned a variant, a set of pipeline settings (today the reranker's candidate counts), by hashing the experiment and key names. Assignment is sticky, so one client always sees the same behavior, and needs no assignment table; a new experiment name reshuffles the keys. The variant goes into the log line and `query_log`, and its settings into the cache key, so variants never share cached answers. `ab_report.py` compares variants on fresh responses only (a cache hit is fast whatever the variant), split into plain and version-comparison requests, with a bootstrap interval on the median. The first experiment, version comparison reranking 3 × 10 candidates instead of 3 × 20, measured on 14 version questions per variant: median 2.36 s → 1.27 s (−1.09 s, 95% interval −1.25 to −0.86), while plain requests, which the setting doesn't touch, moved by +23 ms (interval containing 0, a built-in A/A check). Whether the smaller pool costs answer quality is the offline half's question.

**One database.** pgvector keeps vectors, metadata and keyword search in Postgres. Chunks carry `version` and `doc_type` directly, so filtered searches need no join. An HNSW index serves vector search, and a generated `tsvector` column (stemmed words without stop words) feeds the BM25 word index. A `content_hash` per chunk lets re-ingestion skip unchanged text.

## Evaluation

100 hand-written questions in `data/eval/questions.jsonl`, labeled with the pages that answer them:

| Category | n | Tests |
|---|---|---|
| direct / identifier | 24 | Basic lookup and exact function names |
| paraphrase | 28 | User wording that differs from the docs' wording |
| version_specific | 16 | "In PostgreSQL 17, …": only that version's page counts |
| version_diff | 14 | "Which version added …" |
| cross_page | 10 | Answers spread across several pages |
| unanswerable | 8 | Topics outside the corpus, for later "I don't know" handling |

Labels are page-level (`18:sql-merge.html`) so they survive re-chunking. Every label carries a quote from the passage that answers the question (or a list of quotes when several passages do), which `validate_questions.py` checks against the corpus; version questions also carry quotes that must be *absent* from the versions lacking the feature.

Results (top 10, 92 answerable questions). `eval_retrieval.py` runs the same search code as `/query` (hybrid and reranked; `--rerank 0` for hybrid alone, `--vector-only` for vector search alone); with dedup, a merged result counts for every version it lists:

| Configuration | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|
| Vector search | 0.71 | 0.87 | 0.93 | 0.769 |
| + version filter | 0.72 | 0.87 | 0.93 | 0.784 |
| + cross-version dedup | 0.71 | 0.93 | 0.96 | 0.797 |
| + version filter + dedup | 0.72 | 0.93 | 0.96 | 0.808 |
| Hybrid search | 0.74 | 0.88 | 0.92 | 0.788 |
| + version filter | 0.76 | 0.88 | 0.92 | 0.809 |
| + cross-version dedup | 0.74 | 0.92 | 0.95 | 0.816 |
| + version filter + dedup | 0.76 | 0.92 | 0.95 | 0.831 |
| Hybrid + reranking | 0.83 | 0.93 | 0.95 | 0.861 |
| + version filter | 0.83 | 0.93 | 0.95 | 0.861 |
| + cross-version dedup | 0.83 | 0.96 | 0.97 | 0.884 |
| + version filter + dedup (what `/query` does) | **0.83** | **0.96** | **0.97** | **0.884** |

A page-level hit overstates retrieval on long pages: `functions-json.html` has 85 chunks, and any of them counts. The chunk-level hit also requires the retrieved chunk to contain one of the page's evidence quotes, i.e. the passage that actually answers the question:

| Chunk-level | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|
| Vector search | 0.54 | 0.74 | 0.84 | 0.618 |
| + version filter | 0.55 | 0.74 | 0.85 | 0.632 |
| + cross-version dedup | 0.54 | 0.84 | 0.87 | 0.656 |
| + version filter + dedup | 0.55 | 0.84 | 0.88 | 0.665 |
| Hybrid search | 0.58 | 0.76 | 0.86 | 0.651 |
| + version filter | 0.60 | 0.77 | 0.86 | 0.668 |
| + cross-version dedup | 0.58 | 0.84 | 0.88 | 0.687 |
| + version filter + dedup | 0.60 | 0.84 | 0.88 | 0.700 |
| Hybrid + reranking | 0.72 | 0.86 | 0.89 | 0.766 |
| + version filter | 0.72 | 0.86 | 0.89 | 0.766 |
| + cross-version dedup | 0.71 | 0.88 | 0.91 | 0.785 |
| + version filter + dedup (what `/query` does) | **0.71** | **0.88** | **0.91** | **0.785** |

The quotes were first single words (`rows`, `WHERE`, `GRANT`) found in up to 37 chunks of their page, which made a chunk hit almost free; they are now the answering sentence, found in exactly one chunk per version. With `/query`'s settings, 5 questions retrieve the right page but none of its answering chunks in the top 10 (`->>`, `pg_cancel_backend`, `percentile_cont` for a median, among others; 6 with hybrid search alone, 7 with vector search alone).

The reranking rows were measured after three label fixes: q019 still had a one-word quote (`COALESCE`, which also matched the `NULLIF` passage), and the reranker put valid answers the labels didn't accept first for q029 (`json_to_recordset` expands a JSON array of objects to rows) and q032 (`ALTER SEQUENCE ... RESTART`, which the reference answer mentions). On the fixed labels, hybrid search alone with `/query`'s settings scores chunk-level hit@1 0.61, hit@5 0.84, hit@10 0.88, MRR 0.708; the other hybrid and vector rows use the earlier labels. Page-level, the version filter makes no difference once results are reranked: every chunk's breadcrumb names its version, so for "In PostgreSQL 16, …" the reranker already prefers 16's chunks.

All rows use `ef_search = 200`, and the vector rows were measured before hybrid search. Search takes about 20 ms without reranking and 0.8–1.2 s with it. With pgvector's default of 40 (the first measurements) the same four rows had MRR 0.721, 0.736, 0.749 and 0.759: part of what looked like embedding-model misses was the index skipping the right chunk (see HNSW recall).

Paraphrased questions are the weak spot. Hybrid search helped them most at rank 1 (chunk-level hit@1 0.25 → 0.36, MRR 0.41 → 0.49), and reranking again (hit@1 0.39 → 0.50, MRR 0.516 → 0.595 on the fixed labels), but the answering passage is still in the top 5 for only 68% of them: when neither search method ranks it in the top 20, the reranker never sees it. Top-1 similarity barely separates answerable questions (median 0.755) from unanswerable ones (median 0.706), so a similarity threshold alone won't detect out-of-scope questions.

### Answer quality

`generate_answers.py` answers every question the way `/query` does (version filter + dedup, top 5; the model comparison and version questions below were measured with vector search, the last runs with hybrid search and reranking) and saves each answer with the exact chunks the model saw. `judge_answers.py` then grades them:

- **Free checks:** every `[n]` citation must point to a retrieved chunk, and a regex spots refusals ("the excerpts do not cover …") as a cross-check on the judge.
- **LLM judge:** a local model gets the question, the reference answer, the numbered excerpts and the answer, and returns small labels constrained by a JSON schema (llama-server compiles it into a grammar, so the output always parses): each factual claim, SQL examples included, as supported or unsupported by the excerpts and as agreeing with, contradicting or not mentioned by the reference; coverage of the reference's key points (full/partial/none); and whether the answer says the excerpts don't answer the question. There is no free text beyond the claims: an earlier version had a free-text list of contradictions, and the judge filled it with notes like "the reference does not mention this", each of which counted as a contradiction. The verdict follows from fixed rules, in order: a claim that contradicts the reference is `incorrect` ("the excerpts don't say, but it's in 16" is a wrong answer, not a refusal); a refusal is `refused`; no coverage is `incorrect`; full coverage `correct`, partial `partial`. A refusal needs the judge's label and either a refusal in the answer's first sentence (the regex) or no coverage: the judge also marks answers that hedge and then answer ("there is no section that contrasts them directly", followed by both definitions) as declining, and gives even clean refusals "full" coverage, so neither label alone separates the two. An unanswerable question is `correct` only if the answer declines (by the judge's label or the refusal regex, since the judge missed some clean refusals) with no unsupported claims. Contradicting claims that are themselves refusal sentences ("the excerpts do not state which version…") don't count: the judge sometimes labels them that way.

Correctness and faithfulness are kept apart: an answer can match the reference and still add an unsupported SQL example, and the judge caught exactly that (an invalid `COPY ... ON_ERROR = ignore REJECT_LIMIT = 10`).

Two models answering, both judged by Qwen3.6-35B-A3B (92 answerable questions): Qwen3.6-35B-A3B (MoE, Q4, on a Linux PC) and Qwen3.5-9B (dense, Q8, on the Mac).

| | correct | partial | incorrect | refused | faithfulness | s/answer |
|---|---|---|---|---|---|---|
| **35B-A3B** overall | 0.54 | 0.23 | 0.07 | 0.16 | 0.97 | 5.5 |
| **9B** overall | 0.50 | 0.26 | 0.08 | 0.16 | 0.95 | 13.8 |

| correct, by category | 35B-A3B | 9B |
|---|---|---|
| direct | 0.77 | 0.77 |
| version_specific | 0.81 | 0.75 |
| identifier | 0.64 | 0.55 |
| paraphrase | 0.43 | 0.46 |
| cross_page | 0.40 | 0.20 |
| version_diff | 0.29 | 0.21 |

Both models declined all 8 unanswerable questions without invented facts. The 9B is close overall; it falls behind on cross_page (mostly partial answers that cover one of the pages) and on version_diff, where it makes the most mistakes: it claims JSON_TABLE, EXPLAIN SERIALIZE and NOT ENFORCED exist in PostgreSQL 16 (incorrect 0.21 vs 0.07). The 35B's own errors include virtual generated columns in 17 and EXPLAIN SERIALIZE in 16.

Of the 35B's 15 refused answerable questions, 6 had no expected page in the top 5 (a correct refusal of bad retrieval), 3 had the right page but not the chunk with the answer (e.g. `functions-json.html` without the `->>` row, which the chunk-level hit now measures), and 5 were "which version added X?" questions. The docs never say "added in 17", and 5 chunks can't show that a feature is *absent* from 16, so the model declines to infer it (fixed since, see Version questions below). Retrieval, not the model, is the main limit.

**Version questions.** With per-version retrieval and the term search (35B, the 14 version_diff questions; run 1 was a trial before the committed code, graded against the old q074/q080 references that expected "16" as the first version, and isn't saved):

| version_diff | correct | partial | incorrect | refused |
|---|---|---|---|---|
| single search (full run above) | 0.29 | 0.14 | 0.07 | 0.50 |
| per-version + term search, run 1 | 0.79 | 0.07 | 0.07 | 0.07 |
| per-version + term search, run 2 | 0.57 | 0.36 | 0.07 | 0.00 |

Both runs retrieved the same chunks and named the right version in 13 of 14 answers; they differ in sampling (temperature 0.2) and in where the judge draws the correct/partial line, mostly for answers that give the version but not the reference's one-line description of the feature. The miss is MERGE ... RETURNING: 17's MERGE page isn't in 17's top 4 chunks, so the model sees "no RETURNING" in 16 and RETURNING in 18 and answers 18.

**With hybrid search and reranking.** The same 35B run over all 100 questions with each retrieval change (version_diff through per-version retrieval in every row), 92 answerable:

| 35B-A3B | correct | partial | incorrect | refused | faithfulness |
|---|---|---|---|---|---|
| vector search | 0.59 | 0.26 | 0.07 | 0.09 | 0.97 |
| hybrid search | 0.65 | 0.24 | 0.01 | 0.10 | 0.97 |
| hybrid + reranking | 0.65 | 0.25 | 0.02 | 0.08 | 0.97 |

| correct, by category | vector | hybrid | + reranking |
|---|---|---|---|
| direct | 0.77 | 0.62 | 0.77 |
| version_specific | 0.81 | 0.81 | 0.81 |
| identifier | 0.64 | 0.73 | 0.45 |
| paraphrase | 0.43 | 0.57 | 0.54 |
| cross_page | 0.40 | 0.50 | 0.30 |
| version_diff | 0.57 | 0.71 | 1.00 |

Hybrid search cut wrong answers from 6 to 1. MERGE ... RETURNING is now right: the keyword side brings 17's MERGE chunk into view, and the model answers 17. The two paraphrase questions whose page hybrid search lost (COALESCE for "a default instead of NULL", `SELECT ... FOR UPDATE` for locking rows) had been answered wrongly; now the model says the excerpts don't cover them.

Reranking answers all 14 version questions correctly and fixes `SELECT ... FOR UPDATE` (its passage moves from rank 7 to 1), but the overall score doesn't move. Much of the per-category swing is noise: about half the changed verdicts had the same answering chunks in both runs, so they come from sampling (temperature 0.2) and the judge's correct/partial line. The identifier drop is all of that kind; one run per configuration can't resolve differences of a few questions. Two losses are real retrieval: the `string_agg` passage (a comma-separated string from many rows) and the passage explaining BRIN (BRIN vs B-tree for an append-only log) fell out of the top 5. One is a new failure mode: neither search method finds COALESCE for "a default instead of NULL", and where hybrid search returned unrelated chunks the model declined, the reranker picked JSON `DEFAULT ... ON EMPTY` passages that look relevant, and the model answered wrongly with confidence. A reranker makes the top 5 more convincing whether or not the answer is among the candidates.

**Paired comparison.** `compare_runs.py A.json B.json` pairs two runs question by question (retrieval results or judgments) and reports the difference with a 95% bootstrap interval (10,000 resamples of the questions) and an exact McNemar test on the binary metric (chunk hit@5 or correct); `--a`/`--b` take repeat runs per side and average each question. Reranking vs hybrid alone: retrieval chunk MRR 0.708 → 0.785, +0.077 [+0.014, +0.144], a real gain (21 questions better, 11 worse; hit@5 5 gained, 1 lost, p = 0.22, too few changes to tell). Answers: score (correct 1, partial 0.5) 0.772 → 0.777, +0.005 [−0.054, +0.065]; 11 questions became correct and 11 stopped being correct (p = 1.0). The answer-level difference is indistinguishable from noise with one run per side.

**Checking the judge.** `hand_grade.py` picks 21 judged answers stratified by verdict and category and writes a local web page to grade them blind (question, reference, excerpts and answer, no verdict). Blind agreement with the judge was low: 7/21 (33%), Cohen's kappa 0.08. Most of the gap was the rubric, not the judge: the hand grades marked "declined although the answer exists" as `incorrect` where the rubric says `refused`, and drew the correct/partial line differently. A second page shows each disagreement with both grades and the judge's labels, and the hand grades were settled after reading them. The disagreements and the 9B run also exposed judge bugs (notes counted as contradictions, conflicts with the excerpts graded as wrong answers, clean refusals missed), fixed with the per-claim reference labels and verdict rules above. The current judge agrees with 18/21 settled grades (86%, kappa 0.80), an upper bound since the grades were settled after seeing the judge's reasoning. One remaining judge error: it accepted "a primary key can't prevent overlaps, use EXCLUDE" for a PostgreSQL 18 question whose answer is `PRIMARY KEY (..., WITHOUT OVERLAPS)`.

## Setup

Requirements: Python 3.12+, Docker, and [llama.cpp](https://github.com/ggml-org/llama.cpp) (`brew install llama.cpp`).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the embedding model ([nomic-ai/nomic-embed-text-v1.5-GGUF](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF), file `nomic-embed-text-v1.5.Q8_0.gguf`) into the repo root and start the embedding server:

```bash
llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --embedding --port 8081
```

Start the reranker ([bge-reranker-v2-m3](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF), downloaded on first run, 636 MB). The question and a chunk are scored as one input and chunks alone reach 512 tokens, so the batch size is raised:

```bash
llama-server -hf gpustack/bge-reranker-v2-m3-GGUF:Q8_0 --reranking --port 8083 -ub 2048 -b 2048
```

Start an LLM server for answers. Any OpenAI-compatible chat server works; with llama.cpp and [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (about 22 GB at Q4):

```bash
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --port 8082 -c 16384 --reasoning off
```

To use a server on another machine, set `LLM_URL`, e.g. `export LLM_URL=http://192.168.1.50:8095/v1/chat/completions`.

Start the database (listens on host port 5433; the schema is applied on first start):

```bash
docker compose up -d --wait
```

## Pipeline

```bash
# 1. Fetch and parse the docs (raw HTML is cached in data/raw/)
python fetch_parse_docs.py --contact you@example.com

# 2. Split pages into chunks -> data/chunks/chunks.jsonl
python chunk_docs.py

# 3. Load pages and chunks into Postgres and embed them
python embed_ingest.py

# 4. Check eval labels, then measure retrieval (results saved in data/eval/results/)
python validate_questions.py
python eval_retrieval.py [--filter-version] [--dedup] [--rerank N] [--vector-only] [--show-misses]

# 5. Answer every question with an LLM, then grade the answers (saved in data/eval/answers/, data/eval/judgments/)
python generate_answers.py --name qwen3.6-35b-a3b --llm-url $LLM_URL
python judge_answers.py data/eval/answers/<file>.json --judge-url $LLM_URL
python compare_runs.py <run A>.json <run B>.json   # paired A/B of two retrieval results or two judgments

# 6. Check the judge against hand grades (writes a grading page to data/eval/hand_grades/)
python hand_grade.py page data/eval/judgments/<file>.json
python hand_grade.py score data/eval/judgments/<file>.json <downloaded grades>.json
```

## API

```bash
uvicorn app.main:app --reload    # needs the database, embedding, reranker and LLM servers running
```

Interactive docs at http://localhost:8000/docs (the Authorize button takes an API key).

Every `/query` and `/ingest` request needs an API key. Create one (it's printed once; only its hash is stored):

```bash
docker exec -i ragapi-db psql -U rag -d rag < db/schema.sql   # once, on a database created before API keys, caching, the query log and A/B variants
export RAG_API_KEY=$(python manage_keys.py create my-laptop --scopes query ingest)
python manage_keys.py list                                     # names, status, limits; never the keys
python manage_keys.py limit my-laptop query --per-minute 30 --burst 15
python manage_keys.py revoke my-laptop
```

| Endpoint | Does |
|---|---|
| `POST /query` (scope `query`) | `{"question", "version"?, "doc_type"?, "k"?, "generate"?, "compare_versions"?}` → an answer citing the chunks as `[n]`, plus the top-k chunks with URL, heading path, scores (`similarity`, `bm25`, the fused `rrf` that picks the reranker's candidates, and the `rerank` score they're ordered by) and the versions they apply to. `"generate": false` skips the LLM and returns chunks only. For "which version …?" questions (or `"compare_versions": true`), chunks come from each version and `version_presence` lists the versions whose docs contain each identifier in the question. `cached` is true when the response was stored for an earlier identical request |
| `POST /ingest` (scope `ingest`) | One page (`version`, `doc_type`, `section_title`, `page`, `url`, `text` as markdown) → chunked, upserted, new or changed chunks embedded. Re-sending an unchanged page is a no-op |
| `GET /health` | No key needed. 200 if the database and the embedding, reranker and LLM servers respond, else 503 |

```bash
curl -s localhost:8000/query -H "Authorization: Bearer $RAG_API_KEY" -H 'content-type: application/json' \
  -d '{"question": "how do I create an index without locking the table", "version": 17}'
```

Every response carries `X-Request-ID` and `Server-Timing` (per-stage milliseconds, `curl -i` shows them), each request is logged with its stage timings, and every authenticated `/query` is stored in `query_log`:

```bash
docker exec -it ragapi-db psql -U rag -d rag -c "SELECT created_at, status, cached, total_ms, timings, question FROM query_log ORDER BY id DESC LIMIT 10"
```

An A/B experiment assigns each API key a variant of the pipeline settings (`rerank_pool`, `compare_rerank_pool`); the variant is logged with each request, and `ab_report.py` compares the variants' error rates and latency:

```bash
AB_EXPERIMENT='{"name": "compare-pool", "variants": {"control": {}, "pool10": {"compare_rerank_pool": 10}}}' uvicorn app.main:app
python ab_report.py [--experiment compare-pool] [--control control] [--since 2026-10-01]
```

Tests run against the local database and the embedding, reranker and LLM servers; the ingest test adds and removes a made-up page, and the auth tests create and delete their own keys:

```bash
pytest
```

| Script | Main options |
|---|---|
| `fetch_parse_docs.py` | `--versions 16 17 18`, `--delay 1.0`, `--raw-dir`, `--out` |
| `chunk_docs.py` | `--target 450`, `--max-tokens 512`, `--in`, `--out` |
| `embed_ingest.py` | `--batch-size 32`, `--test-query`; env `DATABASE_URL`, `EMBED_URL`, `EMBED_MODEL` |
| `generate_answers.py` | `--name` (required), `--llm-url`, `--k 5`, `--ids`, `--limit` |
| `judge_answers.py` | answers file, `--judge-url`, `--ids`, `--limit` |
| `hand_grade.py` | `page` / `review` / `score`, `--seed 0` |
| `compare_runs.py` | `A.json B.json`, or `--a` / `--b` with repeat runs per side |
| `manage_keys.py` | `create NAME --scopes query ingest`, `list`, `limit NAME SCOPE --per-minute N --burst N`, `revoke NAME`; env `DATABASE_URL` |
| API (`app/variants.py`) | env `AB_EXPERIMENT` (JSON: `name`, `variants` with optional `weight` and setting overrides; unset = no experiment) |
| `ab_report.py` | `--experiment` (default: latest logged), `--control control`, `--since` |
| API (`app/main.py`) | env `LLM_MAX_PENDING` (4 answers generating or queued at the LLM) |
| API (`app/cache.py`) | env `CACHE_TTL_DAYS` (7) |
| API (`app/observability.py`) | env `LOG_LEVEL` (INFO; successful `/health` checks log at DEBUG) |
| API (`app/generate.py`) | env `LLM_URL` (default `http://localhost:8082/v1/chat/completions`), `LLM_MODEL`, `LLM_TIMEOUT` (120 s) |
| API (`app/rerank.py`) | env `RERANK_URL` (default `http://localhost:8083/v1/rerank`), `RERANK_TIMEOUT` (30 s) |

## Repository layout

```
app/                  FastAPI app (main.py), API keys (auth.py), rate limits (ratelimit.py), response cache (cache.py), timings and query log (observability.py), A/B variants (variants.py), request/response models, shared hybrid search and reranking, version comparison, answer generation
tests/                API tests
fetch_parse_docs.py   crawl postgresql.org and convert pages to markdown
chunk_docs.py         split pages into embedding-sized chunks
embed_ingest.py       load into Postgres and embed chunks
validate_questions.py check eval labels against the corpus
eval_retrieval.py     retrieval metrics (hit@k, MRR, page- and chunk-level) on the eval set
generate_answers.py   answer every eval question with an LLM
judge_answers.py      grade saved answers with an LLM judge and citation checks
hand_grade.py         blind hand-grading page and agreement with the judge
compare_runs.py       paired A/B comparison of two eval runs (bootstrap interval, McNemar)
ab_report.py          compare A/B variants' latency and errors from query_log
manage_keys.py        create, list, limit and revoke API keys
data/eval/            eval questions, saved retrieval results, answers, judgments and hand grades
db/schema.sql         tables, indexes, the BM25 word index, API keys, rate-limit buckets, the response cache and the query log (with A/B variant)
docker-compose.yml    Postgres + pgvector
data/parsed/          parsed corpus (docs.jsonl)
```

## License

MIT
