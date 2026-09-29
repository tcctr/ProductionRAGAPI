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
question ──embed + keywords──▶ hybrid search ──────┘──▶ prompt with chunk text ──▶ LLM ──▶ cited answer
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
| Reranking | Planned |
| Auth, rate limiting, caching | Planned |
| Observability, A/B testing | Planned |

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

Results (top 10, 92 answerable questions). `eval_retrieval.py` runs the same search code as `/query` (hybrid, or `--vector-only`); with dedup, a merged result counts for every version it lists:

| Configuration | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|
| Vector search | 0.71 | 0.87 | 0.93 | 0.769 |
| + version filter | 0.72 | 0.87 | 0.93 | 0.784 |
| + cross-version dedup | 0.71 | 0.93 | 0.96 | 0.797 |
| + version filter + dedup | 0.72 | 0.93 | 0.96 | 0.808 |
| Hybrid search | 0.74 | 0.88 | 0.92 | 0.788 |
| + version filter | 0.76 | 0.88 | 0.92 | 0.809 |
| + cross-version dedup | 0.74 | 0.92 | 0.95 | 0.816 |
| + version filter + dedup (what `/query` does) | **0.76** | 0.92 | 0.95 | **0.831** |

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
| + version filter + dedup (what `/query` does) | **0.60** | 0.84 | 0.88 | **0.700** |

The quotes were first single words (`rows`, `WHERE`, `GRANT`) found in up to 37 chunks of their page, which made a chunk hit almost free; they are now the answering sentence, found in exactly one chunk per version. With `/query`'s settings, 6 questions retrieve the right page but none of its answering chunks in the top 10 (`->>`, `pg_cancel_backend`, `percentile_cont` for a median, among others; 7 with vector search alone).

All rows use `ef_search = 200`, and the vector rows were measured before hybrid search. With pgvector's default of 40 (the first measurements) the same four rows had MRR 0.721, 0.736, 0.749 and 0.759: part of what looked like embedding-model misses was the index skipping the right chunk (see HNSW recall).

Paraphrased questions are the weak spot. Hybrid search helped them most at rank 1 (chunk-level hit@1 0.25 → 0.36, MRR 0.41 → 0.49), but the answering passage is still in the top 5 for only 64% of them (68% with vector search alone), which reranking targets next. Top-1 similarity barely separates answerable questions (median 0.755) from unanswerable ones (median 0.706), so a similarity threshold alone won't detect out-of-scope questions.

### Answer quality

`generate_answers.py` answers every question the way `/query` does (version filter + dedup, top 5; the results below were measured with vector search, before hybrid search) and saves each answer with the exact chunks the model saw. `judge_answers.py` then grades them:

- **Free checks:** every `[n]` citation must point to a retrieved chunk, and a regex spots refusals ("the excerpts do not cover …") as a cross-check on the judge.
- **LLM judge:** a local model gets the question, the reference answer, the numbered excerpts and the answer, and returns small labels constrained by a JSON schema (llama-server compiles it into a grammar, so the output always parses): each factual claim, SQL examples included, as supported or unsupported by the excerpts and as agreeing with, contradicting or not mentioned by the reference; coverage of the reference's key points (full/partial/none); and whether the answer says the excerpts don't answer the question. There is no free text beyond the claims: an earlier version had a free-text list of contradictions, and the judge filled it with notes like "the reference does not mention this", each of which counted as a contradiction. The verdict follows from fixed rules, in order: a claim that contradicts the reference is `incorrect` ("the excerpts don't say, but it's in 16" is a wrong answer, not a refusal); a refusal is `refused`; no coverage is `incorrect`; full coverage `correct`, partial `partial`. An unanswerable question is `correct` only if the answer declines (by the judge's label or the refusal regex, since the judge missed some clean refusals) with no unsupported claims. Contradicting claims that are themselves refusal sentences ("the excerpts do not state which version…") don't count: the judge sometimes labels them that way.

Correctness and faithfulness are kept apart: an answer can match the reference and still add an unsupported SQL example, and the judge caught exactly that (an invalid `COPY ... ON_ERROR = ignore REJECT_LIMIT = 10`).

Two models answering, both judged by Qwen3.6-35B-A3B (92 answerable questions): Qwen3.6-35B-A3B (MoE, Q4, on a Linux PC) and Qwen3.5-9B (dense, Q8, on the Mac).

| | correct | partial | incorrect | refused | faithfulness | s/answer |
|---|---|---|---|---|---|---|
| **35B-A3B** overall | 0.54 | 0.22 | 0.07 | 0.17 | 0.97 | 5.5 |
| **9B** overall | 0.50 | 0.24 | 0.08 | 0.18 | 0.95 | 13.8 |

| correct, by category | 35B-A3B | 9B |
|---|---|---|
| direct | 0.77 | 0.77 |
| version_specific | 0.81 | 0.75 |
| identifier | 0.64 | 0.55 |
| paraphrase | 0.43 | 0.46 |
| cross_page | 0.40 | 0.20 |
| version_diff | 0.29 | 0.21 |

Both models declined all 8 unanswerable questions without invented facts. The 9B is close overall; it falls behind on cross_page (mostly partial answers that cover one of the pages) and on version_diff, where it makes the most mistakes: it claims JSON_TABLE, EXPLAIN SERIALIZE and NOT ENFORCED exist in PostgreSQL 16 (incorrect 0.21 vs 0.07). The 35B's own errors include virtual generated columns in 17 and EXPLAIN SERIALIZE in 16.

Of the 35B's 16 refused answerable questions, 6 had no expected page in the top 5 (a correct refusal of bad retrieval), 3 had the right page but not the chunk with the answer (e.g. `functions-json.html` without the `->>` row, which the chunk-level hit now measures), and 6 were "which version added X?" questions. The docs never say "added in 17", and 5 chunks can't show that a feature is *absent* from 16, so the model declines to infer it (fixed since, see Version questions below). Retrieval, not the model, is the main limit.

**Version questions.** With per-version retrieval and the term search (35B, the 14 version_diff questions; run 1 was a trial before the committed code, graded against the old q074/q080 references that expected "16" as the first version, and isn't saved):

| version_diff | correct | partial | incorrect | refused |
|---|---|---|---|---|
| single search (full run above) | 0.29 | 0.07 | 0.07 | 0.57 |
| per-version + term search, run 1 | 0.79 | 0.07 | 0.07 | 0.07 |
| per-version + term search, run 2 | 0.57 | 0.36 | 0.07 | 0.00 |

Both runs retrieved the same chunks and named the right version in 13 of 14 answers; they differ in sampling (temperature 0.2) and in where the judge draws the correct/partial line, mostly for answers that give the version but not the reference's one-line description of the feature. The miss is MERGE ... RETURNING: 17's MERGE page isn't in 17's top 4 chunks, so the model sees "no RETURNING" in 16 and RETURNING in 18 and answers 18.

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
python eval_retrieval.py [--filter-version] [--dedup] [--vector-only] [--show-misses]

# 5. Answer every question with an LLM, then grade the answers (saved in data/eval/answers/, data/eval/judgments/)
python generate_answers.py --name qwen3.6-35b-a3b --llm-url $LLM_URL
python judge_answers.py data/eval/answers/<file>.json --judge-url $LLM_URL

# 6. Check the judge against hand grades (writes a grading page to data/eval/hand_grades/)
python hand_grade.py page data/eval/judgments/<file>.json
python hand_grade.py score data/eval/judgments/<file>.json <downloaded grades>.json
```

## API

```bash
uvicorn app.main:app --reload    # needs the database, embedding server and LLM server running
```

Interactive docs at http://localhost:8000/docs.

| Endpoint | Does |
|---|---|
| `POST /query` | `{"question", "version"?, "doc_type"?, "k"?, "generate"?, "compare_versions"?}` → an answer citing the chunks as `[n]`, plus the top-k chunks with URL, heading path, scores (`similarity`, `bm25`, and the fused `rrf` they're ordered by) and the versions they apply to. `"generate": false` skips the LLM and returns chunks only. For "which version …?" questions (or `"compare_versions": true`), chunks come from each version and `version_presence` lists the versions whose docs contain each identifier in the question |
| `POST /ingest` | One page (`version`, `doc_type`, `section_title`, `page`, `url`, `text` as markdown) → chunked, upserted, new or changed chunks embedded. Re-sending an unchanged page is a no-op |
| `GET /health` | 200 if the database, the embedding server and the LLM server respond, else 503 |

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "how do I create an index without locking the table", "version": 17}'
```

Tests run against the local database, embedding server and LLM server; the ingest test adds and removes a made-up page:

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
| API (`app/generate.py`) | env `LLM_URL` (default `http://localhost:8082/v1/chat/completions`), `LLM_MODEL`, `LLM_TIMEOUT` (120 s) |

## Repository layout

```
app/                  FastAPI app (main.py), request/response models, shared hybrid search, version comparison, answer generation
tests/                API tests
fetch_parse_docs.py   crawl postgresql.org and convert pages to markdown
chunk_docs.py         split pages into embedding-sized chunks
embed_ingest.py       load into Postgres and embed chunks
validate_questions.py check eval labels against the corpus
eval_retrieval.py     retrieval metrics (hit@k, MRR, page- and chunk-level) on the eval set
generate_answers.py   answer every eval question with an LLM
judge_answers.py      grade saved answers with an LLM judge and citation checks
hand_grade.py         blind hand-grading page and agreement with the judge
data/eval/            eval questions, saved retrieval results, answers, judgments and hand grades
db/schema.sql         tables, indexes and the BM25 word index
docker-compose.yml    Postgres + pgvector
data/parsed/          parsed corpus (docs.jsonl)
```

## License

MIT
