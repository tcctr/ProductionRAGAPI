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
                                      Postgres + pgvector (HNSW + full-text)
                                                   │
question ──embed──▶ nearest chunks ────────────────┘──▶ prompt with chunk text ──▶ LLM ──▶ cited answer
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
| Answer evaluation (LLM judge) | In progress |
| Hybrid search, reranking | Planned |
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

**One database.** pgvector keeps vectors, metadata and full-text search in Postgres. Chunks carry `version` and `doc_type` directly, so filtered searches need no join. An HNSW index serves vector search and a GIN index on a generated `tsvector` column serves keyword search, which hybrid search will combine. A `content_hash` per chunk lets re-ingestion skip unchanged text.

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

Labels are page-level (`18:sql-merge.html`) so they survive re-chunking. Every label carries a quote that `validate_questions.py` checks against the corpus; version questions also carry quotes that must be *absent* from the versions lacking the feature.

Results (vector search, top 10, 92 answerable questions). `eval_retrieval.py` runs the same search code as `/query`; with dedup, a merged result counts for every version it lists:

| Configuration | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|
| Vector search | 0.71 | 0.87 | 0.93 | 0.769 |
| + version filter | 0.72 | 0.87 | 0.93 | 0.784 |
| + cross-version dedup | 0.71 | 0.93 | 0.96 | 0.797 |
| + version filter + dedup (what `/query` does) | 0.72 | 0.93 | 0.96 | 0.808 |

All rows use `ef_search = 200`. With pgvector's default of 40 (the first measurements) the same four rows had MRR 0.721, 0.736, 0.749 and 0.759: part of what looked like embedding-model misses was the index skipping the right chunk (see HNSW recall).

Paraphrased questions are the weak spot (hit@5 0.82 with dedup, but hit@1 only 0.43): the right page is usually retrieved but not ranked first, which reranking targets next. Top-1 similarity barely separates answerable questions (median 0.757) from unanswerable ones (median 0.711), so a similarity threshold alone won't detect out-of-scope questions.

### Answer quality

`generate_answers.py` answers every question the way `/query` does (version filter + dedup, top 5) and saves each answer with the exact chunks the model saw. `judge_answers.py` then grades them:

- **Free checks:** every `[n]` citation must point to a retrieved chunk, and a regex spots refusals ("the excerpts do not cover …") as a cross-check on the judge.
- **LLM judge:** a local model gets the question, the reference answer, the numbered excerpts and the answer, and returns small labels constrained by a JSON schema (llama-server compiles it into a grammar, so the output always parses): each factual claim, SQL examples included, as supported or unsupported by the excerpts; statements contradicting the reference; coverage of the reference's key points (full/partial/none); and whether it refused. The verdict follows from fixed rules: a refusal is `refused` (correct only for unanswerable questions, and only with no unsupported claims), a contradiction or no coverage is `incorrect`, full coverage `correct`, partial coverage `partial`.

Correctness and faithfulness are kept apart: an answer can match the reference and still add an unsupported SQL example, and the judge caught exactly that (an invalid `COPY ... ON_ERROR = ignore REJECT_LIMIT = 10`).

First run, Qwen3.6-35B-A3B answering and judging its own answers (92 answerable questions):

| | correct | partial | incorrect | refused | faithfulness |
|---|---|---|---|---|---|
| overall | 0.55 | 0.20 | 0.07 | 0.18 | 0.97 |
| direct | 0.77 | 0.23 | 0.00 | 0.00 | 1.00 |
| version_specific | 0.81 | 0.06 | 0.12 | 0.00 | 0.97 |
| paraphrase | 0.46 | 0.21 | 0.07 | 0.25 | 0.97 |
| version_diff | 0.21 | 0.07 | 0.07 | 0.64 | 0.92 |

All 8 unanswerable questions were refused without invented facts. Of the 17 refused answerable questions, 6 had no expected page in the top 5 (a correct refusal of bad retrieval), 3 had the right page but not the chunk with the answer (e.g. `functions-json.html` without the `->>` row: page-level hit@k overstates retrieval on long pages), and 7 were "which version added X?" questions. The docs never say "added in 17", and 5 chunks can't show that a feature is *absent* from 16, so the model declines to infer it. The judge also makes mistakes (it graded one answer that agreed with the reference as `incorrect`), so these numbers are provisional until it is checked against hand grades. Next: that check, then the same run with a 9B dense model.

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
python eval_retrieval.py [--filter-version] [--dedup] [--show-misses]

# 5. Answer every question with an LLM, then grade the answers (saved in data/eval/answers/, data/eval/judgments/)
python generate_answers.py --name qwen3.6-35b-a3b --llm-url $LLM_URL
python judge_answers.py data/eval/answers/<file>.json --judge-url $LLM_URL
```

## API

```bash
uvicorn app.main:app --reload    # needs the database, embedding server and LLM server running
```

Interactive docs at http://localhost:8000/docs.

| Endpoint | Does |
|---|---|
| `POST /query` | `{"question", "version"?, "doc_type"?, "k"?, "generate"?}` → an answer citing the chunks as `[n]`, plus the top-k chunks with URL, heading path, similarity and the versions they apply to. `"generate": false` skips the LLM and returns chunks only |
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
| `generate_answers.py` | `--name` (required), `--llm-url`, `--k 5`, `--limit` |
| `judge_answers.py` | answers file, `--judge-url`, `--ids`, `--limit` |
| API (`app/generate.py`) | env `LLM_URL` (default `http://localhost:8082/v1/chat/completions`), `LLM_MODEL`, `LLM_TIMEOUT` (120 s) |

## Repository layout

```
app/                  FastAPI app (main.py), request/response models, shared search, answer generation
tests/                API tests
fetch_parse_docs.py   crawl postgresql.org and convert pages to markdown
chunk_docs.py         split pages into embedding-sized chunks
embed_ingest.py       load into Postgres and embed chunks
validate_questions.py check eval labels against the corpus
eval_retrieval.py     retrieval metrics (hit@k, MRR) on the eval set
generate_answers.py   answer every eval question with an LLM
judge_answers.py      grade saved answers with an LLM judge and citation checks
data/eval/            eval questions, saved retrieval results, answers and judgments
db/schema.sql         tables and indexes
docker-compose.yml    Postgres + pgvector
data/parsed/          parsed corpus (docs.jsonl)
```

## License

MIT
