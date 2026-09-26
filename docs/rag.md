# RAG / Knowledge Base

The bot can attach relevant local knowledge-base content to AI prompts. This is optional.

## How it works

1. Documents are stored in `KB_PATH`.
2. Files are chunked using the smart chunker.
3. A retrieval strategy (hybrid dense + BM25 by default) selects the most relevant chunks for a query.
4. Selected context is included in the AI request, up to `RAG_MAX_CHARS`.

## Embedding strategy (CPU queries, GPU index builds)

The pipeline deliberately **splits the two sides of embedding** so that per-request
RAG never has to swap models on the chat backend:

- **Query embeddings — in-process on CPU.** Each user query is embedded locally by
  `sentence-transformers` (`EMBED_BACKEND=local`, model `LOCAL_EMBED_MODEL`, default
  `BAAI/bge-m3`). This runs inside the bot process and never touches the inference
  backend, so there is no per-request model swap with the chat model. It requires the
  `rag-cpu` extra (sentence-transformers).
- **Index builds — on the GPU backend.** Building or rebuilding the vector index is a
  one-time, batch-heavy job. By default it runs on the inference backend's `/embeddings`
  endpoint (`INDEX_EMBED_BACKEND=backend`, model `INDEX_EMBED_MODEL`), which uses the
  GPU and finishes in well under a minute for a few thousand chunks (versus ~20 min on
  pure CPU). The backend must have the **same** embedding model loaded. If the backend is
  unreachable, index builds fall back to local CPU so a reindex never hard-fails.

### Why mixing CPU and GPU embeddings is safe

Both sides use the **same model** (bge-m3), so their vectors are interchangeable —
cosine similarity between a CPU query vector and a GPU-built index vector is valid. We
verified this empirically: rebuilding the full 1314-chunk index on the GPU backend and
re-running identical queries produced rankings that were **identical to 5-decimal
precision** (mean `|score|` delta ≈ 0, top-1 and top-3 matches for every query; only
near-tied chunks deep in the top-10 swapped). The practical consequence: build the index
once on the GPU for speed, serve queries in-process on CPU with zero backend round-trips,
and either side can be rebuilt later without invalidating the other.

## Retrieval methods

Controlled by `RAG_RETRIEVAL_METHOD`:

- `vector` (default): semantic search over the embedding index. A SQLite-backed index in `<KB_PATH>/.vector_index_cache/` caches embeddings so restarts do not require re-embedding.
- `keyword`: TF-IDF-style heuristic scoring based on filenames, headers, and body text overlap. Works without a vector backend.

If vector search is unavailable, keyword search can be used as a fallback.

### Hybrid dense + BM25 retrieval

By default (`RAG_HYBRID_ENABLED=1`) the vector path fuses **two** signals with reciprocal
rank fusion (RRF):

- **Dense** — cosine similarity between the query embedding and each chunk (captures
  meaning / paraphrase).
- **Lexical (BM25)** — an exact-term index over every chunk (captures names, spell titles,
  stat blocks, proper nouns that dense vectors can under-weight).

RRF merges the two orderings without needing score calibration, so a query still finds the
right chunk when the player's phrasing differs from the KB's vocabulary. Set
`RAG_HYBRID_ENABLED=0` to use dense-only retrieval.

### Min-attachment relevance floor (opt-in)

`RAG_MIN_ATTACH_SCORE` (default `0` = off) drops **dense** chunks whose cosine similarity
is below the threshold *before* hybrid fusion, so weak vector hits cannot crowd out strong
BM25 matches. Lexical-only matches still survive via the hybrid step. Tune it from the
`Vector scores for ...: top=... median=... min=...` log line.

### Per-file chunk budget + attach-time relevance floor

Two knobs control how much of each matched file actually reaches the prompt:

- **`RAG_MAX_CHUNKS_PER_FILE`** (default `3`) caps how many of a file's ranked
  chunks are attached. They're joined into one per-file entry, so lowering this
  trims the "whole file" bloat — e.g. a spell query no longer drags in four
  neighbouring sections when one is enough.
- **`RAG_ATTACH_FLOOR`** (default `0.50`, `0` = off) is a per-chunk relevance gate
  applied at *attach* time (after hybrid fusion / rerank). A ranked chunk whose
  original **dense cosine similarity** is below the floor is skipped, so only
  genuinely relevant sections are attached. It gates on the raw dense score — not
  the rank-based RRF score — and **lexical-only chunks** (exact-term BM25 hits with
  no dense score) are always kept. A safety net guarantees an over-aggressive floor
  can never produce an empty context: if it would drop everything, selection falls
  back to the unfiltered ranking.

## Low-confidence query rewriting (vector path)

Vector search alone can miss the right chunk when a player's phrasing differs
from the KB's vocabulary. To cover that case without slowing down every
query, the rewriter only runs when the index is *not* confident:

1. The query is embedded and ranked as usual (one embedding call).
2. If the top similarity score is **below `RAG_REWRITE_MIN_SCORE`**
   (default `0.35`), the LLM generates up to `RAG_QUERY_MAX_EXPANSIONS`
   alternative phrasings (wall-clock budget: `RAG_REWRITE_BUDGET_SECONDS`).
3. All expansions are embedded in **one batched call**, and every ranking is
   merged with reciprocal rank fusion (RRF) — order-based, so no score
   calibration is needed.
4. If the rewrite or the expansion embedding fails for any reason, retrieval
   silently falls back to the original ranking.

Confident queries (top score ≥ threshold) pay nothing extra. Each query logs
its score distribution (`Vector scores for ...: top=... median=... min=...`)
so you can tune `RAG_REWRITE_MIN_SCORE` from real traffic — raise it if the
rewriter triggers too often, lower it if answers still miss on unusual
phrasing. Set `RAG_QUERY_REWRITER=0` to disable the feature entirely.

The rewrite always uses **the same model as the main completion call** for
that request (a per-character model override is forwarded into the rewriter),
so a single-model local backend never has to load a second set of weights.

## Request serialization (single local backend)
The bot assumes one inference backend serving one model. AI requests are
therefore serialized **process-wide**: at most one request (RAG retrieval +
completion) is in flight at any time, served FIFO. Requests from different
channels queue up rather than running in parallel — a second `/ai` while the
first is still generating simply waits its turn.

## Smart chunking

The chunker uses a few strategies depending on file size and structure:

- **Full document** for small files, to preserve context
- **Header-based splitting** for larger documents, with minimum-size merging
- **Adaptive paragraph splitting** as a fallback for dense content

This helps queries hit relevant sections without being drowned out by unrelated content.

## Supported file types

The knowledge base accepts and meaningfully reads/indexes:

- `.txt`
- `.md`
- `.csv`
- `.html`
- `.xml`
- `.rtf`

These are the expected file types for KB use. Storage itself does not strictly enforce only these extensions — uploaded files use MIME-based inference and fall back to `.txt` when the extension or MIME type is unknown — but files whose extensions are not in the set above are generally not read or indexed by the same path.

## Commands

- `/upload_kb` — add files to the knowledge base
- `/list_kb_docs` — list indexed documents
- `/reindex_kb` — rebuild the vector index from scratch

## Useful settings

| Variable | Purpose |
|---|---|
| `KB_PATH` | Where KB files are stored |
| `CHUNK_SIZE` | Display-only "approx N chunks" estimate in `/list_kb_docs`; the chunker uses fixed character-based limits |
| `RAG_MAX_DOCS` | Max documents attached per query |
| `RAG_MAX_CHARS` | Max RAG context chars sent to the LLM |
| `RAG_WINDOW_LINES` | Window around each match anchor |
| `RAG_RETRIEVAL_METHOD` | `vector` or `keyword` |
| `RAG_QUERY_REWRITER` | Enable low-confidence query rewriting (`1`/`0`) |
| `RAG_REWRITE_MIN_SCORE` | Similarity threshold below which the rewriter triggers |
| `RAG_QUERY_MAX_EXPANSIONS` | Max alternative phrasings generated per rewrite |
| `RAG_REWRITE_BUDGET_SECONDS` | Wall-clock cap for the LLM rewrite call |
| `EMBED_BACKEND` | `local` (in-process CPU query embeddings, recommended) or `remote` (legacy backend `/embeddings`) |
| `LOCAL_EMBED_MODEL` | In-process query-embedding model (default `BAAI/bge-m3`) |
| `INDEX_EMBED_BACKEND` | Where index builds run: `backend` (GPU, default) or `local` (CPU) |
| `INDEX_EMBED_MODEL` | Model slug sent to the backend for index builds (must match `LOCAL_EMBED_MODEL`) |
| `RAG_HYBRID_ENABLED` | Fuse dense + BM25 via RRF (`1`/`0`, default on) |
| `RAG_MIN_ATTACH_SCORE` | Opt-in dense-similarity floor before fusion (`0` = off) |
| `RAG_MAX_CHUNKS_PER_FILE` | Max chunks attached per file (default `3`) |
| `RAG_ATTACH_FLOOR` | Per-chunk dense-score gate at attach time (`0` = off, default `0.50`) |
