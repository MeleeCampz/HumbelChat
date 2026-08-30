# RAG / Knowledge Base

The bot can attach relevant local knowledge-base content to AI prompts. This is optional.

## How it works

1. Documents are stored in `KB_PATH`.
2. Files are chunked using the smart chunker.
3. A retrieval strategy selects the most relevant chunks for a query.
4. Selected context is included in the AI request, up to `RAG_MAX_CHARS`.

## Retrieval methods

Controlled by `RAG_RETRIEVAL_METHOD`:

- `vector` (default): semantic search using embeddings from the configured inference backend. A SQLite-backed index in `<KB_PATH>/.vector_index_cache/` caches embeddings so restarts do not require re-embedding.
- `keyword`: TF-IDF-style heuristic scoring based on filenames, headers, and body text overlap. Works without a vector backend.

If vector search is unavailable, keyword search can be used as a fallback.

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
| `EMBEDDING_MODEL` | Embedding model used for vector search (set to match your inference backend's model) |
