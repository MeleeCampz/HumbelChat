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
