# 📦 Vector Search Implementation Notes

**Status: ✅ Complete — All phases merged into `master`**

This document tracks the vector search migration from keyword-only to hybrid RAG retrieval.

---

## Phase 1: Embedding Engine Swap ✅

Replaced local `fastembed`/Ollama dependency with an OpenAI-compatible inference backend (`nomic-embed-text:latest`).

### Files Created/Modified
- `kb/embedder.py` — Async wrapper for an OpenAI-compatible `/embeddings` endpoint using `httpx`
- `kb/vector_db.py` — Refactored to use `OpenAIEmbedder`; `from_kb_path()` is async; `.query()` returns `(display_name, score)` tuples

### Design Decisions
- Uses `httpx` for async compatibility with Discord.py event loop (no blocking calls)
- Matches OpenAI-compatible payload format: `{ "model": "...", "input": [...] }`
- Handles batch encoding efficiently (8 docs/batch to avoid OOM on large KBs)
- Vector retrieval is async-aware; builds index in background

---

## Phase 2: Smart Chunking & Persistence ✅

Smart document chunking with persistent SQLite-backed index.

### Files Created/Modified
- `kb/chunker.py` — Head-aware splitting with minimum-size merging and adaptive paragraph fallback
- `kb/index.py` — SQLite-persisted `KBIndexStore` for instant bot restarts + incremental updates
- `kb/vector_db.py` — Uses chunked content instead of whole-file blobs

### Chunking Strategies
1. **Full document** for small files (≤8000 chars) — preserves semantic context
2. **Header-based splitting** with min-size merging for larger docs — prevents tiny broken fragments
3. **Adaptive paragraph splitting** as fallback, with structural awareness for dense content

### Index Features
- SQLite caching: index saved to `kb/.index_cache/vector_index.db` and loaded on restart
- Incremental updates: adding/removing a file re-indexes only that file
- Auto-invalidation when KB files are newer than cache

---

## Phase 3: Query Rewriting ✅

Automatic LLM-powered query expansion for enhanced retrieval coverage.

### Files Created/Modified
- `kb/query_rewriter.py` — `QueryRewriter` class with dynamic expansion via the configured AI backend
- `create_query_rewriter()` factory function with configurable model, domain context, and max expansions

### How It Works
- No hardcoded synonym lists — LLM generates context-aware expansions based on KB domain
- Example: `"What do they eat in Humblewood?"` → `["Humblewood food sources", "diet menu ingredients"]`
- Fallback to original query if rewrite fails or is disabled

---

## Phase 4: Unified Retrieval Layer ✅

Unified `retrieve_kb_documents()` entry point with strategy switching and adaptive fallback.

### Files Created/Modified
- `kb/retrievers.py` — Complete rewrite with dual-strategy support (vector + keyword), pure cosine ranking, flashrank-inspired marginal utility reordering, and adaptive-k threshold selection
- `bot_core.py` — Updated to use `retrieve_kb_documents()` instead of direct KB reader calls

### Key Features
- **Strategy switching**: `RAG_RETRIEVAL_METHOD=vector|keyword` (default: `vector`)
- **Automatic fallback**: If vector index is unavailable or empty, falls back to keyword/TF-IDF
- **Pure vector ranking** during stabilization phase (no hybrid scoring)
- **Marginal utility reordering** for optimal document selection
- **RAG_MAX_CHARS** hard cap prevents context bloat

---

## 📐 Current Architecture Flow

```
User Query
    │
    ▼
┌──────────────┐
│  Chunker     │  Smart splitting (header/paragraph/adaptive)
│  (optional)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐      ┌──────────────────┐
│ Query        │─────►│ nomic-embed-text │
│ Rewriter     │      │ embedding model  │
│ (optional)   │      └────────┬─────────┘
└──────┬───────┘              │
       │                      ▼
       ▼           ┌──────────────────┐
    Vector         │  SQLite Index    │◄── kb/.index_cache/
    Search         │  (cached)        │     vector_index.db
       │           └──────────────────┘
       ▼
┌──────────────┐
│ Marginal     │  FlashRank-style greedy selection
│ Utility      │
│ Reordering   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ RAG_MAX_CHARS │  Hard cap on context sent to LLM
│ Cap          │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ KB Context   │  Injected into AI prompt
│ Injection    │
└──────────────┘

Fallback: If vector unavailable → Keyword/TF-IDF (kb.scorch)
```

---

## 📋 Environment Variables for RAG

| Variable | Default | Description |
|---|---|---|
| `RAG_RETRIEVAL_METHOD` | `vector` | `vector` or `keyword` |
| `RAG_MAX_DOCS` | `4` | Max docs per query |
| `RAG_MAX_CHARS` | `24000` | Hard cap on RAG context chars |
| `RAG_WINDOW_LINES` | `80` | Lines above/below match anchors |
| `KB_DEFAULT_KB` | `humblewood` | Default KB name |
| `CHUNK_SIZE` | `2000` | Target chunk size |

## ✅ Implementation Checklist

- [x] Vector index compiles cleanly
- [x] Import test passes (all modules importable)
- [x] Chunker splits docs correctly (header + adaptive tests)
- [x] SQLite persistence works (save/load/shutdown)
- [x] Incremental updates work (add/remove single docs)
- [x] Query rewriter factory works
- [x] Unified retriever with fallback chain
- [x] All code merged to `master`

> **Note:** No hardcoded model names anywhere in new code. Runtime resolves `INFER_URL` + `INFER_API_KEY`. Default fallback enforced with explicit error messaging if backend is unreachable.
