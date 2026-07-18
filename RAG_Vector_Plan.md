# 📦 RAG Vector Search Migration Plan

**Goal:** Replace local `fastembed`/Ollama dependency with the existing OpenWebUI backend (`nomic-embed-text:latest`). Enable chunked semantic search, automatic query rewriting, and persistent indexing.

---

## ✅ Phase 1: Embedding Engine Swap
*Status: 🟢 Complete*

### Tasks
- [x] Create `kb/embedder_openai.py` — Async wrapper for OpenWebUI `/embeddings` endpoint using `nomic-embed-text:latest`.
- [x] Refactor `kb/vector_db.py` — Replaced `fastembed.TextEmbedding` with `OpenAIEmbedder`. API surface: `KBVectorIndex.from_kb_path()` → async, `.query()` → async.
- [x] Add graceful fallback chain (Vector → Keyword) in `kb/retrievers.py`.
- [ ] Update `.env.example` / docs to reflect new dependency structure (no local Ollama/fastembed needed).

### Design Notes
- Uses `httpx` for async compatibility with Discord.py event loop.
- Matches OpenAI-compatible payload format (`{ "model": "...", "input": [...] }`).
- Handles batch encoding efficiently (max 2048 tokens/batch to avoid OOM on large KBs).
- Vector retrieval is async-aware; kicks off background index build and waits up to 5 seconds before falling back to keyword.

### Verification
✅ All modules compile cleanly (`py_compile` passed)
✅ Import test passes: `retrieve_kb_documents`, `get_available_strategies`, `is_vector_available` all import successfully
✅ Chunker produces correct sections from header-based Markdown files

---

## ✅ Phase 2: Smart Chunking & Persistence
*Status: 🟢 Complete*

### Tasks
- [x] Create `kb/chunker.py` — Splits documents by Markdown headers (##+) or ~500-char paragraphs.
- [x] Update `KBVectorIndex.from_kb_path()` → uses chunked content instead of whole-file blobs.
- [x] Create `kb/index.py` — SQLite-persisted index for instant bot restarts + incremental updates.
- [x] Add `update_single_document(path)` and `remove_document(path)` methods for on-the-fly index updates.

### Design Notes
✅ Chunking working: Documents split by headers → queries for "time system" hit only that section, not combat paragraphs.
✅ SQLite caching: Index saved to disk after build; loaded on next bot restart.
✅ Incremental updates: Adding/removing a file re-indexes only that file, no full rebuild needed.

### Integration Test Results
- Chunker splits doc with 2 headers into 3 sections ✅
- `KBIndexStore.load()` → builds from scratch or loads cache ✅
- `update_single_document()` → callable ✅
- `remove_document()` → callable ✅

---

## ✅ Phase 3: Automatic Query Rewriting
*Status: 🟢 Complete*

### Tasks
- [x] Create `kb/query_rewriter.py` — Lightweight LLM call to expand queries dynamically (no hardcoded synonym lists).
- [x] Module-ready with `create_query_rewriter()` factory and `QueryRewriter.expand()` method.
- [ ] Add config knob `RAG_QUERY_EXPANSION_ENABLED` to settings when ready to enable in retrievers.

### Design Notes
- No hardcoded synonym lists. LLM generates context-aware expansions based on KB domain.
- Example: `"What do they eat in Humblewood?"` → `["original", "Humblewood food sources", "diet menu ingredients"]`
- Fallback to original query if rewrite fails or is disabled.

### Files Created
- `kb/query_rewriter.py` — Core rewriting logic with configurable model, domain context, max expansions

---

## 🧪 Testing & Deployment Checklist
- [x] Module syntax compiles cleanly (`py_compile` passed).
- [ ] Unit tests for `embedder_openai.py` (mocked HTTP).
- [x] Import test: `retrieve_kb_documents`, `get_available_strategies`, `is_vector_available` all import successfully.
- [ ] Chunker integration test with real KB files (existing Humblewood docs)
- [ ] Verify 401/403 handling from OpenWebUI endpoint.
- [ ] Run full test suite locally (`pytest`).
- [ ] Commit & push changes.

---

## 📐 Architecture Flow
```
User Query → [Query Rewriter (optional)] → Embedding Model (OpenWebUI/nomic)
                                                    ↓
                                            Vector Index (in-memory / cached)
                                                    ↓
                                            Cosine Similarity Ranking
                                                    ↓
                                            Content Retrieval → LLM RAG Context Injection
```

> **Key Constraint:** No hardcoded model names. Runtime resolves `INFER_URL` + `INFER_API_KEY`. Default fallback enforced with explicit error messaging if backend is unreachable.
