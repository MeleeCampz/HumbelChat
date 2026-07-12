# HumbleBot RAG — Research & Current State (2026-07-11)

**Last Updated:** 2026-07-11
**Status:** Implemented — filesystem-based RAG with graceful fallbacks to existing metadata system

---

## ✅ What Works Today

### Knowledge Base Discovery
- `GET /api/v1/knowledge/` → returns all KBs with file counts ✓
- `GET /api/v1/knowledge/{kb_id}/files` → returns 16 files in HumbleWood KB with full metadata (IDs, filenames, sizes, hashes) ✓

### Commands
- `/upload_kb` — 3-step upload flow (upload → process → add to KB) ✓
- `/list_kb_docs` — lists all documents across KBs with directory structure ✓
- `/character` — model switching ✓
- `/ocr`, `/summarize`, `/translate`, `/remind` — all working ✓

---

## 🟢 Implementation Status (2026-07-11)

### Filesystem RAG Reader (NEW)

When the shared Docker volume (`/home/meleecampz/.open-webui/data/knowledge:/shared-knowledge:ro`) is mounted in this container, the bot will now:

1. **Read actual KB content** directly from OWUI's file store
2. **Score chunks against each user query** using entity matching + keyword overlap scoring
3. **Inject relevant passages as system context** before sending to the LLM

The implementation is in `main.py` with three key functions:

| Function | Purpose |
|----------|---------|
| `_parse_owui_chunks_txt()` | Reads `.chunks.txt` files (OWUI's chunk format) |
| `_find_kb_chunks_on_disk()` | Scans `/shared-knowledge/{kb_id}/` recursively for all chunk files |
| `_rank_chunks_by_relevance()` | Scores chunks with: entity match (+15), keyword overlap (+1–8), size bonus (+1) |

#### File structure expected by the bot:
```
/shared-knowledge/{kb_id}/{file_uuid}/{name}.chunks.txt   ← OWUI standard layout
/shared-knowledge/{kb_id}/*.chunks.txt                    ← flat layout (fallback)
```

### Graceful Fallback
If the shared volume is not mounted (current state), the bot falls back to:
- Original metadata-only RAG (file names and sizes as system prompt context)
- The `retrieval=True` + `knowledge_bases:[id]` flags in API requests (no change)

---

## ⏳ Remaining Tasks

| Task | Owner | Status |
|------|-------|--------|
| Verify `/shared-knowledge` mount reflects OWUI KB data | User (docker/host) | Pending |
| Restart bot after mount is active | User | Pending |
| Test `/ai `Who is Alderheart?" and verify lore content injection | User | Pending |

---
