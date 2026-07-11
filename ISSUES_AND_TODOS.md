# HumbleBot - Current State & Issues Tracker

**Last Updated:** 2026-07-11

---

## ✅ Resolved / Working

- Bot restarts successfully; routing endpoints remain active
- Knowledge base exists in OpenWebUI: **HumbleWood** (ID: `7924210c-1fc0-4657-ab92-58002b17b463`)
- Model metadata correctly shows HumbleWood KB attached to both `humblechatsystem` and `trixysmoldersome` models
- Model resolution works: `KB_KB_ID` resolves lazily when `/ai` command is used
- RAG flags (`retrieval=True`, `knowledge_bases=[...]`) are correctly set in `extra_body` passed to the OpenAI-compatible API

---

## ❌ Unresolved Issues

### Issue #1: Knowledge Base Has NO Documents (ROOT CAUSE)
**Status:** 🔴 BLOCKING — Everything stems from this

The HumbleWood knowledge base exists in OpenWebUI but has **zero documents** attached (`"files": null`). This means there is nothing for RAG to retrieve.

- The KB was created and accessible via the HumbleBot web UI
- But no actual lore files (Alderheart, NPCs, locations, etc.) were uploaded or indexed into it
- The model `humblechatsystem` reports "no knowledge base entries" because there are literally no chunks in the vector store

**Evidence:**
```json
{
  "id": "7924210c-1fc0-4657-ab92-58002b17b463",
  "name": "HumbleWood",
  "files": null,     // <-- No files!
  ...
}
```

**Required Fix:** Upload DnD campaign lore documents (NPCs, locations, plot notes, house rules) into the HumbleWood knowledge base via the OpenWebUI web interface at `http://192.168.178.96:3000`. Documents must be indexed for RAG to work.

---

### Issue #2: Discord Bot RAG Doesn't Retrieve Context
**Status:** 🟡 Related to Issue #1 — will resolve once docs are added

- The bot's `ask_ai_with_model()` correctly sends `retrieval=True` and the KB ID in `extra_body`
- When tested directly via curl, OpenWebUI's RAG endpoint returns generic "I don't have access to your knowledge base" because no documents exist
- Once documents are uploaded and indexed, this should start working automatically

---

### Issue #3: Knowledge Base ID Resolution Fails at Startup
**Status:** 🟡 MINOR — Works on lazy resolution but could be improved

- `_resolve_kb_id()` may fail during startup (first time the bot runs)
- There's a fallback to lazy resolution when `/ai` is first called, which works
- Consider making KB resolution more robust or pre-loading it

---

## 📋 TODO / Action Items

### Priority 1: Knowledge Base Content (BLOCKING)
- [ ] **Upload lore documents to HumbleWood KB** via OpenWebUI web UI (`http://192.168.178.96:3000`)
  - NPC entries (Alderheart, any other DnD characters)
  - Location descriptions (Humblewood area, landmarks, dungeons)
  - Plot/campaign notes and backstory
  - House rules and mechanics references
- [ ] Verify documents are indexed (check KB page shows document count > 0)
- [ ] Test RAG works by asking about an uploaded entry

### Priority 2: Bot Configuration Improvements
- [ ] Improve KB resolution to handle edge cases (network failures, stale cache)
- [ ] Add health check endpoint to verify RAG pipeline is functional
- [ ] Consider caching KB documents list for faster lookups

### Priority 3: Testing & Verification
- [ ] End-to-end test: ask bot about uploaded lore → verify context appears in response
- [ ] Test with both `humblechatsystem` and `trixysmoldersome` models
- [ ] Verify character role mapping still works post-fixes

### Priority 4: Documentation
- [ ] Document how to add new lore to the knowledge base
- [ ] Update README with troubleshooting steps for RAG issues
- [ ] Add setup instructions for first-time HumbleWood KB population

---

## 🔧 Technical Details

### Key Configuration Values
| Setting | Value | File/Location |
|---------|-------|---------------|
| API Base URL | `http://192.168.178.96:3000` | `.env` |
| Model ID | `humblechatsystem` or `trixysmoldersome` | `.env` / `characters.json` |
| KB Name | `HumbleWood` | OpenWebUI admin panel |
| KB ID | `7924210c-1fc0-4657-ab92-58002b17b463` | OpenWebUI / API response |
| RAG Flag | `extra_body["retrieval"] = True` | `main.py:188` |

### Relevant Code Paths
- `/ai` command handler → `resolve_knowledge_base_id()` → `_resolve_kb_id()`
- Chat completion → `ask_ai_with_model()` → sets `extra_body` with RAG flags (line 186-190)
- Model routing → `characters.json` maps role names to model slugs

### OpenWebUI API Endpoints Used
- `GET /api/v1/models` — lists available models with KB attachments
- `POST /api/v1/chat/completions` — chat completion with RAG via `extra_body`
- `GET /api/v1/knowledge/{id}` — knowledge base metadata
- `GET /api/v1/config/all` — system configuration

### Known Limitations of Current Approach
- OpenWebUI's `/knowledge/{id}/query` endpoint is not exposed (returns 405)
- No direct access to the underlying vector store (ChromaDB/Qdrant) from outside
- RAG relies entirely on OpenWebUI proxying chunks into the system prompt
- If documents aren't uploaded to KB, there's no fallback retrieval mechanism
