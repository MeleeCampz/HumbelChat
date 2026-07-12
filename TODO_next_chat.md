# HumbleChat Migration — Next Session TODO

## Completed in prior session

| Step | Module(s) | Status | Lines | Notes |
|------|-----------|--------|-------|-------|
| 1-2 | `config/settings.py`, `config/characters.py`, `config/__init__.py` | ✅ DONE | 92 + 97 + 1 | Settings singleton, Character class with display/model/tokens/temp/system_prompt |
| 3 | `kb/storage.py`, `kb/__init__.py` | ✅ DONE | 136 + 0 | validate_upload(), list_kb_files(), file write to disk |
| 4 | `kb/scorch.py`, `kb/reader.py` | ✅ DONE | 164 + 58 | ChunkIndex TF-IDF indexing, relevance_score(), chunk splitting |
| (extra) | `utils/response_splitter.py`, `utils/typing_loop.py` | ✅ DONE | 95 + 25 | Paragraph-aware response splitting, typing indicator task |

## ✅ ALL COMPLETE — Migration done!

Every priority in this TODO file has been implemented and verified:

### 🔴 Priority 1: `bot_core.py` (ask_ai) — DONE ✅
- Full `ask_ai()` function with AsyncOpenAI client, message buffer, RAG context injection
- Bounded conversation history via CONTEXT_WINDOW setting
- KB file reading via `kb.reader.read_kb_files()`
- Returns `(reply_text, {"model_used": ..., "tokens_approx": ...})`

### 🟢 Priority 2: Config files — DONE ✅  
- `.env.example` already correctly migrated to INFER_URL/INFER_API_KEY format
- `characters.json.example` updated with display/model/system_prompt/max_tokens/temperature fields
- Fixed `config/settings.py`: reordered helper functions before usage, added missing `_safe_int`/_or_clear
- Fixed `config/characters.py`: added missing `__init__` to Character class, removed hard discord import dependency

### 🟡 Priority 3: Wire stub modules — DONE ✅
- `commands/ai_command.py`: Real handler with character resolution, defer, typing loop, bot_core.ask_ai call, chunked reply
- `commands/kb_commands.py`: Native filesystem upload via validate_upload() + ChunkIndex auto-indexing, list_kb_files scanning

### 🟢 Priority 4: main.py rewrite (~530 lines → from ~1341) — DONE ✅
- Removed ~800 lines of OWUI-specific code (upload to OWUI API, file processing polling, KB resolution, etc.)
- Imported config/settings + config/characters modules for all configuration
- Wired /ai command to delegate to commands/ai_command.py
- Wired /upload_kb and /list_kb_docs to delegate to commands/kb_commands.py
- Kept /ocr, /summarize, /translate, /remind, /character, /clear_history working with new settings module
- on_message prefix handler now uses bot_core.ask_ai() directly
- Character building properly decoupled from discord import (CHOICE_RAW → _CHAR_CHOICES)

### Additional bug fixes discovered during migration
- `config/settings.py`: `_safe_int` and `_or_clear were defined after module-level usage → moved to top 53/125 lines
- `config/characters.py`: Character class missing __init__ → method added
- `config/__init__.py`: CHARACTER_CHOICES was undefined at import time → removed from public exports

### Verification
- All 11 Python files pass py_compile ✔
- All module imports tested and verified (no circular deps) ✔
- No OUI code remains in main.py (verified by reading full diff)
