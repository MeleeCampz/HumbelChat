# Development Roadmap & TODO

This file tracks the progress of the HumbleChat bot migration and upcoming feature implementations.

## ❌ INCOMPLETE — Active Development

### Priority 1: Complete `bot_core.py`
**Status:** Stub implemented (17 lines). Needs ~200 lines.
- [ ] Implement `ask_ai()` function with HTTP client integration (`AsyncOpenAI`).
- [ ] Implement KB state management (`_kb_id`, `_kb_name`) and `set_kb_state()`.
- [ ] Ensure full provider-agnostic client functionality using `INFER_URL` and `INFER_API_KEY`.

### Priority 2: Wire `commands/ai_command.py`
**Status:** Stub implemented (raises `NotImplementedError`). Needs ~150 lines.
- [ ] Resolve active character from the new configuration system.
- [ ] Implement interaction deferral and the typing indicator loop.
- [ ] Resolve KB name/state.
- [ ] Call `bot_core.ask_ai()` and handle chunked responses in Discord.
- [ ] Support character overrides.

### Priority 3: Wire `commands/kb_commands.py`
**Status:** Stub implemented (raises `NotImplementedError`). Needs ~100 lines.
- [ ] Implement `handle_upload_kb`: Validate $\rightarrow$ Write to `KB_PATH` $\rightarrow$ Auto-index chunks.
- [ ] Implement `list_kb_docs`: Scan local directory for files instead of calling OWUI API.
- [ ] Add MIME type detection and file size validation.

### Priority 4: Main Integration & Cleanup (`main.py`)
**Status:** Needs ~300 line rewrite/refactor.
- [ ] Update imports to use new `config.*` and `kb.*` modules.
- [ ] Remove all legacy OpenWebUI (OWUI) specific code and HTTP helpers.
- [ ] Wire slash command handlers to the new core logic.
- [ ] Implement `/character` dropdown using the new `Character` class.
- [ ] Replace KB resolution logic with local filesystem lazy-resolve from `KB_PATH`.

---

## 🚀 Upcoming Features (Next Session Goals)

### Knowledge Base Enhancements
- [ ] Add `delete_kb_doc` command to remove files from the local filesystem and update the index.
- [ ] Implement a "Re-index" command to manually trigger `scron.py` processing on an existing directory.
- [ ] Add support for `.pdf` or `.docx` parsing via specialized libraries (e.g., `pypdf`).

### UI/UX & User Experience
- [ ] Implement Discord **Embeds** for character information displays to improve readability.
- [ ] Improve error handling for failed AI API requests (retry logic or user-friendly error messages).
- [ ] Add a `/status` command to display bot health, number of indexed KB files, and the currently connected model.

### Infrastructure & Deployment
- [ ] Finalize `docker-compose.yml` for easy one-command deployment (including an OpenAI-compatible backend like Ollama).
- [ ] Implement a `.env.example` validation/walkthrough script.
- [ ] Update `.env.example` and `characters.json.example` to the new unified format.
