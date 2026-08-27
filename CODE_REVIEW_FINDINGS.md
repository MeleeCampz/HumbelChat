# Discord AI Bot — Code Review Findings
*Review date: 2026-08-20 · Bot: HumbleChat · Repo: /home/user/discord-ai-bot*
*Status updated: 2026-08-27 — items marked ✅ have been fixed and covered by tests (142 passing).*

**Status legend:** ✅ fixed · ⏳ partially addressed · ❌ open

---

## 1. BUGS (observed in logs or traceable in code)

### 1.1 CRITICAL — RAG budget can deliver **zero** documents — ✅ fixed (`22a1720`)
**Where:** `bot_core/ai_client.py`, RAG loop (~lines 78-95)
**Symptom (bot.log 2026-08-02 15:15):**
```
RAG: skipped doc 'playing-the-game.md (relevant chunks)' (39333 chars) — remaining budget 23965 chars
RAG: included 0/20 documents (~0K chars) — budget cap reached
ask_ai → ... rag_chars=35 ...
```
The `break` on the first doc that exceeds the remaining budget means that if the *first* (highest-ranked) doc is larger than the entire budget, **no docs at all** are attached. The model then answers with no KB context and either says "(empty response)" or falls back to generic D&D knowledge.

**Fix direction:** Use `continue` instead of `break`, or split/trim oversized chunks to fit the budget.

---

### 1.2 Character `temperature` is never applied — ✅ fixed (`b6f9587`)
**Where:** `characters.json` defines `"temperature": 0.7` per character; `bot_core/ai_client.py` hardcodes `temperature=0.7`.
The per-character value is silently ignored. Changing it in `characters.json` has no effect.

---

### 1.3 `AsyncOpenAI` client created per request — ✅ fixed (`2987598`)
**Where:** `bot_core/ai_client.py` → `_make_client()` called inside `ask_ai()` and also by every utility command.
Each call opens a new HTTP session. Over a busy session this wastes TCP handshakes and prevents connection keep-alive.

**Fix direction:** Create one module-level `AsyncOpenAI` instance and reuse it.

---

### 1.4 In-memory history lost on every restart — ✅ fixed (`9bfa1c2`)
**Where:** `bot_core/history.py` — `_chat_history` is a plain dict.
Every bot restart (or Discord gateway reconnect that triggers a re-import) wipes all per-channel conversation context. The logs show multiple restarts per day.

**Fix direction:** Persist to SQLite or a JSON file; load on startup.

---

### 1.5 Single-instance lock checks a port that is never bound — ✅ fixed (`39b76bc`, tests in `cd54cee`)
**Where:** `main.py` → `_enforce_single_instance()` connects to `127.0.0.1:18765`.
Nothing in the codebase listens on port 18765, so the check always passes (connection refused → `except: pass`). The real guard is only the PID file, which is fine, but the port check is dead/misleading code.

---

### 1.6 Typing indicator expires on long AI calls — ✅ fixed (`f68ac76`)
**Where:** `main.py` `on_message` calls `message.channel.typing()` **once**.
AI calls routinely take 60-180 s (see logs). Discord's typing indicator expires after ~10 s. The bot appears dead to users.
`utils/typing_loop.py` exists and is used by the `/ai` slash command, but **not** by the `!ai` prefix path in `on_message`.

**Fix direction:** Use `typing_loop_task` in `on_message` as well.

---

### 1.7 Reminders lost on restart — ✅ fixed (`03d4734`)
**Where:** `commands/utility_commands.py` → `_send_reminder` uses `asyncio.create_task(asyncio.sleep(delay))`.
If the bot restarts before the delay elapses, the reminder is silently lost. There is no persistence.

---

### 1.8 OCR / Summarize / Translate use `DEFAULT_MODEL`, not the character's model — ✅ fixed (`39b76bc`)
**Where:** `commands/utility_commands.py` — all three call `_make_client()` then pass `DEFAULT_MODEL` (or `FALLBACK_MODELS`).
If the user has set a specific model per character (e.g. `"trixysmoldersome"`), these utility commands ignore it and use the global default, which may not even exist on the backend (see §2.4).

---

### 1.9 OCR: no download size / timeout guard — ✅ fixed (`39b76bc`)
**Where:** `commands/utility_commands.py` `handle_ocr_command`
`client.get(image.url)` has no `timeout` parameter and no max-size check. A large attachment or slow CDN can hang the command indefinitely.

---

### 1.10 Test MagicMock pollution in production log — ✅ fixed (`39b76bc`, tests in `cd54cee`)
**Where:** `logs/bot.log` (Aug 10-11)
```
ERROR root: Failed to send AI response: object MagicMock can't be used in 'await' expression
INFO  bot: Logged in as <MagicMock name='mock.user' ...>
```
The test suite imports `main.py` (which calls `logging.getLogger()` and attaches handlers to the same `bot.log` / `dev.log` files) and then runs with mocked discord objects. Test output interleaves with production logs, making them hard to read and potentially corrupting the log file.

**Fix direction:** Have tests use a separate log file or suppress file handlers; or restructure so `main.py` doesn't configure logging at import time.

---

### 1.11 `_send_reminder` lazy-imports `main` → circular import fragility — ✅ fixed (`03d4734` — reminders refactored into `bot_core.reminders`, no lazy `main` import remains)
**Where:** `commands/utility_commands.py:58`
```python
from main import bot as _bot
```
This triggered `NameError: name 'settings' is not defined` in test runs (Aug 11). The pattern is fragile: `main.py` imports command modules at the top level, which import `main` lazily. Any reordering of imports or a new transitive import can break it.

**Fix direction:** Pass the bot instance via a setter/dependency-injection pattern, or use `discord.Client.get_bot()` if available.

---

## 2. RUNTIME / OPERATIONAL ISSUES (from bot.log)

### 2.1 Vector index rebuilt from scratch on every (or many) startups — ✅ fixed (`0d65167` — content-hash cache key)

Seen repeatedly:
```
kb.index: Building vector index from scratch for 'data/knowledge'
... 150-720 embedding API calls ...
kb.index: Index built and saved to cache (1501 docs)
```
The cache key / invalidation logic appears to be triggering full rebuilds. Each rebuild costs 150-720 embedding API calls and 30-90 s of latency.

### 2.2 Embedding endpoint instability
Observed errors: `400 Bad Request`, `405 Method Not Allowed`, `500 Internal Server Error`, `501 Not Implemented`.
The keyword-search fallback works but degrades answer quality significantly (see §2.1).

### 2.3 Frequent AI request timeouts (120 s+) — ✅ fixed (`_scaled_timeout` in `bot_core/ai_client.py`, applied to both streaming and non-streaming paths)

With RAG context of 240 K+ chars (~60 K tokens), the backend model often times out:
```
ERROR bot.commands.ai_command: AI request failed: Request timed out.
```
The timeout was a flat 120 s (`REQUEST_TIMEOUT`) even for 60 K+ token prompts.

**Fix:** `_scaled_timeout()` adds 0.5 s per 1000 chars of prompt (≈250 tokens), capped at `REQUEST_TIMEOUT * 4`. Applied to both `ask_ai` and `ask_ai_stream`.

### 2.4 Model-not-found errors (repeated) — ✅ mitigated (`3ff2167` — stale-model guard on both chat and utility paths)

```
Model 'qwen3.6:latest' not found ...
Model 'unsloth/gemma-4-12b-it-GGUF' not found ...
Model 'base_model_id:ministral-3:14b' not found ...
```
The model name in `characters.json` or `.env` doesn't match what the backend has loaded. The error message is clear but the root cause (mismatched model name) is a configuration issue that recurs.

### 2.5 "Characters loaded: (none)" on several recent starts — ✅ fixed (`03d4734` — absolute-path resolution)

```
2026-08-12 22:50:52 [INFO] bot: Characters loaded: (none)
2026-08-12 22:54:00 [INFO] bot: Characters loaded: (none)
```
Suggests `characters.json` was temporarily empty or the relative path `characters.json` failed when the working directory changed. The `load_characters(pathlib.Path("characters.json"))` call uses a **relative** path, which breaks if the bot is started from a different CWD.

### 2.6 Duplicate log lines — ✅ fixed (handlers on `bot` logger only, `propagate=False`, discord logger isolated)
Every log message appeared twice (e.g. `2026-07-19 05:23:53,940 [INFO] bot: ...` followed by the same line with ANSI colors). This is because the console handler and the `discord` library's own logging both propagate to the root logger.

**Fix:** `main.py` attaches all handlers directly to the `bot` logger (never root) with `propagate=False`, and gives the `discord` logger its own console-only handler with propagation disabled.

---

## 3. DESIGN / IMPROVEMENT OPPORTUNITIES

### 3.1 No concurrency or rate limiting — ✅ fixed (`0d65167` — per-channel rate limit)

Multiple users can fire `/ai` simultaneously. Each one triggers a full RAG retrieval + embedding + LLM call. There is no semaphore, queue, or per-channel rate limit. A burst of 5 users could saturate the backend.

### 3.2 No streaming — ✅ fixed (`0d65167`)

All responses are non-streaming (`stream=False`). For 60-180 s generation times, users see nothing until the entire response is ready. Streaming would dramatically improve perceived latency.

### 3.3 RAG context is prepended to the system prompt — ✅ fixed (`0d65167` — RAG in user message)

The 24-285 K chars of KB context is stuffed into the system message. This is token-expensive and may dilute the actual system prompt. A dedicated "context" message role (if the backend supports it) or a separate user/assistant pair would be cleaner.

### 3.4 `characters.json` loaded at import time with relative path — ✅ fixed (`03d4734`)

`main.py` line 72: `load_characters(pathlib.Path("characters.json"))` — CWD-dependent. Should be `pathlib.Path(__file__).parent / "characters.json"`.

### 3.5 No input validation on user prompts — ✅ fixed (`0d65167` — input length cap)

`on_message` in `main.py` passes the raw prompt to `ask_ai` with no length check. A user could paste 100 K chars of text, which combined with RAG context would exceed any reasonable context window.

### 3.6 `_or_clear` helper is misleading — ✅ fixed (renamed to `_history_reset_flag`, returns `bool`)
`config/settings.py`: the function name `_or_clear` and its docstring didn't clearly convey that it returns the string `"clear"` if the env var equals "clear", else `None`. The resulting variable `CHAT_HISTORY_RESET` was of type `str | None` but its only meaningful value was `"clear"`.

**Fix:** `_history_reset_flag()` now returns a proper `bool`; accepts `clear`/`1`/`true`/`yes` (case-insensitive). Docs and `.env.example` updated.

### 3.7 No structured error taxonomy — ✅ fixed (`bot_core/errors.py`, wired into both chat paths)
All AI errors were caught as generic `Exception` and string-formatted into Discord messages.

**Fix:** `bot_core/errors.py` defines `AIError` subclasses (`TimeoutError`, `ModelNotFoundError`, `BackendDownError`) plus `classify_ai_error()`. Both `ask_ai` and `ask_ai_stream` classify failures and surface a user-friendly message; `handle_ai_command` has a dedicated `ValueError` branch. Covered by `tests/test_error_taxonomy.py`.

### 3.8 `response_splitter` uses a fixed 1900-char chunk — ✅ fixed (`DISCORD_SAFE_CHUNK`, header-aware budget, line-level fallback)
Discord's actual limit is 2000 chars, but the old splitter could emit chunks whose *header + body* exceeded the safe budget, and it had no handling for single lines longer than a whole chunk (long URLs).

**Fix:** `utils/response_splitter.py` now uses an explicit `DISCORD_SAFE_CHUNK = 1990` budget that accounts for the `[N/M]` metadata prefix, subtracts the header length from the per-chunk body budget, and falls back to line-level (then hard) splitting for oversized paragraphs. Covered by extended `tests/test_response_splitter.py`.

### 3.9 No health-check endpoint or liveness probe — ✅ fixed (`bot_core/health.py`, `AI_HEALTH_CHECK_INTERVAL`)
If the AI backend goes down, every user request used to fail with a timeout after 120 s.

**Fix:** `bot_core/health.py` performs a lightweight `GET /models` probe at startup (any HTTP response = reachable; only timeouts/connection errors = down) and logs OK/DOWN. Setting `AI_HEALTH_CHECK_INTERVAL=N` starts a periodic background monitor that logs state changes. Covered by `tests/test_health_probe.py`.

### 3.10 `requirements.txt` is minimal (139 bytes) — ✅ fixed (`39b76bc` — all deps pinned)

Only 3-4 packages listed. The actual dependency tree (openai, httpx, discord.py, python-dotenv, numpy, sqlite3) is not fully pinned, which risks environment drift.

---

## 4. MINOR / COSMETIC

| # | Location | Note |
|---|----------|------|
| 4.1 ✅ | `main.py` | `_enforce_single_instance` re-imports `socket` and `os` inside the function despite having them at module level. Fixed: uses module-level `os`. |
| 4.2 ✅ | `main.py` | `log_top_kb_files` is imported inside `on_ready` but the function is trivially importable at top level. Fixed: top-level import (also `typing_loop_task`, `send_long_response`, `RateLimitError`). |
| 4.3 ✅ | `ai_client.py` | `approx_tokens = len(reply_text.split())` is a very rough estimate (word count ≠ tokens). Fixed: char-based estimate (`len // 4`). |
| 4.4 ✅ | `ai_command.py` / `main.py` | Typing tasks were created with bare `asyncio.create_task`. Fixed: `utils/background_tasks.spawn_tracked_task()` keeps a strong reference in a set until completion and logs unhandled exceptions; covered by `tests/test_background_tasks.py`. |
| 4.5 | `characters.json` | `"temperature"` field present but never read (see §1.2). |
| 4.6 | `docs/` | Documentation (6 files) is good but doesn't mention the RAG budget bug (§1.1) or the streaming limitation. |

---

## 5. SUMMARY / PRIORITY MATRIX

*Updated 2026-08-22 — see status marks in each section. Test suite: 124 passing.*

| Priority | Item | Impact | Status |
|----------|------|--------|--------|
| **P0** | 1.1 — RAG zero-doc budget bug | Bot gives empty/wrong answers | ✅ `22a1720` |
| **P0** | 2.4 — Model not found | Bot completely non-functional | ✅ `3ff2167` (guard) |
| **P1** | 1.4 — History lost on restart | UX regression | ✅ `9bfa1c2` |
| **P1** | 1.6 — Typing indicator (prefix path) | Bot appears dead | ✅ `f68ac76` |
| **P1** | 1.3 — No client reuse | Latency / resource waste | ✅ `2987598` |
| **P1** | 1.2 — Temperature ignored | Config silently ignored | ✅ `b6f9587` |
| **P2** | 1.7 — Reminders lost | Feature unreliable | ✅ `03d4734` |
| **P2** | 2.1 — Index rebuild every start | Wasted API calls, slow startup | ✅ `0d65167` |
| **P2** | 3.1 — No rate limiting | Backend saturation | ✅ `0d65167` |
| **P2** | 3.2 — No streaming | Poor perceived latency | ✅ `0d65167` |
| **P2** | 3.3 — RAG in system prompt | Token cost | ✅ `0d65167` |
| **P2** | 3.5 — No input cap | Prompt injection / OOM | ✅ `0d65167` |
| **P3** | 1.5 — Dead port check | Misleading code | ✅ `39b76bc` + tests `cd54cee` |
| **P3** | 1.8 — Utility model bypass | Wrong model | ✅ `39b76bc` |
| **P3** | 1.9 — OCR no guard | DoS vector | ✅ `39b76bc` |
| **P3** | 1.10 — Test log pollution | Ops confusion | ✅ `39b76bc` + tests `cd54cee` |
| **P3** | 1.11 — Circular import | Fragile | ✅ `03d4734` |
| **P3** | 2.5 / 3.4 — Relative paths | Breaks on CWD change | ✅ `03d4734` |
| **P3** | 3.10 — Unpinned deps | Env drift | ✅ `39b76bc` |
| **P3** | 2.6 — Duplicate log lines | Ops confusion | ✅ handlers on `bot` logger only |
| **P3** | 3.6 — `_or_clear` naming | Readability | ✅ `_history_reset_flag() -> bool` |
| **P3** | 3.7 — Error taxonomy | UX / retry logic | ✅ `bot_core/errors.py` |
| **P3** | 3.8 — Splitter edge cases | Discord limit | ✅ header-aware safe chunk budget |
| **P3** | 3.9 — Health check | Fail-fast | ✅ `bot_core/health.py` |
| **P3** | Various minor (§4) | Code quality | ✅ imports, token estimate, task retention |

**Still open:** §2.2 (embedding endpoint instability — backend-side only; keyword fallback works).

*Note:* while verifying the fixes above, a latent bug was found and fixed: `main.py` imported `_CHARACTERS` by value *before* `load_characters()` ran, so slash-command character choices were silently empty after every restart. Character choices are now built via `config.characters.get_character_choices()` after loading.
