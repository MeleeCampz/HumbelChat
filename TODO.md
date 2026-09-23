# TODO — Improvement Log (priority-sorted)

> Generated from a full code + docs audit (2026-09-21). 480 tests passing after P0 #1–#4 + P1 #5/#14 (was 462 at audit).
> Priority: **P0** = correctness bug (wrong behavior / data loss) · **P1** = robustness / user-visible failure · **P2** = performance / UX · **P3** = DX, docs, nice-to-have.
> Conventions: each item lists **Where**, **Problem**, **Fix**, **Verify**. Ticks mark completed items.

---

## P0 — Correctness bugs

### 1. `/reindex_kb` leaves RAG serving the stale in-memory index
- [x] **Fixed (2026-09-21).** **Where:** `commands/kb_commands.py` (`handle_reindex_kb`) + `kb/retrievers.py` (singleton `_index_store`). The handler rebuilt a throwaway `KBIndexStore` and never invalidated the singleton the live RAG path used, so `/ai` kept serving the old index until restart (and the sanity check validated the stale one). **Fix:** added `replace_index_store(store, kb_path)` (+ `reset_index_store`) in `kb/retrievers.py`; the handler now rebuilds in a separate store and swaps it into the singleton **before** the sanity check, so RAG serves the fresh index immediately and the check validates what's actually live. An *empty* rebuild does not clobber a good index (degrades to old index + keyword fallback). Replaced store shut down out-of-band. **Verify:** `tests/test_p0_regressions.py::TestIndexStoreSwap` (swap/reset, reindex installs fresh store, empty-rebuild no-clobber) + a conftest fixture that resets the singleton between tests.

### 2. Conversation history only persists after exceeding the cap
- [x] **Fixed (2026-09-21).** **Where:** `bot_core/ai_client.py` (`ask_ai`). `set_history()` (the only disk write) was gated on `len(history) > 2*CONTEXT_WINDOW`, so a restart before a channel hit its cap lost every recent turn. **Fix:** `set_history(guild_id, channel_id, history[-max_entries:])` now runs on **every** turn, persisting the trimmed window to `data/history.json` and capping in-memory to the same window. **Verify:** `tests/test_p0_regressions.py::TestHistoryPersistedEachTurn` (first-turn-under-cap writes disk; RAG blob / username still kept out of persisted history).

### 3. Concurrent KB index mutations race (lost chunks)
- [x] **Fixed (2026-09-22).** **Where:** `kb/index.py` (all public mutations); lazy-init race in `kb/retrievers.py` (`update_kb_document`, `sync_kb_store`). **Problem:** Read-modify-write of `self._index._docs` spanned `await` points (embedding calls) with **no lock**. Two concurrent `/upload_kb` (or `/upload_kb` + `/sync_kb`) both read the old docs, both merge, last write wins → the first file's chunks are dropped in memory **and** on disk. **Fix:** per-store `asyncio.Lock` (`KBIndexStore._mutation_lock`, created once lazily) held across the **entire** read → embed → merge → persist sequence in `load` / `rebuild` / `update_single_document` / `remove_document` / `sync_changes` / `shutdown`. `load()` delegates to a lock-free `_load_inner()` so the held-lock paths (`sync_changes` / `update_single_document`) can bootstrap the index without deadlocking on the non-reentrant lock. The same init race in the retrievers' lazy store creation is serialized behind the existing `_index_init_lock`. Queries (`get_index`) stay lock-free. **Verify:** `tests/test_p0_3_4_regressions.py::TestConcurrentIndexMutations` — two concurrent `update_single_document` with a sleeping fake embedder → both files present in memory **and** SQLite; `/sync_kb` racing `/upload_kb` keeps both files; embed overlap peak == 1 (lock actually serializes).

### 4. Synchronous file IO blocks the event loop
- [x] **Fixed (2026-09-22).** **Where:** `kb/reader.py`, `kb/storage.py`, `kb/chunker.py`, `kb/index.py`, `kb/retrievers.py` (keyword path), `commands/kb_commands.py` (`/upload_kb`, `/list_kb_docs`, `/reindex_kb`). **Problem:** Direct `read_bytes()`/writes from async handlers. For a large KB this froze the *entire* bot (gateway, typing, voice) during the scan. **Fix:** everything blocking now runs in the default thread pool: `read_kb_files_async` / `get_relevant_chunks_async` (reader), `validate_upload_async` / `list_kb_files_async` (storage), `Chunker.split_file` → sync `split_file_sync` core, and in the index: directory walks (`_iter_files`), SQLite cache reads (`_read_cache_rows_async`), whole-file reads (`_read_file_text`), re-chunking (`_rechunk`), and the full SQLite persist (`_persist_index_to_db` — the sync core of `_save_to_disk_from`). `_retrieve_keyword` is now `async` and awaits the reader wrappers; the command handlers await the async storage wrappers (incl. the post-upload `read_text` chunk estimate and `get_root_directories`). **Also fixed here:** a latent `IndexError` in `kb/reader.py` `get_relevant_chunks` — the `any_hits` tuples are `(hit_count, line_idx)` but one comprehension unpacked them as `(li, _hc)`, using the hit count as a line index; surfaced on small files (hit count ≥ line count). **Verify:** `tests/test_p0_3_4_regressions.py::TestBlockingIoOffloaded` — a 0.5 s sleeping `read_bytes` stub no longer stalls a concurrent heartbeat (≥10 ticks in flight); `list_kb_files_async` / `validate_upload_async` execute off the loop thread (tid check). Full suite: 474 passed.

---

## P1 — Robustness / user-visible failures

### 5. `botctl.sh logs` uses an invalid tmux command
- [x] **Fixed (2026-09-22).** **Where:** `botctl.sh` (`cmd_logs`). **Problem:** `tmux follow` does not exist (verified: absent from tmux 3.5a command list) → `./botctl.sh logs` errored out. **Fix:** `cmd_logs` now checks the session exists (clear "No bot session" error + exit 1 otherwise), then branches on TTY: `[ -t 0 ] && [ -t 1 ]` → `exec tmux attach -t $SESSION` (interactive live tail, Ctrl-b d to detach); anything piped/redirected → `exec tmux capture-pane -p -S -2000 -t $SESSION | tail -n 2000` (one-shot dump, pipe-friendly). **Verify:** `tests/test_scripts.py::TestBotctlLogs` — with a stub `tmux` on PATH: non-TTY run (session alive) uses `capture-pane` and never `follow`/`attach`; non-TTY with no session exits 1 with a clear stderr; a real **pty** run takes the `attach` branch (asserted in the recorded tmux argv + pane output).

### 6. Hardcoded 30 s retry for AI 429, `Retry-After` ignored
- [x] **Fixed (2026-09-22).** **Where:** `bot_core/errors.py` (`classify_ai_error` + new `_extract_retry_after` / `_parse_retry_after`), `config/settings.py` (`AI_RETRY_AFTER_FALLBACK_S`, default 30). **Problem:** HTTP 429 → `RateLimitError(retry_after=30)`; the backend's `Retry-After` header (on `exc.response.headers` of the openai SDK's `APIStatusError`) was discarded. **Fix:** the classifier now extracts `Retry-After` from the exception's httpx response, parses it as delta-seconds **or HTTP-date** (`email.utils.parsedate_to_datetime`, tz-naive → UTC), clamps to [5, 120] s so a hostile header can neither stall the bot nor make it hammer the backend, and falls back to `AI_RETRY_AFTER_FALLBACK_S` (30) only when the header is absent/malformed. **Verify:** `tests/test_p1_6_7_ai_retry.py::TestParseRetryAfter` (seconds, clamp min/max, malformed → fallback, HTTP-date future/past, custom fallback), `TestExtractRetryAfter`, `TestClassify429UsesHeader` (real `openai.RateLimitError` + httpx response: `Retry-After: 5` → 5; malformed/absent → 30; 3600 → 120; env fallback honored) — all against genuine SDK exceptions, not fakes.

### 7. No retry on transient AI 5xx
- [x] **Fixed (2026-09-22).** **Where:** `bot_core/ai_client.py` (new `_call_completion_with_retry` + `_is_transient_api_failure`, wired into `ask_ai`'s completion call), `bot_core/errors.py` (new step 4b: HTTP 5xx → `BackendDownError`). **Problem:** single attempt; a transient backend 5xx/timeout/connection blip failed the whole turn, and a persistent 5xx even leaked out unclassified (generic `unknown`) because no classifier heuristic matched 5xx. **Fix:** the completion call is now retried **once** (2 attempts total) with exponential backoff (1.5 s base, ×2 — overridable module constant so tests don't sleep). Transient = `APIStatusError` with 429/≥500, `APITimeoutError`, `APIConnectionError`, `httpx` timeout/connection errors; 4xx (auth/validation/model-not-found) re-raises immediately, and the RAG-rewrite sub-call in `kb.retrievers` is untouched (it keeps its own `RAG_REWRITE_BUDGET_SECONDS` budget). A persistent 5xx now classifies as `BackendDownError` → the friendly “had a problem… try again” `ValueError` instead of a raw traceback. **Verify:** `tests/test_p1_6_7_ai_retry.py::TestIsTransientApiFailure` (5xx/429/timeout/connection transient; 400/401/404 not) and `TestAskAiCompletionRetry` (mocked client that 500s once then succeeds → reply + 2 calls; persistent 500 → exactly 2 calls then friendly error; 400 → exactly 1 call; timeout → retried and succeeds; patched `asyncio.sleep` proves exactly one 1.5 s backoff; 429 end-to-end surfaces the header's 7 s). `TestClassify5xx` covers the new 500/502 → backend-down mapping. Full suite: 507 passed.

### 8. `IndexError` on empty `choices`
- [x] **Fixed (2026-09-22).** **Where:** every `resp.choices[0].message.content` read — `bot_core/ai_client.py` (`ask_ai`), `commands/utility_commands.py` (`/ocr`, `/summarize`, `/translate`), `commands/session_commands.py` (end-of-session overview), and `kb/query_rewriter.py` (RAG-rewrite sub-call). The TODO named the first two, but the identical latent `IndexError` existed in all six. **Problem:** some backends complete the request but return an empty `choices` array; `resp.choices[0].message.content` then crashed with an unhelpful `IndexError` (or `AttributeError` on a `None` message) instead of a friendly error. **Fix:** one shared guard — `bot_core.errors.extract_reply_text(resp, *, default)` — replaces all six inline reads. It returns `default` when the message *content* is empty (preserving each call site's existing placeholder, e.g. `"(no text found)"`), and raises a new `AIBackendError` (category `backend_error`, a subclass of `AIError`) when *choices* is empty or the response is `None`. Each call site already sat in a `try`, so the `AIBackendError` now degrades gracefully: `ask_ai`'s `except` converts it to a user-friendly `ValueError` (added `"backend_error"` to the friendly-categories list; `classify_ai_error`'s step-1 `isinstance(AIError)` pass-through guarantees the nice `user_message` survives); `/ocr` + `/translate` surface their existing `⚠️ …failed` follow-ups; `/summarize` retries the next fallback model; the session overview falls back to `_fallback_overview`; the query rewriter returns no expansions. **Verify:** `tests/test_p1_8_empty_choices.py` — `TestExtractReplyText` (normal content returned; `None`/`""` content → default placeholder; `choices == []` → `AIBackendError` and *not* `IndexError`; `None` response → `AIBackendError`; category/`user_message` shape), `TestAskAiEmptyChoices` (end-to-end: stub `choices == []` → friendly `ValueError` containing "empty response", no `IndexError`), `TestUtilityCommandsEmptyChoices` (`/ocr` + `/translate` with `choices == []` → friendly "OCR failed"/"Translation failed", no traceback leaked). Full suite: 515 passed.

### 9. Deleted active character breaks `/ai` with `Character None not found`
- [x] **Fixed (2026-09-22).** **Where:** `commands/ai_command.py` (`handle_ai_command` step 2) + the persisted active-char key (`bot_core.history.get_active_char_key`). **Problem:** `get_active_char_key` already falls back to the default character *only when no per-channel key exists*; when a key *is* persisted but its character was later removed from `characters.json`, `get_character(char_key)` returns None and the handler rendered ``f"Character `{character_name}` not found"`` — interpolating `character_name`, which is **None** in the normal (implicit) `/ai` case — so the user saw `Character `None` not found.` and `/ai` hard-returned (no reply at all) until the persona was re-picked. **Fix:** track whether the request was *explicit* (`character_name is not None`). If the lookup fails and it was **explicit**, still name the typed character (unchanged, now guaranteed non-None). If it was **implicit** (stale persisted key), fall back to `default_character()`, log the stale key, send a one-time heads-up ("Your saved character `{key}` no longer exists, so I'm using **{default}**… use `/character set`"), and *continue* the request instead of returning. **Verify:** `tests/test_ai_command.py::TestP1_9StaleActiveCharacter` — (a) no character + persisted stale key → `ask_ai` awaited with the default model, no `Character `None``, a notice naming the stale key + default, and the AI reply still delivered (the TODO's verify case); (b) explicit missing character → still names the typed char, never `None`; (c) regression guard: a *valid* active key produces no spurious fallback warning. Full suite: 518 passed.

### 10. Malformed `RAG_REWRITE_MIN_SCORE` crashes at import
- [x] **Fixed (2026-09-22).** **Where:** `config/settings.py` (`RAG_REWRITE_MIN_SCORE`). **Problem:** it was the one numeric env var read with a bare `float(os.getenv("RAG_REWRITE_MIN_SCORE", "0.35") or 0.35)`; any unparseable value (e.g. `RAG_REWRITE_MIN_SCORE=abc`) raised `ValueError` the instant `config.settings` was imported, killing the whole bot — unlike every other numeric env var, which routes through `_safe_int` / `_safe_float`. **Fix:** one line → `_safe_float(os.getenv("RAG_REWRITE_MIN_SCORE"), 0.35)`, so garbage/empty/unset all fall back to `0.35`. (Consumers read it inside functions — `kb/retrievers.py:409` — so the import-time change is safe.) **Verify:** `tests/test_settings_numeric_env.py` — garbage `"abc"` → import succeeds and value == 0.35 (the TODO's case); valid `"0.7"` → 0.7; empty string → 0.35; unset → 0.35. Full suite: 522 passed.

### 11. No `on_error` handler for application commands
- [x] **Fixed (2026-09-22).** **Where:** `main.py` (new `@bot.tree.error async def on_app_command_error`). **Problem:** unhandled exceptions in a slash-command body surfaced as Discord's generic “Application command failed” popup — nothing logged in bot context, no user-facing message. **Fix:** a global `tree.on_error` (discord.py 2.7 wraps every non-`AppCommandError` into `CommandInvokeError` and dispatches to the Tree's `on_error` in `_dispatch_error`), which logs the full traceback via `log.error(..., exc_info=error)` and sends a short, *ephemeral*, user-scoped message. It branches on the error type (`CheckFailure` → “no permission”; `CommandSignatureMismatch` → “internal error, sync the commands”; `CommandInvokeError` → names the wrapped `.original` exception without leaking a raw traceback) and uses the primary response when the interaction hasn't been answered yet, falling back to a follow-up.
- **Verify:** `tests/test_main_lifecycle.py::TestOnAppCommandError` — unhandled body exception → friendly ephemeral message naming `ValueError: boom` (no `Traceback`), `CheckFailure` → permission text, `CommandSignatureMismatch` → generic internal-error text, primary response used when `is_done() is False`, follow-up fallback when the primary send raises. Full suite: 535 passed.

### 12. Missing `on_shutdown` cleanup
- [x] **Fixed (2026-09-22).** **Where:** `main.py` (new `@bot.event async def on_shutdown()`); `kb/retrievers.py::shutdown_vector_store()` is now actually invoked (it was previously dead code). **Problem:** no graceful-close hook — on SIGTERM/SIGINT the vector index store (the one in-memory resource worth flushing) plus in-flight typing loops and tracked background tasks were left to the OS. **Fix:** `on_shutdown` (1) cancels every live task in `bot.typing_tasks`, (2) cancels everything in `utils.background_tasks._ACTIVE_BACKGROUND_TASKS`, (3) `await kb.retrievers.shutdown_vector_store()` — each step wrapped so one failure can't prevent the rest, with a start/finish log line. **Verify:** `tests/test_main_lifecycle.py::TestOnShutdown` — `on_shutdown` awaits `shutdown_vector_store()` once (the TODO's case); cancels a live tracked bg task + typing task; does **not** cancel already-done tasks. Full suite: 535 passed.

### 13. Duplicate `_resolve_bot` in reminders (dead code)
- [x] **Fixed (2026-09-22).** **Where:** `bot_core/reminders.py`. **Problem:** `_resolve_bot` was defined twice — an early copy (imported `main.bot` directly and checked `is_ready`) was *shadowed* by a later copy (delegates to `bot_core.channel_delivery.get_bot()`), so the first was unreachable dead code that misled readers about how delivery actually resolves the bot. **Fix:** deleted the first (dead) definition; kept the `channel_delivery`-based one, which is what the single call site (`_fire`, `bot = _resolve_bot()`) always ran. **Verify:** `grep "def _resolve_bot" bot_core/reminders.py` now returns exactly one definition; the 28 reminder tests still pass (they monkeypatch the live `_resolve_bot`, whose behaviour is unchanged since the removed copy was never executed). Full suite: 522 passed.

### 14. `start_bot.sh` references nonexistent `stop_bot.sh`
- [x] **Fixed (2026-09-22).** **Where:** `start_bot.sh` (stale/live-PID guard message). **Problem:** Told the user to run `./stop_bot.sh first`, but the file doesn't exist (stop is `./botctl.sh stop`). **Fix:** Message now reads `Use ./botctl.sh stop first.` **Verify:** `tests/test_scripts.py::TestStartBotStopHint` — running the (temp-copied) script with a **live** PID in `.bot.pid` → "already running" message contains `./botctl.sh stop` and no `stop_bot.sh`; a **stale** PID is cleaned up and startup proceeds with no `stop_bot.sh` mention; both scripts grep-clean for `stop_bot.sh`.

### 15. No cap on `/remind` delay
- [x] **Fixed (2026-09-22).** **Where:** `bot_core/reminders.py` (`schedule_reminder` + new `MAX_REMINDER_DELAY_SEC` / `clamp_reminder_delay`) and `commands/utility_commands.py` (`/remind`). **Problem:** `/remind` accepted any delay — e.g. `10^9` s ≈ 31 years — and the reminder (its persisted JSON entry *and* its background asyncio task) lived until it fired, with no ceiling. **Fix:** a single 30-day cap shared by both layers — `MAX_REMINDER_DELAY_SEC` + `clamp_reminder_delay()` in the store. `schedule_reminder` clamps unconditionally (so the invariant holds no matter who calls it, including `rearm_pending_reminders`); the `/remind` command clamps *and* tells the user ("…you asked for more") instead of silently shortening it. **Verify:** `tests/test_reminders.py::TestP1_15DelayCap` — `clamp_reminder_delay` (under/at/over cap; negatives pass through); 40-day `schedule_reminder` → stored `delay_sec` == cap and `fires_at` reflects the clamp (the TODO's case); normal delay preserved; `/remind` 1000 h → scheduled at the cap + "max I can schedule" notice; `/remind` 2 h → no cap notice (regression guard). Full suite: 527 passed.

### 16. SSRF / scheme guard + true size caps for URL fetches
- [x] **Fixed (2026-09-22).** **Where:** new `utils/url_fetch.py` (shared `validate_url` + `fetch_url`), `commands/kb_commands.py` (`/upload_kb` URL path), `commands/utility_commands.py` (`/summarize` URL path). **Problem:** both did `client.get(url)` + `resp.text`/`resp.content` — the *entire* body was buffered into memory before a length check (a 2 GB response was fully held before "too large" fired), and there was no guard on scheme/host (only httpx's own non-http rejection; `file://`, `data:`, and internal/metadata hosts had no explicit guard). **Fix:** one shared hardened fetcher. `validate_url` rejects non-`http(s)` schemes and always-blocked hosts (localhost, `127.0.0.1`, `[::1]`, `0.0.0.0`, link-local `169.254.169.254` cloud metadata, reserved); broader private ranges (10/8, 172.16/12, 192.168/16) are opt-in via `block_private=True` so a self-hosted bot can still fetch LAN docs by default. `fetch_url` uses `client.stream()` + `aiter_bytes()`, aborting mid-download the moment `len(buf) + len(chunk) > max_bytes` (so an oversized body is rejected *without* buffering the whole thing), and **re-validates the final URL after redirects** so a `http://ok.example → http://169.254.169.254/` hop can't slip past the guard. `/summarize` caps at 1 MB (it only ever slices 32 KB); `/upload_kb` caps at the existing 20 MB (`UPLOAD_MAX_DOWNLOAD_BYTES`). **Verify:** `tests/test_url_fetch.py` — `validate_url` (http/https ok; `file://`, `data:`, `gopher:`, `ftp:`, `javascript:`, empty rejected; loopback/link-local/reserved hosts always rejected; private LAN allowed by default, blocked when opted in); `fetch_url` streaming (small body returned; oversized body aborts *mid-stream* after the chunk that crosses the cap — never reads the last chunk; `file://` rejected with the client *never constructed* (no network attempt); redirect to a link-local host re-validated and blocked); command wiring (`/summarize` + `/upload_kb` refuse `file://` with "URL not allowed" and refuse an oversized stream with "too large"). Also rewired the 5 pre-existing raw-`httpx` tests (`test_scan_fixes.py` `TestUploadKBUrl`/`TestSummarizeUrl`, `test_kb_commands.py::test_upload_kb_with_url`) to drive the real streaming path. Full suite: 560 passed.

---

## P2 — Performance

### 17. Vector query is pure-Python cosine over all docs
- [x] **Fixed (2026-09-22).** `kb/vector_db.py` — added a lazily-built, object-identity-cached numpy `(matrix, norms)` pair (`_ensure_matrix`) and a shared `_rank()` that does a single float32 matmul `sims = matrix @ q / (norms·||q||)` with `np.argpartition` top-k; all three methods (`query`, `query_with_embeddings`, `rank_texts`) now call it. Any numpy failure (missing module, ragged embeddings) falls back to the original pure-Python `_rank_py`. **Verify:** `tests/test_p2_vector_numpy.py` — top-k (order + scores within 1e-4) matches the reference pure-Python ranking for top_n ∈ {1,3,5,10,100}; matrix is built once and reused across queries (identity check) and invalidated on doc replacement; 5 000×768 benchmark: **0.26 ms/query vs 503 ms (≈1955× speedup)**. All 53 existing retrieval tests (query_rewrite, kb_vector_index, kb_sync) pass unchanged.

### 18. `list_kb_files` hashes every file fully
- [x] **Fixed (2026-09-22).** `kb/storage.py` — added a hidden sidecar `.sha256_cache.json` inside the KB root (dot-prefixed so the existing "skip hidden entries" rule keeps it out of listings) keyed by KB-relative path with a `(size, mtime)` guard. `list_kb_files` now loads the cache, only re-hashes when size or mtime changed, prunes stale keys on a full recursive root scan (never on a subfolder scan), and saves only when dirty. **Verify:** `tests/test_p2_storage_hash_cache.py` — a second listing of an unchanged KB makes **zero** `Path.read_bytes` calls and returns byte-identical metadata; a changed file (new size + bumped mtime) is re-read exactly once; deleted files are pruned; the cache file is hidden and never listed; a subfolder scan does not prune siblings.

### 19. Index reload re-chunks every file on every start
- [x] **Fixed (2026-09-22).** `kb/index.py` + `kb/vector_db.py` — every source file's SHA-256 is now stored in the cache (new `file_hash` column; schema v3→v4 with an `ALTER TABLE` migration in `_conn`). `_chunks_valid` takes a fast path: if the cached rows carry a `file_hash` that matches the file on disk, the chunks are valid (the chunker is a deterministic function of the file's bytes) and re-chunking is skipped entirely — one file read for the hash instead of read + regex-split + ChunkInfo build. Legacy caches (no `file_hash`) still use the original re-chunk-and-compare path. The hash is computed in `_embed_files` while already reading each file, propagated through the load/sync/update merge paths and persisted. **Verify:** `tests/test_p2_file_hash_skip.py` — a fresh build persists a per-file `file_hash` matching the file bytes; a reload of an unchanged KB makes **zero** `Chunker.split_file_sync` (re-chunk) calls; editing one file invalidates only that file (exactly one re-chunk of `two.txt`).

### 20. Full SQLite rewrite per single-file update
- [x] **Fixed (2026-09-22).** `kb/index.py` — new `_persist_upsert()` writes directly to the live DB (no tmp-file swap): it `DELETE`+`INSERT`s only the changed files' rows and deletes rows for files no longer in the index, leaving every other row's SQLite page/rowid untouched. `_save_to_disk`/`_save_to_disk_from` now take an optional `changed` set; single-file/multi-file paths (`update_single_document`, `remove_document`, `sync_changes`, `_load_incremental`) pass it (incremental), while `_build_fresh` and `shutdown` keep the atomic full rewrite — which remains the periodic compaction pass. A pre-v3 (v2, no `source_file` column) cache falls back to the full rewrite. **Verify:** `tests/test_p2_incremental_persist.py` — updating one file keeps the *other* files' rowids identical (a full rewrite would renumber all); a full `rebuild()` renumbers 1..N (compaction path); the persisted DB round-trips identically (same sources + chunk count, edited content reflected); removing a file deletes only its rows.

### 21. Embedder opens a new HTTP client per batch
- [x] **Fixed (2026-09-22).** `kb/embedder.py` + `config/settings.py` + `main.py` — added `EMBED_TIMEOUT` (default 30, was a hardcoded `timeout=30`) and a process-wide lazily-created shared `httpx.AsyncClient` (`_get_client()`, double-checked under an `asyncio.Lock`) that `_call_api` reuses for every batch/endpoint/attempt, so connections are kept alive instead of a fresh handshake per batch. `close_client()` (idempotent; resets the client but keeps the lock) is wired into `main.on_shutdown`. **Verify:** `tests/test_p2_embedder_client.py` — 5 texts at batch_size 2 → 3 batches but exactly **one** client built; separate `encode()` calls reuse the same client; the client is constructed with `EMBED_TIMEOUT`; `close_client()` awaits `aclose()` and resets the global, and is idempotent.

### 22. Legacy cache re-keying is O(rows × files)
- [x] **Fixed (2026-09-22).** `kb/index.py` — `_read_cache_rows` now lazily resolves the KB file list **once** (only when the first legacy/basename row is seen) into a `basename → [rel_key]` map and matches every row against it in a single pass; a fully-modern cache (all keys contain "/") never walks the directory at all. The SELECT was also normalized to a uniform 6-column aliased query so v2 (no `source_file`) / v3 (no `file_hash`) / v4 caches unpack identically. **Verify:** `tests/test_p2_legacy_rekey.py` — N legacy rows over M files → exactly **one** `_iter_kb_files` call (spy); re-keying is correct (unique basename → KB-relative path); a fully-modern cache makes at most one walk; an ambiguous basename (two files share it) is left for re-embed rather than guessed.

### 23. Move `import pickle` out of the hot loop
- [x] **Verified (2026-09-22).** `kb/index.py` — already satisfied: there is a single top-level `import pickle` (line 61) and **no** in-loop import (grep confirms the only `pickle.*` uses are `pickle.loads`/`pickle.dumps` in `_read_cache_rows`/`_persist_index_to_db`/`_persist_upsert`, all referencing the module-level import). No change needed.

---

## P3 — UX / features / DX / docs

### 24. Streaming AI responses
- [ ] **Where:** `bot_core/ai_client.py` (`stream=False`), `commands/ai_command.py`, `main.py::on_message`.
- **Problem:** User waits for the full generation (up to minutes) and sees nothing; the typing indicator only covers ~30 s (`utils/typing_loop.py`).
- **Fix:** Stream completion chunks → accumulate, and either (a) deliver progressively (edit a placeholder message in ≤ 2 000-char chunks) or (b) at minimum keep typing alive for the whole generation and show progress hints. Also fixes: typing task started before the channel queue in `ai_command.py` (indicator dies while still queued).
- **Verify:** Manual + test that a stub streamed reply results in an edited message; typing loop test for >30 s generations.

### 25. Stop/cancel for in-flight `/ai`
- [ ] **Where:** `commands/ai_command.py`, `bot_core/ai_client.py`.
- **Problem:** A stuck/slow request cannot be cancelled by the user.
- **Fix:** Track the current task per channel (reuse the `bot.typing_tasks` pruning pattern); `/ai stop` (or a stop word) cancels the awaited task and reports "cancelled".
- **Verify:** Test: task cancelled mid-generation → user notified, lock released, no orphaned typing task.

### 26. Global AI lock queue feedback (and/or per-guild lock)
- [ ] **Where:** `bot_core/ai_client.py` (global `asyncio.Lock`), channel queue in `main.py`.
- **Problem:** The lock serializes *all* channels; with `REQUEST_TIMEOUT * 4` ≈ 480 s worst case, other channels wait silently for minutes.
- **Fix (pick one or combine):** Per-guild locks instead of global; or queue position feedback ("you're #2, ~30 s behind"); or a warning edit if wait > 60 s.
- **Verify:** Test: two guilds → concurrent; same guild → serialized (whichever scheme is chosen).

### 27. Packaging: `pyproject.toml` with optional extras
- [ ] **Where:** `requirements.txt` (no `pyproject.toml`; `faster-whisper 1.2.1` + `fastembed 0.8.0` are labelled "optional" in comments but always installed).
- **Problem:** Heavy STT/vector deps installed by default; no proper packaging metadata.
- **Fix:** Add `pyproject.toml` (setuptools) with `base` deps and extras: `pip install -e .[stt,rag-vector]`; keep `requirements.txt` for pin compatibility or deprecate it.
- **Verify:** Clean venv installs of base vs extras; import-guard tests for optional deps still pass.

### 28. CI + linting + type checking
- [ ] **Where:** repo root (no `.github/workflows`, no ruff/mypy config, no coverage config).
- **Problem:** No automated gate on push; no static analysis.
- **Fix:** GitHub Actions (or equivalent) running `pytest` (+ `--cov`) on a pinned Python matrix; add `ruff` (lint + format) and `mypy --strict` for new modules, relaxing per-module as needed.
- **Verify:** CI green on current tree after baseline fixes.

### 29. Pin the Python version
- [ ] **Where:** repo root (no `.python-version`; code uses `str | None` → needs ≥3.10; dev env is 3.12).
- **Fix:** Add `.python-version` (3.12) + `requires-python` in the new `pyproject.toml`; mention in README.
- **Verify:** CI matrix uses the same version.

### 30. Docs debt
- [ ] **Where:** `docs/README.md`, root `README.md`, `.env.example`, `temp/Todos.md`.
- **Problem:**
  - `docs/README.md` index misses `permissions.md` and `recording-to-transcript.md`.
  - Root `README.md` "project structure" block is stale (lists 3 files; omits `utils/`, `kb/`, most `bot_core/` submodules).
  - `.env.example` doesn't document `BOT_NO_LOG_FILES` / `BOT_DISCORD_LOG_LEVEL`.
  - Carried-over item from `temp/Todos.md`: **"Remove any mentions of Beyond20"** — still present in `commands/ai_command.py`, `bot_core/response_splitter.py`, `config/settings.py` (strings/comments).
- **Fix:** Update each doc; strip Beyond20 mentions (keep behavior); add the two missing env vars to the example.
- **Verify:** `grep -ri beyond20` → zero hits; doc index matches actual files in `docs/`.

### 31. `/upload_kb` unused `kb_name` parameter
- [ ] **Where:** `commands/kb_commands.py` (`handle_upload_kb`).
- **Fix:** Remove the dead parameter (or wire it up if the API was ever meant to accept a sub-KB name).
- **Verify:** Grep for callers; tests pass.

### 32. Consistency: boolean-ish env parsing
- [ ] **Where:** `config/settings.py` (`EMBED_FORMAT` etc. use `not in ("0","false","no")`).
- **Problem:** `"False"` (capital F) parses as true; inconsistent with the `_safe_bool`-style helpers elsewhere.
- **Fix:** Use one shared helper for all bool-ish env vars.
- **Verify:** Table-driven test for all boolean settings.

### 33. Deprecated `IntentFlags.messages`
- [ ] **Where:** `main.py` (sets both legacy `messages` and `guild_messages`).
- **Problem:** `messages` (non-guild DM intent) is deprecated by Discord; keep only if DMs are actually used.
- **Fix:** Drop it unless DM support is required; document.
- **Verify:** Bot connects and receives guild messages (and DMs, if retained).

### 34. `BOT_DISCORD_LOG_LEVEL` ignored when `BOT_NO_LOG_FILES=1`
- [ ] **Where:** `main.py` logging setup (discord-logger config lives inside the file-logging branch).
- **Fix:** Configure the `discord` logger level independently of the file-logging toggle.
- **Verify:** With `BOT_NO_LOG_FILES=1` + level set, console shows the expected discord log lines.

### 35. Split `bot_core/voice_recorder.py` (1 294 lines)
- [x] **Done (2026-09-23).** **Where:** `bot_core/voice/` package (new) + `bot_core/voice_recorder.py` (now a 69-line re-exporting facade).
- **Problem:** Monolithic module (recording lifecycle, DAVE decryption, session glue).
- **Fix:** Split into `voice/capture.py` (constants, `_Speaker`, `_SpeakerLog`, timeline WAV writers), `voice/dave.py` (RTP/DAVE decrypt + Opus decode, pure helpers), `voice/session.py` (`VoiceRecorder` state machine + bot wiring), `voice/recover.py` (crash recovery). `bot_core.voice_recorder` re-exports the full public + internal surface, so every existing import path (production code and the test suite, **unchanged**) keeps working. Docs (`README.md`, `docs/voice-recording.md`, `docs/recording-to-transcript.md`) updated to the new layout.
- **Verify:** Full suite 592 passed (voice: 119 passed, `tests/` untouched); `ruff check` clean; mypy error count on the voice code unchanged by the move (12 → 12, all pre-existing type debt). Module sizes: capture 201, dave 163, recover 186, session 750, facade 69 — all well under the ~600-line target (session holds the state machine, which was the bulk of the file).

### 36. 0-byte uploads accepted
- [ ] **Where:** `kb/storage.py` (`validate_upload`).
- **Problem:** Empty files pass validation and chunk into nothing; listing then shows an empty doc (see the comment in `list_kb_docs`).
- **Fix:** Reject 0-byte uploads with a clear message.
- **Verify:** Test: empty bytes → `UploadValidationError`.

### 37. Cosmetic: markdown injection in stored history usernames
- [ ] **Where:** `bot_core/ai_client.py` (user prompt prefix `**{username}:**`).
- **Problem:** A display name containing `**` etc. breaks the prompt formatting.
- **Fix:** Sanitize/escape the username in the prefix (or use a plain-text marker).
- **Verify:** Test with a name containing `**bold**`.

---

## Done / verified non-issues (no action needed)

- Rate limiter GC in `ai_client` — asyncio single-threaded, fine.
- `.env` not tracked by git (only `.env.example`) — good.
- PID file + stale-PID handling in `main.py` — correct.
- `utils/background_tasks._handle_done_task` correctly skips `task.exception()` for cancelled tasks.
- Embedder endpoint fallback for Ollama `/v1` URLs — works.
- Chunk sizes (≤8 000 chars, max 7 500) vs nomic 8 192-token limit — safe.
- 462 tests passing at audit time.

---

### Suggested working order
1. **P0 (items 1–4)** — ✅ DONE (2026-09-21/22). All four fixed + tested (474 tests).
2. **P1 (5–16)** — #5 + #14 ✅ done (2026-09-22). Next: #6/#7 as one coherent "AI error handling" change; #16 bundles the URL hardening.
3. **P2 (17–23)** — #17 + #21 together (retrieval perf), then #18–#20 (index IO), #22–#23 are trivial cleanups.
4. **P3** — #24/#25/#26 are the big UX projects; #27–#29 are one packaging/tooling batch; #30 is docs pass.
