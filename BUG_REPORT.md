# HumbelChat — Bug & Design-Finding Report

Date: 2026-07-08 · Scope: full-repo scan (~16.5k LOC Python) + verification pass
Method: file-by-file read of every module in `bot_core/`, `commands/`, `kb/`, `config/`, `utils/`; flagged findings re-verified by targeted re-read and (where possible) reproduction.

Note: the test suite could not be executed inside this sandbox (`.venv` is Windows-built; system Python lacks `openai/discord.py`). All findings below are static-analysis + logic-reproduction results.

---

## HIGH — data loss / correctness

### H1. Session folder index collision → silent overwrite of previous session notes
**File:** `bot_core/sessions.py` L184–199 (`_next_session_index`), L216–221 (`_session_file_path`), L355–366 (`_write_session_file`)

`_next_session_index()` counts existing folders with regex `^{prefix}_(\d+)_` — it **requires a trailing underscore after the digits**. But *unnamed* sessions produce folders like `2026-07-08_01` (no trailing underscore, see `_session_file_path`: the name suffix is only added when a name exists). Those folders are invisible to the index counter.

Result: a second `/start_session` on the same day **without a name** computes index `01` again → `_write_session_file()` does `mkdir(exist_ok=True)` + `os.replace(tmp, notes.md)` and **overwrites the first session's notes.md**.

Reproduced (logic simulation of L184–199/L216–221):
```
folders on disk: ["2026-07-08_01", "2026-07-08_02"]   # two unnamed sessions
computed index: 1 -> stem: 2026-07-08_01 | collision: True
```
This is a very reachable path: two unnamed `/start_session` runs in one day.

**Fix:** regex `^{prefix}_(\d+)(?:_|$)`; additionally, before creating, loop until the target dir does not exist.

### H2. `load_persisted()` aborts mid-load on `"None"` keys → history + active-character map lost
**File:** `bot_core/history.py` L73–92

The module's own comment (L18–19) states DM turns are persisted under the string key `"None"`. On load, `int(g_str)` / `int(c_str)` raises `ValueError: invalid literal for int() with base 10: 'None'` for those entries. The exception is caught only by the **outer** `except`, so:

- loading of the history dict stops at the failing key (channels loaded before it stay, everything after is skipped), and
- the entire `active_characters` section (loaded afterwards) is **never applied**.

Any bot that ever handled a DM turn silently loses its per-channel character selection on every restart, plus part of the history.

**Fix:** parse keys via a helper (`_parse_id("None") -> None`), or wrap each entry in its own try/except. Also: `_save_to_disk()` writes `"None"` keys — round-trip must be symmetric.

---

## MEDIUM

### M1. Crash recovery loses every speaker's identity and WAV files collide
**File:** `bot_core/voice/recover.py` L216–220 (`_user_id_from_log_name`) vs `bot_core/voice/session.py` L506 (`_log_path_for`)

Runtime frame logs are named `<safe_display>_<user_id>.wav.log`. `_user_id_from_log_name` strips only `.log`, leaving `..._<uid>.wav`; the regex `_(\d+)$` then **never matches** → every speaker gets `user_id = 0`. With `display_name = ""` in recovery, all speakers map to `_wav_filename(0, "") == "user-0.wav"` — with multiple speakers, later WAVs **overwrite** earlier ones.

Reproduced:
```
log_name = 'Alice_1234567890.wav.log' -> regex match: None -> user_id 0
```
Why tests miss it: `tests/test_voice_recorder.py` creates synthetic logs named `Bob_777.log` (no `.wav`), which the regex happens to match. Test fixtures don't reflect the real naming path.

**Fix:** strip `.wav.log` (or use `Path(...).stem` twice); add a regression test using `_log_path_for`-shaped names.

### M2. `/upload_kb` filename corruption: leading segments of hex-ish names are stripped
**File:** `kb/storage.py` L170–178 (`_sanitize_filename`)

The UUID-stripper removes the first `_`-segment if **all** its chars are hex digits — no length or "is really a UUID" check. Real filenames beginning with hex-able words lose that word:
- `cafe_notes.md` → `notes.md`
- `add_this.md` → `this.md` (`add` = valid hex!)
- `face_lift.md`, `bad_day.md`, `dead_end.md`, `beef_recipe.md` → first word stripped
- `_leading_underscore.md` → stripped too (`all()` over empty string is True)

**Fix:** only strip a leading segment when it is 32 chars (or parses via `uuid.UUID`).

### M3. HTTP 400 always classified as "model not found"
**File:** `bot_core/errors.py` L207

```python
if status_code in (400, 404) or "model not found" in error_msg ...
    return ModelNotFoundError(...)
```
A legitimate 400 (`context_length_exceeded`, malformed request, unsupported parameter — e.g. `response_format`/`max_tokens` rejections) is reported to the user as "model not available on the backend", sending them debugging the wrong thing. This is the classifier used by every AI path.

**Fix:** for 400, inspect the error `code`/message first (`context`, `invalid`, `unsupported`) and fall back to a generic `BadRequest` category; keep 404 → ModelNotFound.

### M4. Reminder store grows unboundedly (memory + JSON file)
**File:** `bot_core/reminders.py` L190–193, L72

Fired reminders are only marked `"fired": True`; they are never removed from `_reminders` nor from the persisted file. `_deliver_locks` (L72, L159) also accumulates one `asyncio.Lock` per reminder id forever. Over months of use both grow without bound; every `_save()` rewrites the whole file.

**Fix:** delete the entry after successful delivery (or after N days), and pop the lock in `_fire`'s finally.

### M5. `/reindex_kb` deletes the live index DB file and rebuilds it in place
**File:** `kb/index.py` L334–345; `commands/kb_commands.py` (`handle_reindex_kb`)

The handler builds a **new** `KBIndexStore(kb_path)` "to avoid disturbing the live singleton", but both stores share the same `persist_dir/vector_index.db`. `load(force_rebuild=True)` starts with `self._db_path.unlink()`. So during a rebuild:
- the on-disk cache of the *live* index is gone; if the rebuild then fails (backend down, crash), the next bot start has no usable cache and falls back to keyword retrieval until a successful rebuild;
- incremental persistence from the live store (`_persist...`, opening/closing sqlite per op) can interleave with the rebuild's writes.

**Fix:** build into a temp/staging persist dir and atomically swap the file on success (or version the filename).

### M6. Overdue reminders re-fire immediately on every reconnect (bounded, but noisy)
**File:** `bot_core/reminders.py` (`rearm_pending_reminders`, `_fire`)

`on_ready` runs again on full gateway reconnects; each re-arm restarts tasks with `delay = max(fires_at - now, 0)`, so a reminder whose earlier delivery attempts failed fires immediately on every reconnect. Total re-fires are bounded by `MAX_DELIVERY_ATTEMPTS` (verified — the attempts counter persists via `_save()`/`_load()`), so this is intended retry behavior; noting only that with a flapping gateway the user can get several near-instant late pings rather than spaced-out retries. Optional polish: space re-arms by `min(remaining_backoff, …)` instead of 0.

---

## LOW / polish

> **Status (2026-07):** L1–L9 fixed on branch `fix/audit-findings`; L10 reviewed and kept as a deliberate tradeoff (documented in code).

### L1. `channel_queue` timing logs are wrong
**File:** `utils/channel_queue.py` L68–77
`waited_s = time.monotonic() - started` is computed **before** `await q.get()` — always ≈0, so "ACQUIRED after Xs wait" never shows real waits. And `held_s = now - (started + waited_s)` measures queue+hold together, not the hold. Logging-only; fix by timing around `q.get()`.

### L2. `chunker._hash` uses built-in `hash()` — non-deterministic across processes
**File:** `kb/chunker.py` L441–444. `PYTHONHASHSEED` randomization means `header_hash` differs between runs. Currently harmless (the field is written but never compared across processes), but any future persistence/dedup use would silently break. Prefer `hashlib.blake2b(text.digest_size=8)`.

### L3. "Newest file" guesses after add_document/add_transcript
**File:** `commands/session_commands.py` (`_add_document_from_attachment` tail), `commands/recording_commands.py` (`_run_transcription`)
Both report the just-created file as `sorted(dir.iterdir())[-1]`. Concurrent uploads/transcripts can make the bot name the wrong file in its confirmation. Cosmetic; have `sessions.add_document/add_transcript` return the actual path.

### L4. `/remind` unit rendering: `"30 s"` → "Reminder set for **30** from now" (empty unit)
**File:** `commands/utility_commands.py` (`unit_singular = time_unit.rstrip("s")`). `"s".rstrip("s") == ""`. Use an explicit singular map.

### L5. `ask_ai_stream`: `stream` referenced where it may be unbound
**File:** `bot_core/ai_client.py` L1025 (`getattr(stream, "_final_finish_reason", ...)`). If `create()` raised and `_friendly_ai_error` ever *returned* instead of raising, this is a `NameError`. Currently unreachable (`_friendly_ai_error` always raises, L744–756) — keep as defensive note; initialize `stream = None` to be safe.

### L6. `handle_upload_kb` attachment path has no pre-read size guard
**File:** `commands/kb_commands.py` — `attachment.read()` buffers the whole file before `validate_upload_async` checks `MAX_FILE_SIZE`. Bounded by Discord's upload cap, so limited risk; consider checking `attachment.size` first (as `_add_document_from_attachment` already does).

### L7. Sync-send fragility in `/session_notes view`
**File:** `commands/session_commands.py` (`_show_session_notes`) — if a mid-list `followup.send` fails, remaining parts are lost and the exception propagates unhandled. Wrap loop in try/except with a final "…delivery interrupted" note.

### L8. Settings naming inconsistency
**File:** `config/settings.py` — `RERANK_TOP_K` reads env var `RAG_VECTOR_TOP_K`. Document or alias both.

### L9. Character display-name collisions are silent
**File:** `config/characters.py` — `_CHAR_DISPLAY_MAP = {c.display: c ...}` silently keeps the last character when two share a display name; `/character set <display>` then always picks the same one. Log a warning on collision.

### L10. Per-channel queues never reclaimed
**File:** `utils/channel_queue.py` — `_queues` entries live for the process lifetime (documented tradeoff; a dict of tiny objects, fine in practice).

---

## Verified NOT bugs / good hardening seen during second pass
- `utils/url_fetch.py`: scheme allow-list, DNS pinning, streamed size caps, redirect re-validation — solid.
- `bot_core/voice/session.py`: lock discipline around speaker maps/logs is correct; pending-packet buffer bounded (20/ssrc); `_process_mapped_packet` writes under the same lock `stop()` closes logs with.
- `bot_core/reminders.py`: `_is_transient`, attempt-capping and stale-grace logic behave as documented.
- `utils/embed_formatter.py`: fence-balanced splitting, empty-field-name protection, MAX_FIELDS overflow parking — all check out; pathological inputs degrade to plain text via `build_embed`'s never-raise contract.
- `bot_core/command_sync.py`: correctly distinguishes local-cache clear vs remote delete; propagates sync payload errors instead of reporting false success.
- `_friendly_ai_error` always raises → the "empty response" fallback paths in ai_client are safe.

## Fix status

All HIGH and MEDIUM findings (H1, H2, M1–M5) plus all LOW items (L1–L9; L10 documented) are fixed on branch `fix/audit-findings`, each with regression tests in `tests/test_audit_regressions.py` (+ updated delivery-semantics tests in `tests/test_channel_delivery.py`). Full suite: green.

## Suggested priority order
1. **H1** (session overwrite) and **H2** (persisted-load abort) — both silently destroy user data today.
2. **M1** (multi-speaker recovery collision), **M3** (400 misclassification — affects every AI error message).
3. **M2**, **M4**, **M5**.
4. Low items as polish; add regression tests mirroring real file naming (`*.wav.log`) and unnamed same-day sessions, which is exactly where the current suite's synthetic fixtures diverge from production behavior.
