# Whole-repo code review — findings report (2026-10)

Scope: all `bot_core/`, `kb/`, `commands/`, `config/`, `utils/` modules (~19.2k lines) plus
`main.py`, cross-checked against Ruff (196 production findings, saved during the review) and the
#73 incident. Severity: **H** = affects users now, **M** = latent, triggers under specific
configs/inputs, **L** = hygiene/robustness.

---

## 1. Confirmed / latent bugs (fix recommended)

### 1.1 `bot_core/tts.py` — chat-model swap-back repeats the #73 bug (latent H, config-gated)

`_swap_chat_model_back()` → `_ensure_slug_loaded(DEFAULT_MODEL, …)` POSTs

```json
{"model_path": "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M"}
```

This is exactly the payload shape Studio rejected in #73 (`Repo id must use alphanumeric
chars…`). Three echoes of the same class of bug:

1. No repo/quant split — `model_path` must be `unsloth/Qwen3.8-27B-GGUF` with
   `gguf_variant: "Qwen3.8-27B-NVFP4_50-Q4K_M"` (see `bot_core/images.py::_warm_chat_reload`,
   fixed in af654b2).
2. The residency poll compares `m.get("id") == model_path` — Studio reports loaded models as
   the **bare** repo id, so even a successful load would never be confirmed.
3. The load `POST` response status is never checked, and the poll loop just times out after
   `_CHAT_RELOAD_WAIT_S` — a permanent 500 is silent.

**Why it hasn't bitten yet:** `TTS_BACKEND=ollama` in `.env`, and every entry point early-returns
for non-studio backends. Flip `TTS_BACKEND=studio` and every `/speak` silently loses its
"next question finds the chat model warm" optimization — the exact #73 symptom.

**Fix:** extract the repo:quant split from `images.py` into one shared helper (e.g.
`utils/studio_model.py`) and use it in both `_ensure_slug_loaded` call sites; check the POST
status and log a warning on failure; compare residency as `id == repo` (+ variant where
reported).

### 1.2 `commands/roll_items_command.py` — button records use the **raw user table string** (M)

`pairs = [(table, i.name) …]` uses the user-typed `table` (e.g. `Magic_Items` or
`magic_items.csv`), while `item_state` keys and `Item.table` use the **canonical** CSV stem
(`magic_items`). Consequences:

- `build_view()`'s `state.get(table)` lookup misses the bucket → buttons render with the wrong
  🚫/↩️ label after rehydrate or a toggle (`_handle_toggle` rebuilds from `_sibling_pairs`, i.e.
  the same raw strings).
- The toggle itself still works (`item_state.toggle_item` lowercases and strips `.csv`), so
  state is correct but the **rendered labels are wrong** until a roll re-sends the view.

`commands/item_commands.py` does this right (uses `item.table`).

**Fix:** in `roll_items_command.py`, resolve the canonical stem (`load_items` already matched
one — return/use `result["items"][0].table`) and build `pairs` from it.

**Related edge (`bot_core/roll_buttons.py`):** `encode_custom_id` truncates names >~90 chars
without marking the truncation; `decode_custom_id` then returns the truncated name and
`toggle_item` adds a dead state entry instead of toggling the real item. Consider storing the
pair in the message record and resolving clicks through the record (already the fallback via
`_message_pairs`) instead of trusting the decoded name.

### 1.3 Untracked fire-and-forget tasks (M — the #73 task-GC class)

Strong-reference bug pattern (loop keeps only weak refs to tasks; CPython docs require a
strong ref for fire-and-forget):

| Site | Status |
|---|---|
| `bot_core/images.py:442` warm reload | **fixed** — `_background_tasks` set (af654b2) |
| `bot_core/tts.py:169` | tracked via module global `_chat_reload_task` ✓ |
| `bot_core/reminders.py:300/359` | tracked via `_tasks` / `_retry_tasks` dicts ✓ |
| `bot_core/sessions.py:567` `_index_session_file` | **untracked** — `fut` is local, dropped on return |
| `kb/retrievers.py:126/155` `_safe_shutdown_store` | **untracked** ×2 |

In practice these survive because the await-chain holds refs, but the #73 investigation proved
this pattern makes failures invisible in logs. **Fix:** use `utils.background_tasks.spawn_tracked_task`
at the three remaining sites (sessions auto-index, both `_safe_shutdown_store` calls).

### 1.4 `/help` lists itself under "Other" (L)

`_CATEGORIES` in `commands/help_command.py` has no `"help"` entry, so `/help` shows in the
overview under "Other". One-line fix.

---

## 2. Consistency / robustness issues

### 2.1 Settings bool parsing drifts from the `_safe_bool` convention (L/M)

After the P3 #32 consolidation, two flags still parse booleans ad-hoc:

- `SESSION_IMAGES: str = os.getenv(..., "1")` + `session_images.py` checks `== "0"`.
  `SESSION_IMAGES=false` in `.env` would **enable** the feature.
- `CONTEXTUAL_BACKGROUND_FILL` uses `in ("1","true","yes")` directly — misses `on/y/true` variants
  handled by `_safe_bool`.

Both documented, but they're the only bool-ish env vars a user can trip over. Recommend
migrating to `_safe_bool` (keeping current defaults).

### 2.2 Naive local-time timestamps (L)

`bot_core/sessions.py` (7×), `commands/session_commands.py` (2×), `kb/storage.py` (2×),
`bot_core/ai_client.py:631` format epoch times with naive `datetime.fromtimestamp(...)` in the
**server's** TZ. Display-only, so harmless per se — but `sessions.py:632` re-parses
`YYYY-MM-DD HH:MM` back with `.timestamp()` (local, DST-sensitive), and a session folder written
before a DST change re-reads an hour off afterwards. Only matters if the container TZ ≠ user TZ
or across DST boundaries; worth one shared tz-aware helper if sessions are ever named/compared
programatically.

### 2.3 Chunker size-cap contradiction (L)

`Chunker.MAX_CHUNK_SIZE = 7500` ("embedder 2048-token safety margin") but the full-document
strategy keeps files ≤ **8000** chars intact — a 7 501–8 000 char file becomes one chunk above
the stated cap (and > 2048 tokens for many tokenizers → backend-side truncation). Align the two
constants (e.g. full-doc threshold = `MAX_CHUNK_SIZE`).

### 2.4 `retrievers` singleton ignores `kb_path` once built (L, by design — document it)

`_ensure_index_store(kb_path)` returns the cached singleton regardless of `kb_path` on later
calls; `update_kb_document` falls back to `KB_PATH` when no store exists. Fine for the
single-KB design, but if `SESSIONS_NOTES_DIR` is ever moved **outside** `KB_PATH`
(documented Obsidian setup!), session files index into the KB store anyway via
`update_single_document(path)` — works today, but the mismatch deserves a comment/assert.

### 2.5 Rerank content→name map collision (L)

`kb/retrievers.py` builds `name_map.setdefault(content, name)` to map rerank scores back to
chunk names. Two distinct chunks with **identical content** (plausible in duplicated KB
entries) collapse to one name — gate scores then apply to only one of them. Harmless in
selection (dedup τ 0.92 catches near-dupes); note only.

### 2.6 `typing_loop_task` swallows all non-cancel errors (L)

`except Exception: pass` for permission/channel errors is intentional and bounded (10 s
cadence, 30 s default), but a permanently failing channel keeps re-trying every 10 s for the
whole duration. Acceptable; a stop-after-2-consecutive-failures would be tidier.

---

## 3. Ruff triage (196 production findings)

| Rule | Count | Verdict |
|---|---|---|
| `BLE001` blind `except Exception` | 75 | ~90% intentional "never break the bot/user turn" guards with logging — matches module docstring contracts (sessions, retrievers, turn_logger, vault_sync…). Keep. |
| `TRY400` (`log.error` without `exc_info`) | 27 | Mostly cosmetic; converting to `log.exception`/`exc_info=True` would improve tracebacks in exactly the places the check-logs skill has to dig through. Low-risk sweep. |
| `S110` `try/except/pass` | 19 | Highest concentration in Discord-delivery fallbacks where the reply path is genuinely terminal. Add `# noqa: S110 — <reason>` to make the intent explicit rather than silencing the rule repo-wide. |
| `E402` imports not at top | 15 | Deliberate lazy imports (circular-import avoidance: `config.settings` reads inside functions so tests can monkeypatch; heavy libs lazy). Keep. |
| `TRY300` return-in-try | 12 | Style only. Skip. |
| `DTZ006/005` naive datetimes | 9 | See §2.2. |
| `B023` loop-closure capture | 6 | Checked individually: the reminders closure is explicitly guarded (comment in `_schedule_retry`); remaining ones bind values at definition. No live bug found. |

Net: no additional real bugs beyond §1.1–§1.3 surfaced by lint.

---

## 4. Test hygiene

`tests/test_p2_vector_numpy.py::TestBenchmark::test_numpy_is_faster_than_python_at_scale`
asserts a wall-clock speedup > 1.0× and fails under full-suite load while passing in isolation
(observed twice this session). Recommend `@pytest.mark.skipif`-free but tolerant: assert
`< 3.0×` headroom, or mark it `benchmark` and exclude from the default run so a real
regression signal isn't drowned by a known-flaky timing test.

---

## 5. What's solid (no action)

- **Concurrency architecture**: `_gen_lock` global model-swap window, per-channel `/ai` run
  registry (`ai_runs`) + per-channel outbound queue (`channel_queue`), mutation lock in
  `KBIndexStore`, index-init lock in `retrievers` — all coherent and correctly used.
- **Blocking-IO discipline**: consistent `asyncio.to_thread`/executor offloading (reader,
  chunker, roll/item commands, vault git calls) per the P0 #4 convention.
- **Never-break-a-turn contracts** in `sessions`, `turn_logger`, `history`, `reranker`,
  `lexical`, `query_rewriter`, `agent_loop` (graceful degradation to single-turn fallback) —
  consistently implemented and documented.
- **Atomic writes** everywhere that matters (sessions notes, reminders, index cache, item
  state, roll button records, context-lines sidecar): temp file + `os.replace`.
- **SSRF/size hardening** in `url_fetch`, attachment guards, prompt-budget trimming
  (history first, then lowest-ranked RAG docs), embedding-space versioning via
  `effective_embedding_model()` metadata → automatic reindex on backend switch.
- **`command_sync`** correctly models Discord's global-vs-guild registration semantics.
- **`embed_formatter`** respects all Discord embed limits (4096/1024/25 fields) with
  fence-safe splitting.

---

## 6. Suggested action order

1. **§1.1** shared repo:quant split helper used by `images.py` + `tts.py` (+ POST status check).
   Small, high value, prevents a repeat of #73 the moment `TTS_BACKEND=studio` is tried.
2. **§1.3** three `create_task` sites → `spawn_tracked_task`.
3. **§1.2** canonical table stem in `roll_items_command` pairs.
4. **§2.1** `_safe_bool` migration for `SESSION_IMAGES` / `CONTEXTUAL_BACKGROUND_FILL`.
5. **§1.4** + **§4** + **TRY400 sweep** as a hygiene pass.

None of these are urgent for the current production config; 1–2 are the ones most likely to
turn into a real incident.

---

## 7. Implementation status (2026-10-29, same session)

Implemented on `feat/tool-calling` (commit `a97d331`; hermetic tests only — no live Studio
verification in this pass; the host was out of GPU headroom, so the TTS fix ships
untested-against-Studio but payload-identical to the live-verified #73 shape):

- **§1.1 done** — new `utils/studio_model.py` (`split_repo_variant` / `load_payload` /
  `is_resident`); `bot_core/tts._ensure_slug_loaded` now splits repo:quant, checks the POST
  status (returns False instead of silently polling), polls residency by bare id + quant,
  and gives the blocking load POST the full wait budget (was a 30 s client timeout against a
  ~70 s POST-blocking load — a second latent bug found during implementation).
  `bot_core/images.py` stays untouched — its inline equivalent is pinned by the live-verified
  #73 test suite. New tests: `tests/test_studio_model.py` (12), `tests/test_tts_swapback.py`
  (+5: payload split, 4xx-no-poll, bare-id+quant residency, wrong-quant rejection).
- **§1.3 done** — `sessions._index_session_file` and both `retrievers` shutdown tasks now use
  `spawn_tracked_task`.
- **§1.2 done** — `roll_items_command` pairs now use the canonical `Item.table` stem.
- **§2.1 done** — `SESSION_IMAGES` and `CONTEXTUAL_BACKGROUND_FILL` migrated to `_safe_bool`
  (`SESSION_IMAGES=false` no longer means enabled); `session_images` gate + tests updated.
- **§1.4 done** — `/help` categorized under "AI & Chat".
- **§4 done** — benchmark test switched to best-of-rounds timing (regression signal intact,
  absolute cap relaxed to 10 ms); passed inside a full-suite run.
- **TRY400 sweep** — deliberately NOT done in this pass (27 cosmetic log-quality edits across
  many modules; low value, needless diff noise). Left open.

Full suite after all changes: **1279 passed, 0 failed**.

### Second batch (same day — "fix the remaining issues before I test")

- **§2.2 done** — new `utils/timefmt.py` (`DISPLAY_FMT` + `fmt_local`/`parse_local`);
  the five duplicated `fromtimestamp().strftime("%Y-%m-%d %H:%M")` sites (sessions,
  session_commands, ai_client) and the ad-hoc note re-parse now share one symmetric
  pair. Naive-local storage semantics (and existing note files) unchanged; the DST
  ambiguity is documented at the single parse site. The §2.2 note about `kb/storage.py`
  was **wrong** — it already uses tz-aware UTC; left untouched.
- **§2.3 done, deeper than first reported** — while fixing it, two more oversized-chunk
  holes surfaced: `_split_adaptive`'s `line_count < 50` single-block shortcut had NO
  size check (one-line 8 KB files bypassed everything), and neither `_split_by_paragraphs`
  nor `_split_recursive_by_size` nor even `_hard_split` could split a single giant
  paragraph/line. Fix: `FULL_DOC_CHAR_LIMIT = MAX_CHUNK_SIZE` (7500), adaptive shortcut
  size-guarded, `_hard_split` slices monster lines at word boundaries, and
  `split_file_sync` now enforces a hard cap on ALL chunks from every strategy.
  Existing caches are keyed by SOURCE-FILE hash (P2 #19), so unchanged KB files are
  NOT invalidated — only re-ingested files re-chunk. New test:
  `tests/test_chunker_full_doc_limit.py`.
- **§2.4 done** — `_ensure_index_store` logs a warning when asked for a KB path different
  from the built singleton (`_warn_if_wrong_path`).
- **§2.5 done** — the rerank block no longer builds a `content→name` map: candidates are
  ranked as (name, content) pairs, so identical-content chunks keep their own names and
  scores; `rerank_gate` derives from the ranked candidates.
- **§2.6 done** — `typing_loop_task` now breaks on `discord.Forbidden`/`NotFound`
  (the docstring always promised this); transient errors still retry.
  New test: `tests/test_typing_loop.py`.
- **§1.2 follow-up (custom_id truncation) done** — `_handle_toggle` prefers the canonical
  message record over decoding rendered buttons (long names were truncated in `custom_id`
  and could leak into records/rehydrated views); record saves unified in
  `_save_records_logged` with `exc_info`. New test in `tests/test_roll_buttons.py`.
- **TRY400 sweep done** — 22 `log.error(...: %s", exc)` sites across 13 modules now carry
  `exc_info=True` (AI request failures, index saves, batch images, session reminders,
  OCR/translate/summarize/TTS, command sync, subfolder creation, roll record saves).
  Two candidates skipped (they hold no exception object).

Full suite after second batch: **1292 passed, 0 failed**.
