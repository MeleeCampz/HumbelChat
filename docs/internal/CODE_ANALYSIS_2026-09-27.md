# Code Analysis — 2026-09-27

Full in-depth audit of the HumbelChat codebase (backlog item #2). Goal: find
real bugs, confirm them with regression tests, fix them, and catalogue
improvements for the backlog.

## Scope & method

- **In depth:** all of `bot_core/`, `kb/`, `commands/`, `utils/`,
  `config/settings.py`, `main.py` (~15 K lines of production code). Every
  module was read end-to-end; locking, error paths, and edge cases were
  traced, not skimmed.
- **Lighter pass:** voice subsystem internals (`bot_core/transcriber.py`,
  `bot_core/voice/*`) — structure reviewed, no line-by-line trace.
- **Baselines captured first** (before any change): pytest
  `7 failed / 717 passed / 6 skipped` (all 7 failures pre-existing and
  environment-related: Windows-local path resolution ×2, embed-formatter ×3,
  ai-command persona ×1, p5 streaming total-cap ×1), ruff clean, mypy
  290 issues across 36 files.
- **Rules:** no behaviour changes beyond confirmed bugs; the freshly tuned
  RAG pipeline (F1–F7) was only touched if a genuine bug was found (none);
  one logical fix per commit, each with a regression test; full pytest after
  every fix — final state `7 failed / 721 passed`, i.e. **zero new failures**,
  +4 new passing tests.

## Confirmed bugs (fixed)

### B1 — P4 truncation retry used the wrong timeout · `bot_core/ai_client.py`

**Severity: medium.** When a thinking model exhausts its whole `max_tokens`
budget on reasoning (`finish_reason="length"`, empty content), `ask_ai`
retries once with a 4× token budget (P4). But the retry passed
`timeout=ctx.timeout_sec` — a timeout scaled by `_scaled_timeout()` for the
*original, smaller* budget. A model that needed more tokens also needs more
wall-clock time, so the retry was systematically under-timed and would time
out in exactly the scenario P4 exists to rescue (worst case: short character
budgets like 2 K → 8 K tokens, a ~3 s shortfall at default settings).

**Fix:** re-scale the timeout for the retry budget:
`retry_timeout = _scaled_timeout(ctx.total_chars, retry_budget)`.

**Regression test:** `tests/test_p4_truncated_response.py::
test_retry_timeout_scales_with_retry_budget` — asserts the retry call's
timeout exceeds the first call's by exactly the output-budget delta.

### B2 — Unbound `recorder` in start-recording error handlers · `commands/recording_commands.py`

**Severity: low.** In `handle_start_recording`, both `except` handlers call
`recorder.discard()`. If `_get_recorder(bot)` itself raised (e.g. a
`discord.Forbidden` from voice state), `recorder` was never bound → the
handler crashed with `NameError`, masking the real error and leaving the
user without *any* message.

**Fix:** initialise `recorder = None` before the `try`; guard both
`discard()` calls with `if recorder is not None`.

**Regression test:** `tests/test_stop_recording_stt.py::
test_get_recorder_failure_does_not_raise_nameerror` — makes `_get_recorder`
raise `Forbidden`, asserts the friendly permission message is sent and no
exception propagates.

### B3 — Malformed embedding response → unclassified `KeyError` · `kb/embedder.py`

**Severity: low-medium.** `_remote_encode` trusts that the backend returns
one vector per input. A malformed `/embeddings` response with *fewer*
`data` entries than inputs (some backends truncate or drop items) slipped
past the retry loop, filled only part of `all_embeddings`, and then raised a
raw `KeyError` at order-reconstruction — bypassing the normal failure path:
no per-batch local-CPU fallback, no keyword-only retrieval fallback, just an
unclassified traceback.

**Fix:** validate `len(vectors) == len(batch)` inside the same `try` that
wraps `_call_api`, raising `EmbeddingError` — which routes into the existing
per-batch local fallback (when enabled) or the caller's keyword fallback.

**Regression tests:** `tests/test_p2_embedder_client.py::
TestMalformedResponseLength` (2 tests: raises classified `EmbeddingError`
without fallback; falls back to local vectors with order preserved when
fallback is enabled).

## Minor findings (documented, not fixed)

Deliberately left alone — no user-visible failure in any realistic scenario;
fixing would be churn beyond "confirmed bugs".

| # | Location | Finding | Why not fixed |
|---|----------|---------|---------------|
| M1 | `kb/storage.py` | Uploading a file literally named `..` sanitises to a hidden `.txt` in the KB root: the indexer skips dotfiles, so the user is told "uploaded" but nothing is ever retrievable. | Purely cosmetic edge case; no security issue (stays inside the KB root). A one-line guard could be added if it ever bites. |
| M2 | `utils/embed_formatter.py` | When the *first* chunk overflows even after the rebuild pass, the fallback plain-text path drops `title_override`. | Cosmetic (title only), and only in a rare overflow combination. |
| M3 | `commands/utility_commands.py` | `/summarize` and `/translate` hard-split long input by character count, which can cut mid-word at chunk boundaries (the OCR path uses paragraph-aware splitting). | Cosmetic — the LLM handles mid-word splits fine; no data loss. |

## Backlog improvements

| # | Item | Notes |
|---|------|-------|
| I1 | **mypy cleanup campaign** — ~~290 issues across 36 files~~ → **done (2026-10)**: `mypy bot_core commands kb utils config main.py` now reports **0 errors** across all 50 source files. | Executed incrementally, one module per session, no behaviour changes except two sanctioned bug fixes: (1) `await stream.close()` in `ai_client.py` (the un-awaited async close leaked streaming HTTP connections), and (2) `message.guild_id` → `message.guild.id if message.guild else 0` in `main.py`'s prefix path (DM messages have no `guild_id`). Kept in default mode (no strict promotion). |
| I2 | **Document the `_looks_german` trigger** in `kb/retrievers.py`. With `RAG_REWRITE_ALL_QUERIES=1` (default) every query is rewritten, so the German-detection heuristic only matters behind the `RAG_REWRITE_ALL_QUERIES=0` kill-switch. | Docs/comment only — no code change needed. |
| I3 | **The 7 pre-existing test failures** are environment artifacts (Windows-local path expectations ×2, embed-formatter/ai-command/p5 mismatches ×5). | Worth a pass to either fix the tests or mark them platform-conditional so CI runs green. |

## Per-module notes (highlights)

Everything below was reviewed and found **sound** — these are the
non-obvious invariants a future maintainer should not break:

- **`bot_core/sessions.py`** — session files use atomic writes
  (temp + `os.replace`) and per-store mutation locks; fire-and-forget index
  tasks are safe because each store serialises mutations on `_mut_lock`.
- **`kb/index.py`** — all mutations under the store lock, atomic persists,
  and legacy-format migrations run idempotently on load. The most complex
  module in the repo; no locking gaps found.
- **`kb/storage.py`** — path traversal is guarded (resolved paths must stay
  under the KB root), extension allowlist + size limits enforced. See M1 for
  the one cosmetic edge case.
- **`kb/lexical.py`** — clean Okapi BM25 implementation, correct IDF and
  length normalisation.
- **`bot_core/reminders.py`** — delivery locking is sound: the fired-check
  happens *inside* the per-id lock, so two concurrent firings cannot
  double-deliver.
- **`utils/url_fetch.py`** — robust SSRF protection: DNS pinning (connect to
  the resolved IP, verify hostname), redirect re-verification, size caps.
- **`bot_core/ai_client.py`** — the global AI slot serialises all LLM calls
  process-wide (single local backend); `_scaled_timeout` scales with both
  prompt chars and output budget, capped at 8× base; P1 #7 transient retry
  is bounded to one attempt. (B1 above was its only real defect.)
- **`commands/kb_commands.py`** — URL uploads reuse the SSRF-guarded fetcher
  with a streamed size cap before anything touches disk.
- **`utils/embed_formatter.py`** — complex but carefully bounded: every
  overflow path has a measured fallback, and nothing can produce an
  over-length embed (see M2 for one cosmetic gap).
- **`main.py`** — orchestrator wiring, logging setup, and event handlers are
  all clean; shutdown closes the shared httpx client and voice state.

## Result

| | Before | After |
|---|--------|-------|
| pytest | 7 failed / 717 passed / 6 skipped | 7 failed / **721** passed / 6 skipped (same 7 pre-existing failures) |
| ruff | clean | clean |
| mypy | 290 issues | 290 issues at analysis time; **0** after the I1 campaign (2026-10) |

Three real bugs fixed, four regression tests added, zero behaviour changes
beyond the fixes, zero new test failures.

## Addendum — backlog item #4 (same day): the 7 pre-existing failures

Working through the 7 baseline failures surfaced one more production bug:

### B4 — Stream total-cap timeout misclassified at the boundary · `bot_core/ai_client.py`

**Severity: low.** The streaming loop caps each chunk wait at
`min(gap_deadline, hard_deadline)` and, after an `asyncio.TimeoutError`,
decides between "total time budget" and "per-chunk timeout" by re-comparing
`loop.time() >= hard_deadline`. On Windows the event-loop timer can fire a
few ms **early** (reproduced: up to ~6 ms), so a wait that was bounded by the
total cap could be classified as a per-chunk timeout — wrong error detail in
logs and user messages right at the budget boundary.

**Fix:** record up front which budget bound the wait
(`total_cap_binding = gap_deadline >= hard_deadline`) and use that flag for
the classification; the `loop.time()` comparison remains as a second guard.

The other six failures were test-side staleness / environment pollution, no
product bugs:

- **path-resolution ×2** — `main.py` runs `load_dotenv()` at import and `.env`
  sets `KB_PATH=./data/knowledge` (Docker-relative). Any later
  `importlib.reload(config.settings)` (used by two test files) re-read the
  polluted environment. Fixed in `tests/conftest.py`: pre-set absolute
  `KB_PATH` before anything imports `main` (`load_dotenv` never overrides
  existing variables).
- **embed-wiring ×3** — written before streaming became the default
  (`AI_STREAM=1`); they patched `ask_ai` but the command now routes through
  the real `ask_ai_stream`. Fixed by pinning `AI_STREAM=False` in those
  tests (they exercise the delivery path, not streaming).
- **ai-command persona ×1** — asserted an exact system-prompt match; the
  global `<stat-block-format>` appendix is now added to every character.
  Fixed to assert prefix + appendix presence.

Final state after #4: **0 failed / 728 passed / 6 skipped**, ruff clean.
