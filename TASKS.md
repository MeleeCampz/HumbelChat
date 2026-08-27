# Code-Review Task List

*Tracking fixes for `CODE_REVIEW_FINDINGS.md`. Updated 2026-08-27.*

## Status

- ✅ done and tested
- 🔄 in progress
- ⬜ open

## P0

| # | Task | Commit | Tests | Status |
|---|------|--------|-------|--------|
| 1.1 | RAG zero-doc budget bug | `22a1720` | `tests/test_adaptive_rag.py` | ✅ |
| 2.4 | Model-not-found guard (stale slugs) | `3ff2167` | `tests/test_bot_core.py` | ✅ |

## P1

| # | Task | Commit | Tests | Status |
|---|------|--------|-------|--------|
| 1.2 | Honour per-character temperature | `b6f9587` | `tests/test_bot_core.py` | ✅ |
| 1.3 | Reuse single `AsyncOpenAI` client | `2987598` | `tests/test_bot_core.py` | ✅ |
| 1.4 | Persist chat history + active character | `9bfa1c2` | `tests/test_history_persistence.py` | ✅ |
| 1.6 | Typing indicator on `!ai` prefix path | `f68ac76` | `tests/test_ai_command.py` | ✅ |

## P2

| # | Task | Commit | Tests | Status |
|---|------|--------|-------|--------|
| 1.7 | Persist reminders across restarts | `03d4734` | `tests/test_reminders.py` | ✅ |
| 2.1 | Vector index content-hash cache | `0d65167` | `tests/test_kb_vector_index.py` | ✅ |
| 3.1 | Per-channel rate limit | `0d65167` | `tests/test_bot_core.py` | ✅ |
| 3.2 | Streaming responses | `0d65167` | `tests/test_ai_command.py` | ✅ |
| 3.3 | RAG context in user message | `0d65167` | `tests/test_adaptive_rag.py` | ✅ |
| 3.5 | Input length cap | `0d65167` | `tests/test_ai_command.py` | ✅ |

## P3

| # | Task | Commit | Tests | Status |
|---|------|--------|-------|--------|
| 1.5 | Remove dead port check in single-instance lock | `39b76bc` | `tests/test_main_startup_guards.py` | ✅ |
| 1.8 | Utility commands use active character model | `39b76bc` | `tests/test_utility_commands.py` | ✅ |
| 1.9 | OCR download timeout + size + type guards | `39b76bc` | `tests/test_utility_commands.py` | ✅ |
| 1.10 | Suppress test file-log handlers | `39b76bc` | `tests/test_main_startup_guards.py` | ✅ |
| 1.11 | Circular import (`main` ↔ commands) | `03d4734` | `tests/test_utility_commands.py` | ✅ |
| 2.5 | Absolute `characters.json` path | `03d4734` | `tests/test_path_resolution.py` | ✅ |
| 3.4 | CWD-independent `characters.json` | `0d65167` / `03d4734` | `tests/test_path_resolution.py` | ✅ |
| 3.10 | Pin all dependencies | `39b76bc` | — | ✅ |

## P3 (second wave — low priority / operational)

| # | Task | Notes | Tests | Status |
|---|------|-------|-------|--------|
| 2.3 | Prompt-size-aware timeout | `_scaled_timeout()` in `ai_client.py`, both paths | `tests/test_bot_core.py` (suite) | ✅ |
| 2.6 | Duplicate log lines | Handlers on `bot` logger only, no propagation | `tests/test_main_startup_guards.py` | ✅ |
| 3.6 | `_or_clear` → `_history_reset_flag()` | Returns `bool`; accepts clear/1/true/yes | suite | ✅ |
| 3.7 | Structured error taxonomy | `bot_core/errors.py` + wiring in chat paths | `tests/test_error_taxonomy.py` | ✅ |
| 3.8 | Splitter edge cases | Header-aware safe chunk budget, line-level fallback | `tests/test_response_splitter.py` (extended) | ✅ |
| 3.9 | Backend health probe | `bot_core/health.py`, startup + optional periodic | `tests/test_health_probe.py` | ✅ |
| 4.1 | Redundant local imports in `_enforce_single_instance` | Uses module-level `os` | suite | ✅ |
| 4.2 | Lazy import of `log_top_kb_files` in `on_ready` | Top-level import (also typing/splitter/RateLimitError) | `tests/test_bot_startup.py` | ✅ |
| 4.3 | Token estimate was word count | Char-based estimate (`len // 4`) | suite | ✅ |
| 4.4 | Typing task GC risk | `utils/background_tasks.spawn_tracked_task()` | `tests/test_background_tasks.py` | ✅ |

## Open (not yet fixed)

| # | Task | Notes | Status |
|---|------|-------|--------|
| 2.2 | Embedding endpoint instability | Backend-side; keyword fallback works | ⬜ |

## Test Suite

**142 passing** as of this update. Run with `python -m pytest tests/ -q`.
