# Code-Review Task List

*Tracking fixes for `CODE_REVIEW_FINDINGS.md`. Updated 2026-08-22.*

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

## Open (not yet fixed)

| # | Task | Notes | Status |
|---|------|-------|--------|
| 2.2 | Embedding endpoint instability | Backend-side; keyword fallback works | ⬜ |
| 2.3 | AI request timeouts (120 s+) | Backend-dependent; see 3.7 | ⬜ |
| 2.6 | Duplicate log lines | Handler propagation | ⬜ |
| 3.6 | Rename `_or_clear` | Naming nit | ⬜ |
| 3.7 | Structured error taxonomy | Timeout / model-not-found / rate-limit / down | ⬜ |
| 3.8 | Response splitter edge cases | Markdown expansion | ⬜ |
| 3.9 | Health-check endpoint | Fail-fast on backend down | ⬜ |
| 4.* | Cosmetic items | See §4 | ⬜ |

## Test Suite

**124 passing** as of `cd54cee`. Run with `python -m pytest tests/ -q`.
