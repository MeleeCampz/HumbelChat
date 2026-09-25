---
name: run-tests
description: Run and interpret the HumbelChat pytest suite (Windows + venv specifics). Use when asked to run tests, check if tests pass, debug a test failure, or add new tests.
---

# Run tests (HumbelChat)

## Running the suite

Always use the project venv (Python 3.12), never system Python:

```bash
.venv/Scripts/python.exe -m pytest -q                 # full suite (~30s)
.venv/Scripts/python.exe -m pytest tests/<file>.py    # single file
.venv/Scripts/python.exe -m pytest -k <expr>          # by name/expression
```

## Windows quirks (this repo is developed on Windows/Git Bash)

- `tests/test_scripts.py` skips automatically on Windows (`pty` is Unix-only). A
  "6 skipped" result is normal and healthy here, not a failure.
- Text fixtures: `Path.write_text()` translates `\n` → `\r\n` on Windows. If a
  test compares hashes or exact bytes of written text, write with
  `write_text(..., newline="\n")` so fixtures are byte-identical across platforms.
- Deprecation warning from `discord.player` importing `audioop` is pre-existing
  and harmless on Python 3.12 (removal lands in 3.13).

## Interpreting failures

1. Read the failing assertion with context — most past failures were test-side
   environment artifacts, not product bugs. Check the product code before changing it.
2. Verify hypotheses with a quick one-off: `.venv/Scripts/python.exe -c "..."`.
3. Fix the minimal side (usually the test), re-run the single file, then the full suite.

## Adding tests

- Follow existing layout: `tests/test_<area>.py`, plain pytest, `tmp_path` fixtures.
- Keep tests hermetic: temp dirs for KB/storage, no network, no Discord client.
- Mock at module boundaries (see `_ReadCounter` pattern in
  `tests/test_p2_storage_hash_cache.py` for counting calls via `patch.object`).

## After fixes

Commit test-only fixes with message prefix `fix(tests):`. Ask before pushing to
master — see the `safe-push` skill.
