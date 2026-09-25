---
name: run-tests
description: Run and interpret the pytest suite. Use when asked to run tests, check if tests pass, debug a test failure, or add new tests.
---

# Run tests

## Running the suite

Always use the project venv, never system Python:

```bash
.venv/Scripts/python.exe -m pytest -q                 # full suite
.venv/Scripts/python.exe -m pytest tests/<file>.py    # single file
.venv/Scripts/python.exe -m pytest -k <expr>          # by name/expression
```

(On Unix: `.venv/bin/python -m pytest …`.)

## Interpreting failures

1. Read the failing assertion with context. Distinguish product bugs from
   test-side environment artifacts (paths, line endings, locale, platform
   skips) before changing anything.
2. Verify hypotheses with a quick one-off: `.venv/Scripts/python.exe -c "..."`.
3. Fix the minimal side, re-run the single file, then the full suite.
4. Skips are not failures — report pass/skip counts separately and only
   investigate a skip if it is new or platform-wrong.

## Cross-platform notes

- Text fixtures: `Path.write_text()` translates `\n` → `\r\n` on Windows.
  If a test compares hashes or exact bytes of written text, write with
  `write_text(..., newline="\n")` so fixtures are byte-identical everywhere.
- Unix-only modules (`pty`, `fcntl`, …) must be import-guarded or the test
  skipped on other platforms.

## Adding tests

- Follow existing layout: `tests/test_<area>.py`, plain pytest, `tmp_path`
  fixtures.
- Keep tests hermetic: temp dirs for storage/state, no network, no live
  service clients.
- Mock at module boundaries; count calls with `patch.object` wrappers when
  the assertion is about *how often* something happened.

## After fixes

Commit test-only fixes with message prefix `fix(tests):`. See the `safe-push`
skill before pushing.
