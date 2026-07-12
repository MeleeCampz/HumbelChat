---
name: python-file-corruption
description: Fix write_file artifact injection (extra ** and CJK ） characters) that corrupts generated Python files, using compile-check and targeted stripping.
source: auto-skill
extracted_at: '2026-07-12T15:14:47.807Z'

---

When writing new files for a project (especially `.py` files), verify compilation before declaring success. write_file occasionally injects generation artifacts like `**`, `**:` or CJK fullwidth closing parentheses ） into the content.

## Diagnosis
1. Read back any file that might have been corrupted by checking for trailing ** on lines: grep -n '**' *.py or just compile-check.
2. Use python3 -c "import py_compile; py_compile.compile('/path/to/file.py', doraise=True); print('OK')" to verify compilation.

## Fix procedure
- When corruption is found, strip all ** artifacts from the file:
  ```bash
  python3 << 'EOF'
  path = '/home/user/discord-ai-bot/kb/scorch.py'
  with open(path) as f: text = f.read()
  open(path, 'w').write(text.replace('**', ''))
  EOF
  ```
- Also handle standalone **: or ): patterns that may not be caught by simple replacement.
- Re-run compile-check to confirm clean import-free syntax.

## Prevention
After creating any files worth keeping: run compile on each .py file immediately before writing any more code. Fail fast rather than carrying corrupted modules forward.
