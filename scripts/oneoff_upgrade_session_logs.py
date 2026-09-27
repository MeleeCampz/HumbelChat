"""One-off: upgrade legacy session folders to the AI-merged session log.

For every session folder under data/knowledge/session_notes/ that still has
the mechanical '## Documents (combined uploads + transcripts — full text)'
section, this script runs the same AI merge /end_session uses and replaces
that section with '## Session Log (combined from all uploads — AI-merged)'.

A .bak copy of each notes.md is written first.  Folders without a mechanical
section or where the AI call fails are left untouched.

Run:  .venv/Scripts/python.exe scripts/oneoff_upgrade_session_logs.py
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import re
import shutil
import sys

# The .env INFER_URL points at host.docker.internal (docker-internal); from the
# host the same backend is on loopback.  Set BEFORE config.settings is imported;
# load_dotenv() below must not override it.
os.environ["INFER_URL"] = "http://127.0.0.1:8888/v1"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)

import config.settings  # noqa: E402,F401  (reads env at import time)
from commands.session_commands import _generate_merged_log  # noqa: E402

ROOT = pathlib.Path("data/knowledge/session_notes")
DOCS_RE = re.compile(
    r"^## Documents \(combined uploads \+ transcripts — full text\)\s*$",
    re.MULTILINE,
)


async def upgrade(folder: pathlib.Path) -> None:
    notes = folder / "notes.md"
    text = notes.read_text(encoding="utf-8")
    m = DOCS_RE.search(text)
    if not m:
        print(f"SKIP {folder.name}: no mechanical Documents section")
        return

    name_m = re.match(r"# Session: (.+)", text)
    session = {
        "name": name_m.group(1).strip() if name_m else None,
        "dir": str(folder),
        "file": str(notes),
        "notes": [],
    }
    merged = await _generate_merged_log(session, None, 0)
    if not merged:
        print(f"FAIL {folder.name}: merge returned None — file left unchanged")
        return

    # The Documents section is the last top-level section (appended by the
    # layout migration); internal H2s belong to its content, so replace from
    # the heading to EOF.
    new_text = (
        text[:m.start()].rstrip() + "\n\n"
        "## Session Log (combined from all uploads — AI-merged)\n\n"
        + merged.strip() + "\n"
    )
    shutil.copy2(notes, notes.with_suffix(".md.bak"))
    tmp = notes.with_suffix(".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    os.replace(tmp, notes)
    print(f"OK   {folder.name}: notes.md {len(text)} -> {len(new_text)} chars "
          f"(merged log: {len(merged)} chars; backup: notes.md.bak)")


async def main() -> None:
    folders = [p for p in sorted(ROOT.iterdir())
               if p.is_dir() and not p.name.startswith(".")]
    print(f"Upgrading {len(folders)} session folder(s) ...")
    for folder in folders:
        await upgrade(folder)


if __name__ == "__main__":
    asyncio.run(main())
