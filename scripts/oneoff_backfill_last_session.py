"""One-off: register the newest legacy session folder as ``last_ended``.

Background
----------
The three legacy session folders (DerAnfang, Die_zweite_Session,
Auf_Nach_Alderheart) predate session-state persistence: they exist on disk
under ``data/knowledge/session_notes/`` but ``data/sessions.json`` has
``last_ended: null``, so the last-session continuity context
(``_build_last_session_context``) never fires for them.

This script picks the newest folder, extracts the ``## Overview`` and the
AI-merged ``## Session Log`` sections from its ``notes.md``, and writes a
faithful ``last_ended`` entry into ``sessions.json``:

* the merged log is split on top-level ``## `` sections into individual
  note entries (oldest → newest) so the context budget drops OLDEST
  sections first with an omission marker, instead of dropping one giant
  entry wholesale;
* ``overview_delivered`` is set to True so the next ``/start_session``
  does not post the stale overview into Discord.

A timestamped backup of the previous ``sessions.json`` is written next to
it.  Run at most once:

    .venv/Scripts/python.exe scripts/oneoff_backfill_last_session.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SESSIONS_JSON = REPO / "data" / "sessions.json"
NOTES_ROOT = REPO / "data" / "knowledge" / "session_notes"


def _section(text: str, heading_prefix: str, to_eof: bool = False) -> str:
    """Return the body of the first ``## <heading_prefix>...`` section.

    ``to_eof=True`` takes everything until end of file — needed for the
    Session Log, whose own content uses ``##`` scene headings (the
    renderer writes it last, so EOF is its true boundary).
    """
    m = re.search(
        r"^##\s+" + re.escape(heading_prefix) + r".*?$", text,
        flags=re.MULTILINE,
    )
    if not m:
        return ""
    rest = text[m.end():]
    if to_eof:
        return rest.strip()
    nxt = re.search(r"^##\s+", rest, flags=re.MULTILINE)
    body = rest[: nxt.start()] if nxt else rest
    return body.strip()


def main() -> int:
    folders = sorted(
        p for p in NOTES_ROOT.iterdir()
        if p.is_dir() and (p / "notes.md").is_file()
    )
    if not folders:
        print("no session folders found — nothing to do")
        return 0
    newest = folders[-1]
    text = (newest / "notes.md").read_text(encoding="utf-8")

    # Overview heading is language-dependent ("Overview" / "Übersicht").
    overview = _section(text, "Overview") or _section(text, "Übersicht")
    merged = _section(text, "Session Log", to_eof=True)
    if not overview and not merged:
        print(f"no Overview/Session Log sections in {newest.name}/notes.md")
        return 1

    # Split the merged log into per-section note entries (oldest first).
    parts = re.split(r"(?m)^(?=##\s)", merged)
    note_bodies = [p.strip() for p in parts if p.strip()]
    if not note_bodies and merged:
        note_bodies = [merged]

    m = re.match(r"(\d{4}-\d{2}-\d{2})_\d+_(.+)", newest.name)
    name = m.group(2) if m else newest.name
    day = datetime.strptime(m.group(1), "%Y-%m-%d") if m else datetime.now()
    ts = day.replace(hour=12).timestamp()  # date-only folders: noon local

    state = json.loads(SESSIONS_JSON.read_text(encoding="utf-8"))
    if state.get("last_ended"):
        print("sessions.json already has a last_ended entry — aborting (no clobber)")
        return 1

    backup = SESSIONS_JSON.with_name(f"sessions.json.bak-{int(time.time())}")
    shutil.copy2(SESSIONS_JSON, backup)

    state["last_ended"] = {
        "name": name,
        "started_at": None,
        "ended_at": ts,
        "overview": overview,
        "overview_delivered": True,
        "merged_log": merged,
        "notes": [[ts, body] for body in note_bodies],
    }
    SESSIONS_JSON.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"backfilled last_ended from {newest.name}")
    print(f"  name={name!r} overview={len(overview)} chars "
          f"merged_log={len(merged)} chars note_entries={len(note_bodies)}")
    print(f"  backup: {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
