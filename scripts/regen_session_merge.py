"""One-off: regenerate an ended session's AI-merged log with the CURRENT prompt.

Use after a SESSION_MERGE_PROMPT change to re-process already-stored session
logs (e.g. #74, where the old preservation-only prompt made the model echo a
single uploaded document back almost verbatim).

The bot OWNS ``data/sessions.json`` while it runs — stop it first, then run
this inside the bot container so ``.env``/config resolve exactly as in
production:

    docker compose stop bot
    docker compose run --rm --entrypoint python bot \
        scripts/regen_session_merge.py "To the roots"
    docker compose start bot

With no argument it lists what it can regenerate.  Two target kinds:

* a session still in live state (active or most-recently-ended) — the stored
  ``merged_log`` is replaced, ``notes.md`` re-rendered from state, saved and
  re-indexed;
* an OLDER ended session that only exists as a folder on disk — its
  ``## Session Log (… AI-merged)`` section in ``notes.md`` is rewritten in
  place (state untouched).

The target must be ENDED either way — an active session's log is written by
/end_session.  The matching needle is matched against the session name AND
the folder name (older untitled sessions are only reachable by folder name).
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import re
import sys
from typing import Any

# Allow running both from the repo root (dev) and /app (container).
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_MERGED_HEADER = "## Session Log (combined from all uploads — AI-merged)"


def _state_candidates(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for key in ("session", "last_ended"):
        s = state.get(key)
        if isinstance(s, dict) and s:
            out.append((key, s))
    return out


def _folder_title(notes_path: pathlib.Path) -> str:
    try:
        m = re.search(r"^# Session: (.+)$", notes_path.read_text(encoding="utf-8"), re.M)
        return (m.group(1).strip() if m else "")
    except OSError:
        return ""


def _find_folder_session(needle: str) -> pathlib.Path | None:
    """Session folder on disk whose folder name or notes title matches *needle*."""
    from bot_core import sessions as S
    needle = needle.lower()
    root = pathlib.Path(S.notes_dir())
    hits = [
        d for d in sorted(root.iterdir())
        if d.is_dir() and (d / "notes.md").exists()
        and (needle in d.name.lower() or needle in _folder_title(d / "notes.md").lower())
    ]
    return hits[0] if len(hits) == 1 else None


async def _regenerate(session: dict[str, Any], from_state: bool) -> int:
    from bot_core import sessions as S
    from commands.session_commands import _generate_merged_log
    from kb.retrievers import update_kb_document

    name = session.get("name") or "(untitled)"
    print(f"Regenerating merged log for {name!r} ({'live state' if from_state else 'folder only'}) …")
    merged = await _generate_merged_log(session, None, None)
    if not merged:
        print("Merge failed (AI unavailable?) — nothing changed.")
        return 1

    notes_path = pathlib.Path(session["file"])
    if from_state:
        session["merged_log"] = merged
        S._write_session_file(session)
        S._save()
    else:
        # Surgical rewrite: replace the (always-last) merged section in place —
        # a full re-render would drop the stored overview, which only lives in
        # notes.md for sessions no longer in state.
        text = notes_path.read_text(encoding="utf-8")
        i = text.find(_MERGED_HEADER)
        new = (text[:i] if i != -1 else text.rstrip() + "\n\n") + _MERGED_HEADER + f"\n\n{merged}\n"
        tmp = notes_path.with_suffix(".tmp")
        tmp.write_text(new, encoding="utf-8")
        tmp.replace(notes_path)

    for p in S._session_index_paths(session):
        if pathlib.Path(p).exists():
            await update_kb_document(p)
    print(f"Done: {len(merged)} chars stored, {notes_path.name} re-written + re-indexed.")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    needle = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()

    from bot_core import sessions as S
    S.load_persisted()

    if not needle:
        for key, s in _state_candidates(S._state):
            status = "active" if s is S._state.get("session") else "ended"
            print(f"state[{key}]: {s.get('name') or '(untitled)'} [{status}]"
                  f"{' — has stored merged log' if s.get('merged_log') else ''}")
        root = pathlib.Path(S.notes_dir())
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            if d.is_dir() and (d / "notes.md").exists():
                print(f"folder: {d.name} — “{_folder_title(d / 'notes.md') or '(untitled)'}”")
        return 0

    # 1) live state first …
    hits = [s for _k, s in _state_candidates(S._state)
            if needle in str(s.get("name", "")).lower()]
    if len(hits) == 1:
        session = hits[0]
        if not session.get("ended_at"):
            print(f"Refusing: {session.get('name')!r} is still ACTIVE — end it first "
                  f"(/end_session writes the log with the current prompt).")
            return 1
        return asyncio.run(_regenerate(session, from_state=True))

    # … then an older folder-only session.
    folder = _find_folder_session(needle)
    if folder is None:
        print(f"No UNIQUE ended session matching {needle!r} (state or folders).")
        return 1
    session = {
        "name": _folder_title(folder / "notes.md") or folder.name,
        "dir": str(folder),
        "file": str(folder / "notes.md"),
    }
    return asyncio.run(_regenerate(session, from_state=False))


if __name__ == "__main__":
    raise SystemExit(main())
