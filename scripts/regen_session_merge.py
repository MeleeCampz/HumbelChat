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

With no argument it lists the regeneratable sessions (current state only holds
the active session and the most recently ended one).  The target must be
ENDED — an active session's log is written by /end_session.
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import sys
from typing import Any

# Allow running both from the repo root (dev) and /app (container).
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _candidates(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for key in ("session", "last_ended"):
        s = state.get(key)
        if isinstance(s, dict) and s:
            out.append((key, s))
    return out


async def _regenerate(session: dict[str, Any]) -> int:
    from bot_core import sessions as S
    from commands.session_commands import _generate_merged_log
    from kb.retrievers import update_kb_document

    name = session.get("name") or "(untitled)"
    print(f"Regenerating merged log for {name!r} …")
    merged = await _generate_merged_log(session, None, None)
    if not merged:
        print("Merge failed (AI unavailable?) — state untouched.")
        return 1

    session["merged_log"] = merged
    S._write_session_file(session)
    for p in S._session_index_paths(session):
        if pathlib.Path(p).exists():
            await update_kb_document(p)
    S._save()
    print(f"Done: {len(merged)} chars stored, notes.md re-rendered + re-indexed.")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    needle = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()

    from bot_core import sessions as S
    S.load_persisted()

    hits = [
        (key, s) for key, s in _candidates(S._state)
        if not needle or needle in str(s.get("name", "")).lower()
    ]
    if not needle:
        for key, s in _candidates(S._state):
            status = "active" if s is S._state.get("session") else "ended"
            print(f"{key}: {s.get('name') or '(untitled)'} [{status}]"
                  f"{' — has stored merged log' if s.get('merged_log') else ''}")
        return 0

    if len(hits) != 1:
        print(f"No UNIQUE ended session matching {needle!r} "
              f"({len(hits)} hit(s) in current state).")
        return 1
    _key, session = hits[0]
    if not session.get("ended_at"):
        print(f"Refusing: {session.get('name')!r} is still ACTIVE — end it first "
              f"(/end_session writes the log with the current prompt).")
        return 1
    return asyncio.run(_regenerate(session))


if __name__ == "__main__":
    raise SystemExit(main())
