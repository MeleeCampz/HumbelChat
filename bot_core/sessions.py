"""Global session store with per-session notes files and next-session reminders.

Sessions are a global (bot-wide) bookkeeping concept used by the
``/start_session``, ``/end_session``, ``/remind_next_session`` and
``/session_notes`` commands:

* At most ONE session is active at a time.  State survives bot restarts via
  a JSON file (``data/sessions.json`` — same pattern as ``reminders.json``).
* Each session gets one markdown notes file inside the knowledge base
  (``<KB_PATH>/session_notes/``) so it is automatically part of RAG and can
  be edited on disk by the user at any time.
* Next-session reminders are plain persisted events: they fire when the NEXT
  session starts, so no live asyncio tasks are needed and restarts are
  inherently safe (nothing to re-arm).

Safety rules enforced by :func:`start_session`:

* a new session can only be started **max once per hour** (measured from the
  last start, regardless of whether the previous session was ended);
* while an active session is younger than 12 h, the user must end it first;
* an active session older than 12 h is considered stale and is auto-ended.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import threading
import time
from datetime import datetime

log = logging.getLogger("bot.sessions")

# ── Defaults / limits ────────────────────────────────────────────────────

#: Max one new session per this many seconds (safety rule, see module doc).
MIN_START_INTERVAL_SEC: int = 3600
#: An active session older than this is auto-ended by /start_session.
STALE_SESSION_SEC: int = 12 * 3600
#: Max length of a user-supplied session name (also used for filenames).
MAX_NAME_LEN: int = 40

# ── Disk layout ──────────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _REPO_ROOT / "data" / "sessions.json"

_store_path: pathlib.Path | None = None
_lock = threading.Lock()

_state: dict = {
    "session": None,          # the active session (None when no session is running)
    "last_ended": None,       # most recently ended session (overview pending delivery?)
    "last_start_at": None,    # epoch seconds of the most recent start
    "next_session_reminders": [],  # [{channel_id, message, created_at}]
}


# ── Path resolution ──────────────────────────────────────────────────────

def _resolve_path() -> pathlib.Path | None:
    global _store_path
    if _store_path is not None:
        return _store_path if _store_path else None
    env = os.environ.get("SESSIONS_PERSIST_FILE")
    if env is not None:
        _store_path = pathlib.Path(env) if env else None
        return _store_path if _store_path else None
    return _DEFAULT_PATH


def notes_dir() -> pathlib.Path:
    """Directory holding the per-session markdown notes files.

    Lives inside the knowledge base so session notes are automatically part
    of the RAG-enabled documents (and show up in /list_kb_docs).
    """
    from config.settings import KB_PATH
    return pathlib.Path(KB_PATH) / "session_notes"


# ── Persistence ──────────────────────────────────────────────────────────

def _save() -> None:
    path = _resolve_path()
    if path is None:
        return
    with _lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(_state, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, path)
        except Exception as e:
            log.warning("Failed to persist sessions: %s", e)


def load_persisted() -> None:
    """Load persisted session state (if any) into memory.

    Safe to call repeatedly; a corrupt file is logged and ignored.  Unlike
    reminders there is nothing to re-arm — next-session reminders are pure
    data and fire from the /start_session handler.  A pending manual
    overview survives restarts too and is delivered at the next start.
    """
    path = _resolve_path()
    if path is None or not path.exists():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _state["session"] = raw.get("session")
        _state["last_ended"] = raw.get("last_ended")
        _state["last_start_at"] = raw.get("last_start_at")
        reminders = raw.get("next_session_reminders", [])
        _state["next_session_reminders"] = reminders if isinstance(reminders, list) else []
        log.info("Loaded persisted session state (%d queued next-session reminder(s))",
                 len(_state["next_session_reminders"]))
    except Exception as e:
        log.warning("Could not load sessions file: %s", e)


# ── Session helpers ──────────────────────────────────────────────────────

def _sanitize_name(name: str | None) -> str:
    """Reduce a user-supplied session name to filename-safe characters."""
    if not name:
        return ""
    safe = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE).strip()
    safe = re.sub(r"\s+", " ", safe)
    return safe[:MAX_NAME_LEN]


def _next_session_index(now: datetime) -> int:
    """1-based per-day index for the new session (date is part of the file name)."""
    prefix = now.strftime("%Y-%m-%d")
    try:
        files = list(notes_dir().glob(f"{prefix}_*"))
    except OSError:
        return 1
    best = 0
    for f in files:
        m = re.match(rf"^{re.escape(prefix)}_(\d+)_", f.name)
        if m:
            try:
                best = max(best, int(m.group(1)))
            except ValueError:
                pass
    return best + 1


def _session_file_path(started_at: float, name: str | None, index: int) -> pathlib.Path:
    """File name always carries date + increasing index; the custom name is optional."""
    dt = datetime.fromtimestamp(started_at)
    safe = _sanitize_name(name)
    stem = f"{dt.strftime('%Y-%m-%d')}_{index:02d}" + (f"_{safe}" if safe else "")
    return notes_dir() / f"{stem}.md"


def _session_file_content(session: dict) -> str:
    """Render the session's markdown file from its state."""
    started = datetime.fromtimestamp(session["started_at"]).strftime("%Y-%m-%d %H:%M")
    ended = (datetime.fromtimestamp(session["ended_at"]).strftime("%Y-%m-%d %H:%M")
             if session.get("ended_at") else None)
    lines = [
        f"# Session: {session.get('name') or 'Untitled'}",
        "",
        f"- Started: {started}",
    ]
    if ended:
        lines.append(f"- Ended: {ended}")
    lines += ["", "## Notes", ""]
    for ts, text in session.get("notes", []):
        t = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        lines.append(f"- ({t}) {text}")
    if not session.get("notes"):
        lines.append("(no notes)")
    if session.get("overview"):
        lines += ["", "## Overview (written when the session ended)", "", str(session["overview"]).strip()]
    return "\n".join(lines) + "\n"


def _write_session_file(session: dict) -> None:
    """(Re)write the session's notes file. Failures are logged, never raised."""
    path = pathlib.Path(session.get("file") or "")
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_session_file_content(session), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:
        log.warning("Failed to write session notes file %s: %s", path, e)


def _index_session_file(path: pathlib.Path) -> None:
    """Best-effort: keep the vector index in sync so notes are RAG-searchable.

    Mirrors /upload_kb's auto-index step.  Never raises — if the embedding
    backend is down the file is still picked up on the next index load or
    via /reindex_kb.
    """
    try:
        import asyncio
        from kb.retrievers import update_kb_document

        loop = asyncio.get_running_loop()
        fut = loop.create_task(update_kb_document(path))

        def _on_done(t: "asyncio.Task") -> None:
            try:
                if not t.result():
                    log.warning("Session notes file %s not auto-indexed — run /reindex_kb.", path.name)
            except Exception as e:
                log.warning("Auto-index of session notes failed for %s: %s", path.name, e)

        fut.add_done_callback(_on_done)
    except RuntimeError:
        # No running event loop (tests, CLI) — skip indexing.
        pass
    except Exception as e:
        log.warning("Auto-index of session notes failed for %s: %s", path, e)


def _reindex_notes_file(session: dict) -> None:
    """Re-read the session file from disk and sync state + index.

    The user may have edited the markdown file on disk; treat it as the new
    source of truth (notes bullets are re-parsed, overview is kept in memory).
    """
    path = pathlib.Path(session.get("file") or "")
    if not path or not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning("Could not read session notes file %s: %s", path, e)
        return
    notes: list[list] = []
    for line in text.splitlines():
        m = re.match(r"^\s*-\s*\(([^)]+)\)\s*(.+?)\s*$", line)
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M").timestamp()
            except ValueError:
                continue
            notes.append([ts, m.group(2)])
    session["notes"] = notes
    _index_session_file(path)


# ── Public API — session lifecycle ───────────────────────────────────────

def get_current_session() -> dict | None:
    """The current session if one is active, else None."""
    s = _state.get("session")
    if s and not s.get("ended_at"):
        return s
    return None


def get_last_session() -> dict | None:
    """Most recent session (active or the last ended one) — for /session_notes view."""
    return _state.get("session") or _state.get("last_ended")


def _pending_manual_overview() -> dict | None:
    """Return the last-ENDED session whose AI overview has not been delivered yet.

    The overview of a manually ended session is delivered when the NEXT
    session starts (per spec).  Once delivered, ``overview_delivered`` is set
    so it is never sent twice.
    """
    s = _state.get("last_ended")
    if s and not s.get("overview_delivered") and s.get("overview"):
        return s
    return None


def start_session(name: str | None = None) -> tuple[dict, dict | None]:
    """Start a new global session.

    Enforces the safety rules (see module docstring).  Returns
    ``(new_session, closed_info)`` where ``closed_info`` is either
    ``{"kind": "manual", "session": s}`` for a previously ended session whose
    AI overview should now be delivered, or
    ``{"kind": "stale", "session": s}`` for a stale session that was just
    auto-ended without an AI overview — or None when no previous session
    existed.

    Raises:
        ValueError: with a user-facing message when the start is refused;
            state is left unchanged.
    """
    now = time.time()
    last_start = _state.get("last_start_at") or 0
    current = get_current_session()

    if last_start and (now - last_start) < MIN_START_INTERVAL_SEC:
        wait_min = max(1, int((MIN_START_INTERVAL_SEC - (now - last_start)) / 60))
        msg = (f"A new session can only be started once per hour — please try again in "
               f"about {wait_min} minute(s).")
        log.info("start_session refused: %s", msg)
        raise ValueError(msg)

    closed_info: dict | None = None
    if current is not None:
        age = now - current["started_at"]
        if age >= STALE_SESSION_SEC:
            # Stale session — end it without an AI overview and start fresh.
            closed = _end_session_internal(current, overview=None)
            closed_info = {"kind": "stale", "session": closed}
            log.info("Auto-ended stale session %s (age %.1f h)", current.get("name"), age / 3600)
        else:
            msg = ("A session is already active — end it first with `/end_session` "
                   "(or start again once it is older than 12 h to auto-end it).")
            log.info("start_session refused: session %s still active (age %.1f h)",
                     current.get("name"), age / 3600)
            raise ValueError(msg)
    else:
        # No active session — but a previously ended one may have an
        # overview that is due for delivery now.
        pending = _pending_manual_overview()
        if pending is not None:
            closed_info = {"kind": "manual", "session": pending}
            pending["overview_delivered"] = True

    dt = datetime.fromtimestamp(now)
    safe_name = _sanitize_name(name)
    session = {
        "id": dt.strftime("%Y%m%d%H%M%S"),
        "name": safe_name,
        "started_at": now,
        "ended_at": None,
        "notes": [],          # [[epoch, text], ...]
        "overview": None,     # AI overview written on end (or None)
        "file": str(_session_file_path(now, safe_name, _next_session_index(dt))),
    }
    _state["session"] = session
    _state["last_start_at"] = now
    _write_session_file(session)
    _index_session_file(pathlib.Path(session["file"]))
    _save()
    log.info("Session started: %s (file: %s)", safe_name or "(untitled)", session["file"])
    return session, closed_info


def _end_session_internal(session: dict, overview: str | None) -> dict:
    """Shared end logic — sets ended_at, moves to last_ended, persists."""
    session["ended_at"] = time.time()
    if overview is not None:
        session["overview"] = overview
        session["overview_delivered"] = False  # delivered at the NEXT start
    else:
        session["overview_delivered"] = True   # nothing to deliver (stale end)
    _state["last_ended"] = session
    _state["session"] = None
    _write_session_file(session)
    _index_session_file(pathlib.Path(session.get("file", "")))
    _save()
    return session


def end_session(overview: str | None = None, name: str | None = None) -> dict | None:
    """End the current session.

    *name* (optional) renames the session in its file/state; *overview* is
    stored in the session file and returned to the caller for delivery.
    Returns the ended session dict, or None when no session was active.
    """
    session = get_current_session()
    if session is None:
        return None
    if name:
        safe = _sanitize_name(name)
        if safe:
            session["name"] = safe
    return _end_session_internal(session, overview=overview)


# ── Public API — notes ───────────────────────────────────────────────────

def add_note(text: str, author: str = "") -> dict | None:
    """Append a timestamped note to the current session.

    Returns the updated session, or None when no session is active.  The
    notes file is rewritten and re-indexed for RAG.
    """
    session = get_current_session()
    if session is None:
        return None
    clean = " ".join(str(text).split())
    if not clean:
        return None
    text = f"{clean} (by {author})" if author else clean
    entry = [time.time(), text]
    session.setdefault("notes", []).append(entry)
    _write_session_file(session)
    _index_session_file(pathlib.Path(session["file"]))
    _save()
    log.info("Session note added to %s: %.60s", session.get("name"), clean)
    return session


def get_notes(session: dict | None = None) -> list[list]:
    """Notes of *session* (default: current, else last known)."""
    s = session if session is not None else _state.get("session")
    if not s:
        return []
    return [list(n) for n in s.get("notes", [])]


def refresh_notes_from_disk(session: dict | None = None) -> list[list]:
    """Re-read the notes file from disk (user may have edited it) and sync state."""
    s = session if session is not None else _state.get("session")
    if not s:
        return []
    _reindex_notes_file(s)
    _save()
    return [list(n) for n in s.get("notes", [])]


# ── Public API — next-session reminders ──────────────────────────────────

def queue_next_session_reminder(channel_id: int, message: str) -> dict:
    """Queue a reminder to be delivered when the NEXT session starts.

    Persists immediately; fires from :func:`deliver_queued_reminders` inside
    the /start_session handler (survives restarts without any re-arming).
    """
    clean = " ".join(str(message).split())
    entry = {
        "channel_id": int(channel_id),
        "message": clean,
        "created_at": time.time(),
    }
    _state.setdefault("next_session_reminders", []).append(entry)
    _save()
    log.info("Queued next-session reminder in channel %s: %.60s (%d queued)",
             channel_id, clean, len(_state["next_session_reminders"]))
    return entry


def list_queued_reminders() -> list[dict]:
    return [dict(r) for r in _state.get("next_session_reminders", [])]


async def deliver_queued_reminders(bot) -> int:
    """Send all queued next-session reminders to their channels.

    Called by the /start_session handler right after a new session started.
    Returns the number of reminders that were sent (and removed from the
    queue).  Unresolvable channels are dropped with a warning.
    """
    queued = list(_state.get("next_session_reminders", []))
    if not queued:
        return 0
    sent = 0
    delivered: list[dict] = []
    for r in queued:
        try:
            chan = bot.get_channel(r["channel_id"])
            if chan is None:
                log.warning("Next-session reminder channel %s not found — dropping",
                            r["channel_id"])
                continue
            await chan.send(f"⏰ **Next-session reminder:** {r['message']}")
            sent += 1
            delivered.append(r)
        except Exception as e:
            log.error("Failed to deliver next-session reminder in channel %s: %s",
                      r["channel_id"], e)
    if delivered:
        # Remove exactly the reminders that were delivered (by identity),
        # so failed/unresolvable ones stay queued for the next start.
        keys = {(d["channel_id"], d["message"], d.get("created_at")) for d in delivered}
        _state["next_session_reminders"] = [
            r for r in _state.get("next_session_reminders", [])
            if (r["channel_id"], r["message"], r.get("created_at")) not in keys
        ]
        _save()
    return sent


def cancel_queued_reminder(index: int) -> bool:
    """Remove a queued reminder by position (0-based). Returns True if removed."""
    q = _state.get("next_session_reminders", [])
    if 0 <= index < len(q):
        q.pop(index)
        _save()
        return True
    return False
