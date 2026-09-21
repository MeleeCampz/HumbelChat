"""Global session store with per-session folders and next-session reminders.

Sessions are a global (bot-wide) bookkeeping concept used by the
``/start_session``, ``/end_session``, ``/remind_next_session`` and
``/session_notes`` commands:

* At most ONE session is active at a time.  State survives bot restarts via
  a JSON file (``data/sessions.json`` — same pattern as ``reminders.json``).
* Each session lives in its own folder inside the knowledge base
  (``<KB_PATH>/session_notes/<date>_<index>[_<name>]/``) so it is
  automatically part of RAG and can be edited on disk at any time:

      notes.md            -- markdown rendering of the session (notes +
                             pointers to attachments/transcripts)
      attachments/        -- one file per uploaded .txt/.md document
      transcripts/        -- one file per voice-channel transcript

  ``notes.md`` is the only file whose bullets are re-parsed into in-memory
  state (see :func:`refresh_notes_from_disk`); attachments and transcripts
  are standalone documents the KB indexer chunks on their own.

* Next-session reminders are plain persisted events: they fire when the NEXT
  session starts, so no live asyncio tasks are needed and restarts are
  inherently safe (nothing to re-arm).

Safety rules enforced by :func:`start_session`:

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

#: An active session older than this is auto-ended by /start_session.
STALE_SESSION_SEC: int = 12 * 3600
#: Max length of a user-supplied session name (also used for filenames).
MAX_NAME_LEN: int = 40
#: Session sub-folders that hold standalone documents (indexed by the KB).
_SUBDIR_ATTACHMENTS = "attachments"
_SUBDIR_TRANSCRIPTS = "transcripts"

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
    """Directory holding the per-session folders.

    Lives inside the knowledge base so session notes, attachments and
    transcripts are automatically part of the RAG-enabled documents (and
    show up in /list_kb_docs).
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


def _sanitize_stem(name: str | None) -> str:
    """Reduce an arbitrary title/filename to a filesystem-safe stem (no ext)."""
    if not name:
        return ""
    # Keep the extension-free basename safe: alphanumerics, underscore, dash.
    safe = re.sub(r"[^\w\-]+", "_", str(name).strip(), flags=re.UNICODE)
    safe = re.sub(r"\s+", "_", safe)
    safe = safe.strip("_").replace("..", "_")
    return safe[:MAX_NAME_LEN]


def _next_session_index(now: datetime) -> int:
    """1-based per-day index for the new session (date is part of the folder name)."""
    prefix = now.strftime("%Y-%m-%d")
    try:
        folders = [p for p in notes_dir().glob(f"{prefix}_*") if p.is_dir()]
    except OSError:
        return 1
    best = 0
    for f in folders:
        m = re.match(rf"^{re.escape(prefix)}_(\d+)_", f.name)
        if m:
            try:
                best = max(best, int(m.group(1)))
            except ValueError:
                pass
    return best + 1


def _session_dir(session: dict) -> pathlib.Path:
    """The per-session folder (absolute). Prefer the stored ``dir`` key; fall
    back to the notes file's parent so partially-built sessions still resolve."""
    d = session.get("dir")
    if d:
        p = pathlib.Path(d)
        if p.exists() or p.parent.exists():
            return p
    file = session.get("file")
    if file:
        return pathlib.Path(file).parent
    return notes_dir()


def _session_file_path(started_at: float, name: str | None, index: int) -> pathlib.Path:
    """Each session is a folder: ``<date>_<index>[_<name>]/notes.md``."""
    dt = datetime.fromtimestamp(started_at)
    safe = _sanitize_name(name)
    stem = f"{dt.strftime('%Y-%m-%d')}_{index:02d}" + (f"_{safe}" if safe else "")
    return notes_dir() / stem / "notes.md"


def _unique_path(directory: pathlib.Path, stem: str, ext: str) -> pathlib.Path:
    """Return ``directory/<stem><ext>`` or ``<stem>_<n><ext>`` if taken."""
    base = directory / f"{stem}{ext}"
    if not base.exists():
        return base
    n = 2
    while (directory / f"{stem}_{n}{ext}").exists():
        n += 1
    return directory / f"{stem}_{n}{ext}"


def _next_transcript_name(directory: pathlib.Path) -> str:
    """Next sequential transcript file name in *directory*.

    Sequential (``transcript_01.md``, ``transcript_02.md", …) so the name is
    stable, deterministic and reflects the transcript's position in the
    session — not the (potentially misleading) completion timestamp.
    """
    best = 0
    try:
        for f in directory.iterdir():
            m = re.match(r"^transcript_(\d+)\.md$", f.name)
            if m:
                best = max(best, int(m.group(1)))
    except OSError:
        pass
    return f"transcript_{best + 1:02d}.md"


def _session_docs(session: dict, subdir: str) -> list[str]:
    """Names of standalone documents currently in one of the session's sub-folders."""
    d = _session_dir(session) / subdir
    if not d.exists():
        return []
    try:
        return sorted(
            f.name for f in d.iterdir()
            if f.is_file() and not f.name.startswith(".")
            and f.suffix.lower() in {".md", ".txt"}
        )
    except OSError:
        return []


def _doc_pointer_lines(session: dict, subdir: str) -> list[str]:
    """Display-only pointer lines for a session's standalone documents.

    No ``(<ts>)`` prefix, so :func:`_reindex_notes_file` ignores them — they
    live in the notes file purely so ``/session_notes`` can list them; the
    full text is in the standalone file.
    """
    return [f"- {name}" for name in _session_docs(session, subdir)]


def _session_file_content(session: dict) -> str:
    """Render the session's ``notes.md`` from its state.

    Real notes are timestamped bullets; attachments and transcripts are
    rendered as pointer lines (no ``(<ts>)`` prefix) so
    :func:`_reindex_notes_file` keeps them out of the parsed note list while
    they still show up in ``/session_notes``.
    """
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
    for subdir, heading in ((_SUBDIR_ATTACHMENTS, "## Attachments"),
                            (_SUBDIR_TRANSCRIPTS, "## Transcripts")):
        pointers = _doc_pointer_lines(session, subdir)
        if pointers:
            lines += ["", heading, ""] + pointers
    if session.get("overview"):
        lines += ["", "## Overview (written when the session ended)", "", str(session["overview"]).strip()]
    return "\n".join(lines) + "\n"


def _write_session_file(session: dict) -> None:
    """(Re)write the session's ``notes.md``. Failures are logged, never raised."""
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
    """Best-effort: keep the vector index in sync so a file is RAG-searchable.

    Mirrors /upload_kb's auto-index step.  Never raises — if the embedding
    backend is down the file is still picked up on the next index load or
    via /reindex_kb.
    """
    if not path.exists():
        return
    try:
        import asyncio
        from kb.retrievers import update_kb_document

        loop = asyncio.get_running_loop()
        fut = loop.create_task(update_kb_document(path))

        def _on_done(t: "asyncio.Task") -> None:
            try:
                if not t.result():
                    log.warning("Session file %s not auto-indexed — run /reindex_kb.", path.name)
            except Exception as e:
                log.warning("Auto-index of session file failed for %s: %s", path.name, e)

        fut.add_done_callback(_on_done)
    except RuntimeError:
        # No running event loop (tests, CLI) — skip indexing.
        pass
    except Exception as e:
        log.warning("Auto-index of session file failed for %s: %s", path, e)


def _session_index_paths(session: dict) -> list[pathlib.Path]:
    """All indexable files for a session: notes.md + attachments + transcripts."""
    paths: list[pathlib.Path] = []
    f = session.get("file")
    if f and pathlib.Path(f).exists():
        paths.append(pathlib.Path(f))
    for subdir in (_SUBDIR_ATTACHMENTS, _SUBDIR_TRANSCRIPTS):
        for name in _session_docs(session, subdir):
            paths.append(_session_dir(session) / subdir / name)
    return paths


def _index_session_paths(session: dict) -> None:
    for p in _session_index_paths(session):
        _index_session_file(p)


def _reindex_notes_file(session: dict) -> None:
    """Re-read the session's ``notes.md`` from disk and sync state + index.

    The user may have edited the markdown file on disk; treat it as the new
    source of truth.  Only real ``(<timestamp>)`` note bullets are re-parsed
    into state; attachment/transcript pointer lines (no timestamp) are left
    for display only.  The overview stays in memory.
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
    _index_session_paths(session)


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
    current = get_current_session()

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
    file_path = _session_file_path(now, safe_name, _next_session_index(dt))
    session = {
        "id": dt.strftime("%Y%m%d%H%M%S"),
        "name": safe_name,
        "started_at": now,
        "ended_at": None,
        "notes": [],          # [[epoch, text], ...] — full, un-split text
        "overview": None,     # AI overview written on end (or None)
        "dir": str(file_path.parent),
        "file": str(file_path),
    }
    _state["session"] = session
    _state["last_start_at"] = now
    _write_session_file(session)
    _index_session_paths(session)
    _save()
    log.info("Session started: %s (dir: %s)", safe_name or "(untitled)", session["dir"])
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
    _index_session_paths(session)
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

    The note is stored whole (whitespace-normalized to a single line — the
    shape the notes file uses, so it stays re-parseable).  It is rendered and
    split into Discord-sized messages only at *display* time by
    ``/session_notes`` (see ``session_commands``); it is never pre-chunked,
    so the notes file and the KB chunker see one coherent entry.

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
    session.setdefault("notes", []).append([time.time(), text])
    _write_session_file(session)
    _index_session_paths(session)
    _save()
    log.info("Session note added to %s: %.60s", session.get("name"), clean)
    return session


def _append_document_file(session: dict, subdir: str, title: str, text: str) -> pathlib.Path:
    """Write *text* as one standalone ``.md`` file inside a session sub-folder.

    Returns the written path.  The file keeps the raw document text (a
    markdown header naming the source is prepended for context); the KB
    indexer chunks it on its own.
    """
    directory = _session_dir(session) / subdir
    directory.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_stem(os.path.splitext(title or "")[0]) or "document"
    path = _unique_path(directory, stem, ".md")
    header = (title or stem).strip()
    body = f"# {header}\n\n{text.rstrip()}\n"
    try:
        path.write_text(body, encoding="utf-8")
    except OSError as e:
        log.warning("Failed to write session document %s: %s", path, e)
    return path


def add_transcript(text: str, title: str = "", session: dict | None = None) -> tuple[dict | None, int]:
    """Store a finished voice-channel transcript for a session.

    Called from the STT background job after ``/stop_recording``.  The full
    transcript is written as its own ``.md`` file under the session's
    ``transcripts/`` folder so the KB indexer chunks it by header/paragraph —
    no more 1500-char bullets, and long transcripts stay cleanly searchable.
    A short pointer line is added to the notes so the transcript shows up in
    ``/session_notes`` and links back to the file.

    *session* pins the target — normally the session that was active when the
    recording stopped, which may have ended by the time transcription
    finishes.  When omitted, the currently active session is used.

    Returns ``(session, n_files)`` where ``n_files`` is 1 on success and 0
    when the transcript is empty or no session is available.  Never raises —
    STT delivery must not be blocked by note bookkeeping.
    """
    try:
        if session is None:
            session = get_current_session()
        if session is None:
            return None, 0
        clean = str(text).strip()
        if not clean:
            return None, 0

        header = (title.strip() or "Voice channel transcript")
        directory = _session_dir(session) / _SUBDIR_TRANSCRIPTS
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / _next_transcript_name(directory)
        path.write_text(f"# {header}\n\n{clean}\n", encoding="utf-8")

        _write_session_file(session)
        _index_session_paths(session)
        _save()
        log.info("Transcript saved for %s: %s (%d chars)",
                 session.get("name"), path.name, len(clean))
        return session, 1
    except Exception as e:  # pragma: no cover - defensive
        log.warning("Could not add transcript to session: %s", e)
        return None, 0


def add_document(text: str, title: str = "", session: dict | None = None) -> tuple[dict | None, int]:
    """Store an uploaded text document (``.txt`` / ``.md``) for a session.

    The whole file is written as its own ``.md`` under the session's
    ``attachments/`` folder (one file per upload, so each gets its own
    semantic KB chunks — no pre-splitting).  A short pointer line is added to
    the notes so the document shows up in ``/session_notes`` and links back to
    the file.

    *session* pins the target (defaults to the current one).  Returns
    ``(session, n_files)`` where ``n_files`` is 1 on success and 0 for
    empty/whitespace-only input or when no session is available.  Never
    raises — storing a document must not be blocked by note bookkeeping.
    """
    try:
        if session is None:
            session = get_current_session()
        if session is None:
            return None, 0
        clean = str(text).strip()
        if not clean:
            return None, 0

        header = (title.strip() or "Uploaded document")
        path = _append_document_file(session, _SUBDIR_ATTACHMENTS, title or "document", clean)

        _write_session_file(session)
        _index_session_paths(session)
        _save()
        log.info("Document saved for %s: %s (%d chars)",
                 session.get("name"), path.name, len(clean))
        return session, 1
    except Exception as e:  # pragma: no cover - defensive
        log.warning("Could not add document to session: %s", e)
        return None, 0


def get_notes(session: dict | None = None) -> list[list]:
    """Notes of *session* (default: current, else last known).

    Each entry is ``[epoch, text]`` where *text* is the full (un-split) note
    or an attachment/transcript pointer line.
    """
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
    queue).

    Delivery uses :func:`bot_core.channel_delivery.send_to_channel` (cache,
    then REST fallback).  Reminders whose channel cannot be reached are NOT
    silently dropped: they stay queued with an attempt counter and are
    retried at the next /start_session — until they fail
    ``MAX_DELIVERY_ATTEMPTS`` times, after which they are dropped with a
    loud error log.
    """
    from bot_core.channel_delivery import ChannelNotDeliverableError, send_to_channel
    from bot_core.reminders import MAX_DELIVERY_ATTEMPTS

    queued = list(_state.get("next_session_reminders", []))
    if not queued:
        return 0
    sent = 0
    delivered: list[dict] = []
    failed: list[dict] = []
    for r in queued:
        try:
            # send_to_channel uses the local cache first, then falls back to a
            # REST fetch — so channels missing from the gateway cache (e.g. a
            # private channel with View Channel denied on @everyone) are still
            # attempted instead of being silently skipped.
            await send_to_channel(
                bot, r["channel_id"],
                f"⏰ **Next-session reminder:** {r['message']}",
            )
            sent += 1
            delivered.append(r)
        except ChannelNotDeliverableError as e:
            log.error(
                "Could not deliver next-session reminder to channel %s: %s — "
                "leaving it queued; fix the channel permissions and it will be "
                "retried at the next /start_session.",
                r["channel_id"], e.reason)
            failed.append(r)
        except Exception as e:
            log.error("Failed to deliver next-session reminder in channel %s: %s",
                      r["channel_id"], e)

    if failed:
        # Count attempts per entry; drop ones that keep failing so a
        # permanently undeliverable reminder cannot retry forever.
        q = _state.get("next_session_reminders", [])
        drop_ids: set[int] = set()
        for r in failed:
            key = (r["channel_id"], r["message"], r.get("created_at"))
            entry = next((x for x in q if (x["channel_id"], x["message"],
                                           x.get("created_at")) == key), None)
            if entry is None:
                continue
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            if entry["attempts"] >= MAX_DELIVERY_ATTEMPTS:
                log.error(
                    "DROPPED next-session reminder after %d failed attempts — "
                    "channel: %s, message: %.60s. Re-issue it in a channel the "
                    "bot can post to.", entry["attempts"], r["channel_id"],
                    r["message"])
                drop_ids.add(id(entry))
        if drop_ids:
            _state["next_session_reminders"] = [x for x in q if id(x) not in drop_ids]
    if delivered:
        # Remove exactly the reminders that were delivered (by identity),
        # so failed/unresolvable ones stay queued for the next start.
        keys = {(d["channel_id"], d["message"], d.get("created_at")) for d in delivered}
        _state["next_session_reminders"] = [
            r for r in _state.get("next_session_reminders", [])
            if (r["channel_id"], r["message"], r.get("created_at")) not in keys
        ]
    if delivered or failed:
        # Persist: delivered entries are removed above; failed ones keep their
        # incremented attempt counter so retries are bounded.
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
