"""Global session store with per-session folders and next-session reminders.

Sessions are a global (bot-wide) bookkeeping concept used by the
``/start_session``, ``/end_session``, ``/remind_next_session`` and
``/session_notes`` commands:

* At most ONE session is active at a time.  State survives bot restarts via
  a JSON file (``data/sessions.json`` — same pattern as ``reminders.json``).
* Each session lives in its own folder inside the knowledge base
  (``<KB_PATH>/session_notes/SESSION_<NN>[_<name>]/`` — a globally unique
  running number, NO date in the folder name; the date lives in the file
  header) and can be edited on disk at any time:

      notes.md            -- the ONLY RAG-indexed file for the session:
                             header (name + started/ended dates + a pointer
                             to the notes file) + brief end-of-session
                             overview (when present) + the canonical
                             ``## Session Log`` or a combined
                             ``## Documents`` section with the full text of
                             every uploaded document / transcript
      .attachments/       -- hidden dot-dir (the KB indexer skips dot-dirs):
          notes.md        -- the session's timestamped notes live HERE, as
                             an attachment file, not inside notes.md
          <upload>.md     -- raw uploaded .txt/.md files (one per upload)
      .transcripts/       -- raw voice-channel transcripts; likewise hidden

  Uploading a document or finishing a transcript writes the raw file into
  the hidden dot-dir and re-renders ``notes.md`` with the document's full
  text combined in — one session = one RAG document, so near-duplicate
  uploads can no longer be attached multiple times by retrieval.  The notes
  attachment participates in that combined ``## Documents`` fallback (when
  no AI-merged log exists), so its content stays RAG-reachable either way.

  The notes attachment is the only file whose bullets are re-parsed into
  in-memory state (see :func:`refresh_notes_from_disk`); ``notes.md`` is
  only parsed as a fallback for pre-migration (legacy) layouts.

* Next-session reminders are plain persisted events: they fire when the NEXT
  session starts, so no live asyncio tasks are needed and restarts are
  inherently safe (nothing to re-arm).

Safety rules enforced by :func:`start_session`:

* while an active session is younger than 12 h, the user must end it first;
* an active session older than 12 h is considered stale and is auto-ended.

Crash recovery (#28): if the bot stopped (or crashed) while a session was
still active, :func:`load_persisted` resumes that session at startup —
validates it, recreates its notes file when missing, re-reads user edits
from disk and logs loudly so the session is *continued* instead of silently
lost or blocking ``/start_session`` with an unexplained refusal.
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

from utils.timefmt import fmt_local, parse_local
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import discord

log = logging.getLogger("bot.sessions")

# ── Defaults / limits ────────────────────────────────────────────────────

#: An active session older than this is auto-ended by /start_session.
STALE_SESSION_SEC: int = 12 * 3600
#: Max length of a user-supplied session name (also used for filenames).
MAX_NAME_LEN: int = 40
#: Hidden session sub-folders holding raw documents. The dot prefix keeps
#: them out of the KB indexer (``kb.index._is_hidden_kb_path`` skips any
#: dot-dir) — their content reaches RAG only via the combined ``notes.md``.
_SUBDIR_ATTACHMENTS = ".attachments"
_SUBDIR_TRANSCRIPTS = ".transcripts"
#: Session folder names are ``SESSION_<NN>[_<name>]`` — the date is NOT part
#: of the name (it lives in the notes.md header); NN is a globally unique
#: running index (never resets per day).
_SESSION_PREFIX = "SESSION_"
#: Reserved attachment file holding the session's timestamped notes
#: (``.attachments/notes.md``) instead of a ``## Notes`` section inside the
#: session's main ``notes.md``.
_NOTES_FILE = "notes.md"
#: One timestamped note bullet: ``- (<timestamp>) <text>``.
_NOTE_BULLET_RE = re.compile(r"^\s*-\s*\(([^)]+)\)\s*(.+?)\s*$")

# ── Disk layout ──────────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _REPO_ROOT / "data" / "sessions.json"

_store_path: pathlib.Path | None = None
_lock = threading.Lock()

_state: dict[str, Any] = {
    "session": None,          # the active session (None when no session is running)
    "last_ended": None,       # most recently ended session (overview pending delivery?)
    "last_start_at": None,    # epoch seconds of the most recent start
    "next_session_reminders": [],  # [{channel_id, message, created_at}]
}


# ── Path resolution ──────────────────────────────────────────────────────

_persist_disabled_warned = False


def _resolve_path() -> pathlib.Path | None:
    global _store_path, _persist_disabled_warned
    if _store_path is not None:
        return _store_path if _store_path else None
    env = os.environ.get("SESSIONS_PERSIST_FILE")
    if env is not None:
        if not env and not _persist_disabled_warned:
            # An EMPTY value (as opposed to an unset variable) explicitly disables
            # persistence — that silently lost the active session across the
            # 2026-10-20 restarts. Make it impossible to miss in the logs.
            log.warning(
                "SESSIONS_PERSIST_FILE is set to an EMPTY value — session persistence is "
                "DISABLED: the active session and queued next-session reminders are "
                "memory-only and will be LOST on every bot restart (crash recovery #28 "
                "is off too). Comment out the line in .env to use the default (%s).",
                _DEFAULT_PATH)
            _persist_disabled_warned = True
        _store_path = pathlib.Path(env) if env else None
        return _store_path if _store_path else None
    return _DEFAULT_PATH


def notes_dir() -> pathlib.Path:
    """Directory holding the per-session folders (``SESSION_NN[_name]/``).

    Lives inside the knowledge base by default so each session's combined
    ``notes.md`` is automatically part of the RAG-enabled documents (and
    shows up in /list_kb_docs).  The raw dot-dir files inside each session
    folder (uploads, transcripts, the notes attachment) are hidden from the
    indexer.  Override with ``SESSIONS_NOTES_DIR`` to point inside an
    Obsidian campaign vault (#19, docs/obsidian-vault.md).
    """
    from config.settings import SESSIONS_NOTES_DIR
    return pathlib.Path(SESSIONS_NOTES_DIR)


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

    Also runs the one-time, idempotent layout migration (legacy date-named
    folders → ``SESSION_NN``, visible ``attachments/`` / ``transcripts/`` →
    hidden dot-dirs, inline ``## Notes`` → notes attachment) — deliberately
    BEFORE the vector index is first loaded, so stale per-file rows are
    pruned and the recombined notes file is embedded exactly once — and the
    persisted state is re-pointed at the renamed folders.
    """
    renames: dict[str, str] = {}
    try:
        renames = migrate_legacy_session_dirs()
    except Exception as e:  # pragma: no cover - defensive
        log.warning("Legacy session layout migration failed: %s", e)

    path = _resolve_path()
    if path is None or not path.exists():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        paths_fixed = False
        # Rename maps accumulate across processes: a map persisted by an
        # earlier migration is re-applied to the state (a no-op once the
        # paths are already modern) so state written between a manual
        # migration and the next load still gets re-pointed.
        saved_map = raw.get("renames")
        merged_map = dict(saved_map) if isinstance(saved_map, dict) else {}
        merged_map.update(renames)
        if merged_map:
            paths_fixed = _apply_folder_renames(raw, merged_map)
        _state["session"] = raw.get("session")
        _state["last_ended"] = raw.get("last_ended")
        _state["last_start_at"] = raw.get("last_start_at")
        reminders = raw.get("next_session_reminders", [])
        _state["next_session_reminders"] = reminders if isinstance(reminders, list) else []
        log.info("Loaded persisted session state (%d queued next-session reminder(s))",
                 len(_state["next_session_reminders"]))
        if paths_fixed:
            # The migration renamed the folders the state still pointed at —
            # persist the corrected paths (and the map) immediately.
            _state["renames"] = {k: v for k, v in merged_map.items() if isinstance(v, str)}
            _save()
    except Exception as e:
        log.warning("Could not load sessions file: %s", e)

    # Crash recovery (#28): a session that was active when the bot last
    # stopped must be resumed, validated and made usable again.
    try:
        recover_active_session()
    except Exception as e:  # pragma: no cover - defensive
        log.warning("Active-session recovery failed: %s", e)


def recover_active_session() -> dict[str, Any] | None:
    """Resume and repair the session that was active when the bot last stopped.

    Called from :func:`load_persisted` at startup.  A crashed (or never
    cleanly stopped) bot leaves ``session`` set in the persisted file; this
    makes sure the resumed session is usable again:

    * a missing/pointless notes file is recreated from the stored state;
    * notes are re-read from disk so edits made while the bot was down
      (e.g. via Obsidian) are picked up;
    * a corrupt persisted session (no usable ``started_at`` / no location)
      is dropped and persisted as such, so it cannot block ``/start_session``
      forever;
    * the resume is logged loudly (age included) so the operator sees that
      the old session continues instead of a new one being started.

    Returns the resumed session, or None when no session was active.  Indexing
    is best-effort here — at import time there is usually no event loop yet;
    :func:`index_current_session` covers that from ``on_ready``.
    """
    s = _state.get("session")
    if not isinstance(s, dict):
        return None
    if s.get("ended_at"):
        # Defensive: an ended session belongs in last_ended, not active.
        log.info("Persisted 'session' already ended (%s) — treating as no active session",
                 s.get("name"))
        _state["session"] = None
        _save()
        return None
    started = s.get("started_at")
    if not isinstance(started, (int, float)) or started <= 0:
        log.warning("Dropping corrupt persisted active session (no usable started_at): %r",
                    {k: v for k, v in s.items() if k not in ("notes", "overview", "merged_log")})
        _state["session"] = None
        _save()
        return None
    # Re-point the notes file when it is missing but the dir's copy exists.
    path = pathlib.Path(s["file"]) if s.get("file") else None
    if (path is None or not path.exists()) and s.get("dir") \
            and (pathlib.Path(s["dir"]) / "notes.md").exists():
        path = pathlib.Path(s["dir"]) / "notes.md"
        s["file"] = str(path)
    file_missing = path is None or not path.exists()
    if file_missing:
        if path is not None and path.parent.is_dir():
            # The folder survived but notes.md didn't — recreate it in place.
            pass
        else:
            # No usable location at all — rebuild the location from state so
            # the session keeps working (crash mid-creation, folder deleted,
            # a vault sync that did not pull it back).  Bump past any existing
            # sibling folders, same guard as start_session.
            file_path = _session_file_path(s.get("name") or None, 1)
            while file_path.parent.exists():
                file_path = _bump_file_index(file_path, s.get("name") or None)
            path = file_path
            s["file"] = str(file_path)
            s["dir"] = str(file_path.parent)
        log.warning("Resumed session %r had no notes file on disk — recreating at %s",
                    s.get("name"), s["file"])
    # Ensure the notes attachment exists too (deleted folder / legacy session)
    # BEFORE any disk re-read, so state notes survive as the source.
    if not _notes_attachment_path(s).is_file():
        _write_notes_attachment(s)
    if file_missing:
        _write_session_file(s)      # recreate from state (nothing to clobber)
    else:
        # Disk is the source of truth: re-read user edits made while the bot
        # was down WITHOUT re-rendering (a render would drop any manual
        # changes outside the header sections).
        _reindex_notes_file(s)      # indexing inside is best-effort
    age_h = (time.time() - started) / 3600
    if age_h >= STALE_SESSION_SEC / 3600:
        log.warning(
            "Resumed active session %r after restart, but it is already %.1f h old — "
            "the next /start_session will auto-end it as stale (>%d h). End it "
            "explicitly with /end_session if you want an AI overview.",
            s.get("name") or "(untitled)", age_h, STALE_SESSION_SEC // 3600)
    else:
        log.info(
            "Resumed active session %r after restart (started %.1f h ago) — the session "
            "continues; end it with /end_session or upload documents/notes as usual.",
            s.get("name") or "(untitled)", age_h)
    return s


async def index_current_session() -> bool:
    """(Re)index the active session's notes file into the KB (best-effort).

    Called from ``on_ready`` after a restart with a resumed session, when an
    event loop exists — so RAG serves the latest content even if the notes
    were edited while the bot was down.  Returns True when indexing succeeded.
    """
    s = get_current_session()
    if s is None:
        return False
    paths = _session_index_paths(s)
    if not paths:
        return False
    from kb.retrievers import update_kb_document
    results = [await update_kb_document(p) for p in paths]
    ok = all(results)
    if not ok:
        log.warning("Re-index of resumed session %r failed — run /reindex_kb.",
                    s.get("name"))
    return ok


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


def _next_session_index() -> int:
    """Next globally unique session index (max existing SESSION_NN + 1).

    Without a date in the folder name the index can no longer restart per
    day — it must stay unique across the whole notes directory.
    """
    try:
        folders = [p for p in notes_dir().iterdir() if p.is_dir()]
    except OSError:
        return 1
    best = 0
    for f in folders:
        m = re.match(rf"^{_SESSION_PREFIX}(\d+)(?:_|$)", f.name)
        if m:
            try:
                best = max(best, int(m.group(1)))
            except ValueError:
                pass
    return best + 1


def _session_dir(session: dict[str, Any]) -> pathlib.Path:
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


def _session_file_path(name: str | None, index: int) -> pathlib.Path:
    """Each session is a folder: ``SESSION_<NN>[_<name>]/notes.md``."""
    safe = _sanitize_name(name)
    stem = f"{_SESSION_PREFIX}{index:02d}" + (f"_{safe}" if safe else "")
    return notes_dir() / stem / "notes.md"


def _bump_file_index(file_path: pathlib.Path, name: str | None) -> pathlib.Path:
    """Next free sibling path for the same name (folder index +1).

    Used by :func:`start_session`'s collision guard and by crash recovery so
    a recreated session never points at an existing folder.
    """
    m = re.match(rf"^{_SESSION_PREFIX}(\d+)(?:_|$)", file_path.parent.name)
    index = (int(m.group(1)) + 1) if m else 2
    return _session_file_path(name, index)


def _unique_path(directory: pathlib.Path, stem: str, ext: str) -> pathlib.Path:
    """Return ``directory/<stem><ext>`` or ``<stem>_<n><ext>`` if taken."""
    base = directory / f"{stem}{ext}"
    if not base.exists():
        return base
    n = 2
    while (directory / f"{stem}_{n}{ext}").exists():
        n += 1
    return directory / f"{stem}_{n}{ext}"


def _session_docs(session: dict[str, Any], subdir: str) -> list[str]:
    """Names of standalone uploaded documents in one of the session's sub-folders.

    The reserved notes attachment (``.attachments/notes.md``) is NOT listed —
    the notes have their own display section; this lists *uploads/transcripts*
    only (used by /session_info and the AI document assembly).
    """
    d = _session_dir(session) / subdir
    if not d.exists():
        return []
    try:
        return sorted(
            f.name for f in d.iterdir()
            if f.is_file() and not f.name.startswith(".")
            and f.suffix.lower() in {".md", ".txt"}
            and not (subdir == _SUBDIR_ATTACHMENTS and f.name == _NOTES_FILE)
        )
    except OSError:
        return []


#: Total char budget for the assembled session-document text fed to
#: summary-type model calls (recap, merged log, highlight selection).  It
#: exists only to keep the prompt size bounded for very long sessions; each
#: included document keeps its header + full text (trimmed from the tail if it
#: alone exceeds the whole budget).
SESSION_DOCS_BUDGET = 60_000


def session_documents_text(session: dict[str, Any]) -> tuple[str, list[str]]:
    """Full text of the session's own documents (attachments + transcripts).

    Returns ``(text, refs)`` where *refs* are ``<subdir>/<filename>`` labels
    (one per non-empty document found).  *text* is the concatenation of those
    documents (each with a small ``Source:`` header so the model can tell them
    apart), trimmed to :data:`SESSION_DOCS_BUDGET` chars in total.  When the
    budget is exhausted the remaining documents are dropped with a note so the
    model knows more material exists for the session.
    """
    parts: list[str] = []
    refs: list[str] = []
    omitted: list[str] = []
    budget_left = SESSION_DOCS_BUDGET
    for subdir in (_SUBDIR_ATTACHMENTS, _SUBDIR_TRANSCRIPTS):
        for name in _session_docs(session, subdir):
            ref = f"{subdir}/{name}"
            path = _session_dir(session) / ref
            try:
                body = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if not body:
                continue
            refs.append(ref)
            if budget_left <= 0:
                omitted.append(ref)
                continue
            # Keep each document whole when it fits; trim a single document
            # that alone exceeds the whole budget from the tail (the head
            # carries the session's most relevant opening material).
            if len(body) > budget_left:
                body = body[:budget_left].rstrip() + "\n…[document truncated]"
            parts.append(f"Source: {ref}\n\n{body}")
            budget_left -= len(body)
    if omitted:
        parts.append("…[additional documents not included due to size: "
                     + ", ".join(omitted) + "]")
    return "\n".join(parts).strip(), refs


def _documents_block_for_dir(folder: pathlib.Path) -> list[str]:
    """Render the combined ``## Documents`` lines for one session folder.

    Every uploaded document / transcript in the hidden dot-dirs contributes
    one ``### <filename>`` subsection with its full text (the file's own
    leading H1 is dropped — it duplicates the subsection heading).  The
    reserved notes attachment participates too (so live notes stay
    RAG-reachable while no AI-merged log exists), but only when it actually
    holds note bullets — an empty placeholder is skipped.  Uploaded raw
    files are immutable once written, so rendering is deterministic and
    idempotent: same dot-dir contents ⇒ byte-identical section.
    """
    lines: list[str] = []
    for subdir in (_SUBDIR_ATTACHMENTS, _SUBDIR_TRANSCRIPTS):
        d = folder / subdir
        if not d.exists():
            continue
        try:
            files = sorted(
                f for f in d.iterdir()
                if f.is_file() and not f.name.startswith(".")
                and f.suffix.lower() in {".md", ".txt"}
            )
        except OSError:
            continue
        for f in files:
            try:
                body = f.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if not body:
                continue
            if subdir == _SUBDIR_ATTACHMENTS and f.name == _NOTES_FILE:
                # Skip the notes attachment while it is just a placeholder —
                # the (no notes) skeleton adds nothing to the RAG document.
                if not any(_NOTE_BULLET_RE.match(l) for l in body.splitlines()):
                    continue
            # Drop the raw file's own leading "# Title" line — the subsection
            # heading below already names the document, and a nested H1 would
            # otherwise split the chunker section in two.
            first, _, rest = body.partition("\n")
            if re.match(r"^#\s+", first):
                body = rest.strip()
            lines += ["", f"### {f.name}", "", body]
    if not lines:
        return []
    return ["", "## Documents (combined uploads + transcripts — full text)", ""] + lines


def _documents_section_lines(session: dict[str, Any]) -> list[str]:
    """The combined ``## Documents`` section for *session*'s folder."""
    return _documents_block_for_dir(_session_dir(session))


def _session_file_content(session: dict[str, Any]) -> str:
    """Render the session's main ``notes.md`` from its state + raw dot-dir files.

    Timestamped notes are NOT part of this file any more — they live in the
    notes attachment (``.attachments/notes.md``); the header carries a
    pointer to it.  The session's RAG content is the AI-merged canonical log
    (``## Session Log``, written at end when the AI combined all player
    uploads into one de-duplicated record); without it — AI unavailable,
    stale auto-end, legacy sessions — the notes attachment plus uploaded
    documents and transcripts appear with their FULL text under ``##
    Documents`` instead, so content stays reachable either way.  The brief
    end-of-session overview sits right after the header.
    """
    started = fmt_local(session["started_at"])
    ended = (fmt_local(session["ended_at"])
             if session.get("ended_at") else None)
    n_notes = len(session.get("notes") or [])
    lines = [
        f"# Session: {session.get('name') or 'Untitled'}",
        "",
        f"- Started: {started}",
    ]
    if ended:
        lines.append(f"- Ended: {ended}")
    lines.append(f"- Notes: {_SUBDIR_ATTACHMENTS}/{_NOTES_FILE} ({n_notes} note(s))")
    # The brief end-of-session overview leads the body (the reader gets the
    # digest before the long log).
    if session.get("overview"):
        lines += ["", "## Overview (written when the session ended)", "", str(session["overview"]).strip()]
    # The AI-merged canonical log (written at end) replaces the mechanical
    # full-text copy — it is the de-duplicated combination of all uploads.
    # Without it (AI unavailable, stale auto-end, legacy sessions) the
    # mechanical section keeps every document's content reachable via RAG.
    if session.get("merged_log"):
        lines += ["", "## Session Log (combined from all uploads — AI-merged)",
                  "", str(session["merged_log"]).strip()]
    else:
        lines += _documents_section_lines(session)
    return "\n".join(lines) + "\n"


def _write_session_file(session: dict[str, Any]) -> None:
    """(Re)write the session's main ``notes.md``. Failures are logged, never raised."""
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


def _notes_attachment_path(session: dict[str, Any]) -> pathlib.Path:
    """The session folder's notes attachment (``.attachments/notes.md``)."""
    return _session_dir(session) / _SUBDIR_ATTACHMENTS / _NOTES_FILE


def _notes_attachment_content(session: dict[str, Any]) -> str:
    """Render the notes attachment file from state: header + note bullets.

    Deliberately NO ``##`` section heading below the title — when the
    combined ``## Documents`` fallback embeds this file, the bullets must
    nest under the file's ``###`` subsection, not open a competing ``##``.
    """
    lines = [f"# Session Notes: {session.get('name') or 'Untitled'}", ""]
    for ts, text in session.get("notes", []):
        lines.append(f"- ({fmt_local(ts)}) {text}")
    if not session.get("notes"):
        lines.append("(no notes)")
    return "\n".join(lines) + "\n"


def _write_notes_attachment(session: dict[str, Any]) -> None:
    """(Re)write the session's notes attachment. Failures logged, never raised."""
    path = _notes_attachment_path(session)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_notes_attachment_content(session), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:
        log.warning("Failed to write session notes attachment %s: %s", path, e)


def _write_session_files(session: dict[str, Any]) -> None:
    """Write BOTH session files: notes attachment first, then the main file
    (whose ``## Documents`` fallback embeds the freshly written notes)."""
    _write_notes_attachment(session)
    _write_session_file(session)


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
        from utils.background_tasks import spawn_tracked_task

        asyncio.get_running_loop()  # raises RuntimeError when no loop → skip
        # Tracked task — an untracked create_task can be GC'd mid-flight, the
        # same failure class as the #73 warm-reload bug (code review §1.3).
        fut = spawn_tracked_task(
            update_kb_document(path), name=f"session-index:{path.name}"
        )

        def _on_done(t: "asyncio.Task[Any]") -> None:
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


def _session_index_paths(session: dict[str, Any]) -> list[pathlib.Path]:
    """Indexable files for a session: ONLY the main ``notes.md``.

    The combined notes file carries the full text of the notes attachment,
    every upload and every transcript (see :func:`_documents_section_lines`);
    the raw dot-dir files are hidden from the KB indexer and must never be
    indexed on their own — that is what caused near-duplicate session content
    to be attached multiple times by RAG.
    """
    f = session.get("file")
    if f and pathlib.Path(f).exists():
        return [pathlib.Path(f)]
    return []


def _index_session_paths(session: dict[str, Any]) -> None:
    for p in _session_index_paths(session):
        _index_session_file(p)


def _reindex_notes_file(session: dict[str, Any]) -> None:
    """Re-read the session's notes file from disk and sync state + index.

    The user may have edited the markdown on disk; treat it as the new
    source of truth.  Source of record is the notes attachment
    (``.attachments/notes.md``) — it holds ONLY note bullets, so parsing
    the whole file is safe.  A session folder still in the LEGACY layout
    (no attachment yet, a ``## Notes`` section inside notes.md) is parsed
    with the old section scoping so document text in the same file cannot
    be mistaken for notes.  When neither source carries a parsable notes
    region (new-format main file, attachment deleted) the in-memory notes
    are kept rather than wiped.  The overview stays in memory.
    """
    notes: list[list[float | str]] = []
    att_path = _notes_attachment_path(session)
    text: str | None = None
    if att_path.is_file():
        try:
            text = att_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            log.warning("Could not read session notes attachment %s: %s", att_path, e)
            return
        for line in text.splitlines():
            bm = _NOTE_BULLET_RE.match(line)
            if bm:
                try:
                    ts = parse_local(bm.group(1))
                except ValueError:
                    continue
                notes.append([ts, bm.group(2)])
    else:
        # Legacy fallback: notes bullets inside the main notes.md's ## Notes
        # section (scoped so combined document text is never mis-parsed).
        path = pathlib.Path(session.get("file") or "")
        if not path or not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            log.warning("Could not read session notes file %s: %s", path, e)
            return
        m = re.search(r"^## Notes\s*$", text, flags=re.MULTILINE)
        if not m:
            # New-format file without a notes attachment — nothing on disk to
            # re-parse; keep the in-memory notes (recovery recreates the
            # attachment from state).
            return
        section = text[m.end():]
        cut = re.search(r"^## ", section, flags=re.MULTILINE)
        if cut:
            section = section[:cut.start()]
        for line in section.splitlines():
            bm = _NOTE_BULLET_RE.match(line)
            if bm:
                try:
                    ts = parse_local(bm.group(1))
                except ValueError:
                    continue
                notes.append([ts, bm.group(2)])
    session["notes"] = notes
    _index_session_paths(session)


# ── Layout migration (legacy date-folders / visible sub-dirs / inline notes) ──

#: Legacy session folder names: ``<YYYY-MM-DD>_<NN>[_<name>]``.
_LEGACY_FOLDER_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d+)(?:_(.*))?$")
#: Current session folder names: ``SESSION_<NN>[_<name>]``.
_SESSION_FOLDER_RE = re.compile(rf"^{_SESSION_PREFIX}(\d+)(?:_|$)")


def _migrate_legacy_session_dir(folder: pathlib.Path) -> bool:
    """Migrate ONE session folder's raw documents to the hidden dot-dir layout.

    Legacy folders stored raw documents in visible ``attachments/`` /
    ``transcripts/`` sub-folders, which the KB indexed file-by-file (the
    near-duplicate RAG problem).  Renames them to the dot-names so the
    indexer skips them; when both exist, merges file-by-file.  Returns True
    when anything changed.
    """
    changed = False
    for old_name, new_name in (("attachments", _SUBDIR_ATTACHMENTS),
                               ("transcripts", _SUBDIR_TRANSCRIPTS)):
        src = folder / old_name
        dst = folder / new_name
        if not src.is_dir():
            continue
        if not dst.exists():
            try:
                os.replace(src, dst)
                changed = True
            except OSError as e:
                log.warning("Could not migrate %s → %s: %s", src, dst, e)
        else:
            try:
                for f in sorted(src.iterdir()):
                    if not f.is_file():
                        continue
                    if (dst / f.name).exists():
                        # Same-named file already migrated — the legacy copy
                        # is a duplicate; drop it so the folder can be removed.
                        f.unlink()
                        changed = True
                    else:
                        os.replace(f, dst / f.name)
                        changed = True
                src.rmdir()  # now empty — drop the legacy folder
            except OSError as e:
                log.warning("Could not merge %s into %s: %s", src, dst, e)
    return changed


def _migrate_notes_to_attachment(folder: pathlib.Path) -> bool:
    """Move ONE folder's inline ``## Notes`` bullets into the notes attachment.

    Legacy ``notes.md`` files carried the live notes as a ``## Notes``
    section.  They are extracted (bullets kept verbatim) into
    ``.attachments/notes.md`` and the section is removed from the main file,
    which gains a ``- Notes:`` pointer under its Started/Ended header lines.
    For folders that never had a ``## Notes`` section an (empty) attachment
    is still created so every session folder has the new form.  Never
    re-renders the whole main file (that would drop a stored overview) —
    the live renderer regenerates it from state on the next change.
    Idempotent: no-op when the attachment already exists.
    """
    notes_md = folder / "notes.md"
    att = folder / _SUBDIR_ATTACHMENTS / _NOTES_FILE
    if att.exists():
        return False
    if not notes_md.is_file():
        return False
    try:
        text = notes_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if re.search(r"^- Notes:", text, flags=re.MULTILINE):
        # Already in the NEW form (pointer present) but the attachment is
        # gone — a deletion, not a legacy layout.  Do not fabricate an empty
        # attachment here (it would clobber the notes recovery path); crash
        # recovery recreates it from persisted state instead.
        return False
    title = "Untitled"
    mt = re.search(r"^# Session:\s*(.*)$", text, flags=re.MULTILINE)
    if mt and mt.group(1).strip():
        title = mt.group(1).strip()
    bullets: list[str] = []
    m = re.search(r"^## Notes\s*$", text, flags=re.MULTILINE)
    if m:
        section = text[m.end():]
        cut = re.search(r"^## ", section, flags=re.MULTILINE)
        if cut:
            section = section[:cut.start()]
        for line in section.splitlines():
            bm = _NOTE_BULLET_RE.match(line)
            if bm:
                bullets.append(f"- ({bm.group(1)}) {bm.group(2)}")
    # Write the attachment first (header + verbatim bullets or placeholder).
    body = [f"# Session Notes: {title}", ""] + (bullets or ["(no notes)"])
    try:
        folder.joinpath(_SUBDIR_ATTACHMENTS).mkdir(parents=True, exist_ok=True)
        att.write_text("\n".join(body) + "\n", encoding="utf-8")
    except OSError as e:
        log.warning("Could not create notes attachment %s: %s", att, e)
        return False
    # Surgical main-file edit: drop the ## Notes section, add the pointer.
    lines = text.splitlines()
    if m:
        # Locate the section bounds in the lines list (start at '## Notes',
        # end at the next '## ' heading or EOF).
        start_i = next(i for i, l in enumerate(lines) if l.strip() == "## Notes")
        end_i = next((i for i, l in enumerate(lines[start_i + 1:], start_i + 1)
                      if l.startswith("## ")), len(lines))
        del lines[start_i:end_i]
        # Trim blank lines left dangling before the next section heading.
        while lines and not lines[-1].strip():
            lines.pop()
    if not any(l.startswith("- Notes:") for l in lines):
        insert_at = 0
        for i, l in enumerate(lines):
            if l.startswith("- Started:") or l.startswith("- Ended:"):
                insert_at = i + 1
        lines.insert(insert_at, f"- Notes: {_SUBDIR_ATTACHMENTS}/{_NOTES_FILE}")
    try:
        notes_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as e:
        log.warning("Could not update migrated notes file %s: %s", notes_md, e)
        return False
    return True


def _migrate_legacy_notes_file(folder: pathlib.Path) -> None:
    """Append the combined Documents section to a migrated session's notes.md.

    For sessions that are NOT in memory we must not re-render the whole file
    (that would drop the stored overview), so the section is appended instead;
    the live renderer regenerates it from the dot-dirs on the next change.
    Idempotent: no-op when a ``## Documents`` section already exists — and a
    session whose AI-merged ``## Session Log`` is already present never gets
    one (the merged log intentionally replaces the mechanical copy).
    """
    notes = folder / "notes.md"
    if not notes.is_file():
        return
    try:
        text = notes.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    if re.search(r"^## Documents\b", text, flags=re.MULTILINE):
        return
    if re.search(r"^## Session Log", text, flags=re.MULTILINE):
        return
    block = _documents_block_for_dir(folder)
    if not block:
        return
    if not text.endswith("\n"):
        text += "\n"
    try:
        notes.write_text(text + "\n".join(block) + "\n", encoding="utf-8")
    except OSError as e:
        log.warning("Could not update migrated notes file %s: %s", notes, e)


def migrate_legacy_session_dirs() -> dict[str, str]:
    """One-time, idempotent migration of ALL session folders.

    Called from :func:`load_persisted` at startup.  Brings every folder to
    the current form:

    1. **Rename** legacy date-named folders (``2026-10-20_02_Title``) to
       ``SESSION_<NN>[_Title]`` — renumbered chronologically (date, old
       index) continuing after the highest existing ``SESSION_NN``, so the
       numbers are globally unique once the date is gone.
    2. **Raw documents:** visible ``attachments/`` / ``transcripts/`` →
       hidden dot-dirs (old behavior).
    3. **Notes:** the inline ``## Notes`` section → the notes attachment
       (``.attachments/notes.md``) + a pointer line in the main file.

    Returns ``{old_folder_name: new_folder_name}`` for the renames actually
    performed (empty dict when nothing changed — every run after the first).
    :func:`load_persisted` uses the map to re-point the persisted session
    state at the renamed folders.
    """
    try:
        root = notes_dir()
        if not root.exists():
            return {}
        folders = [p for p in sorted(root.iterdir())
                   if p.is_dir() and not p.name.startswith(".")]
    except OSError:
        return {}

    # 1. Chronological renumbering of the legacy date folders.
    max_idx = 0
    legacy: list[pathlib.Path] = []
    for p in folders:
        m_new = _SESSION_FOLDER_RE.match(p.name)
        if m_new:
            try:
                max_idx = max(max_idx, int(m_new.group(1)))
            except ValueError:
                pass
            continue
        if _LEGACY_FOLDER_RE.match(p.name):
            legacy.append(p)
    renames: dict[str, str] = {}
    # Sort by (date, old index) — ISO dates sort chronologically by name.
    legacy.sort(key=lambda p: (_LEGACY_FOLDER_RE.match(p.name).group(1),
                               int(_LEGACY_FOLDER_RE.match(p.name).group(2))))
    for p in legacy:
        m = _LEGACY_FOLDER_RE.match(p.name)
        title = f"_{m.group(3)}" if m.group(3) else ""  # the trailing ``_<name>`` part
        while True:
            max_idx += 1
            new_name = f"{_SESSION_PREFIX}{max_idx:02d}" + title
            if not (root / new_name).exists():
                break
        try:
            os.replace(p, root / new_name)
            renames[p.name] = new_name
            log.info("Renamed legacy session folder %s → %s", p.name, new_name)
        except OSError as e:
            log.warning("Could not rename session folder %s → %s: %s", p.name, new_name, e)
            continue
        p = root / new_name
        # 2 + 3. Content migration on the renamed folder.
        try:
            content_changed = False
            if _migrate_legacy_session_dir(p):
                content_changed = True
            if _migrate_notes_to_attachment(p):
                content_changed = True
            _migrate_legacy_notes_file(p)
            if content_changed:
                log.info("Migrated session folder to the current layout: %s", p.name)
        except Exception as e:  # one bad folder must not block the rest
            log.warning("Legacy migration failed for %s: %s", p, e)

    # Content-only migration of already-SESSION_-named folders (idempotent).
    for p in folders:
        if p.name in renames or not _SESSION_FOLDER_RE.match(p.name):
            continue
        if not (root / p.name).exists():
            continue
        try:
            content_changed = False
            if _migrate_legacy_session_dir(p):
                content_changed = True
            if _migrate_notes_to_attachment(p):
                content_changed = True
            _migrate_legacy_notes_file(p)
            if content_changed:
                log.info("Migrated session folder to the current layout: %s", p.name)
        except Exception as e:
            log.warning("Legacy migration failed for %s: %s", p, e)
    return renames


def _apply_folder_renames(raw: dict[str, Any], renames: dict[str, str]) -> bool:
    """Re-point the persisted session state at folders renamed by the migration."""
    changed = False
    for key in ("session", "last_ended"):
        s = raw.get(key)
        if not isinstance(s, dict):
            continue
        d = s.get("dir")
        if d:
            p = pathlib.Path(d)
            if p.name in renames:
                s["dir"] = str(p.parent / renames[p.name])
                changed = True
        f = s.get("file")
        if f:
            p = pathlib.Path(f)
            if p.parent.name in renames:
                s["file"] = str(p.parent.parent / renames[p.parent.name] / p.name)
                changed = True
    return changed


# ── Public API — session lifecycle ───────────────────────────────────────

def get_current_session() -> dict[str, Any] | None:
    """The current session if one is active, else None."""
    s: dict[str, Any] | None = _state.get("session")
    if s and not s.get("ended_at"):
        return s
    return None


def get_last_session() -> dict[str, Any] | None:
    """Most recent session (active or the last ended one) — for /session_notes view."""
    return _state.get("session") or _state.get("last_ended")


def _pending_manual_overview() -> dict[str, Any] | None:
    """Return the last-ENDED session whose AI overview has not been delivered yet.

    The overview of a manually ended session is delivered when the NEXT
    session starts (per spec).  Once delivered, ``overview_delivered`` is set
    so it is never sent twice.
    """
    s: dict[str, Any] | None = _state.get("last_ended")
    if s and not s.get("overview_delivered") and s.get("overview"):
        return s
    return None


def start_session(name: str | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
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

    closed_info: dict[str, Any] | None = None
    if current is not None:
        age = now - current["started_at"]
        if age >= STALE_SESSION_SEC:
            # Stale session — end it without an AI overview and start fresh.
            closed = _end_session_internal(current, overview=None)
            closed_info = {"kind": "stale", "session": closed}
            log.info("Auto-ended stale session %s (age %.1f h)", current.get("name"), age / 3600)
        else:
            # Name the active session explicitly — after a restart the user
            # should see that the old session was resumed and can continue it
            # (#28) instead of getting an unexplained refusal.
            started_when = fmt_local(current["started_at"])
            msg = (f"A session is already active: **{current.get('name') or 'Untitled'}** "
                   f"(started {started_when}) — you can continue it as usual, or end it "
                   "first with `/end_session` (or start again once it is older than 12 h "
                   "to auto-end it).")
            log.info("start_session refused: session %s still active (age %.1f h)",
                     current.get("name"), age / 3600)
            raise ValueError(msg)
    else:
        # No active session — but a previously ended one may have an
        # overview that is due for delivery now.
        pending = _pending_manual_overview()
        if pending is not None:
            # The flag is set only after Discord accepts the overview message.
            # Marking it here would drop the overview if the followup never lands.
            closed_info = {"kind": "manual", "session": pending}

    dt = datetime.fromtimestamp(now)
    safe_name = _sanitize_name(name)
    index = _next_session_index()
    file_path = _session_file_path(safe_name, index)
    # Defensive collision guard: never point a new session at an existing
    # folder, no matter how it got there (manual rename, clock skew, a
    # partially-created session). Bump until the path is free.
    while file_path.parent.exists():
        file_path = _bump_file_index(file_path, safe_name)
    session = {
        "id": dt.strftime("%Y%m%d%H%M%S"),
        "name": safe_name,
        "started_at": now,
        "ended_at": None,
        "notes": [],          # [[epoch, text], ...] — full, un-split text
        "overview": None,     # AI overview written on end (or None)
        "merged_log": None,   # AI-merged canonical session log (or None → mechanical fallback)
        "dir": str(file_path.parent),
        "file": str(file_path),
    }
    _state["session"] = session
    _state["last_start_at"] = now
    _write_session_files(session)
    _index_session_paths(session)
    _save()
    log.info("Session started: %s (dir: %s)", safe_name or "(untitled)", session["dir"])
    return session, closed_info


def _end_session_internal(session: dict[str, Any], overview: str | None,
                          merged_log: str | None = None) -> dict[str, Any]:
    """Shared end logic — sets ended_at, moves to last_ended, persists."""
    session["ended_at"] = time.time()
    if overview is not None:
        session["overview"] = overview
        session["overview_delivered"] = False  # delivered at the NEXT start
    else:
        session["overview_delivered"] = True   # nothing to deliver (stale end)
    if merged_log is not None:
        session["merged_log"] = merged_log
    _state["last_ended"] = session
    _state["session"] = None
    _write_session_files(session)
    _index_session_paths(session)
    _save()
    return session


def mark_overview_delivered(session: dict[str, Any]) -> None:
    """Remember that the previous session's overview was posted to Discord."""
    session["overview_delivered"] = True
    _save()


def end_session(overview: str | None = None, name: str | None = None,
                merged_log: str | None = None) -> dict[str, Any] | None:
    """End the current session.

    *name* (optional) renames the session in its file/state; *overview* is
    stored in the session file and returned to the caller for delivery;
    *merged_log* is the AI-merged canonical session log that becomes the
    session's RAG content (None → the mechanical full-text copy is kept).
    Returns the ended session dict, or None when no session was active.
    """
    session = get_current_session()
    if session is None:
        return None
    if name:
        safe = _sanitize_name(name)
        if safe:
            session["name"] = safe
    return _end_session_internal(session, overview=overview, merged_log=merged_log)


# ── Public API — notes ───────────────────────────────────────────────────

def add_note(text: str, author: str = "") -> dict[str, Any] | None:
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
    _write_session_files(session)
    _index_session_paths(session)
    _save()
    log.info("Session note added to %s: %.60s", session.get("name"), clean)
    return session


def _append_document_file(session: dict[str, Any], subdir: str, title: str,
                          text: str) -> pathlib.Path:
    """Write *text* as one standalone ``.md`` file inside a session sub-folder.

    Returns the written path.  The file keeps the raw document text (a
    markdown header naming the source is prepended for context).  It is NOT
    indexed directly — the caller re-renders ``notes.md``, which combines the
    full text into the session's single RAG document.
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


def add_document(text: str, title: str = "",
                 session: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, int]:
    """Store an uploaded text document (``.txt`` / ``.md``) for a session.

    The whole file is written as a raw ``.md`` under the session's hidden
    ``.attachments/`` folder (one file per upload, never indexed directly),
    and the combined ``notes.md`` is re-rendered with the document's full text
    so each upload keeps its own semantic KB chunks inside the session's
    single RAG document — no pre-splitting.
    The document shows up in ``/session_notes`` via its dot-dir listing.

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


def get_notes(session: dict[str, Any] | None = None) -> list[list[float | str]]:
    """Notes of *session* (default: current, else last known).

    Each entry is ``[epoch, text]`` where *text* is the full (un-split) note
    or an attachment/transcript pointer line.
    """
    s = session if session is not None else _state.get("session")
    if not s:
        return []
    return [list(n) for n in s.get("notes", [])]


def refresh_notes_from_disk(session: dict[str, Any] | None = None) -> list[list[float | str]]:
    """Re-read the notes file from disk (user may have edited it) and sync state."""
    s = session if session is not None else _state.get("session")
    if not s:
        return []
    _reindex_notes_file(s)
    _save()
    return [list(n) for n in s.get("notes", [])]


# ── Public API — next-session reminders ──────────────────────────────────

def queue_next_session_reminder(channel_id: int, message: str) -> dict[str, Any]:
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


def list_queued_reminders() -> list[dict[str, Any]]:
    return [dict(r) for r in _state.get("next_session_reminders", [])]


async def deliver_queued_reminders(bot: "discord.Client") -> int:
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
    delivered: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
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
                      r["channel_id"], e, exc_info=True)
            failed.append(r)

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
