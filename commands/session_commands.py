"""Session slash commands — /start_session, /end_session,
/remind_next_session, /session_notes and /session_info.

Delegates all state handling to ``bot_core.sessions``; this module only
translates between Discord interactions and the session store, plus the
AI-generated end-of-session recap — a very brief digest (what happened +
suggested next steps) that is stored with the session at /end_session and
delivered when the NEXT session starts (same model-resolution pattern as
/summarize in commands/utility_commands.py).
"""
from __future__ import annotations

import logging
import pathlib
from datetime import datetime
from typing import Any

import discord

from bot_core.ai_client import _make_client, _validate_model, complete_text
from bot_core.history import get_active_char_key
from config.characters import get_character
from config.settings import (
    DEFAULT_MODEL,
    DEFAULT_SESSION_MERGE_PROMPT,
    DEFAULT_SESSION_RECAP_PROMPT,
    MERGE_LOG_REASONING_EFFORT,
    RECAP_REASONING_EFFORT,
    SESSION_MERGE_PROMPT,
    SESSION_RECAP_PROMPT,
    SUMMARY_CALL_MAX_TOKENS,
)

log = logging.getLogger("bot.session_commands")

# Max chars of the recap posted to the channel (Discord limit is 2000).
_OVERVIEW_POST_LIMIT = 1800
# Total budget (chars) for the session documents that feed the AI recap.
# The full session log(s) are the authoritative source for the recap — this
# exists only to keep the prompt size bounded for very long sessions; each
# included document keeps its header + full text (trimmed from the tail if it
# alone exceeds the whole budget).
_OVERVIEW_DOC_BUDGET = 60_000
# Discord message cap is 2000 chars; keep view messages under it.
_VIEW_MSG_LIMIT = 1900
# How many recent notes to show (in full) in /session_info.
_INFO_NOTE_LIMIT = 10

# ── /session_notes document uploads ──────────────────────────────────────────
#: Text-document extensions accepted by ``/session_notes`` with a file
#: attached. Deliberately limited to plain text / markdown (what was asked for);
#: everything else is rejected before it is read or stored.
_SESSION_DOC_EXTENSIONS = {".txt", ".md"}
#: Max size for a session-notes document upload (bytes).
_SESSION_DOC_MAX_BYTES = 2 * 1024 * 1024


def _merge_prompt() -> str:
    """System prompt for the /end_session AI session-log merge.

    Customizable via SESSION_MERGE_PROMPT in .env (see config/settings.py);
    falls back to the built-in default when unset/empty.
    """
    return SESSION_MERGE_PROMPT.strip() or DEFAULT_SESSION_MERGE_PROMPT


def _recap_prompt() -> str:
    """System prompt for the /end_session recap of the session.

    The recap (what happened + suggested next steps) is stored with the
    session and delivered when the NEXT session starts.

    Customizable via SESSION_RECAP_PROMPT in .env (see config/settings.py);
    falls back to the built-in default when unset/empty.
    """
    return SESSION_RECAP_PROMPT.strip() or DEFAULT_SESSION_RECAP_PROMPT


# ── Overview source assembly ────────────────────────────────────────────

def _session_documents_text(session: dict[str, Any]) -> tuple[str, list[str]]:
    """Full text of the session's own documents (attachments + transcripts).

    Returns ``(text, refs)`` where *refs* are ``<subdir>/<filename>`` labels
    (one per non-empty document found).  *text* is the concatenation of those
    documents (each with a small ``Source:`` header so the model can tell them
    apart), trimmed to ``_OVERVIEW_DOC_BUDGET`` chars in total.  When the
    budget is exhausted the remaining documents are dropped with a note so the
    model knows more material exists for the session.
    """
    from bot_core import sessions as S

    parts: list[str] = []
    refs: list[str] = []
    omitted: list[str] = []
    budget_left = _OVERVIEW_DOC_BUDGET
    for subdir in (S._SUBDIR_ATTACHMENTS, S._SUBDIR_TRANSCRIPTS):
        for name in S._session_docs(session, subdir):
            ref = f"{subdir}/{name}"
            path = S._session_dir(session) / ref
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


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_bot() -> discord.Client | None:
    """Running bot reference with side-effect-free resolution.

    Delegates to :func:`bot_core.channel_delivery.get_bot`, which reads the
    already-loaded ``__main__``/``main`` module from ``sys.modules`` and logs
    a real reason when no logged-in bot is found.  (A lazy ``from main
    import bot`` here was dangerous: if run inside a script process it would
    re-execute main.py's module level and return a fresh, never-logged-in
    duplicate — see the get_bot docstring.)
    """
    from bot_core.channel_delivery import get_bot
    return get_bot()


def _resolve_overview_model(guild_id: int | None, channel_id: int | None) -> str:
    """Model for the AI overview: active character's model, else DEFAULT_MODEL."""
    char = get_character(get_active_char_key(guild_id, channel_id))
    model = (char.model if (char and char.model) else "").strip()
    return model or DEFAULT_MODEL or ""


def _fallback_recap(session: dict[str, Any]) -> str:
    """Plain-text recap used at /end_session when the AI backend is unavailable.

    Lists the last few notes so an ended session is never left without any
    record; a note to that effect when there are no notes either.
    """
    notes = session.get("notes", [])
    if notes:
        lines = ["AI recap unavailable (backend error) — notes recorded during this session:"]
        lines += [f"- {text}" for _ts, text in notes[-5:]]
        return "\n".join(lines)
    return "(AI recap unavailable and no notes were recorded for this session)"


def _plain_digest(session: dict[str, Any]) -> str:
    """Plain-text digest of a previous session WITHOUT a stored AI recap.

    Used at /start_session (no AI call there): lists the last few notes so a
    stale auto-ended session still gets a short record.  Returns "" when the
    session has no notes to show.
    """
    notes = session.get("notes", [])
    if not notes:
        return ""
    lines = ["No AI recap available for this session — last notes recorded:"]
    lines += [f"- {text}" for _ts, text in notes[-5:]]
    return "\n".join(lines)


async def _generate_recap(
    session: dict[str, Any], guild_id: int | None, channel_id: int | None,
) -> str:
    """Very brief AI recap of the session, generated at /end_session.

    Sources (strongest first): the session's own documents (attachments/ and
    transcripts/) then its notes.  The prompt instructs the model to write it
    in the language of the sources — no hard-coded language detection on this
    side (see DEFAULT_SESSION_RECAP_PROMPT in config/settings.py).

    Falls back to :func:`_fallback_recap` when no model is configured or the
    request fails — /end_session must always produce a recap.
    """
    docs_text, doc_names = _session_documents_text(session)
    notes = session.get("notes", [])
    if not doc_names and not notes:
        return ""  # nothing recorded — no recap to store

    model = await _validate_model(_make_client(), _resolve_overview_model(guild_id, channel_id))
    if not model:
        log.warning("No model available for session recap — using plain-text fallback")
        return _fallback_recap(session)

    notes_text = "\n".join(f"- {text}" for _ts, text in notes[-50:]) or "(no notes)"
    source_listing = ", ".join(doc_names) if doc_names else "(none)"
    log.info(
        "Session recap for %r: %d document(s) [%s], %d chars docs",
        session.get("name"), len(doc_names), source_listing, len(docs_text),
    )

    user_content = (
        f"Session name: {session.get('name') or 'Untitled'}\n"
        f"Session files included: {source_listing}\n\n"
        f"## Session files (authoritative — the session's own record)\n"
        f"{docs_text or '(no documents for this session)'}\n\n"
        f"## Session notes\n{notes_text}"
    )

    client = _make_client()
    try:
        # The stored session digest is accuracy-critical (#20): it stays at
        # the model default (max effort) unless RECAP_REASONING_EFFORT lowers
        # it — latency matters less than getting the recap right.
        summary = await complete_text(
            client,
            model=model,
            messages=[
                {"role": "system", "content": _recap_prompt()},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=max(0, SUMMARY_CALL_MAX_TOKENS) or None,
            reasoning_effort=RECAP_REASONING_EFFORT,
        )
        if not summary.strip():
            raise ValueError("empty recap")
        return summary.strip()
    except Exception as e:
        log.error("Session recap AI request failed (%s): %s", model, e)
        return _fallback_recap(session)


async def _generate_merged_log(
    session: dict[str, Any], guild_id: int | None, channel_id: int | None,
) -> str | None:
    """AI-merged canonical session log from the session's own documents.

    Every player log / transcript of the session covers the same chronological
    events with heavy overlap.  The merge combines them into ONE complete,
    well-formatted, de-duplicated session log — that single text replaces the
    mechanical full-text copy as the session's RAG content (see
    ``sessions._session_file_content``).

    Returns None when there are no documents or the AI is unavailable — the
    caller then keeps the mechanical fallback so document content stays
    reachable via RAG even with a dead backend.
    """
    docs_text, doc_names = _session_documents_text(session)
    if not doc_names:
        return None
    model = await _validate_model(_make_client(), _resolve_overview_model(guild_id, channel_id))
    if not model:
        log.warning("No model available for session-log merge — keeping mechanical document copy")
        return None

    source_listing = ", ".join(doc_names)
    log.info(
        "Merging session log for %r: %d document(s) [%s], %d chars docs",
        session.get("name"), len(doc_names), source_listing, len(docs_text),
    )

    user_content = (
        f"Session name: {session.get('name') or 'Untitled'}\n"
        f"Session files included: {source_listing}\n\n"
        f"## Session files (the only sources — one log per player)\n{docs_text}"
    )

    client = _make_client()
    try:
        # Model-max output (max_tokens omitted): the merged log is a full
        # narrative record and benefits from the model's whole output capacity.
        # Reasoning stays at the model default here — long-form merging quality
        # matters more than latency, and /end_session already defers (#9).
        # MERGE_LOG_REASONING_EFFORT (#20) can dial it down; when unset (None)
        # the param is omitted entirely. complete_text runs streamed internally,
        # so AI_TIMEOUT_S is an idle watchdog even for this long
        # thinking+generation call (no total-time cap).
        merged = await complete_text(
            client,
            model=model,
            messages=[
                {"role": "system", "content": _merge_prompt()},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            reasoning_effort=MERGE_LOG_REASONING_EFFORT,
        )
        if not merged.strip():
            raise ValueError("empty merged log")
        return merged.strip()
    except Exception as e:
        log.error("Session-log merge AI request failed (%s): %s", model, e)
        return None


def _truncate(text: str, limit: int = _OVERVIEW_POST_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n…(truncated — full overview in the session file)"


# ── /start_session ───────────────────────────────────────────────────────

async def handle_start_session(interaction: discord.Interaction, name: str | None = None) -> None:
    """Start a new global session (with the once-per-hour safety rule)."""
    from bot_core import sessions as S

    try:
        session, closed_info = S.start_session(name=name)
    except ValueError as e:
        await interaction.response.send_message(f"⚠️ {e}")
        return

    # Defer before any AI/network work; all output goes through followup.
    await interaction.response.defer()

    parts = [
        f"✅ **Session started:** {session.get('name') or '(untitled)'}",
        f"📄 Notes file: `{pathlib.Path(session['file']).name}` (editable on disk, RAG-enabled)",
    ]

    if closed_info and closed_info["kind"] == "stale":
        parts.append("♻️ The previous session was older than 12 h — it was auto-ended without an AI recap.")

    # Deliver the stored recap of the previous session (what happened + next
    # steps) — generated at ITS /end_session, so no AI call happens here.
    # A stale auto-end has no stored recap: fall back to a plain digest of
    # its last notes so it is not left without any record.
    if closed_info:
        prev = closed_info["session"]
        text = (prev.get("overview") or _plain_digest(prev)).strip()
        if text:
            await interaction.followup.send(
                f"🔁 **Previous session recap** ({prev.get('name') or 'untitled'})\n\n"
                f"{_truncate(text)}"
            )
            # Mark delivered only AFTER Discord accepted the message — a
            # failed send must not lose the stored recap (a later
            # /start_session will retry delivery).
            if closed_info["kind"] == "manual":
                S.mark_overview_delivered(prev)

    # Deliver queued next-session reminders to their channels.
    bot = _get_bot()
    if bot is not None:
        try:
            n_rem = await S.deliver_queued_reminders(bot)
            if n_rem:
                parts.append(f"⏰ Delivered {n_rem} queued next-session reminder(s) to their channel(s).")
        except Exception as e:
            log.error("Next-session reminder delivery failed: %s", e)

    await interaction.followup.send("\n".join(parts))


# ── /end_session ─────────────────────────────────────────────────────────

async def handle_end_session(interaction: discord.Interaction, name: str | None = None) -> None:
    """End the current session and write its AI recap (what happened + next steps)."""
    from bot_core import sessions as S

    session = S.get_current_session()
    if session is None:
        await interaction.response.send_message(
            "⚠️ There is no active session to end. Start one with `/start_session`."
        )
        return

    # Defer first — the AI recap call can exceed Discord's 15 s window.
    await interaction.response.defer()

    channel_id = interaction.channel_id

    # Pick up any manual edits to the notes file before summarizing.
    S.refresh_notes_from_disk(session)

    recap = await _generate_recap(session, interaction.guild_id, channel_id)
    merged_log = await _generate_merged_log(session, interaction.guild_id, channel_id)
    ended = S.end_session(overview=recap or None, name=name, merged_log=merged_log)
    if ended is None:  # defensive — should not happen
        await interaction.followup.send("⚠️ Could not end the session.")
        return

    fname = pathlib.Path(ended["file"]).name
    merge_line = ("\n📜 Session log combined from all uploads."
                  if merged_log else "")
    recap_block = f"\n\n{_truncate(recap)}" if recap else "\n(no recap — nothing was recorded for this session)"
    await interaction.followup.send(
        f"🔚 **Session ended:** {ended.get('name') or '(untitled)'}\n"
        f"📄 Recap saved to `{fname}`.{merge_line}{recap_block}"
    )


# ── /remind_next_session ─────────────────────────────────────────────────

async def handle_remind_next_session(interaction: discord.Interaction, message: str) -> None:
    """Queue a reminder for the NEXT session start; starts one if none is active."""
    from bot_core import sessions as S

    # Fail fast: this reminder would be delivered to THIS channel at the next
    # session start — refuse up front if we cannot post here.
    from bot_core.channel_delivery import can_post_in_channel
    if not await can_post_in_channel(interaction):
        await interaction.response.send_message(
            "⚠️ I can't send messages in this channel (missing **View Channel** / "
            "**Send Messages**) — the reminder could never be delivered here. "
            "Queue it in a channel I can post to, or give me access first.",
            
        )
        return

    # A slash command always arrives in a channel; the guard only satisfies
    # mypy (interaction.channel is typed as ... | None).
    if interaction.channel is None:
        await interaction.response.send_message("⚠️ This command needs to run in a channel.")
        return
    entry = S.queue_next_session_reminder(interaction.channel.id, message)
    queued = len(S.list_queued_reminders())

    active = S.get_current_session()
    if active is None:
        try:
            S.start_session()
            start_note = "No session was active — a new one has been started."
        except ValueError as e:
            start_note = f"No session was active, but starting one failed: {e}"
    else:
        start_note = "A session is currently active — this reminder waits for the NEXT session."

    await interaction.response.send_message(
        f"📌 **Next-session reminder queued** ({queued} in queue):\n"
        f"“{entry['message']}”\n"
        f"{start_note}\n"
        f"It will be delivered here when the next session starts.",
        
    )


# ── /session_notes ───────────────────────────────────────────────────────

async def handle_session_notes(
    interaction: discord.Interaction,
    note: str | None = None,
    file: discord.Attachment | None = None,
) -> None:
    """Add a note (or an uploaded ``.txt`` / ``.md`` document) to the current session.

    There is no view action any more — ``/session_info`` shows what a session
    contains (notes, documents, transcripts).
    """
    from bot_core import sessions as S

    # A document upload wins over an inline note: it is the richer action
    # and keeps a single, unambiguous result.
    if file is not None:
        await _add_document_from_attachment(interaction, file)
        return

    if not note or not note.strip():
        await interaction.response.send_message(
            "⚠️ Provide a `note` or attach a `.txt`/`.md` file — e.g. "
            "`/session_notes note: \"remember the API key\"`. "
            "Use `/session_info` to view session contents.",
        )
        return

    session = S.get_current_session()
    if session is None:
        await interaction.response.send_message(
            "⚠️ There is no active session to add notes to. Start one with `/start_session` first.",
        )
        return
    author = (interaction.user.display_name or "").strip() if interaction.user else ""
    updated = S.add_note(note, author=author)
    if updated is None:
        await interaction.response.send_message("⚠️ Could not add the note.")
        return
    n = len(updated.get("notes", []))
    await interaction.response.send_message(
        f"📝 Note added to session **{updated.get('name') or '(untitled)'}** ({n} note(s) total).\n"
        f"📄 `{pathlib.Path(updated['file']).name}`",
    )


async def _add_document_from_attachment(interaction: discord.Interaction, file: discord.Attachment) -> None:
    """Store an uploaded ``.txt``/``.md`` file in the active session.

    Reads the attachment, validates its type and size, decodes it as UTF-8 text
    and hands it to ``sessions.add_document`` — which saves it as a raw
    ``.md`` file under the session's hidden ``.attachments/`` folder (one file
    per upload, never indexed directly) and re-renders the combined
    ``notes.md`` with its full text.  Any validation failure is reported back
    to the user and the session is left untouched.
    """
    from bot_core import sessions as S

    session = S.get_current_session()
    if session is None:
        await interaction.response.send_message(
            "⚠️ There is no active session to add a document to. Start one with `/start_session` first.",
            
        )
        return

    filename = (file.filename or "").strip()
    ext = pathlib.PurePosixPath(filename).suffix.lower()
    if ext not in _SESSION_DOC_EXTENSIONS:
        allowed = ", ".join(sorted(_SESSION_DOC_EXTENSIONS))
        await interaction.response.send_message(
            f"⚠️ Only text documents can be added to session notes ({allowed}) — got `{ext or 'no extension'}`. "
            f"Use `note:` for free text.",
            
        )
        return

    if file.size is not None and file.size > _SESSION_DOC_MAX_BYTES:
        await interaction.response.send_message(
            f"⚠️ Document too large: {file.size:,} bytes (max {_SESSION_DOC_MAX_BYTES:,}). "
            f"Trim it or upload in smaller parts.",
            
        )
        return

    try:
        data = await file.read()
    except Exception as e:
        log.error("Failed to read session-notes document %r: %s", filename, e)
        await interaction.response.send_message(f"⚠️ Could not read `{filename or 'attachment'}`: {e.__class__.__name__}.")
        return

    if not data:
        await interaction.response.send_message(f"⚠️ `{filename or 'attachment'}` is empty — nothing to add.")
        return
    if len(data) > _SESSION_DOC_MAX_BYTES:
        await interaction.response.send_message(
            f"⚠️ Document too large: {len(data):,} bytes (max {_SESSION_DOC_MAX_BYTES:,}). "
            f"Trim it or upload in smaller parts.",
        )
        return

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        await interaction.response.send_message(
            f"⚠️ `{filename or 'attachment'}` is not valid UTF-8 text. "
            f"Only plain-text documents (`.txt` / `.md`) can be added to session notes.",
            
        )
        return

    if not text.strip():
        await interaction.response.send_message(f"⚠️ `{filename or 'attachment'}` is empty — nothing to add.")
        return

    title = filename or "document"
    updated, n = S.add_document(text, title=title, session=session)
    if updated is None or n == 0:
        await interaction.response.send_message("⚠️ Could not add the document.")
        return

    # The raw document lives in the hidden .attachments/ folder; its full
    # text is combined into notes.md — the session's single RAG document.
    # NOTE: we deliberately do NOT show the stored filename — picking it as
    # "newest file in the dir" raced with concurrent uploads and named the
    # wrong file (audit L3). add_document() does not return the path.
    from bot_core import sessions as _S
    await interaction.response.send_message(
        f"📎 Document **{title}** added to session **{updated.get('name') or '(untitled)'}**\n"
        f"📄 Saved under `{_S._SUBDIR_ATTACHMENTS}/` (kept on disk; full text combined "
        f"into the session notes file, which is RAG-enabled).",
    )
    return


def _chunk_display(text: str, limit: int = _VIEW_MSG_LIMIT) -> list[str]:
    """Word-wrap *text* into Discord-safe message pieces of at most *limit* chars.

    Notes are stored whole (no pre-chunking); they are split into readable
    messages ONLY at display time, here.  A long note simply continues on the
    next message instead of being cut off.  A single word longer than *limit*
    still forms its own (unavoidable) piece.
    """
    parts: list[str] = []
    cur = ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > limit:
            parts.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        parts.append(cur)
    return [p for p in parts if p]


def _format_size(num_bytes: int) -> str:
    """Human-readable file size for the /session_info document listing."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _build_info_parts(session: dict[str, Any]) -> list[str]:
    """Build the display-only /session_info messages for one session.

    Shows the session's identity/timing, its notes (last few in full,
    word-wrapped across messages when long) and its raw documents
    (attachments / transcripts) listed with name and size — their full text
    is combined into notes.md, the session's single RAG document.
    """
    from bot_core import sessions as S

    status = "**current**" if S.get_current_session() else "**last (ended)**"
    lines = [f"ℹ️ **Session info — {session.get('name') or '(untitled)'}** ({status})"]

    started = datetime.fromtimestamp(session["started_at"]).strftime("%Y-%m-%d %H:%M")
    timing = f"🕒 Started: {started}"
    if session.get("ended_at"):
        ended = datetime.fromtimestamp(session["ended_at"]).strftime("%Y-%m-%d %H:%M")
        elapsed = session["ended_at"] - session["started_at"]
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        timing += f" — ended: {ended} (lasted {h:d}:{m:02d}:{s:02d})"
    lines.append(timing)

    if session.get("overview"):
        lines.append(f"📄 Recap: “{_truncate(session['overview'], 300)}”")
    lines.append("")

    notes = S.get_notes(session)
    lines.append(f"**📝 Notes ({len(notes)})**")
    if not notes:
        lines.append("(no notes yet — add one with `/session_notes note: \"...\"`)")
    else:
        shown = notes[-_INFO_NOTE_LIMIT:]
        for _ts, text in shown:
            lines.append(f"- {text}")
        if len(notes) > len(shown):
            lines.append(f"… and {len(notes) - len(shown)} earlier note(s).")

    for subdir, heading in ((S._SUBDIR_ATTACHMENTS, "📎 Attachments"),
                            (S._SUBDIR_TRANSCRIPTS, "🎙️ Transcripts")):
        docs = S._session_docs(session, subdir)
        if docs:
            lines.append("")
            lines.append(f"**{heading} ({len(docs)})**")
            for name in docs:
                size = "?"
                try:
                    size = _format_size((S._session_dir(session) / subdir / name).stat().st_size)
                except OSError:
                    pass
                lines.append(f"- `{name}` — {size}")

    lines.append("")
    lines.append(f"📄 Folder: `{pathlib.Path(session['dir']).name}/` (notes.md = single RAG document; "
                 f"raw files kept in hidden folders)")
    lines.append("(long notes are split across messages; full text is in the session folder)")

    return _chunk_display("\n".join(lines))


async def handle_session_info(interaction: discord.Interaction) -> None:
    """Show info about the current session (or the last ended one)."""
    from bot_core import sessions as S

    session = S.get_current_session() or S.get_last_session()
    if session is None:
        await interaction.response.send_message(
            "ℹ️ No sessions yet. Start one with `/start_session`."
        )
        return

    # Pick up manual edits to the notes file (re-parses real notes).
    S.refresh_notes_from_disk(session)

    # Notes are stored whole and only split at display time — so a long note
    # needs several messages.  Send them sequentially (no defer needed: the
    # view is local and fast, well within the 15 s response window).
    parts = _build_info_parts(session)
    if not parts:
        await interaction.response.send_message("ℹ️ No notes yet.")
        return
    # First part via the initial response (no defer needed: the view is local
    # and fast). Later parts are best-effort: one failed followup used to blow
    # up the whole loop unhandled, losing every remaining part (audit L7).
    await interaction.response.send_message(parts[0])
    failed_at = -1
    for i, part in enumerate(parts[1:], start=1):
        try:
            await interaction.followup.send(part)
        except discord.DiscordException as e:
            log.warning("session-notes view: part %d/%d failed to deliver: %s",
                        i + 1, len(parts), e)
            failed_at = i
            break
    if failed_at > 0:
        missing = len(parts) - failed_at
        try:
            await interaction.followup.send(
                f"⚠️ Delivery interrupted — {missing} of {len(parts)} part(s) "
                f"could not be sent."
            )
        except discord.DiscordException:
            pass  # nothing left to try; the warning above is for the logs
