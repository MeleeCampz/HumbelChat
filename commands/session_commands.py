"""Session slash commands — /start_session, /end_session,
/remind_next_session and /session_notes.

Delegates all state handling to ``bot_core.sessions``; this module only
translates between Discord interactions and the session store, plus the
AI-generated end-of-session overview (same model-resolution pattern as
/summarize in commands/utility_commands.py).
"""
from __future__ import annotations

import logging
import pathlib

import discord

from bot_core.ai_client import _make_client, _validate_model
from bot_core.history import get_active_char_key, get_history
from config.characters import get_character
from config.settings import (
    DEFAULT_MODEL,
    DEFAULT_SESSION_SUMMARY_PROMPT,
    SESSION_SUMMARY_PROMPT,
)

log = logging.getLogger("bot.session_commands")

# Max chars of the overview posted to the channel (Discord limit is 2000).
_OVERVIEW_POST_LIMIT = 1800
# How many recent chat messages feed the AI overview.
_OVERVIEW_HISTORY_MESSAGES = 30


def _summary_prompt() -> str:
    """System prompt for the /end_session AI overview.

    Customizable via SESSION_SUMMARY_PROMPT in .env (see config/settings.py);
    falls back to the built-in default when unset/empty.
    """
    return SESSION_SUMMARY_PROMPT.strip() or DEFAULT_SESSION_SUMMARY_PROMPT


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_bot():
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


def _resolve_overview_model(guild_id: int | None, channel_id: int) -> str:
    """Model for the AI overview: active character's model, else DEFAULT_MODEL."""
    char = get_character(get_active_char_key(guild_id, channel_id))
    model = (char.model if (char and char.model) else "").strip()
    return model or DEFAULT_MODEL or ""


def _fallback_overview(session: dict, notes: list[list]) -> str:
    """Plain-text overview used when the AI backend is unavailable."""
    from datetime import datetime
    started = datetime.fromtimestamp(session["started_at"]).strftime("%Y-%m-%d %H:%M")
    ended = (datetime.fromtimestamp(session.get("ended_at") or session["started_at"])
             .strftime("%Y-%m-%d %H:%M"))
    lines = [
        f"Session **{session.get('name') or 'untitled'}** — {started} → {ended}",
        "",
        "AI overview unavailable (backend error) — notes recorded in this session:",
    ]
    if notes:
        for ts, text in notes[-30:]:
            lines.append(f"- {text}")
    else:
        lines.append("- (no notes were added)")
    return "\n".join(lines)


async def _generate_overview(session: dict, guild_id: int | None, channel_id: int) -> str:
    """AI overview of the session from its notes + recent chat in this channel.

    Falls back to a plain-text note listing when no model is configured or
    the request fails — /end_session must always produce an overview.
    """
    notes = session.get("notes", [])
    notes_text = "\n".join(f"- {text}" for _ts, text in notes[-50:]) or "(no notes)"

    hist = get_history(guild_id if guild_id is not None else 0, channel_id)
    chat_lines = []
    for m in hist[-_OVERVIEW_HISTORY_MESSAGES:]:
        role = {"user": "User", "assistant": "AI"}.get(m.get("role"), m.get("role"))
        chat_lines.append(f"[{role}]: {m.get('content', '')}")
    chat_text = "\n\n".join(chat_lines) or "(no recent chat in this channel)"

    model = await _validate_model(_make_client(), _resolve_overview_model(guild_id, channel_id))
    if not model:
        log.warning("No model available for session overview — using plain-text fallback")
        return _fallback_overview(session, notes)

    client = _make_client()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": _summary_prompt(),
                },
                {
                    "role": "user",
                    "content": (
                        f"Session name: {session.get('name') or 'untitled'}\n\n"
                        f"## Session notes\n{notes_text}\n\n"
                        f"## Recent chat (channel where the session was ended)\n{chat_text}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        summary = resp.choices[0].message.content or ""
        if not summary.strip():
            raise ValueError("empty overview")
        return summary.strip()
    except Exception as e:
        log.error("Session overview AI request failed (%s): %s", model, e)
        return _fallback_overview(session, notes)


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
        parts.append("♻️ The previous session was older than 12 h — it was auto-ended without an AI overview.")

    # Deliver the explicitly-ended previous session's overview (if any).
    if closed_info and closed_info["kind"] == "manual":
        prev = closed_info["session"]
        if prev.get("overview"):
            await interaction.followup.send(
                f"📄 **Overview of previous session** ({prev.get('name') or 'untitled'}):\n\n"
                f"{_truncate(prev['overview'])}",
            )

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
    """End the current session and write its AI overview."""
    from bot_core import sessions as S

    session = S.get_current_session()
    if session is None:
        await interaction.response.send_message(
            "⚠️ There is no active session to end. Start one with `/start_session`.",
            
        )
        return

    # Defer first — the AI overview call can exceed Discord's 15 s window.
    await interaction.response.defer()

    guild_id = interaction.guild_id or 0
    channel_id = interaction.channel_id

    # Pick up any manual edits to the notes file before summarizing.
    S.refresh_notes_from_disk(session)

    overview = await _generate_overview(session, interaction.guild_id, channel_id)
    ended = S.end_session(overview=overview, name=name)
    if ended is None:  # defensive — should not happen
        await interaction.followup.send("⚠️ Could not end the session.")
        return

    fname = pathlib.Path(ended["file"]).name
    await interaction.followup.send(
        f"🔚 **Session ended:** {ended.get('name') or '(untitled)'}\n"
        f"📄 Overview saved to `{fname}`.\n\n"
        f"{_truncate(overview)}",
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
    action: str = "view",
    note: str | None = None,
) -> None:
    """Add a note to the current session or view notes (current/last session)."""
    from bot_core import sessions as S

    act = (action or "view").strip().lower()

    if act == "add":
        if not note or not note.strip():
            await interaction.response.send_message(
                "⚠️ Please provide a note, e.g. `/session_notes action: add note: \"remember the API key\"`.",
                
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
        return

    if act != "view":
        await interaction.response.send_message(
            f"⚠️ Unknown action ``{action}``. Use `add` or `view`."
        )
        return

    # view — current session if active, else the most recent one.
    session = S.get_current_session() or S.get_last_session()
    if session is None:
        await interaction.response.send_message(
            "ℹ️ No sessions yet. Start one with `/start_session`."
        )
        return

    notes = S.refresh_notes_from_disk(session)
    status = "**current**" if S.get_current_session() else "**last (ended)**"
    lines = [f"📝 **Session notes — {session.get('name') or '(untitled)'}** ({status})"]
    if session.get("overview"):
        lines.append(f"📄 Overview: “{_truncate(session['overview'], 300)}”")
    lines.append("")
    if not notes:
        lines.append("(no notes yet — add one with `/session_notes action: add note: \"...\"`)")
    else:
        shown = notes[-25:]
        for _ts, text in shown:
            lines.append(f"- {text}")
        if len(notes) > len(shown):
            lines.append(f"… and {len(notes) - len(shown)} earlier note(s).")
    lines.append(f"\n📄 File: `{pathlib.Path(session['file']).name}` (editable on disk)")

    body = "\n".join(lines)
    if len(body) > 1900:
        body = body[:1900].rstrip() + "\n…(truncated — see the file for the full notes)"
    await interaction.response.send_message(body)
