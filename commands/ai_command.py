"""Handler for the /ai slash command.

This module is a helper — it does NOT import discord directly.  Instead, main.py
calls ``handle_ai_command(interaction, message, character)`` with whatever Discord
interaction object it receives.
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("bot.commands.ai")


async def handle_ai_command(
    interaction,                             # any Discord Interaction type
    message: str,
    character_name: str | None = None,
) -> None:
    """Core handler for ``/ai``.

    Flow (mirrors the old inline ai_command in main.py):
      1. Resolve active character (via config.characters).
      2. Defer slash-com response + start typing loop.
      3. Call bot_core.ask_ai() with Character.config.model_slug.
      4. Send chunked reply via utils.response_splitter.
    """

    # ── 1. Resolve character / model slug ─────────────────────────
    from config.characters import CHARACTER_CHOICES, get_character, default_character

    char_key = (
        character_name
        if character_name is not None
            and any(c.key == character_name for c in CHARACTER_CHOICES)
        else _get_active_char(interaction)  # per-guild/channel lookup
    )

    char_obj = get_character(char_key)
    if char_obj is None:
        await interaction.followup.send(f"Character ``{character_name}`` not found.", ephemeral=True)
        return

    model_slug = char_obj.model or ""

    # ── 2. Defer (avoids "Interaction has already been responded to") ─
    try:
        await interaction.response.defer()
    except Exception:
        pass  # stale slash — no harm continuing (typing loop will fail safely)

    # ── 3. Start typing indicator (background) ────────────────────
    if hasattr(interaction, "channel") and interaction.channel is not None:
        from utils.typing_loop import typing_loop_task
        asyncio.create_task(typing_loop_task(interaction.channel))

    # ── 4. Ask AI ────────────────────────────────────────────────
    from bot_core import ask_ai

    reply_text, _extra = await ask_ai(
        user_message=message,
        model_slug=model_slug,
        guild_id=interaction.guild_id or 0,
        channel_id=interaction.channel_id,
        username=(getattr(interaction, "user", None)
                      and getattr(getattr(interaction, "user"), "display_name", "")
                  ) or "",
    )

    # ── 5. Send chunked reply ────────────────────────────────────
    from utils.response_splitter import send_long_response

    try:
        await send_long_response(interaction, reply_text, str(char_obj.display))
    except Exception as exc:
        log.error("Failed to send AI response: %s", exc)


# ── Small helpers ────────────────────────────────────────────────

_ACTIVE_CHARS: dict[tuple[int, int], str] = {}   # (guild_id, channel_id) → char_key


def _get_active_char(interaction) -> str:
    guild_id = getattr(interaction, "guild_id", None) or 0
    channel_id = getattr(interaction, "channel_id", 0)
    return _ACTIVE_CHARS.get((guild_id, channel_id))
