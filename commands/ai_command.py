"""AI chat slash command handler.

P2-4: Uses streaming responses when available so the user sees text
appear progressively instead of waiting for the full reply.
Set AI_STREAMING=0 in .env to disable and use the classic non-streaming path.
"""
from __future__ import annotations

import asyncio
import logging
import os

from config.characters import get_character, default_character
from bot_core.history import get_active_char_key

log = logging.getLogger("bot.commands.ai_command")

def _streaming_enabled() -> bool:
    """Check if streaming is enabled (read at call time for testability)."""
    return os.getenv("AI_STREAMING", "1") not in ("0", "false", "no")


async def handle_ai_command(
    interaction: "discord.Interaction",
    message: str,
    character_name: str | None = None,
) -> None:
    """Core handler for /ai."""
    # 1. Immediate deferral to prevent "Application did not respond"
    try:
        if not interaction.response.is_done():
            await interaction.response.defer()
    except Exception as e:
        log.error("Error deferring interaction: %s", e)
        return

    # 2. Resolve character
    char_key = character_name
    if char_key is None:
        char_key = get_active_char_key(interaction.guild_id, interaction.channel_id)

    char_obj = get_character(char_key)
    if char_obj is None:
        await interaction.followup.send(f"Character `{character_name}` not found.", ephemeral=True)
        return

    model_slug = char_obj.model or ""
    user_id = getattr(getattr(interaction, "user", None), "id", None)
    username = getattr(getattr(interaction, "user", None), "display_name", "") or ""
    guild_id = interaction.guild_id or 0
    channel_id = interaction.channel_id

    # 3. Start typing indicator (visible while waiting for the first chunk)
    typing_task = None
    if hasattr(interaction, "channel") and interaction.channel is not None:
        try:
            from utils.typing_loop import typing_loop_task

            typing_task = asyncio.create_task(typing_loop_task(interaction.channel))
            bot_ref = getattr(interaction, "bot", None)
            if bot_ref is not None:
                tasks = getattr(bot_ref, "typing_tasks", None)
                if isinstance(tasks, list):
                    tasks.append(typing_task)
        except Exception as e:
            log.warning("Typing loop error: %s", e)

    try:
        if _streaming_enabled():
            # ── P2-4: Streaming path ────────────────────────────────
            from bot_core.ai_client import ask_ai_stream
            from utils.stream_response import stream_ai_response

            await stream_ai_response(
                interaction,
                ask_ai_stream(
                    user_message=message,
                    model_slug=model_slug,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    username=username,
                    user_id=user_id,
                ),
            )
        else:
            # ── Non-streaming path ──────────────────────────────────
            from bot_core.ai_client import ask_ai
            from utils.response_splitter import send_long_response

            reply_text, _extra = await ask_ai(
                user_message=message,
                model_slug=model_slug,
                guild_id=guild_id,
                channel_id=channel_id,
                username=username,
                user_id=user_id,
            )
            await send_long_response(interaction, reply_text, str(char_obj.display))

    except Exception as e:
        log.error("AI request failed: %s", e)
        # Friendly error — rate limit, input too long, model missing, etc.
        await _safe_followup(interaction, f"❌ {e}")
        return

    if typing_task is not None:
        typing_task.cancel()


async def _safe_followup(interaction, text: str) -> None:
    """Send a followup, tolerating already-responded interactions."""
    try:
        if hasattr(interaction, "followup") and not interaction.response.is_done():
            await interaction.response.send_message(text, ephemeral=True)
        else:
            await interaction.followup.send(text, ephemeral=True)
    except Exception as e:
        log.warning("Follow-up send failed: %s", e)
