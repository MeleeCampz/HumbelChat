"""AI chat slash command handler.

P2-4: Uses streaming responses when available so the user sees text
appear progressively instead of waiting for the full reply.
Set AI_STREAMING=0 in .env to disable and use the classic non-streaming path.

Beyond20-style embeds: on the NON-STREAMING path, replies are rendered as
Discord embeds (title + description + inline fields) when EMBED_FORMAT is
enabled.  Streaming stays plain text by design — a frozen embed cannot grow
via edits, and live typing is the better UX for streamed replies.
"""
from __future__ import annotations

import logging
import os

import config.settings as _settings
from config.characters import get_character
from bot_core import ai_client
from bot_core.history import get_active_char_key
from utils.background_tasks import spawn_tracked_task
from utils.channel_queue import channel_slot
from utils.response_splitter import send_long_response, send_long_response_embedded
from utils.stream_response import stream_ai_response
from utils.typing_loop import typing_loop_task

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
        await interaction.followup.send(f"Character `{character_name}` not found.")
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
            # §4.4: keep a strong reference to the typing task so it cannot
            # be garbage-collected before completing.
            typing_task = spawn_tracked_task(
                typing_loop_task(interaction.channel),
                name=f"typing-{channel_id}",
            )
            # Retain a per-bot reference for diagnostics / cleanup too.
            bot_ref = getattr(interaction, "bot", None)
            if bot_ref is not None:
                tasks = getattr(bot_ref, "typing_tasks", None)
                if not isinstance(tasks, list):
                    tasks = []
                    setattr(bot_ref, "typing_tasks", tasks)
                # Prune finished tasks so the diagnostics list can't grow
                # unbounded on long uptimes.
                bot_ref.typing_tasks = [t for t in tasks if not t.done()]
                bot_ref.typing_tasks.append(typing_task)
        except Exception as e:
            log.warning("Typing loop error: %s", e)

    # Hold this channel's reply slot for the ENTIRE request + delivery.
    # Without this, a second /ai in the same channel can interleave its
    # messages with this one (Discord orders by send time, not request
    # time) — e.g. the overflow part of a long reply lands after the next
    # user's request. See utils/channel_queue.py.
    async with channel_slot(channel_id, name="ai-command"):
        try:
            if _streaming_enabled():
                # ── P2-4: Streaming path ────────────────────────────────
                await stream_ai_response(
                    interaction,
                    ai_client.ask_ai_stream(
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
                reply_text, _extra = await ai_client.ask_ai(
                    user_message=message,
                    model_slug=model_slug,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    username=username,
                    user_id=user_id,
                )
                if _settings.EMBED_FORMAT:
                    # Beyond20-style embed delivery.  Returns False (having
                    # sent nothing) when the reply is too small to benefit
                    # from an embed or a Discord API error occurs — in both
                    # cases fall back to the classic plain-text chunks so the
                    # user always gets an answer.
                    delivered = await send_long_response_embedded(
                        interaction, reply_text, str(char_obj.display)
                    )
                    if not delivered:
                        await send_long_response(interaction, reply_text, str(char_obj.display))
                else:
                    await send_long_response(interaction, reply_text, str(char_obj.display))

        except ValueError as e:
            # §3.7: classified errors (timeout / model-not-found / backend-down /
            # rate-limit / input too long) surface a user-friendly message.
            log.warning("AI request rejected: %s", e)
            await _safe_followup(interaction, str(e))
            return
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
            await interaction.response.send_message(text)
        else:
            await interaction.followup.send(text)
    except Exception as e:
        log.warning("Follow-up send failed: %s", e)
