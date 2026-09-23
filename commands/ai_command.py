"""AI chat command handlers (slash ``/ai`` + shared turn runner for prefix).

Replies are delivered as Discord embeds (title + description + inline fields)
when EMBED_FORMAT is enabled; plain prose still works and tiny/empty replies
fall back to plain-text chunks so the user always gets an answer.

P3 additions:
  * #24 streaming — when AI_STREAM is on, the completion is consumed token-by-
    token and a placeholder message is edited live as the text grows. The
    typing indicator now stays alive for the whole turn (queue wait +
    generation + delivery) instead of dying after 30 s.
  * #25 stop — the AI call + delivery run in a cancelable child task that is
    registered in :mod:`bot_core.ai_runs`, so ``/ai stop`` can cancel it.
  * #26 queue feedback — if other /ai requests are already in the global queue
    (the bot serializes AI requests process-wide), the user is told how many
    are ahead of them instead of waiting silently.

The delivery helpers accept a generic *source*: either a
``discord.Interaction`` (slash) or a ``discord.Message`` (prefix), so the exact
same turn logic serves both entry points.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import config.settings as _settings
from config.characters import get_character, default_character
from bot_core import ai_client
from bot_core.history import get_active_char_key
from bot_core.ai_runs import register_run, clear_run
from utils.background_tasks import spawn_tracked_task
from utils.channel_queue import channel_slot
from utils.response_splitter import (
    send_long_response,
    send_long_response_embedded,
    DISCORD_SAFE_CHUNK,
)
from utils.typing_loop import typing_loop_task

if TYPE_CHECKING:  # discord is referenced only in string annotations
    import discord

log = logging.getLogger("bot.commands.ai_command")


# ─────────────────────────────────────────────────────────────────────────
#  Generic source helpers (work for both Interaction and Message)
# ─────────────────────────────────────────────────────────────────────────

def _source_channel(source):
    """Best-effort TextChannel for a source (interaction or message)."""
    if source is None:
        return None
    return getattr(source, "channel", None)


async def _deliver_error(source, text: str) -> None:
    """Send an error to the user, tolerating already-responded interactions."""
    try:
        if hasattr(source, "followup") and getattr(source, "response", None) is not None \
                and not source.response.is_done():
            await source.response.send_message(text)
        elif hasattr(source, "followup"):
            await source.followup.send(text)
        else:
            await source.channel.send(text)
    except Exception as e:
        log.warning("Error follow-up send failed: %s", e)


async def _deliver_final(source, reply_text: str, char_name: str) -> None:
    """Standard delivery: embed when enabled, else paragraph-aware chunks."""
    if _settings.EMBED_FORMAT:
        delivered = await send_long_response_embedded(source, reply_text, char_name)
        if not delivered:
            await send_long_response(source, reply_text, char_name)
    else:
        await send_long_response(source, reply_text, char_name)


async def _make_placeholder(source) -> object | None:
    """Post the initial streaming placeholder, or None on failure."""
    try:
        if hasattr(source, "followup"):
            return await source.followup.send("⌨️ *Thinking…*")
        return await source.channel.send("⌨️ *Thinking…*")
    except Exception as e:
        log.warning("streaming placeholder send failed: %s", e)
        return None


async def _deliver_streaming(source, user_message: str, model_slug: str,
                             guild_id: int, channel_id, username: str,
                             user_id, char_name: str) -> None:
    """P3 #24: stream the completion, editing one placeholder live.

    A ``⌨️ *Thinking…*`` placeholder is posted and edited (throttled) with the
    growing text. On completion the placeholder is removed and the reply is
    delivered through the normal (embed/chunk) path so formatting is
    consistent with the non-streaming path. On any failure the placeholder is
    removed and the error propagates to the caller.
    """
    edit_interval = _settings.AI_STREAM_EDIT_INTERVAL_S
    placeholder = await _make_placeholder(source)
    buf = ""
    deleted = False
    try:
        loop = asyncio.get_running_loop()
        last_edit = 0.0
        async for text in ai_client.ask_ai_stream(
            user_message=user_message,
            model_slug=model_slug,
            guild_id=guild_id,
            channel_id=channel_id,
            username=username,
            user_id=user_id,
        ):
            buf = text
            if placeholder is not None and loop.time() - last_edit >= edit_interval:
                last_edit = loop.time()
                preview = buf[: DISCORD_SAFE_CHUNK - 10]
                try:
                    await placeholder.edit(content=preview)
                except Exception:
                    pass

        reply_text = buf
        if placeholder is not None:
            try:
                await placeholder.delete()
                deleted = True
            except Exception:
                pass
        await _deliver_final(source, reply_text, char_name)
    except BaseException:
        if placeholder is not None and not deleted:
            try:
                await placeholder.delete()
            except Exception:
                pass
        raise


# ─────────────────────────────────────────────────────────────────────────
#  Shared turn runner (used by both the slash and prefix paths)
# ─────────────────────────────────────────────────────────────────────────

async def run_ai_turn(
    source,
    *,
    user_message: str,
    model_slug: str,
    guild_id: int,
    channel_id,
    username: str = "",
    user_id=None,
    char_name: str,
) -> None:
    """Hold this channel's slot for the ENTIRE request + delivery, then deliver.

    Without the slot, a second /ai in the same channel can interleave its
    messages with this one (Discord orders by send time, not request time —
    see utils/channel_queue.py).
    """
    async with channel_slot(channel_id, name="ai-turn"):
        try:
            if _settings.AI_STREAM:
                await _deliver_streaming(
                    source, user_message, model_slug, guild_id, channel_id,
                    username, user_id, char_name,
                )
            else:
                reply_text, _extra = await ai_client.ask_ai(
                    user_message=user_message,
                    model_slug=model_slug,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    username=username,
                    user_id=user_id,
                )
                await _deliver_final(source, reply_text, char_name)
        except asyncio.CancelledError:
            raise
        except ValueError as e:
            # §3.7: classified errors (timeout / model-not-found / backend-down /
            # rate-limit / input too long) surface a user-friendly message.
            log.warning("AI request rejected: %s", e)
            await _deliver_error(source, str(e))
        except Exception as e:
            log.error("AI request failed: %s", e)
            await _deliver_error(source, f"❌ {e}")


async def notify_if_queued(source) -> None:
    """P3 #26: let the user know how many AI requests are ahead of them."""
    ahead = ai_client.ai_queue_depth() + (1 if ai_client.ai_slot_busy() else 0)
    if ahead <= 0:
        return
    text = (
        f"⏳ {ahead} other request(s) are ahead of you in the AI queue — "
        "starting yours next. (Use `/ai stop` to bail.)"
    )
    try:
        if hasattr(source, "followup") and getattr(source, "response", None) is not None \
                and not source.response.is_done():
            await source.response.send_message(text)
        elif hasattr(source, "followup"):
            await source.followup.send(text)
        else:
            await source.channel.send(text)
    except Exception as e:
        log.warning("AI queue notify failed: %s", e)


def start_typing(source, channel_id) -> asyncio.Task | None:
    """Start an until-cancelled typing loop; keep a strong ref + diagnostics.

    P3 #24: ``duration_sec=None`` runs the loop until it is cancelled, so it
    covers queue wait + generation + delivery (the old 30 s loop died while a
    request was still queued).
    """
    channel = _source_channel(source)
    if channel is None:
        return None
    try:
        task = spawn_tracked_task(
            typing_loop_task(channel, duration_sec=None),
            name=f"typing-{channel_id}",
        )
        bot_ref = getattr(source, "bot", None)
        if bot_ref is not None:
            tasks = getattr(bot_ref, "typing_tasks", None)
            if not isinstance(tasks, list):
                tasks = []
                setattr(bot_ref, "typing_tasks", tasks)
            bot_ref.typing_tasks = [t for t in tasks if not t.done()]
            bot_ref.typing_tasks.append(task)
        return task
    except Exception as e:
        log.warning("Typing loop error: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────
#  Slash command entry point
# ─────────────────────────────────────────────────────────────────────────

async def handle_ai_command(
    interaction: "discord.Interaction",
    message: str,
    character_name: str | None = None,
) -> None:
    """Core handler for the ``/ai`` slash command."""
    # 1. Immediate deferral to prevent "Application did not respond"
    try:
        if not interaction.response.is_done():
            await interaction.response.defer()
    except Exception as e:
        log.error("Error deferring interaction: %s", e)
        return

    # 2. Resolve character.
    #    ``character_name`` is the *explicit* request (None when the user just
    #    typed /ai with no persona). The fallback is the channel's persisted
    #    active key, which itself defaults to the default character when no
    #    per-channel key exists (see bot_core.history.get_active_char_key).
    char_key = character_name
    explicit = character_name is not None
    if char_key is None:
        char_key = get_active_char_key(interaction.guild_id, interaction.channel_id)

    char_obj = get_character(char_key)
    if char_obj is None:
        if explicit:
            await interaction.followup.send(f"Character `{character_name}` not found.")
            return
        # P1 #9: stale persisted key → fall back to default + tell the user.
        char_obj = default_character()
        log.info(
            "/ai: persisted active character %r not found for guild=%s channel=%s; "
            "falling back to default %r.",
            char_key, interaction.guild_id, interaction.channel_id, char_obj.key,
        )
        await interaction.followup.send(
            f"⚠️ Your saved character `{char_key}` no longer exists, so I'm using "
            f"**{char_obj.display}** for this message. Use `/character set` to "
            "pick a different one."
        )

    model_slug = char_obj.model or ""
    user_id = getattr(getattr(interaction, "user", None), "id", None)
    username = getattr(getattr(interaction, "user", None), "display_name", "") or ""
    guild_id = interaction.guild_id or 0
    channel_id = interaction.channel_id
    channel_key = channel_id if channel_id is not None else 0
    display = str(char_obj.display)

    # 3. AI queue feedback (P3 #26) before we start waiting on our own turn.
    await notify_if_queued(interaction)

    # 4. Typing for the whole turn (cancelled in finally).
    typing_task = start_typing(interaction, channel_id)

    # 5. Run the turn in a cancelable child task (P3 #25) so /ai stop works.
    run_task = spawn_tracked_task(
        run_ai_turn(
            interaction,
            user_message=message,
            model_slug=model_slug,
            guild_id=guild_id,
            channel_id=channel_id,
            username=username,
            user_id=user_id,
            char_name=display,
        ),
        name=f"ai-run-{channel_key}",
    )
    register_run(channel_key, run_task)
    try:
        await run_task
    except asyncio.CancelledError:
        # /ai stop sent its own acknowledgement; nothing more to deliver here.
        log.info("ai_command: in-flight run for channel %s cancelled (/ai stop)", channel_key)
    finally:
        clear_run(channel_key, run_task)
        if typing_task is not None:
            typing_task.cancel()
