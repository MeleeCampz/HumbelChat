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
)
from utils.typing_loop import typing_loop_task

if TYPE_CHECKING:  # discord is referenced only in string annotations
    import discord

log = logging.getLogger("bot.commands.ai_command")


def resolve_turn_character(guild_id, channel_id, character_name: str | None):
    """Resolve the persona for one turn.

    ``character_name`` is the explicit ``/ai`` choice (a key or display name),
    or ``None`` to use the channel's active character.

    Returns ``(character, warning)``. ``character`` is ``None`` only when the
    user named a persona that does not exist. ``warning`` is set when a
    persisted active key is stale and the turn falls back to the default.
    """
    explicit = character_name is not None
    char_key = character_name
    if char_key is None:
        char_key = get_active_char_key(guild_id, channel_id)

    char_obj = get_character(char_key)
    if char_obj is not None:
        return char_obj, None

    if explicit:
        return None, None

    char_obj = default_character()
    log.info(
        "persisted active character %r not found for guild=%s channel=%s; "
        "falling back to default %r.",
        char_key, guild_id, channel_id, char_obj.key,
    )
    warning = (
        f"⚠️ Your saved character `{char_key}` no longer exists, so I'm using "
        f"**{char_obj.display}** for this message. Use `/character set` to "
        "pick a different one."
    )
    return char_obj, warning


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


async def _deliver_streaming(source, user_message: str, model_slug: str,
                             guild_id: int, channel_id, username: str,
                             user_id, char_name: str, char_key: str | None) -> None:
    """P3 #24: consume the streamed completion, then deliver it through the normal
    (embed/chunk) path.

    The stream is consumed internally for timing + mid-flight cancellation, but NO
    placeholder message is posted — so the original slash command stays visible and
    the reply appears exactly like the non-streaming path (one final embed). This
    avoids Discord hiding the command when a bot follow-up is sent first.
    """
    buf = ""
    async for text in ai_client.ask_ai_stream(
        user_message=user_message,
        model_slug=model_slug,
        guild_id=guild_id,
        channel_id=channel_id,
        username=username,
        user_id=user_id,
        char_key=char_key,
    ):
        buf = text
    await _deliver_final(source, buf, char_name)


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
    char_key: str | None = None,
) -> None:
    """Hold this channel's slot for the ENTIRE request + delivery, then deliver.

    Without the slot, a second /ai in the same channel can interleave its
    messages with this one (Discord orders by send time, not request time —
    see utils/channel_queue.py).
    """
    channel_key = channel_id if channel_id is not None else 0
    async with channel_slot(channel_id, name="ai-turn"):
        # Register only once this turn holds the slot. A later /ai in the
        # same channel is queued here and must not replace this task, or
        # /ai stop would cancel the waiter instead of the turn in progress.
        run_task = asyncio.current_task()
        if run_task is not None:
            register_run(channel_key, run_task)
        try:
            if _settings.AI_STREAM:
                await _deliver_streaming(
                    source, user_message, model_slug, guild_id, channel_id,
                    username, user_id, char_name, char_key,
                )
            else:
                reply_text, _extra = await ai_client.ask_ai(
                    user_message=user_message,
                    model_slug=model_slug,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    username=username,
                    user_id=user_id,
                    char_key=char_key,
                )
                await _deliver_final(source, reply_text, char_name)
        except asyncio.CancelledError:
            raise
        except ValueError as e:
            # §3.7: classified errors (timeout / model-not-found / backend-down /
            # rate-limit / input too long) surface a user-friendly message.
            log.warning("AI request rejected: %s", e)
            await _deliver_error(source, str(e))
        except Exception:
            log.exception("AI request failed")
            await _deliver_error(source, "❌ Something went wrong while generating a reply. Please try again.")
        finally:
            if run_task is not None:
                clear_run(channel_key, run_task)


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
        bot_ref = getattr(source, "client", None)
        if bot_ref is None:
            from bot_core.channel_delivery import get_bot
            bot_ref = get_bot()
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

    # 2. Resolve character (explicit choice, else the channel's active key).
    char_obj, warning = resolve_turn_character(
        interaction.guild_id, interaction.channel_id, character_name,
    )
    if char_obj is None:
        await interaction.followup.send(f"Character `{character_name}` not found.")
        return
    if warning:
        await interaction.followup.send(warning)

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
            char_key=char_obj.key,
        ),
        name=f"ai-run-{channel_key}",
    )
    try:
        await run_task
    except asyncio.CancelledError:
        # /ai stop sent its own acknowledgement; nothing more to deliver here.
        log.info("ai_command: in-flight run for channel %s cancelled (/ai stop)", channel_key)
    finally:
        clear_run(channel_key, run_task)
        if typing_task is not None:
            typing_task.cancel()
