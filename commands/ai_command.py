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
  * #70 long-turn progress — fast turns still show nothing but the typing
    indicator; a turn without a reply after ``_PROGRESS_DELAY_S`` gets ONE
    follow-up "🧠 Thinking…" message that is edited with elapsed time until
    the reply lands (flipping to "✍️ Writing…" when the first stream token
    arrives), then deleted — the final embed/chunks are delivered as usual.
    Same live-status pattern as /generate_image; a broken progress display
    never fails the turn.

The delivery helpers accept a generic *source*: either a
``discord.Interaction`` (slash) or a ``discord.Message`` (prefix), so the exact
same turn logic serves both entry points.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, cast

import config.settings as _settings
from config.characters import Character, get_character, default_character
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

    # A turn can be triggered from a slash interaction or a prefix message;
    # the delivery helpers duck-type over both.
    Source = discord.Interaction | discord.Message

log = logging.getLogger("bot.commands.ai_command")

# #70: long-turn progress message (see module docstring).
_PROGRESS_DELAY_S = 15.0   # silence before the first "Thinking…" appears
_PROGRESS_TICK_S = 30.0    # elapsed-time refresh while still running


def resolve_turn_character(
    guild_id: int | None,
    channel_id: int | None,
    character_name: str | None,
) -> tuple[Character | None, str | None]:
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

def _source_channel(source: Source | None) -> discord.TextChannel | None:
    """Best-effort TextChannel for a source (interaction or message)."""
    if source is None:
        return None
    # String form: discord is a TYPE_CHECKING-only import.
    return cast("discord.TextChannel | None", getattr(source, "channel", None))


async def _deliver_error(source: Source, text: str) -> None:
    """Send an error to the user, tolerating already-responded interactions."""
    try:
        # Only slash Interactions have a deferred ``response``; prefix
        # Messages go straight to followup/channel sends.
        response = getattr(source, "response", None)
        if response is not None and not response.is_done():
            await response.send_message(text)
        elif hasattr(source, "followup"):
            await source.followup.send(text)
        else:
            await source.channel.send(text)
    except Exception as e:
        log.warning("Error follow-up send failed: %s", e)


async def _deliver_final(source: Source, reply_text: str, char_name: str) -> None:
    """Standard delivery: embed when enabled, else paragraph-aware chunks."""
    if _settings.EMBED_FORMAT:
        delivered = await send_long_response_embedded(source, reply_text, char_name)
        if not delivered:
            await send_long_response(source, reply_text, char_name)
    else:
        await send_long_response(source, reply_text, char_name)


async def _deliver_streaming(source: Source, user_message: str, model_slug: str,
                             guild_id: int, channel_id: int | None, username: str,
                             user_id: int | None, char_name: str,
                             char_key: str | None,
                             on_first_token: Callable[[], None] | None = None) -> None:
    """P3 #24: consume the streamed completion, then deliver it through the normal
    (embed/chunk) path.

    The stream is consumed internally for timing + mid-flight cancellation, but NO
    placeholder message is posted here — the reply appears exactly like the
    non-streaming path (one final embed). Long-turn feedback lives in
    :class:`_TurnProgress` (started by ``run_ai_turn``): it stays out of the way
    on fast turns and only posts ONE delayed "Thinking…" message that is deleted
    when this delivery runs.
    """
    buf = ""
    first = True
    async for text in ai_client.ask_ai_stream(
        user_message=user_message,
        model_slug=model_slug,
        guild_id=guild_id,
        channel_id=channel_id,
        username=username,
        user_id=user_id,
        char_key=char_key,
    ):
        if first:
            first = False
            if on_first_token is not None:
                try:
                    on_first_token()
                except Exception as exc:  # noqa: BLE001 — progress must never fail the turn
                    log.debug("on_first_token hook failed: %s", exc)
        buf = text
    await _deliver_final(source, buf, char_name)


# ─────────────────────────────────────────────────────────────────────────
#  Shared turn runner (used by both the slash and prefix paths)
# ─────────────────────────────────────────────────────────────────────────

async def run_ai_turn(
    source: Source,
    *,
    user_message: str,
    model_slug: str,
    guild_id: int,
    channel_id: int | None,
    username: str = "",
    user_id: int | None = None,
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
        progress = _TurnProgress(source)
        progress.start()
        try:
            if _settings.AI_STREAM:
                await _deliver_streaming(
                    source, user_message, model_slug, guild_id, channel_id,
                    username, user_id, char_name, char_key,
                    on_first_token=progress.mark_writing,
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
            await progress.finish()


class _TurnProgress:
    """#70: delayed progress message for long /ai turns (pattern of /generate_image).

    Fast turns show nothing but the typing indicator. If no reply arrives
    within ``_PROGRESS_DELAY_S``, ONE follow-up (slash) / channel (prefix)
    "🧠 Thinking…" message appears and is edited with elapsed time every
    ``_PROGRESS_TICK_S`` — flipping to "✍️ Writing…" once the first stream
    token arrives. When the turn ends it is deleted; the final embed/chunks
    are delivered through the normal path. A broken progress display never
    fails the turn.
    """

    def __init__(self, source: Source) -> None:
        self._source = source
        self._msg: Any = None
        self._t0: float | None = None
        self._writing = False
        self._task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        self._t0 = time.monotonic()
        try:
            self._task = spawn_tracked_task(
                self._ticker(), name="ai-turn-progress"
            )
        except Exception as e:  # noqa: BLE001 — progress must never fail the turn
            log.warning("turn progress start failed: %s", e)
            self._task = None

    def mark_writing(self) -> None:
        """First stream token arrived — generation is visibly underway."""
        self._writing = True

    async def finish(self) -> None:
        """Cancel the ticker and delete the progress message (best effort)."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except BaseException:  # noqa: BLE001 — already cancelled / interrupted
                pass
            self._task = None
        if self._msg is not None:
            try:
                await self._msg.delete()
            except Exception as e:  # noqa: BLE001 — cosmetic only
                log.debug("turn progress delete failed: %s", e)
            self._msg = None

    def _render(self) -> str:
        elapsed = int(time.monotonic() - (self._t0 or time.monotonic()))
        stamp = f"{elapsed // 60}m {elapsed % 60:02d}s" if elapsed >= 60 else f"{elapsed}s"
        return ("✍️ Writing…" if self._writing else "🧠 Thinking…") + f" ({stamp})"

    async def _send(self, text: str) -> Any:
        # Interactions have followup; prefix Messages go to the channel.
        if hasattr(self._source, "followup"):
            return await self._source.followup.send(text)
        return await self._source.channel.send(text)

    async def _ticker(self) -> None:
        try:
            await asyncio.sleep(_PROGRESS_DELAY_S)
            while True:
                if self._msg is None:
                    self._msg = await self._send(self._render())
                else:
                    await self._msg.edit(content=self._render())
                await asyncio.sleep(_PROGRESS_TICK_S)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — progress must never fail the turn
            log.debug("turn progress ticker failed: %s", e)


async def notify_if_queued(source: Source) -> None:
    """P3 #26: let the user know how many AI requests are ahead of them."""
    ahead = ai_client.ai_queue_depth() + (1 if ai_client.ai_slot_busy() else 0)
    if ahead <= 0:
        return
    text = (
        f"⏳ {ahead} other request(s) are ahead of you in the AI queue — "
        "starting yours next. (Use `/ai stop` to bail.)"
    )
    try:
        response = getattr(source, "response", None)
        if response is not None and not response.is_done():
            await response.send_message(text)
        elif hasattr(source, "followup"):
            await source.followup.send(text)
        else:
            await source.channel.send(text)
    except Exception as e:
        log.warning("AI queue notify failed: %s", e)


def start_typing(source: Source, channel_id: int | None) -> asyncio.Task[Any] | None:
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
        bot_ref: Any = getattr(source, "client", None)
        if bot_ref is None:
            from bot_core.channel_delivery import get_bot
            bot_ref = get_bot()
        if bot_ref is not None:
            tasks = getattr(bot_ref, "typing_tasks", None)
            if not isinstance(tasks, list):
                tasks = []
                setattr(bot_ref, "typing_tasks", tasks)
            # Dynamic bookkeeping attribute on the live client (see docstring).
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
