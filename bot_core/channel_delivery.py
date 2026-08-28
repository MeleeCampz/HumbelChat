"""Shared channel-delivery helpers for background senders (reminders etc.).

Why this module exists: background delivery must NOT rely on the local
channel cache.  ``bot.get_channel()`` only returns channels the bot is a
member of AND that Discord included in its gateway payload — a private
channel with a View-Channel deny on @everyone is missing from the cache,
so ``get_channel`` returned ``None`` and delivery was silently skipped
while still being logged as "fired" (root cause found 2026-08-28).

:func:`send_to_channel` therefore has a REST fallback via
``bot.http.get_channel(channel_id)`` and raises on genuine failures so
callers can log loudly and keep the item in their queue.

:func:`can_post_in_channel` is an upfront permission check used by the
``/remind`` handler to refuse scheduling into a channel the bot cannot
actually write to (fail fast, at schedule time).
"""
from __future__ import annotations

import logging

import discord

log = logging.getLogger("bot.delivery")


class ChannelNotDeliverableError(Exception):
    """The target channel exists but the bot cannot post to it (or at all)."""

    def __init__(self, channel_id: int, reason: str) -> None:
        super().__init__(f"channel {channel_id} not deliverable: {reason}")
        self.channel_id = channel_id
        self.reason = reason


def get_bot() -> discord.Client | None:
    """Resolve the running, logged-in bot WITHOUT any side effects.

    Resolution order (first usable hit wins):
      1. ``sys.modules["__main__"]`` — when the app runs as a script
         (``python main.py``, the normal case) the fully-loaded entry module
         IS ``__main__``; reading its ``bot`` global never triggers an import;
      2. ``sys.modules["main"]`` — hosts that loaded ``main`` as a regular
         module (the test suite imports it via conftest);
      3. a fresh ``import main`` — last resort only, if neither cached copy
         exists yet.

    Usability rule: a bot is usable once it has logged in (``bot.user`` is
    set) — we deliberately do NOT gate on ``is_ready()``.  discord.py sets
    that internal flag *after* dispatching READY to user callbacks, so during
    on_ready the cache is already populated and sends succeed; gating on
    ``is_ready()`` rejected exactly those working deliveries (observed live
    2026-08-28).  By contrast ``bot.user`` IS set before on_ready dispatches,
    which is why it's the correct "real bot" signal.

    Every miss is logged (with traceback on import errors) so a silent
    ``None`` can never hide the real cause again — that was the 2026-08-28
    "no bot object available" mystery: a lazy ``from main import bot`` inside
    a running script re-executed main.py's module level and returned the fresh
    (never-logged-in) duplicate instead of the live bot.
    """
    import sys

    def _cached():
        seen = set()
        for name in ("__main__", "main"):
            mod = sys.modules.get(name)
            if mod is not None and id(mod) not in seen:
                seen.add(id(mod))
                yield name, mod

    # Pass 1: a logged-in bot from any already-cached module.  Login state is
    # the "this is the real, live client" signal — a stray fresh duplicate
    # produced by a mid-process re-import has user == None and is skipped.
    for name, mod in _cached():
        b = getattr(mod, "bot", None)
        if b is not None and getattr(b, "user", None) is not None:
            return b

    # Pass 2 (startup window): nothing logged in yet.  In production the ENTRY
    # module (__main__, i.e. `python main.py`) unambiguously hosts the live bot
    # object — it was created at module level and its login is simply still in
    # flight, so trust it without requiring user.  We do NOT do this for a
    # secondary copy (sys.modules["main"]) because THAT is where a re-executed
    # duplicate would hide.
    for name, mod in _cached():
        if name == "__main__":
            b = getattr(mod, "bot", None)
            if b is not None:
                return b
            break
        log.warning("get_bot(): '%s.bot' exists but is not logged in yet (user=None) — ignoring secondary copy",
                    name)

    # Pass 3: exotic hosts that haven't cached main at all.
    try:
        import main as _m
        b = getattr(_m, "bot", None)
        if b is not None:
            return b
    except Exception:
        log.exception("get_bot(): could not import 'main' to find the bot — "
                      "delivery will fail for this event")
        return None

    log.warning(
        "get_bot(): no bot found in %s — is main.py still starting up, or did "
        "something restart the process?",
        [name for name, _ in _cached()] or "no cached modules")
    return None


# Back-compat alias for the old private name.
_get_bot = get_bot


async def send_to_channel(bot, channel_id: int, content: str):
    """Send ``content`` to ``channel_id`` with a REST fallback.

    Resolution order:
      1. local cache — ``bot.get_channel()`` (fast path, no network);
      2. REST fetch — ``bot.http.get_channel()`` (catches channels missing
         from the gateway cache);
      3. raise :class:`ChannelNotDeliverableError` when the channel is not
         found at all or sending fails (403 Forbidden / no permissions).

    Raises on failure — callers are expected to log loudly and keep their
    item queued rather than silently dropping it.  Returns the channel
    object that was used for the send.
    """
    chan = None
    if bot is not None:
        try:
            chan = bot.get_channel(channel_id)
        except Exception as e:  # pragma: no cover - defensive
            log.warning("get_channel(%s) raised %r", channel_id, e)

    via = "cache"
    if chan is None and bot is not None:
        try:
            chan = await bot.http.get_channel(channel_id)
            via = "rest"
        except discord.Forbidden:
            raise ChannelNotDeliverableError(
                channel_id,
                "bot has no access to this channel (permission denied — "
                "check the channel's @everyone / role overwrites)",
            ) from None
        except discord.NotFound:
            raise ChannelNotDeliverableError(
                channel_id,
                "channel does not exist or is outside every guild the bot is in",
            ) from None
        except Exception as e:  # rate limit, connection drop mid-reconnect, etc.
            log.warning("REST channel lookup for %s failed: %r", channel_id, e)
            raise ChannelNotDeliverableError(
                channel_id, f"channel lookup failed ({e!r})") from e

    if chan is None:
        # The send path only gets here when no usable bot OR channel exists.
        raise ChannelNotDeliverableError(
            channel_id,
            "no bot instance or channel available to deliver the message",
        )

    try:
        await chan.send(content)
    except discord.Forbidden:
        guild = getattr(chan, "guild", None)
        gname = getattr(guild, "name", "?") if guild is not None else "DM"
        cname = getattr(chan, "name", str(channel_id))
        raise ChannelNotDeliverableError(
            channel_id,
            f"bot cannot post in #{cname} (guild: {gname}) — "
            "check View Channel / Send Messages permissions and role overwrites",
        ) from None

    log.info("Delivered message to channel %s via %s", channel_id, via)
    return chan


async def can_post_in_channel(interaction: discord.Interaction) -> bool:
    """Upfront check: can this bot see AND post in the invoking channel?

    Uses the permissions computed for the current interaction, so it is
    authoritative even for private channels that are missing from the bot's
    local cache.  Returns ``False`` (→ refuse to schedule) on any doubt.
    """
    try:
        channel = getattr(interaction, "channel", None)
        if channel is None:
            return False
        if not getattr(interaction, "guild_id", None):
            # DMs: no View-Channel concept to check; delivery-time send
            # (with its REST fallback + loud error logging) is the real check.
            return True
        me = interaction.guild.me  # type: ignore[union-attr]
        if me is None:
            return False
        perm = channel.permissions_for(me)
        return bool(perm.view_channel and perm.send_messages)
    except Exception as e:  # pragma: no cover - defensive
        log.warning("can_post_in_channel check failed: %r", e)
        return False
