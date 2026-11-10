"""`/botperms` support — one-shot channel-delivery permission diagnostic (#79).

Why this module exists
----------------------
The bot has **two delivery paths with different permission requirements**, and
only one of them ever complains:

* **Interaction replies** (``interaction.response.*`` / ``followup.send``) travel
  over the interaction *webhook* Discord handed out when the command was
  invoked.  They never look the channel up and never consult its overwrites, so
  ``/ai``, ``/end_session`` and friends keep working even when the bot cannot
  read the channel at all.
* **Background senders** (session highlights, next-session reminders, roll-button
  re-attach) must resolve the channel **by ID** through
  :func:`bot_core.channel_delivery.resolve_channel` and then ``chan.send()``.
  Those need ``View Channel`` (to fetch the object) plus the normal send
  permissions.

That asymmetry is why a channel can look perfectly healthy in chat and still
archive highlights to disk only.  This module answers the question the logs
cannot: *which* of the two is broken, and *why* — it runs the exact resolution
a background sender runs, and reports the permission bits behind it.

Everything here is read-only; nothing is posted, retried or changed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import discord

from bot_core.channel_delivery import ChannelNotDeliverableError, resolve_channel

log = logging.getLogger("bot.perms_report")

#: ``(attribute, label, why)`` — the attributes this bot actually consumes.
#: Read with ``getattr(..., None)`` so a discord.py rename degrades to "unknown"
#: instead of raising inside a diagnostic.
CHECKS: tuple[tuple[str, str, str], ...] = (
    ("view_channel", "View Channel", "background senders must fetch the channel object"),
    ("send_messages", "Send Messages", "highlights, reminders, roll views"),
    ("read_message_history", "Read Message History", "sending into a channel not cached yet"),
    ("attach_files", "Attach Files", "highlight/image PNGs, /session_file, /speak"),
    ("send_messages_in_threads", "Send Messages in Threads", "sessions or reminders inside threads"),
    ("create_public_threads", "Create Public Threads", "posting into a forum channel"),
    ("manage_threads", "Manage Threads", "archive/edit forum posts"),
    ("embed_links", "Embed Links", "/help and other embed-only replies"),
)

#: Missing any of these breaks a shipped feature (see docs/permissions.md).
REQUIRED: frozenset[str] = frozenset(
    {"view_channel", "send_messages", "read_message_history", "attach_files"}
)

#: Discord caps a message at 2000 characters; keep the report well inside it.
MAX_REPORT_CHARS = 1900

_FOOTER = (
    "_Slash-command replies use the interaction webhook and keep working even "
    "when everything above is denied — only background senders (session "
    "highlights, next-session reminders, roll buttons) depend on these._"
)


@dataclass
class PermsReport:
    """Everything one diagnostic pass learned about one channel."""

    channel_id: int
    name: str = "?"
    kind: str = "unknown"
    parent: str | None = None
    bot_name: str = "?"
    bot_id: int | None = None
    is_guild_member: bool = False
    roles: list[str] = field(default_factory=list)
    #: ``(attribute, label, granted)`` — ``granted is None`` = unknown attribute.
    perms: list[tuple[str, str, bool | None]] = field(default_factory=list)
    #: What ``bot.get_channel()`` (the gateway cache) returned.
    cache_hit: bool = False
    #: Result of the same ID lookup the background senders fall back to.
    resolve_ok: bool = False
    resolve_via: str = ""
    resolve_reason: str = ""
    notes: list[str] = field(default_factory=list)
    verdict: str = ""


def _kind(channel: Any) -> str:
    """Human-readable channel kind, defensive about discord.py shapes."""
    ctype = getattr(channel, "type", None)
    if ctype is None:
        return type(channel).__name__
    name = getattr(ctype, "name", None)
    if isinstance(name, str):
        return name.replace("_", " ")
    return str(ctype)


def _is_forum(kind: str) -> bool:
    return kind in {"forum", "media"}


def _is_thread(kind: str) -> bool:
    return "thread" in kind


def _read_perms(channel: Any, member: Any) -> list[tuple[str, str, bool | None]]:
    """Read every :data:`CHECKS` attribute off ``channel.permissions_for(member)``."""
    out: list[tuple[str, str, bool | None]] = []
    perms_obj = None
    try:
        perms_obj = channel.permissions_for(member)
    except Exception as e:  # noqa: BLE001 - a diagnostic must not raise
        log.warning("permissions_for() failed: %r", e)
    for attr, label, _why in CHECKS:
        value = getattr(perms_obj, attr, None) if perms_obj is not None else None
        out.append((attr, label, bool(value) if value is not None else None))
    return out


def _verdict(rep: PermsReport) -> str:
    """One-line conclusion, ordered by what actually breaks a feature."""
    missing_required = [label for attr, label, granted in rep.perms
                        if attr in REQUIRED and granted is False]
    missing_optional = [label for attr, label, granted in rep.perms
                        if attr not in REQUIRED and granted is False]

    if not rep.resolve_ok:
        return (
            f"❌ **Background delivery will fail in this channel:** {rep.resolve_reason} "
            "Session highlights are archived to disk only, next-session reminders "
            "never arrive, saved roll views are not re-attached. "
            "Fix: channel → Permissions → the bot's role (and the @everyone row) → "
            "allow **View Channel**."
        )

    if missing_required:
        return (
            "❌ **Background delivery will fail in this channel:** denied → "
            + ", ".join(f"**{m}**" for m in missing_required)
            + ". The channel object resolves, so the send itself is what Discord refuses."
        )

    if missing_optional:
        return (
            "✅ **Background delivery works here.** Optional bits denied: "
            + ", ".join(missing_optional)
            + " — only relevant for the features named beside them."
        )

    if not rep.cache_hit:
        return (
            "✅ **Background delivery works here, but slowly:** the channel is missing "
            "from the gateway cache, so every background send pays a REST fetch "
            "(`via rest`). A View-Channel deny on `@everyone` looks exactly like this."
        )

    return "✅ **Background delivery works here** (resolved from the gateway cache)."


async def build_report(bot: Any, channel: Any) -> PermsReport:
    """Probe ``channel`` exactly the way a background sender would.

    Safe with a half-shaped fake ``bot``/``channel`` (the test suite does this)
    and with a live discord.py client: every attribute read is defensive.
    """
    rep = PermsReport(channel_id=int(getattr(channel, "id", 0) or 0))
    rep.name = str(getattr(channel, "name", None) or rep.channel_id)
    rep.kind = _kind(channel)

    parent = getattr(channel, "parent", None)
    if parent is not None:
        rep.parent = f"{getattr(parent, 'name', '?')} ({parent.id}, {_kind(parent)})"

    user = getattr(bot, "user", None)
    if user is not None:
        rep.bot_name = str(getattr(user, "display_name", None) or getattr(user, "name", "?"))
        rep.bot_id = getattr(user, "id", None)

    guild = getattr(channel, "guild", None)
    me = getattr(guild, "me", None) if guild is not None else None
    rep.is_guild_member = me is not None
    if me is not None:
        try:
            rep.roles = [r.name for r in me.roles][:10]
        except Exception:  # noqa: BLE001 - cosmetic only
            rep.roles = []
        rep.perms = _read_perms(channel, me)
    else:
        rep.perms = [(attr, label, None) for attr, label, _why in CHECKS]
        rep.notes.append("The bot's member object is unavailable for this guild.")

    # 1) gateway cache (fast path), 2) REST fallback — exactly resolve_channel's order.
    try:
        rep.cache_hit = bot.get_channel(rep.channel_id) is not None
    except Exception as e:  # noqa: BLE001 - defensive
        log.warning("get_channel(%s) raised %r", rep.channel_id, e)

    try:
        _chan, via = await resolve_channel(bot, rep.channel_id)
        rep.resolve_ok, rep.resolve_via = True, via
    except ChannelNotDeliverableError as e:
        rep.resolve_ok, rep.resolve_via, rep.resolve_reason = False, "failed", e.reason
    except Exception as e:  # noqa: BLE001 - never let a probe kill the command
        rep.resolve_ok, rep.resolve_via, rep.resolve_reason = False, "failed", repr(e)

    if _is_thread(rep.kind) and rep.parent:
        rep.notes.append(
            f"This is a **{rep.kind}** — the parent is {rep.parent}. "
            "Session highlights post to the **ending channel id** recorded at "
            "`/end_session`; if that id is the parent forum instead of the post, "
            "delivery targets a channel that cannot hold messages."
        )
    if _is_forum(rep.kind):
        rep.notes.append(
            "This is a **forum channel** — it is not directly messageable. "
            "A background sender that resolves this id gets an object without "
            "`send()`; the post (thread) is the messageable target."
        )

    rep.verdict = _verdict(rep)
    return rep


async def find_channel(bot: Any, guild: Any, ref: str) -> tuple[Any | None, str]:
    """Resolve a user-supplied channel reference: id, ``<#id>`` or channel name.

    Returns ``(channel, note)`` — ``channel is None`` when it cannot be resolved,
    with ``note`` explaining why (which is itself a useful permission answer).
    """
    text = str(ref).strip().removeprefix("<#").removesuffix(">").strip()
    if not text:
        return None, "Empty channel reference."

    if text.lstrip("#").isdigit():
        cid = int(text.lstrip("#"))
        chan = bot.get_channel(cid)
        if chan is not None:
            return chan, "resolved from the gateway cache"
        try:
            chan = await bot.http.get_channel(cid)
            return chan, "resolved via REST (not in the gateway cache)"
        except discord.Forbidden:
            return None, (
                f"Discord refused to hand channel `{cid}` to the bot "
                "(`403 Missing Access`) — the bot cannot even **read** this channel."
            )
        except discord.NotFound:
            return None, f"Channel `{cid}` does not exist, or it is outside every guild the bot is in."
        except Exception as e:  # noqa: BLE001 - defensive
            return None, f"Channel lookup for `{cid}` failed: {e!r}"

    # Name lookup uses the guild channel cache the interaction already fetched.
    wanted = text.lower().lstrip("#")
    for chan in getattr(guild, "channels", []) or []:
        if str(getattr(chan, "name", "")).lower() == wanted:
            return chan, "resolved by name from the guild channel cache"
    return None, (
        f"No channel named `{text}` in the bot's cached channel list. "
        "Try the raw channel id (right-click the channel → Copy Link, the number "
        "after `/channels/`) — a channel the bot cannot view is never in that list."
    )


def _resolve_line(rep: PermsReport) -> str:
    """Step 2 of the lookup, phrased so a cache hit doesn't read like a REST call."""
    if rep.resolve_ok and rep.resolve_via == "cache":
        return "2. REST fallback → not needed (the cache hit answered it)"
    if rep.resolve_ok:
        return f"2. REST fallback → succeeded (`GET /channels/{rep.channel_id}`)"
    return f"2. REST fallback → **FAILED** — {rep.resolve_reason}"


def format_report(rep: PermsReport) -> str:
    """Render a report as one Discord-safe message body (≤ :data:`MAX_REPORT_CHARS`).

    Assembly order matters when a channel name or a note is absurdly long: the
    verdict and the footer are the parts that actually explain something, so the
    trimming eats optional permissions first, then notes, then the head — never
    the tail.
    """
    why = {attr: w for attr, _label, w in CHECKS}

    def mark(granted: bool | None) -> str:
        return "✅ allowed" if granted else ("❌ denied" if granted is False else "❓ unknown")

    head = [
        f"🔎 **Delivery-permission report — #{rep.name}**",
        f"Channel `{rep.channel_id}` · type: {rep.kind}"
        + (f" · parent: {rep.parent}" if rep.parent else ""),
        f"Bot: **{rep.bot_name}**"
        + (f" (`{rep.bot_id}`)" if rep.bot_id else "")
        + (" — guild member: yes" if rep.is_guild_member else " — guild member: **no**")
        + (f" · roles: {', '.join(rep.roles)}" if rep.roles else ""),
        "",
        "**Background senders resolve the channel like this:**",
        f"1. `bot.get_channel()` (gateway cache) → {'found' if rep.cache_hit else 'not cached'}",
        _resolve_line(rep),
        "",
        "**Permissions for the bot in this channel:**",
    ]
    required_lines = [
        f"• {label}: {mark(granted)} — {why.get(attr, '')} *(required)*"
        for attr, label, granted in rep.perms if attr in REQUIRED
    ]
    optional_lines = [
        f"• {label}: {mark(granted)} — {why.get(attr, '')}"
        for attr, label, granted in rep.perms if attr not in REQUIRED
    ]
    note_lines = [f"⚠️ {note}" for note in rep.notes]
    tail = ["", rep.verdict, "", _FOOTER]

    def build(*blocks: list[str]) -> str:
        parts = [b for b in blocks if b]
        return "\n\n".join("\n".join(p) for p in parts)

    body = build(head, required_lines, optional_lines, note_lines, tail)
    if len(body) > MAX_REPORT_CHARS:
        body = build(head, required_lines, note_lines, tail)
    if len(body) > MAX_REPORT_CHARS:
        body = build(head, required_lines, note_lines[:1], tail)
    if len(body) > MAX_REPORT_CHARS:
        body = build(head, required_lines, tail)
    if len(body) > MAX_REPORT_CHARS:
        # Still too long: the head itself is huge (an enormous channel name or
        # permission reason). Keep the answer, ellipsize the detail.
        tail_body = "\n".join(tail)
        ellipsis_line = "… (detail truncated) …"
        fixed = len(tail_body) + len(ellipsis_line) + 2   # the two "\n\n" separators
        room = max(MAX_REPORT_CHARS - fixed, 40)
        head_line = head[0] if len(head[0]) <= room else head[0][:room - 1] + "…"
        body = build([head_line], [ellipsis_line], tail)
    if len(body) > MAX_REPORT_CHARS:
        # The verdict + footer alone overflow the budget (a pathological reason
        # string). Keep the ANSWER and drop from the front instead.
        body = body[-MAX_REPORT_CHARS:]
    return body
