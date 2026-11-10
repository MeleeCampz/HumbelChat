"""Tests for the `/botperms` diagnostic (bot_core/perms_report + commands/perms_command).

Hermetic: every discord object is a fake. The *real* :func:`resolve_channel` runs
against those fakes, which is the point of the module — the report must reflect
the exact lookup path background senders take (cache, then REST).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import discord

from bot_core import perms_report as P
from bot_core.channel_delivery import ChannelNotDeliverableError


# ── Fakes ────────────────────────────────────────────────────────────────────

class FakePerms:
    """Only the attributes we set exist — missing ones test the defensive getattr."""

    def __init__(self, **granted: bool) -> None:
        self.__dict__.update(granted)

    def __getattr__(self, item: str):  # pragma: no cover - safety net
        raise AttributeError(item)


FULL_OK = dict(
    view_channel=True,
    send_messages=True,
    read_message_history=True,
    attach_files=True,
    send_messages_in_threads=True,
    create_public_threads=True,
    manage_threads=True,
    embed_links=True,
)


class FakeChannel:
    def __init__(self, cid=42, name="humblewood-chat", guild=None, perms=None,
                 ctype=discord.ChannelType.text, parent=None) -> None:
        self.id = cid
        self.name = name
        self.guild = guild
        self.type = ctype
        self.parent = parent
        self._perms = perms if perms is not None else FakePerms(**FULL_OK)

    def permissions_for(self, member) -> FakePerms:
        return self._perms


class FakeBot:
    def __init__(self, cached=None, rest=None, rest_exc=None) -> None:
        self.user = SimpleNamespace(id=7, display_name="Humbel", name="humbel")
        self._cached = cached
        self.http = SimpleNamespace(
            get_channel=AsyncMock(return_value=rest, side_effect=rest_exc)
        )

    def get_channel(self, cid):
        return self._cached if getattr(self._cached, "id", None) == cid else None


def _guild(me=None, channels=()) -> SimpleNamespace:
    return SimpleNamespace(me=me, channels=list(channels))


def _member(perms: FakePerms, role_names=("HumbelBot",)) -> SimpleNamespace:
    return SimpleNamespace(
        roles=[SimpleNamespace(name=n) for n in role_names],
    )


# ── build_report ─────────────────────────────────────────────────────────────

async def test_healthy_cached_channel_resolves_from_cache():
    guild = _guild(me=_member(FakePerms(**FULL_OK)))
    chan = FakeChannel(guild=guild)
    bot = FakeBot(cached=chan)

    rep = await P.build_report(bot, chan)

    assert rep.resolve_ok and rep.resolve_via == "cache"
    assert rep.cache_hit is True
    assert rep.verdict.startswith("✅")
    assert "gateway cache" in rep.verdict


async def test_cache_miss_but_rest_ok_reports_the_slower_path():
    guild = _guild(me=_member(FakePerms(**FULL_OK)))
    chan = FakeChannel(guild=guild)
    bot = FakeBot(cached=None, rest=chan)

    rep = await P.build_report(bot, chan)

    assert rep.cache_hit is False
    assert rep.resolve_ok and rep.resolve_via == "rest"
    assert "missing from the gateway cache" in rep.verdict
    assert "via rest" in rep.verdict


async def test_denied_view_channel_is_named_as_required_and_missing():
    perms = FakePerms(**{**FULL_OK, "view_channel": False})
    guild = _guild(me=_member(perms))
    chan = FakeChannel(guild=guild, perms=perms)

    rep = await P.build_report(bot=FakeBot(cached=chan), channel=chan)

    assert ("view_channel", "View Channel", False) in rep.perms
    assert rep.verdict.startswith("❌")
    assert "**View Channel**" in rep.verdict


async def test_rest_forbidden_verdict_names_the_background_features():
    guild = _guild(me=_member(FakePerms(**FULL_OK)))
    chan = FakeChannel(guild=guild)
    bot = FakeBot(cached=None, rest_exc=discord.Forbidden(MagicMock(), "Missing Access"))

    rep = await P.build_report(bot, chan)

    assert rep.resolve_ok is False
    assert "no access" in rep.resolve_reason
    assert rep.verdict.startswith("❌")
    assert "Background delivery will fail" in rep.verdict
    assert "highlights" in rep.verdict and "reminders" in rep.verdict


async def test_permission_attribute_that_does_not_exist_degrades_to_unknown():
    # Only two bits exist on this Permissions object; everything else must show
    # "unknown" instead of raising inside the diagnostic.
    perms = FakePerms(view_channel=True, send_messages=False)
    guild = _guild(me=_member(perms))
    chan = FakeChannel(guild=guild, perms=perms)

    rep = await P.build_report(FakeBot(cached=chan), chan)

    granted = {attr: g for attr, _label, g in rep.perms}
    assert granted["attach_files"] is None
    assert granted["send_messages"] is False
    body = P.format_report(rep)
    assert "❓ unknown" in body
    assert "**Send Messages**" in rep.verdict  # still flagged as the missing bit


async def test_forum_channel_is_flagged_as_not_messageable():
    guild = _guild(me=_member(FakePerms(**FULL_OK)))
    chan = FakeChannel(guild=guild, name="campaign-discussion",
                       ctype=discord.ChannelType.forum)
    rep = await P.build_report(FakeBot(cached=chan), chan)

    assert rep.kind == "forum"
    assert any("forum channel" in n and "not directly messageable" in n for n in rep.notes)


async def test_thread_report_names_the_parent_channel():
    guild = _guild(me=_member(FakePerms(**FULL_OK)))
    parent = FakeChannel(cid=11, name="posts", ctype=discord.ChannelType.forum, guild=guild)
    thread = FakeChannel(cid=12, name="session-10", guild=guild,
                         ctype=discord.ChannelType.public_thread, parent=parent)

    rep = await P.build_report(FakeBot(cached=thread), thread)

    assert rep.parent and "posts" in rep.parent
    assert any("ending channel id" in n for n in rep.notes)


async def test_missing_guild_member_is_reported_not_crashed():
    chan = FakeChannel(guild=_guild(me=None))
    rep = await P.build_report(FakeBot(cached=chan), chan)

    assert rep.is_guild_member is False
    assert all(g is None for _a, _l, g in rep.perms)
    assert any("member object is unavailable" in n for n in rep.notes)


def test_report_stays_inside_the_discord_message_limit():
    rep = P.PermsReport(
        channel_id=1,
        name="x" * 120,
        kind="public thread",
        parent="y" * 120 + " (2, forum)",
        bot_name="bot",
        bot_id=1,
        is_guild_member=True,
        roles=["r" * 40] * 10,
        perms=[(a, l, True) for a, l, _w in P.CHECKS],
        cache_hit=True,
        resolve_ok=True,
        resolve_via="cache",
        verdict="✅",
        notes=["n" * 300],
    )
    body = P.format_report(rep)
    assert len(body) <= P.MAX_REPORT_CHARS
    assert body.endswith(P._FOOTER)


# ── find_channel ─────────────────────────────────────────────────────────────

async def test_find_channel_by_id_mention_and_name():
    chan = FakeChannel(cid=99, name="humblewood-chat")
    guild = _guild(channels=[chan])
    bot = FakeBot(cached=chan)

    for ref in ("99", "<#99>", "#99", "humblewood-chat", "HUMBLEWOOD-CHAT"):
        found, note = await P.find_channel(bot, guild, ref)
        assert found is chan, ref
        assert note


async def test_find_channel_reports_the_refusal_reason():
    guild = _guild(channels=[])
    bot = FakeBot(cached=None, rest_exc=discord.Forbidden(MagicMock(), "Missing Access"))

    found, note = await P.find_channel(bot, guild, "869602082552954973")
    assert found is None
    assert "403" in note and "read" in note

    found, note = await P.find_channel(bot, guild, "no-such-channel")
    assert found is None
    assert "cached channel list" in note


# ── the command handler ──────────────────────────────────────────────────────

def _interaction(guild, chan, client):
    it = MagicMock()
    it.client = client
    it.guild = guild
    it.channel = chan
    it.user = SimpleNamespace(id=5, name="user")
    it.response.defer = AsyncMock()
    it.followup.send = AsyncMock()
    return it


async def test_botperms_reports_the_invoking_channel(monkeypatch):
    from commands.perms_command import handle_botperms_command

    guild = _guild(me=_member(FakePerms(**FULL_OK)))
    chan = FakeChannel(guild=guild)
    it = _interaction(guild, chan, FakeBot(cached=chan))

    await handle_botperms_command(it)

    body = it.followup.send.await_args.args[0]
    assert "Delivery-permission report" in body
    assert "humblewood-chat" in body
    assert it.response.defer.await_count == 1


async def test_botperms_unresolvable_channel_ref_is_its_own_answer(monkeypatch):
    from commands.perms_command import handle_botperms_command

    guild = _guild(me=_member(FakePerms(**FULL_OK)), channels=[])
    it = _interaction(guild, FakeChannel(guild=guild), FakeBot(
        cached=None, rest_exc=discord.Forbidden(MagicMock(), "Missing Access")))

    await handle_botperms_command(it, channel="869602082552954973")

    body = it.followup.send.await_args.args[0]
    assert "Could not inspect" in body and "403" in body


async def test_botperms_without_channel_object_asks_for_an_id():
    from commands.perms_command import handle_botperms_command

    guild = _guild(me=_member(FakePerms(**FULL_OK)))
    it = _interaction(guild, None, FakeBot())

    await handle_botperms_command(it)

    body = it.followup.send.await_args.args[0]
    assert "channel id" in body


async def test_botperms_in_a_dm_explains_it_needs_a_guild():
    from commands.perms_command import handle_botperms_command

    chan = FakeChannel(guild=None)
    it = _interaction(None, chan, FakeBot())

    await handle_botperms_command(it)

    body = it.followup.send.await_args.args[0]
    assert "guild" in body


def test_even_a_runaway_head_keeps_the_verdict_and_footer():
    """Pathological case: the head alone blows the budget — the tail must survive."""
    rep = P.PermsReport(
        channel_id=1,
        name="z" * 1500,
        kind="public thread " + "q" * 300,
        parent="p" * 300 + " (2, forum)",
        bot_name="bot",
        bot_id=1,
        is_guild_member=True,
        roles=["r" * 60] * 10,
        perms=[(a, l, False) for a, l, _w in P.CHECKS],
        cache_hit=False,
        resolve_ok=False,
        resolve_via="failed",
        resolve_reason="boom " + "x" * 900,
        notes=["n" * 400] * 3,
        verdict="❌ " + ("detail " * 120),
    )
    body = P.format_report(rep)
    assert len(body) <= P.MAX_REPORT_CHARS
    assert body.endswith(P._FOOTER)


def test_a_verdict_bigger_than_the_budget_still_ends_with_the_footer():
    """Last-resort branch: drop from the FRONT so the verdict/footer survive."""
    rep = P.PermsReport(
        channel_id=1, name="c", kind="text", bot_name="bot", bot_id=1,
        is_guild_member=True, perms=[(a, l, False) for a, l, _w in P.CHECKS],
        cache_hit=False, resolve_ok=False, resolve_via="failed",
        verdict="❌ " + ("reason " * 600),
    )
    body = P.format_report(rep)
    assert len(body) <= P.MAX_REPORT_CHARS
    assert body.endswith(P._FOOTER)


def test_resolve_line_phrases_each_lookup_outcome():
    ok_cache = P.PermsReport(1, resolve_ok=True, resolve_via="cache")
    ok_rest = P.PermsReport(2, resolve_ok=True, resolve_via="rest")
    failed = P.PermsReport(3, resolve_ok=False, resolve_via="failed", resolve_reason="403")

    assert "not needed" in P._resolve_line(ok_cache)
    assert "succeeded" in P._resolve_line(ok_rest) and "GET /channels/2" in P._resolve_line(ok_rest)
    assert P._resolve_line(failed).endswith("403") and "FAILED" in P._resolve_line(failed)
