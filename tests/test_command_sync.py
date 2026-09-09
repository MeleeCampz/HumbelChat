"""Unit tests for bot_core.command_sync (the real duplicate-command fix).

These mock the discord tree/client so they run offline and prove the *logic*:
stale guild-scoped commands and stale globals are deleted, and clean sets are
left alone. No real Discord calls are made.

Note: command objects are lightweight fakes exposing only the attributes the
sync code actually uses (``.name``, ``.id``, ``.guild_id``, ``.delete()``) so
the tests don't depend on discord.py's AppCommand payload schema.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import discord
import pytest

from bot_core import command_sync as cs


def _cmd(name: str, guild_id=None) -> SimpleNamespace:
    """A minimal registered-command object whose .delete() we can track."""
    calls: list = []

    async def _delete():
        calls.append(name)

    c = SimpleNamespace(name=name, id=abs(hash(name)) % (10**18), guild_id=guild_id)
    c.delete = _delete
    c._delete_calls = calls
    return c


class _FakeGuild:
    """Hashable guild stand-in (used as a dict key in the mock fetch_commands)."""

    def __init__(self, name="G", id=123):
        self.name = name
        self.id = id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, _FakeGuild) and self.id == other.id


def _guild(name="G"):
    return _FakeGuild(name=name, id=123)


def _bot(guilds, guild_commands, global_commands, live_names):
    bot = SimpleNamespace(guilds=guilds)
    calls = {"fetch_guild": [], "fetch_global": 0, "sync": 0}

    async def fetch_commands(*, guild=None):
        if guild is None:
            calls["fetch_global"] += 1
            return global_commands
        calls["fetch_guild"].append(guild)
        return guild_commands.get(guild, [])

    async def sync():
        calls["sync"] += 1
        return [_cmd(n) for n in live_names]

    bot.tree = SimpleNamespace(fetch_commands=fetch_commands, sync=sync)
    bot._calls = calls
    return bot


@pytest.mark.asyncio
async def test_purges_stale_guild_commands():
    guild = _guild()
    stale = [_cmd("ai", guild_id=123), _cmd("ocr", guild_id=123)]
    bot = _bot([guild], {guild: stale}, [], ["ai"])
    res = await cs.purge_guild_commands(bot)
    assert ("G", ["ai", "ocr"]) in res
    assert stale[0]._delete_calls == ["ai"]
    assert stale[1]._delete_calls == ["ocr"]
    assert bot._calls["fetch_guild"] == [guild]


@pytest.mark.asyncio
async def test_no_guild_commands_is_noop():
    guild = _guild()
    bot = _bot([guild], {guild: []}, [], ["ai"])
    res = await cs.purge_guild_commands(bot)
    assert res == []
    assert bot._calls["fetch_guild"] == [guild]


@pytest.mark.asyncio
async def test_purges_stale_globals_only():
    live = ["ai", "character"]
    registered = [_cmd("ai"), _cmd("character"), _cmd("oldcmd")]
    bot = _bot([], {}, registered, live)
    deleted = await cs.purge_stale_globals(bot)
    assert deleted == ["oldcmd"]
    assert registered[0]._delete_calls == []
    assert registered[1]._delete_calls == []
    assert registered[2]._delete_calls == ["oldcmd"]


@pytest.mark.asyncio
async def test_full_sync_reports_guilds_and_globals():
    guild = _guild()
    stale_guild = [_cmd("ai", guild_id=123)]
    live = ["ai", "sync"]
    registered = [_cmd("ai"), _cmd("sync")]
    bot = _bot([guild], {guild: stale_guild}, registered, live)
    report = await cs.sync_commands(bot)
    assert report["guilds"] == ["G"]
    assert report["global_deleted"] == []
    assert bot._calls["sync"] >= 1


def _bot_sync_fails(guilds, guild_commands, global_commands, exc):
    """Like _bot, but tree.sync() raises (e.g. HTTP 50035 bad payload)."""
    bot = _bot(guilds, guild_commands, global_commands, ["ai"])

    async def _raise():
        raise exc

    bot.tree.sync = _raise
    return bot


@pytest.mark.asyncio
async def test_purge_stale_globals_propagates_sync_failure():
    """A rejected upload (50035) must NOT be swallowed into an empty list —
    that made /sync report success while commands were never registered."""
    exc = discord.HTTPException(MagicMock(), "50035: Invalid Form Body")
    bot = _bot_sync_fails([], {}, [], exc)
    with pytest.raises(discord.HTTPException):
        await cs.purge_stale_globals(bot)


@pytest.mark.asyncio
async def test_full_sync_reports_error_instead_of_silence():
    """sync_commands surfaces the upload failure in report['error']."""
    exc = discord.HTTPException(MagicMock(), "50035: Invalid Form Body")
    bot = _bot_sync_fails([], {}, [], exc)
    report = await cs.sync_commands(bot)
    assert report["error"]
    assert report["global_deleted"] == []
    assert report["guild_names"] == []


@pytest.mark.asyncio
async def test_missing_delete_is_swallowed():
    guild = _guild()
    gone = _cmd("ghost", guild_id=123)

    async def _raise():
        raise discord.NotFound(MagicMock(), "gone")

    gone.delete = _raise
    alive = _cmd("real", guild_id=123)

    bot = _bot([guild], {guild: [gone, alive]}, [], ["ai"])
    res = await cs.purge_guild_commands(bot)
    # The ghost raises NotFound (swallowed, not counted); 'real' is deleted.
    assert ("G", ["real"]) in res
    assert alive._delete_calls == ["real"]


@pytest.mark.asyncio
async def test_sync_handler_shows_upload_failure():
    """When the upload is rejected, /sync must say FAILED — not 'synced'."""
    from unittest.mock import patch
    from tests._shared import Interaction
    from commands.sync_command import handle_sync_command

    ix = Interaction(client=MagicMock())  # handler reads interaction.client
    report = {
        "guilds": [], "guild_names": [], "global_deleted": [],
        "error": "Failed to upload commands to Discord (HTTP status 400, error code 50035)",
    }
    with patch.object(cs, "sync_commands", return_value=report):
        await handle_sync_command(ix)

    assert any("sync FAILED" in s for s in ix._sent)
    assert any("50035" in s for s in ix._sent)
    assert not any("Commands synced." in s for s in ix._sent)
