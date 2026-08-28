"""Tests for bot_core.channel_delivery + reminder delivery semantics.

Covers the 2026-08-28 root cause: reminders must not be silently skipped
when the target channel is missing from the local cache, and a failed
delivery must never be marked "fired".

Hermeticity notes:
  * The /remind guard tests stub ``bot_core.reminders.schedule_reminder`` so
    no real asyncio task is spawned (a live background sleep would outlive
    the test and pollute later ones).
  * ``main.bot`` mutations in the _fire tests are always restored.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import discord

from bot_core.channel_delivery import (
    ChannelNotDeliverableError,
    can_post_in_channel,
    get_bot,
    send_to_channel,
)


@pytest.fixture
def store_path(tmp_path, monkeypatch):
    """Point the reminder store at a temp file and clear shared state."""
    from bot_core import reminders as R
    p = tmp_path / "reminders_cd.json"
    monkeypatch.setenv("REMINDERS_PERSIST_FILE", str(p))
    monkeypatch.setattr(R, "_store_path", None, raising=False)
    R._reminders.clear()
    R._tasks.clear()
    R._retry_tasks.clear()
    yield p
    # Cancel any in-session retry tasks so no 30s/120s sleep outlives the test.
    for t in list(R._retry_tasks.values()):
        try:
            t.cancel()
        except Exception:
            pass
    R._reminders.clear()
    R._tasks.clear()
    R._retry_tasks.clear()


def _forbidden():
    # Real discord.py signature: HTTPException(response, message).
    return discord.Forbidden(MagicMock(), "Missing Access")


def _not_found():
    return discord.NotFound(MagicMock(), "no such channel")


def _bot(cache_get=None, rest_get=None, ready=True):
    bot = MagicMock()
    # get_bot() only considers a bot usable once logged in (user is set).
    bot.user = MagicMock()
    bot.is_ready.return_value = ready
    if cache_get is not None:
        bot.get_channel = MagicMock(side_effect=cache_get)
    else:
        bot.get_channel.return_value = None
    if rest_get is not None:
        bot.http.get_channel = AsyncMock(side_effect=rest_get)
    else:
        bot.http.get_channel = AsyncMock()  # trackable: assert_not_awaited()
    return bot


class TestGetBot:
    """get_bot must find the LIVE logged-in bot from sys.modules WITHOUT
    re-importing main — a lazy `from main import bot` inside a running script
    re-executes main.py's module level and returns a fresh, never-logged-in
    duplicate (the 2026-08-28 "no bot object available" incident)."""

    def test_returns_logged_in_bot_from_main_module(self, monkeypatch):
        fake = types.ModuleType("fake_main")
        live = _bot()
        fake.bot = live
        real = sys.modules.get("main")
        try:
            monkeypatch.setitem(sys.modules, "main", fake)
            assert get_bot() is live
        finally:
            if real is not None:
                sys.modules["main"] = real

    def test_skips_unlogged_duplicate_and_finds_logged_in_one(self):
        """Encodes the 2026-08-28 incident: a fresh duplicate module (bot.user is
        None — what a re-executed main.py looks like) must be SKIPPED, and the
        logged-in bot cached in another module wins. This is exactly why get_bot
        checks ``user`` instead of trusting the first ``bot`` it finds."""
        dup = types.ModuleType("fake_dup")
        fresh = MagicMock()
        fresh.user = None          # re-executed main.py: never logged in
        dup.bot = fresh
        live_mod = types.ModuleType("fake_live")
        live = _bot()              # user is set → usable
        live_mod.bot = live
        saved = {k: sys.modules.get(k) for k in ("main", "__main__")}
        try:
            sys.modules["__main__"] = dup       # duplicate shadows __main__
            sys.modules["main"] = live_mod      # real logged-in bot cached elsewhere
            assert get_bot() is live, "logged-in bot must win over fresh duplicate"
        finally:
            for k, v in saved.items():
                if v is not None:
                    sys.modules[k] = v
                else:
                    sys.modules.pop(k, None)

    def test_returns_none_when_no_logged_in_bot_and_import_blocked(self):
        """With no cached module exposing a logged-in bot AND ``import main``
        blocked (sys.modules['main'] = None → ImportError), get_bot must
        return None — it never raises and the miss was logged."""
        real_main = sys.modules.get("main")     # conftest's import
        main_mod = sys.modules.get("__main__")
        had_bot = hasattr(main_mod, "bot") if main_mod else False
        saved_bot = getattr(main_mod, "bot", None) if (main_mod and had_bot) else None
        try:
            if main_mod is not None and had_bot:
                delattr(main_mod, "bot")       # __main__ offers no usable bot either
            sys.modules["main"] = None          # blocks `import main` in get_bot
            assert get_bot() is None
        finally:
            sys.modules.pop("main", None)
            if real_main is not None:
                sys.modules["main"] = real_main
            if main_mod is not None and had_bot:
                main_mod.bot = saved_bot


class TestSendToChannel:

    @pytest.mark.asyncio
    async def test_fast_path_cache_hit(self):
        chan = MagicMock()
        chan.send = AsyncMock()
        bot = _bot(cache_get=lambda cid: chan)
        result = await send_to_channel(bot, 111, "hello")
        assert result is chan
        chan.send.assert_awaited_once_with("hello")
        bot.http.get_channel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rest_fallback_when_cache_misses(self):
        """Core regression: a channel missing from the local cache must still
        be reached via REST instead of being silently skipped."""
        fetched = MagicMock()  # what http.get_channel returns
        fetched.send = AsyncMock()
        bot = _bot(cache_get=lambda cid: None, rest_get=lambda cid: fetched)
        result = await send_to_channel(bot, 222, "hi")
        assert result is fetched
        bot.http.get_channel.assert_awaited_once_with(222)
        fetched.send.assert_awaited_once_with("hi")

    @pytest.mark.asyncio
    async def test_no_bot_instance_raises(self):
        with pytest.raises(ChannelNotDeliverableError):
            await send_to_channel(None, 333, "x")

    @pytest.mark.asyncio
    async def test_forbidden_on_lookup_raises_clean_error(self):
        bot = _bot(cache_get=lambda cid: None,
                   rest_get=lambda cid: (_ for _ in ()).throw(_forbidden()))
        with pytest.raises(ChannelNotDeliverableError) as ei:
            await send_to_channel(bot, 444, "x")
        assert "no access" in str(ei.value).lower()

    @pytest.mark.asyncio
    async def test_not_found_on_lookup_raises_clean_error(self):
        bot = _bot(cache_get=lambda cid: None,
                   rest_get=lambda cid: (_ for _ in ()).throw(_not_found()))
        with pytest.raises(ChannelNotDeliverableError) as ei:
            await send_to_channel(bot, 555, "x")
        assert "does not exist" in str(ei.value)

    @pytest.mark.asyncio
    async def test_forbidden_on_send_raises_clean_error(self):
        chan = MagicMock()
        chan.guild = None
        chan.name = "private-bot"
        chan.send = AsyncMock(side_effect=_forbidden())
        bot = _bot(cache_get=lambda cid: chan)
        with pytest.raises(ChannelNotDeliverableError) as ei:
            await send_to_channel(bot, 666, "x")
        assert "cannot post" in str(ei.value).lower()


class TestCanPostInChannel:

    def _ix(self, view: bool, send: bool, guild_id=1, channel=None):
        ix = MagicMock()
        if channel is None:
            channel = MagicMock()
            perm = MagicMock()
            perm.view_channel = view
            perm.send_messages = send
            channel.permissions_for.return_value = perm
        ix.channel = channel
        ix.guild_id = guild_id
        if guild_id:
            ix.guild.me = MagicMock()
        return ix

    @pytest.mark.asyncio
    async def test_allows_when_permissions_present(self):
        assert await can_post_in_channel(self._ix(True, True)) is True

    @pytest.mark.asyncio
    async def test_denies_without_view_channel(self):
        """The exact private-bot scenario: View Channel denied on @everyone."""
        assert await can_post_in_channel(self._ix(False, True)) is False

    @pytest.mark.asyncio
    async def test_denies_without_send_messages(self):
        assert await can_post_in_channel(self._ix(True, False)) is False

    @pytest.mark.asyncio
    async def test_dm_is_permissive(self):
        ix = self._ix(True, True, guild_id=None)
        assert await can_post_in_channel(ix) is True

    @pytest.mark.asyncio
    async def test_missing_channel_denies(self):
        ix = self._ix(True, True)
        ix.channel = None
        assert await can_post_in_channel(ix) is False


class TestReminderFireDelivery:
    """_fire() must only mark a reminder fired after real delivery, and a
    failed delivery must keep it queued (bounded retries)."""

    @pytest.mark.asyncio
    async def test_success_marks_fired(self, store_path):
        from bot_core import reminders as R
        chan = MagicMock()
        chan.send = AsyncMock()
        bot = _bot(cache_get=lambda cid: chan)

        import main
        old = getattr(main, "bot", None)
        try:
            main.bot = bot
            rid = "abcd1234"
            R._reminders[rid] = {"channel_id": 9, "message": "m", "delay_sec": 0,
                                 "fires_at": 0, "created_at": 0, "fired": False}
            await R._fire(rid, 9, "m", 0)  # noqa: SLF001 - whitebox
            assert R._reminders[rid]["fired"] is True, "must only fire on success"
        finally:
            if old is not None:
                main.bot = old
            R._reminders.pop(rid, None)

        chan.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failure_keeps_queued_and_counts_attempt(self, store_path):
        from bot_core import reminders as R
        bot = _bot(cache_get=lambda cid: None,
                   rest_get=lambda cid: (_ for _ in ()).throw(_forbidden()))

        import main
        old = getattr(main, "bot", None)
        rid = "efgh5678"
        try:
            main.bot = bot
            R._reminders[rid] = {"channel_id": 999, "message": "will fail",
                                 "delay_sec": 0, "fires_at": 0,
                                 "created_at": 0, "fired": False}
            await R._fire(rid, 999, "will fail", 0)  # noqa: SLF001
            assert rid in R._reminders, "failed reminder must stay queued"
            assert R._reminders[rid]["fired"] is False
            assert R._reminders[rid]["attempts"] == 1
        finally:
            if old is not None:
                main.bot = old
            R._reminders.pop(rid, None)

    @pytest.mark.asyncio
    async def test_failure_drops_after_max_attempts(self, store_path):
        from bot_core import reminders as R
        bot = _bot(cache_get=lambda cid: None,
                   rest_get=lambda cid: (_ for _ in ()).throw(_not_found()))

        import main
        old = getattr(main, "bot", None)
        rid = "ijkl9012"
        try:
            main.bot = bot
            R._reminders[rid] = {"channel_id": 998, "message": "doomed",
                                 "delay_sec": 0, "fires_at": 0,
                                 "created_at": 0, "fired": False}
            for _ in range(R.MAX_DELIVERY_ATTEMPTS):
                if rid not in R._reminders:
                    break
                await R._fire(rid, 998, "doomed", 0)  # noqa: SLF001
            assert rid not in R._reminders, "must be dropped after max attempts"
        finally:
            if old is not None:
                main.bot = old
            R._reminders.pop(rid, None)

    @pytest.mark.asyncio
    async def test_transient_failure_schedules_in_session_retry(self, store_path, monkeypatch):
        """2026-08-28 regression: a fire that hits an unavailable bot
        (gateway blip / mid-reconnect / teardown) must schedule an in-session
        retry (30s, then 120s) instead of only waiting for the next restart.

        NOTE: a connected-but-is_ready()-False bot is NOT transient here —
        discord.py sets that flag after dispatching READY, so sends succeed
        during on_ready and the flag-lag window.  See channel_delivery docstring.
        """
        from bot_core import reminders as R
        monkeypatch.setattr(R, "_resolve_bot", lambda: None)  # bot unavailable
        rid = "trans001"
        try:
            R._reminders[rid] = {"channel_id": 7, "message": "m", "delay_sec": 0,
                                 "fires_at": 0, "created_at": 0, "fired": False}
            await R._fire(rid, 7, "m", 0)  # noqa: SLF001
            assert rid in R._reminders, "transient failure must keep it queued"
            assert R._reminders[rid]["attempts"] == 1
            assert rid in R._retry_tasks, "in-session retry must be scheduled"
        finally:
            R.cancel_reminder(rid)
            R._reminders.pop(rid, None)

    @pytest.mark.asyncio
    async def test_hard_failure_does_not_schedule_in_session_retry(self, store_path):
        from bot_core import reminders as R
        import main
        bot = _bot(cache_get=lambda cid: None,
                   rest_get=lambda cid: (_ for _ in ()).throw(_forbidden()))
        old = getattr(main, "bot", None)
        rid = "hard0002"
        try:
            main.bot = bot
            R._reminders[rid] = {"channel_id": 8, "message": "m", "delay_sec": 0,
                                 "fires_at": 0, "created_at": 0, "fired": False}
            await R._fire(rid, 8, "m", 0)  # noqa: SLF001
            assert rid in R._reminders
            assert rid not in R._retry_tasks, "permission failures must not auto-retry"
        finally:
            if old is not None:
                main.bot = old
            R.cancel_reminder(rid)
            R._reminders.pop(rid, None)

    @pytest.mark.asyncio
    async def test_retry_recovers_when_bot_returns(self, store_path, monkeypatch):
        """The whole point of the in-session retry: a bot-availability blip at
        fire time must not lose the reminder — once the bot is back (and the
        backoff elapses) the same reminder delivers normally."""
        from bot_core import reminders as R
        monkeypatch.setattr(R, "_RETRY_DELAYS", (0,))  # no real waiting in tests
        rid = "rec0v3r4"
        state = {"bot": None}                          # unavailable during fire #1
        monkeypatch.setattr(R, "_resolve_bot", lambda: state["bot"])
        try:
            R._reminders[rid] = {"channel_id": 7, "message": "recover me", "delay_sec": 0,
                                 "fires_at": 0, "created_at": 0, "fired": False}
            await R._fire(rid, 7, "recover me", 0)  # noqa: SLF001
            assert rid in R._retry_tasks, "retry must be pending"

            state["bot"] = _bot(cache_get=lambda cid: MagicMock(send=AsyncMock()))
            await asyncio.sleep(0.05)                 # let sleep(0) retry + fire run
            assert R._reminders[rid]["fired"] is True, "retry must deliver after recovery"
        finally:
            R.cancel_reminder(rid)
            R._reminders.pop(rid, None)

    def test_is_transient_classification(self):
        from bot_core import reminders as R
        # Process/connection-layer failures: retry in-session.
        assert R._is_transient("no ready bot instance available to deliver the message")
        assert R._is_transient("no bot object available to deliver")
        assert R._is_transient(
            "channel lookup failed (ServerDisconnectedError('connection closed by peer'))")
        # Channel-level errors: keep failing identically -> NOT transient.
        assert not R._is_transient("channel 123 not deliverable: bot has no access to this channel (permission denied)")
        assert not R._is_transient("channel does not exist or is outside every guild the bot is in")
        assert not R._is_transient("bot cannot post in #x — check View Channel / Send Messages permissions and role overwrites")


class TestRemindCommandPermissionGuard:

    @pytest.mark.asyncio
    async def test_refuses_when_bot_cannot_post(self, ix):
        """Regression: /remind in a channel where the bot lacks View Channel
        must be refused up front instead of scheduling an undeliverable
        reminder. schedule_reminder is stubbed so nothing real is scheduled."""
        from commands.utility_commands import handle_remind_command

        channel = MagicMock()
        channel.id = 123456
        perm = MagicMock()
        perm.view_channel = False
        perm.send_messages = True
        channel.permissions_for.return_value = perm
        ix.channel = channel
        ix.guild_id = 1
        ix.guild.me = MagicMock()
        ix.channel_id = 123456

        with patch("bot_core.reminders.schedule_reminder") as sched:
            await handle_remind_command(ix, time_value=30, time_unit="seconds",
                                        message="never delivered")
        assert any("can't send messages" in m for m in ix._sent)
        sched.assert_not_called()

    @pytest.mark.asyncio
    async def test_schedules_when_bot_can_post(self, ix):
        from commands.utility_commands import handle_remind_command

        channel = MagicMock()
        channel.id = 123456
        perm = MagicMock()
        perm.view_channel = True
        perm.send_messages = True
        channel.permissions_for.return_value = perm
        ix.channel = channel
        ix.guild_id = 1
        ix.guild.me = MagicMock()
        ix.channel_id = 123456

        with patch("bot_core.reminders.schedule_reminder",
                   return_value="stub-rid") as sched:
            await handle_remind_command(ix, time_value=30, time_unit="seconds",
                                        message="will deliver")
        assert any("Reminder set for" in m for m in ix._sent)
        sched.assert_called_once()
        args = sched.call_args.args
        assert args[0] == 123456 and "will deliver" in args[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
