"""Tests for the global session store (bot_core.sessions)."""
from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot_core import sessions as S


@pytest.fixture(autouse=True)
def clean_state():
    """Reset in-memory session state around each test (persistence is already
    isolated to a temp file by the autouse fixture in conftest.py)."""
    S._state["session"] = None
    S._state["last_ended"] = None
    S._state["last_start_at"] = None
    S._state["next_session_reminders"] = []
    yield
    S._state["session"] = None
    S._state["last_ended"] = None
    S._state["last_start_at"] = None
    S._state["next_session_reminders"] = []


class TestStartSession:

    def test_start_creates_state_and_file(self):
        session, closed = S.start_session(name="My Session")
        assert closed is None
        assert session["name"] == "My Session"
        assert session["ended_at"] is None
        assert S.get_current_session() is session
        f = pathlib.Path(session["file"])
        assert f.exists()
        content = f.read_text(encoding="utf-8")
        assert "# Session: My Session" in content
        # Filename always carries date + increasing index.
        m = datetime.fromtimestamp(session["started_at"]).strftime("%Y-%m-%d")
        assert session["file"].endswith(".md")
        assert m in f.name

    def test_start_without_name(self):
        session, _ = S.start_session()
        assert session["name"] == ""
        assert "(no notes)" in pathlib.Path(session["file"]).read_text(encoding="utf-8")

    def test_second_start_within_hour_refused(self):
        S.start_session(name="A")
        with pytest.raises(ValueError, match="once per hour"):
            S.start_session(name="B")
        # State unchanged — still the first session.
        assert S.get_current_session()["name"] == "A"

    def test_start_after_ended_still_within_hour_refused(self):
        """The 1h rule is measured from the last START, even after a clean end."""
        S.start_session(name="A")
        S.end_session(overview="done")
        with pytest.raises(ValueError, match="once per hour"):
            S.start_session(name="B")

    def test_active_young_session_requires_end_first(self):
        S.start_session(name="A")
        # Simulate 2h elapsed (past the 1h cooldown, but session still active).
        S._state["last_start_at"] = time.time() - 2 * 3600
        S.get_current_session()["started_at"] = time.time() - 2 * 3600
        with pytest.raises(ValueError, match="end it first"):
            S.start_session(name="B")

    def test_stale_session_auto_ended(self):
        S.start_session(name="Old")
        old = S.get_current_session()
        # Simulate 13h elapsed — stale.
        old["started_at"] = time.time() - 13 * 3600
        S._state["last_start_at"] = time.time() - 13 * 3600
        new, closed_info = S.start_session(name="New")
        assert closed_info is not None and closed_info["kind"] == "stale"
        assert closed_info["session"]["name"] == "Old"
        assert closed_info["session"]["ended_at"] is not None
        # Old file records the end; new session is active.
        old_file = pathlib.Path(closed_info["session"]["file"])
        assert "- Ended:" in old_file.read_text(encoding="utf-8")
        assert S.get_current_session() is new

    def test_name_sanitized_for_filename(self):
        session, _ = S.start_session(name="../evil name?!  ")
        assert "/" not in session["file"].split("/")[-1]
        assert "?" not in pathlib.Path(session["file"]).name
        assert "evil" in pathlib.Path(session["file"]).name


class TestEndSession:

    def test_end_without_active_returns_none(self):
        assert S.end_session(overview="x") is None

    def test_end_stores_overview_and_writes_file(self):
        session, _ = S.start_session(name="S1")
        ended = S.end_session(overview="We did things.")
        assert ended["ended_at"] is not None
        assert ended["overview"] == "We did things."
        content = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        assert "## Overview" in content
        assert "We did things." in content
        assert S.get_current_session() is None

    def test_end_with_name_renames(self):
        session, _ = S.start_session(name="Old name")
        ended = S.end_session(overview=None, name="New name")
        assert ended["name"] == "New name"
        content = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        assert "# Session: New name" in content


class TestNotes:

    def test_add_note_requires_active_session(self):
        assert S.add_note("hello") is None

    def test_add_note_appends_and_writes_file(self):
        session, _ = S.start_session(name="N1")
        updated = S.add_note("remember the deploy", author="Alice")
        assert len(updated["notes"]) == 1
        ts, text = updated["notes"][0]
        assert isinstance(ts, float)
        assert "remember the deploy" in text
        assert "Alice" in text
        content = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        assert "remember the deploy" in content

    def test_refresh_notes_from_disk(self):
        """User edits the file on disk → state picks up the new notes."""
        session, _ = S.start_session(name="E1")
        f = pathlib.Path(session["file"])
        # Append a note line in the exact format the bot writes.
        ts = time.time()
        stamp = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        with f.open("a", encoding="utf-8") as fh:
            fh.write(f"\n- ({stamp}) manually added note\n")
        notes = S.refresh_notes_from_disk()
        assert any("manually added note" in t for _ts, t in notes)

    def test_view_last_session_notes(self):
        session, _ = S.start_session(name="V1")
        S.add_note("one")
        S.end_session(overview=None)
        assert S.get_current_session() is None
        last = S.get_last_session()
        assert last["name"] == "V1"
        notes = S.get_notes(last)
        assert len(notes) == 1


class TestNextSessionReminders:

    def test_queue_and_list(self):
        S.queue_next_session_reminder(111, "call mom")
        S.queue_next_session_reminder(222, "water plants")
        q = S.list_queued_reminders()
        assert len(q) == 2
        assert q[0]["channel_id"] == 111
        assert q[1]["message"] == "water plants"

    def test_cancel_by_index(self):
        S.queue_next_session_reminder(111, "a")
        S.queue_next_session_reminder(222, "b")
        assert S.cancel_queued_reminder(0) is True
        assert S.cancel_queued_reminder(5) is False
        assert [r["message"] for r in S.list_queued_reminders()] == ["b"]

    def test_persistence_survives_reload(self):
        """Queued reminders + active session survive a simulated restart."""
        S.start_session(name="Persist")
        S.add_note("survive me")
        S.queue_next_session_reminder(333, "still here?")

        # Simulate process restart: wipe memory, reload from disk.
        S._state["session"] = None
        S._state["last_ended"] = None
        S._state["last_start_at"] = None
        S._state["next_session_reminders"] = []
        S.load_persisted()

        assert S.get_current_session() is not None
        assert S.get_current_session()["name"] == "Persist"
        notes = S.get_notes()
        assert any("survive me" in t for _ts, t in notes)
        q = S.list_queued_reminders()
        assert len(q) == 1 and q[0]["channel_id"] == 333

    @pytest.mark.asyncio
    async def test_deliver_sends_to_channels_and_clears_queue(self):
        S.queue_next_session_reminder(111, "first")
        S.queue_next_session_reminder(222, "second")
        chan1 = MagicMock()
        chan1.send = AsyncMock()
        chan2 = MagicMock()
        chan2.send = AsyncMock()
        bot = MagicMock()
        bot.get_channel.side_effect = lambda cid: {111: chan1, 222: chan2}.get(cid)

        sent = await S.deliver_queued_reminders(bot)
        assert sent == 2
        chan1.send.assert_awaited_once()
        assert "first" in chan1.send.await_args.args[0]
        assert S.list_queued_reminders() == []

    @pytest.mark.asyncio
    async def test_deliver_keeps_unresolvable_channels_queued(self):
        import discord
        S.queue_next_session_reminder(111, "ok")
        S.queue_next_session_reminder(999, "gone channel")
        chan = MagicMock()
        chan.send = AsyncMock()
        bot = MagicMock()
        bot.get_channel.side_effect = lambda cid: chan if cid == 111 else None
        # REST fallback must report NotFound (the cache missed on purpose).
        async def http_get(cid):
            raise discord.NotFound("no such channel", response=None)
        bot.http.get_channel = AsyncMock(side_effect=http_get)

        sent = await S.deliver_queued_reminders(bot)
        assert sent == 1
        remaining = S.list_queued_reminders()
        # The unresolvable one stays queued (its channel may come back) and
        # its attempt counter is recorded so retries are bounded.
        assert len(remaining) == 1 and remaining[0]["message"] == "gone channel"
        assert remaining[0].get("attempts") == 1

    @pytest.mark.asyncio
    async def test_deliver_drops_after_repeated_failures(self):
        import discord
        S.queue_next_session_reminder(999, "always broken")
        bot = MagicMock()
        bot.get_channel.return_value = None
        async def http_get(cid):
            raise discord.NotFound("no such channel", response=None)
        bot.http.get_channel = AsyncMock(side_effect=http_get)

        from bot_core.reminders import MAX_DELIVERY_ATTEMPTS
        for _ in range(MAX_DELIVERY_ATTEMPTS):
            assert await S.deliver_queued_reminders(bot) == 0
        # After enough failures the entry is dropped instead of retrying forever.
        assert S.list_queued_reminders() == []

    @pytest.mark.asyncio
    async def test_deliver_empty_queue(self):
        bot = MagicMock()
        assert await S.deliver_queued_reminders(bot) == 0


class TestNaming:

    def test_index_increments_per_day(self):
        d = S.notes_dir()
        d.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        (d / f"{today}_01_existing.md").write_text("# old\n", encoding="utf-8")
        session, _ = S.start_session(name="Indexed")
        assert f"{today}_02_Indexed.md" == pathlib.Path(session["file"]).name

    def test_persistence_disabled_via_empty_env(self, monkeypatch):
        monkeypatch.setenv("SESSIONS_PERSIST_FILE", "")
        monkeypatch.setattr(S, "_store_path", None, raising=False)
        session, _ = S.start_session(name="NoPersist")
        assert S.get_current_session() is session  # in-memory still works
