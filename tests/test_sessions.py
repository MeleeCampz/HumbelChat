"""Tests for the global session store (bot_core.sessions)."""
from __future__ import annotations

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

    def test_start_creates_state_and_folder(self):
        session, closed = S.start_session(name="My Session")
        assert closed is None
        assert session["name"] == "My Session"
        assert session["ended_at"] is None
        assert S.get_current_session() is session
        f = pathlib.Path(session["file"])
        assert f.exists()
        assert f.name == "notes.md"  # per-session folder
        # Folder carries date + increasing index + name
        m = datetime.fromtimestamp(session["started_at"]).strftime("%Y-%m-%d")
        d = f.parent
        assert d.name.startswith(m)
        assert "My Session" in d.name
        assert session["dir"] == str(d)
        content = f.read_text(encoding="utf-8")
        assert "# Session: My Session" in content

    def test_start_without_name(self):
        session, _ = S.start_session()
        assert session["name"] == ""
        f = pathlib.Path(session["file"])
        assert f.name == "notes.md"
        assert "(no notes)" in f.read_text(encoding="utf-8")

    def test_start_while_active_refused(self):
        """Only one session at a time — an active (young) session blocks a new start."""
        S.start_session(name="A")
        with pytest.raises(ValueError, match="end it first"):
            S.start_session(name="B")
        # State unchanged — still the first session.
        assert S.get_current_session()["name"] == "A"

    def test_start_right_after_end_allowed(self):
        """No cooldown: a new session can start immediately after a clean end."""
        S.start_session(name="A")
        S.end_session(overview="done")
        new, _ = S.start_session(name="B")
        assert S.get_current_session() is new
        assert new["name"] == "B"

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

    def test_name_sanitized_for_foldername(self):
        session, _ = S.start_session(name="../evil name?!  ")
        folder = pathlib.Path(session["dir"])
        assert "/" not in folder.name
        assert "?" not in folder.name
        assert "evil" in folder.name


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


class TestAddDocument:
    """sessions.add_document() — an uploaded .txt/.md file → its own file in the
    session's attachments/ folder (no pre-chunking; KB chunks it on its own)."""

    def test_no_active_session_returns_none(self):
        session, n = S.add_document("hello")
        assert session is None and n == 0

    def test_empty_text_is_noop(self):
        S.start_session(name="T")
        session, n = S.add_document("   ")
        assert n == 0
        assert S.get_notes(session) == []

    def test_document_written_as_own_file(self):
        S.start_session(name="D")
        session, n = S.add_document("the answer is 42", title="answer.txt")
        assert n == 1
        # A standalone .md file exists under the session's attachments/ folder
        att_dir = pathlib.Path(session["dir"]) / "attachments"
        files = [f for f in att_dir.iterdir() if f.is_file()]
        assert len(files) == 1
        body = files[0].read_text(encoding="utf-8")
        assert "the answer is 42" in body
        # The notes state is unchanged (no pre-chunking, no pointer stored).
        assert S.get_notes(session) == []
        # notes.md lists the attachment under a pointer section
        content = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        assert "## Attachments" in content
        assert files[0].name in content

    def test_default_title_used(self):
        S.start_session(name="D")
        session, n = S.add_document("just words")
        assert n == 1
        att_dir = pathlib.Path(session["dir"]) / "attachments"
        assert len(list(att_dir.iterdir())) == 1
        assert "just words" in att_dir.iterdir().__next__().read_text(encoding="utf-8")

    def test_long_document_not_chunked(self):
        S.start_session(name="D")
        long_text = ("word " * 2000).strip()  # ~10k chars — stays in ONE file
        session, n = S.add_document(long_text, title="big.md")
        assert n == 1  # one file, not several bullets
        att_dir = pathlib.Path(session["dir"]) / "attachments"
        files = [f for f in att_dir.iterdir() if f.is_file()]
        assert len(files) == 1
        body = files[0].read_text(encoding="utf-8")
        # The whole document is preserved verbatim (KB will chunk it).
        assert long_text in body

    def test_pins_to_given_session(self):
        """*session* pins the target — the document lands in THAT session's folder."""
        S.start_session(name="Old")
        old = S.get_current_session()
        S.end_session(overview="done")
        S._state["last_start_at"] -= 2 * 3600
        S.start_session(name="New")

        session, n = S.add_document("pinned doc", title="doc.md", session=old)
        assert n == 1 and session is old
        att_dir = pathlib.Path(old["dir"]) / "attachments"
        assert any("pinned doc" in f.read_text(encoding="utf-8") for f in att_dir.iterdir())
        # The new active session's folder has no attachment
        new_dir = pathlib.Path(S.get_current_session()["dir"]) / "attachments"
        assert not new_dir.exists() or len(list(new_dir.iterdir())) == 0

    def test_repeated_uploads_get_distinct_files(self):
        S.start_session(name="D")
        S.add_document("one", title="same.txt")
        S.add_document("two", title="same.txt")
        att_dir = pathlib.Path(S.get_current_session()["dir"]) / "attachments"
        names = sorted(f.name for f in att_dir.iterdir())
        assert len(names) == 2  # unique_path disambiguates duplicates


class TestAddTranscript:
    """sessions.add_transcript() — a finished transcript → its own file in the
    session's transcripts/ folder (no pre-chunking; KB chunks it on its own)."""

    def test_no_active_session_returns_none(self):
        session, n = S.add_transcript("hello")
        assert session is None and n == 0

    def test_empty_text_is_noop(self):
        S.start_session(name="T")
        session, n = S.add_transcript("   ")
        assert n == 0
        assert S.get_notes(session) == []

    def test_transcript_written_as_own_file(self):
        S.start_session(name="T")
        session, n = S.add_transcript("hello there\nsecond line",
                                      title="Voice channel transcript — #vc (2026-08-31 22:00, 10s)")
        assert n == 1
        tdir = pathlib.Path(session["dir"]) / "transcripts"
        files = [f for f in tdir.iterdir() if f.is_file()]
        assert len(files) == 1
        body = files[0].read_text(encoding="utf-8")
        assert "hello there" in body and "second line" in body
        assert "Voice channel transcript" in body
        # notes state unchanged; notes.md lists the transcript pointer
        assert S.get_notes(session) == []
        content = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        assert "## Transcripts" in content and files[0].name in content

    def test_long_transcript_not_chunked(self):
        S.start_session(name="T")
        long_text = ("word " * 2000).strip()  # ~10k chars — stays in ONE file
        session, n = S.add_transcript(long_text, title="Voice channel transcript")
        assert n == 1
        tdir = pathlib.Path(session["dir"]) / "transcripts"
        files = [f for f in tdir.iterdir() if f.is_file()]
        assert len(files) == 1
        assert long_text in files[0].read_text(encoding="utf-8")

    def test_pins_to_ended_session_when_pinned(self):
        S.start_session(name="Old")
        old = S.get_current_session()
        S.end_session(overview="done")
        S._state["last_start_at"] -= 2 * 3600
        S.start_session(name="New")

        session, n = S.add_transcript("pinned words", title="Voice channel transcript",
                                      session=old)
        assert n == 1 and session is old
        tdir = pathlib.Path(old["dir"]) / "transcripts"
        assert any("pinned words" in f.read_text(encoding="utf-8") for f in tdir.iterdir())


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
        (d / f"{today}_01_existing").mkdir()  # a pre-existing session folder
        session, _ = S.start_session(name="Indexed")
        assert pathlib.Path(session["dir"]).name == f"{today}_02_Indexed"

    def test_persistence_disabled_via_empty_env(self, monkeypatch):
        monkeypatch.setenv("SESSIONS_PERSIST_FILE", "")
        monkeypatch.setattr(S, "_store_path", None, raising=False)
        session, _ = S.start_session(name="NoPersist")
        assert S.get_current_session() is session  # in-memory still works
