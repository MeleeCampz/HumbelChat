"""Tests for the session slash-command handlers (commands.session_commands)."""
from __future__ import annotations

import pathlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot_core import sessions as S
from tests._shared import make_interaction


@pytest.fixture(autouse=True)
def clean_state():
    S._state["session"] = None
    S._state["last_ended"] = None
    S._state["last_start_at"] = None
    S._state["next_session_reminders"] = []
    yield
    S._state["session"] = None
    S._state["last_ended"] = None
    S._state["last_start_at"] = None
    S._state["next_session_reminders"] = []


def _mock_ai(content="We did great work today.", fail=False):
    """Patch the AI client used by session_commands."""
    inst = MagicMock()
    if fail:
        inst.chat.completions.create = AsyncMock(side_effect=RuntimeError("backend down"))
    else:
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=content))]
        inst.chat.completions.create = AsyncMock(return_value=resp)
    return patch("commands.session_commands._make_client", return_value=inst), \
           patch("commands.session_commands._validate_model", new=AsyncMock(return_value="test-model"))


class TestStartSessionCommand:

    @pytest.mark.asyncio
    async def test_start_success(self, ix):
        from commands.session_commands import handle_start_session
        await handle_start_session(ix, name="Kickoff")
        assert any("Session started" in m and "Kickoff" in m for m in ix._sent)
        assert S.get_current_session() is not None

    @pytest.mark.asyncio
    async def test_start_while_active_refused(self, ix):
        from commands.session_commands import handle_start_session
        S.start_session(name="A")
        await handle_start_session(ix, name="B")
        assert any("end it first" in m for m in ix._sent)
        assert S.get_current_session()["name"] == "A"

    @pytest.mark.asyncio
    async def test_start_refused_while_active(self, ix):
        from commands.session_commands import handle_start_session
        S.start_session(name="A")
        S._state["last_start_at"] = time.time() - 2 * 3600
        S.get_current_session()["started_at"] = time.time() - 2 * 3600
        await handle_start_session(ix, name="B")
        assert any("end it first" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_start_delivers_previous_overview_and_reminders(self, ix):
        from commands.session_commands import handle_start_session
        # Previous session with an overview + a queued reminder.
        S.start_session(name="Prev")
        S.end_session(overview="Did the old stuff.")
        S._state["last_start_at"] = time.time() - 2 * 3600  # past cooldown
        S.queue_next_session_reminder(777, "bring coffee")

        chan = MagicMock()
        chan.send = AsyncMock()
        bot = MagicMock()
        bot.get_channel.side_effect = lambda cid: chan if cid == 777 else None

        with patch("commands.session_commands._get_bot", return_value=bot):
            await handle_start_session(ix, name="Next")

        assert any("Overview of previous session" in m and "Did the old stuff." in m for m in ix._sent)
        assert any("Delivered 1 queued next-session reminder" in m for m in ix._sent)
        chan.send.assert_awaited_once()
        assert "bring coffee" in chan.send.await_args.args[0]
        assert S.list_queued_reminders() == []

    @pytest.mark.asyncio
    async def test_start_stale_session_note(self, ix):
        from commands.session_commands import handle_start_session
        S.start_session(name="Ancient")
        old = S.get_current_session()
        old["started_at"] = time.time() - 13 * 3600
        S._state["last_start_at"] = time.time() - 13 * 3600
        await handle_start_session(ix, name="Fresh")
        assert any("auto-ended" in m for m in ix._sent)


class TestEndSessionCommand:

    @pytest.mark.asyncio
    async def test_end_without_active(self, ix):
        from commands.session_commands import handle_end_session
        await handle_end_session(ix)
        assert any("no active session" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_end_with_ai_overview(self, ix):
        from commands.session_commands import handle_end_session
        S.start_session(name="Workday")
        S.add_note("shipped the feature", author="Alice")
        p1, p2 = _mock_ai(content="Shipped X; follow up on Y.")
        with p1, p2:
            await handle_end_session(ix)
        assert any("Session ended" in m and "Workday" in m for m in ix._sent)
        assert any("Shipped X; follow up on Y." in m for m in ix._sent)
        ended = S.get_last_session()
        assert ended["overview"] == "Shipped X; follow up on Y."
        import pathlib
        content = pathlib.Path(ended["file"]).read_text(encoding="utf-8")
        assert "## Overview" in content and "Shipped X" in content

    @pytest.mark.asyncio
    async def test_end_ai_failure_falls_back(self, ix):
        from commands.session_commands import handle_end_session
        S.start_session(name="BrokenAI")
        S.add_note("note that survives", author="Bob")
        p1, p2 = _mock_ai(fail=True)
        with p1, p2:
            await handle_end_session(ix)
        ended = S.get_last_session()
        assert "AI overview unavailable" in ended["overview"]
        assert "note that survives" in ended["overview"]

    @pytest.mark.asyncio
    async def test_end_uses_custom_summary_prompt(self, ix, monkeypatch):
        """SESSION_SUMMARY_PROMPT from settings is used as the system prompt."""
        import commands.session_commands as sc
        from commands.session_commands import handle_end_session
        S.start_session(name="Custom")
        inst = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        inst.chat.completions.create = AsyncMock(return_value=resp)
        monkeypatch.setattr(sc, "_make_client", lambda: inst)
        monkeypatch.setattr(sc, "_validate_model", AsyncMock(return_value="test-model"))
        monkeypatch.setattr(sc, "SESSION_SUMMARY_PROMPT", "MY CUSTOM PROMPT")

        await handle_end_session(ix)

        sys_msg = inst.chat.completions.create.await_args.kwargs["messages"][0]
        assert sys_msg["role"] == "system"
        assert sys_msg["content"] == "MY CUSTOM PROMPT"

    @pytest.mark.asyncio
    async def test_end_falls_back_to_default_prompt(self, ix, monkeypatch):
        """Empty SESSION_SUMMARY_PROMPT → built-in default prompt is used."""
        import commands.session_commands as sc
        from commands.session_commands import handle_end_session
        S.start_session(name="Default")
        inst = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        inst.chat.completions.create = AsyncMock(return_value=resp)
        monkeypatch.setattr(sc, "_make_client", lambda: inst)
        monkeypatch.setattr(sc, "_validate_model", AsyncMock(return_value="test-model"))
        monkeypatch.setattr(sc, "SESSION_SUMMARY_PROMPT", "")

        await handle_end_session(ix)

        sys_msg = inst.chat.completions.create.await_args.kwargs["messages"][0]
        assert "session overviews" in sys_msg["content"]

    @pytest.mark.asyncio
    async def test_end_renames(self, ix):
        from commands.session_commands import handle_end_session
        S.start_session(name="Old")
        p1, p2 = _mock_ai(content="ok")
        with p1, p2:
            await handle_end_session(ix, name="Renamed")
        assert S.get_last_session()["name"] == "Renamed"


class TestRemindNextSessionCommand:

    @pytest.mark.asyncio
    async def test_reminder_with_no_active_session_starts_one(self, ix):
        from commands.session_commands import handle_remind_next_session
        await handle_remind_next_session(ix, "check the build")
        assert len(S.list_queued_reminders()) == 1
        assert S.get_current_session() is not None
        assert any("a new one has been started" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_reminder_with_active_session_waits(self, ix):
        from commands.session_commands import handle_remind_next_session
        S.start_session(name="Active")
        await handle_remind_next_session(ix, "later thing")
        assert len(S.list_queued_reminders()) == 1
        assert any("waits for the NEXT session" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_reminder_after_ended_session_starts_new_one(self, ix):
        """No cooldown: after a session ends, /remind_next_session starts a fresh one."""
        from commands.session_commands import handle_remind_next_session
        S.start_session(name="A")
        S.end_session(overview=None)
        await handle_remind_next_session(ix, "queued anyway")
        assert len(S.list_queued_reminders()) == 1
        assert S.get_current_session() is not None
        assert any("a new one has been started" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_refused_when_bot_cannot_post_here(self, ix):
        """Regression: a next-session reminder destined for a channel the bot
        cannot post to must be refused up front (it could never be delivered).
        """
        from commands.session_commands import handle_remind_next_session

        channel = MagicMock()
        perm = MagicMock()
        perm.view_channel = False
        perm.send_messages = True
        channel.permissions_for.return_value = perm
        ix.channel = channel
        ix.guild_id = 1
        ix.guild.me = MagicMock()

        await handle_remind_next_session(ix, "never delivered")
        assert any("can't send messages" in m for m in ix._sent)
        assert S.list_queued_reminders() == []


class TestSessionNotesCommand:

    @pytest.mark.asyncio
    async def test_add_without_active_refused(self, ix):
        from commands.session_commands import handle_session_notes
        await handle_session_notes(ix, action="add", note="hello")
        assert any("no active session" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_add_requires_note_text(self, ix):
        from commands.session_commands import handle_session_notes
        S.start_session(name="N")
        await handle_session_notes(ix, action="add", note=None)
        assert any("Please provide a note" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_add_and_view(self, ix):
        from commands.session_commands import handle_session_notes
        ix.user = MagicMock()
        ix.user.display_name = "Alice"
        S.start_session(name="Notes")
        await handle_session_notes(ix, action="add", note="remember the deploy")
        assert any("Note added" in m for m in ix._sent)

        await handle_session_notes(ix, action="view")
        view_msgs = [m for m in ix._sent if "Session notes" in m]
        assert view_msgs and "remember the deploy" in view_msgs[-1]
        assert "Alice" in view_msgs[-1]  # author attribution

    @pytest.mark.asyncio
    async def test_view_after_end_shows_last(self, ix):
        from commands.session_commands import handle_session_notes
        S.start_session(name="Done")
        S.add_note("final note")
        S.end_session(overview="wrapped up")
        await handle_session_notes(ix, action="view")
        view_msgs = [m for m in ix._sent if "Session notes" in m]
        assert any("last (ended)" in m and "final note" in m for m in view_msgs)

    @pytest.mark.asyncio
    async def test_view_no_sessions(self, ix):
        from commands.session_commands import handle_session_notes
        await handle_session_notes(ix, action="view")
        assert any("No sessions yet" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_unknown_action(self, ix):
        from commands.session_commands import handle_session_notes
        await handle_session_notes(ix, action="destroy", note=None)
        assert any("Unknown action" in m for m in ix._sent)


class TestAddTranscript:
    """sessions.add_transcript() — automatic transcript -> session notes."""

    def test_appends_single_bullet_with_header(self):
        S.start_session(name="T")
        session, n = S.add_transcript("hello there", title="Voice channel transcript — #vc (2026-08-31 22:00, 10s)")
        assert n == 1
        notes = S.get_notes(session)
        assert len(notes) == 1
        ts, text = notes[0]
        assert isinstance(ts, float)
        assert "Voice channel transcript — #vc" in text
        assert "part 1/1" in text
        assert "hello there" in text

    def test_long_transcript_is_chunked(self):
        S.start_session(name="T")
        long_text = "word " * 2000  # ~10k chars -> several bullets
        session, n = S.add_transcript(long_text.strip(), title="Voice channel transcript")
        assert n > 1
        notes = S.get_notes(session)
        texts = [t for _ts, t in notes]
        assert "part 1/" + str(n) in texts[0]
        assert f"continued, part {n}/{n}" in texts[-1]
        # no bullet exceeds the display-safe limit (header overhead included)
        assert all(len(t) <= 2000 for t in texts)
        # full text is preserved across chunks (whitespace-normalized)
        joined = " ".join("".join(t.split()) for t in texts)
        assert long_text.strip().split()[0] in joined

    def test_no_active_session_returns_none(self):
        session, n = S.add_transcript("hello")
        assert session is None and n == 0

    def test_empty_text_is_noop(self):
        S.start_session(name="T")
        session, n = S.add_transcript("   ")
        assert n == 0
        assert S.get_notes(session) == []

    def test_pins_to_ended_session_when_pinned(self):
        # STT completes after the recording's session ended — the transcript
        # must still land in THAT session's file, not the new active one.
        S.start_session(name="Old")
        old = S.get_current_session()
        S.end_session(overview="done")
        S._state["last_start_at"] -= 2 * 3600
        S.start_session(name="New")

        session, n = S.add_transcript("pinned words", title="Voice channel transcript",
                                      session=old)
        assert n == 1 and session is old
        assert "pinned words" in pathlib.Path(old["file"]).read_text(encoding="utf-8")
        assert all("pinned words" not in t for _ts, t in S.get_notes())

    @pytest.mark.asyncio
    async def test_bullet_lands_in_notes_file_and_view(self, ix):
        from commands.session_commands import handle_session_notes
        S.start_session(name="T")
        S.add_transcript("the key is under the bridge", title="Voice channel transcript — #vc")
        path = pathlib.Path(S.get_current_session()["file"])
        on_disk = path.read_text(encoding="utf-8")
        assert "under the bridge" in on_disk
        # and it shows up in /session_notes view
        await handle_session_notes(ix, action="view")
        view_msgs = [m for m in ix._sent if "Session notes" in m]
        assert any("under the bridge" in m for m in view_msgs)
