"""Tests for the global session store (bot_core.sessions)."""
from __future__ import annotations

import pathlib
import re
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

    def test_end_stores_merged_log(self):
        session, _ = S.start_session(name="M1")
        S.add_document("Player A log: we looted the cave.", title="a.md")
        ended = S.end_session(overview="Short overview.",
                              merged_log="## The Cave\nWe looted the cave together.")
        assert ended["merged_log"].startswith("## The Cave")

    def test_merged_log_replaces_mechanical_documents_section(self):
        """#11 follow-up: the AI-merged canonical log is the session's RAG
        content — the mechanical full-text copy must NOT also be rendered."""
        session, _ = S.start_session(name="M2")
        S.add_document("Player A log: we looted the cave.", title="a.md")
        S.end_session(overview=None,
                      merged_log="## The Cave\nCombined: we looted the cave.")
        content = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        assert "## Session Log (combined from all uploads — AI-merged)" in content
        assert "Combined: we looted the cave." in content
        # no mechanical copy of the raw upload next to the merged log
        assert "## Documents" not in content

    def test_without_merged_log_mechanical_section_stays(self):
        """AI unavailable / stale end / legacy session → the mechanical
        full-text copy keeps document content reachable via RAG."""
        session, _ = S.start_session(name="M3")
        S.add_document("Player A log: we looted the cave.", title="a.md")
        S.end_session(overview=None)  # no merged log
        content = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        assert "## Documents" in content
        assert "Player A log: we looted the cave." in content
        assert "## Session Log" not in content

    def test_merged_log_survives_persistence_reload(self):
        session, _ = S.start_session(name="M4")
        S.add_document("raw upload body", title="a.md")
        S.end_session(overview=None, merged_log="merged canonical log text")
        assert S.get_last_session()["merged_log"] == "merged canonical log text"
        S.load_persisted()
        reloaded = S.get_last_session()
        assert reloaded["merged_log"] == "merged canonical log text"
        # and a fresh render from the reloaded state still uses the merged log
        S._write_session_file(reloaded)
        content = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        assert "## Session Log" in content and "## Documents" not in content


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
    """sessions.add_document() — an uploaded .txt/.md file → raw file in the
    session's hidden .attachments/ folder (out of RAG) + full text combined
    into notes.md, the session's single RAG document."""

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
        # A raw .md file exists under the session's HIDDEN .attachments/ folder
        att_dir = pathlib.Path(session["dir"]) / ".attachments"
        files = [f for f in att_dir.iterdir() if f.is_file()]
        assert len(files) == 1
        body = files[0].read_text(encoding="utf-8")
        assert "the answer is 42" in body
        # The notes state is unchanged (no pre-chunking, no pointer stored).
        assert S.get_notes(session) == []
        # notes.md combines the FULL text under a Documents section
        content = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        assert "## Documents" in content
        assert files[0].name in content
        assert "the answer is 42" in content
        # Only notes.md is indexable for the session
        assert S._session_index_paths(session) == [pathlib.Path(session["file"])]

    def test_default_title_used(self):
        S.start_session(name="D")
        session, n = S.add_document("just words")
        assert n == 1
        att_dir = pathlib.Path(session["dir"]) / ".attachments"
        assert len(list(att_dir.iterdir())) == 1
        assert "just words" in att_dir.iterdir().__next__().read_text(encoding="utf-8")

    def test_long_document_not_chunked(self):
        S.start_session(name="D")
        long_text = ("word " * 2000).strip()  # ~10k chars — stays in ONE file
        session, n = S.add_document(long_text, title="big.md")
        assert n == 1  # one file, not several bullets
        att_dir = pathlib.Path(session["dir"]) / ".attachments"
        files = [f for f in att_dir.iterdir() if f.is_file()]
        assert len(files) == 1
        body = files[0].read_text(encoding="utf-8")
        # The whole document is preserved verbatim (KB will chunk it).
        assert long_text in body
        # ...and combined verbatim into notes.md as well.
        assert long_text in pathlib.Path(session["file"]).read_text(encoding="utf-8")

    def test_pins_to_given_session(self):
        """*session* pins the target — the document lands in THAT session's folder."""
        S.start_session(name="Old")
        old = S.get_current_session()
        S.end_session(overview="done")
        S._state["last_start_at"] -= 2 * 3600
        S.start_session(name="New")

        session, n = S.add_document("pinned doc", title="doc.md", session=old)
        assert n == 1 and session is old
        att_dir = pathlib.Path(old["dir"]) / ".attachments"
        assert any("pinned doc" in f.read_text(encoding="utf-8") for f in att_dir.iterdir())
        # The pinned doc is combined into the OLD session's notes.md only
        assert "pinned doc" in pathlib.Path(old["file"]).read_text(encoding="utf-8")
        new_file = pathlib.Path(S.get_current_session()["file"]).read_text(encoding="utf-8")
        assert "pinned doc" not in new_file
        # The new active session's folder has no attachment
        new_dir = pathlib.Path(S.get_current_session()["dir"]) / ".attachments"
        assert not new_dir.exists() or len(list(new_dir.iterdir())) == 0

    def test_repeated_uploads_get_distinct_files(self):
        S.start_session(name="D")
        S.add_document("one", title="same.txt")
        S.add_document("two", title="same.txt")
        att_dir = pathlib.Path(S.get_current_session()["dir"]) / ".attachments"
        names = sorted(f.name for f in att_dir.iterdir())
        assert len(names) == 2  # unique_path disambiguates duplicates

    def test_rerender_is_idempotent(self):
        """Same state + same dot-dir contents ⇒ byte-identical notes.md."""
        S.start_session(name="D")
        S.add_document("stable content", title="doc.md")
        session = S.get_current_session()
        first = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        S._write_session_file(session)
        second = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        assert first == second


class TestAddTranscript:
    """sessions.add_transcript() — a finished transcript → raw file in the
    session's hidden .transcripts/ folder (out of RAG) + full text combined
    into notes.md, the session's single RAG document."""

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
        tdir = pathlib.Path(session["dir"]) / ".transcripts"
        files = [f for f in tdir.iterdir() if f.is_file()]
        assert len(files) == 1
        body = files[0].read_text(encoding="utf-8")
        assert "hello there" in body and "second line" in body
        assert "Voice channel transcript" in body
        # notes state unchanged; notes.md combines the full transcript text
        assert S.get_notes(session) == []
        content = pathlib.Path(session["file"]).read_text(encoding="utf-8")
        assert "## Documents" in content and files[0].name in content
        assert "hello there" in content and "second line" in content

    def test_long_transcript_not_chunked(self):
        S.start_session(name="T")
        long_text = ("word " * 2000).strip()  # ~10k chars — stays in ONE file
        session, n = S.add_transcript(long_text, title="Voice channel transcript")
        assert n == 1
        tdir = pathlib.Path(session["dir"]) / ".transcripts"
        files = [f for f in tdir.iterdir() if f.is_file()]
        assert len(files) == 1
        assert long_text in files[0].read_text(encoding="utf-8")
        assert long_text in pathlib.Path(session["file"]).read_text(encoding="utf-8")

    def test_pins_to_ended_session_when_pinned(self):
        S.start_session(name="Old")
        old = S.get_current_session()
        S.end_session(overview="done")
        S._state["last_start_at"] -= 2 * 3600
        S.start_session(name="New")

        session, n = S.add_transcript("pinned words", title="Voice channel transcript",
                                      session=old)
        assert n == 1 and session is old
        tdir = pathlib.Path(old["dir"]) / ".transcripts"
        assert any("pinned words" in f.read_text(encoding="utf-8") for f in tdir.iterdir())
        assert "pinned words" in pathlib.Path(old["file"]).read_text(encoding="utf-8")


class TestLegacyMigration:
    """migrate_legacy_session_dirs() — visible attachments/ + transcripts/
    (indexed file-by-file) → hidden dot-dirs + combined notes.md."""

    def _legacy_folder(self, name: str = "2026-01-01_01_Legacy") -> pathlib.Path:
        folder = S.notes_dir() / name
        (folder / "attachments").mkdir(parents=True)
        (folder / "transcripts").mkdir()
        (folder / "attachments" / "old_doc.md").write_text(
            "# old doc\n\nlegacy upload text\n", encoding="utf-8")
        (folder / "transcripts" / "transcript_01.md").write_text(
            "# Voice channel transcript\n\nlegacy words\n", encoding="utf-8")
        (folder / "notes.md").write_text(
            "# Session: Legacy\n\n## Notes\n\n(no notes)\n",
            encoding="utf-8")
        return folder

    def test_visible_subdirs_renamed_and_notes_combined(self):
        folder = self._legacy_folder()
        changed = S.migrate_legacy_session_dirs()
        assert changed == 1
        assert not (folder / "attachments").exists()
        assert not (folder / "transcripts").exists()
        assert (folder / ".attachments" / "old_doc.md").is_file()
        assert (folder / ".transcripts" / "transcript_01.md").is_file()
        content = (folder / "notes.md").read_text(encoding="utf-8")
        assert "## Documents" in content
        assert "legacy upload text" in content
        assert "legacy words" in content

    def test_migration_is_idempotent(self):
        folder = self._legacy_folder()
        S.migrate_legacy_session_dirs()
        first = (folder / "notes.md").read_text(encoding="utf-8")
        assert S.migrate_legacy_session_dirs() == 0
        # No double-appended Documents section.
        second = (folder / "notes.md").read_text(encoding="utf-8")
        assert first == second
        assert second.count("## Documents") == 1

    def test_merge_when_dot_dir_already_exists(self):
        folder = self._legacy_folder()
        dot = folder / ".attachments"
        dot.mkdir()
        (dot / "old_doc.md").write_text("# old doc\n\nalready migrated\n", encoding="utf-8")
        (dot / "new_doc.md").write_text("# new doc\n\ndot-dir only\n", encoding="utf-8")
        S.migrate_legacy_session_dirs()
        assert not (folder / "attachments").exists()
        names = {f.name for f in dot.iterdir()}
        assert names == {"old_doc.md", "new_doc.md"}

    def test_noop_without_legacy_folders(self):
        folder = S.notes_dir() / "2026-01-02_01_Modern"
        (folder / ".attachments").mkdir(parents=True)
        assert S.migrate_legacy_session_dirs() == 0


class TestNotesSectionParsing:
    """refresh_notes_from_disk re-parses ONLY the ## Notes section — bullet-
    looking lines inside ## Documents must never become notes."""

    def test_document_bullets_not_parsed_as_notes(self):
        S.start_session(name="P")
        session = S.get_current_session()
        path = pathlib.Path(session["file"])
        path.write_text(
            "# Session: P\n\n## Notes\n"
            "- (2026-01-01 12:00) real note\n"
            "\n## Documents (combined uploads + transcripts — full text)\n\n"
            "### doc.md\n"
            "- (2026-01-01 13:00) looks like a note but is document text\n",
            encoding="utf-8")
        notes = S.refresh_notes_from_disk(session)
        assert len(notes) == 1
        assert notes[0][1] == "real note"

    def test_user_edited_notes_still_reparsed(self):
        S.start_session(name="P")
        session = S.get_current_session()
        S.add_note("first")
        path = pathlib.Path(session["file"])
        text = path.read_text(encoding="utf-8")
        # Insert a hand-added bullet right after the existing one (still inside ## Notes)
        m = re.search(r"^(- \([^)]+\) first)$", text, flags=re.MULTILINE)
        assert m, "expected the 'first' note bullet in notes.md"
        text = text[:m.end()] + "\n- (2026-01-01 09:00) hand-added" + text[m.end():]
        path.write_text(text, encoding="utf-8")
        notes = S.refresh_notes_from_disk(session)
        assert [t for _ts, t in notes] == ["first", "hand-added"]


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
    async def test_unexpected_error_counts_as_an_attempt(self, monkeypatch):
        S.queue_next_session_reminder(111, "boom")

        async def explode(*args, **kwargs):
            raise RuntimeError("discord down")

        monkeypatch.setattr("bot_core.channel_delivery.send_to_channel", explode)
        assert await S.deliver_queued_reminders(MagicMock()) == 0
        remaining = S.list_queued_reminders()
        assert len(remaining) == 1
        assert remaining[0].get("attempts") == 1

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
