"""Tests for the /stop_recording -> STT wiring in commands/recording_commands.

Verifies that after a recording stops, transcription is spawned in the
background (when enabled), skipped when disabled or transcribe=false, and
that the user-facing message reflects what will happen. No network: the
transcriber module and task spawner are faked.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import commands.recording_commands as rc


class _FakeRecorder:
    def __init__(self, manifest: dict) -> None:
        self._manifest = manifest
        self.is_recording = True

    def stop(self):
        self.is_recording = False
        return self._manifest


def _make_env(tmp_path: Path):
    """Build (ix, bot) with a stopped-ready recorder + on-disk recording dir."""
    d = tmp_path / "rec"
    d.mkdir()
    (d / "Alice_1.wav").write_bytes(b"RIFF....")
    manifest = {
        "duration_s": 5.0,
        "manifest_path": str(d / "manifest.json"),
        "speakers": [
            {"user_id": 1, "display_name": "Alice", "wav_file": "Alice_1.wav",
             "frames_captured": 10, "spoken_duration_s": 2.0},
        ],
    }
    (d / "manifest.json").write_text(json.dumps(manifest))

    bot = MagicMock()
    bot._voice_recorder = _FakeRecorder(manifest)
    bot.get_guild.return_value = None  # no voice client to disconnect

    ix = _ix()
    return ix, bot, manifest


def _ix():
    sent: list[str] = []

    async def on_send(content="", ephemeral=False, file=None, files=None):
        sent.append(str(content))

    ix = MagicMock()
    ix.followup.send.side_effect = on_send
    ix.response.defer = AsyncMock()
    ix.guild_id = 42
    ix._sent = sent
    return ix


@pytest.fixture
def env(tmp_path, monkeypatch):
    ix, bot, manifest = _make_env(tmp_path)
    monkeypatch.setattr(rc, "get_bot", lambda: bot)
    return tmp_path, ix, bot, manifest


class TestStopRecordingSttWiring:
    @pytest.mark.asyncio
    async def test_spawns_transcription_when_enabled(self, env, monkeypatch):
        _, ix, _, _ = env
        spawned: list = []

        def fake_spawn(coro, *, name=None):
            spawned.append(name)
            coro.close()  # don't run it; wiring under test is the spawn call
            return MagicMock()

        monkeypatch.setattr(rc, "spawn_tracked_task", fake_spawn)
        monkeypatch.setattr(rc, "STT_ENABLED", True)

        await rc.handle_stop_recording(ix, leave_channel=False)
        assert len(spawned) == 1 and spawned[0] == "stt-transcription"
        # user is told transcription is running
        assert any("Transcribing" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_skips_when_stt_disabled(self, env, monkeypatch):
        _, ix, _, _ = env
        spawned: list = []
        monkeypatch.setattr(rc, "spawn_tracked_task", lambda c, **k: (c.close(), MagicMock())[1])
        monkeypatch.setattr(rc, "STT_ENABLED", False)

        await rc.handle_stop_recording(ix, leave_channel=False)
        assert spawned == []
        assert not any("Transcribing" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_skips_when_transcribe_flag_false(self, env, monkeypatch):
        _, ix, _, _ = env
        spawned: list = []
        monkeypatch.setattr(rc, "spawn_tracked_task", lambda c, **k: (c.close(), MagicMock())[1])
        monkeypatch.setattr(rc, "STT_ENABLED", True)

        await rc.handle_stop_recording(ix, leave_channel=False, transcribe=False)
        assert spawned == []
        assert not any("Transcribing" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_skips_when_no_speech_captured(self, env, monkeypatch):
        _, ix, bot, _ = env
        # recorder with a speaker who captured zero frames
        manifest = dict(bot._voice_recorder._manifest)
        manifest["speakers"] = [dict(manifest["speakers"][0], frames_captured=0)]
        bot._voice_recorder = _FakeRecorder(manifest)

        spawned: list = []
        monkeypatch.setattr(rc, "spawn_tracked_task", lambda c, **k: (c.close(), MagicMock())[1])
        monkeypatch.setattr(rc, "STT_ENABLED", True)

        await rc.handle_stop_recording(ix, leave_channel=False)
        assert spawned == []


class TestStartRecordingOrdering:
    """Regression: /start_recording must arm the recorder BEFORE joining voice.

    Discord announces SSRC -> user (op-11/op-5) during channel.connect();
    calling start() afterwards wipes those mappings and the whole recording
    comes out empty (silent failure, no WAVs).
    """

    @pytest.mark.asyncio
    async def test_start_called_before_channel_connect(self, env, monkeypatch):
        _, ix, bot, _ = env
        order: list[str] = []

        class _FakeRec:
            is_recording = False

            def start(self, **kw):
                order.append("start")

            def snapshot(self):
                return {}

        monkeypatch.setattr(rc, "_get_recorder", lambda b: _FakeRec())

        async def fake_ensure(*a, **k):
            order.append("connect")

        monkeypatch.setattr(rc, "_ensure_bot_in_channel", fake_ensure)

        # user is in a voice channel
        from types import SimpleNamespace
        ix.user = SimpleNamespace(voice=SimpleNamespace(channel=SimpleNamespace(id=7, name="vc")))

        await rc.handle_start_recording(ix)
        assert order == ["start", "connect"], f"recorder must arm before the voice join: {order}"


class TestRunTranscriptionDelivery:
    @pytest.mark.asyncio
    async def test_posts_summary_and_attaches_transcript(self, env, tmp_path, monkeypatch):
        _, ix, _, manifest = env

        import bot_core.transcriber as T
        report = T.TranscriptionReport(model="m", language_requested="", started_at=1.0)
        report.speakers.append(T.SpeakerResult(user_id=1, display_name="Alice",
                                               wav_file="Alice_1.wav", text="hello there"))
        report.finished_at = 2.0

        monkeypatch.setattr(T, "transcribe_recording", AsyncMock(return_value=report))
        monkeypatch.setattr(T, "write_transcript", lambda out_dir, m, r: (out_dir / "transcript.json"))
        (Path(manifest["manifest_path"]).parent / "transcript.json").write_text("{}")

        await rc._run_transcription(ix, manifest)
        assert any("Transcription done" in m for m in ix._sent)
        assert any("hello there" in m for m in ix._sent)

    @pytest.mark.asyncio
    async def test_reports_failure_gracefully(self, env, monkeypatch):
        _, ix, _, manifest = env
        import bot_core.transcriber as T
        monkeypatch.setattr(T, "transcribe_recording", AsyncMock(side_effect=RuntimeError("boom")))

        await rc._run_transcription(ix, manifest)
        assert any("Transcription failed" in m for m in ix._sent)


class TestTranscriptSessionWiring:
    """_run_transcription must fold the finished transcript into the active
    session's notes (STT_ADD_TO_SESSION) and say so in the Discord message."""

    @pytest.fixture(autouse=True)
    def _clean_sessions(self):
        from bot_core import sessions as S
        S._state["session"] = None
        S._state["last_ended"] = None
        yield
        S._state["session"] = None
        S._state["last_ended"] = None

    def _report(self):
        import bot_core.transcriber as T
        report = T.TranscriptionReport(model="m", language_requested="", started_at=1.0)
        report.speakers.append(T.SpeakerResult(user_id=1, display_name="Alice",
                                               wav_file="Alice_1.wav", text="hello there"))
        report.finished_at = 2.0
        return report

    @pytest.mark.asyncio
    async def test_adds_transcript_to_active_session(self, env, tmp_path, monkeypatch):
        from bot_core import sessions as S
        _, ix, _, manifest = env
        S.start_session(name="T")

        import bot_core.transcriber as T
        monkeypatch.setattr(T, "transcribe_recording", AsyncMock(return_value=self._report()))
        monkeypatch.setattr(T, "write_transcript", lambda out_dir, m, r: (out_dir / "transcript.json"))
        (Path(manifest["manifest_path"]).parent / "transcript.json").write_text("{}")

        await rc._run_transcription(ix, manifest)

        notes = S.get_notes()
        assert any("hello there" in t for _ts, t in notes)
        done_msgs = [m for m in ix._sent if "Transcription done" in m]
        assert done_msgs and any("Added to the active session" in m for m in done_msgs)

    @pytest.mark.asyncio
    async def test_no_active_session_still_delivers(self, env, tmp_path, monkeypatch):
        from bot_core import sessions as S
        _, ix, _, manifest = env
        assert S.get_current_session() is None

        import bot_core.transcriber as T
        monkeypatch.setattr(T, "transcribe_recording", AsyncMock(return_value=self._report()))
        monkeypatch.setattr(T, "write_transcript", lambda out_dir, m, r: (out_dir / "transcript.json"))
        (Path(manifest["manifest_path"]).parent / "transcript.json").write_text("{}")

        await rc._run_transcription(ix, manifest)

        done_msgs = [m for m in ix._sent if "Transcription done" in m]
        assert done_msgs  # transcript still delivered to the channel...
        assert not any("Added to the active session" in m for m in done_msgs)  # ...but no note added
        assert S.get_notes() == []

    @pytest.mark.asyncio
    async def test_skipped_when_flag_disabled(self, env, tmp_path, monkeypatch):
        from bot_core import sessions as S
        _, ix, _, manifest = env
        S.start_session(name="T")
        monkeypatch.setattr(rc, "STT_ADD_TO_SESSION", False)

        import bot_core.transcriber as T
        monkeypatch.setattr(T, "transcribe_recording", AsyncMock(return_value=self._report()))
        monkeypatch.setattr(T, "write_transcript", lambda out_dir, m, r: (out_dir / "transcript.json"))
        (Path(manifest["manifest_path"]).parent / "transcript.json").write_text("{}")

        await rc._run_transcription(ix, manifest)

        assert S.get_notes() == []
        done_msgs = [m for m in ix._sent if "Transcription done" in m]
        assert not any("Added to the active session" in m for m in done_msgs)

    @pytest.mark.asyncio
    async def test_pins_to_session_active_at_stop_even_if_it_ends(self, env, tmp_path, monkeypatch):
        """The transcript must land in the session that was active when the
        recording stopped — even if it ends before STT finishes and a new
        session is started meanwhile."""
        from bot_core import sessions as S
        _, ix, _, manifest = env
        S.start_session(name="Old")
        old = S.get_current_session()

        # User ends the session and starts a new one while STT is running.
        S.end_session(overview="done")
        S._state["last_start_at"] -= 2 * 3600  # bypass start cooldown in start_session
        S.start_session(name="New")
        assert S.get_current_session() is not old

        import bot_core.transcriber as T
        monkeypatch.setattr(T, "transcribe_recording", AsyncMock(return_value=self._report()))
        monkeypatch.setattr(T, "write_transcript", lambda out_dir, m, r: (out_dir / "transcript.json"))
        (Path(manifest["manifest_path"]).parent / "transcript.json").write_text("{}")

        await rc._run_transcription(ix, manifest, session_at_stop=old)

        # transcript went to the OLD (ended) session's file...
        old_file = Path(old["file"]).read_text(encoding="utf-8")
        assert "hello there" in old_file
        # ...and NOT into the new active session's notes
        assert all("hello there" not in t for _ts, t in S.get_notes())
        done_msgs = [m for m in ix._sent if "Transcription done" in m]
        assert any("ended session" in m and "Old" in m for m in done_msgs)

    @pytest.mark.asyncio
    async def test_stop_recording_passes_session_at_stop(self, env, monkeypatch):
        """handle_stop_recording must capture the active session at stop time
        and hand it to the transcription task."""
        from bot_core import sessions as S
        _, ix, _, _ = env
        S.start_session(name="AtStop")
        captured: list = []

        def fake_run(ix_arg, manifest_arg, session_at_stop=None):
            # records the bound args at call time; returns an unawaited dummy
            captured.append(session_at_stop)

            async def _noop():
                pass
            return _noop()

        monkeypatch.setattr(rc, "_run_transcription", fake_run)
        monkeypatch.setattr(rc, "spawn_tracked_task",
                            lambda coro, **k: (coro.close(), MagicMock())[1])
        monkeypatch.setattr(rc, "STT_ENABLED", True)

        await rc.handle_stop_recording(ix, leave_channel=False)

        # the third arg must be the session that was active at stop time
        assert captured == [S.get_current_session()]
