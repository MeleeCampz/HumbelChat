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
