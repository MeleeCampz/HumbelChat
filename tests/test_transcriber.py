"""Tests for bot_core.transcriber (STT via /v1/audio/transcriptions).

Covers:
  * 48 kHz -> 16 kHz mono WAV conversion (downmix + resample),
  * transcribe_wav() success / missing-file / backend-error paths,
  * transcribe_recording() per-speaker aggregation, and
  * write_transcript() output + manifest pointer.

No network access: the AsyncOpenAI client is replaced with a fake.
"""
from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import bot_core.transcriber as T


# ── helpers ──────────────────────────────────────────────────────────────────
def _make_wav(path: Path, *, rate: int = 48_000, channels: int = 1, ms: int = 200) -> None:
    """Write a small real WAV (constant-amplitude 'tone')."""
    n_frames = rate * ms // 1000
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        if channels == 1:
            wf.writeframes(struct.pack(f"<{n_frames}h", *([1000] * n_frames)))
        else:
            # interleaved L,R with different levels to verify downmix
            samples = []
            for _ in range(n_frames):
                samples.extend([1000, 3000])
            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


class _FakeTranscriptions:
    def __init__(self, resp=None, exc: Exception | None = None) -> None:
        self._resp = resp
        self._exc = exc
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._resp


class _FakeAudio:
    def __init__(self, resp=None, exc: Exception | None = None) -> None:
        self.transcriptions = _FakeTranscriptions(resp=resp, exc=exc)


class _FakeClient:
    def __init__(self, resp=None, exc: Exception | None = None) -> None:
        self.audio = _FakeAudio(resp=resp, exc=exc)


# ── 16 kHz conversion ────────────────────────────────────────────────────────
class TestTo16kMonoWav:
    def test_48k_mono_resampled_to_16k(self, tmp_path):
        p = tmp_path / "a.wav"
        _make_wav(p, rate=48_000, ms=200)
        out = T._to_16k_mono_wav(p)
        with wave.open(io.BytesIO(out), "rb") as wf:
            assert wf.getframerate() == 16_000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            # 200 ms at 16 kHz == 3200 frames
            assert abs(wf.getnframes() - 3200) <= 2

    def test_stereo_downmixed_to_mono(self, tmp_path):
        p = tmp_path / "s.wav"
        _make_wav(p, rate=48_000, channels=2, ms=100)
        out = T._to_16k_mono_wav(p)
        with wave.open(io.BytesIO(out), "rb") as wf:
            assert wf.getnchannels() == 1
            raw = wf.readframes(wf.getnframes())
        samples = struct.unpack(f"<{len(raw) // 2}h", raw)
        # average of 1000 and 3000 is 2000
        assert all(abs(s - 2000) <= 1 for s in samples[:50])

    def test_already_16k_passes_through(self, tmp_path):
        p = tmp_path / "b.wav"
        _make_wav(p, rate=16_000, ms=50)
        out = T._to_16k_mono_wav(p)
        with wave.open(io.BytesIO(out), "rb") as wf:
            assert wf.getframerate() == 16_000
            assert abs(wf.getnframes() - 800) <= 2

    def test_shrinks_file_size(self, tmp_path):
        p = tmp_path / "c.wav"
        _make_wav(p, rate=48_000, ms=1000)
        out = T._to_16k_mono_wav(p)
        # 48k mono -> 16k mono is a ~3x shrink (minus header noise)
        assert len(out) < p.stat().st_size / 2.5

    def test_bad_bit_depth_raises(self, tmp_path):
        p = tmp_path / "d.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(3)  # 24-bit — unsupported
            wf.setframerate(48_000)
            wf.writeframes(b"\x00" * 960)
        with pytest.raises(T.TranscriptionError):
            T._to_16k_mono_wav(p)


# ── transcribe_wav ───────────────────────────────────────────────────────────
class TestTranscribeWav:
    @pytest.mark.asyncio
    async def test_success_returns_text(self, tmp_path, monkeypatch):
        p = tmp_path / "spk.wav"
        _make_wav(p)
        fake = _FakeClient(resp={"text": "  hello world ", "language": "en"})
        monkeypatch.setattr(T, "_stt_client", lambda: fake)

        r = await T.transcribe_wav(p, model="test-model")
        assert r.ok
        assert r.text == "hello world"
        assert r.language == "en"
        assert r.wav_file == "spk.wav"
        # uploaded as a (name, bytes, content-type) tuple with the 16 kHz WAV
        call = fake.audio.transcriptions.calls[0]
        name, data, ctype = call["file"]
        assert name == "spk.wav" and ctype == "audio/wav"
        with wave.open(io.BytesIO(data), "rb") as wf:
            assert wf.getframerate() == 16_000
        assert call["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_language_forwarded_when_set(self, tmp_path, monkeypatch):
        p = tmp_path / "spk.wav"
        _make_wav(p)
        fake = _FakeClient(resp={"text": "hallo"})
        monkeypatch.setattr(T, "_stt_client", lambda: fake)

        r = await T.transcribe_wav(p, model="m", language="de")
        assert r.ok
        assert fake.audio.transcriptions.calls[0]["language"] == "de"

    @pytest.mark.asyncio
    async def test_missing_file_reports_error(self, tmp_path):
        r = await T.transcribe_wav(tmp_path / "nope.wav", model="m")
        assert not r.ok
        assert "not found" in r.error

    @pytest.mark.asyncio
    async def test_backend_error_captured_not_raised(self, tmp_path, monkeypatch):
        p = tmp_path / "spk.wav"
        _make_wav(p)
        err = SimpleNamespace(message="STT model 'small' is not downloaded.")
        fake = _FakeClient(exc=Exception(err))
        monkeypatch.setattr(T, "_stt_client", lambda: fake)

        r = await T.transcribe_wav(p, model="small")
        assert not r.ok
        assert "not downloaded" in r.error

    @pytest.mark.asyncio
    async def test_corrupt_wav_reports_error(self, tmp_path, monkeypatch):
        p = tmp_path / "bad.wav"
        p.write_bytes(b"RIFF....not a real wav")
        fake = _FakeClient(resp={"text": "x"})
        monkeypatch.setattr(T, "_stt_client", lambda: fake)

        r = await T.transcribe_wav(p, model="m")
        assert not r.ok
        assert "could not read WAV" in r.error


# ── transcribe_recording ─────────────────────────────────────────────────────
class TestTranscribeRecording:
    def _manifest(self, tmp_path: Path) -> dict:
        d = tmp_path / "rec"
        d.mkdir()
        wav = d / "Alice_1.wav"
        _make_wav(wav)
        manifest = {
            "duration_s": 2.0,
            "manifest_path": str(d / "manifest.json"),
            "speakers": [
                {"user_id": 1, "display_name": "Alice", "wav_file": "Alice_1.wav",
                 "frames_captured": 5},
                {"user_id": 2, "display_name": "Bob", "wav_file": "Bob_2.wav",
                 "frames_captured": 3},  # file intentionally missing
            ],
        }
        (d / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    @pytest.mark.asyncio
    async def test_per_speaker_results_and_settings(self, tmp_path, monkeypatch):
        manifest = self._manifest(tmp_path)
        fake = _FakeClient(resp={"text": "hi there"})
        monkeypatch.setattr(T, "_stt_client", lambda: fake)
        from config import settings as S
        monkeypatch.setattr(S, "STT_MODEL", "fake-model", raising=False)
        monkeypatch.setattr(S, "STT_LANGUAGE", "", raising=False)

        report = await T.transcribe_recording(manifest)
        assert report.model == "fake-model"
        assert len(report.speakers) == 2
        a, b = report.speakers
        assert a.ok and a.user_id == 1 and a.display_name == "Alice"
        assert a.text == "hi there"
        assert not b.ok  # missing file
        assert "not found" in b.error

    @pytest.mark.asyncio
    async def test_all_failures_still_produce_report(self, tmp_path, monkeypatch):
        manifest = self._manifest(tmp_path)
        (tmp_path / "rec" / "Alice_1.wav").unlink()  # remove both files' chance
        fake = _FakeClient(exc=SimpleNamespace(message="boom"))
        monkeypatch.setattr(T, "_stt_client", lambda: fake)

        report = await T.transcribe_recording(manifest)
        assert report.ok_count == 0
        assert len(report.failed) == 2


# ── write_transcript ─────────────────────────────────────────────────────────
class TestResolveRecordingDir:
    def test_uses_manifest_path_parent(self, tmp_path):
        d = tmp_path / "rec"
        d.mkdir()
        (d / "manifest.json").write_text("{}")
        m = {"manifest_path": str(d / "manifest.json"), "speakers": []}
        assert T.resolve_recording_dir(m) == d

    def test_fallback_scans_recordings_dir(self, tmp_path, monkeypatch):
        from config import settings as S
        recs = tmp_path / "recordings"
        (recs / "recording_1").mkdir(parents=True)
        (recs / "recording_1" / "Alice_1.wav").write_bytes(b"x")
        monkeypatch.setattr(S, "RECORDINGS_DIR", recs, raising=False)
        # no manifest_path (as in on-disk manifests)
        m = {"speakers": [{"wav_file": "Alice_1.wav"}]}
        assert T.resolve_recording_dir(m) == recs / "recording_1"

    def test_missing_everything_returns_none(self, tmp_path, monkeypatch):
        from config import settings as S
        monkeypatch.setattr(S, "RECORDINGS_DIR", tmp_path / "nope", raising=False)
        assert T.resolve_recording_dir({"speakers": []}) is None


class TestWriteTranscript:
    def test_writes_json_and_updates_manifest(self, tmp_path):
        d = tmp_path / "rec"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"duration_s": 1.0}))
        manifest = {"manifest_path": str(d / "manifest.json")}

        report = T.TranscriptionReport(model="m", language_requested="", started_at=1.0)
        report.speakers.append(T.SpeakerResult(user_id=1, display_name="Alice",
                                               wav_file="Alice_1.wav", text="hello"))
        report.speakers.append(T.SpeakerResult(user_id=2, display_name="Bob",
                                               wav_file="Bob_2.wav", error="boom"))
        report.finished_at = 3.5

        path = T.write_transcript(d, manifest, report)
        assert path.exists() and path.name == "transcript.json"

        data = json.loads(path.read_text())
        assert data["model"] == "m"
        assert data["elapsed_s"] == 2.5
        assert data["speakers"][0]["text"] == "hello"
        assert data["speakers"][1]["error"] == "boom"
        assert data["speakers"][1]["text"] is None

        m = json.loads((d / "manifest.json").read_text())
        assert m["transcript"] == "transcript.json"
        assert m["stt_model"] == "m"
