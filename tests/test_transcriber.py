"""Tests for bot_core.transcriber (local faster-whisper + HTTP backends).

Covers:
  * 48 kHz -> 16 kHz mono WAV conversion (downmix + resample),
  * transcribe_wav() success / missing-file / backend-error paths on both
    backends,
  * transcribe_recording() per-speaker aggregation + model selection by
    backend,
  * build_interleaved_transcript() timeline merging, and
  * write_transcript() output (json + interleaved txt) + manifest pointer.

No network access: the AsyncOpenAI client and _local_transcribe are faked.
The HTTP-path tests force STT_BACKEND=http; the local-path tests force it to
"local" — each test is explicit about which engine it exercises.
"""
from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bot_core.transcriber as T


@pytest.fixture(autouse=True)
def _http_backend(monkeypatch):
    """Default these tests to the HTTP backend (they fake the OpenAI client).

    Local-backend tests override this explicitly. Also clears any cached
    local models so a real model can never leak into the suite.
    """
    from config import settings as S
    monkeypatch.setattr(S, "STT_BACKEND", "http", raising=False)
    T._local_models.clear()


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


# ── transcribe_wav (local backend) ───────────────────────────────────────────
class TestTranscribeWavLocal:
    @pytest.mark.asyncio
    async def test_success_returns_text_and_segments(self, tmp_path, monkeypatch):
        from config import settings as S
        monkeypatch.setattr(S, "STT_BACKEND", "local", raising=False)
        p = tmp_path / "spk.wav"
        _make_wav(p)

        calls: list[tuple] = []

        def fake_local(wav_path, model_name, language):
            calls.append((wav_path.name, model_name, language))
            return ("hallo welt", "de",
                    [{"start": 0.5, "end": 2.1, "text": "hallo welt"}])

        monkeypatch.setattr(T, "_local_transcribe", fake_local)
        r = await T.transcribe_wav(p, model="large-v3-turbo", language="de")
        assert r.ok
        assert r.text == "hallo welt"
        assert r.language == "de"
        assert r.segments == [{"start": 0.5, "end": 2.1, "text": "hallo welt"}]
        # the model name + language are forwarded to the local engine
        assert calls == [("spk.wav", "large-v3-turbo", "de")]

    @pytest.mark.asyncio
    async def test_local_error_captured_not_raised(self, tmp_path, monkeypatch):
        from config import settings as S
        monkeypatch.setattr(S, "STT_BACKEND", "local", raising=False)
        p = tmp_path / "spk.wav"
        _make_wav(p)

        def boom(wav_path, model_name, language):
            raise RuntimeError("model download failed")

        monkeypatch.setattr(T, "_local_transcribe", boom)
        r = await T.transcribe_wav(p, model="m")
        assert not r.ok
        assert "local whisper" in r.error and "model download failed" in r.error

    @pytest.mark.asyncio
    async def test_missing_file_reports_error(self, tmp_path, monkeypatch):
        from config import settings as S
        monkeypatch.setattr(S, "STT_BACKEND", "local", raising=False)
        r = await T.transcribe_wav(tmp_path / "nope.wav", model="m")
        assert not r.ok
        assert "not found" in r.error


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
        assert report.backend == "http"

    @pytest.mark.asyncio
    async def test_local_backend_selects_local_model(self, tmp_path, monkeypatch):
        from config import settings as S
        manifest = self._manifest(tmp_path)
        monkeypatch.setattr(S, "STT_BACKEND", "local", raising=False)
        monkeypatch.setattr(S, "STT_LOCAL_MODEL", "large-v3-turbo", raising=False)

        def fake_local(wav_path, model_name, language):
            assert model_name == "large-v3-turbo"
            return ("hi", None, [{"start": 0.0, "end": 1.0, "text": "hi"}])

        monkeypatch.setattr(T, "_local_transcribe", fake_local)
        report = await T.transcribe_recording(manifest)
        assert report.backend == "local"
        assert report.model == "large-v3-turbo"
        a = report.speakers[0]
        assert a.ok and a.segments


class TestSttBackendSelection:
    def test_invalid_backend_falls_back_to_local(self, monkeypatch):
        from config import settings as S
        monkeypatch.setattr(S, "STT_BACKEND", "carrier-pigeon", raising=False)
        assert T._stt_backend() == "local"

    def test_explicit_http(self, monkeypatch):
        from config import settings as S
        monkeypatch.setattr(S, "STT_BACKEND", "HTTP", raising=False)
        assert T._stt_backend() == "http"


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


# ── interleaved transcript ───────────────────────────────────────────────────
def _spk(user_id, name, text="", segments=None, error=None):
    return T.SpeakerResult(user_id=user_id, display_name=name, wav_file=f"{name}.wav",
                           text=text, segments=segments or [], error=error)


class TestBuildInterleavedTranscript:
    def test_merges_speakers_chronologically(self):
        report = T.TranscriptionReport(model="m", language_requested="de", started_at=0.0,
                                       backend="local")
        report.speakers.append(_spk(1, "Alice",
                                    segments=[{"start": 2.0, "end": 4.0, "text": "hallo"},
                                              {"start": 9.0, "end": 11.0, "text": "und tschüss"}]))
        report.speakers.append(_spk(2, "Bob",
                                    segments=[{"start": 5.0, "end": 8.0, "text": "servus"}]))

        out = T.build_interleaved_transcript(report)
        lines = [l for l in out.splitlines() if l.strip()]
        assert lines == [
            "[00:02.00] Alice: hallo",
            "[00:05.00] Bob: servus",
            "[00:09.00] Alice: und tschüss",
        ]

    def test_untimed_speakers_appended_at_end(self):
        report = T.TranscriptionReport(model="m", language_requested="", started_at=0.0,
                                       backend="http")
        report.speakers.append(_spk(1, "Alice",
                                    segments=[{"start": 1.0, "end": 2.0, "text": "hi"}]))
        report.speakers.append(_spk(2, "Bob", text="plain text without timestamps"))

        out = T.build_interleaved_transcript(report)
        assert out.index("[00:01.00] Alice: hi") < out.index("[Bob]")
        assert "plain text without timestamps" in out

    def test_all_untimed_still_produces_blocks(self):
        report = T.TranscriptionReport(model="m", language_requested="", started_at=0.0,
                                       backend="http")
        report.speakers.append(_spk(1, "Alice", text="hello"))
        out = T.build_interleaved_transcript(report)
        assert "[Alice]" in out and "hello" in out

    def test_failed_and_empty_speakers_skipped(self):
        report = T.TranscriptionReport(model="m", language_requested="", started_at=0.0,
                                       backend="local")
        report.speakers.append(_spk(1, "Alice", error="boom"))
        report.speakers.append(_spk(2, "Bob"))  # no text
        assert T.build_interleaved_transcript(report) == ""

    def test_timestamp_format_minutes(self):
        report = T.TranscriptionReport(model="m", language_requested="", started_at=0.0,
                                       backend="local")
        report.speakers.append(_spk(1, "Alice",
                                    segments=[{"start": 65.25, "end": 70.0, "text": "x"}]))
        assert "[01:05.25] Alice: x" in T.build_interleaved_transcript(report)


class TestBuildSessionTranscript:
    """build_session_transcript() feeds transcript.txt AND the session notes,
    so it must return something useful for BOTH backends."""

    def test_prefers_chronological_when_timestamps_exist(self):
        report = T.TranscriptionReport(model="m", language_requested="de", started_at=0.0,
                                       backend="local")
        report.speakers.append(_spk(1, "Alice",
                                    segments=[{"start": 2.0, "end": 4.0, "text": "hallo"}]))
        out = T.build_session_transcript(report)
        assert "[00:02.00] Alice: hallo" in out

    def test_falls_back_to_per_speaker_blocks_for_http(self):
        # http backend: no segment timestamps -> interleaved() is empty;
        # the session transcript must still capture who said what.
        report = T.TranscriptionReport(model="m", language_requested="", started_at=0.0,
                                       backend="http")
        report.speakers.append(_spk(1, "Alice", text="hello there"))
        report.speakers.append(_spk(2, "Bob", text="hi back"))

        out = T.build_session_transcript(report)
        assert "[Alice]" in out and "hello there" in out
        assert "[Bob]" in out and "hi back" in out
        # speakers keep their manifest order
        assert out.index("[Alice]") < out.index("[Bob]")

    def test_failed_speakers_skipped_in_fallback(self):
        report = T.TranscriptionReport(model="m", language_requested="", started_at=0.0,
                                       backend="http")
        report.speakers.append(_spk(1, "Alice", error="boom"))
        report.speakers.append(_spk(2, "Bob", text="still here"))

        out = T.build_session_transcript(report)
        assert "[Bob]" in out and "boom" not in out

    def test_empty_when_no_speech(self):
        report = T.TranscriptionReport(model="m", language_requested="", started_at=0.0,
                                       backend="http")
        report.speakers.append(_spk(1, "Alice", error="boom"))
        assert T.build_session_transcript(report) == ""


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

    def test_local_report_writes_interleaved_txt_with_segments(self, tmp_path):
        d = tmp_path / "rec"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"duration_s": 10.0}))
        manifest = {"manifest_path": str(d / "manifest.json")}

        report = T.TranscriptionReport(model="large-v3-turbo", language_requested="de",
                                       started_at=1.0, backend="local")
        report.speakers.append(_spk(1, "Alice",
                                    segments=[{"start": 2.0, "end": 4.0, "text": "hallo"}]))
        report.speakers.append(_spk(2, "Bob",
                                    segments=[{"start": 5.0, "end": 8.0, "text": "servus"}]))
        report.finished_at = 3.5

        path = T.write_transcript(d, manifest, report)
        data = json.loads(path.read_text())
        assert data["backend"] == "local"
        assert data["speakers"][0]["segments"][0]["start"] == 2.0

        txt = (d / "transcript.txt").read_text(encoding="utf-8")
        assert "[00:02.00] Alice: hallo" in txt
        assert "[00:05.00] Bob: servus" in txt

    def test_http_report_writes_per_speaker_txt(self, tmp_path):
        # http backend has no segment timestamps — transcript.txt must still
        # be written via the per-speaker fallback (this is also what lands
        # in the session notes).
        d = tmp_path / "rec"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"duration_s": 10.0}))
        manifest = {"manifest_path": str(d / "manifest.json")}

        report = T.TranscriptionReport(model="qwen3-asr-1.7b", language_requested="",
                                       started_at=1.0, backend="http")
        report.speakers.append(_spk(1, "Alice", text="hello there"))
        report.finished_at = 3.5

        T.write_transcript(d, manifest, report)
        txt = (d / "transcript.txt").read_text(encoding="utf-8")
        assert "[Alice]" in txt and "hello there" in txt


# ── silence trim / chunk planning / segment merge (http pipeline) ────────────
class TestTrimSilence:
    def test_trims_leading_and_trailing_silence(self):
        rate = 16_000
        sil = np.zeros(rate * 2, dtype=np.int16)
        loud = np.full(rate * 1, 8000, dtype=np.int16)   # ~-12 dBFS
        pcm = np.concatenate([sil, loud, sil])
        trimmed, offset = T._trim_silence(pcm, rate=rate, silence_dbfs=-45.0)
        assert len(trimmed) == rate * 1          # only the loud second remains
        assert offset == rate * 2                # started after 2s of silence
        assert np.all(np.abs(trimmed.astype(int)) >= 7999)

    def test_keeps_quiet_speech_above_threshold(self):
        rate = 16_000
        quiet = np.full(rate, 2000, dtype=np.int16)     # ~-24 dBFS, above -45
        pcm = np.concatenate([np.zeros(rate // 2, dtype=np.int16), quiet,
                              np.zeros(rate // 2, dtype=np.int16)])
        trimmed, offset = T._trim_silence(pcm, rate=rate, silence_dbfs=-45.0)
        assert len(trimmed) == rate and offset == rate // 2

    def test_fully_silent_trims_to_empty(self):
        pcm = np.zeros(16_000, dtype=np.int16)
        trimmed, offset = T._trim_silence(pcm, rate=16_000, silence_dbfs=-45.0)
        assert len(trimmed) == 0 and offset == 0

    def test_empty_input(self):
        trimmed, offset = T._trim_silence(np.zeros(0, dtype=np.int16), rate=16_000)
        assert len(trimmed) == 0 and offset == 0


class TestPlanChunks:
    def test_single_chunk_when_small(self):
        assert T._plan_chunks(1000, rate=16_000, max_bytes=None, chunk_seconds=0) == [(0, 1000)]

    def test_size_cap_splits_and_covers(self):
        # cap that allows exactly (2044-44)//2 = 1000 samples per chunk
        ranges = T._plan_chunks(3500, rate=16_000, max_bytes=2044, chunk_seconds=0)
        assert all(e - s <= 1000 for s, e in ranges)
        assert ranges[0] == (0, 1000) and ranges[-1][1] == 3500
        assert sum(e - s for s, e in ranges) == 3500   # contiguous + covering

    def test_time_cap_splits(self):
        ranges = T._plan_chunks(960_000, rate=16_000, max_bytes=None, chunk_seconds=30)
        assert ranges == [(0, 480_000), (480_000, 960_000)]

    def test_tightest_bound_wins(self):
        # size cap allows 1000 samples; time cap allows 2s (32000) -> size wins
        ranges = T._plan_chunks(5000, rate=16_000, max_bytes=2044, chunk_seconds=2)
        assert all(e - s <= 1000 for s, e in ranges)

    def test_zero_bounds_is_single_chunk(self):
        assert T._plan_chunks(12345, rate=16_000, max_bytes=None, chunk_seconds=0) == [(0, 12345)]

    def test_empty_returns_nothing(self):
        assert T._plan_chunks(0, rate=16_000, max_bytes=2044) == []


class TestPcmToWavBytes:
    def test_roundtrip_header_and_payload(self):
        pcm = np.arange(100, dtype=np.int16)
        data = T._pcm_to_wav_bytes(pcm, rate=16_000)
        assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
        with wave.open(io.BytesIO(data), "rb") as wf:
            assert wf.getframerate() == 16_000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 100
            back = np.frombuffer(wf.readframes(100), dtype=np.int16)
        assert list(back) == list(pcm)


class TestMergeSegments:
    def test_shifts_by_chunk_offset(self):
        segs = [{"start": 0.5, "end": 2.0, "text": "hello"},
                {"start": 3.0, "end": 4.5, "text": "world"}]
        out = T._merge_segments(segs, t_start=60.0)
        assert out[0]["start"] == 60.5 and out[0]["end"] == 62.0
        assert out[1]["start"] == 63.0 and out[1]["text"] == "world"

    def test_zero_offset_is_identity(self):
        segs = [{"start": 1.0, "end": 2.0, "text": "x"}]
        assert T._merge_segments(segs, t_start=0.0) == segs


class TestHttpChunkedPipeline:
    """End-to-end http path: trim -> resample -> chunk -> merge (fake backend)."""

    def _wav_with_padding(self, path, *, rate=48_000, loud_ms=1000,
                          lead_silence_ms=2000, tail_silence_ms=2000):
        n = lambda ms: rate * ms // 1000
        samples = np.concatenate([
            np.zeros(n(lead_silence_ms), dtype=np.int16),
            np.full(n(loud_ms), 8000, dtype=np.int16),
            np.zeros(n(tail_silence_ms), dtype=np.int16),
        ])
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
            wf.writeframes(samples.tobytes())

    class _ChunkClient:
        """OpenAI-client-shaped fake: returns one segment per call, times
        relative to that chunk's start (as a real backend would)."""
        def __init__(self):
            self.calls: list[dict] = []
            self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))
        async def _create(self, **kw):
            idx = len(self.calls)
            self.calls.append(kw)
            return {"text": f"part{idx}", "language": "de",
                    "segments": [{"start": 0.1, "end": 0.9, "text": f"part{idx}"}]}

    @pytest.mark.asyncio
    async def test_trims_padding_and_shifts_segments(self, tmp_path, monkeypatch):
        from config import settings as S
        monkeypatch.setattr(S, "STT_TRIM_SILENCE", True, raising=False)
        monkeypatch.setattr(S, "STT_MAX_UPLOAD_MB", 25, raising=False)
        monkeypatch.setattr(S, "STT_CHUNK_SECONDS", 600, raising=False)

        p = tmp_path / "spk.wav"
        self._wav_with_padding(p)  # 2s sil + 1s loud + 2s sil @48k (~5s total)
        fake = self._ChunkClient()
        monkeypatch.setattr(T, "_stt_client", lambda: fake)

        r = await T.transcribe_wav(p, model="m")
        assert r.ok
        # Padding was trimmed -> a single small upload (not the full ~5s).
        assert len(fake.calls) == 1
        uploaded = fake.calls[0]["file"][1]
        with wave.open(io.BytesIO(uploaded), "rb") as wf:
            assert abs(wf.getframerate() - 16_000) <= 1
            assert wf.getnframes() < 48_000          # far less than the ~80k untrimmed
        # The loud part sat at 2-3s on the timeline -> merged segment starts ~2.1s.
        assert len(r.segments) == 1
        assert abs(r.segments[0]["start"] - 2.1) < 0.15

    @pytest.mark.asyncio
    async def test_chunks_long_audio_and_merges_offsets(self, tmp_path, monkeypatch):
        from config import settings as S
        monkeypatch.setattr(S, "STT_TRIM_SILENCE", True, raising=False)
        monkeypatch.setattr(S, "STT_MAX_UPLOAD_MB", 0, raising=False)   # no size cap
        monkeypatch.setattr(S, "STT_CHUNK_SECONDS", 1, raising=False)   # 1s chunks

        p = tmp_path / "spk.wav"
        self._wav_with_padding(p, loud_ms=4000, lead_silence_ms=0, tail_silence_ms=0)
        fake = self._ChunkClient()
        monkeypatch.setattr(T, "_stt_client", lambda: fake)

        r = await T.transcribe_wav(p, model="m")
        assert r.ok
        # 4s of audio in 1s chunks -> 4 uploads; each segment shifted by its chunk.
        assert len(fake.calls) == 4
        starts = [seg["start"] for seg in r.segments]
        assert len(starts) == 4
        expected = [0.1, 1.1, 2.1, 3.1]
        for got, want in zip(starts, expected):
            assert abs(got - want) < 0.05, f"{got} != {want}"
