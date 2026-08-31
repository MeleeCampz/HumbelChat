"""Speech-to-text for voice recordings, on the same OpenAI-compatible backend.

After ``/stop_recording`` writes one WAV per speaker, this module transcribes
each file through the backend's ``/v1/audio/transcriptions`` endpoint (the
same base URL + API key the bot already uses for chat) and writes a
``transcript.json`` next to the recording's ``manifest.json``.

Backend notes (verified against unsloth-studio):
  * The endpoint accepts OpenAI-style multipart uploads (``file`` + ``model``).
  * Valid model slugs are backend-specific — curated defaults on
    unsloth-studio: ``tiny``, ``base``, ``small``, ``large-v3-turbo``,
    ``large-v3``, ``qwen3-asr-0.6b``, ``qwen3-asr-1.7b`` (or any HF repo in
    ``owner/model`` form). The slug is configurable via ``STT_MODEL``.
  * There is a ~25 MB per-request audio limit, so files are downsampled to
    16 kHz mono before upload (Whisper-class models want 16 kHz anyway).

This module is import-safe: no network access happens until
:func:`transcribe_wav` / :func:`transcribe_recording` are called.
"""
from __future__ import annotations

import json
import logging
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("bot.transcriber")

TARGET_SAMPLE_RATE = 16_000   # Whisper-class ASR expects 16 kHz mono
MAX_UPLOAD_BYTES = 24 * 1024 * 1024  # stay under the backend's ~25 MB cap


class TranscriptionError(Exception):
    """Raised when a WAV cannot be prepared for upload (bad format etc.)."""


_shared_stt_client = None


def _stt_client():
    """Shared AsyncOpenAI client for the backend (same URL/key as chat)."""
    global _shared_stt_client
    if _shared_stt_client is None:
        from openai import AsyncOpenAI

        from config.settings import INFER_API_KEY, INFER_URL
        _shared_stt_client = AsyncOpenAI(api_key=INFER_API_KEY, base_url=INFER_URL)
    return _shared_stt_client


def _to_16k_mono_wav(wav_path: Path) -> bytes:
    """Read a WAV and return it as 16 kHz mono 16-bit PCM *WAV* bytes.

    The recorder writes 48 kHz mono; the backend's STT sidecar decodes to
    16 kHz internally, so we do it here to shrink uploads ~3x (the backend
    caps requests at ~25 MB — a 48 kHz WAV hits that in ~4 minutes).
    """
    import io

    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width != 2:
        raise TranscriptionError(f"Unsupported WAV bit depth {sample_width * 8} in {wav_path.name}")

    import numpy as np
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if rate != TARGET_SAMPLE_RATE:
        n_out = int(n_frames * TARGET_SAMPLE_RATE / rate)
        x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        samples = np.interp(x_new, x_old, samples)

    pcm = np.clip(samples, -32768.0, 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(TARGET_SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return buf.getvalue()


@dataclass
class SpeakerResult:
    """Outcome of transcribing one speaker's WAV."""
    user_id: int
    display_name: str
    wav_file: str
    text: str = ""
    language: str | None = None
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class TranscriptionReport:
    """Aggregate result for one recording."""
    model: str
    language_requested: str
    started_at: float
    finished_at: float = 0.0
    speakers: list[SpeakerResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for s in self.speakers if s.ok)

    @property
    def failed(self) -> list[SpeakerResult]:
        return [s for s in self.speakers if not s.ok]


def _error_message(exc: Exception) -> str:
    """Extract a human-readable message from an OpenAI API error."""
    return getattr(exc, "message", None) or str(exc)


async def transcribe_wav(wav_path: Path, *, model: str, language: str = "") -> SpeakerResult:
    """Transcribe one WAV file via ``/v1/audio/transcriptions``.

    Returns a :class:`SpeakerResult` (never raises for expected backend
    errors — those are captured in ``result.error``).
    """
    from config.settings import STT_TIMEOUT

    wav_path = Path(wav_path)
    if not wav_path.exists():
        return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name,
                             error=f"file not found: {wav_path.name}")

    try:
        upload_bytes = _to_16k_mono_wav(wav_path)
    except TranscriptionError as e:
        return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name, error=str(e))
    except Exception as e:  # corrupt/missing WAV
        return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name,
                             error=f"could not read WAV: {e}")

    if len(upload_bytes) > MAX_UPLOAD_BYTES:
        return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name,
                             error=(f"audio too large for backend ({len(upload_bytes) / 1024 / 1024:.1f} MB > "
                                    f"{MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB limit)"))

    fields: dict = {"model": model, "timeout": STT_TIMEOUT}
    if language:
        fields["language"] = language

    client = _stt_client()
    started = time.monotonic()
    try:
        resp = await client.audio.transcriptions.create(
            file=(wav_path.name, upload_bytes, "audio/wav"), **fields,
        )
    except Exception as e:  # noqa: BLE001 - surface backend errors per-speaker
        msg = _error_message(e)
        log.warning("STT failed for %s (model=%s): %s", wav_path.name, model, msg)
        return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name, error=msg)

    elapsed = time.monotonic() - started
    text = ""
    lang: str | None = None
    if isinstance(resp, dict):
        text = resp.get("text") or ""
        lang = resp.get("language")
    else:
        text = getattr(resp, "text", "") or ""
        lang = getattr(resp, "language", None)

    log.info("STT done: %s -> %d chars in %.1fs (model=%s)", wav_path.name, len(text), elapsed, model)
    return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name,
                         text=text.strip(), language=lang, elapsed_s=round(elapsed, 2))


def resolve_recording_dir(manifest: dict) -> Path | None:
    """Locate the recording directory that holds the speakers' WAV files.

    In production the manifest comes from ``VoiceRecorder.stop()`` and always
    carries an absolute ``manifest_path``. The fallback covers manifests loaded
    back from disk (where ``manifest_path`` is absent) by scanning
    ``RECORDINGS_DIR`` for a directory containing the first speaker's WAV.
    """
    from config.settings import RECORDINGS_DIR

    mp = manifest.get("manifest_path")
    if mp:
        candidate = Path(mp).parent
        if (candidate / "manifest.json").exists():
            return candidate
    if RECORDINGS_DIR.exists():
        first_wav = next((s.get("wav_file") for s in manifest.get("speakers", [])), "")
        if first_wav:
            try:
                dirs = sorted(RECORDINGS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            except OSError:
                return None
            for d in dirs:
                if d.is_dir() and (d / first_wav).exists():
                    return d
    return None


async def transcribe_recording(manifest: dict) -> TranscriptionReport:
    """Transcribe every speaker in a recording manifest.

    Files are processed sequentially (one local backend = one slot, same
    reasoning as the global AI lock). Returns a report; per-speaker failures
    never abort the rest.
    """
    from config.settings import STT_LANGUAGE, STT_MODEL

    out_dir = resolve_recording_dir(manifest)
    report = TranscriptionReport(model=STT_MODEL, language_requested=STT_LANGUAGE,
                                 started_at=time.time())

    for sp in manifest.get("speakers", []):
        wav_file = sp.get("wav_file", "")
        wav_path = (out_dir / wav_file) if out_dir else Path(wav_file)
        result = await transcribe_wav(
            wav_path, model=STT_MODEL, language=STT_LANGUAGE or "",
        )
        result.user_id = sp.get("user_id", 0)
        result.display_name = sp.get("display_name", "")
        report.speakers.append(result)

    report.finished_at = time.time()
    return report


def write_transcript(out_dir: Path, manifest: dict, report: TranscriptionReport) -> Path:
    """Write ``transcript.json`` into the recording directory and update the
    on-disk manifest with a pointer to it. Returns the transcript path."""
    out_dir = Path(out_dir)
    payload = {
        "model": report.model,
        "language_requested": report.language_requested or None,
        "started_at": round(report.started_at, 3),
        "finished_at": round(report.finished_at, 3),
        "elapsed_s": round(report.finished_at - report.started_at, 2),
        "speakers": [
            {
                "user_id": s.user_id,
                "display_name": s.display_name,
                "wav_file": s.wav_file,
                "text": s.text if s.ok else None,
                "language": s.language,
                "elapsed_s": s.elapsed_s,
                "error": s.error,
            }
            for s in report.speakers
        ],
    }
    path = out_dir / "transcript.json"
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        # Point the manifest at the transcript (best effort).
        mp = out_dir / "manifest.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            m["transcript"] = path.name
            m["stt_model"] = report.model
            mp.write_text(json.dumps(m, indent=2), encoding="utf-8")
    except OSError as e:  # pragma: no cover - disk errors
        log.error("Failed to write transcript.json: %s", e)
    return path
