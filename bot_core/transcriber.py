"""Speech-to-text for voice recordings.

After ``/stop_recording`` writes one WAV per speaker, this module transcribes
each file and writes a ``transcript.json`` (plus a chronological
``transcript.txt``) next to the recording's ``manifest.json``. When a
session is active, the same transcript text is appended to that session's
notes (see :func:`bot_core.sessions.add_transcript`).

Two backends are supported, selected via ``STT_BACKEND``:

  * ``local`` (default) — faster-whisper running on this machine. It returns
    real per-segment timestamps, which is what makes the interleaved
    chronological transcript possible (the HTTP endpoint below returns plain
text only — verified: unsloth-studio's ``TranscribeRequest`` schema has no
    timestamp/task fields and ignores them). No upload-size cap either.
    The model (default ``large-v3-turbo``, int8 on CPU) is loaded lazily on
    first use and cached for the process lifetime; transcription is
    serialized through one lock so concurrent recordings share the slot.
  * ``http`` — the backend's OpenAI-compatible ``/v1/audio/transcriptions``
    endpoint (same base URL + API key the bot already uses for chat).
    Valid model slugs are backend-specific — curated defaults on
    unsloth-studio: ``tiny``, ``base``, ``small``, ``large-v3-turbo``,
    ``large-v3``, ``qwen3-asr-0.6b``, ``qwen3-asr-1.7b`` (or any HF repo in
    ``owner/model`` form). The slug is configurable via ``STT_MODEL``.
    There is a ~25 MB per-request audio limit, so files are downsampled to
    16 kHz mono before upload (Whisper-class models want 16 kHz anyway).

Timeline note: the recorder writes every speaker's WAV aligned to the shared
recording timeline (file time 0 == recording start), so segment timestamps
from all speakers live on one clock and can be merged directly — no per-
speaker offset correction is needed.

This module is import-safe: importing it never loads a model or touches the
network until :func:`transcribe_wav` / :func:`transcribe_recording` are
called.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("bot.transcriber")

TARGET_SAMPLE_RATE = 16_000   # Whisper-class ASR expects 16 kHz mono


class TranscriptionError(Exception):
    """Raised when a WAV cannot be prepared for upload (bad format etc.)."""


_shared_stt_client = None

# ── Local backend (faster-whisper) state ─────────────────────────────────────
_local_models: dict[str, object] = {}      # model name -> WhisperModel
_local_models_lock = threading.Lock()
_inference_lock = threading.Lock()          # serialize CPU transcriptions


def _stt_client():
    """Shared AsyncOpenAI client for the STT backend.

    Uses ``STT_URL`` (falls back to ``INFER_URL`` when unset) so the STT
    service can live in a different container from the chat LLM. The API key
    reuses ``INFER_API_KEY`` — set it to any non-empty string for backends
    that don't require auth.
    """
    global _shared_stt_client
    if _shared_stt_client is None:
        from openai import AsyncOpenAI

        from config.settings import INFER_API_KEY, STT_URL
        api_key = INFER_API_KEY or "no-key"
        _shared_stt_client = AsyncOpenAI(api_key=api_key, base_url=STT_URL)
    return _shared_stt_client


def _load_to_16k_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV and return ``(int16 mono samples, rate)`` at 16 kHz.

    The recorder writes 48 kHz mono; Whisper-class models want 16 kHz, so we
    downmix (if needed) and resample here — a ~3x size reduction that also
    keeps uploads under the backend's per-request cap. The returned array is
    *always* at ``TARGET_SAMPLE_RATE`` (16 kHz), so callers use the returned
    rate for all sample↔second conversions. Raises :class:`TranscriptionError`
    for unsupported bit depths.
    """
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width != 2:
        raise TranscriptionError(f"Unsupported WAV bit depth {sample_width * 8} in {wav_path.name}")

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if rate != TARGET_SAMPLE_RATE:
        n_out = int(n_frames * TARGET_SAMPLE_RATE / rate)
        x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        samples = np.interp(x_new, x_old, samples)

    # The array is now at TARGET_SAMPLE_RATE (resampled above if needed).
    return np.clip(samples, -32768.0, 32767.0).astype(np.int16), TARGET_SAMPLE_RATE


def _to_16k_mono_wav(wav_path: Path) -> bytes:
    """Read a WAV and return it as 16 kHz mono 16-bit PCM *WAV* bytes.

    Kept for callers/tests that want the full file; the http pipeline uses
    :func:`_load_to_16k_mono` + chunking instead so long recordings can be
    split under the upload cap.
    """
    import io

    pcm, _rate = _load_to_16k_mono(wav_path)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(TARGET_SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return buf.getvalue()


def _trim_silence(
    pcm: np.ndarray, *, rate: int = TARGET_SAMPLE_RATE, silence_dbfs: float = -45.0,
) -> tuple[np.ndarray, int]:
    """Trim leading/trailing samples quieter than *silence_dbfs*.

    Returns ``(trimmed_pcm, trim_start_samples)`` where the offset is in the
    16 kHz domain — callers use it to shift segment timestamps back onto the
    shared timeline. A fully-silent clip trims to empty (offset 0).
    """
    import numpy as np

    n = len(pcm)
    if n == 0:
        return pcm, 0
    # dBFS relative to full scale (32768); -45 dBFS ≈ amplitude 32768*10^(-45/20).
    threshold = 32768.0 * (10.0 ** (silence_dbfs / 20.0))
    loud = np.abs(pcm.astype(np.float32)) >= threshold
    if not loud.any():
        return pcm[:0], 0
    first = int(np.argmax(loud))                 # index of first loud sample
    last = n - 1 - int(np.argmax(loud[::-1]))    # index of last loud sample
    return pcm[first:last + 1], first


def _pcm_to_wav_bytes(pcm: np.ndarray, rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Encode an int16 mono array as a minimal RIFF/WAVE file (44-byte header)."""
    import struct

    data = pcm.tobytes()
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data" + struct.pack("<I", len(data))
    )
    return header + data


def _plan_chunks(n_samples: int, *, rate: int = TARGET_SAMPLE_RATE,
                 max_bytes: int | None = None, chunk_seconds: float = 0.0) -> list[tuple[int, int]]:
    """Split *n_samples* of mono PCM into ``(start_sample, end_sample)`` ranges.

    A range is bounded by whichever comes first: the size cap (a chunk's WAV —
    header + data — must fit under *max_bytes*) or the time cap (*chunk_seconds*
    > 0). If neither bound yields a positive length, the whole clip becomes one
    chunk. Ranges are contiguous and cover ``[0, n_samples)`` exactly.
    """
    if n_samples <= 0:
        return []
    step = None
    if max_bytes is not None and max_bytes > 0:
        # 44-byte WAV header + 2 bytes per sample must fit under the cap.
        step = max(1, (max_bytes - 44) // 2)
    if chunk_seconds and chunk_seconds > 0:
        s = int(chunk_seconds * rate)
        if s <= 0:
            raise ValueError(f"chunk_seconds too small for {rate} Hz: {chunk_seconds}")
        step = s if step is None else min(step, s)
    if step is None or step >= n_samples:
        return [(0, n_samples)]
    return [(i, min(i + step, n_samples)) for i in range(0, n_samples, step)]


def _merge_segments(segs: list[dict], *, t_start: float) -> list[dict]:
    """Shift per-chunk segments back onto the shared timeline.

    Each chunk was transcribed independently, so its segment times are relative
    to that chunk's start. Adding *t_start* (the chunk's offset on the speaker's
    16 kHz audio) puts every segment on one clock — the same clock the recorder
    aligns all speakers' WAVs to.
    """
    return [{"start": round(s.get("start", 0.0) + t_start, 3),
             "end": round(s.get("end", 0.0) + t_start, 3),
             "text": (s.get("text") or "").strip()}
            for s in segs]


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
    # Per-segment timestamps (local backend only): [{"start", "end", "text"}]
    # with times relative to the start of this speaker's WAV — which the
    # recorder aligns to the recording start, i.e. the shared timeline.
    segments: list[dict] = field(default_factory=list)

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
    backend: str = "http"   # "local" (faster-whisper) or "http"

    @property
    def ok_count(self) -> int:
        return sum(1 for s in self.speakers if s.ok)

    @property
    def failed(self) -> list[SpeakerResult]:
        return [s for s in self.speakers if not s.ok]


def _error_message(exc: Exception) -> str:
    """Extract a human-readable message from an OpenAI API error."""
    return getattr(exc, "message", None) or str(exc)


def _extract_segments(resp) -> tuple[str, str | None, list[dict]]:
    """Pull ``(text, language, segments)`` out of a verbose_json response.

    The OpenAI SDK may hand back an object (``.segments`` / ``.start`` ...) or a
    plain dict (``"segments"`` / ``"start"`` ...). Segment times are relative to
    the start of *this chunk* — the caller shifts them onto the shared timeline.
    """
    text = ""
    lang: str | None = None
    segs: list[dict] = []
    if isinstance(resp, dict):
        text = resp.get("text") or ""
        lang = resp.get("language")
        for s in resp.get("segments") or []:
            segs.append({"start": round(s.get("start", 0), 3),
                         "end": round(s.get("end", 0), 3),
                         "text": (s.get("text") or "").strip()})
    elif hasattr(resp, "segments"):
        text = getattr(resp, "text", "") or ""
        lang = getattr(resp, "language", None)
        for s in resp.segments:
            segs.append({"start": round(s.start, 3),
                         "end": round(s.end, 3),
                         "text": (s.text or "").strip()})
    else:
        text = getattr(resp, "text", "") or ""
        lang = getattr(resp, "language", None)
    return text.strip(), lang, segs


def _local_model(name: str):
    """Lazily load (and cache) a faster-whisper model by name.

    First call downloads the model (~1.6 GB for large-v3-turbo) and takes
    ~1 min; afterwards it's an instant dict lookup. Loading happens on the
    calling thread under ``_local_models_lock`` so two recordings can't pull
    the same weights twice.
    """
    with _local_models_lock:
        model = _local_models.get(name)
        if model is None:
            from faster_whisper import WhisperModel

            log.info("Loading local STT model %r (first use downloads it)...", name)
            t0 = time.monotonic()
            model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=12)
            _local_models[name] = model
            log.info("Local STT model %r ready in %.1fs", name, time.monotonic() - t0)
        return model


def _local_transcribe(wav_path: Path, model_name: str, language: str) -> tuple[str, str | None, list[dict]]:
    """Run faster-whisper on one WAV *synchronously* (call via to_thread).

    Returns ``(text, detected_language, segments)`` where each segment is
    ``{"start": s, "end": e, "text": t}`` with times in seconds relative to the
    start of the file. VAD filtering skips long silence stretches so quiet
    recordings don't waste decode time.
    """
    model = _local_model(model_name)
    kwargs: dict = {"vad_filter": True}
    if language:
        kwargs["language"] = language
    with _inference_lock:
        segments, info = model.transcribe(str(wav_path), **kwargs)
        segs = [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
                for s in segments]
    text = " ".join(seg["text"] for seg in segs).strip()
    return text, (info.language if info else None), segs


def _stt_backend() -> str:
    from config.settings import STT_BACKEND

    b = (STT_BACKEND or "local").strip().lower()
    return b if b in ("local", "http") else "local"


async def transcribe_wav(wav_path: Path, *, model: str, language: str = "") -> SpeakerResult:
    """Transcribe one WAV file using the configured backend.

    ``model`` is the model *name*: a faster-whisper model id for the local
    backend (e.g. ``large-v3-turbo``) or a backend slug for the HTTP backend
    (e.g. ``qwen3-asr-1.7b``). Returns a :class:`SpeakerResult` (never raises
    for expected errors — those are captured in ``result.error``).
    """
    from config.settings import STT_TIMEOUT

    wav_path = Path(wav_path)
    if not wav_path.exists():
        return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name,
                             error=f"file not found: {wav_path.name}")

    backend = _stt_backend()
    started = time.monotonic()

    if backend == "local":
        # faster-whisper is CPU-bound and synchronous; run it off the event
        # loop so the bot stays responsive while a recording transcribes.
        try:
            text, lang, segs = await asyncio.to_thread(
                _local_transcribe, wav_path, model, language or "",
            )
        except Exception as e:  # noqa: BLE001 - surface per-speaker
            log.warning("Local STT failed for %s (model=%s): %s", wav_path.name, model, e)
            return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name,
                                 error=f"local whisper: {e}")
        elapsed = time.monotonic() - started
        log.info("Local STT done: %s -> %d chars, %d segment(s) in %.1fs (model=%s)",
                 wav_path.name, len(text), len(segs), elapsed, model)
        return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name,
                             text=text, language=lang, elapsed_s=round(elapsed, 2), segments=segs)

    # ── http backend (OpenAI-compatible /v1/audio/transcriptions) ────────────
    # Pipeline: load + resample to 16 kHz mono -> trim edge silence -> split
    # into chunks that fit under the upload cap -> transcribe each chunk ->
    # merge segments back onto the shared timeline. See _plan_chunks.
    from config.settings import STT_CHUNK_SECONDS, STT_MAX_UPLOAD_MB, STT_SILENCE_DBFS, STT_TRIM_SILENCE

    try:
        pcm16, rate = _load_to_16k_mono(wav_path)
    except TranscriptionError as e:
        return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name, error=str(e))
    except Exception as e:  # corrupt/missing WAV
        return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name,
                             error=f"could not read WAV: {e}")

    t_start = 0.0
    if STT_TRIM_SILENCE:
        pcm16, trim_samples = _trim_silence(pcm16, rate=rate, silence_dbfs=STT_SILENCE_DBFS)
        t_start = trim_samples / rate

    max_bytes = (STT_MAX_UPLOAD_MB * 1024 * 1024) if STT_MAX_UPLOAD_MB > 0 else None
    try:
        ranges = _plan_chunks(len(pcm16), rate=rate, max_bytes=max_bytes,
                              chunk_seconds=float(STT_CHUNK_SECONDS or 0))
    except ValueError as e:
        return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name, error=str(e))

    fields: dict = {
        "model": model,
        "timeout": STT_TIMEOUT,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }
    if language:
        fields["language"] = language

    client = _stt_client()
    text_parts: list[str] = []
    lang: str | None = None
    all_segs: list[dict] = []
    for i, (s, e) in enumerate(ranges):
        chunk_name = wav_path.name if len(ranges) == 1 else f"{wav_path.stem}__chunk{i + 1:02d}.wav"
        try:
            resp = await client.audio.transcriptions.create(
                file=(chunk_name, _pcm_to_wav_bytes(pcm16[s:e]), "audio/wav"), **fields,
            )
        except Exception as exc:  # noqa: BLE001 - surface backend errors per-speaker
            msg = _error_message(exc)
            log.warning("STT failed for %s chunk %d/%d (model=%s): %s",
                        wav_path.name, i + 1, len(ranges), model, msg)
            return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name, error=msg)
        c_text, c_lang, c_segs = _extract_segments(resp)
        text_parts.append(c_text)
        if lang is None and c_lang:
            lang = c_lang
        all_segs.extend(_merge_segments(c_segs, t_start=t_start + s / rate))

    merged_text = " ".join(p for p in text_parts if p).strip()
    elapsed = time.monotonic() - started
    log.info("STT done: %s -> %d chars, %d segment(s) over %d chunk(s) in %.1fs (model=%s)",
             wav_path.name, len(merged_text), len(all_segs), len(ranges), elapsed, model)
    return SpeakerResult(user_id=0, display_name="", wav_file=wav_path.name,
                         text=merged_text, language=lang,
                         elapsed_s=round(elapsed, 2), segments=all_segs)


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
    from config.settings import STT_BACKEND, STT_LANGUAGE, STT_LOCAL_MODEL, STT_MODEL

    backend = _stt_backend()
    # The local engine takes faster-whisper model ids; the HTTP endpoint takes
    # backend slugs — different namespaces, so pick per backend.
    model = STT_LOCAL_MODEL if backend == "local" else STT_MODEL

    out_dir = resolve_recording_dir(manifest)
    report = TranscriptionReport(model=model, language_requested=STT_LANGUAGE,
                                 started_at=time.time(), backend=backend)

    for sp in manifest.get("speakers", []):
        wav_file = sp.get("wav_file", "")
        wav_path = (out_dir / wav_file) if out_dir else Path(wav_file)
        result = await transcribe_wav(
            wav_path, model=model, language=STT_LANGUAGE or "",
        )
        result.user_id = sp.get("user_id", 0)
        result.display_name = sp.get("display_name", "")
        report.speakers.append(result)

    report.finished_at = time.time()
    return report


def build_session_transcript(report: TranscriptionReport) -> str:
    """The transcript text that ``transcript.txt`` and the session notes use.

    Prefers :func:`build_interleaved_transcript` (chronological, per-segment
    timestamps — local backend). When no speaker produced timestamps (http
    backend), falls back to one attributed block per speaker so the session
    notes still capture who said what. Returns "" when there is no speech.
    """
    text = build_interleaved_transcript(report)
    if text.strip():
        return text
    blocks: list[str] = []
    for s in report.speakers:
        if s.ok and s.text:
            name = s.display_name or f"user {s.user_id}"
            blocks.append(f"[{name}]\n{s.text}\n")
    return "\n".join(blocks).rstrip() + ("\n" if blocks else "")


def build_interleaved_transcript(report: TranscriptionReport) -> str:
    """Merge all speakers' timestamped segments into one chronological text.

    Every speaker's WAV is aligned to the recording start by the recorder, so
    segment times from different speakers live on the same clock and can be
    sorted directly. Speakers without timestamps (http backend) are appended
    at the end in their own blocks. Returns "" when there is no speech.
    """
    def _ts(t: float) -> str:
        m = int(t // 60)
        s = t - m * 60
        return f"{m:02d}:{s:05.2f}"

    named = {s.user_id: (s.display_name or f"user {s.user_id}") for s in report.speakers}
    events: list[tuple[float, int, str, str]] = []   # (start, seq, name, text)
    untimed: list[str] = []
    seq = 0
    for s in report.speakers:
        if not s.ok or not (s.text or s.segments):
            continue
        name = s.display_name or f"user {s.user_id}"
        if s.segments:
            for seg in s.segments:
                events.append((seg["start"], seq, name, seg["text"]))
                seq += 1
        else:
            untimed.append(f"[{name}]\n{s.text}\n")

    lines: list[str] = []
    for start, _seq, name, text in sorted(events, key=lambda e: (e[0], e[1])):
        if not text:
            continue
        lines.append(f"[{_ts(start)}] {name}: {text}")

    if untimed:
        if lines:
            lines.append("")
        lines.extend(untimed)
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def write_transcript(out_dir: Path, manifest: dict, report: TranscriptionReport) -> Path:
    """Write ``transcript.json`` + interleaved ``transcript.txt`` into the
    recording directory and update the on-disk manifest with pointers to them.
    Returns the transcript.json path."""
    out_dir = Path(out_dir)
    payload = {
        "model": report.model,
        "backend": report.backend,
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
                **({"segments": s.segments} if s.segments else {}),
            }
            for s in report.speakers
        ],
    }
    path = out_dir / "transcript.json"
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        # Chronological transcript (per-segment timestamps on the local
        # backend; per-speaker blocks as fallback for the http backend).
        transcript_text = build_session_transcript(report)
        if transcript_text:
            (out_dir / "transcript.txt").write_text(transcript_text, encoding="utf-8")

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
