"""Per-speaker audio capture, on-disk frame log, and timeline WAV layout.

Extracted from ``bot_core.voice_recorder`` (P3 #35). Each speaker's decoded
frames are appended to an append-only binary log on disk the moment they
decode (crash durability — a crash loses only the few frames still in
flight, not hours of audio); on stop the logs are read back and laid out
on the shared nominal 20 ms timeline into 16-bit mono WAV files.
"""
from __future__ import annotations

import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48_000          # Discord voice is always 48 kHz
CHANNELS = 1                  # we record mono (one file per speaker)
SAMPLE_WIDTH = 2              # 16-bit PCM
FRAME_SAMPLES = SAMPLE_RATE // 50   # one nominal voice frame == 20 ms of audio
# ── Crash durability ────────────────────────────────────────────────────────
# Every decoded frame is appended to a per-speaker, append-only binary log on
# disk the moment it decodes (instead of sitting in an unbounded RAM list), so
# a crash mid-recording loses only the few frames still in flight — not hours
# of audio. On stop() / after a crash the logs are read back and laid out on
# the same nominal 20 ms grid as before.
#
# Log layout (little-endian):
#   header : "DRAV" + uint32 version(=1) + uint32 sample_rate
#            + uint32 channels + uint32 sample_width + uint32 frame_samples
#   record : int64 ts_us  |  uint32 pcm_len  |  pcm bytes   (repeated)
_SPEAKER_LOG_MAGIC = b"DRAV"
_SPEAKER_LOG_VERSION = 1
_MARKER_NAME = ".recording"               # session marker inside each recording dir

class VoiceRecorderError(Exception):
    """Raised for unrecoverable recorder configuration problems."""

@dataclass
class _Speaker:
    user_id: int
    display_name: str = ""
    ssrc: Optional[int] = None
    # On-disk append-only log for this speaker's decoded PCM (None until the
    # first frame arrives). Replaces the old in-RAM frames list — see the
    # crash-durability note above. Counters are kept here for the manifest.
    log: Optional["_SpeakerLog"] = None
    frame_count: int = 0
    total_pcm_bytes: int = 0
    decode_failures: int = 0
    decrypt_failures: int = 0

    @property
    def duration_s(self) -> float:
        return self.total_pcm_bytes / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)

class _SpeakerLog:
    """Append-only, on-disk log of one speaker's decoded PCM frames.

    Each record is ``int64 ts_us | uint32 pcm_len | pcm bytes`` following a
    small header. Frames are appended (and flushed) the moment they decode,
    so a crash loses at most the frame currently in flight rather than the
    whole in-memory buffer. The same reader (:meth:`read_frames`) is used by
    ``stop()`` and by crash recovery, guaranteeing both paths rebuild an
    identical timeline.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh = open(self.path, "ab")
        hdr = (
            _SPEAKER_LOG_MAGIC
            + struct.pack("<IIIII", _SPEAKER_LOG_VERSION, SAMPLE_RATE,
                          CHANNELS, SAMPLE_WIDTH, FRAME_SAMPLES)
        )
        self._fh.write(hdr)
        self._fh.flush()

    def write_frame(self, ts: float, pcm: bytes) -> None:
        """Append one decoded frame. ``ts`` is an absolute epoch (seconds)."""
        self._fh.write(struct.pack("<qI", int(ts * 1_000_000), len(pcm)))
        self._fh.write(pcm)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover - already closed / OS error
            pass

    @staticmethod
    def read_frames(path: Path) -> list[tuple[float, bytes]]:
        """Read back ``(epoch_seconds, pcm_bytes)`` records in arrival order.

        Tolerates a trailing partial record (a frame that was mid-write when
        the process died) by stopping at the first incomplete one — the loss
        is bounded to that single 20 ms frame. Returns ``[]`` for an empty or
        malformed log.
        """
        data = Path(path).read_bytes()
        if len(data) < 4 or data[:4] != _SPEAKER_LOG_MAGIC:
            return []
        offset = 4 + struct.calcsize("<IIIII")
        out: list[tuple[float, bytes]] = []
        n_rec = struct.calcsize("<qI")
        while offset + n_rec <= len(data):
            ts_us, pcm_len = struct.unpack_from("<qI", data, offset)
            offset += n_rec
            if offset + pcm_len > len(data):
                break  # trailing partial record from a crash — drop it
            out.append((ts_us / 1_000_000.0, data[offset:offset + pcm_len]))
            offset += pcm_len
        return out

    @staticmethod
    def read_meta(path: Path) -> Optional[dict]:
        """Parse the log header (sample_rate/channels/width/frame_samples)."""
        try:
            data = Path(path).read_bytes()
            if len(data) < 4 or data[:4] != _SPEAKER_LOG_MAGIC:
                return None
            version, rate, ch, width, frame_samples = struct.unpack_from(
                "<IIIII", data, 4)
        except (OSError, struct.error):
            return None
        if version != _SPEAKER_LOG_VERSION:
            return None
        return {"sample_rate": rate, "channels": ch,
                "sample_width": width, "frame_samples": frame_samples}

def _write_timeline_wav_from_frames(
    *, path: Path, frames: list[tuple[float, bytes]], origin: float,
    total_samples: int,
) -> tuple[int, int]:
    """Lay *frames* onto the shared nominal 20 ms grid and write a WAV.

    This is the single source of truth for timeline placement — used by both
    ``stop()`` and crash recovery so they produce identical output. Frames are
    anchored at the first frame's arrival offset, then exactly one nominal
    frame (``FRAME_SAMPLES`` = 20 ms) apart, regardless of jittery wall-clock
    arrivals (see :func:`_write_timeline_wav`). Returns ``(gap_frames,
    overlap_frames)`` diagnostics.
    """
    samples = bytearray(total_samples * SAMPLE_WIDTH)  # zero-filled silence
    cursor = int(max(0.0, frames[0][0] - origin) * SAMPLE_RATE) if frames else 0
    gap_frames = overlap_frames = 0
    prev_arr: Optional[int] = None
    for ts, pcm in frames:
        arr = int(max(0.0, ts - origin) * SAMPLE_RATE)
        if prev_arr is not None:
            delta = arr - prev_arr
            if delta > FRAME_SAMPLES + FRAME_SAMPLES // 2:
                gap_frames += 1      # arrival >30 ms late -> jitter hole
            elif delta < FRAME_SAMPLES - FRAME_SAMPLES // 2:
                overlap_frames += 1  # arrival <10 ms early -> burst
        prev_arr = arr
        byte_idx = cursor * SAMPLE_WIDTH
        end_byte = min(len(samples), byte_idx + len(pcm))
        if byte_idx < len(samples):
            samples[byte_idx:end_byte] = pcm[:end_byte - byte_idx]
        cursor += FRAME_SAMPLES
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(samples))
    return gap_frames, overlap_frames

def _wav_filename(user_id: int, display_name: str) -> str:
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in (display_name or "")).strip()
    safe = safe[:40] or f"user-{user_id}"
    return f"{safe}_{user_id}.wav"


def _write_timeline_wav(
    *,
    path: Path,
    frames: list[tuple[float, bytes]],
    origin: float,
    total_samples: int,
) -> tuple[int, int]:
    """Write a speaker's frames onto the shared timeline.

    Frames are placed on a *nominal* grid: the first frame is anchored at its
    wall-clock offset from ``origin``, and every subsequent frame starts
    exactly one nominal frame (``FRAME_SAMPLES`` = 20 ms) after the previous
    one. Placing by raw arrival time would corrupt the audio because voice
    packets arrive with network jitter — intervals anywhere from ~15 ms to
    ~40 ms for frames that each hold a fixed 20 ms of audio: arrivals >20 ms
    apart punch silence holes in the middle of speech, and bursts <20 ms apart
    make one frame overwrite part of the previous one.

    All speakers share the same time base (``origin``), so their WAVs can be
    interleaved for reconstruction. Returns ``(gap_frames, overlap_frames)`` —
    how often consecutive arrivals deviated more than ±50% from the nominal
    20 ms interval — for manifest diagnostics.
    """
    return _write_timeline_wav_from_frames(
        path=path, frames=frames, origin=origin, total_samples=total_samples)
