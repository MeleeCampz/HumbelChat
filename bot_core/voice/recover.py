"""Crash recovery for voice recordings.

Extracted from ``bot_core.voice_recorder`` (P3 #35). Rebuilds a recording
whose process died mid-capture (OOM/segfault/power) from its on-disk frame
logs into WAVs + a manifest flagged ``"recovered": true``. Auto-runs at
startup for orphans older than ~5 minutes (via the bot wiring in
:mod:`bot_core.voice.session`) and is also available as a manual call.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from bot_core.voice.capture import (
    CHANNELS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    _MARKER_NAME,
    _SpeakerLog,
    _wav_filename,
    _write_timeline_wav_from_frames,
)

log = logging.getLogger("bot.voice_recorder")

_RECOVERY_THRESHOLD_S: float = 5 * 60.0   # auto-recover orphans older than this

def find_open_recording(recordings_dir: Path, guild_id: Optional[int] = None) -> Optional[Path]:
    """Newest directory that has a marker and no manifest, optionally for one guild.

    Age is ignored. ``/stop_recording`` uses this after a restart, when the
    in-memory recorder is empty but the frame logs are already on disk.
    """
    recordings_dir = Path(recordings_dir)
    if not recordings_dir.is_dir():
        return None
    best: tuple[float, Path] | None = None
    for entry in recordings_dir.iterdir():
        if not entry.is_dir():
            continue
        marker_path = entry / _MARKER_NAME
        if not marker_path.exists() or (entry / "manifest.json").exists():
            continue
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            started_at = float(marker.get("started_at", 0.0))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        marker_guild = marker.get("guild_id")
        if guild_id is not None and marker_guild not in (None, guild_id):
            continue
        if best is None or started_at >= best[0]:
            best = (started_at, entry)
    return None if best is None else best[1]


def recover_recording(rec_dir: Path, *, now: Optional[float] = None) -> Optional[dict]:
    """Finalize one open recording regardless of how recently it started."""
    return _recover_one(Path(rec_dir), time.time() if now is None else now)


def recover_orphans(
    recordings_dir: Path,
    *,
    threshold_s: float = _RECOVERY_THRESHOLD_S,
    now: Optional[float] = None,
) -> list[dict]:
    """Rebuild WAVs + manifest for recording dirs left open by a crash.

    An *orphan* is a directory under ``recordings_dir`` that has a ``.recording``
    marker but no ``manifest.json`` — i.e. the process captured audio (spilling
    it to per-speaker ``*.log`` files) but never ran :meth:`VoiceRecorder.stop`.
    This happens when the bot is OOM-killed, segfaults, loses power, or is
    stopped (including SIGTERM) while a recording is still open.

    Only orphans whose marker ``started_at`` is at least ``threshold_s`` in the
    past are recovered — a very recent one might belong to a recording that is
    still live (e.g. the bot restarted while a meeting was ongoing), and we
    don't want to silently truncate it. Returns the list of manifests written.
    """
    recordings_dir = Path(recordings_dir)
    if not recordings_dir.is_dir():
        return []
    now = time.time() if now is None else now
    recovered: list[dict] = []
    for entry in sorted(recordings_dir.iterdir()):
        if not entry.is_dir():
            continue
        marker_path = entry / _MARKER_NAME
        if not marker_path.exists() or (entry / "manifest.json").exists():
            continue
        try:
            started_at = float(json.loads(marker_path.read_text()).get("started_at", 0.0))
        except (OSError, ValueError, json.JSONDecodeError):
            log.warning("Recovery: unreadable marker in %s — skipping", entry.name)
            continue
        if now - started_at < threshold_s:
            log.info(
                "Recovery: %s started %.0fs ago (< %.0fs) — leaving it alone",
                entry.name, now - started_at, threshold_s,
            )
            continue
        manifest = _recover_one(entry, now)
        if manifest is not None:
            recovered.append(manifest)
    return recovered


def _recover_one(rec_dir: Path, now: float) -> Optional[dict]:
    """Rebuild one orphaned recording into WAVs + a ``recovered`` manifest."""
    try:
        marker = json.loads((rec_dir / _MARKER_NAME).read_text())
    except (OSError, ValueError, json.JSONDecodeError) as e:
        log.warning("Recovery: could not read marker in %s: %s", rec_dir.name, e)
        return None
    started_at = float(marker.get("started_at", now))

    # Collect this recording's speaker logs (path + decoded frames).
    logs: list[tuple[Path, list[tuple[float, bytes]]]] = []
    for p in sorted(rec_dir.iterdir()):
        if p.suffix != ".log":
            continue
        frames = _SpeakerLog.read_frames(p)
        if not frames:
            continue
        logs.append((p, frames))
    if not logs:
        log.warning("Recovery: %s has no recoverable speaker logs — removing marker", rec_dir.name)
        try:
            (rec_dir / _MARKER_NAME).unlink(missing_ok=True)
        except OSError:  # pragma: no cover
            pass
        return None

    # Shared timeline span across all speakers (same rule as stop()).
    max_end = started_at
    for _, frames in logs:
        first_ts = frames[0][0]
        max_end = max(max_end, first_ts + len(frames) * FRAME_SAMPLES / SAMPLE_RATE)
    total_samples = int(max(0.0, max_end - started_at) * SAMPLE_RATE)

    speakers_out: list[dict] = []
    for log_path, frames in logs:
        user_id = _user_id_from_log_name(log_path.name)
        display_name = ""
        wav_path = rec_dir / _wav_filename(user_id, display_name)
        gap_frames, overlap_frames = _write_timeline_wav_from_frames(
            path=wav_path, frames=frames, origin=started_at, total_samples=total_samples,
        )
        first_ts = frames[0][0]
        last_end_ts = first_ts + len(frames) * FRAME_SAMPLES / SAMPLE_RATE
        speakers_out.append({
            "user_id": user_id,
            "display_name": display_name,
            "ssrc": None,
            "wav_file": wav_path.name,
            "wav_path": str(wav_path),
            "first_speech_offset_s": round(first_ts - started_at, 3),
            "last_speech_offset_s": round(last_end_ts - started_at, 3),
            "spoken_duration_s": round(len(frames) * FRAME_SAMPLES / SAMPLE_RATE, 3),
            "frames_captured": len(frames),
            "decode_failures": None,
            "decrypt_failures": None,
            "jitter_gap_frames": gap_frames,
            "jitter_overlap_frames": overlap_frames,
        })

    manifest = {
        "guild_id": marker.get("guild_id"),
        "channel_id": marker.get("channel_id"),
        "channel_name": marker.get("channel_name", ""),
        "started_at": round(started_at, 3),
        "ended_at": round(now, 3),
        "duration_s": round(now - started_at, 3),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "bit_depth": SAMPLE_WIDTH * 8,
        "recovered": True,
        "recovery_note": (
            "Rebuilt after an unclean shutdown: the bot captured this audio to "
            "disk but was killed before writing WAVs. Audio is complete up to "
            "the last flushed frame; anything in flight at crash time (at most a "
            "few 20 ms frames) is lost. Decode/decrypt failure counts are not "
            "available for recovered recordings."
        ),
        "speakers": speakers_out,
    }

    manifest_path = rec_dir / "manifest.json"
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
    except OSError as e:  # pragma: no cover - disk errors
        log.error("Recovery: could not write manifest for %s: %s", rec_dir.name, e)
        return None

    # Success — remove the intermediate logs + marker.
    try:
        for p in list(rec_dir.iterdir()):
            if p.suffix == ".log" or p.name == _MARKER_NAME:
                p.unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        pass
    log.info(
        "Recovered orphaned recording %s: %d speaker(s), %.1fs -> %s",
        rec_dir.name, len(speakers_out), manifest["duration_s"], rec_dir,
    )
    return manifest


def _user_id_from_log_name(log_name: str) -> int:
    """Recover the user id from a ``<name>_<user_id>.log`` filename."""
    stem = log_name[:-4] if log_name.endswith(".log") else log_name
    m = re.search(r"_(\d+)$", stem)
    return int(m.group(1)) if m else 0
