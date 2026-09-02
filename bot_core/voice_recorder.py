"""Voice-channel audio capture for later STT transcription.

This module lets the bot join a voice channel, capture every participant's
audio *separately* (so each speaker can be attributed and transcribed
independently), and write the result to disk as one 16-bit PCM WAV file per
speaker plus a JSON manifest carrying absolute timestamps so a conversation
can be reconstructed on a shared timeline.

How it works
------------
**Call :meth:`VoiceRecorder.start` *before* joining (or moving into) the
voice channel.** Discord sends the initial ``op-11``/``op-5`` burst (which is
the only place SSRC -> user ids are announced — op-5 otherwise only fires on
speaking-state *changes*) as soon as the voice connection establishes, i.e.
during ``channel.connect()``. If ``start()`` runs afterwards it would wipe the
mappings and every audio packet would be dropped as "unknown SSRC" (observed
live: worked for a solo speaker whose op-5 arrived post-start, captured
nothing when two users' op-5 burst arrived during the join).

Discord voice audio arrives over UDP as RTP packets. Each packet carries an
``SSRC`` (synchronisation source id) in its header, but the SSRC alone does
not say *who* is speaking. The mapping ``SSRC -> user_id`` is published by the
voice WebSocket via **op-5 Speaking** events (and reinforced by op-12/13
client connect/disconnect). We therefore:

  1. Attach a hook to the voice WebSocket so we see every op-5/op-12/op-13
     event and maintain ``ssrc -> user_id``.
  2. Register a UDP socket listener on the connection state so we receive
     every raw audio packet.
  3. For each packet: parse the RTP header (SSRC + timestamp), decrypt the
     transport layer (XChaCha20-Poly1305 or AES-256-GCM, "rtpsize" layout),
     then — because Discord has mandated end-to-end encryption (DAVE) for all
     voice since 2026-03-01 — decrypt the E2EE layer via the ``davey`` session
     that discord.py already maintains. The result is a raw Opus frame.
  4. Decode the Opus frame to 48 kHz mono PCM and append it to that speaker's
     buffer, tagged with an absolute wall-clock timestamp.

On stop, each speaker's buffered frames are laid out on a shared timeline —
anchored at the first frame's arrival offset, then exactly one nominal 20 ms
frame apart (never by raw arrival time, which would smear jitter into silence
holes and overwrites) — and written to ``<speaker>.wav``; a
``manifest.json`` records guild/channel, start/stop times, per-speaker SSRC +
user id + display name, decode statistics, and jitter diagnostics.

Threading note: the UDP listener callback runs on discord.py's voice socket
reader thread (not the event loop), while the op-5 hook runs on the event
loop. The recorder methods used by both are synchronous and only touch
in-memory structures protected by a single lock, so they are safe to call from
either context.

Crash durability
-----------------
Each speaker's decoded frames are appended to an append-only binary log on
disk *as they decode* (``<name>_<user_id>.log``) rather than held in an
unbounded in-memory list. A crash therefore loses only the few frames still in
flight at the moment it happens — not hours of audio — and the RAM footprint
drops to "current frame + file handles" (the old buffer grew ~96 KB/s per
active speaker). On ``stop()`` the logs are read back and laid out on the same
nominal 20 ms grid as before, then deleted. A session that crashed without a
clean stop leaves a ``.recording`` marker + its logs; :func:`recover_orphans`
rebuilds those into WAVs + a manifest flagged ``"recovered": true`` (auto-run
at startup for orphans older than ~5 minutes, and available as a manual call).

This module is intentionally import-safe: importing it never connects to voice
and has no side effects until :func:`attach_to_bot` / :meth:`VoiceRecorder.start`
are called.
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import discord
import nacl.secret

log = logging.getLogger("bot.voice_recorder")

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
_RECOVERY_THRESHOLD_S: float = 5 * 60.0   # auto-recover orphans older than this
_MARKER_NAME = ".recording"               # session marker inside each recording dir

# RTP "rtpsize" transport layout constants.
_RTP_HEADER_SIZE = 12
_AUTH_TAG_LEN = 16            # Poly1305 / GCM tag
_NONCE_TRAILER_LEN = 4        # trailing 32-bit counter nonce


class VoiceRecorderError(Exception):
    """Raised for unrecoverable recorder configuration problems."""


# ── Per-speaker capture buffer ───────────────────────────────────────────────
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


# ── Per-speaker on-disk frame log (crash durability) ────────────────────────
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


# ── The recorder ─────────────────────────────────────────────────────────────
class VoiceRecorder:
    """Captures per-speaker voice audio for one recording session."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self._lock = threading.RLock()
        self._speakers: dict[int, _Speaker] = {}     # user_id -> _Speaker
        self._ssrc_to_user: dict[int, int] = {}      # ssrc -> user_id
        self._decoders: dict[int, Any] = {}          # user_id -> opus.Decoder
        self._recording = False
        self._started_at: float = 0.0
        self._ended_at: Optional[float] = None
        self._guild_id: Optional[int] = None
        self._channel_id: Optional[int] = None
        self._channel_name: str = ""
        self._encryption_mode: str = ""
        self._total_packets = 0
        self._unknown_ssrc_packets = 0
        self._seen_unknown_ssrcs: dict[int, bool] = {}  # ssrc -> True (diagnostics)
        # Diagnostics: per-SSRC arrival counts + hexdumps of first frames.
        self._ssrc_packet_counts: dict[int, int] = {}
        self._hexdumped_ssrcs: set[int] = set()
        # Race fix: voice packets can arrive before the op-5 mapping is recorded
        # (observed 1ms early in live testing). Buffer unmapped frames briefly
        # and replay them once _note_speaker() learns their SSRC.
        self._pending_packets: dict[int, list[tuple[bytes, str, bytes, Any]]] = {}
        self._pending_flushed = 0
        # Pipeline stage diagnostics (transport / dave / decode).
        self._stage_failures: dict[str, int] = {"transport": 0, "dave": 0, "decode": 0}
        # Passthrough-frame accounting (Discord sends ~5% of frames unencrypted
        # even when DAVE is active; davey rejects them with
        # UnencryptedWhenPassthroughDisabled).
        self._passthrough_recovered = 0
        self._passthrough_silence_fallbacks = 0
        # Log the winning DAVE framing variant only once per SSRC.
        self._framing_logged: set[int] = set()
        self._first_stage_error: dict[str, str] = {}
        self._dave_state: dict[str, Any] = {}
        # The VoiceClient this recorder is wired to (set by _wire_voice_client).
        # Used by start() to decide whether an existing SSRC map is still valid.
        self._wired_vc: Optional[Any] = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(
        self,
        *,
        guild_id: Optional[int],
        channel_id: int,
        channel_name: str = "",
        out_dir: Optional[Path] = None,
    ) -> None:
        """Begin a new capture session (resets all state).

        Must be called *before* the voice join so the initial op-11/op-5
        burst is not lost. ``out_dir`` points the recorder at the recording's
        directory immediately, so any audio that starts arriving during the
        handshake lands in the right place.
        """
        if out_dir is not None:
            self.out_dir = Path(out_dir)

        # Clear any stale per-speaker logs / marker left in this directory by a
        # previous session (defensive — normally stop()/discard() remove them).
        with self._lock:
            self._remove_session_artifacts()

        # If the bot is already connected to this exact channel (e.g. a new
        # recording after /stop_recording with leave_channel=false), no fresh
        # handshake happens and Discord will NOT re-send the initial op-11/op-5
        # burst — but the SSRCs from the existing connection are still valid.
        # Keep the map in that case; otherwise a new join/move is coming and
        # old mappings must go (SSRCs are per-connection).
        keep_map = False
        with self._lock:
            vc = self._wired_vc
            if vc is not None:
                ch = getattr(vc, "channel", None)
                try:
                    connected = bool(vc.is_connected())
                except Exception:  # pragma: no cover - defensive
                    connected = False
                keep_map = connected and ch is not None and getattr(ch, "id", None) == channel_id

        with self._lock:
            self._speakers.clear()
            if not keep_map:
                self._ssrc_to_user.clear()
                # Opus decoders are stateful per stream; they're only
                # reusable when the SSRCs (and thus the streams) survive.
                self._decoders.clear()
            self._recording = True
            self._started_at = time.time()
            self._ended_at = None
            self._guild_id = guild_id
            self._channel_id = channel_id
            self._channel_name = channel_name
            self._encryption_mode = ""
            self._total_packets = 0
            self._unknown_ssrc_packets = 0
            self._seen_unknown_ssrcs.clear()
            self._ssrc_packet_counts.clear()
            self._hexdumped_ssrcs.clear()
            self._pending_packets.clear()
            self._pending_flushed = 0
            self._stage_failures = {"transport": 0, "dave": 0, "decode": 0}
            self._passthrough_recovered = 0
            self._passthrough_silence_fallbacks = 0
            self._framing_logged.clear()
            self._first_stage_error = {}
            self._dave_state = {}

        # Write the session marker FIRST so a crash at any point after this is
        # recoverable (the per-speaker logs are created lazily on first frame).
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            marker = {
                "started_at": round(self._started_at, 3),
                "guild_id": guild_id,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "pid": os.getpid(),
            }
            (self.out_dir / _MARKER_NAME).write_text(
                json.dumps(marker, indent=2), encoding="utf-8")
        except OSError as e:  # pragma: no cover - disk errors
            log.error("Could not write recording marker in %s: %s", self.out_dir, e)

        log.info(
            "Voice recording started (guild=%s channel=%s#%s)",
            guild_id, channel_id, channel_name,
        )

    def discard(self) -> None:
        """Abort the current capture session without writing any files.

        Used when the voice join fails after :meth:`start` has already run —
        otherwise a half-started session (with its SSRC mappings) would stay
        armed and leak into the next recording.
        """
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            for sp in self._speakers.values():
                if sp.log is not None:
                    sp.log.close()
            self._remove_session_artifacts()
            self._speakers.clear()
            self._ssrc_to_user.clear()
            self._decoders.clear()
            self._pending_packets.clear()
        log.info("Voice recording discarded (join failed)")

    def stop(self) -> Optional[dict]:
        """Stop capturing and write WAV files + manifest. Returns the manifest.

        Reads each speaker's on-disk frame log back, lays the frames on the
        shared nominal 20 ms grid (identical to the old in-memory path), writes
        one WAV per speaker plus ``manifest.json``, then removes the session
        logs and marker so a later crash-recovery run won't re-process them.
        """
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            self._ended_at = time.time()

        # Read every speaker's frame log back from disk (the single source of
        # truth for what was captured) and close the file handles.
        with self._lock:
            for sp in self._speakers.values():
                if sp.log is not None:
                    sp.log.close()
            speakers_data = {
                uid: (
                    _SpeakerLog.read_frames(sp.log.path)
                    if sp.log is not None else []
                )
                for uid, sp in self._speakers.items()
            }

        # Build the timeline-aligned WAV per speaker.
        speakers_out: list[dict] = []
        max_end = self._started_at
        for uid, frames in speakers_data.items():
            if not frames:
                continue
            first_ts = frames[0][0]
            # Nominal duration (frames x 20 ms), not wall-clock span, so
            # network jitter doesn't inflate the file length.
            end_ts = first_ts + len(frames) * FRAME_SAMPLES / SAMPLE_RATE
            max_end = max(max_end, end_ts)

        total_duration = max(0.0, max_end - self._started_at)
        total_samples = int(total_duration * SAMPLE_RATE)

        with self._lock:
            for user_id, sp in sorted(self._speakers.items()):
                frames = speakers_data.get(user_id) or []
                if not frames:
                    continue
                wav_path = self.out_dir / _wav_filename(user_id, sp.display_name)
                gap_frames, overlap_frames = _write_timeline_wav_from_frames(
                    path=wav_path,
                    frames=frames,
                    origin=self._started_at,
                    total_samples=total_samples,
                )
                first_ts = frames[0][0]
                last_end_ts = first_ts + len(frames) * FRAME_SAMPLES / SAMPLE_RATE
                speakers_out.append({
                    "user_id": user_id,
                    "display_name": sp.display_name,
                    "ssrc": sp.ssrc,
                    "wav_file": wav_path.name,
                    "wav_path": str(wav_path),
                    "first_speech_offset_s": round(first_ts - self._started_at, 3),
                    "last_speech_offset_s": round(last_end_ts - self._started_at, 3),
                    "spoken_duration_s": round(sp.duration_s, 3),
                    "frames_captured": len(frames),
                    "decode_failures": sp.decode_failures,
                    "decrypt_failures": sp.decrypt_failures,
                    "jitter_gap_frames": gap_frames,
                    "jitter_overlap_frames": overlap_frames,
                })

        manifest = {
            "guild_id": self._guild_id,
            "channel_id": self._channel_id,
            "channel_name": self._channel_name,
            "started_at": round(self._started_at, 3),
            "ended_at": round(self._ended_at or time.time(), 3),
            "duration_s": round((self._ended_at or time.time()) - self._started_at, 3),
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "bit_depth": SAMPLE_WIDTH * 8,
            "encryption_mode": self._encryption_mode,
            "total_packets_seen": self._total_packets,
            "unknown_ssrc_packets_dropped": self._unknown_ssrc_packets,
            "unmapped_ssrcs_seen": sorted(self._seen_unknown_ssrcs),
            "ssrc_to_user_at_stop": {str(k): v for k, v in self._ssrc_to_user.items()},
            "ssrc_packet_counts": {str(k): v for k, v in sorted(self._ssrc_packet_counts.items())},
            "pending_packets_replayed": self._pending_flushed,
            "stage_failures": dict(self._stage_failures),
            "first_stage_error": dict(self._first_stage_error),
            "dave_state": self._dave_state,
            "passthrough_frames_recovered": self._passthrough_recovered,
            "passthrough_silence_fallbacks": self._passthrough_silence_fallbacks,
            "speakers": speakers_out,
        }

        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:  # pragma: no cover - disk errors
            log.error("Could not create recordings dir %s: %s", self.out_dir, e)

        manifest_path = self.out_dir / "manifest.json"
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            manifest["manifest_path"] = str(manifest_path)
        except OSError as e:  # pragma: no cover - disk errors
            log.error("Failed to write recording manifest: %s", e)

        # The WAVs + manifest are durable now — remove the intermediate session
        # logs and marker so crash-recovery never re-processes a finished run.
        with self._lock:
            self._remove_session_artifacts()

        log.info(
            "Voice recording stopped: %d speaker(s), %.1fs total -> %s",
            len(speakers_out), manifest["duration_s"], self.out_dir,
        )
        return manifest

    # ── voice-websocket hook (op-5 / op-12 / op-13) ─────────────────────────
    async def on_voice_ws(self, ws: Any, msg: dict) -> None:
        """Installed as the voice WebSocket hook. Maps SSRC -> user_id."""
        try:
            op = msg.get("op")
            data = msg.get("d") or {}
        except AttributeError:  # pragma: no cover - defensive
            return

        # Diagnostics: see exactly what the voice gateway sends us. Op-5/11/13
        # are logged at INFO so they're visible in bot.log without DEBUG.
        if op in (5, 11, 13):
            log.info("voice ws op-%s: %r", op, data)
        else:
            log.debug("voice ws op-%s: %r", op, msg)

        if op == 5:  # Speaking — carries {speaking, delay, ssrc} (+ user_id in v8)
            ssrc = _as_int(data.get("ssrc"))
            user_id = _as_int(data.get("user_id"))
            speaking = _as_int(data.get("speaking"))  # None if the field is absent
            if ssrc is None:
                log.warning("op-5 received without ssrc: %r", data)
                return
            # The voice-gateway op-5 *receive* payload includes the sender's
            # user_id (voice gateway v8). Map it, but ignore "stopped speaking"
            # frames (speaking == 0) for mapping purposes.
            if user_id is not None and speaking != 0:
                self._note_speaker(ssrc, user_id)
            elif ssrc not in self._ssrc_to_user:
                log.warning(
                    "op-5 without resolvable user_id (ssrc=%s) — keys=%r",
                    ssrc, sorted(data.keys()),
                )

        elif op == 11:  # ClientsConnect — {user_ids: [...]} (no SSRC yet)
            for uid in data.get("user_ids") or []:
                uid = _as_int(uid)
                if uid is not None:
                    self._ensure_speaker(uid)

        elif op == 13:  # ClientDisconnect — {user_id}
            user_id = _as_int(data.get("user_id"))
            if user_id is not None:
                log.debug("Client disconnected from voice: %s", user_id)

    def _note_speaker(self, ssrc: int, user_id: int) -> None:
        with self._lock:
            existing = self._ssrc_to_user.get(ssrc)
            if existing is not None and existing != user_id:
                log.warning("SSRC %s remapped from user %s to %s", ssrc, existing, user_id)
            self._ssrc_to_user[ssrc] = user_id
            sp = self._speakers.get(user_id)
            if sp is None:
                sp = _Speaker(user_id=user_id, ssrc=ssrc)
                self._speakers[user_id] = sp
            else:
                sp.ssrc = ssrc
            # Replay frames that arrived before this mapping existed.
            pending = self._pending_packets.pop(ssrc, None)
        if pending:
            log.info("Replaying %d buffered packet(s) for ssrc=%d (user=%s)", len(pending), ssrc, user_id)
            for raw, mode, key, dave in pending:
                self._process_mapped_packet(raw, ssrc, user_id, mode=mode, secret_key=key, dave_session=dave, resolve_name=lambda uid: "")
            with self._lock:
                self._pending_flushed += len(pending)

    def _process_mapped_packet(
        self,
        raw: bytes,
        ssrc: int,
        user_id: int,
        *,
        mode: str,
        secret_key: bytes,
        dave_session: Any,
        resolve_name: Callable[[int], str],
    ) -> None:
        """Steps 2-4 of the capture pipeline for a packet whose SSRC is mapped."""
        # Record the negotiated mode + DAVE session state (diagnostics).
        with self._lock:
            if not self._dave_state:
                try:
                    ready = bool(getattr(dave_session, "ready", False))
                except Exception:
                    ready = None
                self._dave_state = {
                    "mode": mode,
                    "dave_present": dave_session is not None,
                    "dave_ready": ready,
                }

        # 2) Transport-layer decryption -> E2EE-encrypted Opus frame.
        try:
            inner = _decrypt_transport(raw, mode, secret_key)
        except Exception as e:  # bad auth tag / unsupported mode
            self._bump_failure(user_id, "decrypt")
            self._record_stage_error("transport", f"{type(e).__name__}: {e}", raw[:24].hex())
            return

        # 3) E2EE (DAVE) layer -> raw Opus frame.
        #    The transport-decrypted payload may carry a short DAVE framing
        #    prefix that davey expects stripped. Pycord's proven live pipeline
        #    strips the first 8 bytes for the xchacha rtpsize mode before calling
        #    dave.decrypt. We try the full payload first, then the 8-byte-stripped
        #    variant (xchacha rtpsize only), so we work regardless of which framing
        #    Discord used this run; the winning variant + any real error text are
        #    recorded for diagnosis.
        # For the xchacha rtpsize mode pycord's proven pipeline always strips
        # the first 8 bytes before dave.decrypt, so try that variant first; keep
        # the full payload as a fallback for any other framing.
        candidates: list[tuple[str, bytes]] = []
        if mode == "aead_xchacha20_poly1305_rtpsize" and len(inner) > 8:
            candidates.append(("strip8", inner[8:]))
        candidates.append(("full", inner))

        opus_frame: Optional[bytes] = None
        winning = ""
        dave_errors: list[str] = []
        for label, cand in candidates:
            err_box: list[str] = []
            frame = _decrypt_dave(cand, user_id, dave_session, on_error=err_box.append)
            if frame is not None:
                opus_frame = frame
                winning = label
                break
            dave_errors.append(f"{label}: {err_box[0] if err_box else 'returned None'}")
        if opus_frame is None:
            # Passthrough frames: Discord sends ~5% of frames unencrypted even
            # when DAVE is active. Their layout is
            #   [raw_opus][dave_supp_block][rtp_padding]
            # where the supp block ends with <size_byte> 0xFA 0xFA and the RTP
            # padding length is the final byte (RFC 3550). Recover the Opus
            # directly; if that fails, substitute a DTX silence frame so the
            # stateful decoder stays in sync (pycord does the same).
            pt = _extract_passthrough_opus(inner) or _extract_passthrough_opus(candidates[0][1])
            if pt is not None:
                opus_frame = pt
                with self._lock:
                    self._passthrough_recovered += 1
            else:
                opus_frame = b"\xf8\xff\xfe"  # Opus DTX silence (20 ms)
                self._bump_failure(user_id, "decrypt")
                with self._lock:
                    self._passthrough_silence_fallbacks += 1
                self._record_stage_error("dave", "; ".join(dave_errors), inner[:24].hex())
        elif winning != "full" and ssrc not in self._framing_logged:
            with self._lock:
                self._framing_logged.add(ssrc)
            log.info("DAVE decrypt succeeded with %s framing (ssrc=%s)", winning, ssrc)

        # 4) Opus -> PCM.
        pcm = _decode_opus(opus_frame, self._decoder_for(user_id))
        if pcm is None:
            self._bump_failure(user_id, "decode")
            self._record_stage_error("decode", "opus decode returned None / raised", opus_frame[:24].hex())
            return

        now = time.time()
        with self._lock:
            sp = self._speakers.get(user_id)
            if sp is None:  # op-5 for this SSRC may not have arrived yet
                sp = _Speaker(user_id=user_id, ssrc=ssrc)
                self._speakers[user_id] = sp
            if not sp.display_name:
                try:
                    sp.display_name = resolve_name(user_id) or ""
                except Exception:
                    sp.display_name = ""
            # Durability: append the decoded frame to this speaker's on-disk log
            # (created lazily) instead of holding it in an unbounded RAM list.
            if sp.log is None:
                sp.log = _SpeakerLog(self._log_path_for(user_id, sp.display_name))
            sp.log.write_frame(now, pcm)
            sp.frame_count += 1
            sp.total_pcm_bytes += len(pcm)
            self._total_packets += 1

    # ── session-artifact helpers (crash durability) ────────────────────────
    def _log_path_for(self, user_id: int, display_name: str) -> Path:
        """On-disk frame-log path for a speaker in the current recording dir."""
        return self.out_dir / (_wav_filename(user_id, display_name) + ".log")

    def _remove_session_artifacts(self) -> None:
        """Delete this session's per-speaker ``*.log`` files and ``.recording``
        marker from ``out_dir``. Must be called with the lock held."""
        try:
            if not self.out_dir.exists():
                return
            for p in self.out_dir.iterdir():
                if p.name == _MARKER_NAME or p.suffix == ".log":
                    try:
                        p.unlink()
                    except OSError:  # pragma: no cover - benign race
                        pass
        except OSError:  # pragma: no cover - disk errors
            pass

    def install_sigterm_flush(self) -> None:
        """Install a SIGTERM handler that flushes an in-flight recording.

        Docker sends SIGTERM on ``docker stop`` / compose restarts. The default
        action would kill the process and leave the session logs + marker on
        disk (recoverable, but truncated). Instead we run :meth:`stop` so a
        *clean* shutdown always writes complete WAVs + manifest. Hard crashes
        (OOM/segfault/power) can't be caught — that's what crash recovery is
        for. Only installed in the main thread; otherwise it's a no-op.
        """
        if threading.current_thread() is not threading.main_thread():
            log.warning("SIGTERM flush not installed (not on main thread)")
            return
        try:
            signal.signal(signal.SIGTERM, self._sigterm_handler)
        except (ValueError, OSError) as e:  # pragma: no cover - exotic platforms
            log.warning("Could not install SIGTERM handler: %s", e)

    def _sigterm_handler(self, signum: int, frame: Any) -> None:
        log.info("SIGTERM received — flushing in-flight voice recording before exit")
        try:
            if self.is_recording:
                self.stop()
        except Exception:  # pragma: no cover - defensive
            log.exception("Error flushing recording on SIGTERM (logs remain recoverable)")
        os._exit(0)

    def _record_stage_error(self, stage: str, detail: str, hexdump: str) -> None:
        """Count a pipeline-stage failure and log the first one loudly."""
        with self._lock:
            self._stage_failures[stage] = self._stage_failures.get(stage, 0) + 1
            if stage not in self._first_stage_error:
                self._first_stage_error[stage] = detail
                log.warning(
                    "PIPELINE %s FAILED (first): %s | input_head=%s",
                    stage, detail, hexdump,
                )

    def _ensure_speaker(self, user_id: int) -> None:
        with self._lock:
            if user_id not in self._speakers:
                self._speakers[user_id] = _Speaker(user_id=user_id)

    # ── UDP packet path (runs on the socket-reader thread) ───────────────────
    def handle_packet(
        self,
        raw: bytes,
        *,
        mode: str,
        secret_key: bytes,
        dave_session: Any,
        resolve_name: Callable[[int], str],
    ) -> None:
        """Decrypt + decode one raw UDP voice packet and buffer it per speaker."""
        if not self._recording or len(raw) <= _RTP_HEADER_SIZE + 8:
            return

        # 1) Parse the unencrypted RTP header.
        ssrc = struct.unpack_from(">I", raw, 8)[0]

        with self._lock:
            self._ssrc_packet_counts[ssrc] = self._ssrc_packet_counts.get(ssrc, 0) + 1
            # Hexdump the first two frames per SSRC so we can see exactly what
            # is on the wire (RTP? IP-discovery? something else?).
            if ssrc not in self._hexdumped_ssrcs and len(self._hexdumped_ssrcs) < 64:
                self._hexdumped_ssrcs.add(ssrc)
                log.warning(
                    "UDP frame first-seen (ssrc=%d len=%d first_byte=0x%02x): %s",
                    ssrc, len(raw), raw[0], raw[:64].hex(),
                )
            user_id = self._ssrc_to_user.get(ssrc)
            if user_id is None:
                self._unknown_ssrc_packets += 1
                # Track which SSRCs are actually arriving (diagnostics).
                if ssrc not in self._seen_unknown_ssrcs:
                    log.warning(
                        "Unknown SSRC %d arriving on UDP (first packet) — "
                        "no op-5 mapping yet; ssrc_map=%r",
                        ssrc, dict(self._ssrc_to_user),
                    )
                    self._seen_unknown_ssrcs[ssrc] = True
                # Buffer briefly: the op-5 mapping may arrive a few ms later.
                pending = self._pending_packets.setdefault(ssrc, [])
                if len(pending) < 20:
                    pending.append((raw, mode, secret_key, dave_session))
                return

        # Steps 2-4 (decrypt -> decode -> buffer).
        self._process_mapped_packet(
            raw, ssrc, user_id,
            mode=mode, secret_key=secret_key, dave_session=dave_session,
            resolve_name=resolve_name,
        )

    # Per-speaker stateful Opus decoders (one per speaker, keyed by user id).
    def _decoder_for(self, user_id: int) -> Any:
        from discord import opus
        with self._lock:
            d = self._decoders.get(user_id)
            if d is None:
                d = opus.Decoder()
                self._decoders[user_id] = d
            return d

    def _bump_failure(self, user_id: int, kind: str) -> None:
        with self._lock:
            sp = self._speakers.get(user_id)
            if sp is None:
                return
            if kind == "decrypt":
                sp.decrypt_failures += 1
            else:
                sp.decode_failures += 1

    # ── diagnostics ──────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "recording": self._recording,
                "speakers": len(self._speakers),
                "ssrc_map_size": len(self._ssrc_to_user),
                "total_packets": self._total_packets,
                "unknown_ssrc_dropped": self._unknown_ssrc_packets,
            }


# ── Bot wiring (idempotent) ──────────────────────────────────────────────────
_attached = False


def attach_to_bot(bot: discord.Client, out_dir: Path) -> VoiceRecorder:
    """Wire the recorder into ``bot`` so every voice connection is captured.

    This is idempotent: calling it again reuses the same singleton and simply
    re-points it at the (possibly new) output directory.

    discord.py 2.x has no per-connection hook on the client, so joining must go
    through :func:`recorder_voice_cls`: ``channel.connect(cls=recorder_voice_cls(bot))``
    instantiates a :class:`discord.VoiceClient` subclass whose ``__init__``
    installs (a) a voice-WebSocket hook for op-5 SSRC mapping and (b) a UDP
    socket listener for audio packets — both before the client connects.
    """
    global _attached
    recorder = getattr(bot, "_voice_recorder", None)
    if recorder is None:
        recorder = VoiceRecorder(out_dir)
        bot._voice_recorder = recorder  # type: ignore[attr-defined]

    recorder.out_dir = Path(out_dir)

    if not _attached:
        _attached = True
        # Clean shutdowns (docker stop / compose restart) send SIGTERM — flush
        # any in-flight recording so we don't leave an orphan on disk.
        recorder.install_sigterm_flush()
        log.info("Voice recorder attached to bot (out_dir=%s)", out_dir)

    return recorder


def recorder_voice_cls(bot: discord.Client) -> type[discord.VoiceClient]:
    """Build a :class:`discord.VoiceClient` subclass wired for recording.

    Pass it as ``cls=`` to :meth:`discord.VoiceChannel.connect`. The returned
    class wires itself up in ``__init__`` (i.e. before any handshake), so the
    op-5 hook and UDP listener are live from the very first packet.
    """
    recorder = getattr(bot, "_voice_recorder", None)
    if recorder is None:
        raise VoiceRecorderError("attach_to_bot() must be called before joining voice")

    class RecordingVoiceClient(discord.VoiceClient):
        def __init__(self, client: discord.Client, channel: Any) -> None:
            super().__init__(client, channel)
            _wire_voice_client(self, recorder)

    return RecordingVoiceClient


def _wire_voice_client(vc: discord.VoiceClient, recorder: VoiceRecorder) -> None:
    """Install the WS hook + UDP listener on a voice client.

    Called from :class:`RecordingVoiceClient.__init__` (before connect), or as
    a best-effort retrofit when the bot is already in voice for another reason.
    """
    state = vc._connection  # VoiceConnectionState

    with recorder._lock:
        recorder._wired_vc = vc

    # (a) op-5 / op-12 / op-13 hook — read by VoiceConnectionState whenever it
    #     (re)creates the voice WebSocket, so setting it here is safe.
    state.hook = recorder.on_voice_ws

    # (b) raw UDP audio packets.
    def _on_udp(raw: bytes) -> None:
        try:
            mode = state.mode
            key = state.secret_key
            if not isinstance(mode, str) or not isinstance(key, list):
                return  # handshake not complete yet
            recorder.handle_packet(
                raw,
                mode=mode,
                secret_key=bytes(key),
                dave_session=state.dave_session,
                resolve_name=_make_name_resolver(vc),
            )
        except Exception as e:  # never let a bad packet kill the reader thread
            log.debug("voice recorder packet error: %s", e)

    state.add_socket_listener(_on_udp)
    try:
        vc._recorder_wired = True  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - some fakes don't allow attrs
        pass


def _make_name_resolver(vc: discord.VoiceClient) -> Callable[[int], str]:
    """Return a function mapping user_id -> display name (best effort)."""

    def resolve(user_id: int) -> str:
        try:
            guild = vc.guild
            member = guild.get_member(user_id) if guild is not None else None
            if member is not None:
                return member.display_name or ""
        except Exception:
            pass
        return ""

    return resolve


# ── Low-level helpers (pure, unit-testable) ──────────────────────────────────
def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _decrypt_transport(raw: bytes, mode: str, secret_key: bytes) -> bytes:
    """Decrypt the RTP "rtpsize" transport layer, returning the inner payload.

    Handles the two modes Discord currently negotiates with discord.py:
    ``aead_xchacha20_poly1305_rtpsize`` (always available) and
    ``aead_aes256_gcm_rtpsize`` (preferred when the server offers it).
    """
    if len(raw) < _RTP_HEADER_SIZE + _AUTH_TAG_LEN + _NONCE_TRAILER_LEN:
        raise ValueError("packet too short for rtpsize layout")

    first = raw[0]
    csrc_count = first & 0x0F
    xbit = (first >> 4) & 0x01
    header_size = _RTP_HEADER_SIZE + 4 * csrc_count + (4 if xbit else 0)

    header = raw[:header_size]
    # Layout: [header][ciphertext][auth_tag(16)][nonce_counter(4)]
    nonce_counter = raw[-_NONCE_TRAILER_LEN:]
    ct_end = len(raw) - _AUTH_TAG_LEN - _NONCE_TRAILER_LEN
    ciphertext = raw[header_size:ct_end]
    auth_tag = raw[ct_end:len(raw) - _NONCE_TRAILER_LEN]

    if mode == "aead_xchacha20_poly1305_rtpsize":
        box = nacl.secret.Aead(bytes(secret_key))
        nonce = nonce_counter + b"\x00" * 20  # 24-byte XChaCha20 nonce
        return box.decrypt(ciphertext + auth_tag, header, nonce)

    if mode == "aead_aes256_gcm_rtpsize":
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aes = AESGCM(bytes(secret_key))
        # NOTE: positional-only args in the order (nonce, data, associated_data).
        nonce = nonce_counter + b"\x00" * 12  # 16-byte GCM nonce
        return aes.decrypt(nonce, ciphertext + auth_tag, header)

    raise ValueError(f"unsupported transport encryption mode: {mode!r}")


def _extract_passthrough_opus(inner: bytes) -> Optional[bytes]:
    """Recover the raw Opus frame from a DAVE *passthrough* (unencrypted) frame.

    Passthrough layout (reverse-engineered, matches pycord's live traffic):
        [raw_opus][dave_supp_block][rtp_padding]
    where the supp block ends with ``<size_byte> 0xFA 0xFA`` (size counts the
    whole block) and the RTP padding length is the final byte (RFC 3550).
    Returns ``None`` when the layout doesn't parse cleanly.
    """
    data = inner
    # Strip RFC 3550 RTP padding: last byte = number of pad bytes.
    if len(data) >= 4 and 0 < data[-1] <= 64 and len(data) > data[-1] + 4:
        candidate = data[: -data[-1]]
        # Only trust it if the FAFA marker lands inside the trimmed payload.
        if b"\xfa\xfa" in candidate:
            data = candidate
    idx = data.rfind(b"\xfa\xfa")
    if idx >= 2 and idx + 2 <= len(data):
        supp_size = data[idx - 1]
        start = idx + 2 - supp_size
        # Plausible Opus frame: at least a couple of bytes, sane total size.
        if 3 <= supp_size <= 64 and 2 <= start <= 1500:
            return data[:start]
    return None


def _decrypt_dave(
    inner: bytes,
    user_id: int,
    dave_session: Any,
    on_error: Optional[Callable[[str], None]] = None,
) -> Optional[bytes]:
    """Strip the DAVE E2EE layer from a transport-decrypted Opus frame.

    Returns the raw Opus frame, or ``None`` when decryption fails. When no
    DAVE session exists (call downgraded to plaintext / pre-E2EE) the frame is
    already plaintext and is returned unchanged. If ``on_error`` is supplied it
    is called with the real exception text on failure so callers can surface it.
    """
    if dave_session is None:
        return inner
    try:
        import davey  # local import keeps module import-light
        return dave_session.decrypt(user_id, davey.MediaType.audio, inner)
    except Exception as e:  # pragma: no cover - depends on live MLS state
        msg = f"{type(e).__name__}: {e}"
        if on_error is not None:
            try:
                on_error(msg)
            except Exception:
                pass
        log.debug("DAVE decrypt raised (user=%s): %s", user_id, e)
        return None


def _decode_opus(opus_frame: bytes, decoder: Any) -> Optional[bytes]:
    """Decode one Opus frame to 48 kHz *mono* 16-bit PCM (or None on failure).

    Discord voice is encoded as stereo, and discord.py's ``opus.Decoder`` always
    returns interleaved stereo 48 kHz PCM. We downmix to mono here so each
    speaker's WAV stays a single channel (ideal for STT) and every downstream
    byte/duration calculation assumes one channel.

    ``decoder`` must be the per-speaker decoder: Opus decoders are stateful, so
    each speaker needs their own instance fed in frame order.
    """
    try:
        pcm_stereo = decoder.decode(opus_frame)
        if not pcm_stereo:
            return None
        return _downmix_to_mono(pcm_stereo)
    except Exception as e:  # pragma: no cover - codec edge cases
        log.debug("Opus decode failed: %s", e)
        return None


def _downmix_to_mono(stereo_pcm: bytes) -> bytes:
    """Average interleaved stereo 16-bit PCM down to mono.

    ``stereo_pcm`` is interleaved L,R,L,R,... with each sample 2 bytes. For N
    frames there are 4*N bytes total; the result has N mono samples (2*N bytes).
    """
    import array
    n_frames = len(stereo_pcm) // (SAMPLE_WIDTH * 2)  # one frame == L + R
    lr = array.array("h")
    lr.frombytes(stereo_pcm[: n_frames * 2 * SAMPLE_WIDTH])
    out = bytearray(n_frames * SAMPLE_WIDTH)
    for i in range(n_frames):
        l = lr[i * 2]
        r = lr[i * 2 + 1]
        out[i * 2:i * 2 + 2] = struct.pack("<h", (l + r) // 2)
    return bytes(out)


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


# ── Crash recovery (rebuild a recording whose process died mid-capture) ─────
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
    This happens when the bot is OOM-killed, segfaults, or loses power; a clean
    SIGTERM shutdown flushes via the signal handler instead and leaves no orphan.

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
