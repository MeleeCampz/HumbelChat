"""The VoiceRecorder state machine + bot wiring.

Extracted from ``bot_core.voice_recorder`` (P3 #35). :class:`VoiceRecorder`
is the per-session state machine (start/stop, SSRC->user mapping, decode
path, SIGTERM flush); ``attach_to_bot`` / ``recorder_voice_cls`` wire it
into the discord.py client idempotently.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import discord

from bot_core.voice.capture import (
    CHANNELS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    VoiceRecorderError,
    _MARKER_NAME,
    _Speaker,
    _SpeakerLog,
    _wav_filename,
    _write_timeline_wav_from_frames,
)
from bot_core.voice.dave import (
    _RTP_HEADER_SIZE,
    _as_int,
    _decode_opus,
    _decrypt_dave,
    _decrypt_transport,
    _extract_passthrough_opus,
)

log = logging.getLogger("bot.voice_recorder")

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
