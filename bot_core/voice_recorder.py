"""Voice-channel audio capture for later STT transcription.

This module lets the bot join a voice channel, capture every participant's
audio *separately* (so each speaker can be attributed and transcribed
independently), and write the result to disk as one 16-bit PCM WAV file per
speaker plus a JSON manifest carrying absolute timestamps so a conversation
can be reconstructed on a shared timeline.

How it works
------------
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

On stop, each speaker's buffered frames are laid out on a shared timeline
(silence in the gaps) and written to ``<speaker>.wav``; a ``manifest.json``
records guild/channel, start/stop times, per-speaker SSRC + user id + display
name, and decode statistics.

Threading note: the UDP listener callback runs on discord.py's voice socket
reader thread (not the event loop), while the op-5 hook runs on the event
loop. The recorder methods used by both are synchronous and only touch
in-memory structures protected by a single lock, so they are safe to call from
either context.

This module is intentionally import-safe: importing it never connects to voice
and has no side effects until :func:`attach_to_bot` / :meth:`VoiceRecorder.start`
are called.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
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
FRAME_MS = 20                 # one Opus frame == 20 ms at 48 kHz
BYTES_PER_FRAME = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * FRAME_MS // 1000

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
    # list of (absolute_epoch_seconds, pcm_bytes) in arrival order
    frames: list[tuple[float, bytes]] = field(default_factory=list)
    total_pcm_bytes: int = 0
    decode_failures: int = 0
    decrypt_failures: int = 0

    @property
    def duration_s(self) -> float:
        return self.total_pcm_bytes / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)


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
    ) -> None:
        """Begin a new capture session (resets all state)."""
        with self._lock:
            self._speakers.clear()
            self._ssrc_to_user.clear()
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
        log.info(
            "Voice recording started (guild=%s channel=%s#%s)",
            guild_id, channel_id, channel_name,
        )

    def stop(self) -> Optional[dict]:
        """Stop capturing and write WAV files + manifest. Returns the manifest."""
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            self._ended_at = time.time()

        # Build the timeline-aligned WAV per speaker.
        speakers_out: list[dict] = []
        max_end = self._started_at
        with self._lock:
            for sp in self._speakers.values():
                if not sp.frames:
                    continue
                first_ts, _ = sp.frames[0]
                last_ts, last_pcm = sp.frames[-1]
                end_ts = last_ts + len(last_pcm) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
                max_end = max(max_end, end_ts)

        total_duration = max(0.0, max_end - self._started_at)
        total_samples = int(total_duration * SAMPLE_RATE)

        with self._lock:
            for user_id, sp in sorted(self._speakers.items()):
                if not sp.frames:
                    continue
                wav_path = self.out_dir / _wav_filename(user_id, sp.display_name)
                _write_timeline_wav(
                    path=wav_path,
                    frames=sp.frames,
                    origin=self._started_at,
                    total_samples=total_samples,
                )
                first_ts, _ = sp.frames[0]
                last_ts, last_pcm = sp.frames[-1]
                speakers_out.append({
                    "user_id": user_id,
                    "display_name": sp.display_name,
                    "ssrc": sp.ssrc,
                    "wav_file": wav_path.name,
                    "wav_path": str(wav_path),
                    "first_speech_offset_s": round(first_ts - self._started_at, 3),
                    "last_speech_offset_s": round(
                        (last_ts + len(last_pcm) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH))
                        - self._started_at, 3
                    ),
                    "spoken_duration_s": round(sp.duration_s, 3),
                    "frames_captured": len(sp.frames),
                    "decode_failures": sp.decode_failures,
                    "decrypt_failures": sp.decrypt_failures,
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
                return
        if user_id is None:
            return

        # 2) Transport-layer decryption -> E2EE-encrypted Opus frame.
        try:
            inner = _decrypt_transport(raw, mode, secret_key)
        except Exception as e:  # bad auth tag / unsupported mode
            self._bump_failure(user_id, "decrypt")
            log.debug("Transport decrypt failed (ssrc=%s user=%s): %s", ssrc, user_id, e)
            return

        # 3) E2EE (DAVE) layer -> raw Opus frame. Since 2026-03-01 all Discord
        #    voice is end-to-end encrypted; the davey session that discord.py
        #    maintains knows each sender's ratcheted key. During a downgrade or
        #    transition the call may be in *passthrough* (plaintext), which
        #    ``DaveSession.decrypt`` handles by returning the frame unchanged.
        opus_frame = _decrypt_dave(inner, user_id, dave_session)
        if opus_frame is None:
            self._bump_failure(user_id, "decrypt")
            log.debug("DAVE decrypt failed (user=%s)", user_id)
            return

        # 4) Opus -> PCM. Each speaker gets their own stateful decoder so
        #    interleaved multi-speaker audio decodes correctly.
        pcm = _decode_opus(opus_frame, self._decoder_for(user_id))
        if pcm is None:
            self._bump_failure(user_id, "decode")
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
            sp.frames.append((now, pcm))
            sp.total_pcm_bytes += len(pcm)
            self._total_packets += 1

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


def _decrypt_dave(inner: bytes, user_id: int, dave_session: Any) -> Optional[bytes]:
    """Strip the DAVE E2EE layer from a transport-decrypted Opus frame.

    Returns the raw Opus frame, or ``None`` when decryption fails. When no
    DAVE session exists (call downgraded to plaintext / pre-E2EE) the frame is
    already plaintext and is returned unchanged.
    """
    if dave_session is None:
        return inner
    try:
        import davey  # local import keeps module import-light
        return dave_session.decrypt(user_id, davey.MediaType.audio, inner)
    except Exception as e:  # pragma: no cover - depends on live MLS state
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
) -> None:
    """Write a speaker's frames onto a shared timeline (silence in the gaps).

    Each frame is placed at ``frame_ts - origin`` seconds so that all speakers'
    WAVs share the same time base and can be interleaved for reconstruction.
    """
    samples = bytearray(total_samples * SAMPLE_WIDTH)  # zero-filled silence
    for ts, pcm in frames:
        offset_s = max(0.0, ts - origin)
        sample_idx = int(offset_s * SAMPLE_RATE)
        byte_idx = sample_idx * SAMPLE_WIDTH
        end_byte = min(len(samples), byte_idx + len(pcm))
        if byte_idx < len(samples):
            samples[byte_idx:end_byte] = pcm[:end_byte - byte_idx]

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(samples))
