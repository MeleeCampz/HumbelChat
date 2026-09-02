"""Tests for the voice-recording core (bot_core.voice_recorder).

Covers the pure, unit-testable parts of the pipeline:
  * transport-layer decryption (XChaCha20-Poly1305 + AES-256-GCM, "rtpsize"),
  * stereo -> mono downmix,
  * Opus frame decoding to 48 kHz mono PCM,
  * SSRC -> user mapping from voice-WebSocket op-5 / op-11 / op-13 events,
  * the full per-speaker capture path (packet in -> buffered PCM out), and
  * stop() writing one WAV per speaker plus a manifest.json.

No network or live Discord connection is required: packets are synthesised with
the same crypto primitives Discord uses.
"""
from __future__ import annotations

import json
import struct
import time
import wave
from pathlib import Path

import nacl.secret
import pytest

import discord

from bot_core.voice_recorder import (
    SAMPLE_RATE,
    VoiceRecorder,
    VoiceRecorderError,
    _SpeakerLog,
    recover_orphans,
    _decrypt_dave,
    _decrypt_transport,
    _decode_opus,
    _downmix_to_mono,
    _extract_passthrough_opus,
    _write_timeline_wav,
    _wav_filename,
    FRAME_SAMPLES,
    _wire_voice_client,
    attach_to_bot,
    recorder_voice_cls,
)

# A valid Opus "comfort noise" / silence packet — always decodes cleanly.
SILENCE_PACKET = b"\xf8\xff\xfe"


# ── helpers to build Discord-style voice packets ─────────────────────────────
def _rtp_header(ssrc: int, seq: int = 1, ts: int = 0) -> bytes:
    return bytes([0x80, 0x78]) + struct.pack(">H", seq) + struct.pack(">I", ts) + struct.pack(">I", ssrc)


def _make_xchacha_packet(inner: bytes, key: bytes, ssrc: int, counter: int = 1) -> bytes:
    header = _rtp_header(ssrc)
    box = nacl.secret.Aead(key)
    nonce_send = struct.pack(">I", counter) + b"\x00" * 20
    ct = box.encrypt(inner, header, nonce_send).ciphertext  # includes 16-byte tag
    return header + ct + nonce_send[:4]


def _make_aesgcm_packet(inner: bytes, key: bytes, ssrc: int, counter: int = 1) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    header = _rtp_header(ssrc)
    aes = AESGCM(key)
    nonce_send = struct.pack(">I", counter) + b"\x00" * 12
    ct = aes.encrypt(nonce_send, inner, header)  # positional: (nonce, data, associated_data)
    return header + ct + nonce_send[:4]


# ── transport decryption ─────────────────────────────────────────────────────
class TestDecryptTransport:
    def test_xchacha20_rtpsize_roundtrip(self):
        key = nacl.utils.random(32)  # 32-byte XChaCha20 key
        pkt = _make_xchacha_packet(SILENCE_PACKET, key, ssrc=0x1234)
        out = _decrypt_transport(pkt, "aead_xchacha20_poly1305_rtpsize", key)
        assert out == SILENCE_PACKET

    def test_aes256_gcm_rtpsize_roundtrip(self):
        key = nacl.utils.random(32)
        pkt = _make_aesgcm_packet(SILENCE_PACKET, key, ssrc=0x99)
        out = _decrypt_transport(pkt, "aead_aes256_gcm_rtpsize", key)
        assert out == SILENCE_PACKET

    def test_bad_auth_tag_raises(self):
        key = nacl.utils.random(32)
        pkt = bytearray(_make_xchacha_packet(SILENCE_PACKET, key, ssrc=1))
        pkt[-1] ^= 0xFF  # corrupt the nonce trailer / tag region
        with pytest.raises(Exception):
            _decrypt_transport(bytes(pkt), "aead_xchacha20_poly1305_rtpsize", key)

    def test_unsupported_mode_raises(self):
        key = nacl.utils.random(32)
        pkt = _make_xchacha_packet(SILENCE_PACKET, key, ssrc=1)
        with pytest.raises(ValueError):
            _decrypt_transport(pkt, "xsalsa20_poly1305_lite", key)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            _decrypt_transport(b"\x80\x78", "aead_xchacha20_poly1305_rtpsize", b"\x00" * 32)


# ── downmix + decode ─────────────────────────────────────────────────────────
class TestDecode:
    def test_downmix_averages_channels(self):
        assert list(struct.unpack("<h", _downmix_to_mono(struct.pack("<hh", 1, 2)))) == [1]
        assert list(struct.unpack("<h", _downmix_to_mono(struct.pack("<hh", 3, 5)))) == [4]

    def test_downmix_handles_negatives(self):
        assert list(struct.unpack("<h", _downmix_to_mono(struct.pack("<hh", -7, -3)))) == [-5]

    def test_downmix_multiple_frames(self):
        stereo = struct.pack("<hhh hhh".replace(" ", ""), 1, 2, 3, 4, 5, 6)
        mono = _downmix_to_mono(stereo)
        assert list(struct.unpack("<hhh", mono)) == [1, 3, 5]

    def test_decode_silence_packet_to_mono(self):
        from discord import opus
        decoder = opus.Decoder()
        pcm = _decode_opus(SILENCE_PACKET, decoder)
        assert pcm is not None
        # 20 ms of 48 kHz mono 16-bit == 960 samples * 2 bytes == 1920
        assert len(pcm) == 960 * 2

    def test_decode_garbage_packet_is_tolerated(self):
        from discord import opus
        decoder = opus.Decoder()
        # libopus decodes an unrecognised packet as silence (PLC) rather than
        # raising, so the helper must return PCM bytes (not raise) for garbage.
        out = _decode_opus(b"\x00\x01", decoder)
        assert out is None or isinstance(out, (bytes, bytearray))


# ── DAVE layer (passthrough when no session) ─────────────────────────────────
class TestDecryptDave:
    def test_none_session_passthrough(self):
        assert _decrypt_dave(SILENCE_PACKET, 42, None) == SILENCE_PACKET

    def test_bad_session_returns_none(self):
        class _Boom:
            def decrypt(self, *a, **k):
                raise RuntimeError("no MLS state")
        assert _decrypt_dave(SILENCE_PACKET, 42, _Boom()) is None


# ── DAVE passthrough-frame recovery ───────────────────────────────────────────
def _passthrough_frame(opus: bytes, payload_len: int = 9, with_pad: bool = True) -> bytes:
    """Build a Discord passthrough frame: [opus][supp_block][rtp_padding].

    The supp block is ``[payload (s-3 bytes)][size=s][0xFA][0xFA]`` where the
    size byte counts the whole block; RTP padding is RFC 3550 (pad bytes then a
    trailing length byte).
    """
    s = payload_len + 3
    block = bytes(range(1, payload_len + 1)) + bytes([s, 0xFA, 0xFA])
    assert len(block) == s
    frame = opus + block
    if with_pad:
        n = 4
        frame += b"\x00" * n + bytes([n])
    return frame


class TestPassthroughExtraction:
    def test_recovers_opus_with_rtp_padding(self):
        assert _extract_passthrough_opus(_passthrough_frame(SILENCE_PACKET)) == SILENCE_PACKET

    def test_recovers_opus_without_rtp_padding(self):
        f = _passthrough_frame(SILENCE_PACKET, with_pad=False)
        assert _extract_passthrough_opus(f) == SILENCE_PACKET

    def test_recovers_larger_opus_payload(self):
        opus = bytes([0x78, 1, 2, 3, 4, 5, 6, 7, 8])
        assert _extract_passthrough_opus(_passthrough_frame(opus)) == opus

    def test_garbage_returns_none(self):
        assert _extract_passthrough_opus(b"\x01\x02\x03") is None


# ── SSRC -> user mapping from voice WS events ────────────────────────────────
class TestVoiceWsMapping:
    def _recorder(self) -> VoiceRecorder:
        r = VoiceRecorder(Path("/tmp/unused"))
        r.start(guild_id=1, channel_id=2)
        return r

    @pytest.mark.asyncio
    async def test_op5_maps_ssrc_to_user(self):
        r = self._recorder()
        await r.on_voice_ws(None, {"op": 5, "d": {"speaking": 1, "ssrc": 0xAAAA, "user_id": 777}})
        assert r._ssrc_to_user[0xAAAA] == 777
        assert 777 in r._speakers

    @pytest.mark.asyncio
    async def test_op5_stop_speaking_does_not_map(self):
        r = self._recorder()
        await r.on_voice_ws(None, {"op": 5, "d": {"speaking": 0, "ssrc": 0xBBBB, "user_id": 888}})
        assert 0xBBBB not in r._ssrc_to_user

    @pytest.mark.asyncio
    async def test_op11_clients_connect_ensures_speakers(self):
        r = self._recorder()
        await r.on_voice_ws(None, {"op": 11, "d": {"user_ids": [101, 202]}})
        assert 101 in r._speakers and 202 in r._speakers

    @pytest.mark.asyncio
    async def test_op13_is_harmless(self):
        r = self._recorder()
        await r.on_voice_ws(None, {"op": 13, "d": {"user_id": 555}})  # must not raise


# ── full per-speaker capture path ────────────────────────────────────────────
class TestCapturePath:
    def _recorder(self) -> VoiceRecorder:
        r = VoiceRecorder(Path("/tmp/unused"))
        r.start(guild_id=1, channel_id=2, channel_name="General")
        return r

    def test_packet_buffered_per_speaker(self):
        r = self._recorder()
        key = nacl.utils.random(32)
        ssrc = 0x1111
        # Map the SSRC first (as op-5 would), then feed two packets.
        r._note_speaker(ssrc, 4242)
        for i in range(2):
            pkt = _make_xchacha_packet(SILENCE_PACKET, key, ssrc=ssrc, counter=i + 1)
            r.handle_packet(pkt, mode="aead_xchacha20_poly1305_rtpsize",
                            secret_key=key, dave_session=None, resolve_name=lambda uid: "Alice")
        sp = r._speakers[4242]
        assert sp.frame_count == 2
        assert sp.display_name == "Alice"
        assert sp.total_pcm_bytes == 2 * 1920
        # Frames are durably on disk (crash-safe), not in an in-RAM list.
        assert sp.log is not None and sp.log.path.exists()
        assert len(_SpeakerLog.read_frames(sp.log.path)) == 2

    def test_unknown_ssrc_counted_and_dropped(self):
        r = self._recorder()
        key = nacl.utils.random(32)
        pkt = _make_xchacha_packet(SILENCE_PACKET, key, ssrc=0xFFFF)  # unmapped
        r.handle_packet(pkt, mode="aead_xchacha20_poly1305_rtpsize",
                        secret_key=key, dave_session=None, resolve_name=lambda uid: "")
        assert r._unknown_ssrc_packets == 1
        assert not r._speakers

    def test_two_speakers_get_distinct_decoders(self):
        r = self._recorder()
        key = nacl.utils.random(32)
        r._note_speaker(0x1, 1)
        r._note_speaker(0x2, 2)
        d1 = r._decoder_for(1)
        d2 = r._decoder_for(2)
        assert d1 is not d2
        # Both still decode correctly through their own decoder.
        for ssrc, uid in ((0x1, 1), (0x2, 2)):
            pkt = _make_xchacha_packet(SILENCE_PACKET, key, ssrc=ssrc)
            r.handle_packet(pkt, mode="aead_xchacha20_poly1305_rtpsize",
                            secret_key=key, dave_session=None, resolve_name=lambda u: "x")
        assert r._speakers[1].frame_count == 1
        assert r._speakers[2].frame_count == 1


# ── start() state reset & the pre-join op-5 burst ordering ───────────────────
class _FakeVC:
    """Just enough of VoiceClient for start()'s keep-map check."""

    def __init__(self, channel_id=None, connected: bool = True) -> None:
        from types import SimpleNamespace
        self.channel = SimpleNamespace(id=channel_id) if channel_id is not None else None
        self._connected = connected

    def is_connected(self) -> bool:
        return self._connected


class TestStartOrdering:
    """Regression tests for the two-person silent-capture bug.

    Discord sends the initial op-11/op-5 burst (the only place SSRC -> user is
    announced) *during* channel.connect(). The production ordering must be
    start() -> connect(); these tests pin down what start() may and may not
    wipe under each connection situation.
    """

    @pytest.mark.asyncio
    async def test_start_before_op5_burst_captures_both_speakers(self):
        """The fixed production order: start(), then the join's op-11/op-5
        burst, then audio. Both speakers must end up captured."""
        r = VoiceRecorder(Path("/tmp/unused"))
        r.start(guild_id=1, channel_id=2)

        # Burst arrives while the (subsequent) handshake completes:
        await r.on_voice_ws(None, {"op": 11, "d": {"user_ids": [101, 202]}})
        await r.on_voice_ws(None, {"op": 5, "d": {"speaking": 1, "ssrc": 0xA, "user_id": 101}})
        await r.on_voice_ws(None, {"op": 5, "d": {"speaking": 1, "ssrc": 0xB, "user_id": 202}})

        key = nacl.utils.random(32)
        for ssrc in (0xA, 0xB):
            r.handle_packet(_make_xchacha_packet(SILENCE_PACKET, key, ssrc=ssrc),
                            mode="aead_xchacha20_poly1305_rtpsize", secret_key=key,
                            dave_session=None, resolve_name=lambda uid: "x")
        assert r._speakers[101].frame_count == 1
        assert r._speakers[202].frame_count == 1

    @pytest.mark.asyncio
    async def test_start_after_burst_without_connection_clears_map(self):
        """A fresh join is coming (no live voice client for this channel): old
        SSRCs are per-connection and must be wiped."""
        r = VoiceRecorder(Path("/tmp/unused"))
        await r.on_voice_ws(None, {"op": 5, "d": {"speaking": 1, "ssrc": 0xA, "user_id": 101}})
        assert r._ssrc_to_user[0xA] == 101

        r.start(guild_id=1, channel_id=2)  # _wired_vc is None -> full reset
        assert r._ssrc_to_user == {}

    @pytest.mark.asyncio
    async def test_start_keeps_map_when_already_in_same_channel(self):
        """/stop_recording with leave_channel=false: no new handshake happens,
        so Discord won't re-announce the SSRCs — the map must survive."""
        r = VoiceRecorder(Path("/tmp/unused"))
        await r.on_voice_ws(None, {"op": 5, "d": {"speaking": 1, "ssrc": 0xA, "user_id": 101}})

        r._wired_vc = _FakeVC(channel_id=2)  # bot is already connected to ch 2
        r.start(guild_id=1, channel_id=2)
        assert r._ssrc_to_user[0xA] == 101

    @pytest.mark.asyncio
    async def test_start_clears_map_when_in_different_channel(self):
        """Bot sits in another channel: a move is coming, SSRCs will change."""
        r = VoiceRecorder(Path("/tmp/unused"))
        await r.on_voice_ws(None, {"op": 5, "d": {"speaking": 1, "ssrc": 0xA, "user_id": 101}})

        r._wired_vc = _FakeVC(channel_id=999)  # bot connected elsewhere
        r.start(guild_id=1, channel_id=2)
        assert r._ssrc_to_user == {}

    @pytest.mark.asyncio
    async def test_start_clears_map_when_connection_died(self):
        r = VoiceRecorder(Path("/tmp/unused"))
        await r.on_voice_ws(None, {"op": 5, "d": {"speaking": 1, "ssrc": 0xA, "user_id": 101}})

        r._wired_vc = _FakeVC(channel_id=2, connected=False)  # e.g. bot was kicked
        r.start(guild_id=1, channel_id=2)
        assert r._ssrc_to_user == {}

    def test_discard_resets_state_without_touching_wiring(self):
        r = VoiceRecorder(Path("/tmp/unused"))
        r.start(guild_id=1, channel_id=2)
        vc = _FakeVC(channel_id=2)
        r._wired_vc = vc

        r.discard()
        assert not r.is_recording
        assert r._wired_vc is vc  # wiring stays valid for the next start()

        # A discarded session can be cleanly restarted.
        r.start(guild_id=1, channel_id=2)
        assert r.is_recording


# ── stop() output (WAV + manifest) ───────────────────────────────────────────
class TestStopOutput:
    def _recorder(self, tmp_path: Path) -> VoiceRecorder:
        r = VoiceRecorder(tmp_path / "rec")
        r.start(guild_id=1, channel_id=2, channel_name="General")
        return r

    def test_stop_writes_wav_and_manifest(self, tmp_path):
        r = self._recorder(tmp_path)
        key = nacl.utils.random(32)
        ssrc = 0x2222
        r._note_speaker(ssrc, 999)
        for i in range(5):
            pkt = _make_xchacha_packet(SILENCE_PACKET, key, ssrc=ssrc, counter=i + 1)
            r.handle_packet(pkt, mode="aead_xchacha20_poly1305_rtpsize",
                            secret_key=key, dave_session=None, resolve_name=lambda uid: "Bob")

        manifest = r.stop()
        assert manifest is not None
        assert len(manifest["speakers"]) == 1
        sp = manifest["speakers"][0]
        assert sp["user_id"] == 999
        assert sp["display_name"] == "Bob"
        assert sp["frames_captured"] == 5

        # WAV file exists and is a valid 48 kHz mono 16-bit file.
        wav_path = tmp_path / "rec" / sp["wav_file"]
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

        # Manifest file exists and is valid JSON.
        mp = tmp_path / "rec" / "manifest.json"
        assert mp.exists()
        loaded = json.loads(mp.read_text())
        assert loaded["channel_name"] == "General"
        assert len(loaded["speakers"]) == 1

    def test_stop_when_not_recording_returns_none(self, tmp_path):
        r = VoiceRecorder(tmp_path / "rec")  # never start()
        assert r.is_recording is False
        assert r.stop() is None  # not recording -> no-op


# ── bot wiring (recorder_voice_cls / _wire_voice_client) ─────────────────────
class _FakeVoiceState:
    """Just enough of VoiceConnectionState to exercise the wiring."""

    def __init__(self) -> None:
        self.hook = None
        self.mode = None  # not a str until handshake completes
        self.secret_key: list = []
        self.dave_session = None
        self.listeners: list = []

    def add_socket_listener(self, callback) -> None:
        self.listeners.append(callback)


class TestBotWiring:
    def _bot(self):
        class _Bot:
            pass
        return _Bot()

    def test_recorder_voice_cls_requires_attach(self):
        with pytest.raises(VoiceRecorderError):
            recorder_voice_cls(self._bot())  # type: ignore[arg-type]

    def test_attach_is_idempotent_and_repoints_out_dir(self, tmp_path):
        bot = self._bot()
        r1 = attach_to_bot(bot, tmp_path / "a")  # type: ignore[arg-type]
        r2 = attach_to_bot(bot, tmp_path / "b")  # type: ignore[arg-type]
        assert r1 is r2
        assert str(r2.out_dir) == str(tmp_path / "b")

    def test_recording_voice_client_init_wires_state(self, tmp_path, monkeypatch):
        bot = self._bot()
        rec = attach_to_bot(bot, tmp_path)  # type: ignore[arg-type]
        cls = recorder_voice_cls(bot)
        assert issubclass(cls, discord.VoiceClient)

        state = _FakeVoiceState()
        created = {}

        def fake_super_init(self, client, channel):
            created["client"] = client
            created["channel"] = channel
            self._connection = state

        monkeypatch.setattr(discord.VoiceClient, "__init__", fake_super_init)
        vc = cls(bot, "fake-channel")  # type: ignore[arg-type]

        assert created["client"] is bot
        assert created["channel"] == "fake-channel"
        # (bound methods compare by __self__/__func__, not identity)
        assert state.hook == rec.on_voice_ws
        assert len(state.listeners) == 1
        assert getattr(vc, "_recorder_wired", False) is True

    def test_listener_skips_packets_before_handshake(self, tmp_path):
        bot = self._bot()
        rec = attach_to_bot(bot, tmp_path)  # type: ignore[arg-type]
        state = _FakeVoiceState()
        vc = object.__new__(recorder_voice_cls(bot))
        vc._connection = state
        _wire_voice_client(vc, rec)
        # mode/secret_key not set yet -> listener must be a silent no-op.
        state.listeners[0](b"\x80\x78" + b"\x00" * 20)  # type: ignore[index]
        assert rec._total_packets == 0

    def test_listener_buffers_packet_after_handshake(self, tmp_path):
        bot = self._bot()
        rec = attach_to_bot(bot, tmp_path)  # type: ignore[arg-type]
        rec.start(guild_id=1, channel_id=2)
        state = _FakeVoiceState()
        vc = object.__new__(recorder_voice_cls(bot))
        vc._connection = state
        _wire_voice_client(vc, rec)

        key = nacl.utils.random(32)
        ssrc = 0x3333
        rec._note_speaker(ssrc, 777)
        # Simulate a completed handshake.
        state.mode = "aead_xchacha20_poly1305_rtpsize"
        state.secret_key = list(key)
        pkt = _make_xchacha_packet(SILENCE_PACKET, key, ssrc=ssrc)
        state.listeners[0](pkt)  # type: ignore[index]

        assert rec._total_packets == 1
        assert rec._speakers[777].frame_count == 1


# ── filename sanitisation ────────────────────────────────────────────────────
class TestWavFilename:
    def test_sanitises_display_name(self):
        name = _wav_filename(123, "Alice/Bob (x)")
        assert "/" not in name and "(" not in name
        assert name.endswith("_123.wav")

    def test_falls_back_to_user_id_when_blank(self):
        name = _wav_filename(555, "")
        assert name.startswith("user-555") and name.endswith("_555.wav")


# ── Timeline WAV placement (jitter fix) ──────────────────────────────────────
class TestWriteTimelineWav:
    def test_frames_laid_out_on_nominal_grid(self, tmp_path):
        """Consecutive frames must be exactly 20 ms apart regardless of the
        jittery wall-clock arrival times they were tagged with."""
        origin = 1000.0
        # arrivals jitter by +/-8 ms around the nominal 20 ms interval
        ts = origin + 2.5
        frames = []
        for i in range(10):
            frames.append((ts, b"\x01\x02" * (FRAME_SAMPLES // 2)))
            ts += 0.02 + (0.008 if i % 2 else -0.008)
        total = int((frames[0][0] - origin) * SAMPLE_RATE) + len(frames) * FRAME_SAMPLES
        out = tmp_path / "s.wav"
        gaps, overlaps = _write_timeline_wav(path=out, frames=frames, origin=origin, total_samples=total)
        import wave, struct as st
        w = wave.open(str(out))
        s = list(st.unpack(f"<{w.getnframes()}h", w.readframes(w.getnframes())))
        first = int((frames[0][0] - origin) * SAMPLE_RATE)
        # every frame boundary must contain a non-silence sample exactly on grid
        for i in range(10):
            assert s[first + i * FRAME_SAMPLES] != 0, f"frame {i} not at nominal position"
        # file length == anchor + N*20ms (no jitter inflation)
        assert w.getnframes() == total

    def test_no_overlap_when_arrivals_burst(self, tmp_path):
        """Bursty arrivals (<20 ms apart) must not overwrite previous frames."""
        origin = 1000.0
        # little-endian 16-bit samples so each sample is unambiguous
        frames = [(origin + 1.0, struct.pack(f"<{FRAME_SAMPLES}h", *[700] * FRAME_SAMPLES)),
                  (origin + 1.0 + 0.005, struct.pack(f"<{FRAME_SAMPLES}h", *[900] * FRAME_SAMPLES))]
        total = int((frames[0][0] - origin) * SAMPLE_RATE) + len(frames) * FRAME_SAMPLES
        out = tmp_path / "s.wav"
        _write_timeline_wav(path=out, frames=frames, origin=origin, total_samples=total)
        import wave, struct as st
        w = wave.open(str(out))
        s = list(st.unpack(f"<{w.getnframes()}h", w.readframes(w.getnframes())))
        first = int((frames[0][0] - origin) * SAMPLE_RATE)
        # second frame starts exactly one nominal frame later, with its own data
        assert s[first + FRAME_SAMPLES] == 900

    def test_returns_jitter_diagnostics(self, tmp_path):
        origin = 1000.0
        frames = [(origin + 1.0, b"\x01\x02" * (FRAME_SAMPLES // 2))]
        # one late arrival (>30 ms) and one early (<10 ms)
        frames.append((frames[-1][0] + 0.040, b"\x03\x04" * (FRAME_SAMPLES // 2)))
        frames.append((frames[-1][0] + 0.005, b"\x05\x06" * (FRAME_SAMPLES // 2)))
        total = int((frames[0][0] - origin) * SAMPLE_RATE) + len(frames) * FRAME_SAMPLES
        gaps, overlaps = _write_timeline_wav(path=tmp_path / "s.wav", frames=frames, origin=origin, total_samples=total)
        assert gaps == 1 and overlaps == 1


# ── Crash durability: on-disk frame logs ─────────────────────────────────────
class TestSpeakerLog:
    def test_write_then_read_roundtrip(self, tmp_path):
        p = tmp_path / "s.log"
        lg = _SpeakerLog(p)
        lg.write_frame(1000.000, b"\x01\x02" * 8)
        lg.write_frame(1000.020, b"\x03\x04" * 8)
        lg.close()
        frames = _SpeakerLog.read_frames(p)
        assert len(frames) == 2
        assert abs(frames[0][0] - 1000.000) < 1e-6
        assert frames[0][1] == b"\x01\x02" * 8
        assert frames[1][1] == b"\x03\x04" * 8

    def test_read_meta_reports_format(self, tmp_path):
        p = tmp_path / "s.log"
        lg = _SpeakerLog(p)
        lg.write_frame(1.0, b"\x00" * 4)
        lg.close()
        meta = _SpeakerLog.read_meta(p)
        assert meta["sample_rate"] == SAMPLE_RATE
        assert meta["channels"] == 1
        assert meta["frame_samples"] == FRAME_SAMPLES

    def test_trailing_partial_record_is_dropped(self, tmp_path):
        """A frame mid-write at crash time must not corrupt the readable prefix."""
        p = tmp_path / "s.log"
        lg = _SpeakerLog(p)
        lg.write_frame(1000.0, b"\x01" * 16)   # complete record
        lg.close()
        # Simulate a crash that left a header + partial payload on disk.
        with open(p, "ab") as fh:
            import struct as st
            fh.write(st.pack("<qI", int(1000.02 * 1_000_000), 16))
            fh.write(b"\x02" * 5)              # only 5 of 16 bytes made it
        frames = _SpeakerLog.read_frames(p)
        assert len(frames) == 1
        assert frames[0][1] == b"\x01" * 16

    def test_read_frames_on_empty_or_bad_file(self, tmp_path):
        empty = tmp_path / "empty.log"
        empty.write_bytes(b"")
        assert _SpeakerLog.read_frames(empty) == []
        bad = tmp_path / "bad.log"
        bad.write_bytes(b"XXXX" + b"\x00" * 20)
        assert _SpeakerLog.read_frames(bad) == []


# ── Crash durability: stop() reads logs, cleans up ───────────────────────────
class TestStopDurability:
    def test_stop_writes_from_disk_and_removes_logs(self, tmp_path):
        r = VoiceRecorder(tmp_path / "rec")
        r.start(guild_id=1, channel_id=2, channel_name="General")
        key = nacl.utils.random(32)
        ssrc = 0x4444
        r._note_speaker(ssrc, 555)
        for i in range(6):
            pkt = _make_xchacha_packet(SILENCE_PACKET, key, ssrc=ssrc, counter=i + 1)
            r.handle_packet(pkt, mode="aead_xchacha20_poly1305_rtpsize",
                            secret_key=key, dave_session=None, resolve_name=lambda uid: "Carol")
        # A per-speaker log file exists on disk while recording (durability).
        logs = list((tmp_path / "rec").glob("*.log"))
        assert len(logs) == 1

        manifest = r.stop()
        assert manifest is not None
        sp = manifest["speakers"][0]
        assert sp["frames_captured"] == 6
        assert (tmp_path / "rec" / sp["wav_file"]).exists()
        # After a clean stop the intermediate logs + marker are gone.
        assert not list((tmp_path / "rec").glob("*.log"))
        assert not (tmp_path / "rec" / ".recording").exists()

    def test_discard_removes_logs_and_marker(self, tmp_path):
        r = VoiceRecorder(tmp_path / "rec")
        r.start(guild_id=1, channel_id=2)
        key = nacl.utils.random(32)
        r._note_speaker(0x5, 9)
        r.handle_packet(_make_xchacha_packet(SILENCE_PACKET, key, ssrc=0x5),
                        mode="aead_xchacha20_poly1305_rtpsize", secret_key=key,
                        dave_session=None, resolve_name=lambda uid: "")
        assert list((tmp_path / "rec").glob("*.log"))
        r.discard()
        assert not list((tmp_path / "rec").glob("*.log"))
        assert not (tmp_path / "rec" / ".recording").exists()

    def test_start_clears_stale_artifacts(self, tmp_path):
        d = tmp_path / "rec"
        d.mkdir()
        (d / "stale.log").write_bytes(b"junk")
        (d / ".recording").write_text("{}")
        r = VoiceRecorder(d)
        r.start(guild_id=1, channel_id=2)
        assert not (d / "stale.log").exists()
        # start() re-writes a fresh marker for the new session
        assert (d / ".recording").exists()


# ── Crash recovery: recover_orphans() ────────────────────────────────────────
def _make_orphan(rec_dir: Path, started_at: float) -> None:
    """Create a crashed-session dir: marker + one speaker log, no manifest."""
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / ".recording").write_text(json.dumps(
        {"started_at": started_at, "guild_id": 1, "channel_id": 2,
         "channel_name": "crashed-room", "pid": 4242}))
    lg = _SpeakerLog(rec_dir / "Bob_777.log")
    t = started_at + 1.0
    for i in range(5):
        lg.write_frame(t, b"\x07\x08" * (FRAME_SAMPLES // 2))
        t += 0.02
    lg.close()


class TestRecoverOrphans:
    def test_recovers_old_orphan(self, tmp_path):
        rec_dir = tmp_path / "recording_20260101-000000"
        _make_orphan(rec_dir, started_at=time.time() - 3600)  # an hour old
        recovered = recover_orphans(tmp_path)
        assert len(recovered) == 1
        m = recovered[0]
        assert m["recovered"] is True
        assert m["channel_name"] == "crashed-room"
        assert len(m["speakers"]) == 1
        sp = m["speakers"][0]
        assert sp["user_id"] == 777
        assert sp["frames_captured"] == 5
        # WAV written, logs + marker cleaned up, manifest on disk.
        assert (rec_dir / sp["wav_file"]).exists()
        assert not list(rec_dir.glob("*.log"))
        assert not (rec_dir / ".recording").exists()
        assert (rec_dir / "manifest.json").exists()

    def test_skips_recent_orphan_within_threshold(self, tmp_path):
        rec_dir = tmp_path / "recording_recent"
        _make_orphan(rec_dir, started_at=time.time() - 60)  # 60s ago < 5 min
        assert recover_orphans(tmp_path) == []
        # untouched — still an open session
        assert (rec_dir / ".recording").exists()
        assert list(rec_dir.glob("*.log"))

    def test_skips_completed_recording(self, tmp_path):
        rec_dir = tmp_path / "recording_done"
        _make_orphan(rec_dir, started_at=time.time() - 3600)
        (rec_dir / "manifest.json").write_text("{}")  # already finalized
        assert recover_orphans(tmp_path) == []

    def test_no_op_when_dir_missing(self, tmp_path):
        assert recover_orphans(tmp_path / "does-not-exist") == []

    def test_recovers_multiple_speakers_and_drops_empty_logs(self, tmp_path):
        rec_dir = tmp_path / "recording_multi"
        _make_orphan(rec_dir, started_at=time.time() - 3600)
        # a second speaker with real frames...
        lg2 = _SpeakerLog(rec_dir / "Dan_888.log")
        t = time.time() - 3590
        for i in range(4):
            lg2.write_frame(t, b"\x09" * (FRAME_SAMPLES))
            t += 0.02
        lg2.close()
        # ...and a third speaker whose log is empty (no audio) -> dropped.
        _SpeakerLog(rec_dir / "Erin_999.log").close()

        recovered = recover_orphans(tmp_path)
        assert len(recovered) == 1
        uids = sorted(s["user_id"] for s in recovered[0]["speakers"])
        assert uids == [777, 888]   # Erin (empty) excluded


# ── Crash durability: SIGTERM graceful flush ─────────────────────────────────
class TestSigtermFlush:
    def _patch_exit(self, monkeypatch):
        # The real handler ends with os._exit(0), which would kill the test
        # process. Replace it with something that raises SystemExit so we can
        # assert on the flushed state without actually exiting.
        import bot_core.voice_recorder as vr

        def _fake_exit(code):
            raise SystemExit(code)

        monkeypatch.setattr(vr.os, "_exit", _fake_exit)

    def test_sigterm_handler_flushes_recording(self, tmp_path, monkeypatch):
        r = VoiceRecorder(tmp_path / "rec")
        r.start(guild_id=1, channel_id=2)
        key = nacl.utils.random(32)
        r._note_speaker(0x6, 123)
        for i in range(3):
            r.handle_packet(_make_xchacha_packet(SILENCE_PACKET, key, ssrc=0x6),
                            mode="aead_xchacha20_poly1305_rtpsize", secret_key=key,
                            dave_session=None, resolve_name=lambda uid: "Dave")
        assert r.is_recording

        self._patch_exit(monkeypatch)
        with pytest.raises(SystemExit):
            r._sigterm_handler(15, None)

        assert not r.is_recording
        manifest = json.loads((tmp_path / "rec" / "manifest.json").read_text())
        assert manifest["speakers"][0]["frames_captured"] == 3
        assert not list((tmp_path / "rec").glob("*.log"))

    def test_sigterm_handler_noop_when_not_recording(self, tmp_path, monkeypatch):
        r = VoiceRecorder(tmp_path / "rec")  # never start()
        self._patch_exit(monkeypatch)
        with pytest.raises(SystemExit):
            r._sigterm_handler(15, None)
        assert not (tmp_path / "rec" / "manifest.json").exists()
