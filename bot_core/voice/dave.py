"""RTP / DAVE decryption + Opus decode primitives.

Extracted from ``bot_core.voice_recorder`` (P3 #35). Pure, unit-testable
helpers that turn a raw Discord voice packet into decoded 48 kHz mono PCM:
parse the RTP "rtpsize" transport, decrypt the transport layer
(XChaCha20-Poly1305 or AES-256-GCM), strip the E2EE passthrough, decrypt
the DAVE layer via ``davey``, decode the Opus frame, and downmix to mono.
"""
from __future__ import annotations

import logging
import struct
from typing import Any, Callable, Optional

import nacl.secret

from bot_core.voice.capture import SAMPLE_WIDTH

log = logging.getLogger("bot.voice_recorder")

# RTP "rtpsize" transport layout constants.
_RTP_HEADER_SIZE = 12
_AUTH_TAG_LEN = 16            # Poly1305 / GCM tag
_NONCE_TRAILER_LEN = 4        # trailing 32-bit counter nonce

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
