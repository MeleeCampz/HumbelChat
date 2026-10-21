"""TTS client for /speak (#34) — any OpenAI-compatible /v1/audio/speech endpoint.

Unsloth Studio serves ``/v1/audio/speech`` at the same base URL as its chat
API (e.g. a Kokoro GGUF TTS model), so this module reuses the provider-agnostic
``AsyncOpenAI`` pattern from :mod:`bot_core.ai_client`. The synthesized audio
comes back as mp3 bytes, ready for a Discord attachment.
"""
from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from config.settings import (
    TTS_API_KEY,
    TTS_MODEL,
    TTS_TIMEOUT_S,
    TTS_VOICE,
    TTS_URL,
)

log = logging.getLogger("bot.bot_core.tts")


class TTSError(RuntimeError):
    """TTS synthesis failed (backend down, model not loaded, timeout, …)."""


# ── Shared client (singleton) ─────────────────────────────────────────
_tts_client: AsyncOpenAI | None = None


def _make_tts_client() -> AsyncOpenAI:
    global _tts_client
    if _tts_client is None:
        # max_retries=0: the bot owns its retry policy (no retries here —
        # a second synthesis attempt would just double the wait).
        _tts_client = AsyncOpenAI(
            api_key=TTS_API_KEY, base_url=TTS_URL, max_retries=0
        )
    return _tts_client


def tts_configured() -> bool:
    """True when a TTS model is configured (feature on)."""
    return bool(TTS_MODEL)


async def synthesize(text: str) -> bytes:
    """Synthesize ``text`` to mp3 audio bytes.

    Raises :class:`TTSError` on any backend failure or timeout, and
    ``ValueError`` when no TTS model is configured.
    """
    if not TTS_MODEL:
        raise ValueError("TTS_MODEL is not set — /speak is disabled.")
    client = _make_tts_client()
    try:
        resp = await asyncio.wait_for(
            client.audio.speech.create(
                model=TTS_MODEL, voice=TTS_VOICE, input=text, response_format="mp3"
            ),
            timeout=TTS_TIMEOUT_S,
        )
    except asyncio.TimeoutError as e:
        raise TTSError(f"timed out after {TTS_TIMEOUT_S:.0f}s") from e
    except Exception as e:  # APIStatusError / connection errors / …
        raise TTSError(f"{e.__class__.__name__}: {e}") from e

    audio = resp.read()
    if not audio:
        raise TTSError("backend returned no audio data")
    log.info("TTS ok: %d chars -> %d bytes (%s)", len(text), len(audio), TTS_MODEL)
    return audio
