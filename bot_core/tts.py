"""TTS client for /speak (#34) — any OpenAI-compatible /v1/audio/speech endpoint.

Unsloth Studio serves ``/v1/audio/speech`` at the same base URL as its chat
API (e.g. a Kokoro GGUF TTS model), so this module reuses the provider-agnostic
``AsyncOpenAI`` pattern from :mod:`bot_core.ai_client`. The synthesized audio
comes back as mp3 bytes, ready for a Discord attachment.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
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


# ── Unsloth Studio: co-resident load ─────────────────────────────────
# Studio's /v1/audio/speech auto-load cannot bring a TTS model up while the
# chat model holds the GPU, but an explicit POST /api/inference/load can —
# both models then stay resident at once. Studio also idle-unloads audio
# models, so /speak re-loads on demand when it sees "No model loaded".


def _unsloth_api_base() -> str | None:
    """The Studio API origin for an Unsloth-style ``…/v1`` TTS_URL, else None."""
    base = TTS_URL.rstrip("/")
    if not base.endswith("/v1"):
        return None
    return base[: -len("/v1")]


async def _ensure_model_loaded() -> bool:
    """Load the TTS model via Studio's /api/inference/load and wait until resident.

    Returns False when the endpoint is not an Unsloth Studio layout or the
    load does not finish within ``TTS_TIMEOUT_S``.
    """
    api_base = _unsloth_api_base()
    if api_base is None:
        return False
    headers = {"Authorization": f"Bearer {TTS_API_KEY}"} if TTS_API_KEY else {}
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            await http.post(
                f"{api_base}/api/inference/load",
                json={"model_path": TTS_MODEL},
                headers=headers,
            )
            loop = asyncio.get_running_loop()
            deadline = loop.time() + TTS_TIMEOUT_S
            while loop.time() < deadline:
                resp = await http.get(
                    f"{api_base}/api/inference/loaded-models", headers=headers
                )
                data = resp.json().get("data", [])
                if any(
                    m.get("id") == TTS_MODEL and m.get("loaded") for m in data
                ):
                    log.info("TTS model re-loaded: %s", TTS_MODEL)
                    return True
                await asyncio.sleep(5)
    except Exception as e:  # noqa: BLE001 - load is best-effort; report below
        log.warning("TTS auto-load failed: %s: %s", e.__class__.__name__, e)
    return False


def _is_not_loaded_error(e: Exception) -> bool:
    return "no model loaded" in str(e).lower()


async def synthesize(text: str) -> bytes:
    """Synthesize ``text`` to mp3 audio bytes.

    Raises :class:`TTSError` on any backend failure or timeout, and
    ``ValueError`` when no TTS model is configured.
    """
    if not TTS_MODEL:
        raise ValueError("TTS_MODEL is not set — /speak is disabled.")
    client = _make_tts_client()
    try:
        return await _speech_once(client, text)
    except TTSError as e:
        # One bounded recovery: the model was idle-unloaded (or never
        # co-loaded) — load it next to the chat model and retry exactly once.
        if not _is_not_loaded_error(e):
            raise
        if not await _ensure_model_loaded():
            raise
        return await _speech_once(client, text)


async def _speech_once(client: AsyncOpenAI, text: str) -> bytes:
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
