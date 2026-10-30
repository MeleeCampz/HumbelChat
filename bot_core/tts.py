"""TTS client for /speak (#34) — any OpenAI-compatible /v1/audio/speech endpoint.

Unsloth Studio serves ``/v1/audio/speech`` at the same base URL as its chat
API (e.g. a Kokoro GGUF TTS model), so this module reuses the provider-agnostic
``AsyncOpenAI`` pattern from :mod:`bot_core.ai_client`. The synthesized audio
comes back as audio bytes (mp3 from Studio, wav from Ollama), ready for a
Discord attachment.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable

import httpx
from openai import APIStatusError, AsyncOpenAI

from utils.studio_model import is_resident, load_payload

from config.settings import (
    DEFAULT_MODEL,
    TTS_API_KEY,
    TTS_BACKEND,
    TTS_MODEL,
    TTS_TIMEOUT_S,
    TTS_VOICE,
    TTS_URL,
    TTS_X_VECTOR_ONLY_MODE,
    TTS_GPU_IDS,
)

log = logging.getLogger("bot.bot_core.tts")


class TTSError(RuntimeError):
    """TTS synthesis failed (backend down, model not loaded, timeout, …).

    ``retryable`` marks transient server-side failures (e.g. Ollama's MLX
    runner panicking on the first request after a reload) — :func:`synthesize`
    retries those exactly once.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


# ── Shared client (singleton) ─────────────────────────────────────────
_tts_client: AsyncOpenAI | None = None


def _make_tts_client() -> AsyncOpenAI:
    global _tts_client
    if _tts_client is None:
        # max_retries=0: the bot owns its retry policy (no retries here —
        # a second synthesis attempt would just double the wait).
        # Ollama needs no key, but AsyncOpenAI requires a non-empty one.
        _tts_client = AsyncOpenAI(
            api_key=TTS_API_KEY or "ollama", base_url=TTS_URL, max_retries=0
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


async def _ensure_slug_loaded(model_path: str, wait_s: float) -> bool:
    """Load *model_path* via Studio's /api/inference/load and wait until resident.

    Returns False when the endpoint is not an Unsloth Studio layout or the
    load does not finish within *wait_s*. Co-loading works: an explicit load
    brings the new model up NEXT TO the currently resident one (both stay
    loaded at once), so loading the chat model back after a TTS use does not
    kick the TTS model out again.

    #73 class fix (code review 2026-10): Studio's load endpoint takes a bare
    repo id + optional ``gguf_variant`` — sending the colon-suffixed
    ``repo:quant`` slug verbatim fails HF repo-id validation, and residency
    is only ever reported under the bare id. Split via the shared helper;
    check the POST status instead of silently polling a never-loaded model.
    The POST itself blocks until the load completes (observed ~70 s for the
    chat model in the images path), so its timeout IS the wait budget.
    """
    api_base = _unsloth_api_base()
    if api_base is None or not model_path:
        return False
    headers = {"Authorization": f"Bearer {TTS_API_KEY}"} if TTS_API_KEY else {}
    # NOTE: this Studio version silently ignores gpu_ids for audio.cpp GGUF
    # models — a GPU TTS load evicts the resident chat model and the pair
    # swaps back per use (accepted trade-off; kept for future Studio builds).
    extra = {"gpu_ids": TTS_GPU_IDS} if TTS_GPU_IDS else None
    payload = load_payload(model_path, extra)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(wait_s, connect=10.0)) as http:
            resp = await http.post(
                f"{api_base}/api/inference/load",
                json=payload,
                headers=headers,
            )
            if resp.status_code >= 400:
                log.warning(
                    "Model load request failed (%s): HTTP %d — %s",
                    model_path, resp.status_code, (resp.text or "")[:200],
                )
                return False
            loop = asyncio.get_running_loop()
            deadline = loop.time() + wait_s
            while loop.time() < deadline:
                resp = await http.get(
                    f"{api_base}/api/inference/loaded-models", headers=headers
                )
                data = resp.json().get("data", [])
                if any(is_resident(m, model_path) for m in data):
                    log.info("Model re-loaded: %s", model_path)
                    return True
                await asyncio.sleep(5)
    except Exception as e:  # noqa: BLE001 - load is best-effort; report below
        log.warning(
            "Model auto-load failed (%s): %s: %s",
            model_path, e.__class__.__name__, e,
        )
    return False


async def _ensure_model_loaded() -> bool:
    """Load the TTS model via Studio's /api/inference/load and wait until resident.

    Ollama loads on demand by itself — nothing to ensure (and no Studio API).
    """
    if TTS_BACKEND != "studio":
        return True
    return await _ensure_slug_loaded(TTS_MODEL, TTS_TIMEOUT_S)


# ── Chat-model swap-back after /speak ─────────────────────────────────
# A GPU TTS load evicts the resident chat model (see NOTE above), so the next
# question after a /speak would pay the 30–90 s cold load. The chat model is
# needed for the whole session anyway, so bring it back in the BACKGROUND
# right after synthesis — while the user is listening to the audio — and the
# next question finds it warm. Fire-and-forget: never delays the reply.
_chat_reload_task: asyncio.Task[None] | None = None
_CHAT_RELOAD_WAIT_S = 120  # cold chat-model loads can take upwards of a minute


async def _swap_chat_model_back() -> None:
    """Best-effort background reload of the chat model after a TTS evicted it."""
    global _chat_reload_task
    if TTS_BACKEND != "studio":
        return  # Ollama is a separate process — it never evicts Studio's chat model
    if not DEFAULT_MODEL or _unsloth_api_base() is None:
        return
    await _ensure_slug_loaded(DEFAULT_MODEL, _CHAT_RELOAD_WAIT_S)
    _chat_reload_task = None


def _schedule_chat_swap_back() -> None:
    """Schedule :func:`_swap_chat_model_back` without awaiting it (deduped)."""
    global _chat_reload_task
    if TTS_BACKEND != "studio":
        return  # Ollama is a separate process — no swap-back needed
    if not DEFAULT_MODEL or _unsloth_api_base() is None:
        return  # nothing to reload / no Studio API to reload it with
    if _chat_reload_task is not None and not _chat_reload_task.done():
        return  # a reload is already in flight
    try:
        _chat_reload_task = asyncio.get_running_loop().create_task(
            _swap_chat_model_back()
        )
    except RuntimeError:  # no running loop — nothing to schedule against
        pass


def _is_not_loaded_error(e: Exception) -> bool:
    return "no model loaded" in str(e).lower()


async def synthesize(text: str, on_phase: Callable[[str], Any] | None = None) -> bytes:
    """Synthesize ``text`` to mp3 audio bytes.

    Raises :class:`TTSError` on any backend failure or timeout, and
    ``ValueError`` when no TTS model is configured.

    *on_phase* (#70) is an optional progress hook — sync or async callable
    taking a phase name ("speaking", "loading") — so the caller can show the
    user live status during the long load windows. A broken callback never
    fails the synthesis.
    """
    if not TTS_MODEL:
        raise ValueError("TTS_MODEL is not set — /speak is disabled.")
    client = _make_tts_client()

    async def _phase(name: str) -> None:
        if on_phase is None:
            return
        try:
            result = on_phase(name)
            if inspect.isawaitable(result):
                await result
        except Exception as e:  # noqa: BLE001 - progress must never fail the TTS
            log.debug("phase callback failed (%s): %s", name, e)

    try:
        await _phase("speaking")
        audio = await _speech_once(client, text)
    except TTSError as e:
        # One bounded recovery for transient failures: (a) the model was
        # idle-unloaded (Studio) — load it next to the chat model; (b) the
        # server errored on a cold runner (Ollama MLX panic, 5xx) — let it
        # settle. Either way: retry exactly once.
        if not (_is_not_loaded_error(e) or e.retryable):
            raise
        await _phase("loading")
        if not await _ensure_model_loaded():
            raise
        await asyncio.sleep(1.0)
        await _phase("speaking")
        audio = await _speech_once(client, text)
    # The TTS load just evicted the resident chat model — bring it back in
    # the background so the next question doesn't pay the cold load.
    _schedule_chat_swap_back()
    return audio


async def _speech_once(client: AsyncOpenAI, text: str) -> bytes:
    # Qwen3-TTS (Unsloth Studio): clone via speaker embedding only — the base
    # model's ICL reference prompting otherwise leaks the saved voice's
    # transcript tail into the output. Ollama phonon has exactly one fixed
    # voice per language checkpoint and rejects unknown voice names.
    voice = TTS_VOICE if TTS_BACKEND == "studio" else "default"
    # Ollama serves wav/pcm only (mp3 is rejected with a 400); Discord accepts
    # both — wav at 24 kHz mono 16-bit is ~48 KB/s, far under the 10 MB cap.
    response_format = "mp3" if TTS_BACKEND == "studio" else "wav"
    extra_body: dict[str, Any] | None = (
        {"audio_options": {"x_vector_only_mode": True}}
        if TTS_BACKEND == "studio" and TTS_X_VECTOR_ONLY_MODE
        else None
    )
    try:
        resp = await asyncio.wait_for(
            client.audio.speech.create(
                model=TTS_MODEL,
                voice=voice,
                input=text,
                response_format=response_format,
                extra_body=extra_body,
            ),
            timeout=TTS_TIMEOUT_S,
        )
    except asyncio.TimeoutError as e:
        raise TTSError(f"timed out after {TTS_TIMEOUT_S:.0f}s") from e
    except Exception as e:  # APIStatusError / connection errors / …
        # Ollama's MLX runner intermittently panics on the first request
        # after a (re)load — "mlx runner failed" 500 that recovers at once
        # (Windows driver doesn't expose the MLX working-set hint, so it falls
        # back to pageable memory; WDDM lazy commit can lose the race). One
        # bounded retry in synthesize().
        status = e.status_code if isinstance(e, APIStatusError) else None
        retryable = isinstance(status, int) and status >= 500
        raise TTSError(f"{e.__class__.__name__}: {e}", retryable=retryable) from e

    audio = resp.read()
    if not audio:
        raise TTSError("backend returned no audio data")
    log.info("TTS ok: %d chars -> %d bytes (%s)", len(text), len(audio), TTS_MODEL)
    return audio
