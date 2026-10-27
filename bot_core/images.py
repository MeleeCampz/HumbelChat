"""Image generation (#42) — Unsloth Studio diffusion via the OpenAI-compatible
``/v1/images/generations`` endpoint (same host + key as chat), with swap
orchestration.

Architecture constraints (verified 2026-10-26):
- Co-residency is impossible by design: loading the image model EVICTS the
  resident chat model (Studio swaps; context survives via keep-across-unload).
- The working set scales with output resolution (10 GB+ at high res), so the
  default size is capped (IMAGE_DEFAULT_SIZE / IMAGE_MAX_SIZE).
- **No chat-model calls while the image model is resident** — any chat
  completion re-residents the chat model and evicts the image mid-generation
  (this is what made early API load attempts look like "silent failures").
  Hence: prompt rewriting happens BEFORE the swap, and the whole
  load→generate→unload window runs while holding the global AI slot so other
  chat turns queue behind it instead of thrashing the swap.
- The chat model is swapped back AFTER every generation (finally), including
  failed ones — a background warm load makes the next chat turn fast.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time

import httpx
from openai import AsyncOpenAI

from bot_core.ai_client import _create_with_reasoning_fallback
from config.settings import (
    DEFAULT_MODEL,
    IMAGE_DEFAULT_SIZE,
    IMAGE_GEN_TIMEOUT_S,
    IMAGE_GGUF_FILENAME,
    IMAGE_GPU_IDS,
    IMAGE_LOAD_TIMEOUT_S,
    IMAGE_MAX_SIZE,
    IMAGE_MODEL,
    INFER_API_KEY,
    INFER_URL,
)

log = logging.getLogger("bot.bot_core.images")


class ImageGenError(Exception):
    """Image generation / swap orchestration failure (user-safe message)."""


def image_gen_configured() -> bool:
    return bool(IMAGE_MODEL)


# ── Studio management API base (INFER_URL is .../v1; mgmt endpoints live on
# the bare host) ────────────────────────────────────────────────────────────
_STUDIO_BASE = INFER_URL.removesuffix("/v1").removesuffix("/")
_STUDIO_HEADERS = {"Authorization": f"Bearer {INFER_API_KEY}"}


async def _studio(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> dict:
    resp = await client.request(method, f"{_STUDIO_BASE}{path}", headers=_STUDIO_HEADERS, **kwargs)
    if resp.status_code >= 400:
        raise ImageGenError(f"Studio {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


async def _image_status(client: httpx.AsyncClient) -> dict:
    return await _studio(client, "GET", "/api/inference/images/status")


async def ensure_image_loaded() -> None:
    """Load the configured image model if not resident (evicts chat by design)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        st = await _image_status(client)
        if st.get("loaded"):
            return
        log.info("loading image model %s (evicts chat model)", IMAGE_MODEL)
        await _studio(client, "POST", "/api/inference/images/load", json={
            "model_path": IMAGE_MODEL,
            "model_kind": "gguf",
            "gguf_filename": IMAGE_GGUF_FILENAME,
            "gpu_ids": IMAGE_GPU_IDS,
        })
        deadline = time.monotonic() + IMAGE_LOAD_TIMEOUT_S
        while time.monotonic() < deadline:
            await asyncio.sleep(10)
            if (await _image_status(client)).get("loaded"):
                log.info("image model loaded (resident=%s MiB)", st.get("resident_mib"))
                return
        raise ImageGenError(f"image model did not load within {IMAGE_LOAD_TIMEOUT_S:.0f}s")


async def _unload_image() -> None:
    """Swap back: unload the image model. Best-effort — never raises into the
    caller's error path (a stuck unload still ends with chat re-loadable)."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            await _studio(client, "POST", "/api/inference/images/unload")
            log.info("image model unloaded — swapping chat back in")
    except Exception as e:  # noqa: BLE001 — best-effort by design
        log.warning("image unload failed (chat still re-loadable on demand): %s", e)


async def _warm_chat_reload() -> None:
    """Background: re-resident the chat model after an image generation so the
    next chat turn doesn't pay a cold load. Failure is harmless — Studio
    auto-loads the chat model on the bot's next completion request."""
    if not DEFAULT_MODEL:
        return
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            await _studio(client, "POST", "/api/inference/load", json={"model_path": DEFAULT_MODEL})
    except Exception as e:  # noqa: BLE001 — best-effort by design
        log.warning("background chat re-load failed (harmless): %s", e)


# ── Prompt rewriting (chat model — MUST run before the swap) ──────────────
_REWRITE_SYSTEM = (
    "You are a prompt engineer for a semi-realistic painterly fantasy image model set "
    "in Humblewood, a world from the Dungeons & Dragons universe (D&D 5e setting). "
    "Rewrite the user's short request into ONE rich paragraph (max ~80 words) that the "
    "image model can render directly: concrete subject, setting, lighting, mood and "
    "palette. Style: 'semi-realistic painterly fantasy illustration with soft "
    "watercolor touches — detailed fur and feather texture, natural light, storybook "
    "charm'. Characters are D&D player characters, NOT wild animals: bipedal humanoid "
    "animal adventurers standing upright, in fitting clothing and gear (tunic, leather "
    "armor, satchel, tool or weapon matching their role), confident and expressive. "
    "Keep every element the user asked for; do not add named characters or brands they "
    "didn't mention. Hard rules: NO humans — only animals or humanoid animals (bird-"
    "folk and other beast-folk in the D&D Humbelfolk style); no text or lettering in "
    "the image; no modern objects. Species trait: bird-folk have VERY large, oversized "
    "round googly eyes — clearly bigger than a normal animal's, their most prominent "
    "facial feature. If the image contains bird-folk, give EVERY one of them those "
    "eyes; if it contains no birds at all, do not mention googly eyes. Reply with the "
    "rewritten prompt only — no quotes, no commentary."
)


def _rewrite_client() -> AsyncOpenAI:
    # Same host/key as chat (bot_core.ai_client's singleton is private; a small
    # dedicated client keeps this module independently testable).
    return AsyncOpenAI(api_key=INFER_API_KEY, base_url=INFER_URL, max_retries=0)


async def rewrite_prompt(text: str) -> str:
    """Expand a short user request into a rich image prompt via the chat model.

    Runs on the resident chat model — call this BEFORE ensure_image_loaded().
    Raises ImageGenError on failure (caller falls back to the raw prompt).
    """
    client = _rewrite_client()
    try:
        # The chat model is a reasoning model — without an explicit low effort
        # the thinking budget eats the token cap and ``content`` comes back
        # empty (observed live, intermittently even WITH low effort: one run
        # rambled ~700 reasoning tokens and truncated before any content).
        # Same param plumbing + 4xx fallback as the main AI path
        # (bot_core.ai_client, #20); generous cap + one retry on empty.
        rewritten = ""
        for attempt in (1, 2):
            resp = await asyncio.wait_for(
                _create_with_reasoning_fallback(
                    client,
                    "low",
                    model=DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": _REWRITE_SYSTEM},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=2048,
                ),
                timeout=120.0,
            )
            rewritten = (resp.choices[0].message.content or "").strip().strip('"')
            if rewritten:
                return rewritten
            log.warning("prompt rewrite attempt %d came back empty — retrying", attempt)
        raise ImageGenError("empty rewrite")
    except ImageGenError:
        raise
    except Exception as e:  # noqa: BLE001 — fall back to raw prompt in the caller
        raise ImageGenError(f"prompt rewrite failed: {e.__class__.__name__}") from e


# ── Generation (the no-chat-calls window) ─────────────────────────────────
_gen_lock = asyncio.Lock()  # one generation at a time (swap is global state)


def resolve_size(raw: str | None) -> int | None:
    """Parse an optional ``--size`` flag value; clamp to [256, IMAGE_MAX_SIZE].
    Returns None → caller uses IMAGE_DEFAULT_SIZE."""
    if raw is None or not str(raw).strip():
        return None
    try:
        n = int(str(raw).strip())
    except ValueError:
        raise ImageGenError(f"invalid size {raw!r} — give a number of pixels (e.g. 1024)")
    if n < 256:
        return 256
    return min(n, IMAGE_MAX_SIZE)


async def generate_image(prompt: str, size: int | None = None) -> bytes:
    """Generate one image and return PNG bytes.

    Holds the generation lock for the whole load→generate→unload window; the
    CALLER is responsible for holding the global AI slot around this call so
    no other chat turn can re-resident the chat model mid-swap.
    """
    side = size or IMAGE_DEFAULT_SIZE
    async with _gen_lock:
        client = _rewrite_client()  # reused for images.generate (same host)
        try:
            await ensure_image_loaded()
            log.info("generating image (%dx%d): %s", side, side, prompt[:80])
            t0 = time.monotonic()
            resp = await asyncio.wait_for(
                client.images.generate(
                    model=IMAGE_MODEL,
                    prompt=prompt,
                    n=1,
                    size=f"{side}x{side}",  # SDK maps this to width/height
                    response_format="b64_json",
                ),
                timeout=IMAGE_GEN_TIMEOUT_S,
            )
            b64 = resp.data[0].b64_json
            if not b64:
                raise ImageGenError("backend returned no image data")
            png = base64.b64decode(b64)
            log.info("image generated in %.1f s (%d bytes)", time.monotonic() - t0, len(png))
            return png
        finally:
            # ALWAYS swap back, even on failure — never leave the image model
            # resident (it would keep evicting the chat model).
            await _unload_image()
            asyncio.get_running_loop().create_task(_warm_chat_reload())
