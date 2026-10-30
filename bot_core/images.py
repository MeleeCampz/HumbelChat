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
import time
from pathlib import Path

from typing import Any, Callable

import httpx
from openai import APIStatusError, AsyncOpenAI

from bot_core.agent_loop import looks_like_tool_call_fragment, run_agent_loop
from bot_core.ai_client import _create_with_reasoning_fallback
from bot_core.errors import extract_reply_text
from config.settings import (
    DEFAULT_MODEL,
    IMAGE_DEFAULT_SIZE,
    IMAGE_GEN_TIMEOUT_S,
    IMAGE_NEGATIVE_PROMPT,
    IMAGE_GGUF_FILENAME,
    IMAGE_GPU_IDS,
    IMAGE_LOAD_TIMEOUT_S,
    IMAGE_MAX_SIZE,
    IMAGE_MODEL,
    INFER_API_KEY,
    KB_PATH,
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
    """Load the configured image model if not resident (evicts chat by design).

    Hardened 2026-10-27 after back-to-back generation runs: Studio's diffusion
    slot bookkeeping can be mid-load when we ask (409 'A diffusion load is
    already in progress' — our own swap-back racing the next request) or a
    load can STALL without ever reaching ready. Both observed live. Handling:
    409 → just poll for ready; stalled attempt → unload to clear the stuck
    state and retry once (speed is not a concern — reliability is).
    """
    last_err = ""
    for attempt in (1, 2):
        async with httpx.AsyncClient(timeout=30.0) as client:
            st = await _image_status(client)
            if st.get("loaded"):
                return
            log.info("loading image model %s (attempt %d, evicts chat model)",
                     IMAGE_MODEL, attempt)
            try:
                await _studio(client, "POST", "/api/inference/images/load", json={
                    "model_path": IMAGE_MODEL,
                    "model_kind": "gguf",
                    "gguf_filename": IMAGE_GGUF_FILENAME,
                    "gpu_ids": IMAGE_GPU_IDS,
                })
            except ImageGenError as e:
                # 409: a load is already in progress (slot bookkeeping) — the
                # model will become ready; just fall through to polling.
                if "HTTP 409" not in str(e):
                    raise
                log.warning("image load already in progress — polling for ready")
            deadline = time.monotonic() + IMAGE_LOAD_TIMEOUT_S
            while time.monotonic() < deadline:
                await asyncio.sleep(10)
                fresh = await _image_status(client)
                if fresh.get("loaded"):
                    log.info("image model loaded (attempt %d, resident=%s MiB)",
                             attempt, fresh.get("resident_mib"))
                    return
            last_err = f"attempt {attempt}: not ready within {IMAGE_LOAD_TIMEOUT_S:.0f}s"
        # Not ready in time — clear whatever stuck state Studio holds and retry.
        log.warning("image load %s — unloading stuck state and retrying", last_err)
        await _unload_image()
        await asyncio.sleep(15)
    raise ImageGenError(f"image model did not load ({last_err})")


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
    auto-loads the chat model on the bot's next completion request.

    Serialized on the generation lock (2026-10-27, observed live): a warm
    chat-load running CONCURRENTLY with the next image load stalls both for
    minutes — every back-to-back generation idled ~300 s on a stuck image
    load while the previous generation's warm reload was still in flight.
    Two concurrent Studio model loads are the trap; serialized, they're just
    a no-op (chat already resident) or a clean sequential swap. If another
    generation is queued this simply waits behind it — and by then the image
    path has evicted chat again anyway, so skipping the redundant load is
    correct.
    """
    if not DEFAULT_MODEL:
        return
    # Studio's /api/inference/load takes a plain HF repo id plus an optional
    # gguf_variant — it does NOT accept the bot's "repo:quant" model slug.
    # Sending the slug verbatim fails 100% (HF repo-id validation rejects the
    # colon; observed live 2026-10-29: deferred 500 "Repo id must use
    # alphanumeric chars...", chat left evicted until a completion request
    # auto-loaded it). Split at the first ':' into repo + variant.
    repo, _, variant = DEFAULT_MODEL.partition(":")
    async with _gen_lock:
        try:
            # Cheap pre-check: don't even POST when chat is already resident
            # (the common case right after a generation that failed before
            # evicting, or a double-scheduled reload). Match on the repo id —
            # Studio reports loaded models as "owner/repo" without the quant,
            # and verify the quant too when one is requested.
            async with httpx.AsyncClient(timeout=30.0) as client:
                loaded = (await _studio(client, "GET", "/api/inference/loaded-models")).get("data", [])
            if any(
                m.get("id") == repo
                and (not variant or m.get("quant") in (variant, None))
                for m in loaded
            ):
                log.debug("chat model already resident — skipping warm reload")
                return
            payload: dict[str, str] = {"model_path": repo}
            if variant:
                payload["gguf_variant"] = variant
            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=IMAGE_LOAD_TIMEOUT_S) as client:
                await _studio(client, "POST", "/api/inference/load", json=payload)
            log.info("chat model re-loaded in background (%.1f s): %s",
                     time.monotonic() - t0, DEFAULT_MODEL)
        except Exception as e:  # noqa: BLE001 — best-effort by design
            log.warning("background chat re-load failed (harmless): %s", e)


# ── Prompt rewriting (chat model — MUST run before the swap) ──────────────
# ── Rewrite policy (layered: repo file + vault character rules) ────────────
# The prompt-engineering policy is NOT hardcoded here — it lives in two
# editable places so rule changes never need a code edit:
#   1. prompts/image_rewrite_system.md (repo, ships in the image) — world,
#      style, template, budgets, generic rules.
#   2. the vault note HumbleWood/Image Generation Policy.md —
#      CHARACTER-specific rules (googly eyes, species hotfixes, pronouns),
#      editable directly in Obsidian; fills the {{CHARACTER_RULES}} slot.
# Both are re-read on EVERY rewrite call, so edits apply to the next image
# without a restart. Missing files degrade to safe built-ins (logged).

_POLICY_FILE = Path(__file__).resolve().parent.parent / "prompts" \
    / "image_rewrite_system.md"
_CHAR_RULES_FILE = Path(KB_PATH) / "humblewood" / "HumbleWood" \
    / "Image Generation Policy.md"

# Safety net ONLY — the vault note is the source of truth for character rules
# (per-character facts live on the sheets themselves; only cross-cutting rules
# belong in the note, and its googly-eyes rule is mirrored here).
_DEFAULT_CHARACTER_RULES = (
    "Googly eyes are NOT a species trait and belong ONLY to Column Bidae: "
    "never give any other character googly eyes, and if he is not in the "
    "image do not mention googly eyes at all."
)

# Last-resort policy if the repo file is missing from the image entirely.
_FALLBACK_POLICY = (
    "You are a prompt engineer for a semi-realistic fantasy image model set "
    "in Humblewood (D&D 5e world). Rewrite the user's request into ONE rich "
    "prompt: [STYLE] base style 'Semi-realistic fantasy illustration — "
    "detailed fur and feather texture, natural light, storybook charm'; "
    "[SCENE] setting/action ≤45 words; [CHARACTERS] one line per character, "
    "D&D player characters (bipedal animal adventurers, no humans), reuse "
    "each character's Visual Anchor verbatim when their sheet has one, at "
    "most 5 characters; [LIGHTING] ≤15 words. Total under 1300 characters "
    "(2000 for 3+ characters). No negations in the prompt; include user-"
    "requested text VERBATIM.\n"
    "Character rules: {{CHARACTER_RULES}}\n"
    "Ground campaign specifics with the knowledge-base tools before writing."
)


def _load_rewrite_system() -> str:
    """Assemble the rewrite system prompt (repo policy + vault char rules)."""
    try:
        raw = _POLICY_FILE.read_text(encoding="utf-8")
        # The file starts with a human-facing header (how the layering works);
        # only the policy sections themselves belong in the system prompt.
        policy = raw.split("## World", 1)[1] if "## World" in raw else raw
    except OSError as exc:
        log.error("rewrite policy file missing (%s) — using built-in fallback",
                  exc)
        policy = _FALLBACK_POLICY
    try:
        char_rules = _CHAR_RULES_FILE.read_text(encoding="utf-8").strip()
        if not char_rules:
            raise OSError("empty")
    except OSError as exc:
        log.warning("character-rules vault note missing (%s) — using built-in "
                    "default", exc)
        char_rules = _DEFAULT_CHARACTER_RULES
    return policy.replace("{{CHARACTER_RULES}}", char_rules).strip()





def _rewrite_client() -> AsyncOpenAI:
    # Same host/key as chat (bot_core.ai_client's singleton is private; a small
    # dedicated client keeps this module independently testable).
    return AsyncOpenAI(api_key=INFER_API_KEY, base_url=INFER_URL, max_retries=0)


async def rewrite_prompt(text: str) -> str:
    """Expand a short user request into a rich image prompt via the chat model.

    Runs on the resident chat model — call this BEFORE ensure_image_loaded().
    The rewrite goes through the SAME bounded tool-calling loop as normal chat
    (bot_core.agent_loop) with the SAME knowledge-base tools, so campaign
    specifics in the request (characters, companions, places, items) are looked
    up and grounded instead of invented. Speed is not a concern — reliability
    is: the loop NEVER raises; every failure degrades to a single-turn answer
    over whatever context was gathered.
    Raises ImageGenError on failure (caller falls back to the raw prompt).
    """
    client = _rewrite_client()

    async def _complete(
        messages: list[dict], model: str, *, thinking: bool,
    ) -> str:
        # Decision turns: reasoning OFF (fast, no leak); the final turn gets a
        # LOW effort — the chat model is a reasoning model and at default effort
        # its thinking budget can eat the token cap and leave content empty
        # (observed live). Same param plumbing + 4xx fallback as the main AI
        # path (bot_core.ai_client, #20).
        resp = await asyncio.wait_for(
            _create_with_reasoning_fallback(
                client,
                "low" if thinking else "off",
                model=model,
                messages=list(messages),
                # Multi-character prompts run up to ~2000 chars (~500+ tokens)
                # and the chat model is a reasoning model — at 2048 its thinking
                # budget could eat the cap and truncate content mid-sentence
                # (observed live: party prompt died at 671 chars). 4096 leaves
                # headroom for both.
                max_tokens=4096,
            ),
            timeout=120.0,
        )
        return extract_reply_text(resp, default="").strip()

    log.info("prompt rewrite started: %s", text[:120])
    try:
        out, meta = await run_agent_loop(
            complete=_complete,
            model=DEFAULT_MODEL,
            system_prompt=_load_rewrite_system(),
            user_message=text,
        )
        out = (out or "").strip().strip('"')
        if not out:
            raise ImageGenError("empty rewrite")
        # The loop NEVER raises — on total backend failure it returns an
        # apology sentence instead of a prompt. Those sentinels must fall back
        # to the RAW user prompt, not be painted.
        if meta.fallback and any(
            s in out.lower()
            for s in ("unexpected error", "couldn't finish that lookup")
        ):
            raise ImageGenError(f"rewrite unusable ({meta.fallback_reason})")
        # Defense in depth: a mangled tool-call fragment is never a usable
        # image prompt (the loop already rejects these — but the fallback
        # single-turn path can still emit one, observed live 2026-10-28).
        if looks_like_tool_call_fragment(out):
            raise ImageGenError(
                "rewrite returned a tool-call fragment: " + out[:80].replace("\n", " "))
        log.info(
            "prompt rewrite done: %d iteration(s), tools=%s fallback=%s%s — %s",
            meta.iterations, ",".join(meta.tool_calls) or "-",
            meta.fallback,
            f" ({meta.fallback_reason})" if meta.fallback else "",
            out[:200] + ("…" if len(out) > 200 else ""),
        )
        return out
    except ImageGenError:
        raise
    except Exception as e:  # noqa: BLE001 — fall back to raw prompt in the caller
        raise ImageGenError(f"prompt rewrite failed: {e.__class__.__name__}") from e


# ── Generation (the no-chat-calls window) ─────────────────────────────────
_gen_lock = asyncio.Lock()  # one generation at a time (swap is global state)
_background_tasks: set[asyncio.Task] = set()  # strong refs — the loop only
                                               # keeps weak refs to tasks


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


async def prepare_prompt(text: str, raw: bool = False) -> str:
    """Final prompt for the image backend: rewrite via the chat model unless
    ``raw`` (verbatim passthrough).  A failed rewrite degrades to the raw
    text (never blocks a generation over a rewrite hiccup).

    Shared by the live path and the dry-run mode (#65b) so both see exactly
    what would be sent.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if raw:
        return text
    try:
        return await rewrite_prompt(text)
    except ImageGenError as exc:
        log.warning("prompt rewrite failed (%s) — using the raw prompt", exc)
        return text


async def _generate_one(client: AsyncOpenAI, prompt: str, side: int) -> bytes:
    """One ``images.generate`` call with one retry on transient 5xx.

    Observed live: Studio 'Image generation failed' (HTTP 500) mid-run,
    recovered on the next attempt. 4xx is a real config/prompt error →
    no retry.
    """
    # #66: full debug trail of exactly what goes to the backend —
    # dev.log runs at DEBUG so the complete prompt + negative prompt
    # are inspectable after the fact.
    log.debug("final image prompt (%d chars): %s", len(prompt), prompt)
    if IMAGE_NEGATIVE_PROMPT:
        log.debug("image negative prompt: %s", IMAGE_NEGATIVE_PROMPT)
    log.info("generating image (%dx%d): %s", side, side, prompt[:80])
    t0 = time.monotonic()
    for gen_attempt in (1, 2):
        try:
            resp = await asyncio.wait_for(
                client.images.generate(
                    model=IMAGE_MODEL,
                    prompt=prompt,
                    n=1,
                    size=f"{side}x{side}",  # SDK maps this to width/height
                    response_format="b64_json",
                    # Studio's /v1 spec accepts negative_prompt; this
                    # openai SDK version has no native kwarg for it →
                    # extra_body passthrough.
                    extra_body={"negative_prompt": IMAGE_NEGATIVE_PROMPT}
                    if IMAGE_NEGATIVE_PROMPT
                    else None,
                ),
                timeout=IMAGE_GEN_TIMEOUT_S,
            )
            b64 = resp.data[0].b64_json
            if not b64:
                raise ImageGenError("backend returned no image data")
            png = base64.b64decode(b64)
            log.info("image generated in %.1f s (%d bytes)",
                     time.monotonic() - t0, len(png))
            return png
        except (APIStatusError, ImageGenError) as e:
            transient = (
                isinstance(e, APIStatusError) and e.status_code is not None
                and e.status_code >= 500
            ) or isinstance(e, ImageGenError)
            if gen_attempt == 2 or not transient:
                raise
            log.warning("image generation attempt 1 failed (%s: %s) — retrying",
                       e.__class__.__name__, str(e)[:120])
            await asyncio.sleep(5.0)


async def _swap_back() -> None:
    """Swap the chat model back in — ALWAYS, even when a generation failed.

    Never leave the image model resident (it would keep evicting the chat
    model).  The warm reload runs as a background task with a strong
    reference: an unreferenced task can be GC'd before/while it runs
    (observed live 2026-10-29: zero warm-reload log lines across every
    generation despite the code path existing in all deployed builds).
    """
    await _unload_image()
    task = asyncio.get_running_loop().create_task(_warm_chat_reload())
    _background_tasks.add(task)


async def _emit_phase(on_phase: Callable[[str], Any] | None, name: str) -> None:
    """Await an optional sync-or-async phase callback; a callback error is
    logged and NEVER fails the generation (progress must not kill work)."""
    if on_phase is None:
        return
    try:
        res = on_phase(name)
        if asyncio.iscoroutine(res):
            await res
    except Exception as exc:  # noqa: BLE001 — progress must never kill the gen
        log.debug("on_phase(%s) failed: %s", name, exc)


async def generate_image(
    prompt: str,
    size: int | None = None,
    on_phase: Callable[[str], Any] | None = None,
) -> bytes:
    """Generate one image and return PNG bytes.

    Holds the generation lock for the whole load→generate→unload window; the
    CALLER is responsible for holding the global AI slot around this call so
    no other chat turn can re-resident the chat model mid-swap.

    ``on_phase`` (optional, #stall-visibility): awaited with ``"loading"``
    before the image-model load and ``"painting"`` once it is resident — so a
    caller can update the user while the GPU sits idle during the swap. A
    callback error is logged and never fails the generation.
    """
    side = size or IMAGE_DEFAULT_SIZE
    async with _gen_lock:
        client = _rewrite_client()  # reused for images.generate (same host)
        try:
            await _emit_phase(on_phase, "loading")
            await ensure_image_loaded()
            await _emit_phase(on_phase, "painting")
            return await _generate_one(client, prompt, side)
        finally:
            await _swap_back()


async def generate_images(
    prompts: list[str],
    size: int | None = None,
    on_phase: Callable[[str], Any] | None = None,
) -> list[bytes | ImageGenError]:
    """Batch variant of :func:`generate_image` (#69): paint every prompt in
    ONE model-swap window — load once, paint all while the image model is
    resident (a single failure does NOT abort the rest), unload + warm-
    reload the chat model exactly once.

    The caller holds the global AI slot around the whole call (same contract
    as :func:`generate_image`).  ``on_phase`` is awaited with ``"loading"``
    once, then ``"painting i/n"`` per image.  Returns one entry per prompt,
    in order: PNG bytes or the exception that killed it (:class:`ImageGenError`
    or a non-transient :class:`APIStatusError` — both are reported per image
    so ONE bad moment never aborts the rest of the batch).
    """
    side = size or IMAGE_DEFAULT_SIZE
    total = len(prompts)
    results: list[bytes | Exception] = []
    async with _gen_lock:
        client = _rewrite_client()
        try:
            await _emit_phase(on_phase, "loading")
            await ensure_image_loaded()
            for i, prompt in enumerate(prompts, 1):
                await _emit_phase(on_phase, f"painting {i}/{total}")
                try:
                    results.append(await _generate_one(client, prompt, side))
                except (ImageGenError, APIStatusError) as exc:
                    log.error("batch image %d/%d failed: %s", i, total, exc, exc_info=True)
                    results.append(exc)
        finally:
            await _swap_back()
    return results
