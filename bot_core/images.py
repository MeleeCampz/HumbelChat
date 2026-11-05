"""Image generation (#42) — Unsloth Studio diffusion via the OpenAI-compatible
``/v1/images/generations`` endpoint on the IMAGE backend (``IMAGE_INFER_URL``,
GPU-split 2026-11).

Architecture (GPU split, replaces the old swap orchestration):
- The image backend (Studio, 4070 Ti SUPER / GPU 0) is a SEPARATE server from
  chat (Strata on the 5070 / GPU 1). Chat never evicts the image model and
  image work never evicts chat — the two cards are independent.
- The 4070 hosts ONE mode at a time: "rag" (reranker-llama + embedder-llama,
  the default steady state) or "image" (the diffusion model resident, measured
  ~12.5 GiB at Q4_K_M). The bot switches to image mode right BEFORE painting
  and back to rag afterwards (finally, even on failure) via the gpu-switch
  helper (``GPU_SWITCH_URL``; scripts/gpu_switch_helper.py + gpu_switch.sh).
- **Prompt rewriting always runs FIRST, while rag mode is live** — the
  rewrite uses the KB agent tools (incl. the reranker), so it must not run
  mid-switch.
- ``IMAGE_MODE_HOLD_S`` briefly keeps image mode so back-to-back generations
  skip the double switch; ``_gen_lock`` serialises every switch/paint window.
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
    GPU_SWITCH_TIMEOUT_S,
    GPU_SWITCH_URL,
    IMAGE_DEFAULT_SIZE,
    IMAGE_GEN_TIMEOUT_S,
    IMAGE_GGUF_FILENAME,
    IMAGE_GPU_IDS,
    IMAGE_INFER_API_KEY,
    IMAGE_INFER_URL,
    IMAGE_LOAD_TIMEOUT_S,
    IMAGE_MAX_SIZE,
    IMAGE_MODE_HOLD_S,
    IMAGE_MODEL,
    IMAGE_NEGATIVE_PROMPT,
    INFER_API_KEY,
    INFER_URL,
    KB_PATH,
)

log = logging.getLogger("bot.bot_core.images")


class ImageGenError(Exception):
    """Image generation / GPU-mode orchestration failure (user-safe message)."""


def image_gen_configured() -> bool:
    return bool(IMAGE_MODEL)


# ── Studio management API base (IMAGE_INFER_URL is .../v1; mgmt endpoints
# live on the bare image-backend host) ────────────────────────────────────────────────────────────
_STUDIO_BASE = IMAGE_INFER_URL.removesuffix("/v1").removesuffix("/")
_STUDIO_HEADERS = {"Authorization": f"Bearer {IMAGE_INFER_API_KEY}"}


async def _studio(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> dict:
    resp = await client.request(method, f"{_STUDIO_BASE}{path}", headers=_STUDIO_HEADERS, **kwargs)
    if resp.status_code >= 400:
        raise ImageGenError(f"Studio {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


async def _image_status(client: httpx.AsyncClient) -> dict:
    return await _studio(client, "GET", "/api/inference/images/status")


async def _ensure_image_loaded() -> None:
    """Load the diffusion model on the image backend unless already resident.

    Hardened 2026-10-27 after back-to-back generation runs: Studio's diffusion
    slot bookkeeping can be mid-load when we ask (409 'A diffusion load is
    already in progress') or a
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
            log.info("loading image model %s (attempt %d)", IMAGE_MODEL, attempt)
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
        try:
            await _studio(client, "POST", "/api/inference/images/unload")
        except Exception as ue:  # noqa: BLE001 — best-effort state clear
            log.warning("stuck-state unload failed (retrying anyway): %s", ue)
        await asyncio.sleep(15)
    raise ImageGenError(f"image model did not load ({last_err})")


# ── GPU mode switching (rag ⇄ image on the 4070) ──────────────────────
# rag mode (default)  = reranker-llama + embedder-llama loaded on the 4070
# image mode          = the diffusion model resident on the 4070 (~12.5 GiB)
# The 5070 (chat/Strata) is NEVER touched by any of this. The helper service
# (scripts/gpu_switch_helper.py) wraps the compose stop/start + Studio
# images/load|unload calls; scripts/gpu_switch.sh is its manual CLI twin.
_mode_switched_at: float = 0.0   # end of the last painting window (mono time)
_hold_started_at: float = 0.0    # when the current deferred restore was scheduled


def _switcher_enabled() -> bool:
    return bool(GPU_SWITCH_URL)


def _stamp_painting_window() -> None:
    """Record the end of a painting window for the IMAGE_MODE_HOLD_S check."""
    global _mode_switched_at
    _mode_switched_at = time.monotonic()


_background_tasks: set[asyncio.Task] = set()  # strong refs — the loop only
                                              # keeps weak refs to tasks


def _spawn_restore_after_hold() -> None:
    """Fire-and-forget: switch back to rag mode once IMAGE_MODE_HOLD_S expires.

    Serialized on _gen_lock, so it can never interleave with a live painting
    window — a new generation that arrives first either wins (the hold keeps
    image mode for it) or queues behind this switch cleanly.
    """
    async def _later() -> None:
        try:
            await asyncio.sleep(IMAGE_MODE_HOLD_S)
            async with _gen_lock:
                await ensure_image_still_needed_or_restore()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — best-effort by design
            log.error("deferred rag restore failed: %s — reranker/embedder "
                      "may be down; run: scripts/gpu_switch.sh rag", e)

    task = asyncio.create_task(_later())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def ensure_image_still_needed_or_restore() -> None:
    """After the hold window: if another painting window already re-entered
    image mode more recently than the hold started, stay put; else restore."""
    global _hold_started_at
    if _hold_started_at and _mode_switched_at > _hold_started_at:
        log.info("image mode re-entered during the hold window — deferring "
                 "the switch-back again")
        _spawn_restore_after_hold()
        return
    await _switch_gpu_mode("rag")


async def _helper(client: httpx.AsyncClient, method: str, path: str,
                  **kwargs) -> dict:
    """Call the gpu-switch helper; raise ImageGenError on any HTTP failure."""
    try:
        resp = await client.request(method, f"{GPU_SWITCH_URL}{path}", **kwargs)
    except (httpx.HTTPError, OSError) as e:
        raise ImageGenError(f"gpu-switch helper unreachable at {GPU_SWITCH_URL}: {e}")
    if resp.status_code >= 400:
        raise ImageGenError(f"gpu-switch {path} -> HTTP {resp.status_code}: "
                                 f"{resp.text[:200]}")
    return resp.json()


async def _switch_gpu_mode(mode: str) -> None:
    """Ask the helper to put the 4070 into "image" or "rag" mode.

    Idempotent switch: the helper no-ops when already in the requested mode
    (its own state check), so a failed previous restore is healed by the
    next switch attempt. On failure into image mode: log loudly — the card
    stays in rag mode and RAG keeps working (no half-switch state).
    """
    if not _switcher_enabled():
        if mode == "image":
            raise ImageGenError(
                "GPU_SWITCH_URL not set — run scripts/gpu_switch.sh image "
                "on the host before generating images")
        return
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=GPU_SWITCH_TIMEOUT_S) as client:
        out = await _helper(client, "POST", "/mode", json={"mode": mode})
    log.info("gpu mode -> %s (%.1f s): %s", mode, time.monotonic() - t0,
             str(out.get("detail", out))[:160])


# ── Prompt rewriting (chat model — MUST run BEFORE the mode switch) ───────
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





def _image_client() -> AsyncOpenAI:
    """OpenAI client for the IMAGE backend (Studio; IMAGE_INFER_URL).

    bot_core.ai_client's singleton is private; a small dedicated client
    keeps this module independently testable.
    """
    return AsyncOpenAI(api_key=IMAGE_INFER_API_KEY,
                       base_url=IMAGE_INFER_URL, max_retries=0)


def _chat_client() -> AsyncOpenAI:
    """OpenAI client for the CHAT backend (prompt rewriting; INFER_URL)."""
    return AsyncOpenAI(api_key=INFER_API_KEY, base_url=INFER_URL, max_retries=0)


async def rewrite_prompt(text: str) -> str:
    """Expand a short user request into a rich image prompt via the chat model.

    Runs on the chat backend (its own card) while rag mode is live — call
    this BEFORE ensure_image_mode(): the rewrite's KB tools include the
    reranker, which a mode switch takes offline.
    The rewrite goes through the SAME bounded tool-calling loop as normal chat
    (bot_core.agent_loop) with the SAME knowledge-base tools, so campaign
    specifics in the request (characters, companions, places, items) are looked
    up and grounded instead of invented. Speed is not a concern — reliability
    is: the loop NEVER raises; every failure degrades to a single-turn answer
    over whatever context was gathered.
    Raises ImageGenError on failure (caller falls back to the raw prompt).
    """
    client = _chat_client()

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


# ── Generation (the image-mode window) ─────────────────────────────────────────
_gen_lock = asyncio.Lock()  # one generation at a time (mode switches +
                          # diffusion jobs are card-global state)


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


async def ensure_image_mode(on_phase: Callable[[str], Any] | None = None) -> None:
    """Put the 4070 into image mode with the diffusion model ready to paint.

    Cheap when already in image mode with the model loaded (status 200-OK:
    no switch, no load, no phase emitted — the back-to-back generation
    case). Otherwise: switch via the helper (stops reranker/embedder), then
    ensure the model is resident. Raises ImageGenError on failure — the
    card may still be mid-switch; the next generation's rag restore heals it.
    """
    need_load = True
    if _switcher_enabled():
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                mode = (await _helper(client, "GET", "/mode")).get("mode", "")
            except ImageGenError:
                mode = ""  # helper state unknown — ask for the switch anyway
        if mode == "image":
            async with httpx.AsyncClient(timeout=30.0) as client:
                need_load = not (await _image_status(client)).get("loaded")
        else:
            await _switch_gpu_mode("image")
    if need_load:
        await _emit_phase(on_phase, "loading")
        await _ensure_image_loaded()
    else:
        log.debug("image mode already live — no switch, no load")


async def _restore_rag_after_painting() -> None:
    """Switch the 4070 back to rag mode after painting — ALWAYS attempted,
    even when painting failed, so the reranker/embedder are never left down.

    IMAGE_MODE_HOLD_S: after a generation that SWITCHED into image mode the
    switch-back is skipped while the hold window is open, so a back-to-back
    generation finds image mode still live (GET /mode says "image", model
    resident → no switch, no load). The window is measured from the END of
    painting. Best-effort: never raises.
    """
    try:
        if IMAGE_MODE_HOLD_S > 0 and _switcher_enabled():
            global _hold_started_at
            _hold_started_at = time.monotonic()
            log.info("image mode held %.0f s after painting "
                     "(IMAGE_MODE_HOLD_S) — deferred switch-back scheduled",
                     IMAGE_MODE_HOLD_S)
            _spawn_restore_after_hold()
            return
        await _switch_gpu_mode("rag")
    except Exception as e:  # noqa: BLE001 — best-effort by design
        log.error("switch back to rag mode FAILED: %s — reranker/embedder "
                  "may be down; run: scripts/gpu_switch.sh rag", e)


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

    Holds the generation lock for the whole switch→paint→restore window.
    The CALLER must have prepared the prompt BEFORE this call (prompt
    rewriting needs rag mode live — its KB tools include the reranker).
    Chat is unaffected: it runs on a different backend/card throughout.

    ``on_phase`` (optional, #stall-visibility): awaited with ``"loading"``
    only when the mode switch / model load actually happens (a back-to-back
    generation in held image mode emits nothing until ``"painting"``). A
    callback error is logged and never fails the generation.
    """
    side = size or IMAGE_DEFAULT_SIZE
    async with _gen_lock:
        client = _image_client()
        try:
            await ensure_image_mode(on_phase)
            await _emit_phase(on_phase, "painting")
            png = await _generate_one(client, prompt, side)
            _stamp_painting_window()
            return png
        finally:
            await _restore_rag_after_painting()


async def generate_images(
    prompts: list[str],
    size: int | None = None,
    on_phase: Callable[[str], Any] | None = None,
) -> list[bytes | ImageGenError]:
    """Batch variant of :func:`generate_image` (#69): paint every prompt in
    ONE image-mode window — switch + load once, paint all while the image
    model is resident (a single failure does NOT abort the rest), switch
    back to rag mode exactly once.

    The caller must have PREPARED ALL PROMPTS before this call (moment
    selection + rewriting need rag mode live).  ``on_phase`` is awaited with
    ``"loading"`` only when an actual switch/load happens, then
    ``"painting i/n"`` per image.  Returns one entry per prompt, in order:
    PNG bytes or the exception that killed it (:class:`ImageGenError` or a
    non-transient :class:`APIStatusError` — both are reported per image so
    ONE bad moment never aborts the rest of the batch).
    """
    side = size or IMAGE_DEFAULT_SIZE
    total = len(prompts)
    results: list[bytes | Exception] = []
    async with _gen_lock:
        client = _image_client()
        try:
            await ensure_image_mode(on_phase)
            for i, prompt in enumerate(prompts, 1):
                await _emit_phase(on_phase, f"painting {i}/{total}")
                try:
                    results.append(await _generate_one(client, prompt, side))
                except (ImageGenError, APIStatusError) as exc:
                    log.error("batch image %d/%d failed: %s", i, total, exc, exc_info=True)
                    results.append(exc)
            _stamp_painting_window()
        finally:
            await _restore_rag_after_painting()
    return results
