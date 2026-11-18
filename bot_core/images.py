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
import re
import time
from collections import OrderedDict
from pathlib import Path

from typing import Any, Callable

import httpx
from openai import APIStatusError, AsyncOpenAI, OpenAIError

from bot_core import image_prompt
from bot_core.agent_loop import looks_like_tool_call_fragment, run_agent_loop
from bot_core.ai_client import _create_with_reasoning_fallback
from bot_core.errors import extract_reply_text
from config.settings import (
    DEFAULT_MODEL,
    GPU_SWITCH_TIMEOUT_S,
    GPU_HEAL_ON_START,
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
    IMAGE_PROMPT_BASE_CHARS,
    IMAGE_PROMPT_CHARS_PER_FIGURE,
    IMAGE_PROMPT_FOREGROUND_FIGURES,
    IMAGE_PROMPT_MAX_CHARS,
    IMAGE_REASONING_EFFORT,
    IMAGE_REWRITE_MAX_TOKENS,
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
    try:
        resp = await client.request(method, f"{_STUDIO_BASE}{path}", headers=_STUDIO_HEADERS, **kwargs)
    except (httpx.HTTPError, OSError) as e:
        # Connection-level failure (refused/timeout/reset) must surface as the
        # user-safe ImageGenError the command layer catches — not a raw httpx
        # traceback (the image backend being down is an operational state,
        # not a bug).
        raise ImageGenError(f"image backend unreachable at {_STUDIO_BASE}: {e}")
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
_last_bot_switch_at: float = 0.0 # end of the last window where the BOT switched the card
_hold_started_at: float = 0.0    # when the current deferred restore was scheduled
_bot_switched = False            # the current window performed a mode switch


def _switcher_enabled() -> bool:
    return bool(GPU_SWITCH_URL)


def _hold_active() -> bool:
    """Is an IMAGE_MODE_HOLD_S window currently open (armed by a bot switch and
    not yet expired)?"""
    return (_hold_started_at > 0.0
            and (time.monotonic() - _hold_started_at) < IMAGE_MODE_HOLD_S)


def _stamp_painting_window() -> None:
    """Record the end of a painting window for the IMAGE_MODE_HOLD_S check.

    Always stamp the window end (the defer check in
    :func:`ensure_image_still_needed_or_restore` uses it to spot generations
    that painted during a hold). Separately stamp the last BOT-switched
    window — only those earn an IMAGE_MODE_HOLD_S hold at all: a window where
    someone else (manual gpu_switch.sh) had already put the card in image mode
    gets no hold, so the bot switches straight back to rag after it.
    """
    global _mode_switched_at, _last_bot_switch_at
    now = time.monotonic()
    _mode_switched_at = now
    if _bot_switched:
        _last_bot_switch_at = now


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


async def heal_gpu_mode_on_start() -> None:
    """Bot-startup heal: if the switcher is configured, make sure the 4070 is
    in rag mode.

    rag mode is the bot's steady state; image mode only ever exists around a
    painting window. A crash / restart between switch and restore (or a hold
    task killed with the loop) would otherwise leave the reranker/embedder
    stopped for the whole session with no trigger to bring them back.
    Idempotent and cheap when already rag (helper no-ops). Never raises.
    """
    if not (GPU_HEAL_ON_START and _switcher_enabled()):
        return
    try:
        async with _gen_lock:  # serialize with painting windows like every
            await _switch_gpu_mode("rag")  # other mode mutation
    except Exception as e:  # noqa: BLE001 — startup must not fail on this
        log.warning("startup GPU heal failed (continuing; run "
                    "scripts/gpu_switch.sh rag if RAG stays degraded): %s", e)


async def ensure_image_still_needed_or_restore() -> None:
    """After the hold window: if another painting window already re-entered
    image mode more recently than the hold started, stay put; else restore.

    On re-defer the hold anchor MOVES to now — without that, every later
    painting window would always look "re-entered" (its stamp is newer than
    the original arm time) and the restore would defer itself forever,
    leaving image mode stuck on permanently.
    """
    global _hold_started_at
    if _hold_started_at and _mode_switched_at > _hold_started_at:
        log.info("image mode re-entered during the hold window — deferring "
                 "the switch-back again")
        _hold_started_at = time.monotonic()
        _spawn_restore_after_hold()
        return
    _hold_started_at = 0.0
    await _switch_gpu_mode("rag")


async def _helper(client: httpx.AsyncClient, method: str, path: str,
                  **kwargs) -> dict:
    """Call the gpu-switch helper; raise ImageGenError on any HTTP failure."""
    try:
        resp = await client.request(method, f"{GPU_SWITCH_URL}{path}", **kwargs)
    except (httpx.HTTPError, OSError) as e:
        raise ImageGenError(
            f"gpu-switch helper unreachable at {GPU_SWITCH_URL}: {e} "
            "(start it with `docker compose up -d gpu-switch`, or unset "
            "GPU_SWITCH_URL for manual mode via scripts/gpu_switch.sh)")
    if resp.status_code >= 400:
        raise ImageGenError(f"gpu-switch {path} -> HTTP {resp.status_code}: "
                                 f"{resp.text[:200]}")
    return resp.json()


def _notify_retrievers_gpu_mode(mode: str) -> None:
    """Tell kb.retrievers which mode the switched RAG card is in.

    Review 2026-11 (R2b): during an image window the reranker/embedder
    sidecars are STOPPED — retrieval should skip the dense leg outright
    (lexical-only) instead of discovering that via failing ladders every
    turn. Best-effort: a notification failure never affects generation.
    """
    try:
        from kb.retrievers import set_gpu_mode
        set_gpu_mode(mode)
    except Exception as e:  # noqa: BLE001 — notification must never break the switch
        log.debug("could not notify retrievers of gpu mode %r: %s", mode, e)


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
    detail = str(out.get("detail", out))
    if "refused" in detail:
        # The helper deliberately kept the card in rag mode because the image
        # backend is unreachable — do NOT proceed to paint against a card
        # that never got the diffusion slot.
        raise ImageGenError(f"gpu-switch refused the switch to {mode}: {detail[:200]}")
    global _bot_switched
    if mode == "image":
        _bot_switched = True
    log.info("gpu mode -> %s (%.1f s): %s", mode, time.monotonic() - t0, detail[:160])
    _notify_retrievers_gpu_mode(mode)


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
    "Googly eyes are NOT a species trait and belong ONLY to Colum Bidae: "
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
    "Party dossier: {{PARTY_DOSSIER}}\n"
    "Ground campaign specifics with the knowledge-base tools before writing."
)

# The party sheet is a THIRD editable layer: the party itself (names, species,
# pronouns, Visual Anchors) is read straight from the vault on every rewrite,
# so a group request no longer depends on the model successfully retrieving it.
# Until this existed (2026-11-18) group rewrites were pure retrieval gambles:
# when the sheet lookup failed the model invented party members outright —
# "die ganze Gruppe" produced a badger named Bramble and an antlered elf named
# Elara, Colum as a raven and Trixy as a tabby cat, and dropped Seta, Bubo and
# Romi entirely.
_PARTY_SHEET_FILE = Path(KB_PATH) / "humblewood" / "HumbleWood" / "Characters" \
    / "Player Characters.md"

# The base style sentence the policy makes the model open with VERBATIM. Used
# as the structural check that a rewrite is a PROMPT and not narration — and
# kept in sync with prompts/image_rewrite_system.md by a test.
_BASE_STYLE = ("Semi-realistic fantasy illustration — detailed fur and feather "
               "texture, natural light, storybook charm")

_MAX_ANCHOR_CHARS = 900   # a Visual Anchor runs ~550; a runaway quote must not
# monopolise the system prompt (it is truncated with a warning instead)

_NO_DOSSIER = ("(No party dossier available — look the campaign characters up "
               "with the knowledge-base tools.)")


def _parse_party_sheet(raw: str) -> list[tuple[str, str, str]]:
    """(name, species/pronoun line, Visual Anchor) per character block.

    Sheet shape (the convention the sheets follow):
      `# Name` → a free line (species, class, pronouns) → portraits →
      `## Appearance` → `## Visual Anchor …` with the anchor as a `>` quote.
    Characters without an anchor are skipped: without one there is nothing to
    inject, and the model's own no-anchor creature rules apply to them.
    """
    out: list[tuple[str, str, str]] = []
    name = desc = anchor = ""
    expect_anchor = False
    in_fence = False

    def flush() -> None:
        if name and anchor:
            out.append((name, desc, anchor))

    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue    # a '# ' inside a stat block is not a character heading
        if s.startswith("# "):
            flush()
            name, desc, anchor, expect_anchor = s[2:].strip(), "", "", False
        elif name and not desc and s and not s.startswith(("#", ">", "!", "[")):
            desc = s
        elif s.startswith("## "):
            expect_anchor = "visual anchor" in s.lower()
        elif expect_anchor and s.startswith("> "):
            anchor = s[2:].strip()
            expect_anchor = False
    flush()
    return out


def _load_party_dossier() -> str:
    """The party roster + every Visual Anchor, re-read on EVERY rewrite."""
    try:
        raw = _PARTY_SHEET_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("party sheet unreadable (%s) — no party dossier injected", exc)
        return ""
    entries = _parse_party_sheet(raw)
    if not entries:
        log.warning("party sheet has no parsable 'Visual Anchor' quote — "
                    "no party dossier injected")
        return ""
    lines = [
        "The player characters, read from the vault right now. Each Visual Anchor",
        "is given in FULL: when that character is in the image, copy their anchor",
        "VERBATIM from here — do not rephrase it, do not shorten it, and do not use",
        "a tool call to look it up. Anyone not listed is NOT a party member.",
        "",
        "Everything ELSE the request names (an NPC, a merchant, a bandit, a",
        "monster, a place, an item) has no entry here and MUST be looked up in the",
        "knowledge base BEFORE you write it: the session logs carry 'Lexikon:",
        "Personen' rosters that name recurring NPCs and state their species, colours",
        "and gear, and monsters come from the rules. NEVER assign a species or a",
        "colour to a named NPC from imagination. Batch it: emit every lookup you",
        "need as ONE array of tool calls in a SINGLE decision turn — the loop only",
        "has a few turns, so one search per turn runs out of budget.",
    ]
    for name, desc, anchor in entries:
        a = anchor
        if len(a) > _MAX_ANCHOR_CHARS:
            a = a[:_MAX_ANCHOR_CHARS].rsplit(" ", 1)[0] + "…"
            log.warning("Visual Anchor of %s exceeds %d chars — truncated in the "
                        "dossier", name, _MAX_ANCHOR_CHARS)
        lines.append(f"- {name}" + (f" — {desc}" if desc else "")
                     + f"\n  Visual Anchor: {a}")
    return "\n".join(lines)


def _norm_prompt(text: str) -> str:
    """Compare prompts past the model's punctuation habits.

    Live outputs carry stray spaces ('storybook charm . Seta stands…'), so the
    check compares letters/digits only.
    """
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def looks_like_image_prompt(text: str) -> bool:
    """False for narration: the policy makes a real prompt OPEN with the base
    style sentence, so its absence means the model answered with something
    else — a plan, a comment, an apology. Observed live 2026-11-18: a group
    request returned "I need to look up the Player Characters sheet…" and that
    text would otherwise have been PAINTED.

    A paraphrased-but-real prompt still passes on the distinctive opening
    phrase, so the guard rejects narration rather than wording drift.
    """
    n = _norm_prompt(text)
    if not n:
        return False
    if n.startswith(_norm_prompt(_BASE_STYLE)):
        return True
    head = _norm_prompt(_BASE_STYLE.split(" — ")[0])
    return bool(head) and head in n[:220]


# One corrective retry: cheap (the narration cases answered in ~3 s with no
# tool calls) and it recovers the reply instead of dropping to the raw request.
_RETRY_NUDGE = (
    "Your previous answer was not an image prompt — it was a plan or a comment. "
    "The party and their Visual Anchors are already in your system prompt, so no "
    "lookup is needed. Write the final prompt NOW: open with the base style "
    "sentence VERBATIM, then scene, then one line per character, then lighting. "
    "Reply with the prompt only.\n\nRequest: "
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
        char_rules = _strip_editor_preamble(
            _CHAR_RULES_FILE.read_text(encoding="utf-8"))
        if not char_rules:
            raise OSError("empty")
    except OSError as exc:
        log.warning("character-rules vault note missing (%s) — using built-in "
                    "default", exc)
        char_rules = _DEFAULT_CHARACTER_RULES
    return (policy
            .replace("{{CHARACTER_RULES}}", char_rules)
            .replace("{{PARTY_DOSSIER}}", _load_party_dossier() or _NO_DOSSIER)
            .strip())





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


def _make_completer(client: AsyncOpenAI):
    """The completion callable the agent loop drives (shared by both rewrite
    paths — the plan path and the older prose path)."""

    async def _complete(
        messages: list[dict], model: str, *, thinking: bool,
    ) -> str:
        # The loop's `thinking` flag used to gate the effort here, and that made
        # the setting inert: measured live, a plan answer arrives out of a
        # DECISION turn (thinking=False), so a whole rewrite ran reasoning-off
        # and the configured level never reached a single request. The effort is
        # now applied to every rewrite turn — decision, answer, lookup and
        # fallback alike — which is what the §12 measurement actually varied.
        # Same param plumbing + 4xx fallback as the main AI path
        # (bot_core.ai_client, #20).
        resp = await asyncio.wait_for(
            _create_with_reasoning_fallback(
                client,
                IMAGE_REASONING_EFFORT,
                model=model,
                messages=list(messages),
                # IMAGE_REWRITE_MAX_TOKENS grows with the reasoning effort: a
                # plan is short (~600 chars) but the loop's forced-answer turn
                # repeats the gathered context, the older prose path writes
                # ~2000-char prompts, and any of those can be eaten by the
                # thinking budget (observed live: a party prompt died at 671
                # chars at a 2048 cap).
                max_tokens=IMAGE_REWRITE_MAX_TOKENS,
            ),
            timeout=120.0,
        )
        return extract_reply_text(resp, default="").strip()

    return _complete


# ── The plan path (#84): the model plans, the composer writes the prompt ──

_PLAN_FILE = Path(__file__).resolve().parent.parent / "prompts" \
    / "image_plan_system.md"

_NO_ROSTER = ("(No party roster available — look EVERY character up with the "
              "knowledge-base tools and give each one a `look` and a `source`.)")

# One corrective retry, same reasoning as _RETRY_NUDGE below: the failure cases
# answered in ~3 s with no tool calls, so a retry is cheap and recovers the
# answer instead of dropping to the raw request.
_LOOKUP_NUDGE = (
    "These named characters are not in the party sheet, and you have not "
    "retrieved their appearance from the vault: {{NAMES}}.\n"
    "Search the vault for each one NOW — one search per character, all in a "
    "single batch — and open the section that names them if a search found "
    "one. Then output the WHOLE plan again (scene sentence, every figure, the "
    "count) with their `look` written from what you retrieved and `source` "
    "citing the exact \"file\" value it came from. If the vault holds no "
    "appearance for one, leave its `look` out and name it in the scene "
    "instead: an appearance you made up is worse than none.")

_PLAN_RETRY_NUDGE = (
    "Your previous answer had no <image_plan> block, so nothing could be "
    "painted. The party roster and its appearance fields are ALREADY in your "
    "system prompt — no lookup is needed for them. Answer NOW with the "
    "<image_plan> block alone: scene (naming individuals, never a crowd noun), "
    "figures (name + pose, plus `look`/`source` for anyone off the roster), "
    "entities, lighting, palette, mood.\n\nRequest: "
)

# How the answer must look, said by CODE. The shared chat-path addendum used to
# be appended verbatim here and it ended "answer in plain text (no JSON)" —
# measured sitting directly under a contract that demands a JSON block (analysis
# §15), which is the shape of the narration-instead-of-plan failures
# _PLAN_RETRY_NUDGE exists to mop up.
_IMAGE_ANSWER_INSTRUCTION = (
    "Answering: your final answer must be the <image_plan> block of the "
    "contract and nothing else — not prose, not lookup findings, not the image "
    "described in sentences. A reply without that block cannot be painted and is "
    "discarded. On a lookup turn, reply only with the JSON tool call.")


def _strip_editor_preamble(text: str) -> str:
    """Drop the vault note's header, which is written for the human editing it.

    Image Generation Policy.md opens with 610 characters addressed to the vault's
    owner — 'This note is loaded into the system prompt… Edit freely — changes
    apply to the next image… Write direct instructions for the model; keep it
    short.' The prose path already skipped its file's header; the plan path was
    injecting this one whole, telling the model to edit a markdown file.
    """
    i = text.find("\n## ")
    return text[i + 1:].strip() if i >= 0 else text.strip()


def _wrap_request(text: str) -> str:
    """The request as a labelled block, the way ASK_AI sends its context blocks.

    A bare sentence under ~13k characters of rules is how a request gets
    paraphrased instead of answered; ASK_AI gained the same benefit from tagging
    its context ("<history_context> … do NOT re-answer these").
    """
    return f"<request>\n{text}\n</request>"


# What the LAST image in this channel was. A follow-up ("nochmal mit Fackeln",
# "ohne die Banditen") is an edit of a frame, but the rewrite was amnesiac: it
# re-planned from scratch and re-searched the same NPCs. In-memory and bounded —
# a restart forgets, which is the right failure mode for a stale frame.
_PLAN_MEMORY: "OrderedDict[Any, str]" = OrderedDict()
_PLAN_MEMORY_MAX = 12
_PREVIOUS_PLAN_MAX_CHARS = 700


def _plan_note(plan: image_prompt.ImagePlan) -> str:
    """A plan rendered small enough to be worth re-reading every later turn."""
    parts = [f"scene: {plan.scene}"]
    for f in plan.figures:
        pose = str(getattr(f, "pose", "") or "").strip()
        parts.append(f"- {f.name}" + (f": {pose}" if pose else ""))
    if plan.entities:
        parts.append("entities: " + ", ".join(plan.entities))
    for label, value in (("lighting", plan.lighting), ("palette", plan.palette),
                         ("mood", plan.mood)):
        if value:
            parts.append(f"{label}: {value}")
    note = "\n".join(parts)
    return note if len(note) <= _PREVIOUS_PLAN_MAX_CHARS \
        else note[:_PREVIOUS_PLAN_MAX_CHARS].rstrip() + " …"


def _remember_plan(key: Any, plan: image_prompt.ImagePlan) -> None:
    if key is None:
        return
    _PLAN_MEMORY[key] = _plan_note(plan)
    _PLAN_MEMORY.move_to_end(key)
    while len(_PLAN_MEMORY) > _PLAN_MEMORY_MAX:
        _PLAN_MEMORY.popitem(last=False)


def _previous_plan_block(key: Any) -> str:
    note = _PLAN_MEMORY.get(key) if key is not None else None
    if not note:
        return ""
    return ("<previous_image_plan>\n"
            "The plan of the LAST image painted in this channel. The new request "
            "may be an edit of it (\"again, but with torches\", \"without the "
            "bandits\"): keep what still applies and re-use the same characters. "
            "If the new request is unrelated, ignore this block entirely.\n"
            f"{note}\n</previous_image_plan>")


def _load_anchors() -> dict[str, image_prompt.Anchor]:
    """Party anchors, re-read from the vault on EVERY rewrite.

    Re-reading is deliberate: a Visual Anchor the user edits in Obsidian must
    take effect on the next image, not the next bot restart.
    """
    try:
        raw = _PARTY_SHEET_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("party sheet unreadable (%s) — no anchors for the composer", exc)
        return {}
    anchors = {a.name: a for a in image_prompt.parse_party_sheet(raw)}
    untyped = [a.name for a in anchors.values() if not a.typed]
    if untyped:
        log.info("party anchors still in the legacy single-paragraph form for: %s "
                 "(painted verbatim; field overrides fall back to clause "
                 "filtering)", ", ".join(untyped))
    return anchors


def _load_roster_block() -> str:
    """The roster as the plan path needs it: names + a PREVIEW of each field.

    The model must know which field a request is asking to change ("her cloak" →
    `outfit`), but it no longer copies these values — the composer paints them in
    full. A 110-character preview per field is enough to aim an override and keeps
    the system prompt small: the rewrite path used to inject 12,564 characters, of
    which the model had to reproduce 2,757.
    """
    anchors = _load_anchors()
    if not anchors:
        return _NO_ROSTER
    lines = [f"fields you may override: {', '.join(image_prompt.ANCHOR_FIELDS)} "
             "(outfit = clothing, signature = carried/built-in items)", ""]
    for a in anchors.values():
        lines.append(f"- {a.name}" + (f" — {a.desc}" if a.desc else ""))
        if a.typed:
            for f in image_prompt.ANCHOR_FIELDS:
                v = a.fields.get(f, "")
                if not v:
                    continue
                preview = v if len(v) <= 110 else v[:110].rsplit(" ", 1)[0] + "…"
                lines.append(f"    {f}: {preview}")
        else:
            preview = a.blob if len(a.blob) <= 200 else a.blob[:200] + "…"
            lines.append(f"    appearance (one block, painted verbatim; an "
                         f"override replaces the whole look): {preview}")
        for x in a.extra:
            lines.append(f"    extra: {x}")
    return "\n".join(lines)


def _load_plan_system() -> str:
    """Assemble the plan-contract system prompt (repo contract + vault layers)."""
    try:
        raw = _PLAN_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        raise ImageGenError(f"plan contract file missing ({exc})") from exc
    try:
        char_rules = _strip_editor_preamble(
            _CHAR_RULES_FILE.read_text(encoding="utf-8"))
        if not char_rules:
            raise OSError("empty")
    except OSError as exc:
        log.warning("character-rules vault note missing (%s) — using built-in "
                    "default", exc)
        char_rules = _DEFAULT_CHARACTER_RULES
    return (raw.replace("{{CHARACTER_RULES}}", char_rules)
             .replace("{{PARTY_ROSTER}}", _load_roster_block())
             .strip())


_READ_ARG_KEYS = ("path", "paths", "relPath", "relPaths", "file", "files",
                  "folder", "dir")
_FILE_IN_RESULTS = re.compile(r'"(?:file|relPath|path)"\s*:\s*"([^"]+\.(?:md|txt))"')


def _read_paths_from(detail: list[dict]) -> list[str]:
    """Files this rewrite OPENED (kb_read_section and friends).

    `read_section` is called with a folder name rather than a vault path, which
    is why `source_matches` compares path segments instead of whole strings.
    """
    out: list[str] = []
    for d in detail or []:
        args = d.get("args")
        if not isinstance(args, dict):
            continue
        for key in _READ_ARG_KEYS:
            v = args.get(key)
            if isinstance(v, str) and v.strip():
                out.append(v)
            elif isinstance(v, (list, tuple)):
                out.extend(x for x in v if isinstance(x, str) and x.strip())
    return out


def _snippet_paths_from(gathered: str, opened: Sequence[str]) -> list[str]:
    """Files that came back as search HITS — seen, but only as a snippet.

    Every kb_search hit carries its file, so a description taken from a hit is
    grounded in the vault; what it is not is fully read. The composer accepts
    such a `look` only when its wording is traceable to the retrieved text
    (image_prompt.verify_look), which is the difference between reading a note
    and pattern-matching a headline.
    """
    opened_last = {o.strip().rstrip("/").rsplit("/", 1)[-1].lower()
                   for o in opened if o.strip()}
    out: list[str] = []
    for hit in _FILE_IN_RESULTS.findall(gathered or ""):
        last = hit.rsplit("/", 1)[-1].lower()
        if any(last == o or last in o or o in last for o in opened_last):
            continue                      # counted as a read already
        out.append(hit)
    return out


def _budget_kwargs() -> dict[str, int]:
    return {"base": IMAGE_PROMPT_BASE_CHARS,
            "per_figure": IMAGE_PROMPT_CHARS_PER_FIGURE,
            "hard_cap": IMAGE_PROMPT_MAX_CHARS}


async def _rewrite_composed(text: str, complete,
                            memory_key: Any = None) -> str:
    """Plan the image with the chat model, then COMPOSE the prompt in code.

    Raises ImageGenError when no usable plan comes back; rewrite_prompt then
    falls back to the older prose path.
    """
    system = _load_plan_system()
    anchors = _load_anchors()
    prior = _previous_plan_block(memory_key)
    base_msg = _wrap_request(text)
    first_msg = f"{prior}\n\n{base_msg}" if prior else base_msg
    log.info("image plan requested: %s%s", text[:120],
             " (+previous plan injected)" if prior else "")
    for i, msg in enumerate((first_msg, _PLAN_RETRY_NUDGE + base_msg)):
        out, meta = await run_agent_loop(
            complete=complete,
            model=DEFAULT_MODEL,
            system_prompt=system,
            user_message=msg,
            answer_instruction=_IMAGE_ANSWER_INSTRUCTION,
            # No KB context is attached on this path — the paragraph telling the
            # model not to re-verify "what is right there" only ever has the
            # roster PREVIEW in view, and it discourages exactly the lookups a
            # missing NPC appearance needs.
            mention_attached_context=False,
        )
        out = (out or "").strip()
        if meta.fallback and any(
            s in out.lower()
            for s in ("unexpected error", "couldn't finish that lookup")
        ):
            raise ImageGenError(f"plan unusable ({meta.fallback_reason})")
        plan = image_prompt.extract_plan(out)
        if plan is None:
            if i == 0:
                log.warning("rewrite returned no <image_plan> block (%d chars, "
                            "opens %r) — one corrective retry", len(out),
                            out[:70].replace("\n", " "))
                continue
            raise ImageGenError(
                "rewrite produced no plan block: " + out[:80].replace("\n", " "))
        final_plan = plan
        composed = image_prompt.compose_prompt(
            plan, anchors, _BASE_STYLE,
            foreground=IMAGE_PROMPT_FOREGROUND_FIGURES,
            budget_kwargs=_budget_kwargs(),
            # An NPC appearance is only painted when it came from a file THIS
            # rewrite read. The live dry run caught the model writing a confident
            # "large badger-folk with grey-black fur" for Boss Frey — a character
            # the vault has no appearance notes for — after making no tool call at
            # all. Verification against the files actually read is the only rule
            # that holds when the model skips the lookup.
            read_paths=_read_paths_from(meta.tool_calls_detail),
            snippet_paths=_snippet_paths_from(
                meta.gathered_context, _read_paths_from(meta.tool_calls_detail)),
            retrieved=meta.gathered_context,
            # The request's own words: which species the USER named, and the
            # spelling a campaign name really has ("Frey" painted for "Boss Frey"
            # was measured live, analysis §15).
            request_text=text)
        for w in composed.warnings:
            if w.startswith("budget "):
                log.warning("composer budget: %s", w)
            else:
                log.warning("composer: %s", w)
        # #84's enforcement, in code rather than in a rule the model may ignore.
        # A named character the party does not own was supposed to be looked up
        # before being described; when the plan painted it by name anyway — with a
        # `look` it never retrieved, or none at all — the bot runs ONE lookup pass
        # and composes from what that found. Only for names the request itself
        # named: a bystander the model invented is worth dropping, not searching.
        worth = [n for n in composed.ungrounded
                 if any(part.lower() in text.lower() for part in n.split()
                        if len(part) >= 4)]
        if worth:
            log.info("rewrite: %s named without a verified appearance — one "
                     "lookup pass before composing again", ", ".join(worth))
            retry_out, retry_meta = await run_agent_loop(
                complete=complete,
                model=DEFAULT_MODEL,
                initial_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": out},
                    {"role": "user", "content": _LOOKUP_NUDGE.replace(
                        "{{NAMES}}", ", ".join(worth))},
                ],
                answer_instruction=_IMAGE_ANSWER_INSTRUCTION,
                mention_attached_context=False,
            )
            retry = image_prompt.extract_plan(retry_out or "")
            if retry is None:
                log.warning("rewrite: the lookup pass returned no plan — keeping "
                            "the first (%s painted by name)", ", ".join(worth))
            else:
                opened = (_read_paths_from(meta.tool_calls_detail)
                          + _read_paths_from(retry_meta.tool_calls_detail))
                gathered = "\n\n".join(g for g in
                                       (meta.gathered_context,
                                        retry_meta.gathered_context) if g)
                composed = image_prompt.compose_prompt(
                    retry, anchors, _BASE_STYLE,
                    foreground=IMAGE_PROMPT_FOREGROUND_FIGURES,
                    budget_kwargs=_budget_kwargs(),
                    read_paths=opened,
                    snippet_paths=_snippet_paths_from(gathered, opened),
                    retrieved=gathered, request_text=text)
                for w in composed.warnings:
                    if w.startswith("budget "):
                        log.warning("composer budget: %s", w)
                    else:
                        log.warning("composer: %s", w)
                log.info("image prompt recomposed after the lookup pass: %d "
                         "char(s), %d figure(s), %d tool call(s) in the lookup",
                         len(composed.prompt), composed.figures,
                         len(retry_meta.tool_calls))
                final_plan = retry

        _remember_plan(memory_key, final_plan)
        log.info(
            "image prompt composed: %d char(s), %d figure(s) (%d foreground), "
            "%d tool call(s), %d cut(s), %d anchor(s) available%s — %s",
            len(composed.prompt), composed.figures, composed.foreground,
            len(meta.tool_calls), len(composed.cuts), len(anchors),
            " [after retry]" if i else "",
            composed.prompt[:200] + ("…" if len(composed.prompt) > 200 else ""),
        )
        return composed.prompt
    raise ImageGenError("rewrite produced no plan")  # unreachable


async def rewrite_prompt(text: str, memory_key: Any = None) -> str:
    """Turn a short user request into the prompt the image model gets.

    Runs on the chat backend (its own card) while rag mode is live — call this
    BEFORE ensure_image_mode(): the rewrite's KB tools include the reranker,
    which a mode switch takes offline.

    Two paths, in order (#84 — analysis doc §9.3):
    1. **plan + composer** — the model answers with an `<image_plan>` block (a
       scene, the figures, their poses, an optional field override, and a
       looked-up `look` for anyone off the roster) and bot_core.image_prompt
       writes the prompt from the vault Visual Anchor fields. Every guarantee the
       policy used to ask for is now structural: style sentence first, each
       character described exactly once, a figure-count line, no negations, a
       budget that compresses the least distinctive clauses first.
    2. **the older prose path** — the model writes the prompt itself. Kept as the
       fallback: when a backend hiccup produces no plan, a hand-written prompt is
       still better than a raw request, and a hand-written prompt that turns out
       to be narration is still rejected.

    The rewrite goes through the SAME bounded tool-calling loop as normal chat
    (bot_core.agent_loop) with the SAME knowledge-base tools, so campaign
    specifics in the request (characters, companions, places, items) are grounded
    instead of invented. Speed is not a concern — reliability is: the loop NEVER
    raises; every failure degrades to a single-turn answer over the gathered
    context. Raises ImageGenError when both paths fail (the caller then paints
    the raw request rather than failing).
    """
    client = _chat_client()
    complete = _make_completer(client)
    try:
        return await _rewrite_composed(text, complete, memory_key)
    except ImageGenError as exc:
        log.warning("composed rewrite unavailable (%s) — falling back to the "
                    "prose rewrite path", str(exc)[:160])
    except Exception as exc:  # noqa: BLE001 — a composer bug must never block a paint
        log.exception("composer path failed unexpectedly (%s) — falling back to the "
                      "prose rewrite path", exc)
    return await _rewrite_prose(text, complete)


async def _rewrite_prose(text: str, complete) -> str:
    """The pre-#84 path: the model writes the whole prompt itself."""
    log.info("prompt rewrite started (prose path): %s", text[:120])
    system = _load_rewrite_system()
    try:
        # First the request as given; if the answer turns out to be narration
        # rather than a prompt, ONE corrective retry (see _RETRY_NUDGE).
        for i, msg in enumerate((text, _RETRY_NUDGE + text)):
            out, meta = await run_agent_loop(
                complete=complete,
                model=DEFAULT_MODEL,
                system_prompt=system,
                user_message=msg,
            )
            out = (out or "").strip().strip('"')
            if not out:
                raise ImageGenError("empty rewrite")
            # The loop NEVER raises — on total backend failure it returns an
            # apology sentence instead of a prompt. Those sentinels must fall
            # back to the RAW user prompt, not be painted.
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
            if not looks_like_image_prompt(out):
                if i == 0:
                    log.warning(
                        "rewrite answer is not a prompt (%d chars, opens %r) — "
                        "one corrective retry", len(out), out[:70].replace("\n", " "))
                    continue
                raise ImageGenError(
                    "rewrite produced no prompt (base style sentence absent): "
                    + out[:80].replace("\n", " "))
            log.info(
                "prompt rewrite done: %d iteration(s), tools=%s fallback=%s%s%s — %s",
                meta.iterations, ",".join(meta.tool_calls) or "-",
                meta.fallback,
                f" ({meta.fallback_reason})" if meta.fallback else "",
                " [after retry]" if i else "",
                out[:200] + ("…" if len(out) > 200 else ""),
            )
            return out
        raise ImageGenError("rewrite produced no prompt")  # unreachable
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


async def prepare_prompt(text: str, raw: bool = False,
                         memory_key: Any = None) -> str:
    """Final prompt for the image backend: rewrite via the chat model unless
    ``raw`` (verbatim passthrough).  A failed rewrite degrades to the raw
    text (never blocks a generation over a rewrite hiccup).

    Shared by the live path and the dry-run mode (#65b) so both see exactly
    what would be sent. ``memory_key`` (usually ``(guild_id, channel_id)``)
    lets a follow-up request in the same channel see the previous plan.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if raw:
        return text
    try:
        return await rewrite_prompt(text, memory_key)
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
            # A 500 is sometimes the backend having DROPPED the diffusion
            # model (Studio crash / manual unload) — re-ensure residency so
            # the retry does not just fail the same way. A status read is
            # cheap; a reload only happens if actually needed.
            await _reensure_before_retry()


async def _reensure_before_retry() -> None:
    """Best-effort re-load of the diffusion model before a retry (see
    _generate_one). Never raises — a failure here must not mask the original
    error; the rag-restore in the caller's finally still runs."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if (await _image_status(client)).get("loaded"):
                return
        log.info("image model no longer resident before retry — reloading")
        await _ensure_image_loaded()
    except Exception as e:  # noqa: BLE001
        log.warning("re-ensure before retry failed (continuing): %s", e)


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
            # A MANUAL image window (gpu_switch.sh / helper already in image
            # mode without the bot switching it): the sidecars are down just
            # the same — retrieval must know (R2b).
            _notify_retrievers_gpu_mode("image")
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

    IMAGE_MODE_HOLD_S: after a generation where the BOT itself switched the
    card into image mode, the switch-back is deferred by the hold window so a
    back-to-back generation finds image mode still live (GET /mode says
    "image", model resident → no switch, no load). The window is measured
    from the END of painting. Windows where the card was ALREADY in image mode
    (manual gpu_switch.sh overlap) are not held — the bot restores rag
    immediately. Best-effort: never raises.
    """
    try:
        global _hold_started_at
        if IMAGE_MODE_HOLD_S > 0 and _switcher_enabled() and _hold_active():
            # A hold from an earlier bot switch is still open — extend it
            # (back-to-back generations pay no switch) and let the deferred
            # restore judge whether painting re-entered image mode.
            _spawn_restore_after_hold()
            return
        if IMAGE_MODE_HOLD_S > 0 and _switcher_enabled() and _bot_switched:
            _hold_started_at = max(time.monotonic(), _mode_switched_at)
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
    global _bot_switched
    async with _gen_lock:
        _bot_switched = False
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

    An EMPTY batch is a true no-op — no mode switch is paid for nothing.
    """
    side = size or IMAGE_DEFAULT_SIZE
    total = len(prompts)
    results: list[bytes | Exception] = []
    if not prompts:
        return results
    global _bot_switched
    async with _gen_lock:
        _bot_switched = False
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
