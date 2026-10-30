"""Automatic session-highlight images (#69) — key-moment art after /end_session.

Spawned as a tracked background task from ``commands.session_commands``
(``spawn_tracked_task``).  Flow:

1. select up to ``SESSION_IMAGE_MAX_MOMENTS`` key moments from the session's
   record (resident chat model — BEFORE any swap);
2. prepare ALL prompts first, through the existing rewrite pipeline
   (:func:`bot_core.images.prepare_prompt` — KB tools ground campaign
   specifics, Visual Anchors verbatim) — still on the resident chat model;
3. paint the whole batch in ONE model-swap window
   (:func:`bot_core.images.generate_images`: load once → paint ×N → unload +
   one warm chat reload), holding the global AI slot so no chat turn can
   re-resident the chat model mid-batch;
4. post each painted moment to the ending channel as a captioned PNG and
   archive it under the session folder's hidden ``.images/`` directory.

This module NEVER raises — every failure degrades to a friendly channel
message or a log line, and progress display is best-effort (a broken callback
never fails the work).
"""
from __future__ import annotations

import io
import json
import logging
import re
from typing import Any

import discord

from bot_core import images as I
from bot_core.ai_client import _ai_slot, _make_client, _validate_model, complete_text
from bot_core.channel_delivery import (
    ChannelNotDeliverableError,
    get_bot,
    resolve_channel,
)
from bot_core.history import get_active_char_key
from bot_core.sessions import SESSION_DOCS_BUDGET, session_documents_text
from config.characters import get_character
from config.settings import (
    DEFAULT_MODEL,
    SESSION_IMAGE_MAX_MOMENTS,
    SESSION_IMAGES,
    SESSION_IMAGES_PROMPT,
    SUMMARY_CALL_MAX_TOKENS,
)

log = logging.getLogger("bot.session_images")


# ── Gates ────────────────────────────────────────────────────────────────

def will_generate(session: dict[str, Any], channel_id: int | None) -> bool:
    """/end_session gate: should highlight generation start for this session?

    Feature enabled + an image model configured + a known ending channel +
    something actually recorded (notes, documents or a merged log).
    """
    if not SESSION_IMAGES:
        return False
    if not I.image_gen_configured():
        return False
    if not channel_id:
        return False
    if session.get("notes") or session.get("merged_log"):
        return True
    return bool(session_documents_text(session)[1])


def _selection_prompt() -> str:
    """The key-moment selection prompt (env override, else the default)."""
    base = SESSION_IMAGES_PROMPT.strip() or None
    if base is None:
        from config.settings import DEFAULT_SESSION_IMAGES_PROMPT
        base = DEFAULT_SESSION_IMAGES_PROMPT
    return base


# ── Selection ────────────────────────────────────────────────────────────

def _parse_moments(raw: str, max_moments: int | None = None) -> list[dict[str, str]] | None:
    """Parse the model's JSON array into ``[{title, scene}]``.

    Tolerant on purpose: strips code fences, tolerates prose around the
    array, accepts bare strings (used as the scene), drops empty entries and
    truncates to *max_moments*.  Returns None when nothing usable remains.
    """
    limit = max_moments or SESSION_IMAGE_MAX_MOMENTS
    text = (raw or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    moments: list[dict[str, str]] = []
    for item in data[:limit]:
        if isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            scene = str(item.get("scene") or "").strip()
        elif isinstance(item, str):
            title, scene = "", item.strip()
        else:
            continue
        if scene:
            moments.append({"title": title or scene[:60], "scene": scene})
    return moments or None


async def _select_moments(
    session: dict[str, Any],
    guild_id: int | None,
    channel_id: int | None,
) -> list[dict[str, str]] | None:
    """Pick the session's key moments via ONE chat-model call (resident model).

    Returns a list of ``{title, scene}`` dicts, or None when no usable
    selection could be made (no model, backend error, unparseable output).
    """
    char = get_character(get_active_char_key(guild_id, channel_id))
    model = await _validate_model(
        _make_client(), (char.model if (char and char.model) else "").strip() or DEFAULT_MODEL
    )
    if not model:
        log.warning("no model available for highlight selection — skipping")
        return None

    docs_text, doc_names = session_documents_text(session)
    notes = session.get("notes", [])
    notes_text = "\n".join(f"- {text}" for _ts, text in notes[-50:]) or "(no notes)"
    source_listing = ", ".join(doc_names) if doc_names else "(none)"
    parts = [
        f"Session name: {session.get('name') or '(untitled)'}",
        "",
        "## Session files",
        source_listing,
        docs_text or "(none)",
        "",
        "## Session notes",
        notes_text,
    ]
    merged = session.get("merged_log")
    if merged:
        parts += ["", "## Merged session log (canonical narrative)", str(merged)[:SESSION_DOCS_BUDGET]]

    try:
        raw = await complete_text(
            _make_client(),
            model=model,
            messages=[
                {"role": "system", "content": _selection_prompt()},
                {"role": "user", "content": "\n".join(parts)},
            ],
            temperature=0.3,
            max_tokens=max(0, SUMMARY_CALL_MAX_TOKENS) or None,
            reasoning_effort="low",
        )
    except Exception as e:  # noqa: BLE001 — selection is best-effort
        log.error("highlight selection call failed: %s", e, exc_info=True)
        return None

    moments = _parse_moments(raw)
    if not moments:
        log.warning("highlight selection returned no usable moments (%d chars): %.120s",
                    len(raw or ""), raw or "")
    return moments


# ── Delivery helpers ─────────────────────────────────────────────────────

def _archive(session: dict[str, Any], index: int, png: bytes) -> None:
    """Persist a painted moment under the session folder's hidden .images/ dir.

    A disk failure is a log line only — the Discord delivery already happened.
    """
    try:
        import pathlib
        folder = pathlib.Path(session["dir"]) / ".images"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"moment-{index}.png").write_bytes(png)
    except OSError as e:
        log.warning("could not archive highlight image %d for session %r: %s",
                    index, session.get("name"), e)


async def _safe_send(chan: Any, text: str) -> None:
    """Best-effort channel send — a delivery failure is logged, never raised."""
    if chan is None:
        return
    try:
        await chan.send(text)
    except Exception as e:  # noqa: BLE001 — progress/delivery must not fail the work
        log.warning("highlight message send failed: %s", e)


# ── Orchestration ────────────────────────────────────────────────────────

async def _paint_batch(
    prompts: list[str],
    chan: Any,
    name: str,
) -> list[bytes | Exception]:
    """Paint every prepared prompt in ONE swap window, with live progress.

    Posts a progress message to the channel (when reachable), edits it per
    phase, then deletes it.  Holds the global AI slot for the whole batch.
    Returns one entry per prompt: PNG bytes or the ImageGenError that killed
    it (a single failure never aborts the rest).
    """
    # ONE announcement/progress message, edited in place per phase and
    # deleted once the batch is done (kept as a reference so it can be).
    progress: Any = None
    if chan is not None:
        try:
            progress = await chan.send(
                f"🎨 Painting highlights of **{name}** — {len(prompts)} moment(s). "
                "Loading the image model (up to a minute)…"
            )
        except Exception as e:  # noqa: BLE001
            log.warning("highlight progress message failed: %s", e)

    def _on_phase(phase_name: str):
        if progress is None:
            return
        if phase_name == "loading":
            text = "📖 Loading the image model…"
        elif phase_name.startswith("painting"):
            frac = phase_name.split(maxsplit=1)[1] if " " in phase_name else ""
            text = f"🖌️ Painting {frac}…"
        else:
            return

        async def _edit() -> None:
            try:
                await progress.edit(content=text)
            except Exception as e:  # noqa: BLE001 — progress must never kill the batch
                log.debug("highlight progress edit failed: %s", e)

        return _edit()

    async with _ai_slot():
        results = await I.generate_images(prompts, on_phase=_on_phase)

    if progress is not None:
        try:
            await progress.delete()
        except Exception as e:  # noqa: BLE001
            log.debug("highlight progress delete failed: %s", e)
    return results


async def _run_highlights(
    session: dict[str, Any],
    chan: Any,
    guild_id: int | None,
    channel_id: int | None,
) -> None:
    name = session.get("name") or "(untitled)"

    # 1) Select key moments — resident chat model, BEFORE any swap.
    moments = await _select_moments(session, guild_id, channel_id)
    if not moments:
        await _safe_send(chan, f"🎨 No clear key moments to paint for **{name}** — nothing generated.")
        return

    # 2) Prepare ALL prompts first — resident chat model, before the swap.
    prompts: list[str] = []
    for moment in moments:
        text = (f"{moment['title']}\n{moment['scene']}" if moment["title"] else moment["scene"]).strip()
        prepared = await I.prepare_prompt(text)  # rewrite failure degrades to raw text
        if prepared:
            prompts.append(prepared)
    if not prompts:
        await _safe_send(chan, f"🎨 The selected moments for **{name}** came out empty — nothing generated.")
        return

    # 3) Paint the whole batch in ONE model-swap window.
    results = await _paint_batch(prompts, chan, name)

    # 4) Deliver: one captioned PNG per success; a line for each failure.
    failures: list[str] = []
    for index, (moment, result) in enumerate(zip(moments, results), 1):
        if isinstance(result, bytes):
            _archive(session, index, result)
            if chan is not None:
                filename = f"session-{session.get('id', 'unknown')}-moment-{index}.png"
                try:
                    await chan.send(
                        f"🎨 {moment['title']}",
                        file=discord.File(io.BytesIO(result), filename=filename),
                    )
                except Exception as e:  # noqa: BLE001 — archived on disk regardless
                    log.warning("highlight image %d send failed: %s", index, e)
        else:
            failures.append(f"**{moment['title']}** ({str(result)[:80]})")

    if len(results) and not any(isinstance(r, bytes) for r in results):
        await _safe_send(
            chan,
            f"⚠️ None of the highlights for **{name}** could be painted — "
            "the image backend may be down. Try `/generate_image` manually later.",
        )
    elif failures:
        await _safe_send(chan, "⚠️ Could not paint: " + "; ".join(failures))


async def generate_session_highlights(
    session: dict[str, Any],
    channel_id: int | None,
    guild_id: int | None = None,
) -> None:
    """Paint an ended session's key moments into its ending channel.

    Entry point for the background task — NEVER raises.  When the channel
    itself is undeliverable the images are still generated and archived to
    disk (loud log), so a delivery hiccup never wastes the paint time.
    """
    if not will_generate(session, channel_id):
        return

    name = session.get("name") or "(untitled)"
    chan: Any = None
    try:
        chan, _via = await resolve_channel(get_bot(), int(channel_id))
    except ChannelNotDeliverableError as e:
        log.error("highlight channel %s undeliverable — images will only be archived to disk: %s",
                  channel_id, e, exc_info=True)

    try:
        await _run_highlights(session, chan, guild_id, channel_id)
    except Exception as e:  # noqa: BLE001 — background task must never raise
        log.exception("session highlight images failed for %r: %s", name, e)
        await _safe_send(
            chan,
            f"⚠️ The session highlights for **{name}** ran into a problem "
            f"({e.__class__.__name__}) — nothing was lost, the session record is intact.",
        )
