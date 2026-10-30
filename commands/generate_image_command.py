"""/generate_image (#42) — generate an image via the Studio diffusion model.

Flow (see bot_core/images.py for the why):
1. defer with a placeholder ("Generating…")
2. prompt rewrite via the CHAT model (before any swap — no image resident yet)
   unless ``--raw``
3. hold the global AI slot across the whole load→generate→unload window so no
   other chat turn can re-resident the chat model and evict the image mid-run
4. deliver the final text + PNG IN PLACE on the progress message (edit() can
   add attachments — Discord API + discord.py 2.7); only if that edit fails
   does it degrade to a fresh follow-up message
"""
from __future__ import annotations

import io
import logging
import time

import discord

from bot_core.ai_client import _ai_slot
from bot_core.images import (
    IMAGE_DEFAULT_SIZE,
    ImageGenError,
    generate_image,
    image_gen_configured,
    prepare_prompt,
    resolve_size,
)

log = logging.getLogger("bot.commands.generate_image")

_PROMPT_PREVIEW_CHARS = 80


async def handle_generate_image_command(
    interaction: discord.Interaction,
    prompt: str,
    raw: bool = False,
    size: str | None = None,
    dry_run: bool = False,
) -> None:
    """Generate an image from ``prompt`` and send it as a PNG message.

    ``dry_run`` (#65b): do the prompt work (rewrite unless ``raw``) and reply
    with the FINAL prompt that would be sent to the image backend — no model
    swap, no generation. For iterating on prompts without burning generations.

    Progress updates: a short follow-up message is edited at each phase
    (prompt → loading → painting) so the long GPU-idle swap windows read as
    visible progress instead of a frozen "bot is thinking". (discord.py's
    response.defer() takes no content kwarg — hence the separate message.)
    """
    await interaction.response.defer()

    if not image_gen_configured():
        await interaction.followup.send(
            "⚠️ Image generation is not configured. Set `IMAGE_MODEL` in `.env` "
            "(e.g. `unsloth/Qwen-Image-2.1-Turbo-GGUF`) and restart the bot."
        )
        return

    prompt = (prompt or "").strip()
    if not prompt:
        await interaction.followup.send("⚠️ Nothing to paint — give me a prompt.")
        return

    try:
        side = resolve_size(size)
    except ImageGenError as e:
        await interaction.followup.send(f"⚠️ {e}")
        return

    # Progress message (after the early-return checks above): edited per phase
    # AND it is the final answer — edit() can add attachments (Discord API +
    # discord.py 2.7), so the finished text+image replace the last update
    # in place: ONE message, live feedback preserved.
    progress_msg = await interaction.followup.send("🎨 Preparing prompt…")

    async def _progress(text: str) -> None:
        try:
            await progress_msg.edit(content=text)
        except Exception as exc:  # noqa: BLE001 — progress must never fail the gen
            log.debug("progress update failed: %s", exc)

    async def _cleanup() -> None:
        try:
            await progress_msg.delete()
        except Exception as exc:  # noqa: BLE001 — cosmetic only
            log.debug("progress cleanup failed: %s", exc)

    # 1) Rewrite FIRST, on the resident chat model (no image loaded yet).
    final_prompt = await prepare_prompt(prompt, raw=raw)

    # Dry run (#65b): show exactly what would be sent — no swap, no image.
    if dry_run:
        log.info("dry-run: returning final prompt without generating (%d chars)",
                 len(final_prompt))
        await interaction.followup.send(
            "🔍 **Dry run — nothing generated.** Final prompt that would be sent:\n\n" +
            f"```\n{final_prompt}\n```"
        )
        await _cleanup()
        return

    # 2) The no-chat-calls window: hold the AI slot for load→generate→unload.
    await _progress("📖 Prompt ready — loading the image model (up to a minute)…")

    def _phase(name: str):
        if name == "painting":
            # `side` is None when the user gave no size — the effective
            # default is applied inside generate_image; show that, not "None".
            return _progress(f"🖌️ Painting… ({side or IMAGE_DEFAULT_SIZE} px)")
        return None  # "loading" already announced above

    try:
        async with _ai_slot():
            png = await generate_image(final_prompt, size=side, on_phase=_phase)
    except ImageGenError as e:
        log.error("image generation failed: %s", e, exc_info=True)
        await interaction.followup.send(f"⚠️ Image generation failed ({e}). Try again in a moment.")
        await _cleanup()
        return

    preview = prompt[:_PROMPT_PREVIEW_CHARS]
    if len(prompt) > _PROMPT_PREVIEW_CHARS:
        preview += "…"
    filename = f"image-{int(time.time())}.png"
    file = discord.File(io.BytesIO(png), filename=filename)
    # Final answer IN PLACE on the progress message (text + image together).
    try:
        await progress_msg.edit(content=f"🎨 {preview}", attachments=[file])
    except Exception as exc:  # noqa: BLE001 — degrade to a fresh message
        log.warning("final in-place edit failed (%s) — sending fresh message", exc)
        await interaction.followup.send(f"🎨 {preview}", file=file)
