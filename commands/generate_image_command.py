"""/generate_image (#42) — generate an image via the Studio diffusion model.

Flow (see bot_core/images.py for the why):
1. defer with a placeholder ("Generating…")
2. prompt rewrite via the CHAT model (before any swap — no image resident yet)
   unless ``--raw``
3. hold the global AI slot across the whole load→generate→unload window so no
   other chat turn can re-resident the chat model and evict the image mid-run
4. deliver the PNG as a fresh message (Discord edits can't add attachments)
"""
from __future__ import annotations

import io
import logging
import time

import discord

from bot_core.ai_client import _ai_slot
from bot_core.images import (
    ImageGenError,
    generate_image,
    image_gen_configured,
    resolve_size,
    rewrite_prompt,
)

log = logging.getLogger("bot.commands.generate_image")

_PROMPT_PREVIEW_CHARS = 80


async def handle_generate_image_command(
    interaction: discord.Interaction, prompt: str, raw: bool = False, size: str | None = None
) -> None:
    """Generate an image from ``prompt`` and send it as a PNG message."""
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

    # 1) Rewrite FIRST, on the resident chat model (no image loaded yet).
    final_prompt = prompt
    if not raw:
        try:
            final_prompt = await rewrite_prompt(prompt)
        except ImageGenError as e:
            log.warning("prompt rewrite failed (%s) — using the raw prompt", e)

    # 2) The no-chat-calls window: hold the AI slot for load→generate→unload.
    try:
        async with _ai_slot():
            png = await generate_image(final_prompt, size=side)
    except ImageGenError as e:
        log.error("image generation failed: %s", e)
        await interaction.followup.send(f"⚠️ Image generation failed ({e}). Try again in a moment.")
        return

    preview = prompt[:_PROMPT_PREVIEW_CHARS]
    if len(prompt) > _PROMPT_PREVIEW_CHARS:
        preview += "…"
    filename = f"image-{int(time.time())}.png"
    file = discord.File(io.BytesIO(png), filename=filename)
    await interaction.followup.send(f"🎨 {preview}", file=file)
