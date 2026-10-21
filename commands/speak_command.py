"""/speak (#34) — read text aloud via the TTS endpoint and send it as a
playable mp3 attachment (Discord renders a native audio bar for mp3 files)."""
from __future__ import annotations

import io
import logging
import time

import discord

from bot_core.tts import TTSError, synthesize, tts_configured
from config.settings import TTS_MAX_CHARS, TTS_MODEL

log = logging.getLogger("bot.commands.speak")

_CAPTION_PREVIEW_CHARS = 80


async def handle_speak_command(interaction: discord.Interaction, text: str) -> None:
    """Synthesize ``text`` to speech and deliver it as an mp3 message."""
    await interaction.response.defer()

    if not tts_configured():
        await interaction.followup.send(
            "⚠️ TTS is not configured. Set `TTS_MODEL` in `.env` "
            "(name of a TTS model loaded in your backend, e.g. Unsloth Studio) "
            "and restart the bot."
        )
        return

    text = (text or "").strip()
    if not text:
        await interaction.followup.send("⚠️ Nothing to read — give me some text.")
        return
    if len(text) > TTS_MAX_CHARS:
        await interaction.followup.send(
            f"⚠️ Too long to read: {len(text):,} chars "
            f"(limit {TTS_MAX_CHARS:,}). Shorten the text and try again."
        )
        return

    try:
        audio = await synthesize(text)
    except TTSError as e:
        log.error("TTS request failed: %s", e)
        await interaction.followup.send(
            f"⚠️ TTS failed ({e}). Is the TTS model "
            f"`{TTS_MODEL}` loaded in your backend?"
        )
        return

    preview = text[:_CAPTION_PREVIEW_CHARS]
    if len(text) > _CAPTION_PREVIEW_CHARS:
        preview += "…"
    filename = f"speak-{int(time.time())}.mp3"
    file = discord.File(io.BytesIO(audio), filename=filename)
    await interaction.followup.send(f"🔊 {preview}", file=file)
