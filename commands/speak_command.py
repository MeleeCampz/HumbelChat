"""/speak (#34) — read text aloud via the TTS endpoint and send it as a
playable mp3 attachment (Discord renders a native audio bar for mp3 files).

#70: long synthesis (incl. on-demand voice-model loads, up to TTS_TIMEOUT_S)
gets live status updates like /generate_image — one follow-up message edited
per phase (reading aloud → loading voice model), and the finished
caption+mp3 replace the last update in place.
"""
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

    # Progress message (#70, same pattern as /generate_image): edited per
    # phase AND it is the final answer — edit() can add attachments (Discord
    # API + discord.py 2.7), so the finished caption+mp3 replace the last
    # update in place: ONE message, live feedback preserved.
    progress_msg = await interaction.followup.send("🎙️ Reading aloud…")

    async def _progress(text_update: str) -> None:
        try:
            await progress_msg.edit(content=text_update)
        except Exception as exc:  # noqa: BLE001 — progress must never fail the TTS
            log.debug("progress update failed: %s", exc)

    async def _cleanup() -> None:
        try:
            await progress_msg.delete()
        except Exception as exc:  # noqa: BLE001 — cosmetic only
            log.debug("progress cleanup failed: %s", exc)

    def _phase(name: str):
        if name == "loading":
            return _progress("📖 Loading the voice model (up to a minute) …")
        return None  # "speaking" already announced above

    try:
        audio = await synthesize(text, on_phase=_phase)
    except TTSError as e:
        log.error("TTS request failed: %s", e, exc_info=True)
        await interaction.followup.send(
            f"⚠️ TTS failed ({e}). Is the TTS model "
            f"`{TTS_MODEL}` loaded in your backend?"
        )
        await _cleanup()
        return

    preview = text[:_CAPTION_PREVIEW_CHARS]
    if len(text) > _CAPTION_PREVIEW_CHARS:
        preview += "…"
    filename = f"speak-{int(time.time())}.mp3"
    file = discord.File(io.BytesIO(audio), filename=filename)
    # Final answer IN PLACE on the progress message (caption + mp3 together).
    try:
        await progress_msg.edit(content=f"🔊 {preview}", attachments=[file])
    except Exception as exc:  # noqa: BLE001 — degrade to a fresh message
        log.warning("final in-place edit failed (%s) — sending fresh message", exc)
        await interaction.followup.send(f"🔊 {preview}", file=file)
