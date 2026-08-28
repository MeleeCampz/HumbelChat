"""Utility slash commands: remind, ocr, summarize, translate."""
from __future__ import annotations

import asyncio
import base64
import logging

import discord
import httpx

from config.settings import DEFAULT_MODEL, FALLBACK_MODELS
from bot_core.history import get_active_char_key, get_history
from bot_core.ai_client import _make_client, _validate_model
from config.characters import get_character

log = logging.getLogger("bot.utility_commands")

# OCR download guards (code review §1.9)
OCR_DOWNLOAD_TIMEOUT = 30.0   # seconds
OCR_MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
# NOTE: .heic/.heif are intentionally NOT listed — Discord attachments arrive
# without a usable MIME mapping for them and the vision backend rejects them.
OCR_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif")

# Extension → MIME type for the data-URI sent to the vision model.
_OCR_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}
OCR_TEXT_EXTS = (".txt", ".md", ".csv", ".json", ".log", ".xml", ".html", ".htm", ".ini", ".yaml", ".yml")


def _resolve_utility_model(guild_id: int | None, channel_id: int) -> tuple[str, float | None, int | None]:
    """Resolve (model, temperature, max_tokens) from the channel's active
    character, falling back to DEFAULT_MODEL when unset (code review §1.8).
    """
    char_key = get_active_char_key(guild_id, channel_id)
    char = get_character(char_key)
    model = (char.model if (char and char.model) else "").strip()
    if not model:
        return DEFAULT_MODEL, None, None
    temp = getattr(char, "temperature", None)
    max_tok = getattr(char, "max_tokens", None)
    return model, temp, max_tok


async def _validated_utility_model(guild_id: int | None, channel_id: int) -> str:
    """Resolve + guard against stale model slugs (P0 model-not-found guard)."""
    model, _, _ = _resolve_utility_model(guild_id, channel_id)
    client = _make_client()
    return await _validate_model(client, model)


# ── Remind ───────────────────────────────────────────────────────────────

async def handle_remind_command(
    interaction: discord.Interaction,
    time_value: int,
    time_unit: str,
    message: str,
) -> None:
    """Schedule a one-time reminder."""
    multipliers = {
        "second": 1, "seconds": 1, "s": 1,
        "minute": 60, "minutes": 60, "min": 60, "m": 60,
        "hour": 3600, "hours": 3600, "hr": 3600, "h": 3600,
    }
    unit_lower = time_unit.lower()
    if unit_lower not in multipliers:
        await interaction.response.send_message(
            f"Unknown unit ``{time_unit}``. Use: seconds, minutes, hours.", ephemeral=True
        )
        return

    delay = time_value * multipliers[unit_lower]
    if delay < 10:
        await interaction.response.send_message(
            "Reminder must be at least 10 seconds in the future.", ephemeral=True
        )
        return

    channel_id = interaction.channel.id

    # Fail fast: refuse to schedule into a channel we cannot actually write
    # to (e.g. private channels with View Channel denied on @everyone) —
    # such a reminder could never be delivered.
    from bot_core.channel_delivery import can_post_in_channel
    if not await can_post_in_channel(interaction):
        await interaction.response.send_message(
            "⚠️ I can't send messages in this channel — I'm missing **View Channel** "
            "and/or **Send Messages** permission (check the @everyone / role "
            "overwrites). Set the reminder in a channel I can post to, or give "
            "me access first.", ephemeral=True,
        )
        return

    await interaction.response.defer(ephemeral=True)

    # Persist + schedule via the reminder store (survives restarts)
    from bot_core.reminders import schedule_reminder
    schedule_reminder(channel_id, message, delay)

    unit_singular = time_unit.rstrip("s") if time_value != 1 else time_unit
    prompt_text = "✅ Reminder set for **" + str(time_value) + " " + unit_singular + "** from now!"
    confirmation = prompt_text + f'\n📝 I\'ll ping you with: "{message}"'
    await interaction.followup.send(confirmation, ephemeral=True)


# ── OCR ──────────────────────────────────────────────────────────────────

async def handle_ocr_command(
    interaction: discord.Interaction,
    image: discord.Attachment | None = None,
) -> None:
    """Vision-based OCR."""
    await interaction.response.defer(ephemeral=True)

    if not image:
        await interaction.followup.send("⚠️ Please attach an image.", ephemeral=True)
        return

    # Non-image guards (code review §1.9): refuse text/unknown extensions
    # *before* spending a download on a file the vision model can't read.
    fn = (image.filename or "").lower()
    if fn.endswith(OCR_TEXT_EXTS):
        await interaction.followup.send(
            f"⚠️ `{image.filename}` looks like a text file — no OCR needed. "
            "Attach an actual image (PNG/JPG/GIF/WebP).",
            ephemeral=True,
        )
        return
    if fn and not fn.endswith(OCR_IMAGE_EXTS):
        await interaction.followup.send(
            f"⚠️ `{image.filename}` doesn't look like an image. "
            f"Supported: {', '.join(sorted(set(OCR_IMAGE_EXTS)))}.",
            ephemeral=True,
        )
        return

    # Download guards (code review §1.9): timeout + max size on the bytes.
    # Read via the Discord attachment API (discord.py's own session applies
    # its own timeout; we bound this read step explicitly).
    try:
        img_data = await asyncio.wait_for(image.read(), timeout=OCR_DOWNLOAD_TIMEOUT)
    except asyncio.TimeoutError:
        await interaction.followup.send("⚠️ Timed out downloading the image.", ephemeral=True)
        return
    if not img_data:
        await interaction.followup.send("⚠️ Could not download the image.", ephemeral=True)
        return
    if len(img_data) > OCR_MAX_DOWNLOAD_BYTES:
        await interaction.followup.send(
            f"⚠️ Image too large to process (> {OCR_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB).",
            ephemeral=True,
        )
        return

    # MIME detection — map the real extension instead of defaulting to PNG.
    mime = next((m for ext, m in _OCR_MIME_MAP.items() if fn.endswith(ext)), "image/png")

    b64 = base64.b64encode(img_data).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"

    model = await _validated_utility_model(
        interaction.guild_id, interaction.channel_id
    )
    client = _make_client()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this image accurately."},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            temperature=0,
            max_tokens=4096,
        )
    except Exception as e:
        log.error("OCR request failed: %s", e)
        await interaction.followup.send(
            f"⚠️ OCR failed: {e.__class__.__name__}. The AI backend may be down or the model may not support images.",
            ephemeral=True,
        )
        return
    reply = resp.choices[0].message.content or "(no text found)"

    MAX_LEN = 1900
    if len(reply) <= MAX_LEN:
        await interaction.followup.send(f"🔍 Extracted text:\n\n{reply}", ephemeral=True)
    else:
        # Paragraph-aware split (avoids mid-word hard cuts).
        from utils.response_splitter import _split_long_message
        chunks = _split_long_message(reply, "")
        for i, chunk in enumerate(chunks, start=1):
            await interaction.followup.send(
                f"🔍 OCR (part {i}/{len(chunks)})\n\n{chunk}", ephemeral=True
            )


# ── Summarize ────────────────────────────────────────────────────────────

async def handle_summarize_command(
    interaction: discord.Interaction,
    file_url: str | None = None,
) -> None:
    """Summarize recent chat history or a file from a URL."""
    await interaction.response.defer(ephemeral=True)

    text = ""
    src = ""
    if file_url:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(file_url, timeout=10.0)
                resp.raise_for_status()
                text = resp.text[:32000]
                src = f"file from `{file_url[:80]}...`"
        except Exception as e:
            log.error("Failed to fetch file_url: %s", e)
            await interaction.followup.send(f"⚠️ Error fetching URL: {e}", ephemeral=True)
            return
    else:
        guild_id = interaction.guild_id or 0
        ch_history = get_history(guild_id, interaction.channel_id)
        parts = []
        for msg in ch_history[-30:]:
            role_name = {"user": "User", "assistant": "AI"}.get(msg["role"], msg["role"])
            parts.append(f"[{role_name}]: {msg['content']}")
        text = "\n\n".join(parts) if parts else "(no history)"
        src = "recent conversation"

    if not text.strip() or text == "(no history)":
        await interaction.followup.send("⚠️ Nothing to summarize.", ephemeral=True)
        return

    guild_id = interaction.guild_id or 0
    primary_model, char_temp, _ = _resolve_utility_model(guild_id, interaction.channel_id)
    client = _make_client()

    models_to_try = [primary_model]
    models_to_try.extend(FALLBACK_MODELS)

    summary = None
    for model in models_to_try:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Summarize the following text from {src}. Be concise but complete.",
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            summary = resp.choices[0].message.content or "(empty)"
            break  # Success!
        except Exception as e:
            log.error("Summarize error with model %s: %s", model, e)
            continue

    if summary is None:
        await interaction.followup.send(
            "⚠️ Failed to generate summary after trying multiple models.", ephemeral=True
        )
    else:
        MAX_LEN = 1900
        if len(summary) <= MAX_LEN:
            await interaction.followup.send(f"📄 **Summary** ({src}):\n\n{summary}", ephemeral=True)
        else:
            for i in range(0, len(summary), MAX_LEN):
                await interaction.followup.send(
                    f"📄 **Summary** ({src}) (part {i // MAX_LEN + 1}):\n\n{summary[i:i + MAX_LEN]}",
                    ephemeral=True,
                )


# ── Translate ────────────────────────────────────────────────────────────

async def handle_translate_command(
    interaction: discord.Interaction,
    target_language: str,
    source_language: str | None = None,
) -> None:
    """Translate text via the AI provider."""
    parts = target_language.split(":", 1)
    tgt = parts[0].strip()
    text_to = parts[1].strip() if len(parts) > 1 else None

    if not text_to:
        guild_id = interaction.guild_id or 0
        ch_hist = get_history(guild_id, interaction.channel_id)
        last_user = [m["content"] for m in reversed(ch_hist) if m["role"] == "user"]
        text_to = last_user[0] if last_user else None

    if not text_to:
        await interaction.followup.send(
            "⚠️ No text to translate.  Provide text as ``/translate Spanish: Hello world``.",
            ephemeral=True,
        )
        return

    src_clause = f" from {source_language}" if source_language else ""

    model = await _validated_utility_model(interaction.guild_id, interaction.channel_id)
    client = _make_client()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"Translate{src_clause} text into {tgt}. Return ONLY translated text.",
                },
                {"role": "user", "content": text_to},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
    except Exception as e:
        log.error("Translate request failed: %s", e)
        await interaction.followup.send(f"⚠️ Translation failed: {e.__class__.__name__}. Please try again.", ephemeral=True)
        return
    translated = resp.choices[0].message.content or "(translation failed)"

    MAX_LEN = 1900
    if len(translated) <= MAX_LEN:
        await interaction.followup.send(f"🌐 Translated to **{tgt}**:\n\n{translated}", ephemeral=True)
    else:
        for i in range(0, len(translated), MAX_LEN):
            await interaction.followup.send(
                f"🌐 Translated to **{tgt}** (part {i // MAX_LEN + 1})\n\n{translated[i:i + MAX_LEN]}",
                ephemeral=True,
            )
