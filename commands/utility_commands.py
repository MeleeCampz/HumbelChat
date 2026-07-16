"""Utility slash commands: remind, ocr, summarize, translate."""
from __future__ import annotations

import asyncio
import base64
import logging
import httpx

import discord
import discord.app_commands as app_commands
from openai import AsyncOpenAI

log = logging.getLogger("bot.utility_commands")


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
    await interaction.response.defer(ephemeral=True)
    asyncio.create_task(_send_reminder(channel_id, message, delay=delay))

    unit_singular = time_unit.rstrip("s") if time_value != 1 else time_unit
    prompt_text = "\u2705 Reminder set for **" + str(time_value) + " " + unit_singular + "** from now!"
    confirmation = prompt_text + f'\n\U0001f4dd I"ll ping you with: "{message}"'
    await interaction.followup.send(confirmation, ephemeral=True)


async def _send_reminder(channel_id: int, message: str, delay: int) -> None:
    """Background reminder sender. Sleeps *delay* seconds before sending."""
    from main import bot  # noqa: circular import

    await asyncio.sleep(delay)
    try:
        chan = bot.get_channel(channel_id)
        if chan:
            await chan.send(f"\u23f0 **Reminder:** {message}")
    except Exception as e:
        log.error("Failed to send reminder: %s", e)


# ── OCR ──────────────────────────────────────────────────────────────────

async def handle_ocr_command(
    interaction: discord.Interaction,
    image: discord.Attachment = None,
) -> None:
    """Vision-based OCR."""
    from config.settings import settings  # noqa: local-only

    await interaction.response.defer(ephemeral=True)

    if not image:
        await interaction.followup.send("\u26a0\ufe0f Please attach an image.", ephemeral=True)
        return

    async with httpx.AsyncClient() as client:
        img_resp = await client.get(image.url)
    img_data = img_resp.content

    # MIME detection
    mime = "image/png"
    fn   = (image.filename or "").lower()
    if fn.endswith((".jpg", ".jpeg")):  mime = "image/jpeg"
    if fn.endswith((".gif")):           mime = "image/gif"
    if fn.endswith((".webp")):          mime = "image/webp"

    b64   = base64.b64encode(img_data).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"

    client = AsyncOpenAI(
        api_key=settings.INFER_API_KEY or "local-model-key",
        base_url=settings.INFER_URL,
    )
    resp = await client.chat.completions.create(
        model=settings.DEFAULT_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Extract all text from this image accurately."},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]}],
        temperature=0, max_tokens=4096,
    )
    reply = resp.choices[0].message.content or "(no text found)"

    MAX_LEN = 1900
    if len(reply) <= MAX_LEN:
        await interaction.followup.send(f"\U0001f50d Extracted text:\n\n{reply}", ephemeral=True)
    else:
        for i in range(0, len(reply), MAX_LEN):
            await interaction.followup.send(f"\U0001f50d OCR (part {i//MAX_LEN+1})\n\n{reply[i:i+MAX_LEN]}", ephemeral=True)


# ── Summarize ────────────────────────────────────────────────────────────

async def handle_summarize_command(
    interaction: discord.Interaction,
    file_url: str | None = None,
) -> None:
    """Summarize recent chat history or a file from a URL."""
    from config.settings import settings  # noqa: local-only
    from bot_core import _chat_history  # noqa: local-only

    await interaction.response.defer(ephemeral=True)

    text = ""
    src  = ""
    if file_url:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(file_url, timeout=10.0)
                resp.raise_for_status()
                text = resp.text[:32000]
                src  = f"file from `{file_url[:80]}...`"
        except Exception as e:
            log.error("Failed to fetch file_url: %s", e)
            await interaction.followup.send(f"\u26a0\ufe0f Error fetching URL: {e}", ephemeral=True)
            return
    else:
        guild_id = interaction.guild_id or 0
        ch_history = _chat_history.get(guild_id, {}).get(interaction.channel_id, [])
        parts = []
        for msg in ch_history[-30:]:
            role_name = {"user": "User", "assistant": "AI"}.get(msg["role"], msg["role"])
            parts.append(f"[{role_name}]: {msg['content']}")
        text     = "\n\n".join(parts) if parts else "(no history)"
        src      = "recent conversation"

    if not text.strip() or text == "(no history)":
        await interaction.followup.send("\u26a0\ufe0f Nothing to summarize.", ephemeral=True)
        return

    client = AsyncOpenAI(
        api_key=settings.INFER_API_KEY or "local-model-key",
        base_url=settings.INFER_URL,
    )

    # Define fallback models in case the primary one fails
    models_to_try = [settings.DEFAULT_MODEL]
    if settings.DEFAULT_MODEL not in ["gemma4:latest", "llama3:latest"]:  # noqa: simple check for common defaults
        models_to_try.extend(["gemma4:latest", "llama3:latest"])

    summary = None
    for model in models_to_try:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"Summarize the following text from {src}. Be concise but complete."},
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
        await interaction.followup.send("\u26a0\ufe0f Failed to generate summary after trying multiple models.", ephemeral=True)
    else:
        MAX_LEN = 1900
        if len(summary) <= MAX_LEN:
            await interaction.followup.send(f"\U0001f4dd **Summary** ({src}):\n\n{summary}", ephemeral=True)
        else:
            for i in range(0, len(summary), MAX_LEN):
                await interaction.followup.send(f"\U0001f4dd **Summary** ({src}) (part {i//MAX_LEN+1}):\n\n{summary[i:i+MAX_LEN]}", ephemeral=True)


# ── Translate ────────────────────────────────────────────────────────────

async def handle_translate_command(
    interaction: discord.Interaction,
    target_language: str,
    source_language: str | None = None,
) -> None:
    """Translate text via the AI provider."""
    from config.settings import settings  # noqa: local-only
    from bot_core import _chat_history  # noqa: local-only

    parts   = target_language.split(":", 1)
    tgt     = parts[0].strip()
    text_to = parts[1].strip() if len(parts) > 1 else None

    if not text_to:
        guild_id  = interaction.guild_id or 0
        ch_hist   = _chat_history.get(guild_id, {}).get(interaction.channel_id, [])
        last_user = [m["content"] for m in reversed(ch_hist) if m["role"] == "user"]
        text_to   = last_user[0] if last_user else None

    if not text_to:
        await interaction.followup.send(
            "\u26a0\ufe0f No text to translate.  Provide text as ``/translate Spanish: Hello world``.", ephemeral=True
        )
        return

    src_clause = f" from {source_language}" if source_language else ""

    client = AsyncOpenAI(api_key=settings.INFER_API_KEY or "local-model-key", base_url=settings.INFER_URL)
    resp = await client.chat.completions.create(
        model=settings.DEFAULT_MODEL,
        messages=[
            {"role": "system",
             "content": f"Translate{src_clause} text into {tgt}. Return ONLY translated text."},
            {"role": "user", "content": text_to},
        ],
        temperature=0.3, max_tokens=4096,
    )
    translated = resp.choices[0].message.content or "(translation failed)"

    MAX_LEN = 1900
    if len(translated) <= MAX_LEN:
        await interaction.followup.send(f"\U0001f310 Translated to **{tgt}**:\n\n{translated}", ephemeral=True)
    else:
        for i in range(0, len(translated), MAX_LEN):
            await interaction.followup.send(
                f"\U0001f310 Translated to **{tgt}** (part {i//MAX_LEN+1})\n\n{translated[i:i+MAX_LEN]}", ephemeral=True
            )
