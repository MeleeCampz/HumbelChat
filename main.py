"""Main entry point for the Discord AI bot.

This file wires everything together — bot setup, event handlers, slash commands,
import from config / kb / bot_core modules, and startup logic.

OWUI-dependent code (~400 lines) has been removed; KB operations now use native
filesystem reads via kb.storage and kb.reader instead of the OpenWebUI API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import sys
import uuid
from dotenv import load_dotenv

load_dotenv()

import discord
from discord.ext import commands
import discord.app_commands as app_commands
from openai import AsyncOpenAI
import httpx

# ── Import our migrated modules ────────────────────────────────────────
from config.settings import settings, _Settings
from config.characters import (
    load_characters,
    get_character,
    default_character,
)
from bot_core import ask_ai as core_ask_ai, ensure_history, clear_history as core_clear_history
from bot_core import _chat_history  # for prefix command history lookup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("bot")

INTENTS = discord.Intents.default()
INTENTS.messages       = True
INTENTS.guilds         = True
INTENTS.guild_messages  = True
INTENTS.message_content = True

# ── Character loading ════ (populates _CHARACTERS global from characters.json)
load_characters(pathlib.Path("characters.json"))

# Discord Choice objects — built from config at bot startup time (after character loading)
from config.characters import _CHARACTERS

_Character_RAW_CHOICES = [
    {"name": c.key, "value": c.key}
    for c in _CHARACTERS
]
_CHAR_CHOICES: list[app_commands.Choice[str]] = [
    app_commands.Choice(name=c["name"], value=c["value"])
    for c in _Character_RAW_CHOICES
]

# Per-guild / per-channel active character map
_ACTIVE_CHARACTERS: dict[tuple[int, int], str] = {}


def _get_active_character_key(gid: int | None, cid: int) -> str:
    if gid is not None and (gid, cid) in _ACTIVE_CHARACTERS:
        return _ACTIVE_CHARACTERS[(gid, cid)]
    return default_character().key


# ── Bot setup ─────────────────────────────────────────────────────────
bot = commands.Bot(
    command_prefix=settings.BOT_PREFIX,
    intents=INTENTS,
)


# ════════════════════════════════════════════════════════════════════════
#  Slash commands (import-based — delegates to dedicated modules)
# ════════════════════════════════════════════════════════════════════════

@bot.tree.command(
    name="ai",
    description="Send a prompt to the AI and get a reply.",
)
@app_commands.choices(character=_CHAR_CHOICES)
async def ai_command(
    interaction: discord.Interaction,
    message: str,
    character: app_commands.Choice[str] | None = None,
):
    """AI chat command — wired to ``commands/ai_command.py``."""
    # Delegate fully
    from commands.ai_command import handle_ai_command

    char_name = character.value if character else None
    await handle_ai_command(interaction, message, char_name)


@bot.tree.command(
    name="character",
    description="Manage AI character/persona settings.",
)
@app_commands.describe(
    action="list / set / show / reset",
    name="Character key (e.g. System or Trixy Smoldersome)",
)
async def character_command(
    interaction: discord.Interaction,
    action: str = "list",
    name: str | None = None,
):
    """Switch or list characters — uses config.characters globals."""
    await interaction.response.defer(ephemeral=True)

    active_key = default_character().key
    from config.characters import _CHARACTERS
    for c in _CHARACTERS:
        if c.key == _get_active_character_key(interaction.guild_id, interaction.channel_id):
            active_key = c.key
            break

    current_char = get_character(active_key)

    if action == "list":
        lines = ["**Available characters:**\n"]
        for char in __import__('config.characters', fromlist=['_CHARACTERS'])._CHARACTERS:
            marker = f" ← current" if char.key == active_key else ""
            lines.append(f"  • `{char.key}` — display: `{char.display or char.key}`{marker}")
        await interaction.followup.send("\n".join(lines), ephemeral=True)

    elif action == "set":
        if name is None:
            await interaction.followup.send("Please provide a character key: `/character set <name>`", ephemeral=True)
            return
        import config.characters as _cc
        char_obj = get_character(name) or (char for char in _cc._CHARACTERS if getattr(char, "key", "") == name)
        if char_obj is None:
            avail = ", ".join(f"`{c.key}`" for c in _cc._CHARACTERS)
            await interaction.followup.send(
                f"Unknown character ``{name}``. Available: {avail}", ephemeral=True
            )
            return
        if interaction.guild_id is not None:
            _ACTIVE_CHARACTERS[(interaction.guild_id, interaction.channel_id)] = char_obj.key
        await interaction.followup.send(
            f"Switched to **{char_obj.display}** (model: ``{char_obj.model or '(none set)'}\n)", ephemeral=True
        )

    elif action == "show":
        display = current_char.display if current_char else "Default"
        model   = current_char.model if current_char else "(not set)"
        await interaction.followup.send(
            f"**Current character:** `{display}`\n**Model:** ``{model}\n``", ephemeral=True
        )

    elif action == "reset":
        if interaction.guild_id is not None:
            _ACTIVE_CHARACTERS.pop((interaction.guild_id, interaction.channel_id), None)
        default_name = default_character().display or "Default"
        await interaction.followup.send(
            f"Reverted to default character: **{default_name}**", ephemeral=True
        )

    else:
        await interaction.followup.send(
            f"Unknown action ``{action}``. Use: list, set, show, reset.", ephemeral=True
        )


@bot.tree.command(name="clear_history", description="Clear conversation history for this channel.")
async def clear_history_command(interaction: discord.Interaction):
    guild_id = interaction.guild_id or 0
    cid      = interaction.channel_id
    await core_clear_history(guild_id, cid)
    await interaction.response.send_message("Conversation history cleared.", ephemeral=True)


# ════════════════════════════════════════════════════════════════════════
#  Utility slash commands (ocr, summarize, translate, remind)
# ════════════════════════════════════════════════════════════════════════

@bot.tree.command(
    name="remind",
    description="Schedule a reminder for yourself.",
)
@app_commands.describe(
    time_value="Amount of time (number)",
    time_unit="Unit of time (seconds, minutes, hours)",
    message="What you want to be reminded about",
)
async def remind_command(
    interaction: discord.Interaction,
    time_value: int,
    time_unit: str,
    message: str,
):
    """Schedule a one-time reminder. Stays inline for now."""

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
    confirmation = prompt_text + f'\n📝 I"ll ping you with: "{message}"'
    await interaction.followup.send(confirmation, ephemeral=True)


async def _send_reminder(channel_id: int, message: str, delay: int) -> None:
    """Background reminder sender. Sleeps *delay* seconds before sending."""
    await asyncio.sleep(delay)
    try:
        chan = bot.get_channel(channel_id)
        if chan:
            await chan.send(f"⏰ **Reminder:** {message}")
    except Exception as e:
        log.error("Failed to send reminder: %s", e)


@bot.tree.command(
    name="upload_kb",
    description="Upload a file to the knowledge base for RAG.",
)
@app_commands.describe(
    kb_name="Override KB name (defaults to KB_DEFAULT_KB)",
    url="Remote URL to fetch (optional if attachment provided)",
)
async def upload_kb_command(
    interaction: discord.Interaction,
    kb_name: str | None = None,
    url: str | None = None,
    file: discord.Attachment = None,
):
    """Upload KB via native filesystem — delegates to commands/kb_commands."""
    await interaction.response.defer(ephemeral=True)
    from commands.kb_commands import handle_upload_kb
    await handle_upload_kb(interaction, kb_name=kb_name, url=url, attachment=file)


@bot.tree.command(name="list_kb_docs", description="List all documents in the knowledge base.")
async def list_kb_command(interaction: discord.Interaction):
    """List KB docs from local filesystem — delegates to commands/kb_commands."""
    await interaction.response.defer(ephemeral=True)
    from commands.kb_commands import handle_list_kb_docs
    await handle_list_kb_docs(interaction)


@bot.tree.command(name="reindex_kb", description="Reindex all files in the knowledge base.")
async def reindex_kb_command(interaction: discord.Interaction):
    """Trigger reindexing of all files in the KB."""
    from commands.kb_commands import handle_reindex_kb
    await handle_reindex_kb(interaction)


@bot.tree.command(
    name="ocr",
    description="Extract text from an image (OCR).",
)
async def ocr_command(interaction: discord.Interaction, image: discord.Attachment = None):
    """Vision-based OCR — stays inline because it's provider-specific and simple."""
    await interaction.response.defer(ephemeral=True)

    if not image:
        await interaction.followup.send("⚠️ Please attach an image.", ephemeral=True)
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

    import base64 as _b64
    b64   = _b64.b64encode(img_data).decode("utf-8")
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
        await interaction.followup.send(f"🔍 Extracted text:\n\n{reply}", ephemeral=True)
    else:
        for i in range(0, len(reply), MAX_LEN):
            await interaction.followup.send(f"🔍 OCR (part {i//MAX_LEN+1})\n\n{reply[i:i+MAX_LEN]}", ephemeral=True)


@bot.tree.command(name="summarize", description="Summarize text from a file or recent conversation.")
@app_commands.describe(file_url="Optional URL to fetch text content")
async def summarize_command(interaction: discord.Interaction, file_url: str | None = None):
    """Summarize — uses inline chat with provider."""
    await interaction.response.defer(ephemeral=True)

    text = ""
    src  = ""
    if file_url:
        async with httpx.AsyncClient() as client:
            resp    = await client.get(file_url)
            resp.raise_for_status()
            text     = resp.text[:32000]
            src       = f"file from `{file_url[:80]}...`"
    else:
        guild_id = interaction.guild_id or 0
        ch_history = _chat_history.get(guild_id, {}).get(interaction.channel_id, [])
        parts = []
        for msg in ch_history[-30:]:
            role_name = {"user": "User", "assistant": "AI"}.get(msg["role"], msg["role"])
            parts.append(f"[{role_name}]: {msg['content']}")
        text     = "\n\n".join(parts) if parts else "(no history)"
        src      = "recent conversation"

    if not text.strip():
        await interaction.followup.send("⚠️ Nothing to summarize.", ephemeral=True)
        return

    client = AsyncOpenAI(
        api_key=settings.INFER_API_KEY or "local-model-key",
        base_url=settings.INFER_URL,
    )
    resp = await client.chat.completions.create(
        model=settings.DEFAULT_MODEL,
        messages=[
            {"role": "system",
             "content": f"Summarize the following text from {src}. Be concise but complete."},
            {"role": "user", "content": text},
        ],
        temperature=0.3, max_tokens=2048,
    )
    summary = resp.choices[0].message.content or "(empty)"

    MAX_LEN = 1900
    if len(summary) <= MAX_LEN:
        await interaction.followup.send(f"📝 Summary of {src}:\n\n{summary}", ephemeral=True)
    else:
        for i in range(0, len(summary), MAX_LEN):
            await interaction.followup.send(f"📝 Summary (part {i//MAX_LEN+1})\n\n{summary[i:i+MAX_LEN]}", ephemeral=True)


@bot.tree.command(name="translate", description="Translate text to a language.")
@app_commands.describe(
    target_language="Target language (optionally with source: 'Spanish: Hello')",
    source_language="Optional source language (default: auto-detect)",
)
async def translate_command(
    interaction: discord.Interaction,
    target_language: str,
    source_language: str | None = None,
):
    """Translate — inline provider call."""
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
            "⚠️ No text to translate.  Provide text as ``/translate Spanish: Hello world``.", ephemeral=True
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
        await interaction.followup.send(f"🌐 Translated to **{tgt}**:\n\n{translated}", ephemeral=True)
    else:
        for i in range(0, len(translated), MAX_LEN):
            await interaction.followup.send(
                f"🌐 Translated to **{tgt}** (part {i//MAX_LEN+1})\n\n{translated[i:i+MAX_LEN]}", ephemeral=True
            )


# ════════════════════════════════════════════════════════════════════════
#  Event handlers + startup
# ════════════════════════════════════════════════════════════════════════

@bot.event
async def on_ready():
    log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
    if bot.guilds:
        # For testing locally, we use the first guild found to sync commands quickly.
        target_guild = bot.guilds[0]
        bot.tree.copy_global_to(guild=target_guild)
        await bot.tree.sync(guild=target_guild)
    else:
        await bot.tree.sync()
    char_names = [c.name for c in _CHAR_CHOICES]
    log.info("Characters loaded: %s", ", ".join(char_names) or "(none)")


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    content = message.content.strip()
    if not content.startswith(settings.BOT_PREFIX):
        return

    prompt = content[len(settings.BOT_PREFIX):].strip()
    if not prompt:
        await message.channel.send(f"Usage: {settings.BOT_PREFIX} <your question>")
        return

    guild_id  = message.guild_id or 0
    log.info("%s (%s) in #%s: %s",
             message.author, message.author.id, message.channel.name, prompt[:80])

    await message.channel.typing()

    sys_char = default_character()
    sys_model = sys_char.model if sys_char else settings.DEFAULT_MODEL
    reply, _extra = await core_ask_ai(
        prompt,
        model_slug=sys_model or "",
        guild_id=guild_id,
        channel_id=message.channel.id,
        username=message.author.display_name or "",
    )

    # Inline paragraph-aware chunking for prefix commands
    from utils.response_splitter import send_long_response
    await send_long_response(message, reply, str(sys_char.display))


# ── Single-instance lock ───────────────────────────────────────────────

PIDFILE = pathlib.Path(__file__).parent / ".bot.pid"
SOCK_FILE = pathlib.Path(__file__).parent / ".bot.sock"


def _enforce_single_instance():
    """Exit immediately if another instance of this bot is already running."""
    import socket as _socket

    # Port check
    try:
        test_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        test_sock.settimeout(0.5)
        test_sock.connect(("127.0.0.1", 18765))
        test_sock.close()
        log.info("Another bot instance running (port 18765). Exiting.")
        sys.exit(0)
    except Exception:
        pass

    # PID file check
    if PIDFILE.exists():
        try:
            old_pid = int(PIDFILE.read_text().strip())
            import os as _os, signal as _signal
            _os.kill(old_pid, 0)
            log.info("Another bot instance (PID %d) is already running. Exiting.", old_pid)
            sys.exit(0)
        except (ProcessLookupError, ValueError):
            PIDFILE.unlink(missing_ok=True)

    # Own PID
    PIDFILE.write_text(str(os.getpid()))
    import atexit as _atexit

    @_atexit.register
    def _cleanup_lock():
        try:
            PIDFILE.unlink(missing_ok=True)
        except OSError:
            pass


# ── Startup ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not settings.DISCORD_TOKEN:
        log.error("Please set the DISCORD_BOT_TOKEN environment variable.")
        raise SystemExit(1)

    _enforce_single_instance()

    log.info("Connecting to AI backend at: %s", settings.INFER_URL)
    log.info("Bot prefix: `%s` — Default character: %s",
             settings.BOT_PREFIX, default_character().display or "Default")

    bot.run(settings.DISCORD_TOKEN)

