"""Main entry point for the Discord AI bot.

This file wires everything together — bot setup, event handlers, slash commands,
import from config / kb / bot_core modules, and startup logic.

OWUI-dependent code (~400 lines) has been removed; KB operations now use native
filesystem reads via kb.storage and kb.reader instead of the OpenWebUI API.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys
from dotenv import load_dotenv

load_dotenv()

import discord
from discord.ext import commands
import discord.app_commands as app_commands
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
    name="Character key (e.g. System or Assistant)",
)
async def character_command(
    interaction: discord.Interaction,
    action: str = "list",
    name: str | None = None,
):
    """Switch or list characters — wired to commands/character_commands.py."""
    from commands.character_commands import handle_character_command
    await handle_character_command(interaction, action=action, name=name)


@bot.tree.command(name="clear_history", description="Clear conversation history for this channel.")
async def clear_history_command(interaction: discord.Interaction):
    """Clear history — wired to commands/clear_history_command.py."""
    from commands.clear_history_command import handle_clear_history_command
    await handle_clear_history_command(interaction)


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
    """Schedule a reminder — wired to commands/utility_commands.py."""
    from commands.utility_commands import handle_remind_command
    await handle_remind_command(interaction, time_value, time_unit, message)


@bot.tree.command(
    name="ocr",
    description="Extract text from an image (OCR).",
)
async def ocr_command(interaction: discord.Interaction, image: discord.Attachment = None):
    """Vision-based OCR — wired to commands/utility_commands.py."""
    from commands.utility_commands import handle_ocr_command
    await handle_ocr_command(interaction, image=image)


@bot.tree.command(
    name="summarize",
    description="Summarize recent chat history or a file from a URL.",
)
@app_commands.describe(file_url="Optional URL to fetch text content")
async def summarize_command(interaction: discord.Interaction, file_url: str | None = None):
    """Summarize — wired to commands/utility_commands.py."""
    from commands.utility_commands import handle_summarize_command
    await handle_summarize_command(interaction, file_url=file_url)


@bot.tree.command(
    name="translate",
    description="Translate text into a target language.",
)
@app_commands.describe(
    target_language="Target language (optionally with source: 'Spanish: Hello')",
    source_language="Optional source language (default: auto-detect)",
)
async def translate_command(
    interaction: discord.Interaction,
    target_language: str,
    source_language: str | None = None,
):
    """Translate — wired to commands/utility_commands.py."""
    from commands.utility_commands import handle_translate_command
    await handle_translate_command(interaction, target_language, source_language)


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

    from utils.kb_utils import log_top_kb_files
    log_top_kb_files(settings.KB_PATH)

    char_names = [c.display for c in _CHAR_CHOICES]
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

