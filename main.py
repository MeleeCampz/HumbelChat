"""Main entry point for the Discord AI bot.

Wires everything together: bot setup, event handlers, slash command
registrations, character loading, and startup logic.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

import discord
from discord.ext import commands
import discord.app_commands as app_commands

from config.settings import (
    BOT_PREFIX,
    KB_PATH,
    CHARACTERS_FILE,
    INFER_URL,
    DEFAULT_MODEL,
    DISCORD_TOKEN,
    CHAT_HISTORY_RESET,
)
from config.characters import load_characters, default_character, get_character_choices
from bot_core.ai_client import ask_ai as core_ask_ai
from bot_core.ai_client import RateLimitError
from bot_core.health import start_backend_health_probe
from bot_core.reminders import rearm_pending_reminders
from utils.background_tasks import spawn_tracked_task
from utils.channel_queue import channel_slot
from utils.kb_utils import log_top_kb_files
from utils.response_splitter import send_long_response
from utils.typing_loop import typing_loop_task

# ── Logging setup ───────────────────────────────────────────────────────
# Handlers are attached to the "bot" logger (not root) and the bot hierarchy
# does not propagate, so every record is emitted exactly once. This fixes the
# duplicate-line symptom where the console handler and discord's own logging
# both bubbled records up to the root logger (see code review §2.6).
_NO_FILE_LOGS = os.environ.get("BOT_NO_LOG_FILES") == "1"

log = logging.getLogger("bot")
log.propagate = False

if _NO_FILE_LOGS:
    # Tests / minimal environments: console-only output, no log files.
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.INFO)
else:
    LOG_DIR = pathlib.Path(__file__).resolve().parent / "logs"
    LOG_DIR.mkdir(exist_ok=True)

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    bot_log = RotatingFileHandler(
        LOG_DIR / "bot.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    bot_log.setLevel(logging.INFO)
    bot_log.setFormatter(log_formatter)

    dev_log = RotatingFileHandler(
        LOG_DIR / "dev.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    dev_log.setLevel(logging.DEBUG)
    dev_log.setFormatter(log_formatter)

    log.addHandler(console_handler)
    log.addHandler(bot_log)
    log.addHandler(dev_log)
    log.setLevel(logging.INFO)

    # discord.py logs through the "discord" logger; keep it console-only and
    # stop it from propagating to root (root has no handlers anyway, so this
    # prevents duplicate lines if a handler is ever added to root).
    _discord_logger = logging.getLogger("discord")
    _discord_logger.handlers.clear()
    _discord_logger.addHandler(logging.StreamHandler(sys.stdout))
    _discord_logger.setLevel(logging.INFO)
    _discord_logger.propagate = False

    # The knowledge-base modules log under their own top-level "kb" namespace
    # (kb.index, kb.embedder, kb.retrievers, ...). Attach the same handlers so
    # index-build / embedding failures land in bot.log and dev.log instead of
    # only flashing past on the console via Python's last-resort handler.
    _kb_logger = logging.getLogger("kb")
    _kb_logger.handlers.clear()
    _kb_logger.addHandler(console_handler)
    _kb_logger.addHandler(bot_log)
    _kb_logger.addHandler(dev_log)
    _kb_logger.setLevel(logging.INFO)

# ── Intents ─────────────────────────────────────────────────────────────
INTENTS = discord.Intents.default()
INTENTS.messages = True
INTENTS.guilds = True
INTENTS.guild_messages = True
INTENTS.message_content = True

# ── Character loading ───────────────────────────────────────────────────
load_characters(CHARACTERS_FILE)

# Restore conversation history + active-character selections from disk.
# §debt: CHAT_HISTORY_RESET ("clear"/1/true/yes) is now actually consumed —
# it used to be parsed in settings but never read anywhere.
from bot_core.history import load_persisted, reset_all_history
load_persisted()
if CHAT_HISTORY_RESET:
    log.info("CHAT_HISTORY_RESET set — wiping all stored conversation history.")
    reset_all_history()

# Restore session state (active session + queued next-session reminders).
# Nothing to re-arm — reminders fire from the /start_session handler.
from bot_core.sessions import load_persisted as load_sessions
load_sessions()

# Built *after* load_characters() so the choices reflect the actual registry.
# Reading via get_character_choices() also avoids the import-by-value trap
# (``load_characters`` rebinds the module global instead of mutating the
# originally-imported list in place).
_CHAR_CHOICES: list[app_commands.Choice[str]] = [
    app_commands.Choice(name=c["name"], value=c["value"])
    for c in get_character_choices()
]

# ── Bot setup ───────────────────────────────────────────────────────────
bot = commands.Bot(
    command_prefix=BOT_PREFIX,
    intents=INTENTS,
)
# §4.4: Strong references to background tasks live in utils/background_tasks.
# This list exists only for diagnostics and direct cancellation if needed.
bot.typing_tasks: list[asyncio.Task] = []

# ── One-time command sync on first startup ──────────────────────────────
# Track whether we've synced commands to avoid duplicate registrations.
# Use a marker file in the project root; if it exists, we skip auto-sync.
SYNC_MARKER = pathlib.Path(__file__).parent / ".commands_synced"


async def _ensure_commands_synced() -> None:
    """Sync commands once on first run; skip on subsequent restarts.

    This avoids the duplication problem caused by syncing on every
    on_ready event (which fires on every reconnect).
    """
    if SYNC_MARKER.exists():
        log.info("Commands already synced previously; skipping auto-sync.")
        return

    log.info("First startup detected; syncing commands globally...")
    try:
        await bot.tree.sync()
        SYNC_MARKER.touch(exist_ok=True)
        log.info("Commands synced and marker written.")
    except Exception as e:
        log.error("Initial command sync failed: %s", e)


# ════════════════════════════════════════════════════════════════════════
#  Slash commands — delegate to command modules
# ════════════════════════════════════════════════════════════════════════

@bot.tree.command(name="ai", description="Send a prompt to the AI and get a reply.")
@app_commands.choices(character=_CHAR_CHOICES)
async def ai_command(
    interaction: discord.Interaction,
    message: str,
    character: app_commands.Choice[str] | None = None,
) -> None:
    """AI chat command — delegated to commands/ai_command.py."""
    from commands.ai_command import handle_ai_command
    char_name = character.value if character else None
    await handle_ai_command(interaction, message, char_name)


@bot.tree.command(name="character", description="Manage AI character/persona settings.")
@app_commands.describe(action="list / set / show / reset", name="Character key (e.g. System)")
async def character_command(
    interaction: discord.Interaction,
    action: str = "list",
    name: str | None = None,
) -> None:
    """Switch or list characters — delegated to commands/character_commands.py."""
    from commands.character_commands import handle_character_command
    await handle_character_command(interaction, action=action, name=name)


@bot.tree.command(name="clear_history", description="Clear conversation history for this channel.")
async def clear_history_command(interaction: discord.Interaction) -> None:
    """Clear history — delegated to commands/clear_history_command.py."""
    from commands.clear_history_command import handle_clear_history_command
    await handle_clear_history_command(interaction)


@bot.tree.command(name="remind", description="Schedule a reminder for yourself.")
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
) -> None:
    """Schedule a reminder — delegated to commands/utility_commands.py."""
    from commands.utility_commands import handle_remind_command
    await handle_remind_command(interaction, time_value, time_unit, message)


@bot.tree.command(name="ocr", description="Extract text from an image (OCR).")
async def ocr_command(interaction: discord.Interaction, image: discord.Attachment | None = None) -> None:
    """Vision-based OCR — delegated to commands/utility_commands.py."""
    from commands.utility_commands import handle_ocr_command
    await handle_ocr_command(interaction, image=image)


@bot.tree.command(name="summarize", description="Summarize recent chat history or a file from a URL.")
@app_commands.describe(file_url="Optional URL to fetch text content")
async def summarize_command(interaction: discord.Interaction, file_url: str | None = None) -> None:
    """Summarize — delegated to commands/utility_commands.py."""
    from commands.utility_commands import handle_summarize_command
    await handle_summarize_command(interaction, file_url=file_url)


@bot.tree.command(name="translate", description="Translate text into a target language.")
@app_commands.describe(
    target_language="Target language (optionally with source: 'Spanish: Hello')",
    source_language="Optional source language (default: auto-detect)",
)
async def translate_command(
    interaction: discord.Interaction,
    target_language: str,
    source_language: str | None = None,
) -> None:
    """Translate — delegated to commands/utility_commands.py."""
    from commands.utility_commands import handle_translate_command
    await handle_translate_command(interaction, target_language, source_language)


@bot.tree.command(name="upload_kb", description="Upload a file or URL into the knowledge base.")
@app_commands.describe(
    file="File attachment to upload",
    url="URL to download and upload as a KB document",
    subfolder="Optional subfolder inside the KB directory",
)
async def upload_kb_command(
    interaction: discord.Interaction,
    file: discord.Attachment | None = None,
    url: str | None = None,
    subfolder: str | None = None,
) -> None:
    """Upload a document to the knowledge base — delegated to commands/kb_commands.py."""
    from commands.kb_commands import handle_upload_kb
    await handle_upload_kb(interaction, attachment=file, url=url, subfolder=subfolder)


@bot.tree.command(name="list_kb_docs", description="List all documents in the knowledge base.")
@app_commands.describe(path="Optional subfolder path to list (omit for root-level overview).")
async def list_kb_docs_command(interaction: discord.Interaction, path: str | None = None) -> None:
    """List KB documents — delegated to commands/kb_commands.py."""
    from commands.kb_commands import handle_list_kb_docs
    await handle_list_kb_docs(interaction, subfolder_path=path)


@bot.tree.command(name="reindex_kb", description="Re-index all files in the knowledge base for semantic search.")
async def reindex_kb_command(interaction: discord.Interaction) -> None:
    """Re-index KB files — delegated to commands/kb_commands.py."""
    from commands.kb_commands import handle_reindex_kb
    await handle_reindex_kb(interaction)


@bot.tree.command(
    name="sync",
    description="Re-sync all slash commands with Discord (fixes duplicated command listings).",
)
async def sync_command(interaction: discord.Interaction) -> None:
    """Re-sync commands — delegated to commands/sync_command.py."""
    from commands.sync_command import handle_sync_command
    await handle_sync_command(interaction)


@bot.tree.command(name="start_session", description="Start a new work session.")
@app_commands.describe(name="Optional custom name for the session")
async def start_session_command(interaction: discord.Interaction, name: str | None = None) -> None:
    """Start session — delegated to commands/session_commands.py."""
    from commands.session_commands import handle_start_session
    await handle_start_session(interaction, name=name)


@bot.tree.command(name="end_session", description="End the current session and write its overview.")
@app_commands.describe(name="Optional new name for the session (renames it in the notes file)")
async def end_session_command(interaction: discord.Interaction, name: str | None = None) -> None:
    """End session — delegated to commands/session_commands.py."""
    from commands.session_commands import handle_end_session
    await handle_end_session(interaction, name=name)


@bot.tree.command(
    name="remind_next_session",
    description="Queue a reminder that is delivered when the next session starts.",
)
@app_commands.describe(message="What you want to be reminded about at the next session start")
async def remind_next_session_command(interaction: discord.Interaction, message: str) -> None:
    """Next-session reminder — delegated to commands/session_commands.py."""
    from commands.session_commands import handle_remind_next_session
    await handle_remind_next_session(interaction, message)


@bot.tree.command(name="session_notes", description="Add notes to the current session or view them.")
@app_commands.describe(action="add / view", note="The note text (required for action: add)")
async def session_notes_command(
    interaction: discord.Interaction,
    action: str = "view",
    note: str | None = None,
) -> None:
    """Session notes — delegated to commands/session_commands.py."""
    from commands.session_commands import handle_session_notes
    await handle_session_notes(interaction, action=action, note=note)


# ════════════════════════════════════════════════════════════════════════
#  Event handlers
# ════════════════════════════════════════════════════════════════════════

@bot.event
async def on_ready() -> None:
    log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)

    # One-time sync on first run only — avoids command duplication from
    # repeated syncs on every reconnect.
    await _ensure_commands_synced()

    # NOTE: We no longer auto-sync on every on_ready.
    # Auto-syncing on every reconnect causes command duplication in Discord's cache.
    # Commands are registered once when the bot starts; if they need re-syncing,
    # use the /sync command (which is always available since it's registered at startup).

    # §3.9: fail fast if the AI backend is unreachable; optionally start the
    # periodic liveness probe (AI_HEALTH_CHECK_INTERVAL).
    start_backend_health_probe(bot)

    log_top_kb_files(KB_PATH)

    char_names = [c.name for c in _CHAR_CHOICES]
    log.info("Characters loaded: %s", ", ".join(char_names) or "(none)")

    # Re-arm any persisted reminders that haven't fired yet (survives restarts).
    # on_ready also fires on full gateway reconnects; rearm_pending_reminders()
    # cancels existing live tasks before replacing them (no double-fires).
    n_rearmed = rearm_pending_reminders()
    if n_rearmed:
        log.info("Re-armed %d pending reminder(s)", n_rearmed)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author == bot.user:
        return

    content = message.content.strip()
    if not content.startswith(BOT_PREFIX):
        return

    prompt = content[len(BOT_PREFIX):].strip()
    if not prompt:
        await message.channel.send(f"Usage: {BOT_PREFIX} <your question>")
        return

    guild_id = message.guild_id or 0
    log.info(
        "%s (%s) in #%s: %s",
        message.author,
        message.author.id,
        message.channel.name,
        prompt[:80],
    )

    # §4.4: the tracking set in utils/background_tasks keeps the task alive
    # even if no local references remain.
    typing_task = spawn_tracked_task(
        typing_loop_task(message.channel),
        name=f"typing-{message.channel.id}",
    )
    # Diagnostics list only — prune finished tasks so it can't grow unbounded.
    bot.typing_tasks = [t for t in bot.typing_tasks if not t.done()]
    bot.typing_tasks.append(typing_task)

    sys_char = default_character()
    sys_model = sys_char.model if sys_char else DEFAULT_MODEL

    # Hold this channel's reply slot for the ENTIRE request + delivery so a
    # concurrent prefix command in the same channel can't interleave its
    # messages with this one (see utils/channel_queue.py).
    async with channel_slot(message.channel.id, name="prefix-command"):
        try:
            reply, _extra = await core_ask_ai(
                prompt,
                model_slug=sys_model or "",
                guild_id=guild_id,
                channel_id=message.channel.id,
                username=message.author.display_name or "",
                user_id=message.author.id,
            )
        except RateLimitError as e:
            typing_task.cancel()
            await message.channel.send(f"⏳ Rate limit reached — please try again in {e.retry_after}s.")
            return
        except ValueError as e:
            typing_task.cancel()
            await message.channel.send(f"⚠️ {e}")
            return

        typing_task.cancel()

        await send_long_response(message, reply, str(sys_char.display))


# ── Single-instance lock ────────────────────────────────────────────────

PIDFILE = pathlib.Path(__file__).parent / ".bot.pid"

def _enforce_single_instance() -> None:
    """Exit immediately if another instance of this bot is already running."""
    # PID file check (the port check was removed — nothing ever bound 18765,
    # so it was dead code; see code review §1.5)
    if PIDFILE.exists():
        try:
            old_pid = int(PIDFILE.read_text().strip())

            os.kill(old_pid, 0)
            log.info("Another bot instance (PID %d) is already running. Exiting.", old_pid)
            sys.exit(0)
        except (ProcessLookupError, ValueError):
            PIDFILE.unlink(missing_ok=True)

    # Own PID
    PIDFILE.write_text(str(os.getpid()))
    import atexit as _atexit

    @_atexit.register
    def _cleanup_lock() -> None:
        try:
            PIDFILE.unlink(missing_ok=True)
        except OSError:
            pass


# ── Startup ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        log.error("Please set the DISCORD_BOT_TOKEN environment variable.")
        raise SystemExit(1)

    _enforce_single_instance()

    log.info("Connecting to AI backend at: %s", INFER_URL)
    log.info(
        "Bot prefix: `%s` — Default character: %s",
        BOT_PREFIX,
        default_character().display or "Default",
    )

    bot.run(DISCORD_TOKEN)
