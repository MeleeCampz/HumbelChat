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
from bot_core.ai_runs import register_run, clear_run
from commands.ai_command import run_ai_turn, start_typing, notify_if_queued
from bot_core.health import start_backend_health_probe
from bot_core.reminders import rearm_pending_reminders
from utils.background_tasks import spawn_tracked_task
from utils.kb_utils import log_top_kb_files

# ── Logging setup ───────────────────────────────────────────────────────
# Handlers are attached to the "bot" logger (not root) and the bot hierarchy
# does not propagate, so every record is emitted exactly once. This fixes the
# duplicate-line symptom where the console handler and discord's own logging
# both bubbled records up to the root logger (see code review §2.6).
_NO_FILE_LOGS = os.environ.get("BOT_NO_LOG_FILES") == "1"

log = logging.getLogger("bot")
log.propagate = False


def _configure_discord_logger(dev_handler=None) -> None:
    """Configure the discord.py logger (P3 #34).

    The discord logger is kept console-only and stop-propagating. Its level is
    driven by ``BOT_DISCORD_LOG_LEVEL`` **in both** file and console-only modes
    — previously the level (and the whole config) lived inside the file-logging
    branch, so ``BOT_DISCORD_LOG_LEVEL`` was silently ignored when
    ``BOT_NO_LOG_FILES=1``. A ``dev_handler`` (dev.log) is attached only when a
    file logger is present, so voice-gateway/DAVE debug lines reach dev.log.
    """
    _discord_logger = logging.getLogger("discord")
    _discord_logger.handlers.clear()
    _discord_logger.addHandler(logging.StreamHandler(sys.stdout))
    _discord_console_level = os.environ.get("BOT_DISCORD_LOG_LEVEL", "INFO")
    _discord_logger.setLevel(getattr(logging, _discord_console_level.upper(), logging.INFO))
    if dev_handler is not None:
        _discord_logger.addHandler(dev_handler)
    _discord_logger.propagate = False


if _NO_FILE_LOGS:
    # Tests / minimal environments: console-only output, no log files.
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.INFO)
    # P3 #34: still honour BOT_DISCORD_LOG_LEVEL in console-only mode.
    _configure_discord_logger()
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

    _configure_discord_logger(dev_handler=dev_log)

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
# P3 #33: the bot only serves *guild* prefix commands and slash commands — it
# never handles DM (non-guild) messages — so the deprecated ``messages`` intent
# is not set. ``guild_messages`` + ``message_content`` are enough for the
# prefix path; ``guilds`` for command/voice state. (Discord has deprecated the
# legacy non-guild ``messages`` intent, and it only matters if you add DM
# support later.)
INTENTS = discord.Intents.default()
INTENTS.guilds = True
INTENTS.guild_messages = True
INTENTS.message_content = True
# Voice recording: we need voice-state updates (to see who is in a channel and
# to resolve speaker display names) for /start_recording & /stop_recording.
INTENTS.voice_states = True

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

    On the first run we do a *full* sync via :mod:`bot_core.command_sync`,
    which also **purges** any stale guild-scoped / renamed registrations so the
    ``/`` menu shows each command exactly once. On later restarts we skip the
    auto-sync entirely — this avoids the duplication problem caused by syncing
    on every on_ready event (which fires on every reconnect). If commands ever
    get out of sync afterwards, use the ``/sync`` command.
    """
    if SYNC_MARKER.exists():
        log.info("Commands already synced previously; skipping auto-sync.")
        return

    log.info("First startup detected; syncing (and purging stale) commands...")
    try:
        from bot_core import command_sync
        report = await command_sync.sync_commands(bot)
    except Exception as e:
        # No marker on exception → the next restart retries the sync.
        log.error("Initial command sync failed (will retry on next restart): %s", e)
        return
    if report.get("error"):
        # The upload was rejected (e.g. bad command payload). Do NOT write the
        # marker — otherwise every future restart would skip the sync and new
        # commands would silently never reach Discord.
        log.error(
            "Initial command sync FAILED — commands not registered, marker NOT "
            "written (will retry on next restart): %s", report["error"],
        )
        return
    SYNC_MARKER.touch(exist_ok=True)
    log.info(
        "Commands synced and marker written. Removed stale: guilds=%s global=%s",
        [n for n, _ in report["guild_names"]], report["global_deleted"],
    )


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


@bot.tree.command(
    name="ai_stop",
    description="Cancel the in-flight /ai reply in this channel (P3 #25).",
)
async def ai_stop_command(interaction: discord.Interaction) -> None:
    """Stop an in-flight /ai — delegated to commands/ai_stop_command.py."""
    from commands.ai_stop_command import handle_ai_stop_command
    await handle_ai_stop_command(interaction)


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


@bot.tree.command(name="sync_kb", description="Re-index only new, renamed, edited or deleted KB files — unchanged files are skipped.")
async def sync_kb_command(interaction: discord.Interaction) -> None:
    """Fast diff-based KB sync — delegated to commands/kb_commands.py."""
    from commands.kb_commands import handle_sync_kb
    await handle_sync_kb(interaction)


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


@bot.tree.command(
    name="session_notes",
    description="Add notes (or a .txt/.md document) to the current session, or view them.",
)
@app_commands.describe(action="add / view")
@app_commands.describe(note="The note text (for action: add, free text)")
@app_commands.describe(file="A .txt or .md file to add as a whole document (for action: add)")
async def session_notes_command(
    interaction: discord.Interaction,
    action: str = "view",
    note: str | None = None,
    file: discord.Attachment | None = None,
) -> None:
    """Session notes — delegated to commands/session_commands.py."""
    from commands.session_commands import handle_session_notes
    await handle_session_notes(interaction, action=action, note=note, file=file)


@bot.tree.command(
    name="start_recording",
    description="Join your voice channel and record each participant's audio separately (for STT).",
)
async def start_recording_command(interaction: discord.Interaction) -> None:
    """Start voice recording — delegated to commands/recording_commands.py."""
    from commands.recording_commands import handle_start_recording
    await handle_start_recording(interaction)


@bot.tree.command(
    name="stop_recording",
    description="Stop the active voice recording and save per-speaker WAV files + a timestamped manifest.",
)
@app_commands.describe(leave_channel="Whether the bot should leave the voice channel afterwards (default true)")
@app_commands.describe(transcribe="Transcribe each speaker's audio with STT after stopping (default true; requires STT_ENABLED)")
async def stop_recording_command(
    interaction: discord.Interaction,
    leave_channel: bool = True,
    transcribe: bool = True,
) -> None:
    """Stop voice recording — delegated to commands/recording_commands.py."""
    from commands.recording_commands import handle_stop_recording
    await handle_stop_recording(interaction, leave_channel=leave_channel, transcribe=transcribe)


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

    # Crash durability: attach the voice recorder (installs the SIGTERM flush
    # handler) and recover any recording left open by an unclean shutdown.
    try:
        _recover_crashed_recordings(bot)
    except Exception:  # pragma: no cover - never block startup on recovery
        log.exception("Crashed-recording recovery failed (continuing)")


@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError,
) -> None:
    """P1 #11: global error handler for application (slash) commands.

    Without this, any unhandled exception in a command body surfaces in
    Discord as the generic "Application command failed" popup with no log
    trail. ``bot.tree.on_error`` is invoked for every top-level command
    failure (``_from_interaction`` catches it, wrapping non-``AppCommandError``
    exceptions into ``CommandInvokeError``) — see ``discord.app_commands.tree``.
    We log the full traceback and send a short, friendly *ephemeral* follow-up
    so only the invoking user sees it.
    """
    cmd = getattr(interaction, "command", None)
    log.error(
        "Application command %r failed with %s: %s", getattr(cmd, "name", "?"),
        type(error).__name__, error,
        exc_info=error,
    )

    # Derive a user-facing message. ``CommandInvokeError`` carries the *real*
    # exception in ``.original`` (and as ``__cause__``), so we can name the
    # original cause without leaking a raw traceback.
    original = getattr(error, "original", None) or error
    if isinstance(error, app_commands.CommandSignatureMismatch):
        user_msg = "⚠️ The bot ran into an internal error on that command. Please try again — and let the owner know if it keeps happening."
    elif isinstance(error, app_commands.CheckFailure):
        user_msg = "⚠️ You don't have permission to use this command."
    elif isinstance(error, app_commands.CommandInvokeError):
        user_msg = f"⚠️ {type(original).__name__}: {original}"
    else:
        user_msg = f"⚠️ {type(error).__name__}: {error}"

    # Best-effort: this handler may run after the interaction already responded
    # (deferred → followed up) or not.  Try the primary response first, then
    # fall back to a follow-up.
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message(user_msg, ephemeral=True)
            return
    except Exception:
        pass
    try:
        await interaction.followup.send(user_msg, ephemeral=True)
    except Exception as e:  # pragma: no cover - purely defensive
        log.warning("on_app_command_error: follow-up send failed: %s", e)


@bot.event
async def on_shutdown() -> None:
    """P1 #12: graceful-close hook.

    Runs on ``SIGTERM``/``SIGINT`` (the only signals discord.py lets the event
    loop handle cleanly).  Stops in-flight work, cancels any tracked background
    tasks so they don't leak, and flushes the vector index store to disk (this
    was previously dead code — :func:`kb.retrievers.shutdown_vector_store` was
    defined but never called).  All state in this bot is already persisted
    immediately, so nothing more is needed beyond flushing the in-memory index.
    """
    log.info("Bot shutting down — cleaning up background tasks + vector store")
    # 1. Stop any per-channel typing indicator loops (tracked as
    #    bot.typing_tasks in the /ai and prefix handlers).
    for t in list(getattr(bot, "typing_tasks", []) or []):
        try:
            if not t.done():
                t.cancel()
        except Exception:
            pass
    # 2. Cancel any other tracked background tasks (spawned via
    #    utils.background_tasks.spawn_tracked_task — strong refs are kept in
    #    _ACTIVE_BACKGROUND_TASKS so they aren't GC'd before we can cancel).
    from utils.background_tasks import _ACTIVE_BACKGROUND_TASKS
    for t in list(_ACTIVE_BACKGROUND_TASKS):
        try:
            if not t.done():
                t.cancel()
        except Exception:
            pass
    # 3. Flush + close the vector index store (the one resource that would
    #    otherwise be left to the OS).
    from kb.retrievers import shutdown_vector_store
    try:
        await shutdown_vector_store()
    except Exception as e:
        log.warning("Vector store shutdown failed: %s", e)
    # 4. Close the shared embeddings HTTP client (P2 #21) so the connection
    #    pool is torn down cleanly instead of lingering until process exit.
    from kb.embedder import close_client
    try:
        await close_client()
    except Exception as e:
        log.warning("Embeddings client shutdown failed: %s", e)
    log.info("Bot shutdown complete")


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

    sys_char = default_character()
    sys_model = sys_char.model if sys_char else DEFAULT_MODEL

    # P3 #26: let the user know if other AI requests are already ahead.
    await notify_if_queued(message)

    # P3 #24: typing runs for the whole turn (queue + generation + delivery);
    # the shared runner does the request + delivery with the same streaming /
    # embed behaviour as /ai. The run is registered so /ai stop can cancel it.
    channel_key = message.channel.id
    typing_task = start_typing(message, channel_key)
    run_task = spawn_tracked_task(
        run_ai_turn(
            message,
            user_message=prompt,
            model_slug=sys_model or "",
            guild_id=guild_id,
            channel_id=message.channel.id,
            username=message.author.display_name or "",
            user_id=message.author.id,
            char_name=str(sys_char.display),
        ),
        name=f"ai-run-{channel_key}",
    )
    register_run(channel_key, run_task)
    try:
        await run_task
    except asyncio.CancelledError:
        # /ai stop sent its own acknowledgement; nothing more to deliver here.
        log.info("prefix: in-flight run for channel %s cancelled (/ai stop)", channel_key)
    finally:
        clear_run(channel_key, run_task)
        if typing_task is not None:
            typing_task.cancel()


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

def _recover_crashed_recordings(bot_obj) -> None:
    """Attach the voice recorder and recover orphaned recordings at startup.

    Attaching installs the SIGTERM flush handler (so ``docker stop`` / compose
    restarts always write complete WAVs). :func:`recover_orphans` then rebuilds
    any recording whose process died mid-capture (OOM/segfault/power) — but
    only if its session marker is at least ~5 minutes old, so a still-live
    recording (bot restarted while a meeting was going) is left untouched.
    """
    from config.settings import RECORDINGS_DIR
    from bot_core.voice_recorder import attach_to_bot, recover_orphans

    attach_to_bot(bot_obj, RECORDINGS_DIR)  # also installs the SIGTERM flush hook
    recovered = recover_orphans(RECORDINGS_DIR)
    if recovered:
        log.info("Recovered %d crashed recording(s) at startup", len(recovered))


def run_bot() -> None:
    """P3 #27: entry point for the ``discord-ai-bot`` console script.

    Extracted from the module-level ``__main__`` block so the bot can be
    launched via ``python -m main`` *and* the installed ``discord-ai-bot``
    console script (see ``pyproject.toml``) without duplicating the startup
    sequence. Behaviour is unchanged: guard on the token, enforce the single
    instance lock, log the connection target, then run the bot.
    """
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


if __name__ == "__main__":
    run_bot()
