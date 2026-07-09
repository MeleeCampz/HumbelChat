import asyncio
import json
import os
import pathlib
import logging
from dotenv import load_dotenv

load_dotenv()  # Load .env variables into os.environ

import discord
from discord.ext import commands
import discord.app_commands as app_commands
from openai import AsyncOpenAI

# ─────────────────────────── Configuration ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("bot")

INTENTS = discord.Intents.default()
INTENTS.messages = True
INTENTS.guilds = True
INTENTS.guild_messages = True
INTENTS.message_content = True

DISCORD_TOKEN  = os.getenv("DISCORD_BOT_TOKEN", "")

# API Key — local endpoints typically ignore it, but we pass something non-empty
OPENAI_API_KEY = (os.getenv("OPENWEBUI_API_KEY") or
                  os.getenv("OPENAI_API_KEY", "local-model-key"))

# Base URL for the local inference backend:
#   OpenWebUI (default):  http://localhost:8080/v1
#   SillyTavern proxy:    http://localhost:5100/v1/openai
API_BASE_URL = os.getenv("OPENAI_API_URL", "http://localhost:8080/v1")

# Fallback values used when no characters are configured.
DEFAULT_MODEL      = os.getenv("MODEL_NAME", "default-model-name")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant embedded in a Discord bot.",
)

CONTEXT_WINDOW: int = 10
prefix         = os.getenv("BOT_PREFIX", "!ai")

# ─────────────────────────── Characters ──────────────────────────────
CHARACTERS_FILE = pathlib.Path(__file__).parent / "characters.json"

try:
    with open(CHARACTERS_FILE, encoding="utf-8") as fh:
        _char_cfg = json.load(fh)
except FileNotFoundError:
    log.warning("characters.json not found — using single default character.")
    _char_cfg = {
        "default": "Default",
        "characters": {
            "Default": {"model": DEFAULT_MODEL},
        },
    }

CHARACTERS: dict       = _char_cfg.get("characters", {})
DEFAULT_CHARACTER     = _char_cfg.get("default", list(CHARACTERS)[0]) if CHARACTERS else "Default"

# Pre-built Discord slash-command choices (dropdown) — populated at startup from characters.json
_CHAR_CHOICES = [
    discord.app_commands.Choice(name=name, value=name)
    for name in CHARACTERS.keys()
]

# Per-guild channel active character  (key = (guild_id, channel_id))
_active_characters: dict[tuple[int, int], str] = {}


def _get_char_model(gid: int | None, cid: int) -> str:
    """Return the model slug for the current active character."""
    if gid is not None and (gid, cid) in _active_characters:
        name = _active_characters[(gid, cid)]
    else:
        name = DEFAULT_CHARACTER
    char_data = CHARACTERS.get(name, {})
    return char_data.get("model", DEFAULT_MODEL)


def _switch_character(gid: int | None, cid: int, name: str) -> tuple[bool, str]:
    """Set active character. Returns (ok, message)."""
    if name not in CHARACTERS:
        avail = ", ".join(f"`{k}`" for k in CHARACTERS)
        return False, f"Unknown character '{name}'. Available: {avail}"
    if gid is not None:
        _active_characters[(gid, cid)] = name
    entry = CHARACTERS[name].get("model", "")
    return True, f"Switched to **{name}** ({entry})"


# ─────────────────────────── Bot Setup ───────────────────────────────
bot = commands.Bot(command_prefix=prefix, intents=INTENTS)

# Per-channel conversation history: guild_id -> channel_id -> [messages]
chat_histories: dict[int, dict[int, list[dict]]] = {}


def _ensure_history(guild_id: int, channel_id: int) -> None:
    if guild_id not in chat_histories:
        chat_histories[guild_id] = {}
    if channel_id not in chat_histories[guild_id]:
        chat_histories[guild_id][channel_id] = []


def _clear_history(guild_id: int, channel_id: int) -> None:
    chat_histories.get(guild_id, {}).pop(channel_id, None)


# ─────────────────────────── Helpers ─────────────────────────────────


async def ask_ai_with_model(
    user_message: str,
    model: str,
    guild_id: int,
    channel_id: int,
) -> str:
    """Send *user_message* to the AI using a specific model slug and return the reply text."""
    timeout_sec = int(os.getenv("AI_REQUEST_TIMEOUT", "120"))
    system_p = DEFAULT_SYSTEM_PROMPT

    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

    _ensure_history(guild_id, channel_id)
    history = chat_histories[guild_id][channel_id]

    messages: list[dict] = [{"role": "system", "content": system_p}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        stream=False,
        timeout=timeout_sec,
        metadata={"r": True},  # ← activate Knowledge Base / RAG
    )

    reply = resp.choices[0].message.content or "(empty response)"

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 2 * CONTEXT_WINDOW:
        chat_histories[guild_id][channel_id] = history[-(2 * CONTEXT_WINDOW):]

    return reply


async def _typing_loop(channel, duration_sec: int = 30):
    """Send typing indicators every 10s for up to duration_sec seconds."""
    end_time = asyncio.get_event_loop().time() + duration_sec
    while asyncio.get_event_loop().time() < end_time:
        try:
            await channel.typing()
        except discord.Forbidden:
            break
        except Exception:
            pass
        await asyncio.sleep(10)


async def _send_long_response(source, reply: str, char_name: str = "") -> None:
    """Send `reply`, chunking it into multiple messages if > 1900 chars.

    Paragraph-aware splitting keeps code blocks and lists intact.
    Works for both Slash Commands (followup.send) and prefix commands (reply).
    """
    MAX_LEN = 1900  # Leave room for "[X/Y] " metadata

    header = f"--- {char_name} ---\n" if char_name else ""

    if len(reply) <= MAX_LEN:
        full_text = (header + reply).strip()
        if hasattr(source, 'followup'):
            await source.followup.send(full_text)
        else:
            await source.reply(full_text)
        return

    paragraphs = reply.split('\n\n')
    chunks, current_chunk = [], ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > MAX_LEN:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Single paragraph too long → force-split by words
            if len(para) > MAX_LEN:
                words = para.split()
                sub = ""
                for w in words:
                    if len(sub) + len(w) + 1 > MAX_LEN:
                        chunks.append(sub.strip())
                        sub = w
                    else:
                        sub += " " + w if sub else w
                current_chunk = sub
            else:
                current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk.strip())

    for i, chunk in enumerate(chunks, 1):
        meta = f"[{i}/{len(chunks)}] "
        display_text = (header + meta + chunk).strip()
        if hasattr(source, 'followup'):
            await source.followup.send(display_text)
        else:
            await source.reply(display_text)


# ─────────────────────────── Slash Commands ──────────────────────────

@bot.tree.command(name="ai", description="Send a prompt to the AI and get a reply.")
@app_commands.choices(character=_CHAR_CHOICES)
async def ai_command(
    interaction: discord.Interaction,
    message: str,
    character: str | None = None,
):
    """AI chat command with optional character override. Falls back to System if none specified."""
    await interaction.response.defer()

    guild_id = interaction.guild_id or 0

    # Determine which character/model to use
    if character is not None:
        if character not in CHARACTERS:
            avail = ", ".join(f"`{k}`" for k in CHARACTERS)
            await interaction.followup.send(
                f"Unknown character `{character}`. Available: {avail}", ephemeral=True
            )
            return
        model_to_use = CHARACTERS[character]["model"]
        char_name = character
    else:
        # No character specified — use active per-channel character or fall back to DEFAULT_CHARACTER (System)
        if guild_id is not None and (guild_id, interaction.channel_id) in _active_characters:
            name = _active_characters[(guild_id, interaction.channel_id)]
        else:
            name = DEFAULT_CHARACTER  # "System"
        char_data = CHARACTERS.get(name, {"model": DEFAULT_MODEL})
        model_to_use = char_data["model"]
        char_name = name

    asyncio.create_task(_typing_loop(interaction.channel))

    reply = await ask_ai_with_model(
        message, model_to_use, guild_id, interaction.channel_id
    )
    await _send_long_response(interaction, reply, char_name)


@bot.tree.command(
    name="character",
    description="Manage AI character/persona settings.",
)
async def character_command(
    interaction: discord.Interaction,
    action: str = "list",
    name: str | None = None,
):
    """Switch or list characters.

    Usage:
        /character              — lists available characters (default)
        /character set <name>   — switch to a character
        /character show         — shows current active character
        /character reset        — revert to default character
    """
    await interaction.response.defer(ephemeral=True)
    if not CHARACTERS:
        await interaction.followup.send("No characters configured.", ephemeral=True)
        return

    guild_id = interaction.guild_id or 0
    current = _active_characters.get((guild_id, interaction.channel_id), DEFAULT_CHARACTER)
    is_current = lambda n: n == current

    if action == "list":
        lines = ["**Available characters:**\n"]
        for cname, cdata in CHARACTERS.items():
            marker = " ← current" if is_current(cname) else ""
            lines.append(f"  • `{cname}` — model: `{cdata['model']}`{marker}")
        await interaction.followup.send("\n".join(lines), ephemeral=True)

    elif action == "set":
        if name is None:
            await interaction.followup.send("Please provide a character name: `/character set <name>`", ephemeral=True)
            return
        ok, msg = _switch_character(guild_id, interaction.channel_id, name)
        await interaction.followup.send(msg, ephemeral=True)

    elif action == "show":
        char_data = CHARACTERS.get(current, {})
        await interaction.followup.send(
            f"**Current character:** `{current}`\n"
            f"**Model:** `{char_data.get('model', 'N/A')}`",
            ephemeral=True,
        )

    elif action == "reset":
        if guild_id is not None:
            _active_characters.pop((guild_id, interaction.channel_id), None)
        await interaction.followup.send(f"Reverted to default character: **{DEFAULT_CHARACTER}**", ephemeral=True)

    else:
        await interaction.followup.send(
            f"Unknown action '{action}'. Use: list, set, show, reset.",
            ephemeral=True,
        )


@bot.tree.command(name="clear_history", description="Clear conversation history for this channel.")
async def clear_history_command(interaction: discord.Interaction):
    if interaction.guild_id and interaction.channel_id:
        _clear_history(interaction.guild_id, interaction.channel_id)
    await interaction.response.send_message("Conversation history cleared.", ephemeral=True)


# ─────────────────────────── Event Handlers ──────────────────────────

@bot.event
async def on_ready():
    log.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    if bot.guilds:
        bot.tree.copy_global_to(guild=discord.Object(id=bot.guilds[0].id))
    await bot.tree.sync()
    log.info("Characters loaded: %s", ", ".join(CHARACTERS) or "(none)")


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    content = message.content.strip()
    if not content.startswith(prefix):
        return

    prompt = content[len(prefix):].strip()
    if not prompt:
        await message.channel.send(f"Usage: {prefix} <your question>")
        return

    guild_id = message.guild_id or 0
    log.info("%s (%s) in #%s: %s",
             message.author, message.author.id, message.channel.name, prompt[:80])

    await message.channel.typing()
    # Prefix command always uses default character (System)
    sys_model = CHARACTERS.get(DEFAULT_CHARACTER, {}).get("model", DEFAULT_MODEL)
    reply = await ask_ai_with_model(prompt, sys_model, guild_id, message.channel.id)
    await _send_long_response(message, reply, DEFAULT_CHARACTER)


# ─────────────────────────── Startup ─────────────────────────────────

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        log.error("Please set the DISCORD_BOT_TOKEN environment variable.")
        raise SystemExit(1)

    log.info("Connecting to local AI backend at: %s", API_BASE_URL)
    log.info("Using default model:  %s", DEFAULT_MODEL)
    bot.run(DISCORD_TOKEN)
