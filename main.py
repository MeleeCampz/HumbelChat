import asyncio
import os
import logging
from dotenv import load_dotenv
load_dotenv()  # Load .env variables into os.environ

import discord
from discord.ext import commands
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
# Prefer OPENWEBUI_API_KEY for OpenWebUI; fall back to OPENAI_API_KEY.
OPENAI_API_KEY = (os.getenv("OPENWEBUI_API_KEY") or
                  os.getenv("OPENAI_API_KEY", "local-model-key"))

# Base URL for the local inference backend:
#   OpenWebUI (default):  http://localhost:8080/v1
#   SillyTavern proxy:    http://localhost:5100/v1/openai
API_BASE_URL = os.getenv("OPENAI_API_URL", "http://localhost:8080/v1")

# The model slug that your local server exposes (e.g. from /api/tags)
MODEL_NAME   = os.getenv("MODEL_NAME", "default-model-name")

SYSTEM_PROMPT  = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant embedded in a Discord bot.",
)
CONTEXT_WINDOW: int = 10
prefix         = os.getenv("BOT_PREFIX", "!ai")

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

async def ask_ai(user_message: str, guild_id: int, channel_id: int) -> str:
    """Send *user_message* to the local AI model and return the reply."""
    # Timeout must exceed local model inference time (~9s for qwen3.6:latest).
    # Also set per-request timeout so fast-fail works on transient errors.
    timeout_sec = int(os.getenv("AI_REQUEST_TIMEOUT", "120"))
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=API_BASE_URL,
        http_client=None,  # use default; we set timeout below
    )

    _ensure_history(guild_id, channel_id)
    history = chat_histories[guild_id][channel_id]

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        stream=False,  # non-streaming — avoids timeout issues with OpenWebUI
        timeout=timeout_sec,   # seconds before giving up
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
            # Bot lacks permission to send typing indicators in this channel
            break
        except Exception:
            pass
        await asyncio.sleep(10)


async def _send_long_response(source, reply: str) -> None:
    """Send `reply`, chunking it into multiple messages if > 1900 chars.

    Paragraph-aware splitting keeps code blocks and lists intact.
    Works for both Slash Commands (followup.send) and prefix commands (reply).
    """
    MAX_LEN = 1900  # Leave room for "[X/Y] " metadata

    if len(reply) <= MAX_LEN:
        if hasattr(source, 'followup'):
            await source.followup.send(reply)
        else:
            await source.reply(reply)
        return

    # Split by double newlines (paragraphs) first for readability
    paragraphs = reply.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > MAX_LEN:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If a single paragraph is longer than MAX_LEN, force-split by words
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

    # Send sequentially with page indicators
    for i, chunk in enumerate(chunks, 1):
        meta = f"[{i}/{len(chunks)}] "
        display_text = meta + chunk

        if hasattr(source, 'followup'):
            await source.followup.send(display_text)
        else:
            await source.reply(display_text)


# ─────────────────────────── Slash Commands ──────────────────────────

@bot.tree.command(name="ai", description="Send a prompt to the AI and get a reply.")
async def ai_command(interaction: discord.Interaction, message: str):
    await interaction.response.defer()

    guild_id = interaction.guild_id or 0
    asyncio.create_task(_typing_loop(interaction.channel))

    reply = await ask_ai(message, guild_id, interaction.channel_id)
    await _send_long_response(interaction, reply)


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
    reply = await ask_ai(prompt, guild_id, message.channel.id)
    await _send_long_response(message, reply)


# ─────────────────────────── Startup ─────────────────────────────────

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        log.error("Please set the DISCORD_BOT_TOKEN environment variable.")
        raise SystemExit(1)

    log.info(f"Connecting to local AI backend at: {API_BASE_URL}")
    log.info(f"Using model:             {MODEL_NAME}")
    bot.run(DISCORD_TOKEN)
