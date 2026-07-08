import asyncio
import os
import logging

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

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME      = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
SYSTEM_PROMPT   = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant embedded in a Discord bot.",
)
CONTEXT_WINDOW: int = 10
prefix          = os.getenv("BOT_PREFIX", "!ai")

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
    """Send *user_message* to the AI model and return the reply."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

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
    )

    reply = resp.choices[0].message.content or "(empty response)"

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 2 * CONTEXT_WINDOW:
        chat_histories[guild_id][channel_id] = history[-(2 * CONTEXT_WINDOW):]

    return reply


async def _typing_loop(channel, duration_sec: int = 30):
    """Send typing indicators every 10 seconds for *duration_sec* seconds."""
    end_time = asyncio.get_event_loop().time() + duration_sec
    while asyncio.get_event_loop().time() < end_time:
        await channel.typing()
        await asyncio.sleep(10)


async def _send_long_response(interaction, reply: str):
    if len(reply) > 2000:
        with open("_tmp.txt", "w") as f:
            f.write(reply)
        await interaction.followup.send(file=discord.File("_tmp.txt"))
        os.remove("_tmp.txt")
    else:
        await interaction.followup.send(reply)


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
    log.info("%s (%s) in #%s: %s", message.author, message.author.id, message.channel.name, prompt[:80])

    await message.channel.typing()
    reply = await ask_ai(prompt, guild_id, message.channel.id)
    await _send_long_response(message, reply)


# ─────────────────────────── Startup ─────────────────────────────────

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        log.error("Please set the DISCORD_BOT_TOKEN environment variable.")
        raise SystemExit(1)
    if not OPENAI_API_KEY:
        log.error("Please set the OPENAI_API_KEY environment variable.")
        raise SystemExit(1)

    bot.run(DISCORD_TOKEN)
