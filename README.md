# Discord AI Bot

A Discord bot that reads text messages from channels and forwards them to an AI (OpenAI), then sends the AI's response back to the channel.

## Features

- **Slash command** `/ai <message>` for direct prompts
- **Prefix commands** `!ai <message>` as a shortcut
- **Conversation history** — remembers context within each channel
- **Clear history** with `/clear_history` when needed
- **Typing indicators** so users see the bot is working
- **Long-message support** — automatically sends large responses as file attachments

## Prerequisites

- Python 3.12+ installed
- A Discord Bot Token (see [Discord Developer Portal](https://discord.com/developers/applications))
- An OpenAI API key (from [OpenAI Platform](https://platform.openai.com/api-keys))

## Setup

### 1. Clone / create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# or: venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Create a `.env` file in the project root

```env
DISCORD_BOT_TOKEN=your-discord-bot-token-here
OPENAI_API_KEY=sk-your-openai-key-here
MODEL_NAME=gpt-3.5-turbo          # or gpt-4o, gpt-4, etc.
SYSTEM_PROMPT=You are a helpful AI assistant embedded in a Discord bot.
BOT_PREFIX=!ai                     # optional; default is !ai
CONTEXT_WINDOW=10                  # how many previous turns to keep per channel
```

### 3. Invite the bot to your server

Go to **Discord Developer Portal → Your Application → Bot** and copy the invite URL, e.g.:

```
https://discord.com/oauth2/authorize?client_id=YOUR_CLIENT_ID&scope=bot%20applications.commands&permissions=8
```

Make sure **MESSAGE CONTENT INTENT** is enabled on the bot settings page.

### 4. Run the bot

```bash
python main.py
```

The bot will appear online in your server. Try `/ai Hello!` in any text channel.

## Configuration Reference

| Variable | Description | Default |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Discord bot token | _(required)_ |
| `OPENAI_API_KEY` | OpenAI API key | _(required)_ |
| `MODEL_NAME` | OpenAI model to use | `gpt-3.5-turbo` |
| `SYSTEM_PROMPT` | System / developer message for the AI | _"You are a helpful AI assistant..."_ |
| `BOT_PREFIX` | Prefix for text commands | `!ai` |
| `CONTEXT_WINDOW` | Number of past turn pairs kept per channel | `10` |

## License

MIT
