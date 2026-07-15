# HumbleChat — Local-Only Discord AI Bot

A self-hosted Discord bot that forwards messages from servers to **OpenWebUI** (your local AI gateway) and returns AI responses as Discord messages. All inference runs through OpenWebUI's OpenAI-compatible API layer — nothing touches a model backend directly.

> **Local-only:** All inference runs on your own hardware. No cloud APIs. Private by design.
> **Gateway-first:** Every request routes through OpenWebUI, giving you centralized access to knowledge bases, user management, and model creation for the future.

---

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
pip install -r requirements.txt

# 2. Copy configuration files
cp .env.example .env                    # Edit .env — DISCORD_BOT_TOKEN is required
cp characters.example.json characters.json  # Configure your AI personas and models

# 3. Run the bot
python main.py
```

After startup, all slash commands are registered globally (may take up to 60 min for Discord to propagate). Try `/ai Hello!` or `/character list` in any text channel.

For full documentation see [docs/README.md](docs/README.md).

---

## Configuration Files

### .env
Contains sensitive credentials and bot configuration:
- `DISCORD_BOT_TOKEN` - Your Discord bot token (required)
- Model endpoints, API keys, and other settings

Copy `.env.example` to `.env` before running the bot. Never commit `.env` to version control.

### characters.json
Defines AI personas and their configurations:
- Each character has a model assignment, system prompt, and optional parameters
- **Private file** - do not share or commit with real character data

Copy `characters.example.json` to `characters.json` and customize for your use case.

---

## License

MIT