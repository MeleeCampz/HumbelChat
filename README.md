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

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env — DISCORD_BOT_TOKEN is required

# 3. Run the bot
python main.py
```

After startup, all slash commands are registered globally (may take up to 60 min for Discord to propagate). Try `/ai Hello!` or `/character list` in any text channel.

For full documentation see [docs/README.md](docs/README.md).

---

## License

MIT
