# HumbelChat - AI-Powered Discord Bot

A feature-rich Discord bot that brings conversational AI capabilities to your server, with built-in RAG (Retrieval-Augmented Generation) for contextual knowledge responses.

## What This Bot Does

**Primary Functions:**
- **Conversational AI**: Engage in natural, multi-turn conversations with an AI assistant
- **RAG Integration**: Pull relevant information from filesystem-based knowledge bases to provide accurate, context-aware responses
- **Slash Commands**: Modern Discord interaction framework with intuitive commands like `/ask`, `/chat`, and `/knowledge`

**Key Features:**
- Personality-driven conversations (configurable character personalities)
- Session memory for coherent multi-message dialogues
- Knowledge base integration via filesystem storage
- Token-limited response handling to prevent Discord limits
- Async-compatible architecture built on discord.py

## Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/MeleeCampz/HumbelChat.git
cd HumbelChat
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Discord bot token and OpenAI API key

# Run the bot
python main.py
```

## Configuration

Set these environment variables in `.env`:
- `DISCORD_TOKEN` - Your Discord bot token
- `OPENAI_API_KEY` - Your OpenAI API key  
- `DEFAULT_PERSONALITY` - Character personality file to load by default
- `MAX_TOKENS` - Maximum tokens per response (default: 1500)

## Project Structure

```
discord-ai-bot/
├── main.py              # Bot entry point and command handlers
├── bot_core.py          # Core bot logic and conversation management
├── config/              # Settings, characters, and environment variables
├── kb/                  # Knowledge base reader and storage (filesystem-based RAG)
├── commands/            # Slash command implementations
└── utils/               # Helper functions for responses and utilities
```

## Building Your Own Knowledge Base

Place `.txt` files in the `kb/` directory to create a filesystem-based knowledge base. The bot will automatically index and search these documents when answering questions.

## License

MIT License - Feel free to fork, modify, and deploy on your own Discord servers.
