# Discord AI Bot

A feature-rich Discord bot that brings conversational AI capabilities to your server, with built-in RAG (Retrieval-Augmented Generation) for contextual knowledge responses.

## What This Bot Does

**Primary Functions:**
- **Conversational AI**: Engage in natural, multi-turn conversations with an AI assistant
- **RAG Integration**: Pull relevant information from filesystem-based knowledge bases to provide accurate, context-aware responses
- **Slash Commands**: Modern Discord interaction framework with intuitive commands like `/ai`, `/character`, and `/upload_kb`

**Key Features:**
- Personality-driven conversations (configurable character personalities)
- Session memory for coherent multi-message dialogues per guild/channel
- Knowledge base integration via filesystem storage (no external APIs needed)
- TF-IDF relevance scoring for chunked KB retrieval
- Paragraph-aware response splitting to respect Discord's 2000-char limit
- Async-compatible architecture built on discord.py

## Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/MeleeCampz/discord-ai-bot.git
cd discord-ai-bot
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

# Configure environment — required before first run:
cp .env.example .env
# Edit .env and set at minimum: DISCORD_BOT_TOKEN and INFER_URL

# Run the bot
python main.py
```

## Configuration

Set these environment variables in `.env`:

| Variable | Description | Default |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Discord bot token (required) | — |
| `INFER_URL` | Base URL for AI inference backend | `http://127.0.0.1:11434/v1` |
| `INFER_API_KEY` | API key for the inference provider | *(empty for local)* |
| `CONTEXT_WINDOW` | Number of message rounds retained per channel | `10` |
| `BOT_PREFIX` | Prefix for non-slash commands (e.g., `!ai`) | `!ai` |
| `AI_REQUEST_TIMEOUT` | HTTP timeout in seconds | `120` |
| `MAX_TOKENS` | Maximum tokens per response | `2000` |
| `KB_PATH` | Path to knowledge base files | `./data/knowledge` (default) |
| `CHUNK_SIZE` | Tokens per KB chunk for indexing | `2000` |

## Project Structure

```
discord-ai-bot/
├── main.py                   # Bot entry point, event handlers, slash command registrations
├── bot_core.py               # Core AI client + conversation history (shared state)
├── config/                   # Settings and character configuration
│   ├── __init__.py           # Package init
│   ├── settings.py           # Environment variable loading & singleton
│   └── characters.py         # Character/persona loading & display mapping
├── commands/                 # Slash command implementations
│   ├── __init__.py           # Package init
│   ├── ai_command.py              # /ai command handler (delegates to bot_core)
│   ├── character_commands.py      # /character command handler
│   ├── clear_history_command.py   # /clear_history handler
│   ├── kb_commands.py             # /upload_kb, /list_kb_docs, /reindex_kb handlers
│   └── utility_commands.py        # /remind, /ocr, /summarize, /translate handlers
├── kb/                     # Knowledge base modules
│   ├── __init__.py           # Package init
│   ├── reader.py             # Filesystem-based KB reading (RAG source)
│   ├── storage.py            # Upload, validate, list KB files
│   └── scorch.py             # TF-IDF relevance scoring for chunks
├── utils/                  # Helper functions
│   ├── __init__.py           # Package init
│   ├── kb_utils.py               # KB logging utilities
│   ├── response_splitter.py      # Long message chunking (paragraph-aware)
│   └── typing_loop.py            # Typing indicator task
├── tests/                  # Unit tests
│   └── test_kb_reader.py
├── docs/                   # Additional documentation
│   └── README.md
├── characters.json         # (optional) Character/persona config — NOT committed
├── .env                    # Environment variables (from .env.example)
├── .env.example            # Example environment variable template
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Slash Commands Reference

All commands begin with `/`:

### `/ai` — AI Chat
```
/ai <message> [character: <name>]
```
Sends prompt to the active AI character. Uses per-channel conversation history and KB context.

### `/character` — Persona Management
```
/character [action: list|set|show|reset] [name: <character_name>]
```

| Action | Description |
|---|---|
| `list` | Lists available characters with current highlight |
| `set` | Switches active character for this server/channel |
| `show` | Shows currently active character and model |
| `reset` | Clears per-channel override, uses default |

### `/upload_kb` — Knowledge Base Upload
```
/upload_kb [kb_name: <name>] [url: <url>] <file_attachment>
```
Uploads `.txt`, `.md`, `.csv`, `.html`, `.xml`, or `.rtf` files to the local KB directory. Auto-chunks uploaded files for RAG indexing.

### `/list_kb_docs` — List KB Documents
Lists all documents in `KB_PATH`. Shows name, size, modification date, and SHA256 prefix.

### `/reindex_kb` — Reindex Knowledge Base
Rebuilds chunk indices (TF-IDF scoring) for all KB files.

### `/clear_history` — Clear Conversation History
Clears the conversation history for this server/channel.

### `/ocr` — Extract Text from Image
Upload an image and get all text extracted via vision AI.

### `/summarize` — Summarize Content
Summarize recent chat history or a file from a URL using AI.

### `/translate` — Translate Text
```
/translate <target_language>: <text>
```
Translates text into the specified target language.

## Prefix Command (Legacy)

```
<BOT_PREFIX><your_question>
```

Example: `!ai What time is it?`

Uses the default character and shares the same history as slash commands.

## Character Configuration (`characters.json`)

Controls AI personas/models. Structure:

```json
{
  "default": "System",
  "characters": {
    "System": {
      "model": "qwen3:latest",
      "system_prompt": ""
    },
    "Assistant": {
      "display": "Chat Assistant",
      "model": "gemma4:latest",
      "system_prompt": "..."
    }
  }
}
```

| Field | Description |
|---|---|
| `default` | Character used when none selected |
| `characters.<name>` | Each key becomes an available persona |
| `display` | Human-readable name (optional) |
| `model` | Model slug for the inference API |
| `system_prompt` | Custom system prompt (optional) |

**Important:** `characters.json` is private and should NOT be committed. It's in `.gitignore`.

## Knowledge Base (RAG)

The bot performs filesystem-based RAG — it reads `.txt` and `.md` files directly from `KB_PATH` at inference time, no external vector database required. Documents are split into chunks via heading-aware splitting and scored using TF-IDF relevance matching.

### Supported File Types
- `.txt` - Plain text files
- `.md` - Markdown files
- `.csv` - CSV spreadsheets
- `.html` - HTML documents
- `.xml` - XML documents
- `.rtf` - Rich Text Format

## License

MIT
