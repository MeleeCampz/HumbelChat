# Discord AI Bot

A feature-rich Discord bot that brings conversational AI capabilities to your server, with built-in RAG (Retrieval-Augmented Generation) for contextual knowledge responses.

## What This Bot Does

**Primary Functions:**
- **Conversational AI**: Engage in natural, multi-turn conversations with an AI assistant
- **Smart RAG Integration**: Pull relevant information from filesystem-based knowledge bases using vector similarity search or keyword matching
- **Slash Commands**: Modern Discord interaction framework with intuitive commands like `/ai`, `/character`, and `/upload_kb`

**Key Features:**
- Personality-driven conversations (configurable character personalities)
- Session memory for coherent multi-message dialogues per guild/channel
- **Knowledge base integration** via filesystem storage with two retrieval strategies:
  - **Vector similarity search** (default): Semantic embedding search using OpenWebUI's `nomic-embed-text` model with persistent SQLite indexing
  - **Keyword/TF-IDF fallback**: Heuristic scoring for environments without vector backend access
- **Smart document chunking**: Head-aware splitting that preserves semantic boundaries for better retrieval quality
- **Paragraph-aware response splitting** to respect Discord's 2000-char limit
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
# Edit .env and set at minimum: DISCORD_BOT_TOKEN, INFER_URL, and INFER_API_KEY (or leave empty for local)

# Run the bot
python main.py
# Or use the startup script for automatic logging:
chmod +x start_bot.sh && ./start_bot.sh
```

## Configuration

Set these environment variables in `.env`:

### Required

| Variable | Description | Default |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Discord bot token (required) | — |
| `INFER_URL` | Base URL for AI inference backend | `http://127.0.0.1:11434/v1` |
| `INFER_API_KEY` | API key for the inference provider | *(empty for local)* |

### AI Provider

| Variable | Description | Default |
|---|---|---|
| `MODEL_NAME` | Default model slug for AI requests | *(empty — uses character config)* |
| `AI_REQUEST_TIMEOUT` | HTTP timeout in seconds | `120` |
| `MAX_TOKENS` | Maximum tokens per response | `2000` |

### Bot Behavior

| Variable | Description | Default |
|---|---|---|
| `CONTEXT_WINDOW` | Number of message rounds retained per channel | `10` |
| `BOT_PREFIX` | Prefix for non-slash commands (e.g., `!ai`) | `!ai` |
| `CHAT_HISTORY_RESET` | Set to "clear" once to wipe all chat history | *(empty)* |

### Knowledge Base & RAG

| Variable | Description | Default |
|---|---|---|
| `KB_PATH` | Path to knowledge base files | `./data/knowledge` |
| `KB_DEFAULT_KB` | Default KB name on startup (folder slug) | `humblewood` |
| `CHUNK_SIZE` | Target chunk size for legacy indexing | `2000` |
| `RAG_MAX_DOCS` | Max documents to attach per RAG query | `4` |
| `RAG_MAX_CHARS` | Hard cap on RAG context chars sent to LLM | `24000` |
| `RAG_WINDOW_LINES` | Lines above/below each match anchor (wider → complete spell/ability blocks) | `80` |
| `RAG_RETRIEVAL_METHOD` | Retrieval strategy: `vector` (semantic search, default) or `keyword` (TF-IDF) | `vector` |

## Architecture

```
┌──────────┐    ┌─────────────┐    ┌─────────────────┐
│  Discord │◄──►│   Bot (Py-  │◄──►│  AI Backend     │    ← Conversational AI
│  Gateway │    │  discord)   │    │  (OpenWebUI, etc.)│    
└──────────┘    └─────────────┘    └─────────────────┘    

                    │
                    ▼
              ┌─────────────┐     ┌───────────────────┐
              │ KB Files    │◄──►│ Vector Index      │
              │ (local fs)  │     │ (SQLite cache)    │     ← RAG Pipeline
              └─────────────┘     └───────────────────┘
                      ▲                  ▲
                      │                  │
          ┌───────────────┐    ┌──────────────────┐
          │ Smart Chunker  │    │ nomic-embed-text │
          │ (header-aware) │    │ embedding model  │
          └───────────────┘    └──────────────────┘
```

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
├── kb/                     # Knowledge base & RAG modules
│   ├── __init__.py           # Package init
│   ├── reader.py             # Filesystem-based KB reading (RAG source)
│   ├── storage.py            # Upload, validate, list KB files
│   ├── scorch.py             # TF-IDF relevance scoring for chunks
│   ├── vector_db.py          # In-memory vector index with cosine similarity
│   ├── embedder_openai.py    # Async embedding via OpenWebUI /embeddings endpoint
│   ├── chunker.py            # Smart document chunking (header-aware + paragraph fallback)
│   ├── index.py              # Persistent SQLite-backed vector index store
│   └── query_rewriter.py     # Automatic LLM-powered query expansion for RAG
├── utils/                  # Helper functions
│   ├── __init__.py           # Package init
│   ├── kb_utils.py               # KB logging utilities
│   ├── response_splitter.py      # Long message chunking (paragraph-aware)
│   └── typing_loop.py            # Typing indicator task
├── tests/                  # Unit tests
│   ├── conftest.py           # Test fixtures
│   ├── test_ai_command.py
│   ├── test_bot_core.py
│   ├── test_kb_commands.py
│   ├── test_kb_reader.py
│   ├── test_adaptive_rag.py  # Vector search integration tests
│   └── ...
├── data/knowledge/         # Knowledge base source files (not committed)
├── docs/                   # Additional documentation
│   └── README.md
├── .env.example            # Example environment variable template
├── characters.json.example # Example character configuration
├── requirements.txt        # Python dependencies
├── start_bot.sh            # Startup script with logging setup
├── README.md               # This file
```

## Slash Commands Reference

All commands begin with `/`:

### `/ai` — AI Chat
```
/ai <message> [character: <name>]
```
Sends prompt to the active AI character. Uses per-channel conversation history and KB context via RAG.

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
Uploads `.txt`, `.md`, `.csv`, `.html`, `.xml`, or `.rtf` files to the local KB directory. Auto-chunks uploaded files for vector indexing.

### `/list_kb_docs` — List KB Documents
Lists all documents in `KB_PATH`. Shows name, size, modification date, and SHA256 prefix.

### `/reindex_kb` — Reindex Knowledge Base
Rebuilds the vector index from scratch using smart chunking and OpenWebUI embeddings.

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

## Knowledge Base (RAG)

### Retrieval Strategies

The bot supports two retrieval strategies controlled by `RAG_RETRIEVAL_METHOD`:

1. **Vector Search** (default, recommended): Documents are semantically chunked and embedded using the configured AI backend's `/embeddings` endpoint (model: `nomic-embed-text:latest`). A SQLite-persisted index ensures fast bot restarts without re-embedding. Uses cosine similarity for relevance ranking.

2. **Keyword/TF-IDF**: Lightweight heuristic scoring of filenames, headers, and body text overlap. Works without any vector backend and serves as automatic fallback when the embedding service is unavailable.

### Smart Chunking

The smart chunker (`kb/chunker.py`) splits documents using three strategies:
- **Full document** for small files (≤8000 chars) to preserve context
- **Header-based splitting** with minimum-size merging for larger documents
- **Adaptive paragraph splitting** as a fallback, with structural awareness for dense content

This ensures queries for specific topics (e.g., "time system") hit only relevant sections — not drowned out by unrelated content.

### Supported File Types
- `.txt` - Plain text files
- `.md` - Markdown files
- `.csv` - CSV spreadsheets
- `.html` - HTML documents
- `.xml` - XML documents
- `.rtf` - Rich Text Format

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

## Testing

```bash
python -m pytest tests/ -v
```

## License

MIT
