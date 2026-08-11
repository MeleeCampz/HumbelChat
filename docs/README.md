# Discord AI Bot Documentation

A self-hosted Discord bot that forwards messages to a local AI backend and returns responses. Designed for privacy-first, on-premises AI inference.

---

## Environment Configuration (.env)

Copy `.env.example` to `.env` and configure the following variables:

### Required
| Variable | Description | Default |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Discord bot token (required) | — |
| `INFER_URL` | Base URL for AI inference backend | `http://127.0.0.1:11434/v1` |
| `INFER_API_KEY` | API key for the inference provider | *(empty for local)* |

### AI Provider
| Variable | Description | Default |
|---|---|---|
| `MODEL_NAME` | Default model slug for requests | *(empty — uses character config)* |
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
| `RAG_RETRIEVAL_METHOD` | Retrieval strategy: `vector` or `keyword` | `vector` |

---

## Architecture Overview

```
┌──────────┐    ┌─────────────┐    ┌─────────────────┐    
│  Discord │◄──►│   Bot (Py-  │◄──►│  AI Backend     │    
│  Gateway │    │  discord)   │    │  (AI backend, etc.)│    
└──────────┘    └─────────────┘    └─────────────────┘    

                    │
                    ▼
              ┌─────────────┐     ┌───────────────────┐
              │ KB Files    │◄──►│ Vector Index      │
              │ (local fs)  │     │ (SQLite cache)    │     
              └─────────────┘     └───────────────────┘
                      ▲                  ▲
                      │                  │
          ┌───────────────┐    ┌──────────────────┐
          │ Smart Chunker  │    │ nomic-embed-text │
          │ (header-aware) │    │ embedding model  │
          └───────────────┘    └──────────────────┘
```

- **Bot framework:** `discord.py` with `app_commands` for slash commands
- **AI client:** `openai.AsyncOpenAI` - routes through OpenAI-compatible gateway
- **Knowledge Base:** Filesystem-based RAG with dual retrieval strategies:
  - **Vector search** (default): Semantic embedding via configured inference backend, SQLite-persisted index
  - **Keyword/TF-IDF fallback**: Heuristic scoring for environments without vector backend
- **Smart chunking:** Head-aware splitting preserves semantic boundaries for better retrieval quality
- **State management:** In-memory dicts keyed by `(guild_id, channel_id)`

---

## Knowledge Base (RAG)

### Retrieval Strategies

1. **Vector Search** (`RAG_RETRIEVAL_METHOD=vector`): Documents are chunked semantically and embedded using the configured AI backend's `/embeddings` endpoint (model: `nomic-embed-text:latest`). A SQLite-persisted index ensures fast bot restarts.

2. **Keyword/TF-IDF** (`RAG_RETRIEVAL_METHOD=keyword`): Lightweight heuristic scoring of filenames, headers, and body text overlap. Works without any vector backend.

### Smart Chunking

The smart chunker splits documents using three strategies:
- **Full document** for small files (≤8000 chars) to preserve context
- **Header-based splitting** with minimum-size merging for larger documents
- **Adaptive paragraph splitting** as fallback, with structural awareness

### Supported File Types
- `.txt` - Plain text files
- `.md` - Markdown files
- `.csv` - CSV spreadsheets
- `.html` - HTML documents
- `.xml` - XML documents
- `.rtf` - Rich Text Format

---

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

**Important:** `characters.json` is private and should NOT be committed. Copy from `characters.json.example`. It's in `.gitignore`.

---

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
/upload_kb <document_file> [url: <url>] [subfolder: <path>]
```
Uploads `.txt`, `.md`, `.csv`, `.html`, `.xml`, or `.rtf` files to the local KB directory. Auto-chunks for vector indexing.

### `/list_kb_docs` — List KB Documents
Lists all documents in `KB_PATH`. Shows name, size, and modification date.

### `/reindex_kb` — Reindex Knowledge Base
Rebuilds the vector index from scratch using smart chunking and configured embeddings.

### `/clear_history` — Clear Conversation History
Clears the conversation history for this server/channel.

---

## Prefix Command (Legacy)

```
<BOT_PREFIX><your_question>
```

Example: `!ai What time is it?`

Uses default character and same history as slash commands.

---

## Startup & Deployment

### Quick Start
```bash
# 1. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env           # Edit with your credentials
cp characters.json.example characters.json  # Configure personas

# 3. Run the bot
python main.py
```

### Using the Startup Script
```bash
./start_bot.sh
```

The startup script is committed as executable. If your checkout loses the execute bit, run `chmod +x start_bot.sh` once as a one-off.

The startup script automatically:
- Creates log directory structure
- Detects and kills existing bot instances (via PID file)
- Streams output to terminal and logs
- Uses `python -u` for unbuffered stdout

### Logging
- **Console**: Visible in terminal
- **bot.log**: INFO+ level, rotated (10 MB × 5 files)
- **dev.log**: DEBUG level, rotated (10 MB × 5 files)

On first run, slash commands are registered globally (may take up to an hour for Discord propagation).

---

## Project Structure

```
discord-ai-bot/
├── main.py                   # Bot entry point, event handlers, slash command registrations
├── bot_core/                 # Core AI client + conversation history
│   ├── __init__.py           # Re-exports for backward compatibility
│   ├── ai_client.py          # Provider calls + RAG orchestration
│   └── history.py            # Per-channel conversation history (singleton)
├── config/                   # Settings and character configuration
│   ├── __init__.py           # Package init (re-exports constants)
│   ├── settings.py           # Environment variable loading & typed constants
│   └── characters.py         # Character/persona loading & display mapping
├── commands/                 # Slash command implementations
│   ├── __init__.py           # Package init
│   ├── ai_command.py              # /ai command handler (delegates to bot_core.ai_client)
│   ├── character_commands.py      # /character command handler
│   ├── clear_history_command.py   # /clear_history handler
│   ├── kb_commands.py             # /upload_kb, /list_kb_docs, /reindex_kb handlers
│   └── utility_commands.py        # /remind, /ocr, /summarize, /translate handlers
├── kb/                     # Knowledge base & RAG modules
│   ├── __init__.py           # Package init
│   ├── reader.py             # Filesystem-based KB reading + relevance scoring
│   ├── storage.py            # Upload, validate, list KB files
│   ├── scorch.py             # TF-IDF relevance scoring for chunks
│   ├── vector_db.py          # In-memory vector index with cosine similarity
│   ├── embedder.py    # Async embedding via configured /embeddings endpoint
│   ├── chunker.py            # Smart document chunking (header-aware + paragraph fallback)
│   ├── index.py              # Persistent SQLite-backed vector index store
│   ├── retrievers.py         # Unified retriever (keyword + vector strategies)
│   └── query_rewriter.py     # Automatic LLM-powered query expansion for RAG
├── utils/                  # Helper functions
│   ├── __init__.py           # Package init
│   ├── kb_utils.py               # KB logging utilities
│   ├── response_splitter.py      # Long message chunking (paragraph-aware)
│   └── typing_loop.py            # Typing indicator task
├── tests/                  # Unit tests (65 tests)
│   ├── conftest.py           # Test fixtures
│   └── ...
├── data/knowledge/         # Knowledge base source files (not committed)
├── characters.json.example # Example character configuration
├── .env.example            # Example environment variable template
├── start_bot.sh            # Startup script with logging setup
└── README.md               # Main documentation
```

---

## Testing

```bash
python -m pytest tests/ -v
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Commands invisible in dropdown | Command sync not propagated | Re-run `python main.py` or wait ~60 min; check `.env` token validity |
| AI responses not appearing | API timeout (default 120s) | Increase `AI_REQUEST_TIMEOUT` in `.env`; check backend logs |
| `characters.json not found` warning | File missing or misnamed | Ensure file exists at project root with valid JSON syntax (use `characters.json.example` as template) |
| KB files not loading | Wrong path or unsupported format | Check `KB_PATH` points to correct directory; use `.txt` or `.md` files |
| Vector search returning 0 results | Embedding backend unreachable | Verify `INFER_URL` and `INFER_API_KEY`; try `RAG_RETRIEVAL_METHOD=keyword` as fallback |
| Double bot instance error | Port/PID conflict from previous run | Check `start_bot.sh` — it auto-kills stale instances via PID file |

---

## License

MIT
