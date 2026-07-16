# Discord AI Bot Documentation

A self-hosted Discord bot that forwards messages to a local AI backend and returns responses. Designed for privacy-first, on-premises AI inference.

---

## Environment Configuration (.env)

Copy `.env.example` to `.env` and configure the following variables:

| Variable | Description | Default |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Discord bot token (required) | — |
| `INFER_URL` | Base URL for AI inference backend | `http://127.0.0.1:11434/v1` |
| `INFER_API_KEY` | API key for the inference provider | *(empty for local)* |
| `CONTEXT_WINDOW` | Number of message rounds retained per channel | `10` |
| `BOT_PREFIX` | Prefix for non-slash commands (e.g., `!ai`) | `!ai` |
| `AI_REQUEST_TIMEOUT` | HTTP timeout in seconds | `120` |
| `KB_PATH` | Path to knowledge base files | `./data/knowledge` (default) |
| `CHUNK_SIZE` | Tokens per KB chunk for indexing | `2000` |

---

## Architecture Overview

```
┌──────────┐    ┌─────────────┐    ┌─────────────────┐    
│  Discord │◄──►│   Bot (Py-  │◄──►│  AI Backend     │    
│  Gateway │    │  discord)   │    │  (Ollama, etc.) │    
└──────────┘    └─────────────┘    └─────────────────┘    

                    │
                    ▼
              ┌─────────────┐
              │ KB Files    │
              │ (local fs)  │
              └─────────────┘
```

- **Bot framework:** `discord.py` with `app_commands` for slash commands
- **AI client:** `openai.AsyncOpenAI` - routes through OpenAI-compatible gateway
- **Knowledge Base:** Filesystem-based RAG - reads `.txt` and `.md` files directly from `KB_PATH`
- **State management:** In-memory dicts keyed by `(guild_id, channel_id)`

---

## Knowledge Base (RAG)

The bot performs filesystem-based RAG instead of querying OpenWebUI's HTTP API. Upload documents via `/upload_kb`, and the bot reads them directly from `KB_PATH`.

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

**Important:** `characters.json` is private and should NOT be committed. It's in `.gitignore`.

---

## Slash Commands Reference

All commands begin with `/`:

### `/ai` — AI Chat
```
/ai <message> [character: <name>]
```
Sends prompt to the active AI character. Uses per-channel conversation history.

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
/upload_kb <document_file>
```
Uploads `.txt`, `.md`, `.csv`, `.html`, `.xml`, or `.rtf` files to the local KB directory.

### `/list_kb_docs` — List KB Documents
Lists all documents in `KB_PATH`. Shows name, size, and modification date.

### `/reindex_kb` — Reindex Knowledge Base
Rebuilds chunk indices for all KB files.

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

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
pip install -r requirements.txt

# 2. Copy configuration files
cp .env.example .env                    # Edit with your credentials
cp characters.example.json characters.json  # Configure personas

# 3. Run the bot
python main.py
```

On first run, slash commands are registered globally (may take up to an hour for Discord propagation).

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Commands invisible in dropdown | Command sync not propagated | Re-run `python main.py` or wait ~60 min; check `.env` token validity |
| AI responses not appearing | API timeout (default 120s) | Increase `AI_REQUEST_TIMEOUT` in `.env`; check backend logs |
| `characters.json not found` warning | File missing or misnamed | Ensure file exists at project root with valid JSON syntax |
| KB files not loading | Wrong path or unsupported format | Check `KB_PATH` points to correct directory; use `.txt` or `.md` files |

---

## License

MIT