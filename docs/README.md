# Environment Configuration (.env)

Copy `.env.example` to `.env` and configure all values.

| Variable | Description | Default |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Discord bot authentication token (required) | — |
| `OPENAI_API_KEY` | API key for local AI backend | `local-model-key` |
| `OPENWEBUI_API_KEY` | OpenWebUI-specific API key for knowledge base features | *(falls back to `OPENAI_API_KEY`)* |
| `OPENAI_API_URL` | Base URL of the inference backend gateway (OpenWebUI) | `http://localhost:8080/v1` |
| `MODEL_NAME` | Default model slug for all chat commands | `default-model-name` |
| `VISION_MODEL` | Model used for OCR/image tasks | *(falls back to `MODEL_NAME`)* |
| `SUMMARIZE_MODEL` | Model used for summarization | *(falls back to `MODEL_NAME`)* |
| `TRANSLATE_MODEL` | Model used for translation | *(falls back to `MODEL_NAME`)* |
| `SYSTEM_PROMPT` | System prompt injected into every chat context | `You are a helpful AI assistant embedded in a Discord bot.` |
| `CONTEXT_WINDOW` | Number of message *rounds* (pairs) retained per channel | `10` |
| `AI_REQUEST_TIMEOUT` | HTTP timeout for API calls, in seconds | `120` |
| `BOT_PREFIX` | Text prefix for non-slash commands (e.g., `!ai`) | `!ai` |
| `KB_KNOWLEDGE_BASE` | Name of the OpenWebUI knowledge base to use (usually "HumbleWood") | `Default` |
| `KB_SHARED_DIR`     | Mount path on bot side for RAG chunk reads from shared KB volume | `/shared-knowledge/kb/uploads` |

---

# Container and Volume Setup for RAG

The bot performs **filesystem-based RAG** — instead of querying OpenWebUI's HTTP API, the bot reads knowledge-base files directly from a Docker named volume declared in each service's compose file. This shared volume lets both Open Web UI (OWUI) and the Discord Bot see the same physical path on disk:

### Steps to configure shared volume

1. Add a named volume declaration to **both** docker-compose.yml files:

```yaml
volumes:
  kb_shared:
    driver: local   # Docker native volume driver
```

2. Open Web UI mounts it read-write for KB uploads — this is where `/kb/uploads` lives.


3. The bot service also mounts the same named volume under `KB_SHARED_DIR`:

```yaml
services:
  open-webui:
    environment:
      OPENWEBUI_API_KEY: "your-owui-key"
      KB_UPLOAD_PATH: /kb/uploads   # OWUI writes there on disk
  
  bot-open-terminal:
    environment:
      KB_SHARED_DIR: "/kb/uploads"    # tell the bot where to find them
  
  bot-open-terminal:
    volumes: 
      - kb_shared:/shared-knowledge/kb/uploads:ro  # read-only mount into container
```

Make sure the volume name (`kb_shared` or whatever you choose) and the driver match in both files. The `KB_SHARED_DIR` env var tells the bot at startup which directory to search.

### After adding KB docs through Open Web UI

When you upload a document via `/upload_kb`, the gateway writes its `.chunks.txt` file inside your shared volume (`KB_SHARED_DIR`). On every /ai query the bot reads from that mount path — no cache invalidation or restart necessary.

---

# Knowledge Base (RAG) Logging

The bot provides console logging for knowledge base document usage:

```
RAG: Attaching 3 KB document(s) to context: ["HumbleWood Lore", "Character Races", "Campaign Guide"]
ask_ai → model=gemma4:latest messages_in_prompt=8 KB_files=3
```

This helps you see which documents are being included in each AI request for debugging and optimization purposes.

---

# Character Configuration (`characters.json`)

Controls which AI personas/models are available. Structure:

```json
{
  "default": "System",
  "characters": {
    "System": {
      "model": "qwen3.6:latest",
      "system_prompt": ""
    },
    "Trixy Smoldersome": {
      "display": "Trixy Smoldersome",
      "model": "hf.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M",
      "system_prompt": "..."
    }
  }
}
```

| Field | Description |
|---|---|
| `default` | Character used when none is explicitly selected |
| `characters.<name>` | Each key becomes an available persona in the `/character list` dropdown and `/ai` autocomplete |
| `display` | Human-readable name shown in Discord UI (optional, falls back to key) |
| `model` | Model slug sent to the inference API for that character |
| `system_prompt` | Custom system prompt for this persona (optional) |
| `max_tokens` | Maximum tokens for responses (optional) |
| `temperature` | Creativity setting 0-1 (optional, default 0.7) |

Characters are loaded at bot startup and synced into a Discord dropdown menu automatically. Per-guild per-channel overrides keep separate active characters — switching one channel doesn't affect others.

---

# Slash Commands Reference

All slash commands begin with `/`. They are registered globally on bot startup (`/character list` is ephemeral by default).

## `/ai` — AI Chat

```
/ai <message> [character: <name>]
```

Sends a prompt to the active AI character and returns a reply. Uses per-channel conversation history (last `CONTEXT_WINDOW` rounds are retained).

| Parameter | Required | Description |
|---|---|---|
| `message` | **Yes** | Your question or prompt |
| `character` | No | Override the active character; dropdown auto-populates from `characters.json`. Falls back to per-channel default (usually "System") |

The bot defers the response and sends typing indicators every 10s while waiting. Replies longer than 1900 characters are automatically split into sequential messages with `[X/Y]` markers. Code blocks and lists are preserved across splits where possible.

---

## `/character` — Persona Management

```
/character [action: <list|set|show|reset>] [name: <character_name>]
```

Manages which AI persona is active per channel. All output is ephemeral (visible only to you).

| Action | Description | Example |
|---|---|---|
| `list` *(default)* | Lists all available characters with their models and highlights the current one | `/character list` |
| `set` | Switches active character for this server/channel pair | `/character set Trixy Smoldersome` |
| `show` | Shows currently active character and its model | `/character show` |
| `reset` | Clears the per-channel override, reverting to the default from `characters.json` | `/character reset` |

**Key detail:** Character selection is scoped to `(guild_id, channel_id)` — switching in one voice/text channel doesn't affect another. The active character persists until reset or the bot restarts.

---

## `/remind` — One-Time Reminders

```
/remind <amount> <unit> <message>
```

Schedules a deferred reminder that pings you back at the specified time. Minimum delay is 10 seconds. Output is ephemeral.

| Parameter | Description | Valid values |
|---|---|---|
| `amount` | Numeric value (integer ≥ 1) | Any positive integer |
| `unit` | Time unit | `seconds`/`s`, `minutes`/`min`/`m`, `hours`/hr`/`h` |
| `message` | Reminder text | Free text (shown when the reminder fires) |

**Examples:**
- `/remind 15 minutes Say hello to Bob`
- `/remind 30s Check the server logs`
- `/remind 2 hours Standup in 30 minutes`

The bot sends a direct message to the channel or DM where you issued the command after the delay expires.

---

## `/ocr` — Image Text Extraction (Vision)

```
/ocr [image: <attached_image>]
```

Downloads an attached image, downloads it from Discord's CDN, encodes it as a base64 data URI with the correct MIME type, and sends it to the vision-capable model for OCR. The response is returned verbatim — no interpretation or summarization.

| Parameter | Description |
|---|---|
| `image` | Attach an image file when invoking the command |

**Supported formats:** PNG, JPEG, GIF, WebP (auto-detected from filename; defaults to PNG).  
Uses `VISION_MODEL` (falls back to `MODEL_NAME`) for inference.

---

## `/summarize` — Text Summarization

```
/summarize [file_url: <optional_URL>]
```

Summarizes content into bullet points using a dedicated model (`SUMMARIZE_MODEL`). The source defaults to the last 30 rounds of the current channel's conversation history.

| Parameter | Description |
|---|---|
| `file_url` | Optional HTTP/HTTPS URL pointing to a text-based file (`.txt`, `.md`, etc.). If omitted, recent chat history is summarized instead |

Output is split into chunks if exceeding 1900 characters with `[X/Y]` markers.

---

## `/translate` — Text Translation

```
/translate <target_language>:<optional_source_text> [source_language: <language>]
```

Translates text using the AI model configured via `TRANSLATE_MODEL`. If no source text is provided, the bot uses the last user message in channel history. The translation prompt enforces **return-only-the-translated-text** — no extra commentary.

| Parameter | Description | Example |
|---|---|---|
| `target_language` | Language to translate *into*. Optionally include source as `source:text` | `Spanish: Hello world` or just `Spanish` |
| `source_language` | *(Optional)* Source language hint (e.g., "English"). If omitted, model auto-detects | `source_language: English` |

---

## `/upload_kb` — Knowledge Base Document Upload

```
/upload_kb <document_file>
```

Uploads an attached file (PDF, TXT, CSV, HTML, XML, MD, RTF) to the configured OpenWebUI knowledge base (`KB_KNOWLEDGE_BASE`). The bot detects MIME type from headers or filename extension. Uses the `OPENWEBUI_API_KEY` for authorization — must be set in `.env`.

Supported file types:
- PDF (`.pdf`)
- Plain text (`.txt`)
- CSV (`.csv`)
- HTML (`.html`)
- XML (`.xml`)
- Markdown (`.md`)
- Rich Text (`.rtf`)

---

## `/list_kb_docs` — Knowledge Base Inventory

```
/list_kb_docs
```

Lists all documents currently stored in the active knowledge base. Shows name, size in bytes, and document ID for each entry. Entries are capped at 50 per display; extras show "... and N more."

---

## `/clear_history` — Conversation Reset

```
/clear_history
```

Clears the conversation history (context window) for this server/channel combination. The AI will start fresh with no memory of previous messages. Ephemeral confirmation sent upon success.

---

# Prefix Command

In addition to slash commands, the bot responds to a legacy prefix format:

```
<BOT_PREFIX><your_question>
```

Default prefix is `!ai`. Example: `!ai What time is it?`

| Details | Value |
|---|---|
| Configurable via `.env` `BOT_PREFIX` variable | `!ai` (default) |
| Always uses the default character (System) regardless of per-channel overrides | Yes |
| Uses the same conversation history as slash commands | Yes |

---

# Architecture Overview

```
┌──────────┐    ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Discord  │◄──►│   Bot (Py-  │◄──►│  OpenWebUI      │◄──►│  LLM Backend │
│  Gateway  │    │  discord)   │    │  Gateway        │    │  (any)       │
└──────────┘    └─────────────┘    └─────────────────┘    └──────────────┘
   Messages       Slash commands    OpenAI-compatible   Document upload/
   + Attachments  + Typing loops    API calls           KB management, user mgmt
```

- **Bot framework:** `discord.py` with `app_commands` (slash commands) and `commands.Bot` (prefix fallback)
- **HTTP client:** `httpx.AsyncClient` for image downloading, file uploads, and knowledge base queries
- **AI client:** `openai.AsyncOpenAI` — routes all inference through OpenWebUI's gateway; the underlying LLM backend is fully abstracted and can be swapped in the future without changing the bot code.
- **State management:** In-memory dicts keyed by `(guild_id, channel_id)` for characters and history; no external database

---

# Startup & Deployment

```bash
pip install -r requirements.txt   # or: pip install discord.py openai httpx python-dotenv
cp .env.example .env              # edit .env with your credentials
python main.py
```

On first run the bot calls `bot.tree.sync()` to register all slash commands globally. If commands don't appear in Discord after startup, try `/character list` — if it works the commands are registered; otherwise re-run sync or wait up to an hour for global propagation.

---

# Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Commands invisible in dropdown | Command sync not propagated | Run `python main.py` again or wait ~60 min; check `.env` token validity |
| `/ocr` → `illegal base64 data at input byte 5` | Image URL sent directly instead of base64 data URI | Verified fixed in latest code — downloads and encodes from Discord CDN |
| Summarize / KB upload → HTTP 400 | Knowledge base payload missing context injection or wrong auth | Ensure `OPENWEBUI_API_KEY` is set; verify OpenWebUI backend is reachable at `API_BASE_URL` |
| AI responses not appearing | API timeout (default 120s) | Increase `AI_REQUEST_TIMEOUT` in `.env`; check backend logs for model load delays |
| `characters.json not found` warning | File missing or misnamed | Ensure `characters.json` exists at project root with valid JSON syntax |

---