# HumbleChat — Local-Only Discord AI Bot

A self-hosted Discord bot that forwards messages from channels to **OpenWebUI** (your local AI gateway) and returns AI responses as Discord messages. All inference runs through OpenWebUI's OpenAI-compatible API layer — nothing touches a model backend directly.

> **Local-only:** All inference runs on your own hardware. No cloud APIs. Private by design.
> **Gateway-first:** Every request routes through OpenWebUI, giving you centralized access to knowledge bases, user management, and model creation for the future.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Container / Volume Setup for Knowledge Base Sharing](#container--volume-setup-for-knowledge-base-sharing)
- [Slash Commands Reference](#slash-commands-reference)
- [Character System](#character-system)
- [Configuration (`.env`)](#configuration-env)
- [Invite the Bot](#invite-the-bot)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [License](#license)

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

## Container / Volume Setup for Knowledge Base Sharing

The bot performs **filesystem-based RAG** — instead of querying OpenWebUI's HTTP API, it reads KB files directly from a shared Docker volume (`KB_SHARED_DIR`). Both containers must share exactly the same physical path on disk:

1. Add a named volume declaration to both `docker-compose.yml` files (name it `kb_shared`):

```yaml
volumes:
  kb_shared:
    driver: local   # docker native volume driver
```

2. Open Web UI mounts it read-write for KB uploads — this is where `/kb/uploads` lives inside OWUI container.


3. Your bot service also mounts the same named volume under `KB_SHARED_DIR`:

```yaml
services:
  open-webui:
    environment:
      OPENWEBUI_API_KEY: "your-owui-key"
  
  # Bot (bot-open-terminal):
  environments:
    KB_SHARED_DIR: /kb/uploads   # tell the bot which directory to search
  
  bot-open-terminal:
    volumes: 
      - kb_shared:/shared-knowledge/kb/uploads:ro   # read-only mount into container image
```

The volume name (`kb_shared` or whatever you prefer) and driver must match in both compose files. The `KB_SHARED_DIR` env var passes that path into the bot at startup so it knows which directory on disk to scan for chunk files.

---

## Slash Commands Reference

All slash commands begin with `/`. Output is visible to everyone unless noted as **ephemeral** (visible only to you).

### `/ai <message> [character: name]`

Send a prompt to the AI and get a reply. Uses per-channel conversation history.

| Parameter | Required | Description |
|---|---|---|
| `message` | **Yes** | Your question or prompt |
| `character` | No | Override the active character; dropdown auto-populates from `characters.json`. Falls back to per-channel default (usually "System") |

The bot defers the response and sends typing indicators every 10s. Replies > 1900 chars are automatically split with `[X/Y]` markers, preserving code blocks and lists.

**Examples:**
- `/ai What's the weather like?`
- `/ai Explain quantum computing [character: Trixy Smoldersome]`

---

### `/character [action] [name]` — Persona Management

Manages which AI persona is active per channel. **Ephemeral.**

| Action | Description | Example |
|---|---|---|
| `list` *(default)* | Lists all available characters with models; highlights current | `/character list` |
| `set` | Switches active character for this channel/guild combo | `/character set Trixy Smoldersome` |
| `show` | Shows currently active character and its model | `/character show` |
| `reset` | Clears override, reverts to default from `characters.json` | `/character reset` |

**Important:** Character selection is scoped to `(guild_id, channel_id)` — switching in one channel doesn't affect another. Overrides persist until reset or bot restart.

---

### `/remind <amount> <unit> <message>` — One-Time Reminders

Schedules a deferred reminder that pings you back at the specified time. **Ephemeral.** Minimum delay: 10 seconds.

| Parameter | Description | Valid values |
|---|---|---|
| `amount` | Numeric value (integer ≥ 1) | Any positive integer |
| `unit` | Time unit | `seconds`/`s`, `minutes`/`min`/`m`, `hours`/`hr`/`h` |
| `message` | Reminder text | Free text |

**Examples:**
- `/remind 15 minutes Say hello to Bob`
- `/remind 30s Check the server logs`
- `/remind 2 hours Standup in 30 minutes`

---

### `/ocr [image: file]` — Image Text Extraction (Vision)

Downloads an attached image from Discord's CDN, encodes it as a proper base64 data URI with correct MIME type, and sends it to the vision-capable model for OCR. Returns extracted text verbatim. **Ephemeral.**

| Parameter | Description |
|---|---|
| `image` | Attach an image file when invoking the command |

**Supported formats:** PNG, JPEG, GIF, WebP (auto-detected from filename; defaults to PNG).

Uses `VISION_MODEL` (falls back to `MODEL_NAME`) for inference.

**Usage:** Type `/ocr`, then drag-drop or paste an image into the attachment slot.

---

### `/summarize [file_url]` — Text Summarization

Summarizes content into clear bullet points using a dedicated model (`SUMMARIZE_MODEL`). **Ephemeral.**

| Parameter | Description |
|---|---|
| `file_url` | Optional HTTP/HTTPS URL to a text-based file (`.txt`, `.md`, etc.). If omitted, recent chat history (last 30 rounds) is summarized instead |

Output is split into chunks if exceeding 1900 characters.

**Examples:**
- `/summarize https://example.com/document.txt`
- `/summarize` *(uses last 30 messages in this channel)*

---

### `/translate <target>[:text] [source_language: lang]` — Translation

Translates text using the AI model (`TRANSLATE_MODEL`). If no source text is provided, uses the last user message in channel history. Enforces **return-only-the-translated-text** — no commentary. **Ephemeral.**

| Parameter | Description | Example |
|---|---|---|
| `target` | Language to translate *into*. Optionally prepend source as `source:text` | `Spanish: Hello world` or just `Spanish` |
| `source_language` | *(Optional)* Source language hint (e.g., "English"). If omitted, model auto-detects | `source_language: English` |

**Examples:**
- `/translate Spanish: Hello world`
- `/translate German` *(translates last message in channel)*
- `/translate French: Bonjour le monde source_language: English`

---

### `/upload_kb <document>` — Knowledge Base Document Upload

Uploads an attached file to the configured OpenWebUI knowledge base (`KB_KNOWLEDGE_BASE`). **Ephemeral.** Uses `OPENWEBUI_API_KEY` for authorization.

**Supported file types:** PDF, TXT, CSV, HTML, XML, MD, RTF (auto-detected from filename or MIME header).

---

### `/list_kb_docs` — Knowledge Base Inventory

Lists all documents in the active knowledge base. **Ephemeral.** Shows name, size in bytes, and document ID per entry. Entries capped at 50; extras show "... and N more."

---

### `/clear_history` — Conversation Reset

Clears conversation history (context window) for this channel/guild combination. The AI starts fresh with no memory of previous messages. **Ephemeral.**

```
/clear_history
```

---

## Character System

The bot supports multiple AI personas, each mapped to a different model. Characters are defined in `characters.json` at the project root:

```json
{
  "default": "System",
  "characters": {
    "System": {
      "model": "qwen3.6:latest",
      "system_prompt": "Optional custom system prompt for this character."
    },
    "Trixy Smoldersome": {
      "model": "some-other-model"
    }
  }
}
```

| Field | Description |
|---|---|
| `default` | Character used when none is explicitly selected (also the fallback for prefix commands) |
| `characters.<name>` | Each key becomes an available persona in `/character list` and `/ai` autocomplete dropdown |
| `model` | Model slug sent to the inference API for that character |

**Selection methods:**
1. **Dropdown** — `/ai <message> [character: ...]` auto-populates from `characters.json`
2. **Per-channel override** — `/character set <name>` applies to this `(guild, channel)` combo until reset
3. **Fallback chain** — per-channel override → default in `characters.json`

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and configure all values.

| Variable | Description | Default |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Discord bot authentication token **(required)** | — |
| `OPENAI_API_KEY` | API key for local AI backend | `local-model-key` |
| `OPENWEBUI_API_KEY` | OpenWebUI-specific auth key for KB features | *(falls back to `OPENAI_API_KEY`)* |
| `OPENAI_API_URL` | Base URL of the OpenWebUI gateway | `http://localhost:8080/v1` |
| `MODEL_NAME` | Default model slug for all chat commands | `default-model-name` |
| `VISION_MODEL` | Model for OCR/image tasks | *(falls back to `MODEL_NAME`)* |
| `SUMMARIZE_MODEL` | Model for summarization | *(falls back to `MODEL_NAME`)* |
| `TRANSLATE_MODEL` | Model for translation | *(falls back to `MODEL_NAME`)* |
| `SYSTEM_PROMPT` | System prompt injected into every chat context | `You are a helpful AI assistant embedded in a Discord bot.` |
| `CONTEXT_WINDOW` | Number of message *rounds* (pairs) retained per channel | `10` |
| `AI_REQUEST_TIMEOUT` | HTTP timeout for API calls, in seconds | `120` |
| `BOT_PREFIX` | Prefix for non-slash commands (e.g., `!ai`) | `!ai` |
| `KB_KNOWLEDGE_BASE` | Name of the OpenWebUI knowledge base to use | `Default` |
| `KB_SHARED_DIR` | Mount path on bot side for RAG chunk reads from shared KB volume | `/shared-knowledge/kb/uploads` |

---

## Invite the Bot

Discord Developer Portal → Your Application → Bot tab:

1. Enable **MESSAGE CONTENT INTENT** under Privileged Gateway Intents
2. Copy the invite URL:
```
https://discord.com/oauth2/authorize?client_id=YOUR_CLIENT_ID&scope=bot%20applications.commands&permissions=8
```

---

## Architecture

```
┌──────────┐    ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Discord  │◄──►│   Bot (Py-  │◄──►│  OpenWebUI      │◄──►│  LLM Backend │
│  Gateway  │    │  discord)   │    │  Gateway        │    │  (any)       │
└──────────┘    └─────────────┘    └─────────────────┘    └──────────────┘
   Messages       Slash commands    OpenAI-compatible   Document upload/
   + Attachments  + Typing loops    API calls           KB management, user mgmt
```

- **Bot framework:** `discord.py` with `app_commands` (slash commands) + `commands.Bot` (prefix fallback)
- **HTTP client:** `httpx.AsyncClient` for image downloading, file uploads, and knowledge base queries
- **AI client:** `openai.AsyncOpenAI` — routes all inference through OpenWebUI's gateway; the underlying LLM backend is fully abstracted and can be swapped in the future without changing the bot code.
- **State management:** In-memory dicts keyed by `(guild_id, channel_id)` for characters and history; no external database

**Why OpenWebUI?** OpenWebUI acts as a single intermediary layer that provides:
- Knowledge base management and RAG (your data stays local)
- User management across your team
- Easy model creation and selection from a UI
- A standardized OpenAI-compatible API endpoint — meaning future gateways can replace OpenWebUI without rewriting the bot

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Commands invisible in dropdown | Command sync not propagated by Discord | Restart bot; wait up to 60 min for global propagation |
| `/ocr` → `illegal base64 data at input byte 5` | Image URL sent directly instead of base64 data URI | **Fixed** — bot now downloads from Discord CDN and encodes as proper base64 data URI |
| Summarize / KB upload → HTTP 400 | Knowledge base payload missing context injection or wrong auth | Ensure `OPENWEBUI_API_KEY` is set; verify OpenWebUI backend is reachable at `API_BASE_URL` |
| AI responses never arrive | API timeout (default 120s) | Increase `AI_REQUEST_TIMEOUT` in `.env`; check backend logs for model load delays |
| `characters.json not found` warning | File missing or misnamed | Ensure `characters.json` exists at project root with valid JSON syntax |

---

## License

MIT
