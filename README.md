# Discord AI Bot

A self-hosted Discord bot with AI chat, configurable AI personas, and optional RAG/knowledge-base lookup from local files.

## Quick start

Pick one of two ways to run it — that's the whole startup story:

### Production — Docker (recommended for the live bot)

The bot runs in its own container with all runtime data bind-mounted from the
repo — see [Docker deployment](./docs/docker.md).

```bash
cp .env.example .env   # then set DISCORD_BOT_TOKEN, INFER_URL, INFER_API_KEY
docker compose up -d --build   # build + start (also the update flow after git pull)
docker compose logs -f bot     # live logs
docker compose down            # stop
```

### Local development — no Docker

One script sets up the venv and runs the bot in a detached tmux session that
survives browser/terminal shutdowns. Requires **Python 3.10+** (3.12 recommended).

```bash
cp .env.example .env   # then set DISCORD_BOT_TOKEN, INFER_URL, INFER_API_KEY
./botctl.sh start      # creates .venv if missing, then launches in tmux
./botctl.sh logs       # watch output · also: stop | restart | status
```

<details>
<summary>Optional extras & running directly</summary>

The heavy STT and vector dependencies are optional extras (a base install runs
the core bot):

```bash
pip install -e .                # base (AI chat + RAG text retrieval)
pip install -e '.[stt]'         # + local STT (faster-whisper)
pip install -e '.[rag-vector]'  # + vector retrieval (fastembed)
pip install -e '.[all]'         # everything
```

`./botctl.sh setup` refreshes the venv after dependency changes. To run in the
foreground without tmux: `./.venv/bin/python main.py` (Windows:
`.venv\Scripts\python main.py`).
</details>

## Docs

- [Docker deployment](./docs/docker.md) — production container setup, ops commands, KB database access
- [Configuration](./docs/configuration.md) — env vars, response-length/token defaults, and fallback behavior
- [Embeds](./docs/embeds.md) — structured Discord-embed rendering for /ai replies
- [Characters](./docs/characters.md) — `characters.json` format and per-character settings
- [RAG / Knowledge Base](./docs/rag.md) — retrieval methods, smart chunking, supported file types
- [Voice Recording](./docs/voice-recording.md) — per-speaker voice capture for STT (pipeline, manifest format, troubleshooting)
- [Commands](./docs/commands.md) — slash command and prefix command reference
- [Item Rolls](./docs/item-rolls.md) — `/roll_items` table formats, rolling mechanics, and pool management
- [Permissions](./docs/permissions.md) — Discord permissions the bot needs, where to set them, and per-channel override traps
- [Troubleshooting](./docs/troubleshooting.md) — common symptoms and fixes

## Commands (brief)

- `/ai` — chat with the active AI character
- `/character` — list, set, show, or reset AI persona
- `/upload_kb` — add a document to the knowledge base
- `/list_kb_docs` — list knowledge base documents
- `/reindex_kb` — rebuild the knowledge base index
- `/clear_history` — clear channel conversation history
- `/sync` — re-sync all slash commands (fixes duplicated commands)
- `/ocr` — extract text from an image
- `/summarize` — summarize chat history or a URL
- `/translate` — translate text into a target language
- `/roll_items` — roll random items from a CSV item table (descriptions + page links, optional CR scaling, no-repeat pool)
- `/item_search` — fuzzy-search the item tables for a specific item
- `/item_stats` — per-rarity counts and remaining pool size
- `/reset_rolls` — return rolled items to the pool
- `/start_session` — start a work session
- `/end_session` — end the session and generate an AI overview
- `/remind_next_session` — queue a reminder for the next session start
- `/session_notes` — add/view notes for the current or last session
- `/start_recording` / `/stop_recording` — capture voice channel audio per speaker (WAV + timestamped manifest) for STT

Prefix command: `<BOT_PREFIX><message>` (example: `!ai hello`)

Full reference: [Commands](./docs/commands.md)

## Project structure

```
discord-ai-bot/
├── main.py            # entry point (wires bot, commands, startup)
├── config/            # settings.py (env), characters.py
├── bot_core/          # ai_client, history, voice/ (recorder), transcriber, sessions, ...
├── commands/          # slash + prefix command handlers
├── kb/                # storage, chunker, embedder, retrievers, vector_db, ...
├── utils/             # embed_formatter, response_splitter, url_fetch, typing_loop, ...
├── data/              # runtime data (knowledge base, recordings, session notes)
├── botctl.sh          # local-dev control (tmux): setup / start / stop / restart / status / logs
├── requirements.txt   # pinned deps (see pyproject.toml for optional extras)
├── pyproject.toml     # packaging: base + [stt] [rag-vector] [dev] extras
└── docs/              # in-depth docs (index: docs/README.md)
```

## License

MIT
