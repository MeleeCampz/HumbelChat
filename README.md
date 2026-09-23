# Discord AI Bot

A self-hosted Discord bot with AI chat, configurable AI personas, and optional RAG/knowledge-base lookup from local files.

## Quick start

Requires **Python 3.10+** (3.12 recommended — see `.python-version`).

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
cp .env.example .env
# edit .env and set at minimum: DISCORD_BOT_TOKEN, INFER_URL, INFER_API_KEY
python main.py
```

Prefer the packaged install (base + optional extras) instead of `requirements.txt`:
```bash
pip install -e .                # base (AI chat + RAG text retrieval)
pip install -e '.[stt]'         # + local STT (faster-whisper)
pip install -e '.[rag-vector]'  # + vector retrieval (fastembed)
pip install -e '.[all]'         # everything
```

Or use a startup script (the bot runs in a detached tmux session so it survives browser/terminal shutdowns):
```bash
./botctl.sh start      # or: restart | stop | status | logs
```
`./start_bot.sh` also works for a simple foreground-style start.

## Docs

- [Configuration](./docs/configuration.md) — env vars, response-length/token defaults, and fallback behavior
- [Embeds](./docs/embeds.md) — structured Discord-embed rendering for /ai replies
- [Characters](./docs/characters.md) — `characters.json` format and per-character settings
- [RAG / Knowledge Base](./docs/rag.md) — retrieval methods, smart chunking, supported file types
- [Voice Recording](./docs/voice-recording.md) — per-speaker voice capture for STT (pipeline, manifest format, troubleshooting)
- [Commands](./docs/commands.md) — slash command and prefix command reference
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
├── botctl.sh          # tmux-based run/stop/restart helper
├── requirements.txt   # pinned deps (see pyproject.toml for optional extras)
├── pyproject.toml     # packaging: base + [stt] [rag-vector] [dev] extras
└── docs/              # in-depth docs (index: docs/README.md)
```

## License

MIT
