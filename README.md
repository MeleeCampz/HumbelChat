# Discord AI Bot

A self-hosted Discord bot with AI chat, configurable AI personas, and optional RAG/knowledge-base lookup from local files.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
cp .env.example .env
# edit .env and set at minimum: DISCORD_BOT_TOKEN, INFER_URL, INFER_API_KEY
python main.py
```

Or use a startup script (the bot runs in a detached tmux session so it survives browser/terminal shutdowns):
```bash
./botctl.sh start      # or: restart | stop | status | logs
```
`./start_bot.sh` also works for a simple foreground-style start.

## Docs

- [Configuration](./docs/configuration.md) — env vars, response-length/token defaults, and fallback behavior
- [Streaming delivery](./docs/streaming.md) — how streamed replies grow via edits, and how long replies split into frozen sections
- [Characters](./docs/characters.md) — `characters.json` format and per-character settings
- [RAG / Knowledge Base](./docs/rag.md) — retrieval methods, smart chunking, supported file types
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

Prefix command: `<BOT_PREFIX><message>` (example: `!ai hello`)

Full reference: [Commands](./docs/commands.md)

## Project structure

```
discord-ai-bot/
├── main.py
├── bot_core/
│   ├── ai_client.py
│   └── history.py
├── config/
│   ├── settings.py
│   └── characters.py
├── commands/
├── kb/
├── utils/
├── botctl.sh          # tmux-based run/stop/restart helper
└── docs/
```

## License

MIT
