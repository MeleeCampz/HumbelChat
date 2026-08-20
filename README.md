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

Or use the startup script:
```bash
./start_bot.sh
```

## Docs

- [Configuration](./docs/configuration.md) — env vars, response-length/token defaults, and fallback behavior
- [Characters](./docs/characters.md) — `characters.json` format and per-character settings
- [RAG / Knowledge Base](./docs/rag.md) — retrieval methods, smart chunking, supported file types
- [Commands](./docs/commands.md) — slash command and prefix command reference
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
└── utils/
```

## License

MIT
