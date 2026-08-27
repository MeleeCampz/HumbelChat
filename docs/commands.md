# Commands

The bot uses Discord slash commands by default, plus a legacy prefix command for backward compatibility.

## Slash commands

All commands begin with `/`.

### `/ai`

```
/ai <message> [character: <name>]
```

Send a message to the active AI character. Uses per-channel conversation history and, if enabled, RAG context.

The optional `character` option takes a **one-off** persona for that single reply — it does not change the channel's active character (use `/character set` for that).

### `/character`

```
/character [action: list|set|show|reset] [name: <character_name>]
```

| Action | Description |
|---|---|
| `list` | List available characters and the current highlight |
| `set` | Switch the active character for this server/channel |
| `show` | Show the active character and model |
| `reset` | Clear the per-channel override and use the default |

### `/upload_kb`

Upload a document to the knowledge base. The expected file types are `.txt`, `.md`, `.csv`, `.html`, `.xml`, and `.rtf`. These are the types the bot will read and index properly. Storage uses MIME-based inference and defaults to `.txt` when the extension is unrecognized, but unsupported extensions are not reliably indexed.

### `/list_kb_docs`

```
/list_kb_docs [path: <subfolder>]
```

Without `path`, shows a root-level overview (top-level folders + files with name, size, and modification date). With `path`, lists everything inside that subfolder.

### `/reindex_kb`

Rebuild the vector index from scratch using smart chunking and configured embeddings.

### `/clear_history`

Clear the conversation history for the current server/channel.

### `/ocr`

Upload an image and extract text from it using vision AI.

### `/summarize`

Summarize recent chat history or content from a URL using AI.

### `/translate`

```
/translate <target_language> [source_language: <language>]
```

Translate text into the target language. Put the text to translate after a colon in the first option — e.g. `/translate Spanish: Hello world`. If no text is given, the bot translates your most recent message in the channel. An optional source language can be given (default: auto-detect).

### `/remind`

```
/remind <time_value> <time_unit: seconds|minutes|hours> <message>
```

Schedule a one-time reminder (minimum 10 seconds ahead); the bot posts a `⏰ Reminder` message in this channel when it fires. Accepted units: seconds, minutes, hours. Reminders persist across restarts (see `REMINDERS_PERSIST_FILE`).

### `/sync`

Re-sync all slash commands with Discord. Use this if commands appear duplicated in the slash command menu or if newly added commands are not showing up.

This command clears Discord's command cache and re-registers the current set of commands globally. It may take a few minutes for Discord to fully update its cache after running `/sync`.

#### Why isn't this done automatically?

The bot used to sync commands on every startup/reconnect, which caused duplicates to accumulate in Discord's cache over time. The fix is a **one-time sync on first run** (tracked by a `.commands_synced` marker file) plus this manual `/sync` command for when you need to refresh.

If you want to force a fresh sync on the next startup (e.g., after adding new commands), delete the `.commands_synced` file in the project root and restart the bot.

## Prefix command

```
<BOT_PREFIX><your_question>
```

Example:
```
!ai What time is it?
```

Uses the default character and shares history with slash commands.

## Notes

- Slash commands are registered globally on first run and may take up to an hour to appear in Discord.
- If commands appear duplicated, run `/sync` to flush Discord's command cache and re-register.
- History is tracked per channel, not globally.
- RAG context is added automatically when relevant documents are available.
