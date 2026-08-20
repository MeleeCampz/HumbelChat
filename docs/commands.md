# Commands

The bot uses Discord slash commands by default, plus a legacy prefix command for backward compatibility.

## Slash commands

All commands begin with `/`.

### `/ai`

```
/ai <message> [character: <name>]
```

Send a message to the active AI character. Uses per-channel conversation history and, if enabled, RAG context.

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

List documents in the knowledge base, including name, size, and modification date.

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
/translate <target_language>: <text>
```

Translate text into the specified target language.

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
