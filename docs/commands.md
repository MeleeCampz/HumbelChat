# Commands

The bot uses Discord slash commands by default, plus a legacy prefix command for backward compatibility.

## Slash commands

All commands begin with `/`.

**Visibility:** all commands post their output **publicly** in the channel, so every message is visible to everyone in the channel.

### `/ai`

```
/ai <message> [character: <name>]
```

Send a message to the active AI character. Uses per-channel conversation history and, if enabled, RAG context. With agent mode enabled (`RAG_AGENT_MODE`, see [rag.md](./rag.md#agent-mode--tool-calling-over-the-kb)) the AI may additionally call knowledge-base tools to follow up on its retrieval.

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


### `/sync_kb`

Fast, diff-based sync: re-indexes only files that were added, renamed, edited, or
deleted on disk (i.e. managed outside `/upload_kb`, which auto-indexes its own
uploads). Unchanged files are never re-embedded, and a failed embedding call can
not wipe the existing cache. Use it after dropping or renaming documents into
the KB folder by hand.
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

### `/speak`

```
/speak <text>
```

Read the text aloud and send it as a playable audio message (mp3 attachment —
Discord shows a native audio bar). Uses any OpenAI-compatible
`/v1/audio/speech` endpoint; Unsloth Studio serves one at the same base URL as
its chat API. Config-gated: with `TTS_MODEL` empty the command replies with a
"not configured" hint instead of failing. See the TTS section in `.env.example`
(`TTS_MODEL`, `TTS_VOICE`, `TTS_URL`, `TTS_API_KEY`, `TTS_TIMEOUT_S`, `TTS_MAX_CHARS`).

### `/generate_image`

```
/generate_image <prompt> [raw: false] [size: 1024]
```

Paint an image from a short prompt via the Unsloth Studio diffusion model and
send it as a PNG. Base style: semi-realistic painterly fantasy with soft
watercolor touches, set in Humblewood (the D&D world). The rewrite enforces
"no humans — only animals or humanoid animals (Humbelfolk style)" drawn as
upright D&D player characters in adventurer gear, bird-folk always get their
signature VERY large googly eyes (only when birds are in the picture), plus no
text/modern objects. The prompt is first
expanded into a rich image prompt by the CHAT model (no extra model involved —
`raw: true` skips the rewrite entirely, sending your text verbatim). Swap-based under the hood: loading the image model evicts
the resident chat model (Studio design), generation runs inside a no-chat-calls
window, and the chat model is swapped back automatically afterwards — expect
tens of seconds per image. Size is capped (`IMAGE_MAX_SIZE`, default 2048)
because the working set scales with resolution (10 GB+ at high res).
Config-gated: with `IMAGE_MODEL` empty the command replies with a "not
configured" hint. See the image section in `.env` (`IMAGE_MODEL`,
`IMAGE_GGUF_FILENAME`, `IMAGE_GPU_IDS`, `IMAGE_DEFAULT_SIZE`, `IMAGE_MAX_SIZE`).

### `/roll_items`

```
/roll_items [table: <name>] [count: 1-25] [cr: <rating>] [rarity: <rarity>] [consume: false]
```

Roll random items from your CSV item table in `data/items/` (override with `ITEMS_DIR`). The reply is an embed where each item shows name, rarity, description, and an item page link. By default rolls are pure random — nothing is tracked; with `consume` on, rolled items are marked so they don't come up again (see below).

- **`table`** — optional. A CSV filename in `data/items/` without the `.csv` suffix (case-insensitive). Omit to use the default table (`magic_items`; override with the `DEFAULT_ITEM_TABLE` env var).
- **`count`** — how many items to roll (default 3). Rolls are **without replacement** — an item never repeats within one roll. If the table has fewer items than requested, everything is returned with a note.
- **`cr`** — optional monster Challenge Rating (`4`, `1/2`, `1/4`, …). When given, it **overrides `count`**: the tier with the highest `min_cr` ≤ CR wins, and each of its per-rarity ranges is rolled independently.
- **`rarity`** — optional filter for plain rolls (e.g. `rare`). Ignored when `cr` is given.
- **`consume`** — when on, rolled items are marked as used: they're excluded from later `consume` rolls until cleared with `/reset_rolls`. Off by default (pure random, nothing tracked).

### `/item_search`

```
/item_search <query> [table: <name>] [limit: 1-25]
```

Fuzzy-search items (emote prefixes ignored) and list each match with description, rarity, table, and page link — so you can see what an item does without opening the link.

### `/item_stats`

```
/item_stats [table: <name>]
```

Per-table counts: total, rolled out, remaining, plus a per-rarity breakdown (footer with grand totals).

### `/reset_rolls`

```
/reset_rolls [table: <name>]
```

Return consumed items to the pool — one table or all — and report how many came back. Clears manual `/item_exclude` marks too.

### `/item_exclude`

```
/item_exclude <name> [table: <name>]
```

Manually mark an item as used (fuzzy name match, emote prefixes ignored) so it won't come up in `consume` rolls until you put it back or reset. If the name is ambiguous, a shortlist is offered instead.

### `/item_include`

```
/item_include <name> [table: <name>]
```

Put one marked/rolled item back into the pool without resetting the whole table.

See **[docs/item-rolls.md](./item-rolls.md)** for the full picture: file layout, both CSV formats, rolling mechanics, the no-repeat pool, and tier tuning.

### Sessions

Sessions are a global (bot-wide) bookkeeping concept: at most one session is active at a time. Each session gets its own folder under `<KB_PATH>/session_notes/` whose `notes.md` combines notes + the AI-merged session log (all player uploads combined into one de-duplicated record at `/end_session`) + the recap — that single file is part of the RAG-enabled knowledge base and can be edited on disk at any time (raw uploads/transcripts are kept in hidden `.attachments/` / `.transcripts/` folders, out of RAG). When a session has ended, its recap + notes are also attached to every AI turn as a stable last-session context block (see `docs/rag.md`).

### `/start_session`

```
/start_session [name: <custom_name>]
```

Start a new session. Safety rules:

- Only one session can be active at a time — if a session is still active and younger than 12 h, you must end it first with `/end_session`.
- If the active session is older than 12 h (stale), it is auto-ended (without an AI recap) and the new one starts.

On success the bot also delivers: all queued next-session reminders to their channels, and — if a previous session existed — its **recap**: a brief digest (~150 words, 6–8 short "what happened" bullets plus 1–3 suggested next steps). The recap is generated by the AI at the previous session's `/end_session` and stored with it, so no AI call happens here. A stale auto-ended session has no stored recap — in that case a plain digest of its last notes is posted instead (skipped entirely when nothing was recorded).

The optional `name` customizes the session name; the notes file always includes the date and an increasing per-day index, e.g. `2026-08-28_01_MyName.md`.

### `/end_session`

```
/end_session [name: <custom_name>]
```

End the current session. The bot (1) generates a brief AI **recap** — max. ~150 words: 6–8 short "What happened" bullets plus suggested "Next steps" — from the session's own documents (the raw files in `.attachments/` and `.transcripts/`) and its notes, stores it with the session (it is delivered when the NEXT session starts), and (2) merges all player uploads/transcripts into ONE de-duplicated, chronological **session log** that becomes the session's RAG content in `notes.md` (AI down → the mechanical full-text copy is kept instead). The recap uses the active character's model, is appended to the session file, and posted here. It is written in the language of the session's own sources — the model determines this itself from the language rule in the system prompt (the built-in default prompt is written in English). The optional `name` renames the session in its file. If the AI backend is unavailable, a plain-text recap listing the notes is written instead.

The system prompt that drives the recap is customizable via `SESSION_RECAP_PROMPT` in `.env` (see [Configuration](./configuration.md)) — leave it empty to use the built-in default.

### `/remind_next_session`

```
/remind_next_session <message>
```

Queue a reminder that is delivered when the **next** session starts (in the channel where it was queued). If no session is active, a new one is started — but the reminder still waits for the *following* session. Queued reminders persist across restarts and are delivered by `/start_session`.

### `/session_notes`

```
/session_notes [note: <text>] [file: <.txt/.md>]
```

Append a timestamped note to the **current** session (requires an active session; notes are stored in the session file and re-indexed for RAG).

**Adding a text document:** also accepts a **file** — a `.txt` or `.md` attachment — instead of (or in place of) the `note` text:

```
/session_notes file: <.txt or .md file>
```

The whole document is stored as one raw file in the session's hidden `.attachments/` folder (out of RAG); at `/end_session` the AI merges all such uploads into the session's single combined log inside the RAG-indexed `notes.md` — same treatment as voice transcripts (`.transcripts/`). Only `.txt` / `.md` are accepted (other types are rejected), the file must be valid UTF-8 and ≤ 2 MB, and a file takes precedence over an inline `note` when both are supplied.

### `/session_info`

```
/session_info
```

Show info about the **current** session — or the most recent ended session if none is active (reads the notes file from disk, so manual edits are picked up). Displays:

- session name and status (`current` / `last (ended)`), with start time, end time and duration for ended sessions;
- the stored recap (truncated) when one exists;
- all notes added to the session (the most recent 10 in full, long ones split across messages) plus the total count;
- uploaded documents (`📎 Attachments`), plus any legacy voice transcripts (`🎙️ Transcripts` from sessions recorded before the voice feature was removed), each listed with filename and size;
- the session folder name (`notes.md` is the single RAG document; raw files stay in the hidden folders).


### `/session_file`

```
/session_file [session: <number or name fragment>]
```

Get an **old session's full notes as a downloadable file** (#43). Without an
argument it lists all campaign sessions (same ordinals the AI sees); with a
number (`/session_file 3`) or a name fragment (`/session_file alderheart`) it
sends a short overview plus that session's `notes.md` attached as a file.

### `/help`

Shows a categorized overview of **all** available commands (AI & Chat, D&D Items, Knowledge Base, Sessions, Utility, Maintenance). The listing is generated from the live command set, so it always matches what the bot actually has.

With an argument — `/help <command>` — it shows a detail view for that one command: its description plus every option (type, required/default, and min–max where applicable).

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
