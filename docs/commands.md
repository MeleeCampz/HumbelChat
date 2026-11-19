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

### `/generate_image`

```
/generate_image <prompt> [raw: false] [size: 1024] [dry: false]
```

Paint an image from a short prompt via the Unsloth Studio diffusion model and
send it as a PNG. Base style: semi-realistic fantasy illustration (explicitly
no painterly/watercolor look), set in Humblewood (the D&D world). The rewrite enforces
"no humans — only animals or humanoid animals (Humbelfolk style)" drawn as
upright D&D player characters in adventurer gear. Googly eyes are NOT a
bird-folk trait — only Colum Bidae (Luma) has them, and they're mentioned only
when he's in the picture. No
modern objects, and no unsolicited text — but user-requested text (signs,
book titles, labels) is included verbatim and NOT suppressed by the negative
prompt. The prompt is first
expanded into a rich image prompt by the CHAT model (no extra model involved —
`raw: true` skips the rewrite entirely, sending your text verbatim). `dry: true`
does the prompt work and replies with the FINAL prompt that would be sent — no
GPU mode switch, no generation (for iterating on prompts without burning images).

Three layers build the rewrite system prompt, all re-read on every generation
(so editing any of them takes effect without a restart): `prompts/image_rewrite_system.md`
(structure, world, hard limits), the vault note `HumbleWood/Image Generation Policy.md`
(the `{{CHARACTER_RULES}}` placeholder), and — since 2026-11-18 — the party roster read
from `HumbleWood/Characters/Player Characters.md` (the `{{PARTY_DOSSIER}}` placeholder):
every player character with their Visual Anchor injected in full, so a group request
cannot drift into invented characters. That third layer exists because the live failure
mode was exactly that: "die ganze Gruppe" produced a badger named Bramble and an antlered
elf named Elara, rendered Colum as a raven and Trixy as a tabby cat, and dropped Seta,
Bubo and Romi — the model had been told to look the party up and its lookup failed.
The answer is only accepted as a prompt if it opens with the base style sentence; a reply
that is really a plan ("I need to look up the Player Characters sheet…") gets ONE
corrective retry and otherwise degrades to painting your raw request instead of the plan.
The dossier covers the PARTY only, so a named NPC it does not list is looked up in the
knowledge base first — as ONE batched array of tool calls in a single decision turn, since
the agent loop has only a few turns and one search per turn runs out of budget before the
model can write (live evidence: with the party fixed but no NPC rule, "Boss Frey" and the
freed merchant "Alisa Pennygleam" came out as invented fox-folk and hare-folk; with the
rule, the rewrite found the vault's actual lizard merchant **Spleck** and his blue-and-yellow
wagon). What the vault does NOT yet hold is NPC appearance data — its `Lexikon: Personen`
entries record roles and bounties, not species or colours — so recurring NPCs stay partly
invented until the vault carries Visual Anchors for them. An amount in a request ('a bag
with 70 gold') describes the scene and is not lettering to paint.
Group scenes carry one more rule after the first A/B test on the rendered images: the
cast must not be named twice. Both six-figure scenes had a collective subject in the
scene sentence ('Party members battle…', 'the whole upright party squares off…') AND six
individually described characters, and the model painted an unnamed crowd on top of the
cast — a second Colum in the background in one image, no individual Colum at all in the
other. So a collective subject is banned, the scene sentence must close with the figure
count ('Exactly N named figures in this image.'), each character is listed exactly once,
and `IMAGE_NEGATIVE_PROMPT` gained `duplicate of the same character, identical clone of a
character, doppelganger of a main figure`.
GPU-split under the hood (#42, 2026-11): the rewrite runs on the CHAT backend
(Strata on the 5070) while the 4070 is still in `rag` mode; only then the bot
switches the 4070 into `image` mode (stop reranker/embedder + load the diffusion
model), paints, and switches back to `rag` mode in a `finally` — expect tens of
seconds per image, dominated by the switch windows (~25–100 s load, ~19 s rag
restore). Chat stays available throughout (different card). Size is capped
(`IMAGE_MAX_SIZE`, default 2048) because the working set grows on top of the
~12.5 GiB resident model.
Config-gated: with `IMAGE_MODEL` empty the command replies with a "not
configured" hint. See the image section in `.env` (`IMAGE_MODEL`,
`IMAGE_GGUF_FILENAME`, `IMAGE_INFER_URL`, `IMAGE_GPU_IDS`, `GPU_SWITCH_URL`,
`IMAGE_MODE_HOLD_S`, `IMAGE_REWRITE_LOOKUP_WAIT_S`, `IMAGE_DEFAULT_SIZE`,
`IMAGE_MAX_SIZE`).
Debug trail (#66): every phase logs to `logs/dev.log` (DEBUG) — the raw user
request, rewrite iterations + tool calls + a 200-char preview of the result,
the FULL final prompt and negative prompt actually sent to the backend, load
attempts, generation timing + byte size. Progress: ONE message carries the
whole run — edited per phase (Preparing prompt → switching the GPU to image
mode / loading the image model → Painting) so the switch windows read as
visible progress instead of a frozen "bot is thinking", and the final answer
(text + PNG) replaces the last update IN PLACE on that same message (edit() can
add attachments); only if that edit fails does the image go out as a separate
reply. The stalls are the GPU-mode switch windows (stop RAG → load image →
paint → unload → restart RAG); with `IMAGE_MODE_HOLD_S` > 0 the switch-back is
deferred and back-to-back generations skip it entirely.

Back-to-back `/image` calls: a rewrite is NOT locked to a painting window (it
runs on the chat model, and its tool calls include the reranker — which a mode
switch takes offline), so the second request's rewrite lands INSIDE the first
request's painting window, where `RAG_IMAGE_MODE_LEXICAL_ONLY` skips the dense
leg and the reranker is stopped outright. When the composer reports a finding
whose fix is READING the vault (a named figure with no verified appearance, a
contradicted species, a description left in the scene sentence), that lookup
waits for the window to close — the rag restore that ends the window brings the
sidecars back, so the wait costs nothing the pipeline was not paying anyway —
and the progress message says why (`IMAGE_REWRITE_LOOKUP_WAIT_S` caps it). A
finding the model fixes from its own words (a collective subject, a repeated
name, a meta aside) never waits.

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

Sessions are a global (bot-wide) bookkeeping concept: at most one session is active at a time. Each session gets its own folder under `<KB_PATH>/session_notes/` named `SESSION_<NN>[_<name>]` (a globally unique running number — no date in the name; the dates live in the file header). Its `notes.md` combines the AI-merged session log (all player uploads combined into one de-duplicated record at `/end_session`) + the recap — that single file is part of the RAG-enabled knowledge base and can be edited on disk at any time. The timestamped notes are stored as their own attachment file, `.attachments/notes.md`, alongside the raw uploads/transcripts in the hidden `.attachments/` / `.transcripts/` folders (out of direct RAG — while no merged log exists the notes + upload full text are combined into `notes.md`'s `## Documents` section). When a session has ended, its recap + notes are also attached to every AI turn as a stable last-session context block (see `docs/rag.md`). After `/end_session`, the bot may also paint automatic **key-moment images** (#69): each is archived in the vault `Images/` folder (sibling of the session-log dir) as `Session_NN_moment-<i>_<title-slug>.webp` — the filename carries a slug of the moment's SHORT title, never the prompt — and embedded in the session's `notes.md` under `## Momente` — see `SESSION_IMAGES_DIR` in [Configuration](./configuration.md).

### `/start_session`

```
/start_session [name: <custom_name>]
```

Start a new session. Safety rules:

- Only one session can be active at a time — if a session is still active and younger than 12 h, you must end it first with `/end_session`.
- If the active session is older than 12 h (stale), it is auto-ended (without an AI recap) and the new one starts.

On success the bot also delivers: all queued next-session reminders to their channels, and — if a previous session existed — its **recap**: a brief digest (~150 words, 6–8 short "what happened" bullets plus 1–3 suggested next steps). The recap is generated by the AI at the previous session's `/end_session` and stored with it, so no AI call happens here. A stale auto-ended session has no stored recap — in that case a plain digest of its last notes is posted instead (skipped entirely when nothing was recorded).

The optional `name` customizes the session name; the folder is `SESSION_<NN>[_Name]` with `NN` a globally unique running index (e.g. `SESSION_07_MyName/`), and the session's dates are recorded in the `notes.md` header.

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

Append a timestamped note to the **current** session (requires an active session; notes are stored in the session's notes attachment (`.attachments/notes.md`) and re-indexed for RAG).

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

### `/botperms`

```
/botperms [channel: <id | <#mention> | name>]
```

Diagnoses **why the bot can answer in a channel but cannot post there on its own** — the exact failure behind highlights being archived to disk only, next-session reminders never arriving, and saved roll views not being re-attached.

Without an argument it inspects the channel the command ran in; with one it inspects a channel by id, mention, or name. It reports:

- whether the channel is in the bot's gateway cache (`bot.get_channel`) and whether the REST fallback (`GET /channels/<id>`) succeeds — the two-step resolution every background sender uses;
- the bot's actual permission bits in that channel (`View Channel`, `Send Messages`, `Read Message History`, `Attach Files`, plus the thread/forum ones);
- the channel type and parent, flagging forum channels and threads (a forum parent is not directly messageable);
- a one-line verdict: what will fail, and which permission to change.

The report ends with the caveat that makes this command necessary: slash-command replies ride the interaction webhook and keep working even when every listed permission is denied. See [permissions.md](./permissions.md).

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
