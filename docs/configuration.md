# Configuration

All environment variables are loaded from `.env` at the project root. Copy `.env.example` to `.env` and edit it before first run.

## Required

| Variable | Description | Default |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Discord bot token | — |
| `INFER_URL` | Base URL for the AI inference backend | `http://127.0.0.1:11434/v1` |
| `INFER_API_KEY` | API key for the inference provider | *(empty for local)* |

## AI provider

| Variable | Description | Default |
|---|---|---|
| `MODEL_NAME` | Default model slug for AI requests | *(empty — uses character config)* |
| `SYSTEM_PROMPT` | Fallback system prompt when a character has none | *(empty)* |
| `FALLBACK_MODELS` | Comma-separated model slugs tried in order if the primary model fails (used by `/summarize` and `/translate`) | *(empty)* |
| `SESSION_RECAP_PROMPT` | System prompt used by `/end_session` to write the AI session recap (a very brief digest: what happened + next steps), delivered at the next `/start_session`; empty = built-in default | *(empty — built-in default)* |
| `AI_TIMEOUT_S` | Maximum silence from the AI backend before a request fails (seconds) — see *Timeouts* below | `120` |
| `AI_HEALTH_CHECK_INTERVAL` | `0` probes once at startup; positive integer repeats liveness checks every N seconds | `0` |
| `AI_HEALTH_CHECK_TIMEOUT` | Timeout for each liveness probe | `5` |
| `SUMMARY_CALL_MAX_TOKENS` | Token cap for summary calls (`/summarize`, `/end_session` overview); `0` = model maximum output length | `0` |
| `MAX_TOKENS` / `MAX_TOKENS_HARD_CAP` | **Deprecated** — no longer applied as global defaults; kept for reference | `8000` / `16384` |

## Response length

Requests omit `max_tokens` by default, so the backend (LM Studio) lets the
model use its **maximum output length**. A per-character `max_tokens` in
`characters.json` is still honoured when explicitly set. Summary calls can be
capped separately with `SUMMARY_CALL_MAX_TOKENS`. If a response is truncated
(`finish_reason` `length`) with an empty answer, the bot reports a friendly
error (the model's own output limit was exhausted); a partial answer is
delivered as-is. Long outputs are split into multiple Discord messages.

## Timeouts

A single knob governs all AI calls: **`AI_TIMEOUT_S`** (default `120`) — the
maximum SILENCE from the backend before a request fails.

* Idle watchdog (most calls): the timeout is applied per read, so any
  arriving chunk resets it — a slow-but-alive generation may run
  indefinitely, while a stalled or dead backend fails within `AI_TIMEOUT_S`
  of silence. There is deliberately no total wall-clock cap; `/ai stop`
  cancels an in-flight response. This covers streamed main-chat replies and
  ALL `complete_text` summary calls (`/summarize`, `/end_session` overview
  and merged log) — they run *streamed internally* (the full text is
  collected and returned), which matters for the merged log: its long hidden
  thinking phase plus long answer must not be bounded by a total-time cap.
* Total wait bound: plain non-streaming main-chat replies (`ask_ai`) deliver
  no bytes until the whole generation is done, so `AI_TIMEOUT_S` bounds the
  entire request there.

Raise `AI_TIMEOUT_S` (e.g. `300`) for models with long silent thinking
phases — the bound that applies to them is the wait for the FIRST chunk
(everything before it is silence). `AI_REQUEST_TIMEOUT` is still read as a
fallback alias for older `.env` files.

One thing to know:

* **LM Studio output cap**: some LM Studio versions silently stop generation
* **LM Studio output cap**: some LM Studio versions silently stop generation
  at roughly 10K–16K tokens regardless of the requested budget (open upstream
  bug, `finish_reason` stays `stop`, so it is undetectable). Short replies
  are unaffected; very long merged session logs may be cut. Watch for fixes
  in LM Studio releases.

## Bot behavior

| Variable | Description | Default |
|---|---|---|
| `CONTEXT_WINDOW` | Number of message rounds retained per channel | `10` |
| `BOT_PREFIX` | Prefix for non-slash commands | `!ai` |
| `CHAT_HISTORY_RESET` | Set to `clear`, `1`, `true`, or `yes` to wipe chat history on startup | *(empty)* |
| `MAX_INPUT_CHARS` | Hard cap on user prompt length (chars) | `50000` |
| `AI_RATE_LIMIT_MAX` | Max AI requests per user in the rate-limit window | `5` |
| `AI_RATE_LIMIT_WINDOW` | Sliding-window size for the per-user AI rate limit (seconds) | `60` |

## Persistence paths

| Variable | Description | Default |
|---|---|---|
| `CHARACTERS_FILE` | Path to `characters.json` | `<repo_root>/characters.json` |
| `HISTORY_PERSIST_FILE` | Where chat history + active-character picks are stored; set empty to keep history in RAM only | `<repo_root>/data/chat_history.json` |
| `REMINDERS_PERSIST_FILE` | Where `/remind` reminders are stored so they survive restarts; set empty to disable persistence | `<repo_root>/data/reminders.json` |
| `SESSIONS_PERSIST_FILE` | Where session state (active session + queued next-session reminders) is stored so it survives restarts; set empty to disable persistence. Session notes files live under `SESSIONS_NOTES_DIR` (default `<KB_PATH>/session_notes/`) | `<repo_root>/data/sessions.json` |
| `SESSIONS_NOTES_DIR` | Where per-session folders live — point inside an Obsidian campaign vault for two-way sync (see below) | `<KB_PATH>/session_notes` |

## Voice recording and STT

| Variable | Description | Default |
|---|---|---|
| `RECORDINGS_DIR` | Where `/start_recording` writes per-recording subdirectories (one WAV per speaker + `manifest.json`) | `<repo_root>/data/recordings` |
| `STT_ENABLED` | Transcribe each speaker's WAV automatically after `/stop_recording`; set `0` to record only | `1` |
| `STT_MODEL` | Model slug for the backend's `/v1/audio/transcriptions` endpoint (unsloth-studio: `tiny`, `base`, `small`, `large-v3-turbo`, `large-v3`, `qwen3-asr-0.6b`, `qwen3-asr-1.7b`, or any HF `owner/model`) | `qwen3-asr-1.7b` |
| `STT_LANGUAGE` | Force a transcription language (e.g. `en`, `de`); empty = auto-detect | *(empty)* |
| `STT_TIMEOUT` | Per-file HTTP timeout in seconds (long recordings + cold model load) | `300` |
| `STT_ADD_TO_SESSION` | Append the finished transcript to the active session's notes when STT completes; set `0` to keep transcripts out of the session notes | `1` |

STT runs on the same OpenAI-compatible backend as chat (`INFER_URL` / `INFER_API_KEY`) — see [Voice Recording — Speech-to-text](./voice-recording.md#speech-to-text).

## Knowledge base and RAG

| Variable | Description | Default |
|---|---|---|
| `KB_PATH` | Path to knowledge base files | `<repo_root>/data/knowledge` |
| `CHUNK_SIZE` | Display-only: used to estimate "approx N chunks" in `/list_kb_docs`. The chunker itself uses fixed character-based limits and ignores this value | `2000` |
| `RAG_MAX_DOCS` | Max documents attached per RAG query | `4` |
| `RAG_MAX_CHARS` | Hard cap on RAG context characters sent to the LLM | `24000` |
| `RAG_WINDOW_LINES` | Lines above/below each match anchor | `80` |
| `RAG_RETRIEVAL_METHOD` | Retrieval strategy: `vector` or `keyword` | `vector` |
| `EMBEDDING_MODEL` | Embedding model name for vector search (OpenAI-compatible /embeddings endpoint) | `nomic-embed-text:latest` |

### Obsidian vault sync (#19)

Vaults kept as private GitHub repos are mirrored into the KB and re-indexed on a
periodic background loop — see [Obsidian Vault Sync](./obsidian-vault.md) for the
full setup guide (Obsidian Git plugin, tokens, two-way campaign vault).

| Variable | Description | Default |
|---|---|---|
| `OBSIDIAN_VAULTS` | Semicolon-separated `name=git_url` entries; each vault is cloned to `<KB_PATH>/<name>/` and pulled + incrementally re-indexed. Empty = feature off. Private repos use a token-embedded URL | *(empty)* |
| `OBSIDIAN_CAMPAIGN_VAULT` | Name of the two-way vault that holds session notes (its URL needs write access); must be one of the names in `OBSIDIAN_VAULTS` | *(empty — all one-way)* |
| `OBSIDIAN_VAULT_PULL_INTERVAL` | Vault pull/commit period in seconds; `0` = one startup pass only | `300` |

## Embed formatting

| Variable | Description | Default |
|---|---|---|
| `EMBED_FORMAT` | Render /ai replies as structured Discord embeds; set to 0/false/no for classic plain text | `1` |

Replies are requested non-streaming and delivered as embeds (title +
description + inline fields); long structured replies become multiple embed
messages, and tiny/unstructured replies fall back to plain text. See
[Embeds](./embeds.md) for details.
