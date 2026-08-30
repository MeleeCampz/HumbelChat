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
| `SESSION_SUMMARY_PROMPT` | System prompt used by `/end_session` to write the AI session overview; empty = built-in default | *(empty — built-in default)* |
| `AI_REQUEST_TIMEOUT` | HTTP timeout in seconds | `120` |
| `AI_HEALTH_CHECK_INTERVAL` | `0` probes once at startup; positive integer repeats liveness checks every N seconds | `0` |
| `AI_HEALTH_CHECK_TIMEOUT` | Timeout for each liveness probe | `5` |
| `MAX_TOKENS` | Baseline max tokens per response | `2000` |
| `MAX_TOKENS_HARD_CAP` | Absolute upper bound applied after character/global value is chosen | `4096` |

## Response length defaults

The bot resolves `max_tokens` for each request using this precedence, then clamps the result to `MAX_TOKENS_HARD_CAP`:

1. Per-character `max_tokens` from `characters.json`, if present
2. Global `MAX_TOKENS` from `.env`
3. `MAX_TOKENS_HARD_CAP` from `.env` (final clamp)

A smaller baseline is usually better for Discord. Very large `max_tokens` values can produce over-long outputs and increase the chance of hitting Discord message limits.

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
| `SESSIONS_PERSIST_FILE` | Where session state (active session + queued next-session reminders) is stored so it survives restarts; set empty to disable persistence. Session notes files always live under `<KB_PATH>/session_notes/` | `<repo_root>/data/sessions.json` |

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

## Embed formatting

| Variable | Description | Default |
|---|---|---|
| `EMBED_FORMAT` | Render /ai replies as Beyond20-style Discord embeds; set to 0/false/no for classic plain text | `1` |

Replies are requested non-streaming and delivered as embeds (title +
description + inline fields); long structured replies become multiple embed
messages, and tiny/unstructured replies fall back to plain text. See
[Embeds](./embeds.md) for details.
