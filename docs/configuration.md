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

## Knowledge base and RAG

| Variable | Description | Default |
|---|---|---|
| `KB_PATH` | Path to knowledge base files | `./data/knowledge` |
| `KB_DEFAULT_KB` | Default KB folder slug on startup | `humblewood` |
| `CHUNK_SIZE` | Target chunk size used by legacy indexing | `2000` |
| `RAG_MAX_DOCS` | Max documents attached per RAG query | `4` |
| `RAG_MAX_CHARS` | Hard cap on RAG context characters sent to the LLM | `24000` |
| `RAG_WINDOW_LINES` | Lines above/below each match anchor | `80` |
| `RAG_RETRIEVAL_METHOD` | Retrieval strategy: `vector` or `keyword` | `vector` |
