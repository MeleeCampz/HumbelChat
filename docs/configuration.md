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
| `HISTORY_PERSIST_FILE` | Where chat history + active-character picks are stored; set empty to keep history in RAM only | `<repo_root>/data/history.json` |
| `REMINDERS_PERSIST_FILE` | Where `/remind` reminders are stored so they survive restarts; set empty to disable persistence | `<repo_root>/data/reminders.json` |
| `SESSIONS_PERSIST_FILE` | Where session state (active session + queued next-session reminders) is stored so it survives restarts; set empty to disable persistence. Session notes files live under `SESSIONS_NOTES_DIR` (default `<KB_PATH>/session_notes/`) | `<repo_root>/data/sessions.json` |
| `SESSIONS_NOTES_DIR` | Where per-session folders live — point inside an Obsidian campaign vault for two-way sync (see below) | `<KB_PATH>/session_notes` |
| `WORLD_CONTEXT_FILE` | Campaign world-info file appended to **every** AI prompt (backlog #25): setting, party names, ongoing hooks. Re-read each turn; missing/empty = feature off. Start from `world.md.example` | `<repo_root>/world.md` |

## Knowledge base and RAG

Mechanics live in [RAG / Knowledge Base](./rag.md); this is the variable reference.

| Variable | Description | Default |
|---|---|---|
| `KB_PATH` | Path to knowledge base files | `<repo_root>/data/knowledge` |
| `CHUNK_SIZE` | Display-only: used to estimate "approx N chunks" in `/list_kb_docs`. The chunker itself uses fixed character-based limits and ignores this value | `2000` |
| `RAG_MAX_DOCS` | Max documents attached per RAG query | `4` |
| `RAG_MAX_CHARS` | Hard cap on RAG context characters sent to the LLM | `24000` |
| `RAG_WINDOW_LINES` | Lines above/below each match anchor | `80` |
| `RAG_RETRIEVAL_METHOD` | Retrieval strategy: `vector` or `keyword` | `vector` |

### Embeddings (split query/index path — no model swapping)

Query embeddings run **in-process on CPU**; index builds run on the GPU backend.
Same model both sides, so vectors are interchangeable — see rag.md for the
empirical verification and the fast reindex flow.

**Index-side serving (since 2026-10-25):** the GPU index embedder is the
`embedder-llama` compose service — official digest-pinned `ggml-org/llama.cpp`
server image, official `Qwen3-Embedding-4B-Q4_K_M.gguf`, OpenAI-compatible
`/v1/embeddings`, pinned to GPU 1 via `CUDA_VISIBLE_DEVICES=1` (same image,
pattern and pinning as `reranker-llama`; the old Ollama `embedder-qwen`
container stays in compose, stopped, as rollback). `EMBED_URL` points at the
compose service name; loopback `:8091` is host debug only. Renaming
`EMBEDDING_MODEL`/`INDEX_EMBED_MODEL` invalidates the vector cache and triggers
a one-time full reindex on next bot start (atomic swap keeps the old cache until
the new one is built).

| Variable | Description | Default |
|---|---|---|
| `EMBED_BACKEND` | `local` (in-process CPU query embeddings) or `remote` (legacy backend `/embeddings`) | `local` |
| `LOCAL_EMBED_MODEL` | In-process query-embedding model | `BAAI/bge-m3` |
| `EMBED_QUERY_INSTRUCT` | Optional English task instruction wrapping every QUERY for instruction-aware models (e.g. Qwen3-Embedding); documents are never wrapped; toggling needs no reindex | *(empty)* |
| `INDEX_EMBED_BACKEND` | Where index builds run: `backend` (GPU) or `local` (CPU) | `backend` |
| `INDEX_EMBED_MODEL` | Exact name the backend serves for `/embeddings` during builds (same model as `LOCAL_EMBED_MODEL`) | *(empty)* |
| `RAG_EMBED_BATCH_SIZE` | Docs per `/embeddings` call during index builds | `8` |

### Hybrid retrieval, query rewriting, attach floors

| Variable | Description | Default |
|---|---|---|
| `RAG_HYBRID_ENABLED` | Fuse dense + BM25 legs via reciprocal rank fusion | `1` |
| `RAG_DENSE_TOP_K` / `RAG_LEXICAL_TOP_K` | Candidate pool per leg before RRF fusion | `48` |
| `RAG_QUERY_REWRITER` | Low-confidence query rewriting (LLM rephrasings when the top score is weak) | `1` |
| `RAG_REWRITE_MIN_SCORE` | Similarity below which the rewriter triggers | `0.35` |
| `RAG_QUERY_MAX_EXPANSIONS` / `RAG_REWRITE_BUDGET_SECONDS` | Rewriter shape / wall-clock cap | `2` / `45` |
| `RAG_MIN_ATTACH_SCORE` | Opt-in dense-similarity floor before fusion (`0` = off) | `0` |
| `RAG_MAX_CHUNKS_PER_FILE` | Max chunks attached per file | `3` |
| `RAG_ATTACH_FLOOR` | Per-chunk dense-score gate at attach time; lexical-only chunks always kept; safety net never yields empty context (`0` = off) | `0.50` |

### Cross-encoder reranking (opt-in sidecar)

A cross-encoder (Qwen3-Reranker-4B, jina wire format) re-scores the fused
candidates and reorders them before attach; any failure degrades to the RRF
order. The 0.50 attach floor is calibrated against these scores — do not
shrink `RERANK_TOP_K` / `RERANK_MAX_CHARS` or the server's `-c` context
without re-running `scripts/rag_floor_probe.py`.

**Serving (since 2026-10-25):** pinned `llama.cpp` **container** on GPU 1
(`reranker-llama` service, official digest-pinned `ggml-org/llama.cpp` server
image; `Qwen3-Reranker-4B-Q4_K_M.gguf` self-converted from official weights —
community GGUFs had dropped the classifier head; see HANDOFF_RERANKER_LLAMA.md),
`--alias reranker-4b --reranking --pooling rank -c 2048` (~2.4 GB flat,
150-pair batch ~2.5 s). The bot reaches it on the compose network as
`RERANK_API_BASE=http://reranker-llama:8080/v1`; loopback `:8090` is host
debug only (llama.cpp's server has no auth — never bind to the LAN).
`RERANK_MODEL=reranker-4b` (the `model` field is required by llama.cpp,
ignored harmlessly by vLLM). Score scale ≈ probability (strong ≈ 0.9+, weak
≈ 0.2–0.5, irrelevant ≈ 1e-4) — the 0.50 floor still separates cleanly.
GPU pinning is `CUDA_VISIBLE_DEVICES=1` inside the container: the Windows
nvidia runtime exposes both GPUs to the container regardless of the compose
`device_ids` reservation (verified), and host-side `nvidia-smi dmon` SM
attribution for guest processes is unreliable under WDDM — verify placement by
VRAM delta, not SM%. The old vLLM container (`reranker-qwen`, `:8000`) stays
in compose as the rollback path.

| Variable | Description | Default |
|---|---|---|
| `RERANK_ENABLED` | Enable the reranker sidecar | `0` |
| `RERANK_MODEL` | Model name for logs/display | *(empty)* |
| `RERANK_API_BASE` / `RERANK_API_FORMAT` | Sidecar URL + wire format (`jina` or `tei`) | — |
| `RERANK_INSTRUCT` | Task instruction sent per request (jina format only) | *(empty)* |
| `RERANK_TOP_K` / `RERANK_MAX_CHARS` | Candidates re-scored / chars fed per candidate (keep at calibration values) | `48` / `1200` |
| `RERANK_TIMEOUT_SECONDS` | Per-request timeout; raise to ~60 for large batches (wave-quantized inference) | `8` |
| `RERANK_USE_EXPANSION` | Also rerank against the first query expansion, keep max per chunk (fixes cross-lingual blindness) | `1` |

### Agent mode — tool calling over the KB

Optional bounded agent loop: the model may call KB tools (`search_kb`,
`read_section`, `list_sections`, `find_in_kb`, `list_sessions`) instead of
answering from attached context alone; independent calls in one turn execute
in parallel. Off by default — zero overhead when off.

| Variable | Description | Default |
|---|---|---|
| `RAG_AGENT_MODE` | `off` (fast path only) / `auto` (trigger: top rerank score below gate, enumeration phrasing, or "search the knowledge base") / `always` (every turn) | `off` |
| `RAG_AGENT_SCORE_GATE` | `auto` trigger threshold on the top rerank score (CRAG-style; calibrated: true hits ≥ 0.65, negatives ≤ 0.36) | `0.55` |
| `RAG_AGENT_BUDGET_SECONDS` | Wall-clock cap per agent turn (primary governor) | `45` |
| `RAG_AGENT_MAX_ITERATIONS` | Decision-turn cap (hard safety net) | `2` |
| `RAG_AGENT_MAX_TOOL_CALLS` | Total tool-call cap across the turn (hard safety net; cheap CPU tools + parallel batches make the budget the binding constraint) | `6` |
| `RAG_AGENT_SEARCH_TOP_K` | Snippet count for the agent's `search_kb` | `8` |

### Per-turn AI record logging

One self-contained JSON per `/ai` turn (prompt, retrieval, agent telemetry,
response, timings) for offline debugging; total — a logger failure never
breaks a turn.

| Variable | Description | Default |
|---|---|---|
| `AI_TURN_LOG_ENABLED` | Enable per-turn records (`0` = zero overhead) | `1` |
| `AI_TURN_LOG_DIR` | Folder for the JSON records | `logs/ai_turns` |
| `AI_TURN_LOG_KEEP` | Max files kept (oldest deleted) | `300` |

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

## Contextual sidecar background fill (#52)

| Variable | Default | Meaning |
|---|---|---|
| `CONTEXTUAL_BACKGROUND_FILL` | `0` | After startup, generate missing contextual-retrieval lines for newly ingested chunks in the background (idle-delayed, batch-paused so live `/ai` keeps working, resumable, reloaded per batch). |
| `CONTEXTUAL_FILL_INITIAL_DELAY_S` | `300` | Seconds after startup before the fill begins. |

The coverage **warning** is always on: if the sidecar doesn't cover every
indexed chunk (e.g. after new data was ingested), startup logs a warning with
the coverage and both fill options. Offline alternative:
`scripts/contextual_reindex.py`.

