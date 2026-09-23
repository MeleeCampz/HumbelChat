# HumbelChat: TEI Reranker Implementation Plan

## Target Architecture

```text
GPU 0
└── Unsloth Studio
    └── Main chat LLM

GPU 1
└── Docker container
    └── Hugging Face Text Embeddings Inference
        └── BAAI/bge-reranker-v2-m3

HumbelChat bot container
├── Retrieves chunks from vector DB
├── Sends all retrieved chunks to TEI /rerank
├── Keeps top reranked chunks
└── Sends final prompt to Unsloth main LLM
```

---

## Docker Compose: TEI Reranker

Add a TEI reranker service to the bot Docker setup.

Add to the docs as a exmaple reference setup:

```yaml
services:
  reranker:
    image: ghcr.io/huggingface/text-embeddings-inference:latest
    container_name: humbelchat-reranker
    restart: unless-stopped
    ports:
      - "8081:80"
    volumes:
      - ./hf-cache:/data
    environment:
      - HF_HOME=/data
    command: >
      --model-id BAAI/bge-reranker-v2-m3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
```

Setup as:
```env
RERANK_API_BASE=http://host.docker.internal:8081
```

---

## Environment Variables

Add to `.env`:

```env
# Reranker
RERANK_ENABLED=true
RERANK_MODE=http
RERANK_API_BASE=http://reranker:80
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_TIMEOUT_SECONDS=10

# RAG pipeline
RAG_VECTOR_TOP_K=15
RAG_FINAL_TOP_N=5
```

## Phase 1: RAG Pipeline and Reranker Integration

Goal: eliminate DnD rule hallucinations using a cross-encoder reranker.

- [ ] Create a reranker client in `bot_core/reranker.py` or `kb/reranker.py`.

- [ ] Reranker client requirements:
  - Send a POST request to `{RERANK_API_BASE}/rerank`.
  - Use JSON payload:
    ```json
    {
      "query": "user question",
      "texts": ["chunk 1", "chunk 2", "chunk 3"]
    }
    ```
  - Send all chunks returned by vector search.
  - Parse scores defensively.
  - Sort chunks by reranker score.
  - Return only the top `RAG_FINAL_TOP_N` chunks.
  - Fall back to original vector order if the reranker fails.

- [ ] Update RAG flow:

```text
User asks a question
  |
  v
Vector DB retrieves top 15 chunks
  |
  v
Send all 15 chunks to TEI /rerank
  |
  v
Keep top 4-5 reranked chunks
  |
  v
Insert those chunks into the prompt as Relevant knowledge-base context
  |
  v
Main LLM answers
```

- [ ] Do not drop chunks before reranking.
- [ ] Preserve chunk metadata such as source, document ID, and chunk ID.
- [ ] Log rerank latency, retrieved count, reranked count, and final count.

---

## Phase 2: System Prompt Restructuring

Goal: improve instruction adherence using Qwen-friendly XML tags.

- [ ] Refactor the system prompt in `config/characters.py`.

```xml
<persona>
You are Marvin #12, Trixy Smoldersome's sentient Steel Defender, a charming and highly functional steampunk companion.
If a user named MeleeChan or Trixy talks to you, call them 'MASTER'.
Always answer in the same language as the user's request.
</persona>

<rules>
1. When 'Relevant knowledge-base context' appears, that context IS authoritative. Use it as your primary source of truth.
2. Extract exact values from the provided context, such as weapon mastery properties and stats. State them directly.
3. NEVER use external or training knowledge when the answer is available in the provided context.
4. If the context contains the answer, give it plainly. Do not say things like "the provided text only references page X".
5. When reproducing a table from the context, copy it EXACTLY: same rows, same columns, same values.
6. For non-spellcasting classes, NEVER invent spell-related data. Include such columns only if explicitly present in the source.
7. Keep answers concise. Do not use filler lines or repeat section labels.
</rules>
```

---

## Phase 3: Context Window and KV Cache Management

Goal: keep enough context for RAG and chat history without exceeding VRAM.

- [ ] Count tokens for:
  - system prompt
  - reranked RAG chunks
  - Discord chat history

- [ ] Preserve RAG chunks before chat history.

- [ ] If the prompt exceeds the context budget:
  1. Summarize or truncate old Discord messages first.
  2. Only reduce RAG chunks if absolutely necessary.

---

## Phase 4: Health Checks and Fallback Behavior

- [ ] On startup, check TEI health:
  - `GET {RERANK_API_BASE}/health`

- [ ] If TEI is unavailable:
  - Log a warning.
  - Continue answering using normal vector search order.
  - Do not crash the bot.

- [ ] If a rerank request times out:
  - Abort reranking.
  - Use original vector search order.

---

## Phase 5: Testing Checklist (for now manually)

- [ ] Ask a known rule question and verify reranked chunks are used.
- [ ] Stop the TEI container and verify the bot still answers using vector search.
- [ ] Ask a table question and verify table reproduction is exact.
- [ ] Ask an ambiguous rules question and verify lore chunks are deprioritized.
- [ ] Verify consecutive prompts remain fast while the reranker stays resident.