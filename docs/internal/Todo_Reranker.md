# HumbelChat: TEI Reranker Implementation Plan

> **Status (2026-10-10): Phase 1 + Phase 4 implemented and measured.**
> The GPU reranker is live in `docker-compose.yml` (`reranker` service, GPU 1)
> and the bot uses `RERANK_MODE=http`. Measured via `scripts/rag_eval.py`
> (33-query golden set): hit rate 28→30/33, **recall@1 36%→70%**, median
> latency +0.7 s vs no rerank (in-process CPU mode measured ~10 s/query and
> is kept only as a fallback). Phases 2–3 remain open; Phase 5's regression
> surface is now covered by the standing eval harness.

## Target Architecture (as built)

```text
GPU 0
└── Unsloth Studio
    └── Main chat LLM

GPU 1
└── Docker container (compose service: reranker)
    └── Hugging Face Text Embeddings Inference
        └── BAAI/bge-reranker-v2-m3

HumbelChat bot container
├── Retrieves chunks from vector DB (top 15 candidates)
├── Sends all retrieved chunks to TEI /rerank   (http://reranker:80)
├── Keeps top reranked chunks
└── Sends final prompt to Unsloth main LLM
```

The compose service lives directly in `docker-compose.yml` (pinned to GPU 1,
shares the `hf-cache/` volume for the model download, no host port published).

---

## Environment Variables (current `.env`)

```env
# Reranker
RERANK_ENABLED=1
RERANK_MODE=http                      # http = TEI endpoint | local = in-process CPU
RERANK_API_BASE=http://reranker:80    # compose network address of the reranker service
RERANK_MODEL=BAAI/bge-reranker-v2-m3  # used by local mode + TEI --model-id
RERANK_TIMEOUT_SECONDS=10

# RAG pipeline
RAG_VECTOR_TOP_K=15                   # candidate chunks handed to the reranker
```

---

## Phase 1: RAG Pipeline and Reranker Integration — DONE

Goal: eliminate DnD rule hallucinations using a cross-encoder reranker.

- [x] Create a reranker client in `kb/reranker.py`.
- [x] Reranker client requirements:
  - [x] Send a POST request to `{RERANK_API_BASE}/rerank`.
  - [x] Use JSON payload `{"query": ..., "texts": [...]}`.
  - [x] Send all chunks returned by vector search (top `RAG_VECTOR_TOP_K`).
  - [x] Parse scores defensively (type check, out-of-range index → fallback).
  - [x] Sort chunks by reranker score; caller keeps the final top-N per file.
  - [x] Fall back to original vector order if the reranker fails.
- [x] Update RAG flow (`kb/retrievers.py`): retrieve 15 → TEI /rerank →
      keep top candidates → inject as "Relevant knowledge-base context".
- [x] Do not drop chunks before reranking (all 15 candidates are scored).
- [x] Preserve chunk metadata such as source, document ID, and chunk ID
      (the reranker scores text only; the caller keeps its own name↔chunk map).
- [x] Log rerank latency, retrieved count, reranked count, and final count
      (`kb.reranker` debug line: `Reranked N candidate(s) -> M kept in X ms`).

## Phase 2: System Prompt Restructuring — OPEN

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

## Phase 3: Context Window and KV Cache Management — OPEN

Goal: keep enough context for RAG and chat history without exceeding VRAM.

- [ ] Count tokens for: system prompt, reranked RAG chunks, Discord chat history.
- [ ] Preserve RAG chunks before chat history.
- [ ] If the prompt exceeds the context budget:
  1. Summarize or truncate old Discord messages first.
  2. Only reduce RAG chunks if absolutely necessary.

## Phase 4: Health Checks and Fallback Behavior — DONE

- [x] On startup, check TEI health (`GET {RERANK_API_BASE}/health` in
      `preload_reranker()`; logs a warning and continues if unreachable).
- [x] If TEI is unavailable: log a warning, keep answering in vector order,
      never crash. Every request retries independently (no cached failure).
- [x] If a rerank request times out (`RERANK_TIMEOUT_SECONDS`): abort, use
      original vector search order.

## Phase 5: Testing Checklist

Regression surface is now automated by `scripts/rag_eval.py` (33-query golden
set through the real `retrieve_kb_documents()` entry point; run in the live
container, results to `data/rag_eval_*.json`). Remaining manual checks:

- [x] Retrieval side of "known rule question" — expected files are attached
      (eval hit rate 30/33; was 28/33 without the reranker).
- [ ] Stop the TEI container and verify the bot still answers using vector search.
- [ ] Ask a table question and verify the FINAL ANSWER reproduces the table exactly
      (generation quality — the eval only proves the right file reached the context).
- [x] Ambiguous rules questions no longer lead with generic stat blocks (eval:
      recall@1 36%→70%, monsters-A-Z crowding resolved).
- [ ] Verify consecutive live prompts remain fast while the reranker stays resident.
