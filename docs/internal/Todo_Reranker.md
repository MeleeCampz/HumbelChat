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

## Reranker swap: bge-reranker-v2-m3 (TEI) → Qwen3-Reranker-4B (vLLM), 2026-10-15

**Why:** TEI cannot serve Qwen3-Reranker — it only supports BERT/XLM-RoBERTa
cross-encoders; the Qwen3ForSequenceClassification path is still an open issue
(huggingface/text-embeddings-inference#643, all PRs closed unmerged). vLLM has
native support (`--hf_overrides ... "is_original_qwen3_reranker": true`).

**What changed:**

- `docker-compose.yml`: new `reranker-qwen` service — `vllm/vllm-openai:v0.31.0`
  (pinned), `boboliu/Qwen3-Reranker-4B-W4A16-G128` with
  `--quantization compressed-tensors` (pre-quantized INT4, W4A16 group-128,
  ~2.6 GB — a third-party 4-bit of the official Qwen3-Reranker-4B),
  `--gpu-memory-utilization 0.5`, `--max-model-len 2048`, GPU 1.
  The chat template is vendored at `deploy/qwen3_reranker.jinja` (vLLM's own
  example, Apache-2.0) and mounted in — without it the model scores with the
  wrong prompt format. The old TEI `reranker` service is profile-gated
  (`tei-rerank`) for rollback.
  - **Why not bitsandbytes/NF4 on the official fp16 weights (the original plan)?**
    In vLLM 0.31.0 BitsAndBytes is an **out-of-tree plugin** (`vllm-bnb-plugin`)
    that is NOT bundled in `vllm/vllm-openai`, so `--quantization bitsandbytes`
    fails with "unknown quantization" in the stock image. compressed-tensors is
    in-tree, so the INT4 checkpoint loads with no custom build. To keep NF4 on
    first-party weights you'd need a custom Dockerfile installing the plugin.
- `kb/reranker.py`: `_rerank_http()` now speaks both wire formats via
  `RERANK_API_FORMAT` — `tei` (unchanged) and `jina` (vLLM's Jina-compatible
  `/rerank`: `documents` field, object response with `results`,
  `relevance_score`). The ≤32 chunking applies to both.
- `.env`: `RERANK_API_BASE=http://reranker-qwen:8000`,
  `RERANK_API_FORMAT=jina`, `RERANK_INSTRUCT=<D&D task instruction>` (folded
  into the template's `<Instruct>` slot per request).

**VRAM budget GPU 1 (16 GB):** vLLM reserves 0.5×16 = 8 GB (INT4 weights + KV);
the Ollama `embedder-qwen` sidecar keeps ~2.5–3 GB outside that reservation.
Headroom for a future Qwen3-Reranker-8B or fp16: limited — plan around it.

**Rollback:** `docker compose --profile tei-rerank up -d reranker`, then set
`RERANK_API_BASE=http://reranker:80` + `RERANK_API_FORMAT=tei` in `.env`.

**Fallbacks if the INT4 checkpoint fails to load:** drop `--quantization`
and point at `Qwen/Qwen3-Reranker-4B` (official fp16, ~8 GB — still fits the
0.5 reservation), or build a custom image with `vllm-bnb-plugin` for NF4.

**First start:** downloads the INT4 checkpoint (~2.6 GB) into `hf-cache/`.
Watch `docker compose logs -f reranker-qwen` for the "Uvicorn running on"
line before testing in Discord.

**Gotcha hit 2026-10-15:** v0.31.0's `vllm-openai` image changed its ENTRYPOINT
to `["vllm", "serve"]` (older images: `["vllm"]`). The compose `command` must
therefore start with the model id, **not** `serve` — a leading `serve` makes the
container crash-loop with `vllm: error: unrecognized arguments: <model id>`
(the subparser eats `serve` as the model tag). Fixed in `docker-compose.yml`.

**Gotcha #2 (found via live 400s, 2026-10-16):** the container booted fine but
every `/rerank` call returned `400 Bad Request` (bot correctly fell back to
pre-rerank order). Root cause: `Qwen3ForSequenceClassification` exists only via
`hf_overrides` — it is NOT in vLLM's model registry — so `--runner auto`
couldn't match it and fell back to `generate`, serving the model as a
generative LLM; the scoring endpoint then rejects requests with 400. Fix:
explicit `--runner pooling` (vLLM's own `qwen3_reranker_online.py` example
passes it). Verified against v0.31.0 sources: `RerankRequest` accepts
`{query, documents, instruction}` (`model` is optional), so the bot's jina
payload was already correct.

**Gotcha #3 (live crash 2026-10-15/16):** with `--gpu-memory-utilization 0.5`
the engine init died with `ValueError: ... 0.28 GiB KV cache is needed, which
is larger than the available KV cache memory (0.06 GiB)`. The INT4 weights +
Marlin kernels loaded fine — the card is simply SHARED with the Ollama
embedder, and vLLM subtracts other processes' VRAM from its 50% budget.
Fix: `--gpu-memory-utilization 0.7` (plenty of KV headroom for 32-pair rerank
batches, still leaves room for the embedder) +
`CUDA_DEVICE_ORDER=PCI_BUS_ID` (host has two different GPUs; vLLM warns that
device "1" is otherwise not guaranteed to be the PCI-bus second card).

**Debugging from the dev sandbox (2026-10-16):** the sandbox can reach
*host-bound* ports via `http://host.docker.internal:<port>` (the Unsloth/LM
Studio backend on :8888 works with `INFER_API_KEY`), but compose-internal
service names (`reranker-qwen:8000`, `embedder-qwen:11434`) don't resolve
there. So `reranker-qwen` publishes its port **loopback-only**
(`127.0.0.1:8000:8000` in docker-compose.yml) — from the sandbox:

```bash
curl -s http://host.docker.internal:8000/health
curl -s http://host.docker.internal:8000/rerank -H 'Content-Type: application/json' \
     -d '{"query": "test", "documents": ["a", "b"]}'

Better: `deploy/reranker_smoke.py` — zero-dependency end-to-end check
(/health + Jina response shape + relevant-outranks-irrelevant for an EN and
a DE query). On the host: `python3 deploy/reranker_smoke.py`; from the
sandbox: `python3 deploy/reranker_smoke.py http://host.docker.internal:8000`.
Exit 0 = healthy.
```

(Loopback, not LAN: vLLM's OpenAI server has no auth. `embedder-qwen`
likewise publishes `127.0.0.1:11435:11434` — host port 11435 so a native
host Ollama on 11434 is unaffected; reachable at
`http://host.docker.internal:11435/v1/embeddings`. Both publishes are inert
until the services are started.)
Also: `kb.reranker` now logs the server's response body (truncated to 300
chars) on HTTP errors, so a future 400 shows *why* immediately instead of
just the status line.
