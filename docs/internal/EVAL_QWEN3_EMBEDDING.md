# Evaluation — Qwen3-Embedding-8B as RAG embedding model + retrieval parameter sweep

**Date:** 2026-10-14 · **Verdict: parameter changes applied; model swap is a Phase-2 decision (needs a dedicated embedding endpoint).**

Companion to `EVAL_EMBEDDINGGEMMA.md` (which tested gemma-2 on a 9-file probe corpus
and kept bge-m3). This round tests on the **full production-shaped index**
(5,387 chunks, all books + session logs) with the 15-case torture suite that exposed
the cross-lingual failure mode in live use.

## Method

- Test index: full `data/knowledge` tree chunked by the production chunker, embedded
  per model (bge-m3 local CPU; Qwen3-Embedding-8B-GGUF via the :8888 GPU backend —
  one-time build ~7 min for 5,387 chunks).
- Fixed query variants: each of the 15 torture queries was expanded **once** through
  the production rewriter and cached, so every config comparison uses identical inputs.
- Pool sweep: per-variant dense + BM25 lists merged with RRF; measured the pool rank
  of the expected answer chunk (strict: ALL markers of a multi-marker case must be in
  the pool). No reranking — isolates the retrieval stage.
- Full pipeline: production `retrieve_kb_documents` path with CPU cross-encoder
  rerank (bge-reranker-v2-m3), rewriter OFF (original query only — the harder
  condition; production adds expansions on top).

## Pool sweep (marker chunk rank, lower = better)

| config | case 0 (DE Trickster's Quiver) | worst marker rank | all 13 within top-50 |
|---|---|---|---|
| bge-m3 d48_l48_k60 **(old production)** | **90** | 90 | 12/13 |
| bge-m3 d96_l96_k60 | **12** | 17 | 13/13 |
| Qwen3-Emb-8B d48_l48_k60 (old top_k!) | **6** | 11 | 13/13 |
| Qwen3-Emb-8B d96_l96_k60 | **5** | 12 | 13/13 |

(d = dense top_k per variant, l = lexical top_k per variant, k = RRF constant.
RRF k=30 never beat k=60; dense 192 added nothing over 96.)

Two independent fixes, each sufficient on its own:

1. **Wider candidate lists (bge-m3, no model change):** d/l 48→96 moves the
   Trickster's Quiver chunk from pool rank 90 to 12 — inside even a 50-wide rerank
   window. Cost is negligible (BM25 + one extra matmul slice).
2. **Qwen3-Embedding-8B:** the same chunk lands at pool rank 4–6 *even with the old
   48/48 settings*; every case's markers sit inside the top ~12 of the pool. The
   cross-lingual DE→EN gap that broke dense retrieval (rank 110 pre-fix) is gone.

## Full pipeline on the Qwen index (original queries only, rerank window 100)

**13/13 positive PASS + 2/2 negative.** Trickster's Quiver is the **top-1 attached
document** for its German question; all four German item/monster queries pass without
any rewriter expansion. The two hallucination traps attach no relevant marker.

### Score thresholds stay valid

Top-1 dense cosine per case: Qwen 0.47–0.81 vs bge-m3 0.47–0.81 — nearly identical
distributions, so `RAG_MIN_ATTACH_SCORE=0.35` and `RAG_REWRITE_MIN_SCORE=0.35` need
no retuning for the swap.

## What was applied (Phase 1 — no new infrastructure)

| setting | old | new | why |
|---|---|---|---|
| `RAG_DENSE_TOP_K` | 48 (default) | **96** | pool-rank fix above |
| `RAG_LEXICAL_TOP_K` | 48 (default) | **96** | same |
| `RERANK_TOP_K` | 200 | **100** | all markers within top ~20 → 5× margin, half the cross-encoder cost/latency |
| `INDEX_EMBED_FALLBACK_LOCAL` | — (always on) | **auto/1/0** | code fix: a mid-reindex backend outage with *different* build/local models would have silently mixed two vector spaces into one index. Fallback now auto-disabled on model mismatch; `=1` kept in `.env` because `gpustack/bge-m3-GGUF` and `BAAI/bge-m3` are the same model under different slugs |

Committed as `f752bb6` (guard + 4 tests); `.env` values are local.

## Golden-set A/B (33 queries, same index, CPU rerank, rewriter off)

| config | hit rate | recall@1 | recall@4 |
|---|---|---|---|
| OLD 48/48/200 | 24/33 | 42% | 73% |
| NEW 96/96/100 | 24/33 | 42% | 73% |

Identical — **no regression** from the parameter change on this set (it targets
the legacy PDF files; the torture suite above is where the pool-rank fix shows).
Same nine misses in both runs, mostly DE queries where a campaign/notes file
outranks the expected rules file.

## Phase 2 — hosting options for the Qwen3-Embedding swap (researched 2026-10-14)

**Blocker found:** :8888 (Unsloth Studio) keeps **one model resident at a time**
(`/v1/models` shows `loaded: true/false`). Query-time remote embedding would swap
the 27B chat model out and the 8B embedder in on *every* RAG query, then swap back
for the next chat completion — measured first-call load ~13 s. That is exactly the
per-request model-swap problem `EMBED_BACKEND=local` was introduced to avoid.
CPU-local 8B is far too slow per query.

### Option A — Ollama sidecar (recommended if a GPU slot exists)

New compose service next to `reranker`; Ollama serves the OpenAI-compatible
`/v1/embeddings` our `Embedder` already speaks, so only `EMBED_URL` changes.
Quantized GGUF tags (official library):

| tag | size | notes |
|---|---|---|
| `qwen3-embedding:8b-q8_0` | 8.0 GB | near-lossless (our gemma Q8_0 check: cosine 0.9998 vs fp) |
| `qwen3-embedding:8b` | 4.7 GB | Q4_K_M default — fine for embeddings, slightly more loss |
| `qwen3-embedding:4b-q8_0` | 4.3 GB | fallback if VRAM is tight |

GPU 1 currently runs the TEI reranker (~2.3 GB): 8b-q8_0 needs ~10.5 GB total,
8b-Q4 ~7 GB. Bonus: Ollama also hosts **Qwen3-Reranker-4B** with a `/api/rerank`
endpoint — a future reranker upgrade path (our `kb/reranker.py` speaks the TEI
dialect `{query, texts}` → `[{index, score}]`; Ollama wants
`{model, query, documents}` → `{results: [...]}`, a small adapter flag).

### Option B — CPU in-process, Qwen3-Embedding-0.6B (no new infrastructure)

Same pattern as today's bge-m3 (`EMBED_BACKEND=local`). Measured on this box
(real index chunks, avg 796 chars): bge-m3 ≈ 1.05 s/chunk, **0.6B ≈ 2.8 s/chunk**
(≈2.7×); per-query embedding ≈ 2.6 s (bge-m3) vs ≈ 5 s (0.6B) for one variant.
The 0.6B has the same 1024-dim output as bge-m3 and is instruction-aware.
Quality on our torture suite: tested in this session (full CPU index build +
eval) — see "Qwen3-Embedding-0.6B CPU results" below when finished. Index build
is one-time (~4 h on this sandbox's CPU; faster if the 0.6B is added to the
:8888 model list and built via `INDEX_EMBED_BACKEND=backend` like today).

### Option C — TEI embeddings sidecar

TEI officially supports all three Qwen3-Embedding sizes (docker examples in its
README), exposes MRL `dimensions`, but loads fp16/bf16: 8B ≈ **16 GB VRAM**,
4B ≈ 8 GB. Heavier than Ollama's quantized GGUFs; pick it only if we want MRL
dimension control or TEI-specific throughput.

### Option D — Unsloth :8888 multi-resident (not available yet)

Open **PR #10876** adds `UNSLOTH_RESIDENT_MODEL_SLOTS` +
`keep_existing_loaded`: each resident GGUF gets its own llama-server process,
requests naming a resident model route to it, default single-model behaviour
unchanged. **Not merged as of 2026-10-14** — would mean running unsloth from a
PR branch for the production chat backend; revisit when it ships.

### Qwen3 family extras (apply to any option)

- **Companion rerankers:** `Qwen3-Reranker-0.6B/4B/8B` — the "built-in
  reranker" exists as sibling models designed to pair with the embedders
  (not literally inside them). Strong multilingual cross-encoders; a later
  upgrade for our TEI/Ollama reranker slot.
- **MRL:** output dimension configurable 32–4096 (TEI/vLLM expose it; Ollama
  returns native dims). Full 4096 is fine — index grows to ~88 MB.
- **Instructions:** `Instruct: <task>\nQuery:<q>` on queries only gives +1–5%
  per Qwen's eval. Our 8B win above was achieved *without* instructions; adding
  them later is a small embedder-side change (queries must never be wrapped in
  the index path).
- **Quantization loss** for embeddings is small (Q8_0 ≈ fp on our corpus);
  Q4_K_M acceptable, Q8_0 preferred when VRAM allows.

### Recipe when approved (Options A/B/C)

Set `EMBED_BACKEND=remote` (A/C) or keep `local` with `LOCAL_EMBED_MODEL=
Qwen/Qwen3-Embedding-0.6B` (B), set `EMBEDDING_MODEL`/`INDEX_EMBED_MODEL`
accordingly, drop `INDEX_EMBED_FALLBACK_LOCAL=1` (auto → off for mismatched
models; keep `=1` only for same-model slug pairs), rebuild container,
`/reindex_kb`. The model-name change in index metadata triggers the one-time
reindex automatically (~7 min on GPU, ~4 h on CPU).

**Reranker:** keep `bge-reranker-v2-m3` on the TEI GPU service. With markers landing
in the top ~12 of the pool, the cross-encoder's job is ordering + gating, which it
already does; jina-reranker-v3 / bge-reranker-v2-gemma are available upgrades but
buy little now and cost setup/VRAM.

## Qwen3-Embedding-0.6B (Option B) — quality unverified

The full 5,387-chunk CPU index build was started and then killed (CPU load),
so the torture-suite comparison for the 0.6B is **not done**. Speed data from
the same session: ≈2.8 s/chunk vs bge-m3's ≈1.05 s on this box; per-query
embedding ≈5 s vs ≈2.6 s (one variant).

Verification recipe when convenient (GPU build, ~5 min, no CPU load):
add `Qwen/Qwen3-Embedding-0.6B` to the :8888 model list, run
`/tmp/rag_qwen06.py` with `INDEX_EMBED_BACKEND=backend`,
`LOCAL_EMBED_MODEL=EMBEDDING_MODEL=<that slug>` (query embedder local or
remote — same space either way), read the pool sweep + 15-case results.
If it clearly beats bge-m3 on the cross-lingual cases, Option B becomes a
zero-new-infrastructure upgrade; if not, go straight to Option A (8B).

## Reproducibility

Probe scripts lived in `/tmp` (disposable): `rag_opt1.py` (variant cache),
`rag_opt2.py` (bge-m3 pool sweep), `rag_opt3.py` (Qwen index build + sweep),
`rag_opt4.py` (full pipeline eval), `rag_opt5.py` (score distributions).
The standing 33-query golden set in `scripts/rag_eval.py` should be re-run after the
Phase-1 `.env` changes land in the container; expected: no regressions, TQ case stable.
