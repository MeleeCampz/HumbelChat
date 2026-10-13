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

## Phase 2 — swapping the embedding model to Qwen3-Embedding-8B (decision needed)

**Blocker found:** :8888 (Unsloth Studio) keeps **one model resident at a time**
(`/v1/models` shows `loaded: true/false`). Query-time remote embedding would swap
the 27B chat model out and the 8B embedder in on *every* RAG query, then swap back
for the next chat completion — measured first-call load ~13 s. That is exactly the
per-request model-swap problem `EMBED_BACKEND=local` was introduced to avoid.
CPU-local 8B is far too slow per query. So the swap needs a **dedicated embedding
endpoint** that never swaps:

- a TEI embeddings container (like the existing `reranker` service) serving
  `Qwen/Qwen3-Embedding-8B` (~16 GB VRAM fp16) on GPU 1, pointed at via the
  already-existing `EMBED_URL`/`EMBED_API_KEY` split; or a smaller quant if VRAM is tight.

Recipe when approved: set `EMBED_BACKEND=remote`, `EMBEDDING_MODEL` and
`INDEX_EMBED_MODEL` to the Qwen slug, drop `INDEX_EMBED_FALLBACK_LOCAL=1` (auto →
off, since models differ), rebuild container, `/reindex_kb` (~7 min on GPU). The
model-name change in index metadata triggers the one-time reindex automatically.

**Reranker:** keep `bge-reranker-v2-m3` on the TEI GPU service. With markers landing
in the top ~12 of the pool, the cross-encoder's job is ordering + gating, which it
already does; jina-reranker-v3 / bge-reranker-v2-gemma are available upgrades but
buy little now and cost setup/VRAM.

## Reproducibility

Probe scripts lived in `/tmp` (disposable): `rag_opt1.py` (variant cache),
`rag_opt2.py` (bge-m3 pool sweep), `rag_opt3.py` (Qwen index build + sweep),
`rag_opt4.py` (full pipeline eval), `rag_opt5.py` (score distributions).
The standing 33-query golden set in `scripts/rag_eval.py` should be re-run after the
Phase-1 `.env` changes land in the container; expected: no regressions, TQ case stable.
