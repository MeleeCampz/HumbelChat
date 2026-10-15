# Post-Mortem: German Trickster's-Quiver query failure (2026-10-15)

The torture-test question ("Gibt es ein Item dass mit hilft nicht aufgedeckt zu werden
wenn ich aus Stealth heraus einen Fernkampfangriff mache?") was answered with the wrong
item, then "no such item in context". The right chunk — `The Griffon's Saddlebag (part 2
of 3).md [Trickster's Quiver *(rare)*]` (2,726 chars, contains the stealth-miss property
at char ~180 and all four specialty arrows) — **was in the index**. This doc records what
the pipeline did to it, measured against the production models.

## Method

Repro ran outside the bot with the real components:

- Dense ranking: `Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M` via :8888 `/v1/embeddings`
  (same model + quant as the bot's Ollama sidecar), instruct-wrapped exactly like
  production, cosine over all 5,387 indexed embeddings.
- Lexical: repo `kb.lexical.BM25` + `kb.retrievers.reciprocal_rank_fusion`.
- Rerank: real `BAAI/bge-reranker-v2-m3` cross-encoder (HF cache), 1,200-char truncation
  like production TEI.
- Selection: repo `kb.retrievers.select_ranked_chunks` with production settings
  (`top_n=4`, `max_chunks_per_file=3`, attach floor 0.35).

## Where the chunk went (measured)

| Stage | TQ position | Verdict |
|---|---|---|
| Dense top-96 | rank 57, cosine 0.571 | in the pool — fine |
| Hybrid RRF (dense + BM25) | **rank 113** | outside `RERANK_TOP_K=100` → cross-encoder never scored it |
| Rerank, German original, window 150 | rank 25, rs 0.003 | demoted: all top-40 candidates scored 0.002–0.018 (noise band) |
| Per-file selection | dropped | part 2's bucket filled by other Saddlebag chunks before rank 25 |
| Rerank, **English expansion**, window 150 | **rank 1, rs 0.963** (next: 0.126) | the cross-encoder is decisive when query language matches the KB |

Two independent defects compounded:

1. **Rerank window too narrow for property-specific queries.** The dense top-96 saturates
   with stealth-item near-duplicates (Cloak of Elvenkind ×2, Boots of Elvenkind ×2,
   Shadow Ward, …); RRF then places TQ at 113–118 depending on phrasing. `RERANK_TOP_K=100`
   was tuned under bge-m3 when markers sat within top ~20 — that margin no longer holds.
2. **Cross-lingual rerank blindness.** bge-reranker-v2-m3 scores German-question /
   English-chunk pairs in a 0.002–0.018 band — no discriminative signal. It ordered by
   surface theme (stealth item *names*), not the property in TQ's text. Against the
   rewriter's English expansion the same model puts TQ first with a 9× margin.

A third, operational defect removed the safety net: the rewrite/expansion leg was failing
in production (`Query rewrite failed ()` = `asyncio.TimeoutError`, empty string). The
10 s budget could not survive :8888 model swaps (16–20 s) whenever another model — e.g. a
coding session on Q4_K_M — was resident. With working expansions, TQ is **hybrid rank 1**
(measured), so the window problem never arises for this query.

## Fixes (branch `feat/embed-endpoint-split`, 2026-10-15)

| Fix | Where | Effect |
|---|---|---|
| `RAG_REWRITE_BUDGET_SECONDS` 10 → 30 | `.env` (gitignored) | rewrite survives model swaps; runs concurrently with dense ranking, so no common-case latency cost |
| `RERANK_TOP_K` 100 → 150 | `.env` (gitignored) | wider cross-encoder rescue window; needs the ≤32-text request chunking already on this branch or TEI answers 422 |
| `RERANK_USE_EXPANSION` (new, default on) | `kb/retrievers.py`, `config/settings.py` | rerank against original AND first expansion, keep max per chunk; failed passes return all-0.0 so the merge degrades to pre-rerank order |

Considered and rejected: raising `RAG_MAX_CHUNKS_PER_FILE` 3 → 5 — with the expansion
rerank TQ is rank 1, so the cap is non-binding for this failure; it would only grow
context usage.

## Verification

- Unit: `tests/test_rag_german_retrieval.py::TestRerankExpansionMaxFusion` (4 tests) —
  max-fusion ordering, disable flag, no-expansion single pass, both-passes-failing.
- Suite: no new failures vs clean tree (5 pre-existing order-dependent flakes, green in isolation).
- End-to-end in Discord: **pending** — requires `docker compose up -d --build` (picks up
  the chunking + expansion-rerank code and the `.env` tuning), then re-fire the torture set
  with the agent session idle on :8888 so IQ4_XS is not evicted mid-query.
