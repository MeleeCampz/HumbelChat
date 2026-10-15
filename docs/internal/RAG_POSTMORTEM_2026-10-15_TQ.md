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
| `RAG_REWRITE_BUDGET_SECONDS` 10 → 45 | `.env` (gitignored) | rewrite survives model swaps; runs concurrently with dense ranking, so no common-case latency cost |
| `RERANK_TOP_K` 100 → 150 | `.env` (gitignored) | wider cross-encoder rescue window; needs the ≤32-text request chunking already on this branch or TEI answers 422 |
| `RERANK_USE_EXPANSION` (new, default on) | `kb/retrievers.py`, `config/settings.py` | rerank against original AND first expansion, keep max per chunk; failed passes return all-0.0 so the merge degrades to pre-rerank order |

Considered and rejected: raising `RAG_MAX_CHUNKS_PER_FILE` 3 → 5 — with the expansion
rerank TQ is rank 1, so the cap is non-binding for this failure; it would only grow
context usage.

## Second failure layer (same day): rewrite "thinking" bug

After rebuilding with the fixes above, the TQ question **still failed** — and the
logs showed the rewrite failing again, this time *not* with a timeout:
```
The model used its full output budget on reasoning and produced no answer.
```

The rewriter disables thinking by sending
`extra_body={"reasoning_effort": "none", "enable_thinking": False}`. Measured
directly against :8888 (IQ4_XS, the bot's model):

| params sent | time | result |
|---|---|---|
| `reasoning_effort=none` **only** | **38.6 s** | content OK — 30× slower generation |
| `enable_thinking=False` **only** | **1.2 s** | clean, fast |
| neither (baseline) | 6.0 s | all 256 tokens spent reasoning → **empty content** |

So on the Unsloth/llama.cpp backend top-level `reasoning_effort="none"` is
*actively harmful*: it defeats a fast generation path, and in the combo it
overrode `enable_thinking` so the model thought its way through the whole
256-token budget → empty expansions → no English query ever reached the reranker
→ the same TQ miss. The 18:16 attempt was the same bug as a timeout (thinking at
~1.4 tok/s exceeded the budget); 18:21 completed but with empty content.

**Fix:** `kb/query_rewriter.py` now sends **only** `enable_thinking=False` (one
plain-call fallback for backends that reject it). Live end-to-end check through
the real `QueryRewriter` class: clean English expansions, ~1 s warm. Also raised
`REWRITE_TIMEOUT_SEC` 10 → 60 (the per-call httpx timeout must exceed the outer
budget — a :8888 model swap mid-call takes ~20 s before the first byte) and
`.env RAG_REWRITE_BUDGET_SECONDS` to 45 so a rewrite survives a swap instead of
timing out. Tests: `tests/test_query_rewrite.py::TestRewriterThinkingParams`.

## Third failure layer (same day): generic expansions + per-file cap

After rebuild #2, attempt 1's rewrite failed (timeout — a model swap overlapped
the query) but attempt 2 ran the FULL fixed pipeline: "Query rewrite triggered
… added 3 expansion list(s) in 5.0s", "Reranked top 150 candidate(s) against 2
queries" — and STILL no answer.

Reproduced the exact pipeline against the production index with live IQ4_XS
expansions. The rewriter's prompt asks for *generic* short phrases, and generic
phrases score near-noise in the cross-encoder against the TQ chunk:

| prompt | sample expansions | TQ rerank rank | TQ score | bucket pos in Saddlebag p2 | in context (cap 3) |
|---|---|---|---|---|---|
| v1 (current, 2 runs) | "equipment hiding attacker from enemies", … | 18 / 12 | 0.030 / 0.106 | **4th** of ~33 | **NO** |
| v2 (mechanic-preserving, 2 runs) | "ranged attack does not reveal position", … | **1 / 1** | 0.900 / 0.710 | **1st** | YES |

So the per-file cap (3) was the coin flip: v1 expansions put TQ 4th among its
own file's chunks → dropped; v2 expansions — which keep the query's specific
mechanics (action type, trigger condition, effect) — put it first with a
decisive margin. **Fix:** `REWRITE_PROMPT_TEMPLATE` gained two rules ("preserve
the SPECIFIC mechanics…", "each distinct mechanic/condition gets its own
expansion line"); pinned by
`tests/test_query_rewrite.py::TestRewriterPromptMechanics`. `.env
RAG_MAX_CHUNKS_PER_FILE` 3 → 5 as a safety margin for other narrow-bucket cases.
Also added chunk-level INFO logging ("attached chunk: …") in `kb/retrievers.py`
so the next failure shows exactly which sections were attached — this
visibility gap is what cost three debug rounds.

## Verification

- Unit: `tests/test_rag_german_retrieval.py::TestRerankExpansionMaxFusion` (4 tests) —
  max-fusion ordering, disable flag, no-expansion single pass, both-passes-failing.
- Suite: no new failures vs clean tree (5 pre-existing order-dependent flakes, green in isolation).
- End-to-end in Discord: **pending** — requires `docker compose up -d --build` (picks up
  the chunking + expansion-rerank code and the `.env` tuning), then re-fire the torture set
  with the agent session idle on :8888 so IQ4_XS is not evicted mid-query.
