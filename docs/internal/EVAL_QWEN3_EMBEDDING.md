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

### Option A — Ollama sidecar (recommended; **use the 4B**)

New compose service next to `reranker`; Ollama serves the OpenAI-compatible
`/v1/embeddings` our `Embedder` already speaks, so only `EMBED_URL` changes.
Quantized GGUF tags (official library):

| tag | size | notes |
|---|---|---|
| **`qwen3-embedding:4b`** | **2.5 GB** | **recommended — Q4_K_M, tested, quality ≈ Q8_0 ≈ 8B** |
| `qwen3-embedding:4b-q8_0` | 4.3 GB | tested; no measurable gain over Q4 on our corpus |
| `qwen3-embedding:8b-q8_0` | 8.0 GB | tested; no measurable gain over the 4B on our corpus |
| `qwen3-embedding:8b` | 4.7 GB | Q4_K_M default — fine for embeddings, slightly more loss |

GPU 1 currently runs the TEI reranker (~2.3 GB): `:4b` needs ~4.8 GB total,
leaving headroom on an 8 GB card for a later Qwen3-Reranker upgrade. Bonus: Ollama also hosts **Qwen3-Reranker-4B** with a `/api/rerank`
endpoint — a future reranker upgrade path (our `kb/reranker.py` speaks the TEI
dialect `{query, texts}` → `[{index, score}]`; Ollama wants
`{model, query, documents}` → `{results: [...]}`, a small adapter flag).

### Option B — CPU in-process, Qwen3-Embedding-0.6B (no new infrastructure)

Same pattern as today's bge-m3 (`EMBED_BACKEND=local`). Measured on this box
(real index chunks, avg 796 chars): bge-m3 ≈ 1.05 s/chunk, **0.6B ≈ 1.5–2.8
s/chunk** (≈2–2.7×); per-query embedding (one batched call of ~4 short
variants) roughly doubles vs bge-m3.
The 0.6B has the same 1024-dim output as bge-m3 and is instruction-aware.
**Tested — does not beat current production end-to-end** (12/13 + 2 NEG vs
bge-m3's 13/13 + 2 NEG; loses case 0 to sibling-item reranker confusion) and
costs ≈2.7× the CPU embedding time per query. Full data: "Qwen3-Embedding-0.6B
(Option B) — full results" below. **Not recommended.**

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

### Recipe when approved (Options A or C)

Set `EMBED_BACKEND=remote` (A/C) with `EMBED_URL` pointed at the new service,
set `EMBEDDING_MODEL`/`INDEX_EMBED_MODEL` to the Qwen slug accordingly, drop
`INDEX_EMBED_FALLBACK_LOCAL=1` (auto → off for mismatched models; keep `=1`
only for same-model slug pairs), rebuild container, `/reindex_kb`. The
model-name change in index metadata triggers the one-time reindex
automatically (~7 min on GPU). Option B is excluded — see its results section.

**Reranker:** keep `bge-reranker-v2-m3` on the TEI GPU service. With markers landing
in the top ~12 of the pool, the cross-encoder's job is ordering + gating, which it
already does; jina-reranker-v3 / bge-reranker-v2-gemma are available upgrades but
buy little now and cost setup/VRAM.

## Qwen3-Embedding-0.6B (Option B) — full results (2026-10-14)

Full 5,387-chunk index built in-process on CPU (batch 8 fastest: ≈0.86 s/chunk
on an idle box; the actual build took 2 h 18 m under shared load), then pool
sweep + 15-case torture eval with the same fixed variants.

**Pool sweep, d96_l96_k60 — worst-marker rank per case**
(4B = `Qwen/Qwen3-Embedding-4B-GGUF:Q8_0` via :8888, GPU build 329 s):

| Case | bge-m3 | Qwen-0.6B | Qwen-4B | Qwen-8B |
|---|---|---|---|---|
| 0 (TQ DE) | 12 | 12–13 | **3** | 4 |
| 1 | 16 | 8 | 3 | 5 |
| 2 | 12 | 6 | 7 | 6 |
| 3 | 14 | 4 | 5 | 6 |
| 4 | 15 | 1 | 1 | 6 |
| 5 | 13 | 1 | 2 | 5 |
| 6 | 17 | 2 | 2 | 6 |
| 7 | 12 | 3 | 2 | 5 |
| 8 | 12 | 2 | 2 | 4 |
| 9 | 12 | 5 | 5 | 5 |
| 10 | 13 | 5 | 7 | 5 |
| 11 | 17 | 1 | 1 | 6 |
| 14 | 12 | 1 | 1 | 4 |
| **worst** | **17** | **12–13** | **7** | **6** |

The 0.6B sits between bge-m3 and the larger Qwens at pool level (worst case
12 vs 17, most cases 1–8). The **4B is statistically indistinguishable from
the 8B** (worst 7 vs 6; it even beats the 8B on case 0: rank 3 vs 4) at half
the size.

**Full pipeline (CPU rerank, window 100):**

| Model | Result |
|---|---|
| bge-m3 (96/96/100) | 13 PASS + 2 NEG |
| Qwen-0.6B | **12 PASS** + 2 NEG — case 0 FAILS |
| Qwen-4B | **13 PASS** + 2 NEG |
| Qwen-8B | 13 PASS + 2 NEG (TQ final rank 1) |

**Full pipeline (CPU rerank, window 100): 12 PASS + 2 NEG — case 0 FAILS.**
Deep dive: TQ is at pool rank 13, but ~13 of the top-25 pool entries are other
Griffon's Saddlebag items; the cross-encoder then picks sibling items
(Vigilante's Mantle, Blazeblood Rapier) over TQ. The 8B puts TQ at pool rank 4
with far fewer siblings above it, so it wins easily; bge-m3 passes with a
different (luckier) sibling ordering. It is a reranker-confusion failure inside
a dense sibling cluster, not a retrieval miss.

**Verdict:** the 0.6B does **not** beat current production end-to-end (it trades
case 0 for better pool ranks elsewhere), while costing ≈2.7× the CPU embedding
time per query. Option B is **not recommended**.

## Qwen3-Embedding-4B — the sweet spot (2026-10-14)

The 4B matches the 8B end-to-end (13/13 + 2 NEG, case 0 passes) and is pool-
equivalent to it, at half the size: `Qwen/Qwen3-Embedding-4B-GGUF` on HF has
official quants Q4_K_M 2.5 GB / Q5_K_M 2.9 GB / Q6_K 3.3 GB / Q8_0 4.3 GB /
f16 8 GB; Ollama ships `qwen3-embedding:4b` (2.5 GB Q4_K_M) and
`:4b-q8_0` (4.3 GB).

**Quantization sweep (all three tested, same harness, 2026-10-14):**

| Quant | Size | Worst pool rank | End-to-end |
|---|---|---|---|
| Q8_0 | 4.28 GB | 7 | 13 PASS + 2 NEG |
| Q6_K | 3.31 GB | 9 | 13 PASS + 2 NEG |
| **Q4_K_M** | **2.50 GB** | **10** | **13 PASS + 2 NEG** |

Case 0 (the discriminating TQ case) sits at pool rank 3 in **all three**
quants; the worst-rank drift (7→9→10) is in mid-tier cases and changes no
outcome — every marker stays ≤10, ten times inside the rerank window.
**Quantization level does not matter on our corpus.**

**Recommendation: swap to the 4B Q4_K_M (2.5 GB), not the 8B.** Same measured
retrieval quality as the 8B and the Q8_0 quant, at a third of the 8B's VRAM.
Hosting = Option A with `qwen3-embedding:4b` (Ollama sidecar on GPU 1 next to
the reranker: ~4.8 GB total with bge-reranker-v2-m3; leaves headroom for a
Qwen3-Reranker-4B (~2.5 GB Q4) later — ~7.3 GB even with an 8B reranker Q4).
If VRAM is not tight, Q6_K/Q8_0 buy nothing measurable on this corpus.

### :8888 on-demand download (new capability, tested)

Unsloth Studio now downloads missing models on request using the
`repo:QUANT` slug format (`Qwen/Qwen3-Embedding-4B-GGUF:Q8_0`). First request
returns `{"error": {"message": "Downloading '…' (4.3 GB). Retry shortly."}}`,
the model then appears in `/v1/models`. Two gotchas hit during testing:

1. **Bare HF repo names silently fall back** — requesting `Qwen/Qwen3-Embedding-4B`
   (no `-GGUF:QUANT`) returned 1024-dim bge-m3 embeddings with no error.
   Always verify the dimensionality after a model change (4B = 2560).
2. **Swap-in latency vs `EMBED_TIMEOUT=30`** — the first embedding batch after
   the chat model was resident took >30 s (model load ≈16 s), so all three
   retries failed and the index build aborted with an empty error. The test
   run used `EMBED_TIMEOUT=300` + a warm-up call before starting. If :8888 is
   ever used as the *production* query embedder, raise `EMBED_TIMEOUT` (or keep
   the embedder warm) — a dedicated Ollama service has no such swap at all.

### Qwen3-Reranker-4B (companion cross-encoder)

Exists in the same family (0.6B/4B/8B). No official Ollama library tag yet;
community GGUFs are pullable (`ollama pull <user>/qwen3-reranker-…`). llama.cpp
/:8888 has no rerank endpoint, so it needs the Ollama sidecar (or TEI if it
gains support). Our `kb/reranker.py` speaks the TEI dialect and would need a
small adapter for Ollama's `/api/rerank` (`{model, query, documents}` →
`{results: [{index, score}]}`). Quality untested — defer until the embedder
swap lands; with markers already at pool rank ≤7 the current bge-reranker-v2-m3
doing fine is the baseline to beat.

## Query-side instruct (2026-10-14, backlog #33)

Qwen's README recommends prefixing each QUERY with a short English task
instruction (`Instruct: {task}\nQuery:{q}`; documents stay plain) and claims a
1–5 % retrieval gain. Tested on our corpus: fresh Q4_K_M index of the live KB
(5,387 chunks via :8888), all 33 golden-set queries (`scripts/rag_eval.py`) embedded
twice — plain vs with
`"Given a question about rules, items, monsters, or lore from a Dungeons &
Dragons 5e campaign, retrieve relevant passages that answer the question."`
— dense top-20 file-stem ranking:

| arm | found (top-20) | hit@10 | best | worst |
|---|---|---|---|---|
| plain | 30/33 | 26 | 1 | 16 |
| **instruct** | **32/33** | **31** | 1 | **13** |

The wins concentrate exactly where Qwen's training data is strongest: German
rule/table/stat-block lookups (e.g. *Gib mir alle Heilzauber bis Stufe 3*
None→13, *Zeig mir die Magier Tabelle* 16→2, *wizard class table* 14→1) and
lore-file disambiguation (*Welche Hintergründe gibt es in Humblewood?* None→6).
No query regressed beyond a rank-1↔2 swap. **Recommendation: enable it for the
Phase-2 Qwen deployment.**

Implementation: `EMBED_QUERY_INSTRUCT` (default empty = off). When set, query
paths in `kb/vector_db.py` (`query`, `query_with_embeddings`, `rank_texts`) wrap
texts before encoding; document/index paths never do — so it can be toggled
**without reindexing**. Tests: `tests/test_query_instruct.py` (8).
Suggested value for `.env`:
```
EMBED_QUERY_INSTRUCT=Given a question about rules, items, monsters, or lore from a Dungeons & Dragons 5e campaign, retrieve relevant passages that answer the question
```
Verify with the torture questions in Discord after the Phase-2 swap (case 0 is
the canary).

## Reproducibility

Probe scripts lived in `/tmp` (disposable): `rag_opt1.py` (variant cache),
`rag_opt2.py` (bge-m3 pool sweep), `rag_opt3.py` (Qwen index build + sweep),
`rag_opt4.py` (full pipeline eval), `rag_opt5.py` (score distributions),
`instruct_ab.py` (index build + plain/instruct golden-set A/B, rows in
`/tmp/instruct_ab_rows.json`). The standing 33-query golden set in
`scripts/rag_eval.py` should be re-run after the Phase-1 `.env` changes land in
the container; expected: no regressions, TQ case stable.
