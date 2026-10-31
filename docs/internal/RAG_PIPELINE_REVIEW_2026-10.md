# RAG pipeline review — #74 (2026-10-31)

Deep start-to-end review of the retrieval pipeline: ingest → chunk → embed → index →
query rewrite → dense + lexical → RRF → rerank → floor/select → prompt attach → agent
tools. Method: full read of every pipeline module, the live production index
(`data/knowledge/.vector_index_cache/vector_index.db`), `docker-compose.yml`, `.env`,
the docs (`docs/rag.md`, `docs/internal/RAG_ARCHITECTURE_RESEARCH.md`), plus two
hermetic probes against the real code paths (dedup wiring, chunk-length distribution).
**Review only — no pipeline code was changed by this item.**

Headline: the pipeline is in genuinely good shape — hybrid retrieval, calibrated
rerank floor, sidecar'd contextual retrieval, and GPU-side embedding/rerank containers
that never disturb the chat model. But one shipped feature (near-duplicate dedup)
is **provably dead in production wiring** (F1), a coverage gap (missing context
lines, F2) silently degrades new data, and a few config/doc issues are worth tidying.

---

## 1. End-to-end map (production config as of 2026-10-31)

```
Discord message
  └─ ai_client._build_ai_request
       ├─ retrieve_kb_documents (kb/retrievers.py, strategy=vector)
       │    ├─ _ensure_index_store          lazy singleton KBIndexStore (SQLite)
       │    ├─ query_with_embeddings        ── EMBED_URL=http://embedder-llama:8080/v1
       │    │     query embedded with EMBED_QUERY_INSTRUCT (docs stay plain) → dense top 96
       │    ├─ dense_scores snapshot        (raw cosine, for the floor fallback)
       │    ├─ RAG_MIN_ATTACH_SCORE=0.35    pre-filters dense leg below 0.35
       │    ├─ LLM query rewrite            ── :8888 chat model (THE ONLY chat-model dependency)
       │    │     runs CONCURRENTLY with dense ranking; thinking off; load-state-aware
       │    │     budget (10 s warm / 90 s cold); original + up to 3 English expansions
       │    ├─ expansion dense rankings     rank_texts(≤3 expansions, each top 96)
       │    ├─ RRF fusion                   original + expansions (k=60)
       │    ├─ BM25 lexical leg             in-memory BM25 over all 5903 chunks, scored for
       │    │     every query variant, RRF-merged, then fused with the dense ranking
       │    ├─ cross-encoder rerank         ── RERANK_API_BASE=http://reranker-llama:8080/v1
       │    │     top 150 candidates × {query, first expansion} (max-fused), jina wire format,
       │    │     ≤32 texts/request, chunk text truncated to 1200 chars for SCORING only
       │    ├─ attach floor 0.50            gated on RERANK scores (RAG_ATTACH_FLOOR_SOURCE=rerank);
       │    │     lexical-only hits bypass the floor; empty-result safety net re-selects unfiltered
       │    ├─ per-file caps               ≤5 chunks/file, ≤24 000 chars/file
       │    └─ near-duplicate dedup         cosine τ=0.92 vs already-selected — BROKEN, see F1
       ├─ select → ≤4 files (RAG_MAX_DOCS), ≤32 000 chars (RAG_MAX_CHARS)
       ├─ last-session context block        (LAST_SESSION_CONTEXT_ENABLED, ≤4 000 chars)
       ├─ _apply_prompt_budget              history first, then lowest-ranked RAG docs (100 000 chars)
       └─ agent mode (RAG_AGENT_MODE=always, kb/tools.py)
            search_kb / read_section / list_sections / list_sessions / find_in_kb
            ≤3 decision turns (thinking OFF), ≤ batch-parallel tools, JSON protocol
  └─ completion → :8888 (Unsloth Studio, Qwen3.8-27B)
```

Index state (live DB, schema v6): **5 903 chunks / 77 files / 5.86 M chars**, model
`qwen3-embedding-4b` @ 2560-dim, chunk size p50 769 / p90 1 979 / max 7 616 chars.
Contextual sidecar (`context_lines.json`): 5 749 keyed lines; **195 indexed chunks
have no context line** (137 in `DnDRules`, 58 in `HumbleWood`). Session notes: 5
session files, 56 chunks, hierarchical `doc_name` (`notes.md [Session: …]`).

## 2. What is solid (keep as-is)

- **Retrieval never touches the chat model except the optional rewrite**, and embed +
  rerank run on dedicated llama.cpp containers (GPU 1) → no model-thrash risk, unlike
  the pre-2026-10-25 design. Failure-open everywhere: every enhancement leg degrades
  to the previous ranking, never to an error.
- **Calibrated attach floor** (0.50 on rerank scores, `RAG_ATTACH_FLOOR_SOURCE=rerank`)
  with a dense-score fallback and an empty-result safety net — the right shape.
- **Cache discipline**: file-hash + content-hash cache validity means unchanged files
  never re-embed; contextual lines live in a sidecar keyed by content hash, so adding
  context never invalidates vectors (the #52 lesson is correctly encoded in code).
- **Hybrid + expansion is coherent**: BM25 scores every query variant (German →
  English expansions exact-match KB terms), RRF fusion is name-keyed, and dedup/rerank
  operate on retrieval names consistently (except the F1 wiring bug).
- **Agent tools are well-bounded** (iterations, tool calls, wall clock, 4 000-char tool
  results, gathered-context re-append on forced answers) and all four tools reuse the
  same index — no second source of truth. `find_in_kb` is a good enumeration answer.
- **Turn telemetry** (`turn_logger`) already records attached files + rerank/dense top
  scores + agent telemetry per turn — an evaluation harness can be built on it (see §5).
- Rerank truncation is **scoring-only** (original chunk text is what gets attached) —
  correct design, the residual issue is the window, not the copy (F4).

## 3. Findings

### F1 — HIGH: near-duplicate dedup is dead in production wiring (proved)

`select_ranked_chunks` dedups a candidate when its embedding (looked up by the
**ranked name**) is within cosine τ=0.92 of an already-selected one. The production
call passes `embeddings_by_name = idx.embedding_map()`, but
(`kb/vector_db.py`) `embedding_map()` is keyed by **`display_name`** while the ranked
tuples — dense, lexical, RRF, rerank — all carry **`retrieval_name()`**, which for any
file in a subfolder is `"<relative path> [<section>]"` ≠ `"<basename> [<section>]"`.

Verified two ways:

- Against the **live index**: retrieval_name ≠ display_name for **5 903 / 5 903**
  chunks — every `.get()` misses, so dedup never fires. There are zero root-level
  files in this KB.
- Hermetic run of the real code path: identical-content nested chunks are both
  attached (prod keys) vs correctly deduped (retrieval-name keys).

Why nobody noticed: unit tests hand-build `embeddings_by_name` with the exact ranked
names (they pass vacuously); probe scripts don't pass `embeddings_by_name` at all; and
the research doc's "dedup notes fired live (Y rows)" benchmark claim could not have come
from this wiring on a nested corpus. Impact today is *modest* (FM2 sibling saturation
is now largely handled by the rerank floor + per-file caps), but the feature is
advertised in `docs/rag.md` and silently does nothing — exactly the class of bug the
§2.5 review fix addressed.

**Fix (one line, single consumer):** in `KBVectorIndex.embedding_map()` key by
`d.retrieval_name()` instead of `d.display_name()` (docstring already says the map
exists *for* selection-time dedup). Add a regression test that builds a store/index
from **nested-source** entries and asserts the dedup note appears via
`idx.embedding_map()` end-to-end, not via a hand-built dict.

### F2 — MED: 195 indexed chunks have no context line; background fill is off

New data ingested since the Oct-23 contextual reindex (137 `DnDRules` + 58
`HumbleWood` chunks) retrieves uncontextualized — for chunk disambiguation this is
exactly what Contextual Retrieval was added for. `CONTEXTUAL_BACKGROUND_FILL` defaults
to **False** and is unset in `.env`, so these stay degraded until someone runs
`scripts/contextual_reindex.py`; the startup worker only warns. Recommendation: set
`CONTEXTUAL_BACKGROUND_FILL=1` in `.env` (it is idle-delayed 300 s, batch-paused,
resumable, and live-reloads per batch by design), or make the startup warning louder
(it already logs, but it is easy to miss). Cheap, zero-risk quality win.

### F3 — MED: with `RAG_AGENT_MODE=always`, the final answer is generated on a NON-thinking decision turn

In `bot_core/agent_loop.py`, the loop ends with a plain-text decision turn (that text
is returned directly) — decision turns run `thinking=False`. Only forced/fallback
answers (iteration or tool-cap reached) get `thinking=True`. So in the usual
`always` flow (1 tool turn + answer), the **answer itself never uses the model's normal
reasoning effort** — even though `AI_REASONING_EFFORT=low` is configured and
`_run_agent_turn`'s docstring promises "final answer with the normal reasoning
settings". This is a quality lever, not a bug per se (it saves ~25 % tokens on answers),
but it should be a *conscious* choice. Options:
1. switch to `RAG_AGENT_MODE=auto` (trigger-gated: enumeration/compound shape or rerank
   score < gate) — answers on plain questions then take the normal thinking path; or
2. keep `always` but make the post-tool answer turn run with thinking ON (small change
   in `run_agent_loop`: after ≥1 tool call, request the final answer explicitly instead
   of accepting a plain-text decision turn).
Worth an A/B on a few campaign rewrite turns before choosing (§5 harness).

### F4 — MED: rerank only sees the first 1 200 chars of 30 % of chunks

`RERANK_MAX_CHARS=1200` truncates scoring input; corpus p90 chunk is 1 979 chars, max
7 616 — **1 764 chunks (30 %) are longer than the scoring window**. Context lines help
(the framing sits in the head), and entry-style sections front-load their answer —
which is why calibration worked — but narrative session logs and long stat blocks can
carry their answer-signal past 1 200 chars and get demoted by a model that never saw
it. Candidate improvement (measure first): for chunks > 1 200 chars score two windows
(head + tail, context line prepended to both) and take the max — doubles request size
only for the 30 % long tail, keeps the existing wire format, and session notes are
exactly the corpus class this bot answers questions about.

### F5 — MED: `reranker-qwen` (vLLM, ~5 GB VRAM) is not profile-gated — `docker compose up -d` still starts it

The llama.cpp reranker (`reranker-llama`) replaced the vLLM Qwen3-Reranker container
on 2026-10-25, but unlike the older TEI service (properly moved to the
`tei-rerank` profile), `reranker-qwen` has **no `profiles:` key** — a fresh
`docker compose up -d --build` starts it on GPU 1 next to `reranker-llama`,
`embedder-llama` and `tts-ollama`, eating ~5 GB of the 16 GB card that is already
committed to the image-generation swap. Add `profiles: ["vllm-rerank"]` to that service
(same pattern as `reranker`). Verify current host state with `docker compose ps` — if
the container was manually stopped it will silently start again on the next full up.

### F6 — LOW: first query after start pays the full index load

`_ensure_index_store` is lazy: the first RAG turn pays SQLite hydration (~135 MB file)
+ cache validation + BM25 build + numpy matrix build inside the reply path (seconds,
and more if new files need embedding). The contextual worker warms the store after
300 s, but anything earlier pays cold. Recommendation: a small startup task that
awaits `_ensure_index_store(KB_PATH)` off the critical path (the pieces exist — just
nothing calls them until the first query).

### F7 — LOW: docs/env drift

- `docs/rag.md` still documents **`/ask_kb`** (command no longer exists) and recommends
  "in-process CPU query embedding" — production runs remote sidecar embedding
  (`EMBED_BACKEND=remote`, `EMBED_URL=http://embedder-llama:8080/v1`) since Phase 2.
  `RAG_CONTEXT_TOKEN_BUDGET` in `.env.example` is a dead variable (settings use
  `PROMPT_BUDGET_CHARS`); `CONTEXTUAL_INDEX_ENABLED` appears only in docs/research and
  was never a settings variable.
- `.env` carries **14 dead `STT_*` variables** plus `SESSION_SUMMARY_PROMPT` and
  `RECORDINGS_DIR` — nothing in the codebase reads them (leftovers from the removed
  voice-transcription line of work). Remove or comment them; dead config is how people
  "tune" things that don't exist.
- The live index predates the §2.3 chunker hard-cap fix: max stored chunk is 7 616
  chars (> `MAX_CHUNK_SIZE=7500`). Harmless (embedder ctx is 8 192 tokens), but the
  next full reindex will normalize it — note it before comparing chunk counts.

### F8 — LOW (verify): worst-case prompt budget vs runtime context

`PROMPT_BUDGET_CHARS=100 000` (default) ≈ 25 K tokens at chars/4. If the Studio runtime
loads the chat model with a smaller context (e.g. 16 K), a maximal RAG attach + history
turn can hit `context_length_exceeded` — handled politely by `bot_core/errors.py`, but
it kills the turn. Not observed in practice (typical prompts are ~10 K tokens); worth
one check of the loaded model's runtime ctx on Studio, and consider capping
`PROMPT_BUDGET_CHARS` to ~0.8 × ctx × 4 if the runtime ctx is < 32 K.

### F9 — INFO: remaining structural ideas (deliberately NOT recommended now)

- **Parent chunks (research doc Tier 2 / small-to-big)**: still unimplemented
  (`parent_chunk` exists nowhere). Correct next big lever if long-chunk precision
  (F4) turns out to matter more than window tricks — but measure F4 first.
- **Canonical-name glossary** (`Column`/`Colum`, `Bees`/`Beast Burrow`, from the #73
  session-quality report): fix at the *vault* layer (a `Glossary.md` file or injected
  canonical names in merge prompts), not in retrieval. Adding alias expansion to BM25
  or embeddings would paper over source corruption.
- **Sentence-window attachment**: `RAG_WINDOW_LINES=80` only affects the keyword
  fallback path now; the vector path attaches whole chunks. Fine as-is; revisit only if
  attached-chunk bloat shows up in answers.
- Do **not** raise `RAG_MAX_DOCS` blindly — the floor + per-file caps are calibrated at
  the current attach budget, and prompt budget pressure (F8) grows with it.

### F10 — INFO: micro-perf notes (no action needed)

- `_llm_expansions` constructs a `QueryRewriter` + a fresh `httpx.AsyncClient` per query
  for the `model_is_loaded` probe (the OpenAI client itself is shared). ~1 ms — fine.
- Rerank batches are sequential (≤32 texts each, ×2 query passes) — correct for a single
  llama.cpp reranker with `--max-num-seqs 32`; parallelizing would just queue server-side.
- `embedding_map()` is rebuilt per retrieval call (~ms for 5.9 K chunks) by design
  (no stale views) — acceptable; it becomes meaningful again once F1 is fixed.

## 4. Suggested action order

| # | Action | Effort | Risk |
|---|--------|--------|------|
| 1 | F1 dedup key fix (`embedding_map` → retrieval_name keys) + regression test | ~10 min | none (single consumer, strictly more correct) |
| 2 | F5 add `profiles: ["vllm-rerank"]` to `reranker-qwen` | 2 min | none (check `docker compose ps` first) |
| 3 | F2 `CONTEXTUAL_BACKGROUND_FILL=1` in `.env` (or run `scripts/contextual_reindex.py`) | 1 min / one run | low (chat-model calls are batch-paused) |
| 4 | F6 warm the index store at startup (background task) | ~20 min | low |
| 5 | F7 docs/env hygiene (rag.md, .env.example, dead STT vars) | ~30 min | none |
| 6 | F3 agent-mode answer-thinking A/B (option 2 first: post-tool answer with thinking ON) | ~30 min + live test | medium — behavioral |
| 7 | F4 long-chunk windowed rerank scoring (head+tail max) behind a probe comparison | ~1 h + measurement | medium — floor calibration constants could shift |
| 8 | F8 confirm Studio runtime ctx vs PROMPT_BUDGET worst case | check only | — |

Items 1–3 are effectively free and each closes a "documented feature that silently
doesn't work" gap — same failure class as §2.5 from the code review.

## 5. Measurement gap (the highest-leverage improvement)

Every improvement above (and everything in the research doc's remaining tiers) needs a
way to answer *"did retrieval get better?"* without manual Discord sampling. The pieces
already exist: `turn_logger` records attached files + rerank/dense top scores + agent
telemetry per turn, and `scripts/rag_probe*.py` replay queries against the live stack.
A **golden-question set** (~30 representative questions from real campaign traffic —
rule lookups, item lookups, German phrasing, compound questions, session recall — each
with the expected file/section) run through `scripts/rag_probe*` before/after each
change would make F1/F3/F4 decisions mechanical instead of vibes-based. Suggested as
its own backlog item before further retrieval tuning.

## 6. Verdict

Architecture: keep. The two-stage design (recall pool → cross-encoder gate → budgeted
attach) is the right one and the failure-open discipline is excellent. The debt is in
**wiring details** (F1), **coverage** (F2), **config hygiene** (F5/F7), and **evaluation
infrastructure** (§5) — not in the architecture. No re-architecture needed or wanted.
