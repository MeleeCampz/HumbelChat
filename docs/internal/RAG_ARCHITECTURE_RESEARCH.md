# RAG Architecture Research & Recommendation (backlog #39 + #41)

**Date:** 2026-10-19
**Status:** RESEARCH / RECOMMENDATION ONLY — no implementation until the user reviews §4 and picks items.
**Companions:** `docs/rag.md` (current pipeline), `RAG_ANALYSIS_2026-09-27.md`,
`RAG_POSTMORTEM_2026-10-15_TQ.md`, `EVAL_QWEN3_EMBEDDING.md`, `BENCHMARK_RAG_2026-10-16.md`.

> **Scope.** Two backlog items tackled together:
> - **#41** — how local-hosted RAG systems are typically built; an overview of typical architectures.
> - **#39** — should the bot use *controlled tool calling* (a bounded search loop) instead of relying on RAG + a single answer turn?
>
> This document is grounded in **our measured failure modes**, not generic best-practice lists. Every recommendation names which failure mode it fixes, its cost under our local constraints (latency / VRAM / local-model reliability), an effort estimate, and how to test it on the existing golden set.

---

## 0. Where we stand today (measured baseline)

### 0.1 Current pipeline (single-turn)

```
query ──► [LLM rewrite/expansion] ─┐        (concurrent, ~0.5–2 s when warm; up to 45 s budget)
       └► dense embed (Qwen3-Emb-4B, CPU) ─► rank top-96
                                            │
      BM25 over every variant (orig + exp.) ┘──► RRF fuse ──► hybrid pool
                                                            │
                              cross-encoder rerank (Qwen3-Reranker-4B, window 150,
                              against original AND first expansion, keep max)
                                                            │
              per-file caps (≤5 chunks/file) + attach floor (dense ≥0.50)
                                                            │
                                   ONE LLM turn: context → system/user prompt → answer
```

All retrieval is in-process or local sidecars; the chat model is a local Qwen3-family 27B GGUF
(`unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS`) via LM Studio. There is **no iteration**: one retrieval
pass, one answer turn, no "not found → search differently" loop, and **no answer-side
verification** (nothing checks that the answer is actually supported by what was attached).

### 0.2 Corpus (measured from `vector_index.db`, 2026-10-19)

| fact | value | why it matters |
|---|---|---|
| chunks | **5,397** | small enough for in-memory dense + BM25 |
| total text | **5.13 M chars ≈ ~1.28 M tokens** | **~10–40× a local chat model's context window → RAG is genuinely required; "stuff the whole KB in the prompt" is not an option** (Anthropic's own rule of thumb: skip RAG only below ~200 K tokens) |
| distinct files | **70** | per-file caps already help |
| two biggest files | Griffon's Saddlebag part 1 & 2 — **~1.0 M chars / ~680 chunks each** | a wall of near-duplicate magic items → the top-k saturation problem is structural, not incidental |
| domain split | `books/` 3.28 M ch · `DnD5_5/` (SRD) 1.57 M ch · campaign dirs (`HumbleWood`, `backgrounds`, `species`, session notes) ≪ 100 K ch each | rules vs lore are cleanly separable; enumeration targets live in small per-item files |

### 0.3 Measured performance (best it has ever been — and still not good enough)

From `BENCHMARK_RAG_2026-10-16.md` (production stack: Qwen3-Emb-4B + Qwen3-Reranker-4B + rewrite):
**38/40 expected-file hits, recall@4 ≈ 95%, zero regressions**, and the canonical TQ torture case
went from "rank 25 in a noise band" to **rank 1 at rerank score 0.94–0.96**.

So the classic cross-lingual / property-lookup failures are *largely fixed*. The user is still
unhappy, which means the residual problems are the **systemic** ones below — the classes that no
single retrieval knob can fix because they are about the *shape of the pipeline* (one turn, no
iteration, no verification) and about *corpus structure* (enumeration, near-duplicates).

### 0.4 Our failure modes (the thing every recommendation must map to)

| ID | Failure mode | Evidence | Persists across configs? |
|---|---|---|---|
| **FM1** | **Enumeration / list queries.** "Which special spells / backgrounds exist in Humblewood?" The answer spans 7 / 3 *small single-item files*; no single chunk enumerates them, so the topically-relevant *aggregate* file (spells.md / World_Overview) wins — correctly by every retriever's metric. | Benchmark Finding 3: **miss in every config since Oct 10** (bge, CPU rerank, TEI, Qwen3). | Yes — structural |
| **FM2** | **Property-specific lookups among near-duplicates.** "Which item has property X?" where the answer is one chunk among dozens of siblings (Cloak/Boots of Elvenkind ×N, Saddlebag items). The right chunk gets crowded out by its own file's siblings. | TQ post-mortem; 0.6B eval "reranker-confusion inside a dense sibling cluster". | Fixed for the *canonical* case via expansion-rerank + wider window, but **fragile** — it depends on the rewriter producing a mechanic-preserving English expansion that lands in time (see FM6). |
| **FM3** | **Single turn, no iteration, no answer-side verification.** If the first retrieval is even slightly off, there is no second chance to search differently, and nothing checks the answer is grounded in what was attached. This is the core of **#39**. | Architectural — `_retrieve_vector` → one `ask_ai`. No loop, no critic. | Yes — architectural |
| **FM4** | **Cross-lingual German query / English KB residual.** Mostly bridged by English expansion + expansion-aware rerank; a fuzzy match (animals/Vulpin) still scores only ~0.50. | Benchmark case 16/26. | Largely fixed, small residual |
| **FM5** | **Negative / unanswerable queries drag junk.** "Capital of France?" attaches **9.4 K chars**, "Steuererklärung in Deutschland?" attaches **13.7 K chars** of on-topic-but-wrong D&D content (rerank tops only 0.26–0.35). The model then either answers from junk or says "not in context" — but it *wasted* the budget and risks a confident wrong answer. | Benchmark Finding 4 (backlog #43). | Yes — no relevance gate before attach/answer |
| **FM6** | **Rewrite latency stalls.** In the production run, **17/42 (40%) of rewrites exhausted the 45 s budget** and were dropped → those queries paid ~50–57 s end-to-end *and* lost their expansions. Root cause: the rewrite uses the **big chat model**, which intermittently leaks into thinking mode or stalls on LM Studio. | Benchmark Finding 2 (top action item). | Yes — operational, but it also *removes the safety net* that fixes FM2. |

**Key insight.** FM1 and FM5 are *retrieval-structure* problems fixable without an agent.
FM3 is the *architectural* problem that tool calling (#39) addresses. FM6 is an operational bug
that quietly degrades FM2's fix. A good plan attacks all of these in priority order, and the
tool-calling work should only start **after** the golden set has grown to actually measure it.

---

## 1. How local RAG systems are typically built (#41)

A useful way to think about "typical" RAG is as a ladder of stages. Each stage adds capability
and cost. Below, "where we are" marks our position.

| Stage | What it adds | Typical cost (latency / VRAM / model) | Where we are |
|---|---|---|---|
| **S0 — Naive RAG** | chunk → embed → cosine top-k → stuff in prompt | trivial; one embedding call | superseded |
| **S1 — Hybrid retrieval** | + BM25 exact-term leg, fused (RRF) so names/stat-blocks/proper-nouns aren't under-weighted by dense vectors | negligible (BM25 is CPU, already cached) | ✅ here |
| **S2 — Reranking** | + cross-encoder re-scores a top-N window → clean score separation between "the answer" and "everything else" | one extra model pass over ~150 chunks; we pay for it (Qwen3-Reranker-4B, ~4–7 s) | ✅ here |
| **S3 — Query transformation** | + rewrite / expansion / HyDE / decomposition / step-back so the *query* matches the KB's vocabulary & language | one small LLM call per query (we already pay it; see FM6) | 🟡 partial (rewrite+expansion only) |
| **S4 — Index-time enrichment** | + contextual retrieval, late chunking, small-to-big parents, dedup/condensation, collection routing — make the *indexed units* easier to retrieve | mostly **one-time offline** LLM/compute; near-zero runtime cost | ❌ not here (biggest untapped wins) |
| **S5 — Self-correcting / agentic generation** | + relevance gating (CRAG/Self-RAG), retrieve-on-demand, iterative retrieval (FLARE), tool-call loops (ReAct/RAG-as-tools) | multiple LLM turns; the only stage that adds *per-query* latency beyond S2–S3 | ❌ not here — this is **#39** |

### 1.1 What "local" specifically changes

- **No per-request model swap.** Our whole design (`EMBED_BACKEND=local`) exists so query
  embedding runs in-process on CPU and the chat model never has to be evicted. Any new stage must
  respect this: it should be (a) in-process/CPU, (b) a *persistent* sidecar that stays resident, or
  (c) an **offline** batch job. "Call a second big model at query time" is the anti-pattern —
  FM6 is exactly what happens when we violate it (the rewrite uses the chat model and stalls).
- **VRAM is shared with D&D.** The user plays while the bot runs. New *runtime* components should
  be tiny (≤ a few GB) or CPU. New *offline* components can use the GPU briefly (chat model
  unloaded), because they run once, not per query.
- **The chat model is a quantized local Qwen3.** Assume function calling *works* but is less
  reliable than frontier APIs → any agentic design must degrade gracefully to single-turn (§3).

### 1.2 The two archetypes at the top of the ladder

- **Self-correcting RAG (CRAG / Self-RAG / FLARE)** — the model *evaluates* its retrieval and
  decides to answer, re-retrieve, or admit "not found." Adds a cheap relevance gate + optional
  second retrieval. Low latency overhead if gated.
- **Agentic RAG (ReAct / RAG-as-tools)** — retrieval is exposed as *tools* the model calls in a
  bounded loop. Highest ceiling (handles multi-hop, enumeration-by-synthesis, "search again"),
  highest cost and reliability risk on a local model. This is #39.

**The short version for #41:** our stack already implements S0–S2 solidly and part of S3. The
high-value, low-risk headroom is in **S4 (index-time enrichment)** — which Anthropic measured as
the single biggest retrieval win — and a *lightweight* slice of S5 (a relevance gate + "say not
found" honesty). Full agentic tool calling (S5) is the bigger bet that pays off only once we can
measure it.

---

## 2. Technique catalogue, mapped to our failure modes

Format per technique: **what / who uses it / fit for this bot (latency·VRAM·local-model) / effort /
fixes which FM / golden-set testable?**

### 2.1 Query side (S3)

| Technique | What it is | Fit here | Effort | Fixes | Testable? |
|---|---|---|---|---|---|
| **Query decomposition** (sub-questioning) | LLM splits a multi-part / "which X have property Y" question into sub-queries, retrieves per sub-query, merges. LlamaIndex `SubQuestionQueryEngine`, LangGraph. | Good for FM1 + multi-part questions. One extra small-LLM call; can run in the existing rewrite budget slot. The decomposition output doubles as a *keyword list* for BM25. | Low–Med | FM1 (partly), FM3 (enables it) | Yes — add multi-part + enumeration queries |
| **HyDE** (hypothetical document embeddings) | Generate a fake "answer document," embed *that*, retrieve real neighbors. Gao et al. 2022 (arXiv:2212.10496). | **Low marginal value for us.** Our LLM expansion already produces English query variants that are a cheaper, safer cousin of HyDE. Anthropic explicitly evaluated HyDE and found contextual retrieval better (§2.2). A generated fake doc can inject false specifics that mislead dense search. | Low (but not worth it) | FM4 (weakly) | Yes, but expected to underperform expansion |
| **Step-back prompting** | LLM first asks "what's the general concept here?" and retrieves on the abstraction, then answers the specific. Zheng et al. 2023 (arXiv:2310.06117). | Fits FM1 well: "which backgrounds exist in Humblewood?" → step back to "Humblewood campaign setting / character backgrounds" → retrieve the *category*, then list members. Cheap (one short call, thinking off). | Low | FM1 | Yes — enumeration queries |
| **Multi-query / LLM keyword lists for BM25** | LLM emits explicit keywords/entities (English) to feed the lexical leg. | We already expand→BM25 (F6 in the 09-27 analysis). The refinement is *structured* entity extraction (item names, stat-block terms) rather than free-form expansions — makes BM25 exact-matches more reliable for FM2. | Low | FM2, FM4 | Yes |
| **Cross-lingual: translate vs bilingual embeddings** | Either translate the query to English before retrieval, or use a multilingual embedder. | We do *both* already: Qwen3-Embedding-4B is multilingual (119 languages incl. German) **and** we generate an English expansion that rescues the cross-lingual rerank gap (TQ rank 25→1). Recommendation: **keep this**; it's working. The only change is making the bridge *reliable* (FM6), not replacing it. | — (maintain) | FM4 | Yes |

**Query-side verdict:** we already do the high-value half of S3. The net-new, cheap wins are
**decomposition + step-back** aimed specifically at **FM1 (enumeration)** — both fit in the existing
rewrite slot with thinking off. HyDE is not worth adding on top of expansion.

### 2.2 Retrieval side (S4) — the biggest untapped headroom

| Technique | What it is | Fit here | Effort | Fixes | Testable? |
|---|---|---|---|---|---|
| **Contextual Retrieval** (Anthropic, Sep 2024) | Prepend a short LLM-written "this chunk is from X about Y" to each chunk *before* embedding + BM25. Measured: −35% top-20 retrieval failure (embeddings), **−49%** (+Contextual BM25), **−67%** (+rerank). One-time cost ~$1.02 / M doc tokens with prompt caching. | **High fit.** Our worst-retrieved content is *decontextualized table rows* — exactly Anthropic's "the company's revenue grew 3%" failure: an armor row, a spell line, or a Saddlebag item reads as gibberish without its section header. We already convert HTML tables to `Columns: …` + `Name | v1 | v2` (F1), but the *row itself* still lacks "this is the Plate Armor row in the equipment table." Contextualizing would also directly help **FM1** (each background/spell item chunk would say "one of Humblewood's campaign backgrounds"). **Cost is offline**: one batch LLM pass over 5,397 chunks (~1.28 M tokens) — run once with the chat model unloaded on GPU, or a small local model; near-zero runtime cost. | Med (one-time job + reindex) | FM1, FM2, general recall | Yes — full golden set before/after reindex |
| **Contextual Compression** (Anthropic companion) | At query time, LLM extracts only the sentences in a retrieved chunk relevant to the query → smaller, denser context. | Moderate fit; helps FM5 (less junk reaches the prompt) but adds a per-query LLM call over top-k chunks. Defer — the attach floor (§2.3 / #43) is a cheaper first cut. | Med | FM5 | Yes (negative set) |
| **Late chunking** (Jina, 2024) | Embed the *whole* document with a long-context embedder, then chunk the resulting passage vector → each small chunk's embedding carries full-document context. | **Limited fit.** Requires a long-context embedder; our Qwen3-Emb-4B is an 8 K-window model and query-side runs in-process on CPU (swapping to a big long-context embedder per query breaks the no-swap design). Conceptually overlaps with contextual retrieval, which we can do offline. **Defer.** | High | FM2 (weakly) | — |
| **Small-to-big / parent-document retrieval** | Retrieve on small precise chunks, but **return the larger parent section/document** to the LLM for context. LlamaIndex node relationships, Weaviate. | **Good fit for D&D.** A stat-block row or item line is the precise retrieval target; the *whole item entry* (name + rarity + all properties) or *whole table section* is what the answer needs. This directly attacks FM2 (return the full Saddlebag item, not a fragment that lost its property line) and helps answers be complete. We already have heading-path chunk labels (`file [Section > Subsection]`), so parents are recoverable from existing structure — mostly a selection-time change. | Low–Med | FM2, answer completeness | Yes — TQ + property queries |
| **Dedup / condensation of near-duplicates** | Cluster near-identical chunks (embedding cosine) and attach one representative (+ maybe a "×N variants" note) instead of N siblings. | **Directly targets FM2.** The Saddlebag parts are ~1.36 M chars of sibling items that saturate top-k. A selection-time dedup (cosine > τ within the rerank window → keep top, drop rest, or merge) is cheap and needs no reindex. Complements small-to-big. | Low–Med | FM2 | Yes — TQ + stealth-item queries |
| **Collection routing / per-domain indexes** | Route the query to a subset of the corpus (rules vs campaign lore vs session notes) before retrieval, or maintain separate indexes with per-collection weights. | **Good fit.** Our corpus splits cleanly (§0.2): `books/`+`DnD5_5/` = rules; small campaign dirs = lore. Routing fixes two things: (a) German session-note BM25 crowding out rulebook answers (documented in 09-27 analysis), and (b) **FM1** — "backgrounds in Humblewood" should search the *campaign* collection, not the whole SRD. Can start as a lightweight path-prefix filter + per-collection RRF weighting, no new infra. | Med | FM1, FM4-crowding | Yes — enumeration + German rule queries |
| **GraphRAG** (Microsoft) | LLM extracts entities/relationships/communities into a graph; local search for specific facts, global (map-reduce) search for corpus-level "sensemaking." `github.com/microsoft/graphrag`. | **Recommend REJECT.** Three reasons: (1) **indexing is LLM-expensive** — the README itself warns it "can be an expensive operation" and needs a strong model to extract entities/relationships over our 5.4 K chunks; that's a large offline compute + quality gamble on a local model. (2) The project is **in maintenance mode** (no new features). (3) It's designed for *corpus-level* questions ("what are the main themes across this corpus?"), whereas our failures are *precise lookups and enumeration within a small, already-structured corpus*. Our 70-file corpus with clean heading structure is exactly the case where hybrid retrieval + contextual enrichment + (later) a bounded tool loop get most of GraphRAG's benefit at a fraction of the cost. | — (reject) | — | n/a |

**Retrieval-side verdict:** the standout is **Contextual Retrieval** (offline, one-time, measured
−49%/−67%) because our corpus is full of decontextualized table rows. Pair it with cheap
selection-time **dedup + small-to-big** for FM2, and a lightweight **collection router** for FM1.
Skip late chunking (embedder constraint) and GraphRAG (cost/maintenance/fit).

### 2.3 Generation side (S5-lite)

| Technique | What it is | Fit here | Effort | Fixes | Testable? |
|---|---|---|---|---|---|
| **Relevance gating / "not in context" honesty** (CRAG's evaluator, prompt-level) | Before/while answering, decide whether the attached context actually answers the query; if not, say so explicitly instead of answering from junk. CRAG (Yan et al. 2024, arXiv:2401.15884) uses a lightweight retrieval evaluator that returns a confidence and triggers different actions. | **High fit, low cost.** We already have the signal: Qwen3 rerank scores separate "real answer" (≥0.50) from "junk" (0.26–0.35) cleanly (Benchmark Finding 1). A prompt instruction + a score threshold tells the model "if nothing here is clearly relevant, say 'I couldn't find that in the rules/campaign'." Directly fixes **FM5** and prevents confident-wrong answers on negatives. No new model. | Low | FM5, FM3 (honesty) | Yes — negative set + hit set (no regressions) |
| **Answer with citations** | Require the answer to cite which attached section each claim comes from; uncited claims are flagged. RAGAS `faithfulness` measures exactly this. | Good fit — cheap prompt change, improves trust and makes FM3 failures *visible* (we can see when the model reaches beyond context). Pairs with the golden-set growth. | Low | FM3 (visibility) | Yes — faithfulness on answer set |
| **Self-RAG / reflection tokens** (Asai et al. 2023, arXiv:2310.11511) | Train a model to emit special "should I retrieve? is this relevant? is this supported?" tokens and adapt retrieval on-demand. | **Not feasible locally as specified** — it needs *fine-tuning* a model with reflection-token training data. But its *idea* (retrieve-on-demand + self-critique) is what the CRAG gate + tool loop give us without training. Adopt the idea, not the method. | High (reject as-specified) | FM3 | — |
| **Iterative retrieval** (FLARE, Jiang et al. 2023, arXiv:2305.06983) | After generating part of an answer, detect low-confidence spans and *retrieve again* to fill gaps. | This is the conceptual bridge to #39. A full FLARE loop (per-sentence confidence) is heavy; a **coarse version** — "if first retrieval's top rerank score < threshold, do one more targeted retrieval before answering" — is cheap and is effectively the *non-tool* half of #39. | Med | FM2, FM3 | Yes — property queries |

### 2.4 Agentic / tool calling (#39) — S5-full

See §3 for the concrete design. Summary of the landscape:

- **ReAct** (Yao et al. 2022, arXiv:2210.03629): interleave reasoning traces + tool actions;
  reasoning helps the model plan and recover from a bad first action. The canonical pattern for
  "search → look at result → decide next step."
- **RAG-as-tools:** expose `search_kb`, `read_chunk`/`open_section`, (optionally) `list_sections`.
  The model retrieves on demand, reads the promising hit in full, and can re-search with a better
  query — which is precisely the "not found → search differently" loop we lack today.
- **Bounded loops:** cap iterations (2–3) and total tool calls; each iteration is one local
  completion (~2–8 s) + one retrieval (~1–4 s). A 2-iteration ceiling keeps worst case ~20–30 s.
- **When single-turn beats an agent:** for the *common* confident lookup (the majority of our
  golden set), single-turn + good retrieval is faster and just as good — an agent would only add
  latency. The agent earns its keep on the *hard tail*: FM1 enumeration, FM2 property-in-a-sea-of-
  -siblings, and multi-hop questions. Hence: **agent mode should be triggered, not default** (§3.4).
- **Reliability on our local model:** Qwen3 is *explicitly trained for tool calling* — the Qwen3
  release notes state "Qwen3 excels in tool calling capabilities," add agentic RL over format-
  following + agent tasks, and recommend Qwen-Agent (which encapsulates tool-call templates/parsers).
  That's a strong prior that function calling will *work*. The local caveats: (a) use **non-thinking
  mode** for the tool-decision turns (fast, less likely to leak into reasoning — the same bug class
  as FM6); (b) validate tool-call JSON strictly and fall back on any parse failure; (c) hard-cap
  iterations so a confused loop can't run away. **Design must degrade gracefully** to single-turn.
- **Open-source patterns:** LlamaIndex agent workflows / `SubQuestionQueryEngine`, LangGraph agent
  recipes, Qwen-Agent — all provide bounded ReAct-style loops with tool schemas we can mirror.

---

## 3. Controlled tool calling design (#39) — concrete proposal

> This is a *proposal for review*, not an implementation plan of record. It answers the handoff's
> required questions: which tools, loop shape, iteration caps, coexistence with single-turn,
> reliability on our local model, and eval plan.

### 3.1 Goals & hard constraints

- Give the model a way to **search again** (with a better query) and **read a hit in full** when
  the first retrieval is insufficient — closing FM3.
- Keep the **common case fast**: confident lookups should still be single-turn.
- Never exceed a bounded wall-clock; **always** fall back to single-turn with best-so-far context.
- No new resident VRAM (tools are in-process Python over the existing index + reranker).

### 3.2 Tool set (minimal, high-leverage)

| Tool | Args | Returns | Why |
|---|---|---|---|
| `search_kb` | `query: str`, `scope?: "rules"\|"lore"\|"all"` | top-k hits as `[ {file, section_path, snippet(≈300 ch), rerank_score} ]` | The retrieval leg; `scope` gives the model collection routing (§2.2) for free |
| `read_section` | `file: str`, `section?: str` (or `chunk_id`) | the **full** section/chunk text (small-to-big, §2.2) | Lets the model pull a complete stat block / item entry instead of a 300-char snippet — fixes FM2's "property line got cut" class |
| `list_sections` | `file: str` or `dir: str` | heading-path outline (we already store these labels) | Enables **enumeration** (FM1): "list the backgrounds in Humblewood" → model lists the dir, then reads each; no aggregate chunk required |

Optional later: `check_answer(claims)` — a cheap critic that returns which claims are supported by
the attached text (RAGAS-faithfulness at runtime). Keep out of v1.

### 3.3 Loop shape

```
system prompt (persona + "you have tools; search before answering; if nothing is relevant, say so")
  + history block
  + current user question
        │
   ┌───▼──────── tool-calling loop (max N=2 iterations, max C=4 tool calls total) ───┐
   │ model emits:  [tool_call search_kb(...)]                                          │
   │ we run it → append tool result as a `tool` message                                │
   │ model emits:  [tool_call read_section(...)] or [final answer with citations]      │
   │ … until the model stops calling tools (answers) or a cap/timeout is hit           │
   └──────────────────────────────────────────────────────────────────────────────────┘
        │
   if any tool-call parse error / iteration cap / wall-clock budget exceeded:
        → FALL BACK to single-turn using the best context gathered so far
          (all search snippets + any read sections), with the honesty instruction.
```

- **Non-thinking mode** (`enable_thinking=False`) on every *tool-decision* turn — fast and avoids
  the FM6 reasoning-leak. The final *answer* turn may keep the user's normal reasoning setting.
- **Wall-clock budget** for the whole loop (e.g. ≤ ~30 s of tool turns) independent of the
  per-read `AI_TIMEOUT_S` idle watchdog, so a stuck backend fails the loop and triggers fallback.
- Tool results are appended as OpenAI-style `tool`/`role:"tool"` messages; the streaming path
  (`ask_ai_stream`) must be extended to interleave — the current stream is single-shot content-only.

### 3.4 Coexistence with the single-turn path (trigger, don't default)

Two modes sharing one retrieval core:

1. **Fast mode (default):** today's pipeline verbatim. Used when the first retrieval is
   *confident* — e.g. top rerank score ≥ ~0.85 and the query isn't an enumeration/multi-part shape.
2. **Agent mode:** triggered when a cheap classifier says "hard." Triggers (any of):
   - top rerank score below a threshold (the CRAG gate — retrieval itself is unsure), **or**
   - query shape = enumeration / multi-part (matched by the decomposition/step-back pass, §2.1) or an explicit `/ai deep` command, **or**
   - FM6-safe: rewrite produced no usable expansion and confidence is low.

This keeps the 30+ confident golden-set queries on the fast path (no latency regression) and spends
the extra turns only where single-turn demonstrably fails (FM1/FM2). A global `RAG_AGENT_MODE=off|auto|always`
env knob controls it; `auto` is the intended default.

### 3.5 Reliability on our local Qwen3 model — assessment

- **Positive prior:** Qwen3 is trained for function calling + agentic use (release notes, §2.4).
  Tool-call format is supported by llama.cpp/LM Studio/Qwen-Agent. IQ4_XS quant at 27B is large
  enough that schema-following should be reliable on *simple* tools with short schemas.
- **Risks & mitigations:** (a) JSON parse failures → strict validation + one retry + fallback;
  (b) the model "forgets" to call a tool and answers from nothing → system prompt mandates search
  first, and we detect zero-tool-call + low-confidence → force one `search_kb`; (c) runaway loops
  → hard caps on iterations/calls/wall-clock; (d) thinking leakage → non-thinking on decision turns.
- **Must verify empirically before trusting it:** a tool-calling reliability probe (see §5) — e.g.
  30 scripted scenarios, measure % that emit valid tool calls, % that terminate with an answer,
  median iterations, fallback rate. This is a prerequisite gate for shipping agent mode.

### 3.6 Token / latency budget per activation

- Each tool turn ≈ one local completion (2–8 s warm) + one retrieval (1–4 s incl. rerank).
- N=2 iterations, C≤4 calls → worst realistic ~4 turns ≈ **15–30 s** of tool activity, then the
  final answer. Acceptable for the hard tail (users are mid-session; a "thinking" indicator is
  already shown), but **not** acceptable as the default — hence §3.4's trigger design.

---

## 4. Prioritized improvement plan (quick wins → bigger bets)

> Every item: what / fixes / effort / how it's measured on the golden set / risk.
> Ordered so each tier is shippable and testable before the next. **Nothing here is implemented yet.**

### Tier 0 — trivial, config/prompt only (do first; low risk)

1. **Stop the rewrite stalls (FM6).** Two independent levers:
   - Drop `RAG_REWRITE_BUDGET_SECONDS` from 45 → **8–10 s** so a stalled rewrite fails fast
     (a 45 s hang is *worse* than no expansion — Benchmark Finding 2).
   - Structurally, stop using the big chat model for the utility call: run expansions with a
     **small dedicated local model** (or the embedding-instruct path) so it can't leak into
     thinking mode or stall on LM Studio. This also *protects the FM2 fix*, which currently
     depends on an expansion landing in time.
   - *Measure:* re-run `rag_benchmark.py`; stalled-rewrite count should drop from 17/42 → ~0;
     median latency down; hit rate unchanged (it was already unaffected).
2. **Answer-honesty + citations prompt (FM3/FM5).** Add to the answer system prompt: cite the
   attached section for each claim; if nothing attached is clearly relevant, say "I couldn't find
   that in the rules/campaign" instead of guessing. *Measure:* negative set attaches junk but the
   *answer* now says "not found"; hit set shows no regressions and gains visible citations.

### Tier 1 — quick wins, minimal infra (retrieval-structure fixes)

3. **Enable the rerank-score attach floor (#43) for negatives (FM5).** Set
   `RAG_ATTACH_FLOOR_SOURCE=rerank` with a Qwen3-calibrated floor ~0.50 (Benchmark Finding 4 shows
   this attaches *nothing* to both negatives and changes nothing on the 38 hits). *Measure:*
   negative set → ~0 junk chars attached; hit set unchanged.
4. **Selection-time dedup of near-duplicates (FM2).** In `select_ranked_chunks`, cluster the
   rerank window by embedding cosine (>τ) and attach one representative per cluster (+ optional
   "×N similar items" note). No reindex. *Measure:* TQ + stealth-item queries; sibling saturation
   drops; hit rate up or steady.
5. **Small-to-big: return the parent section for precise hits (FM2 / completeness).** When a
   stat-block row / item line is the top hit, attach its full section (heading-path parents are
   already stored). *Measure:* property queries return complete entries; TQ stable at rank 1.
6. **Enumeration handling (FM1) — the persistent miss.** Two complementary options:
   - **(a) KB-side index file:** generate a small "Humblewood special spells / backgrounds" listing
     file that ranks first for such queries (cheapest, most reliable).
   - **(b) Directory-aware retrieval:** when the query is an enumeration ("which X in Y?"), list
     the matching directory's files and attach their headers (this is what `list_sections` will do
     in agent mode — §3.2). *Measure:* add both Humblewood enumeration queries to the golden set;
     they should flip from MISS → HIT.

### Tier 2 — moderate effort, higher value (index-time + gated generation)

7. **Contextual Retrieval pass over the KB (§2.2).** One-time offline LLM job: prepend a
   chunk-specific context line to all 5,397 chunks, reindex. Measured −49% retrieval failure
   (−67% with rerank) in Anthropic's study; our decontextualized table rows are the ideal target.
   Run once with the chat model unloaded on GPU (or a small local model). *Measure:* full golden
   set before/after reindex — expect broad recall gains, especially FM1/FM2 and German lookups.
8. **Query decomposition + step-back for hard queries (§2.1).** Extend the rewrite slot to emit
   sub-questions / a step-back abstraction for enumeration & multi-part shapes; feed sub-queries
   through retrieval and RRF-merge. *Measure:* multi-part + enumeration golden queries.
9. **CRAG-style relevance gate before answering (FM5/FM3).** Use the existing rerank score to
   decide answer / (one) re-retrieve / say-not-found — the non-tool half of #39. Cheap, and it's a
   *prerequisite* for trusting agent mode's trigger (§3.4). *Measure:* negative set + low-confidence hits.

### Tier 3 — bigger bets (the structural fix)

10. **Controlled tool calling (#39), per §3.** Build the three tools, the bounded loop, the
    fast/agent trigger split, and the graceful fallback. **Gate it behind:** (a) the golden set
    grown with multi-hop + enumeration + "needs-2nd-retrieval" queries (§5), and (b) a green
    tool-calling reliability probe (§3.5). This is where FM3 is truly closed. *Measure:* new
    agentic metrics — success rate on the hard-tail subset, median iterations, fallback rate,
    latency p50/p95, faithfulness.
11. **Collection routing (rules / lore / session) (§2.2).** Lightweight path-prefix filter +
    per-collection RRF weighting; also fixes German-note crowding of rulebook answers. *Measure:*
    enumeration + German rule queries.

### Revisit / defer / reject

- **GraphRAG — reject** (cost, maintenance mode, wrong fit for a small structured corpus; §2.2).
- **HyDE — defer/reject** (we already have expansion, its safer cousin; Anthropic found it weaker
  than contextual retrieval).
- **Late chunking — defer** (needs a long-context embedder; conflicts with the in-process CPU
  no-swap design).
- **Contextual Compression — defer** behind #3/#9 (attach floor + CRAG gate are cheaper first cuts
  to the same FM5 problem).

---

## 5. Evaluation plan (how every recommendation gets measured)

### 5.1 Keep and GROW the golden set

- `scripts/rag_eval.py` (33-query embedded set) is the standing harness; `rag_benchmark.py` runs
  the fuller 42-case production-stack set. **Grow both with real production failures:** add a
  "failure log → golden query" loop — every bad Discord answer gets logged and its query added as
  a case (query + acceptable file stems). The two Humblewood enumeration queries and any new
  multi-hop cases belong here *before* Tier 3.

### 5.2 Split retrieval quality from answer quality (RAGAS-style)

Today we only measure **retrieval** (hit rate, recall@k). Add the **answer** side so FM3/FM5 are
actually scored:

| Axis | Metric | Source | Measures |
|---|---|---|---|
| Retrieval | hit rate, recall@1/2/4 | existing | right file attached (FM1/FM2) |
| Retrieval | **context precision / context recall** | RAGAS (precision@k over the attached set; degree relevant info is captured) | ranking quality, junk ratio (FM5) |
| Answer | **faithfulness** | RAGAS — fraction of answer claims supported by the *attached* context (0–1) | hallucination / reaching-beyond-context (FM3) |
| Answer | **"not found" correctness** | % of negative queries answered with an honest "not in the KB" vs a confident guess | honesty (FM5) |

RAGAS metrics are LLM-judged; run them with our local model (thinking off, batched) as an offline
eval pass — not on the hot path.

### 5.3 Measuring the tool-calling loop (#39) specifically

Add per-query agentic telemetry to the benchmark:

- **iterations used** (distribution) and **tool calls made** (which tools, how often);
- **success rate** on the hard-tail subset (FM1/FM2/multi-hop) vs the fast-mode baseline;
- **fallback rate** (% of agent activations that hit a cap/parse-error and fell back to single-turn);
- **latency p50/p95** for agent vs fast mode (must confirm the common case stays on fast mode);
- **faithfulness** delta (does the loop actually ground answers better?).

A standalone **tool-calling reliability probe** (§3.5) — ~30 scripted scenarios measuring valid-call
%, termination %, median iterations — is a **prerequisite gate** before agent mode ships.

### 5.4 Guardrails for every change

- Any retrieval change: re-run the full golden set; require **no regression on the 38 hits** (the
  bar our last two swaps met).
- Keep the negative set in every run — it's the only thing that catches FM5 regressions.
- Log attached-chunk names (already added, 10-15 post-mortem) so the next failure is one grep away.

---

## 6. Sources

**Papers (arXiv):**
- Self-RAG — Asai et al. 2023, arXiv:2310.11511 (retrieve-on-demand + reflection tokens).
- Corrective RAG (CRAG) — Yan et al. 2024, arXiv:2401.15884 (lightweight retrieval evaluator → confidence-gated actions; decompose-then-recompose).
- FLARE — Jiang et al. 2023, arXiv:2305.06983 (active/iterative retrieval on low-confidence spans).
- HyDE — Gao et al. 2022, arXiv:2212.10496 (hypothetical document embeddings).
- Step-Back Prompting — Zheng et al. 2023, arXiv:2310.06117 (abstract to high-level concept first).
- ReAct — Yao et al. 2022, arXiv:2210.03629 (interleaved reasoning + tool actions).

**Engineering / product:**
- Anthropic, "Contextual Retrieval," Sep 2024 — anthropic.com/engineering/contextual-retrieval
  (contextual embeddings −35% top-20 failure; +Contextual BM25 −49%; +rerank −67%; ~$1.02/M tokens;
  explicitly evaluated and found HyDE & summary-indexing weaker; "skip RAG below ~200 K tokens").
- Microsoft GraphRAG — github.com/microsoft/graphrag (LLM-expensive indexing; local vs global search;
  currently in maintenance mode).
- Jina AI, "Late Chunking for Fine-grained Text Embedding," 2024 — jina.ai/news/late-chunking-for-fine-grained-text-embedding/.
- Qwen3 release notes (qwenlm.github.io/blog/qwen3) — hybrid thinking/non-thinking modes; "Qwen3 excels
  in tool calling capabilities"; agentic RL over format-following + agent tasks; Qwen-Agent.
- RAGAS metrics — docs.ragas.io (faithfulness = claims grounded in context; context precision = mean
  precision@k over the retrieved set; context recall).

**Internal (measured baselines & rejected options):** `docs/rag.md`,
`docs/internal/RAG_ANALYSIS_2026-09-27.md`, `RAG_POSTMORTEM_2026-10-15_TQ.md`,
`EVAL_QWEN3_EMBEDDING.md`, `BENCHMARK_RAG_2026-10-16.md`; corpus stats measured directly from
`data/knowledge/.vector_index_cache/vector_index.db` on 2026-10-19.
