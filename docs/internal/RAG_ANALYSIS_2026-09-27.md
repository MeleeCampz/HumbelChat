# RAG Deep-Dive — German Query Retrieval (2026-09-27)

Triggered by backlog item #15 follow-up: in Discord, *"Gib mir die Stat block vons
verschiedneen Rüstungen"* got no armor stats back. This document protocols the full
analysis pass over the RAG system, the root causes found, the fixes implemented, and
before/after evidence.

## 1. Protocol (work log)

| Step | What was done | Result |
|---|---|---|
| 1 | Read `logs/bot.log` around the failing query (17:27) | RAG *did* attach 4 files incl. `equipment.md` (~9.7K chars) — but the bot still couldn't answer. So retrieval ran; the **right chunk** was missing. |
| 2 | Read all of `kb/` (retrievers, query_rewriter, chunker, lexical, embedder, index, vector_db) + `.env` RAG settings | Mapped the full pipeline: dense (bge-m3) + BM25 → RRF → per-file selection (`RAG_MAX_DOCS=4`, `RAG_MAX_CHUNKS_PER_FILE=3`). |
| 3 | Built `scripts/rag_probe.py` — runs 33 German + English game queries through the **real production pipeline** (read-only) inside the live container | 26/33 attached an expected file. Armor queries ranked `[Preamble]` above `[Armor]`; BM25 leg dominated by German session notes. Report: `data/rag_probe_results.json`. |
| 4 | `scripts/rag_probe2_embed_compare.py` — compared full-precision local bge-m3 scores vs the GGUF-built index for identical (query, chunk) pairs | Scores **identical** (0.568 vs 0.569 …). The `INDEX_EMBED_MODEL differs` startup warning is a false alarm — quantized GGUF vectors are effectively interchangeable. Index is NOT the problem. |
| 5 | Inspected `equipment.md` structure | SRD files are HTML-table based: the `[Armor]` chunk is one ~3.7K-char `<table>` with 15 armor types; **all** SRD files use tables (monsters-A-Z alone: 235 tables / 705 rows). Tag markup dilutes embeddings and hides item names from BM25. |
| 6 | `scripts/rag_probe3_table_test.py` — embedded original vs plain-text Armor chunk | Plain text helps English queries only marginally; German queries still ~0.41–0.52 with `[Preamble]` competitive. → Not the main lever. |
| 7 | `scripts/rag_probe4_variants.py` — tested per-row chunks, plain-text table, and **simulated LLM-rewritten (English) queries** | Decisive: German raw ≈ 0.41–0.52 everywhere; English query → Armor 0.56+; **rewritten English expansion → 0.66–0.70** on armor chunks while `[Preamble]` drops to ~0.45. |
| 8 | Diagnosed why the rewriter never helped: `RAG_REWRITE_MIN_SCORE=0.35` but real top-scores sit at 0.49–0.67 (even good matches), so the trigger is dead; and the prompt never told the LLM the KB is English / to translate | Root cause #2 confirmed. |
| 9 | Implemented fixes (below) + unit tests (`tests/test_rag_german_retrieval.py`, 17 passed) | — |
| 10 | Deployed: image rebuild → index force-rebuild → bot restart → re-ran probe with the live rewriter enabled (`scripts/rag_probe_after2.py`, calls the real `retrieve_kb_documents()` entry point) | Rewriter now fires for every German query — but see step 11. |
| 11 | Traced rewrite results in `dev.log` + container stdout | **Second bug found:** every rewrite call failed with *"The model used its full token budget on reasoning and produced no answer"* — Qwen3.8 (hybrid thinking model) spends the entire `max_tokens=256` budget on reasoning, leaving nothing for content. The rewriter had been silently dead even when triggered. |
| 12 | Live A/B test against the backend: `max_tokens=1024` → 5.7 s of pure thinking; `extra_body={"enable_thinking": False}` → **0.5 s** with perfect output ("Armor / Healing Potion / Fireball Spell") | Fixed in `kb/query_rewriter.py` (F5): disable thinking for this short utility call, with a plain-retry fallback for backends that reject the param. |
| 13 | Re-ran probe: equipment.md attached again — but now **BM25 on the raw German query** flooded the fusion with monster "stat block" chunks and pushed `[Armor]` (fused rank #4) out of the attachments | Fixed in `kb/retrievers.py` (F6): the lexical leg now scores **every query variant** (original + expansions), so BM25 exact-matches translated terms like "Chain Mail" instead of only German tokens. |
| 14 | Final probe run (`data/rag_probe_results_after.json`) | **29/33** (baseline 26/33); both originally failing armor queries now attach `equipment.md`. See §5. |

## 2. Root causes (why "Rüstung" failed)

1. **Cross-lingual dilution (primary).** The KB is English SRD text; game queries are
   German. bge-m3 bridges the gap only weakly here: even a direct German hit scores
   ~0.45–0.58, and generic prose chunks like `equipment.md [Preamble]` score nearly as
   high as the actual `[Armor]` table chunk. For *"Kettenrüstung"* the real Armor chunk
   ranked **4th** for equipment.md — just below the `RAG_MAX_CHUNKS_PER_FILE=3` cut, so
   the attached "equipment" content contained no armor stats at all.
2. **The LLM query rewriter was dormant.** It is the only component that can translate
   German key terms into English D&D vocabulary (the fix that works — §4), but it only
   fires below `RAG_REWRITE_MIN_SCORE=0.35`, and measured top-scores never go that low.
   Its prompt also never mentioned that the KB is in English.
3. **HTML tables as embedding noise (systemic).** Every SRD chunk is full of
   `<td>/<th>` markup; item names ("Chain Mail", "Goblin") are wrapped in tags, which
   dilutes dense embeddings and keeps BM25 from exact-matching them.
4. **BM25 leg language skew (secondary, by design-ish).** For German queries the lexical
   leg matches the *German session notes* verbatim (they score 14–36 vs ~0 for English
   SRD chunks), so RRF fusion pushes session notes into the 4 attachment slots. Campaign
   memory is valuable, but it crowds out rulebook answers until the dense leg gets a
   language-matched boost (the rewrite provides exactly that).

Non-issues ruled out: GGUF index quantization (vectors verified equivalent), index
freshness (cache HIT, 2962 chunks), embedding backend health.

## 3. Evidence — production probe BEFORE fixes (`data/rag_probe_results.json`)

| Query (DE) | top-1 dense chunk | score | Armor chunk rank |
|---|---|---|---|
| Gib mir die Stat block vons verschiedneen Rüstungen | equipment.md **[Preamble]** | 0.569 | not in top-5 |
| Was ist die Rüstungsklasse einer Kettenrüstung? | equipment.md **[Preamble]** | 0.512 | **4th (cut by per-file cap)** |
| Wie viel kostet eine Plattenrüstung? | equipment.md [Raw Materials] | 0.558 | not in top-5 |
| armor stat block (EN baseline) | monsters-A-Z [Animated Armor] | 0.586 | 3rd |

## 4. Fixes implemented

| # | File | Change |
|---|---|---|
| F1 | `kb/chunker.py` | New `html_tables_to_plain_text()`: every HTML `<table>` is converted to plain text **before chunking** — header row → `Columns: a, b, c` legend, body rows → `Name \| v1 \| v2 …`, colspan category rows → `[label]`. All chunking strategies benefit; item names become clean tokens for both dense and BM25. |
| F2 | `kb/query_rewriter.py` | Prompt now states the KB is **English** (D&D 5e SRD + campaign lore) and requires every expansion to be in English D&D 5e vocabulary, with explicit German→English translation examples ("Rüstung" → "armor", "Heiltrank" → "potion of healing"). |
| F3 | `kb/retrievers.py` | New `_looks_german()` (umlauts or ≥2 German stopwords). German queries **always** route through the rewrite/expansion path, independent of the score threshold — this is what activates F2 for real game traffic. English/low-confidence behaviour unchanged. |
| F4 | index | Full force-rebuild so all 2962+ chunks are re-embedded from the plain-text chunking (one-time). |
| F5 | `kb/query_rewriter.py` | The rewrite LLM call now passes `extra_body={"enable_thinking": False}`. Qwen3-style hybrid-thinking models otherwise burn the whole 256-token budget on reasoning and return no content — the rewriter failed on **every** call (measured: 0.5 s with thinking off vs 5.7 s of pure thinking for zero output). Falls back to a plain retry if the backend rejects the param. |
| F6 | `kb/retrievers.py` | The BM25 lexical leg now scores **all query variants** (original + LLM expansions) and RRF-merges them before fusing with the dense ranking. Before: BM25 ran on the raw German query only, whose tokens ("stat", "block") matched monster-statblock text and German notes verbatim, drowning out the expansion-fused dense ranking (`[Armor]` was fused rank #4 but never attached). |
| F7 | `config/settings.py`, `kb/retrievers.py` | **Rewrite-all-queries by default** (`RAG_REWRITE_ALL_QUERIES=1`, kill-switch `=0`). The `_looks_german()` heuristic is no longer the trigger — with thinking off (F5) the rewrite costs ~0.5–1s, so every query gets expansions: German without umlauts can no longer slip past detection, and mangled English ("wizzard", "armor stat block") gets help too. Probe-verified: 29/33 → **30/33**, Q29 `armor stat block` fixed, no dilution regressions on confident English queries. |

Tests: `tests/test_rag_german_retrieval.py` (table conversion + German detection, 17 passed).
Full suite: no new failures (7 pre-existing failures on clean tree, unrelated: path
resolution / embed formatter / streaming budget).

## 5. Validation after deploy

Full production pipeline (`retrieve_kb_documents`, same entry point as the bot),
33 queries, index rebuilt with plain-text tables, rewriter live.
Results: `data/rag_probe_results_after.json` vs baseline `data/rag_probe_results.json`.

**Overall: 26/33 → 30/33 attached an expected file** (29/33 after F1–F6, 30/33 after F7 rewrite-all).

| Query (DE) | Before (attached files) | After (attached files) |
|---|---|---|
| Gib mir die Stat block vons verschiedneen Rüstungen | gameplay-toolbox, monsters-A-Z, rules-glossary, notes — **no equipment.md** | monsters-A-Z, **equipment.md**, playing-the-game, Warlock ✅ |
| Was ist die Rüstungsklasse einer Kettenrüstung? | classes, monsters-A-Z, rules-glossary, notes — **no equipment.md** | classes, rules-glossary, **equipment.md**, playing-the-game ✅ |
| Wie viel kostet eine Plattenrüstung? | notes, monsters-A-Z, equipment (3rd) | **equipment.md #1**, magic-items, gameplay-toolbox, Artificer ✅ |
| Zeig mir die Waffentabelle mit den Preisen | notes ×2, monsters-A-Z, equipment | **equipment.md #1**, character-creation, rules-glossary, classes ✅ |
| Was kostet ein Heiltrank? | notes, monsters-A-Z, spells, magic-items | **magic-items + equipment** ✅ |
| Erkläre die Regeln für Heimlichkeit | Hedge(!), notes, character-creation | **playing-the-game #1**, monsters-A-Z, monsters, classes ✅ |
| Wie erstelle ich einen neuen Charakter? | notes, Backstory, classes | classes, **character-creation**, rules-glossary, playing-the-game ✅ |

Remaining MISSes (3): *Heilzauber bis Stufe 3* (Cleric.md attached — arguably fine),
*besondere Zaubersprüche in Humblewood* / *Hintergründe in Humblewood* (world-doc recall,
separate campaign-lore issue, not a language problem).

Latency: every query now pays the rewrite call ≈ 0.5–2.2 s (thinking off), so total
retrieval is ≈ 1.2–1.6 s for both German and English queries (was ≈ 0.2 s for confident
English). Negligible next to 17–123 s answer generation; `RAG_REWRITE_ALL_QUERIES=0`
restores the fast path if it ever matters.

## 6. Remaining recommendations (not changed — judgment calls)

- **Latency:** German queries now pay one extra LLM call (≤ `RAG_REWRITE_BUDGET_SECONDS=10`)
  for the expansion. With thinking disabled (F5) it's ~0.5–2.2 s on the local Qwen
  backend; acceptable next to 17–123 s answer generation. Watch `dev.log`
  "Query rewrite triggered … in X.Xs" — and if you ever see *"full token budget on
  reasoning"* warnings again, a thinking model is burning its output budget (F5).
- **`RAG_MAX_CHUNKS_PER_FILE=3`** (`.env`) is tighter than the code default (5). After the
  fixes it's fine; if multi-section answers ("armor AND weapons") ever feel thin, raise to 5.
- **Session-note crowding:** BM25 legitimately favors German notes for German queries. If
  rulebook answers get crowded out again, options are: downweight non-`DnD5_5/` chunks in
  the RRF fusion, or lower `RAG_LEXICAL_TOP_K`.
- **The startup warning** `INDEX_EMBED_MODEL (gpustack/bge-m3-GGUF) differs from … BAAI/bge-m3`
  is cosmetic here (vectors verified equivalent, §1 step 4). Renaming the slug to align
  would force a full reindex for no gain — left as-is.
- Probe scripts kept in `scripts/rag_probe*.py` for future regression checks; rerun any
  time retrieval quality feels off: `docker compose exec -T bot python - < scripts/rag_probe.py`.
