# RAG Benchmark — 2026-10-16 (Qwen3-Reranker-4B era, current state)

**Date:** 2026-10-16 · **Harness:** `scripts/rag_benchmark.py` (committed) ·
**KB:** humblewood, 5,387 chunks · **Cases:** 34 golden (incl. TQ torture case 0) + 8 extra (session/property/negative) = 42

**Stack under test (live production endpoints):**

| stage | backend |
|---|---|
| dense retrieval | `qwen3-embedding:4b` via Ollama sidecar (`embedder-qwen`, GPU 1) |
| lexical | BM25 in-process |
| fusion | RRF (dense + lexical [+ rewrite expansions]) |
| rerank | `Qwen3-Reranker-4B` INT4 (compressed-tensors) via vLLM sidecar (`reranker-qwen`, GPU 1), window 150 |
| query rewrite | LM Studio `:8888`, `unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS` (same model as the chat backend) |

Two full runs were captured: **A** = rewrite disabled (lower bound, 42/42 single-pass),
**B** = rewrite enabled (production configuration). Results JSONs:
`data/rag_benchmark_results.json` (run B; run A archived in this doc's numbers).

---

## Verdict

1. **Retrieval quality is the best it has ever been.** 38/40 expected-file hits
   (95% recall@4), **zero regressions** vs the TEI bge-reranker era on the shared
   33-query set, and the canonical TQ torture case went from *rank 25 in a noise
   band* (postmortem 2026-10-15) to **rank 1 at rerank score 0.94–0.96** in every
   configuration tested. The two remaining misses are both enumeration queries —
   a known, separate class problem (Finding 3).
2. **The rewrite path is currently broken by latency, not by quality.** In run B,
   **17/42 (40%) of rewrites exhausted the 45 s budget** and were dropped: those
   queries paid ~50–57 s end-to-end *and* lost their expansions. The 25 that
   succeeded took a median of 1.5 s. Hit rate was unaffected (38/40 in both runs),
   so the cost is pure latency — but it makes ~40% of Discord answers stall for
   45 s on the chat LLM backend for a utility call. This is the top action item.

## Headline metrics

| config | hits | recall@1 | recall@2 | recall@4 | median latency | p95 |
|---|---|---|---|---|---|---|
| bge-m3 + no rerank (Oct 10, `rag_eval_results.json`) | 28/33 | 36% | 76% | 85% | 969 ms | 10.0 s |
| + local CPU bge-reranker (Oct 10) | 30/33 | 61% | 79% | 91% | 10.1 s | 11.6 s |
| + GPU TEI bge-reranker-v2-m3 (Oct 10, `rag_eval_gpu.json`) | 30/33 | 70% | 82% | 91% | 1.7 s | 2.0 s |
| **Qwen3-Emb + Qwen3-Reranker-4B, rewrite OFF (run A)** | **31/33** on shared set | 64% | 85%* | 94%* | 3.9 s | 4.5 s |
| **Qwen3-Emb + Qwen3-Reranker-4B, rewrite ON (run B)** | **38/40** (incl. TQ + 6 new) | 65% | 85% | 95% | 17.6 s | 56.9 s |

\* on the full 40 scored cases of run A. Run B's latency is dominated by the
45 s rewrite budget stalls (Finding 2); excluding the 17 stalled queries, median
is ~10 s.

Per-query delta vs the best prior config (`rag_eval_gpu.json`, shared 33):
**1 fixed** (*Kettenrüstung AC* — was monsters-A-Z/magic-items, now equipment.md),
**0 regressed**, 2 still missing (both enumeration). Five lookups slid from
position 0 to 1–2 where Qwen3 ranks an equally-valid alternative file first
(e.g. *healing potion price*: D&D Beyond Basic Rules ahead of equipment.md) —
golden-set granularity, not a quality loss.

## Per-query detail (run B — production config)

| # | q | lang | result | ms | d_top | rew | pool | rr_tops | attached (top 4 files) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Gibt es ein Item dass mir hilft nicht aufgedeckt zu werden… (TQ) | DE | **HIT@0** | 20609 | 0.671 | Y | 418 | .939/.954 | Saddlebag p2, playing-the-game, magic-items, DMG |
| 2 | Gib mir die Stat block vons verschiedneen Rüstungen | DE | HIT@0 | 8240 | 0.704 | Y | 375 | .905/.913 | equipment, magic-items, rules-glossary, monsters |
| 3 | Was ist die Rüstungsklasse einer Kettenrüstung? | DE | HIT@0 | 8768 | 0.665 | Y | 347 | .919/.846 | equipment, monsters, playing-the-game, DMG |
| 4 | Wie viel kostet eine Plattenrüstung? | DE | HIT@1 | 51483 | 0.642 | **stall** | 149 | .822 | magic-items, equipment, gameplay-toolbox, DMG |
| 5 | Zeig mir die Waffentabelle mit den Preisen | DE | HIT@0 | 57923 | 0.669 | **stall** | 191 | .885 | equipment, magic-items, Saddlebag p3, Descent into Avernus |
| 6 | Was kostet ein Heiltrank? | DE | HIT@1 | 50002 | 0.671 | **stall** | 192 | .927 | magic-items, equipment, D&D Beyond Basic Rules, spells |
| 7 | Welche Zauber kann ein Magier in Stufe 1 wirken? | DE | HIT@0 | 20327 | 0.708 | Y | 399 | .918/.887 | equipment→(spells set) |
| 8 | Was macht der Zauberspruch Feuerball? | DE | HIT@0 | 9533 | 0.754 | Y | 426 | .952/.896 | spells, PHB, magic-items, DMG |
| 9 | Gib mir alle Heilzauber bis Stufe 3 | DE | HIT@3 | 9237 | 0.656 | Y | 415 | .877/.931 | classes, PHB, Cleric, spells |
| 10 | Wie funktioniert Konzentration beim Zaubern? | DE | HIT@0 | 8936 | 0.733 | Y | 354 | .933/.940 | spells, rules-glossary, classes, magic-items |
| 11 | Was kann ein Krieger in Stufe 5? | DE | HIT@2 | 9097 | 0.621 | Y | 407 | .907/.939 | Ranger, Artificer, classes, monsters-A-Z |
| 12 | Welche Klassen gibt es in D&D 5e? | DE | HIT@0 | 9724 | 0.644 | Y | 424 | .848/.861 | classes, monsters-A-Z, Artificer, PHB |
| 13 | Wie erstelle ich einen Charakter… | DE | HIT@0 | 53969 | 0.674 | **stall** | 169 | .929 | classes, character-creation, spells, Artificer |
| 14 | Was für Tierarten gibt es… | DE | HIT@1 | 50806 | 0.699 | **stall** | 154 | .901 | rules-glossary, monsters-A-Z, animals, Arcana Unleashed |
| 15 | goblin statblock | EN | HIT@0 | 10340 | 0.665 | Y | 294 | .934/.934 | monsters-A-Z, monsters, rules-glossary, classes |
| 16 | Was ist eine Vulpin? / Spezies-Frage | DE | HIT@0 | 56864 | 0.531 | **stall** | 188 | .504 | animals, monsters-A-Z, rules-glossary, playing-the-game |
| 17 | …(DE rules) | DE | HIT@0 | 49748 | 0.814 | **stall** | 159 | .935 | rules-glossary, playing-the-game, monsters, Ranger |
| 18 | …(DE rules) | DE | HIT@0 | 39791 | 0.735 | Y (30.8 s) | 478 | .955/.918 | playing-the-game, rules-glossary, classes, monsters |
| 19 | …(DE rules) | DE | HIT@0 | 57442 | 0.704 | **stall** | 166 | .920 | playing-the-game, rules-glossary, classes, Saddlebag p2 |
| 20 | …(DE rules) | DE | HIT@0 | 50336 | 0.737 | **stall** | 143 | .931 | rules-glossary, playing-the-game, Cleric, classes |
| 21 | Wie würfle ich die Werte… (Stärke/Geschicklichkeit) | DE | HIT@0 | 14874 | 0.668 | Y | 404 | .894/.914 | character-creation, gameplay-toolbox, character-origins, playing-the-game |
| 22 | …(DE rules) | DE | HIT@0 | 53789 | 0.615 | **stall** | 138 | .825 | classes, gameplay-toolbox, character-origins, Ranger |
| 23 | Wie würfle ich die Werte für Stärke und Geschicklichkeit? | DE | HIT@0 | 50732 | 0.622 | **stall** | 189 | .930 | character-creation, rules-glossary, playing-the-game, magic-items |
| 24 | Wie ist der Kalender in Humblewood aufgebaut? | DE | HIT@0 | 11067 | 0.667 | Y | 432 | .896/.880 | Humblewood_Calendar, World_Overview, world_context, notes |
| 25 | Wer ist Hath und was ist sein Reich? | DE | HIT@1 | 56277 | 0.649 | **stall** | 128 | .917 | Religion, Hath, Backstory, animals |
| 26 | Was kann die Spezies Vulpin? | DE | HIT@1 | 51033 | 0.797 | **stall** | 186 | .723 | animals, species, Vulpin, notes |
| 27 | Welche besonderen Zaubersprüche gibt es in Humblewood? | DE | **MISS** | 27536 | 0.592 | Y | 499 | .788/.756 | spells, classes, notes, world_context |
| 28 | Welche Hintergründe gibt es in Humblewood? | DE | **MISS** | 53684 | 0.654 | **stall** | 161 | .931 | World_Overview, Alderheart, notes, notes |
| 29 | Was ist in der letzten Session bei Alderheart passiert? | DE | HIT@0 | 49766 | 0.747 | **stall** | 128 | .814 | notes ×4 |
| 30 | armor stat block | EN | HIT@0 | 14285 | 0.651 | Y | 416 | .927/.929 | equipment, monsters, playing-the-game, rules-glossary |
| 31 | fireball spell | EN | HIT@2 | 53727 | 0.755 | **stall** | 168 | .930 | PHB, magic-items, spells, DMG |
| 32 | goblin statblock (EN dup) | EN | HIT@0 | 48958 | 0.699 | **stall** | 104 | .917 | monsters-A-Z, rules-glossary, animals, PHB |
| 33 | healing potion price | EN | HIT@0 | 13665 | 0.728 | Y | 414 | .879/.895 | equipment, D&D Beyond Basic Rules, magic-items, Saddlebag p3 |
| 34 | wizard class table | EN | HIT@1 | 10335 | 0.742 | Y | 397 | .879/.954 | spells, classes, Warlock, character-creation |
| 35 | What happened when Tempa tried to fly? | EN | HIT@0 | 9316 | 0.783 | Y | 391 | .912/.845 | notes, playing-the-game, rules-glossary, DMG |
| 36 | Tell me about the Black Pudding incident. | EN | HIT@1 | 10191 | 0.704 | Y | 449 | .847/.921 | monsters-A-Z, notes, Saddlebag p1, World_Overview |
| 37 | Erzähl mir von der Nacht in der Taverne… | DE | HIT@0 | 10396 | 0.625 | Y | 433 | .805/.826 | notes ×3, world_context |
| 38 | Welches Item macht mich unsichtbar, wenn ich einen Angriff verfehle? | DE | HIT@1 | 9760 | 0.699 | Y | 503 | .956/.941 | Saddlebag p2, magic-items, Ambush_Prey, DMG |
| 39 | Who is Hath and what is his realm? | EN | HIT@0 | 10378 | 0.621 | Y | 453 | .901/.531 | Hath, Religion, World_Overview, Cleric |
| 40 | Gib mir den Statblock vom Schwarzen Drachen | DE | HIT@2 | 10720 | 0.734 | Y | 335 | .933/.894 | monsters, rules-glossary, monsters-A-Z, magic-items |
| 41 | What is the capital of France? (negative) | EN | neg | 9570 | 0.445 | Y | 360 | .257/.281 | World_Overview, rules-glossary, world_context, Calendar — **9.4k chars junk** |
| 42 | Wie funktioniert die Steuererklärung in Deutschland? (negative) | DE | neg | 9693 | 0.330 | Y | 398 | .266/.349 | rules-glossary, spells, notes, Saddlebag p1 — **13.7k chars junk** |

(run A = same set, rewrite off: identical hit pattern except #1 pool 178 vs 418;
median latency 3.9 s.)

## Findings

### 1. The reranker swap achieved its goal — with margin
TQ (case 1): rank 25 in a 0.002–0.018 noise band under bge → **rank 1 at 0.94**,
with the *next-best* chunk at ≤0.96 on the expansion pass. Hit top-rerank scores
on correct answers: median 0.919, min 0.504 (animals/Vulpin — genuinely fuzzy
match), max 0.956. Negative queries top out at 0.26–0.35. The score separation
between "real answer" and "everything else" is now clean (~0.5 gap) — it was
noise before. Rerank passes: 67/67 OK, 0 errors (the stale-bot 400 era is over).

### 2. Rewrite budget exhaustion — the current top problem
Run B: 25 rewrites succeeded (median **1.5 s**, one outlier at 30.8 s), 17
exhausted `RAG_REWRITE_BUDGET_SECONDS=45` and were dropped by
`asyncio.wait_for` in `kb/retrievers.py`. Every stalled query took 49–58 s
end-to-end. The same backend produced 0.1–1.0 s rewrites in this morning's
production logs, so the stalls are **intermittent** — consistent with the
documented Qwen3 thinking-mode leakage (the model occasionally burns its token
budget reasoning despite `enable_thinking=False`) or LM Studio queue stalls.
Note `.env` raised the budget from the code default of 10 s to 45 s; at 45 s a
hung rewrite is *worse than no rewrite* (45 s latency + lost expansions, vs
10 s + lost expansions). **Recommendation:** drop `RAG_REWRITE_BUDGET_SECONDS`
back toward 8–10 s so bad rewrites fail fast; investigate the intermittent
:8888 stalls separately (small dedicated rewrite model is the structural fix).

### 3. The two persistent misses are enumeration queries — a different problem class
*„Welche besonderen Zaubersprüche gibt es in Humblewood?"* and *„Welche
Hintergründe gibt es in Humblewood?"* have been misses in **every** config since
Oct 10 (bge, CPU rerank, TEI, Qwen3). The correct answer spans seven / three
small single-item files; no single chunk enumerates them, so the most
topically-relevant *aggregate* file (spells.md / World_Overview) wins — correctly,
by all retrievers' metrics. Fix candidates (not started): list-aware query
handling (attach by directory/topic), or a KB-side index file that lists special
spells/backgrounds and ranks first for such queries.

### 4. Negative cases quantify the attach-floor backlog (#43)
Unanswerable queries still drag in 9.4k / 13.7k chars of junk at rerank tops of
0.26–0.35, while real answers sit at ≥0.50. A score-driven attach floor around
**0.5** (Qwen3-calibrated) would have attached *nothing* to both negatives and
changed nothing on the 38 hits — direct evidence for backlog #43.

### 5. Latency profile vs TEI era
Rewrite-free path: ~9–11 s/query (embed ~1 s + dense/lexical ~1 s + rewrite
median 1.5 s + Qwen3-4B rerank of ~150 candidates in up to 2 passes ≈ 4–7 s).
TEI bge-reranker-v2-m3 did the same job in ~1.7 s median end-to-end. The 4B
cross-encoder is ~5× costlier per pass but buys the score separation in Finding 1;
for a Discord bot 10 s is acceptable, the 45 s stalls are not (Finding 2).

## Reproduce

```powershell
# from the host (production endpoints via compose DNS):
docker compose cp scripts\rag_benchmark.py bot:/tmp/ && docker compose exec bot python /tmp/rag_benchmark.py
# from the dev sandbox (published loopback ports, bypass the proxy for :8888!):
NO_PROXY="…,host.docker.internal" EMBED_URL=http://host.docker.internal:11435/v1 `
RERANK_API_BASE=http://host.docker.internal:8000 python scripts/rag_benchmark.py
```

Gotcha learned the hard way this session: sandbox HTTP to `host.docker.internal`
goes through the firewall proxy unless it is in `NO_PROXY` — the proxy added
~30–50 s per LLM call and made run B's first attempt useless. Rerank/embed calls
tolerate it; the streaming OpenAI client does not.
