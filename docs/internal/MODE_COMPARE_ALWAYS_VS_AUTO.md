# Mode compare: `RAG_AGENT_MODE=always` vs `auto` (empirical, 2026-10-23)

Harness: `scripts/rag_mode_compare.py` — 36 prompts × 2 real end-to-end passes
through `ask_ai` (`off` = production fast path; `always` = tools available).
The `auto` counterfactual is computed from pass A's observed rerank top via the
same `agent_trigger_reason()` production uses. Raw results:
`data/evals/mode_compare_20261023T2337UTC.jsonl` (gitignored; regenerate with
the script).

## Setup

- 36 prompts across 7 categories: simple-rules (8), enumeration (6), compound
  (4), negative/impossible (6), session-info (6), items (4), monsters (2);
  mixed DE/EN like the live campaign.
- All backends healthy (chat :8888, embed host.docker.internal:11435,
  rerank host.docker.internal:8000). Zero errors, zero agent fallbacks.
- `RAG_AGENT_MODE` untouched in `.env` (stays `auto`) — mode switching was
  in-process only. History persistence + last-session context disabled for
  isolation/identical conditions.

## Headline numbers

| Metric | Value |
|---|---|
| `always` invoked tools | **31/36 (86%)** — 54 calls total: search_kb ×31, read_section ×14, list_sections ×9 |
| `auto` would trigger | **14/36 (39%)** — enumeration-shape ×7, compound-question ×6, low-rerank (<0.55) ×2 |
| auto-triggered AND tools used (always) | 14/14 — **zero wasted triggers** in this sample |
| tools used (always) but auto silent | 17 — mostly harmless single verification searches; **2 substantive misses** (below) |
| latency mean | fast 29.4 s vs always 31.6 s (+7% overall); only enumeration pays a real cost (+13.6 s) |
| quality regressions (always worse than fast) | **0** of 36 |

## Quality findings

**`always` wins (3 clear):**
- **e04** "Which monsters have appeared in our campaign logs so far?" — fast
  path only saw Session 3's log (top-k retrieval); `always` enumerated
  sessions 1+3 via search/read_section. **`auto` did NOT trigger this one**
  (rerank 0.89, shape regex missed "have appeared … so far"). The single most
  valuable miss in the sample.
- **c04** "What did we find in the tomb and which traps were in it?" — fast
  path honestly said "no tomb in my context" (retrieval attached monster gear);
  `always` found Session 6's burial hall and answered fully. `auto` DID
  trigger (compound-question) — covered.
- **i03** "Find an item that helps restore hit dice faster" — fast gave the
  indirect Potion of Vitality (HP, not HD); `always` found the exact-match
  Rejuvenating Draft. `auto` silent — subtle miss.

**Full-pair reading (all 36 pairs read end-to-end, not just telemetry):**
6 clear `always` wins — e04 and c04 (above), i03 (above), **e03** (DE Int-feats:
fast path deflected to "not in attached sections" + ASI rule; `always` actually
read feats.md and answered concretely, incl. flagging the file's truncation),
**c02** (same deflection pattern — `always` checked feats.md directly), **c03**
(DE Cha-spellcasting races: fast path found only Tiefling; `always` also found
Elf/Elven Lineage and added the Warlock/Sorcerer/Bard class layer). 4 of the 6
are inside auto's trigger set (e03 enumeration, c02/c03/c04 compound); e04 and
i03 are the misses. Marginal `always` improvements: s01/s05 (richer session
summaries via list_sections), m02 (Dust Mephit as the desert answer vs a bare
"none listed"), n03/n06/i04 (search-confirmed absences, explicit "nothing else
listed"), r01/r08 (more structure). The other ~24 pairs are quality-equivalent;
zero regressions. **Honesty held in both modes on all 6 negatives** — no
hallucinated lore.

**Trigger false positives observed: none harmful.** r02 ("…und was darf man
dabei?") and r06 ("Welche … gibt es?") triggered auto; the extra search call
was a cheap verification and answers were equal quality. The score gate fired
twice (n01 0.45, m02 0.54) — both absence questions where searching confirmed
the absence; correct behavior.

## Conclusion + action

`auto` remains the right default: 39% trigger rate with zero wasted triggers
and no observed quality loss *within its trigger set*. But it structurally
misses **cross-log/cross-file enumeration** — the question class where tools
have the highest value and retrieval top-k is weakest (e04, i03-class).

**Trigger improvement candidate (not yet implemented):** extend the
enumeration/compound rules with cross-log phrasing — e.g.
`appeared|so far|across (all |our )?sessions|in our campaign` — which would
have caught e04 at near-zero false-positive cost (no non-enumeration prompt in
the sample matches that phrasing). Re-run this harness after the change to
confirm the trigger set grows only by the intended class.

## Caveats

- n=36, one campaign's KB; latency figures include shared-GPU contention and
  thinking-model variance (per-prompt fast/always deltas swing ±15 s).
- `always` pass ran second per prompt; no evidence of order effects, but the
  +7% mean should not be read as a precise cost.
- The auto counterfactual uses pass A's rerank score; a live `auto` run could
  see slightly different scores (retrieval is deterministic in practice).
