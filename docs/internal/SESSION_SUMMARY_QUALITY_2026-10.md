# Session Log & Auto-Summary Quality Review — 2026-10 (#73)

Scope: the four most recent ended sessions under
`data/knowledge/humblewood/HumbleWood/SessionLogs/` — `2026-10-28_01_To the roots`,
`2026-10-20_02_We fucked up back to Alderheart`, `2026-10-16_01`, `2026-10-07_02`.
Method: raw upload (`.attachments/*.md`) vs the AI artifacts it produced (Overview recap +
`## Session Log` merged log), cross-checked against `logs/dev.log` generation lines.

## 1. What the pipeline actually produces

At `/end_session` (commands/session_commands.py):

1. **Merged log** (`DEFAULT_SESSION_MERGE_PROMPT`): all session uploads/transcripts → one
   polished chronological narrative. Model-max output, reasoning at model default.
2. **Overview/recap** (`DEFAULT_SESSION_RECAP_PROMPT`): ≤150 words, from the RAW docs + notes
   (generated *before/independently of* the merge).
3. notes.md renders: header → `## Notes` → `## Overview` → `## Session Log` (merged log
   *replaces* the mechanical full-text `## Documents` section).

Raw uploads stay on disk in `.attachments/` → re-merge is always possible
(`data/rebuild_merged_log.py`). Log check: only one merge failure ever (2026-09-30, old model
burned its token budget on reasoning — later fixed by model-max output); the four reviewed
runs all succeeded cleanly.

## 2. Overall verdict

**Quality is good — the core design works.** Specifically, verified on the 10-28 session
(raw 10.9 KB → merged 10.3 KB):

- **No hallucination.** Every recap bullet and merged-log passage I spot-checked traces to the
  raw source — including suspicious ones like "Baladin stellt ein lautes Clowns-Zelt auf" and
  "Tempel … Titan and Eva" (raw line 54: temple NAMES, so the English is correct, not mixing).
- **Completeness:** the merged log kept every event, name, the Nat-20, loot (Diadem,
  Küchenschwert mit Runen), the captain's shield tactics, Marvin's death+rescue, even the
  cliffhanger. Nothing invented, nothing padded.
- **Polish:** first-person rough fragments → coherent third-person narrative; "Mein das Zeichen…"
  → correct indirect speech ("sei bereits … aufgetaucht").
- **Next-session value:** overviews are genuinely useful as "where we left off" briefs.

What follows is what I'd change — refinements, not fixes of a broken system.

## 3. Defects found (with evidence) and how I'd rewrite them

### 3.1 Recap labels are not consistently localized (prompt bug-class)
The recap prompt pins the literals `**What happened:**` / `**Next steps:**` while its LANGUAGE
rule demands "ENTIRE recap in the source language". Models split the difference per session:
- 10-28: English labels, German bullets
- 10-20: `**Was passiert ist:** / **Nächste Schritte:**`
- 10-16: `**Was geschah:**` (yet another variant)

**Fix options:** (a) instruct: "translate the two bold labels into the source language", or
(b) mechanically: ask for only the bullets and have `sessions.py` write fixed labels itself.
(b) is deterministic and costs nothing — I'd do (b).

### 3.2 "Next steps" restate the last event instead of naming an open task
10-28 recap: `Sie müssen den Tunnel weiter erkunden; Romi wurde von einem Cube am Kopf
getroffen.` — the second half is an *event already told in bullet 7*, not a next step.
**Prompt rewrite:** "Next steps = open tasks or hooks phrased as actions; NEVER restate an
event already present in the 'What happened' bullets."

### 3.3 Rare coined words slip into recaps
10-20: "Die Gruppe **nächtigt** Richtung Alderheart" (raw said "nächtigen" — itself non-standard
for "übernachten"). Recap-level grammar errors are disproportionately visible (it's the part
re-read at next session start). **Prompt addition:** "Use clean standard language; do not coin
verbs; convert table slang to standard vocabulary in the RECAP (the merged log may keep it)."

### 3.4 Obsidian image embeds are dropped non-deterministically (biggest content loss)
- 10-20 merged log KEPT `![[Eulenbär.png]]` + `![[Zephyr & Co.png]]`.
- 10-28 merged log DROPPED `![[BanditenLagerInDenRoots.png]]` (the bandit-lager map — the most
  useful image of the session). 10-16 also dropped its one embed.
Same prompt, different behaviour per run — the model treats embeds as "filler" sometimes.

**Prompt rewrite (merge):** "Preserve every Obsidian embed `![[...]]` VERBATIM, attached to the
scene where it appears — they are part of the record (maps, battle boards)." This is the only
rule I'd add as a STRICT RULE, because lost images are unrecoverable from the merged text.

### 3.5 Gamer-jargon "corrections" sometimes overreach
"Der zweite Angriff … **krittet**" → "Sein zweiter Angriff **kritisiert**" — real German verb,
wrong meaning (crit → kritisiert). "Stiefelkreisen" → "prügeln" is fine. The merge should
polish grammar but the table's vocabulary (looten, critten, disengagen, Moonbeam indoor) is
identity, not an error.
**Prompt line:** "Fix grammar/spelling, but KEEP common TTRPG anglicisms/jargon; never replace a
game term with a similar-sounding standard verb."

### 3.6 POV handling is left to chance
Raw is first-person ("wir verlieren Column"); the merge chose third-person ("die Gruppe") —
good, but unprompted. Different sessions could drift to different perspectives.
**Prompt line:** "Write the log in consistent third-person ensemble view ('die Gruppe'),
regardless of the sources' POV." (Or pin first-person plural — my preference is third.)

### 3.7 Players' own name spellings contradict each other — correctly preserved, but worth fixing upstream
- "Column" (10-28 raw) vs "Colum" (10-20 raw) — same character, and the prompt's
  "keep names exactly" rule faithfully propagates both.
- "Beast Burrow" (10-20 raw) vs "Bees Burrow" (10-28 raw) — same building.
This is NOT a bot defect; it will confuse RAG lookups though ("Beast Burrow" chunk vs "Bees
Burrow" chunk never match). **Fix upstream:** a small glossary in the vault (e.g.
`HumbleWood/Glossary.md` with canonical spellings), and/or inject "Known canonical names: …"
into the merge/recap user message so the model maps variants consistently.

## 4. Structural suggestions (bigger-lever, optional)

1. **Generate the recap from the merged log instead of the raw drafts.** Today recap and log
   are two independent passes over the same raw text → they can disagree (see 3.7 spellings),
   and the recap must re-compress un-deduplicated drafts. Order: merge → recap(merged_log +
   notes). Cost is lower (one strong pass instead of two), and the brief is then guaranteed to
   match the canonical record it previews in notes.md.
2. **Add a mechanical `## Attachments` footer** to notes.md listing every file in
   `.attachments/` — belt-and-braces against 3.4 even if a model still drops an embed.
3. The 09-30 merge failure mode (reasoning ate the whole budget) is already mitigated
   (model-max output, no `max_tokens` cap) — no action; just don't reintroduce a cap via
   `SUMMARY_CALL_MAX_TOKENS` for the merge call.

## 5. Ready-to-apply prompt deltas

`DEFAULT_SESSION_MERGE_PROMPT` — add to STRICT RULES:
```
- Preserve every Obsidian embed like ![[name.png]] VERBATIM, attached to the scene where
  it appears — they are part of the record (maps, battle boards). Dropping one is a failure.
- Write in consistent third-person ensemble view ("die Gruppe"), regardless of POV used
  in the sources.
- Fix grammar/spelling, but keep common TTRPG jargon/anglicisms (looten, critten,
  disengagen, spell names); never replace a game term with a similar-sounding standard verb.
```
`DEFAULT_SESSION_RECAP_PROMPT` — replace the "Next steps" bullet spec and labels:
```
- **Next steps:** 1–3 short bullets phrased as OPEN TASKS or HOOKS the sources name; never
  restate an event already present in the "What happened" bullets, never speculation.
- Translate the two bold part-labels into the sources' language (e.g. "Was passiert ist:" /
  "Nächste Schritte:") — or better: the bot adds fixed labels, output only the bullets.
- Use clean standard language in the recap; never coin verbs and never invent words.
```

## 6. Status (2026-10-30): merge deltas APPLIED + existing files fixed

- `DEFAULT_SESSION_MERGE_PROMPT` now includes all three deltas: VERBATIM-embeds strict rule,
  pinned third-person ensemble view, TTRPG-jargon preservation. Recap deltas from §5 remain
  proposals (the user asked for merge-prompt changes only).
- Past sessions curated by hand (Studio backend was down, so a model re-merge was not
  possible — the curation was done directly against the raw sources):
  - 10-28: `![[BanditenLagerInDenRoots.png]]` restored at the tunnel-discovery passage;
    "kritisiert" → "krittet"; `last_ended.merged_log` in `sessions.json` re-synced from the
    corrected notes.md (backup: `data/sessions.json.pre-73fix`).
  - 10-16: `(Bildreferenz: BrumbelwellCatacombs.png)` → real `![[BrumbelwellCatacombs.png]]` embed.
  - 10-07: merged log converted from first-person ("wir", 18 sites) to third-person ensemble
    view; "eine Rest" → "eine Rast" typo fixed; both images (webp + `![[Susan.png|334]]`) were
    already preserved — left in place.
  - 10-20: already conformant (2 embeds verbatim, third-person, no jargon damage) — untouched.
- Verification: every merged section greps clean (`wir`/`Wir` = 0, embeds present, no
  `Bildreferenz`, no `kritisiert`); full suite 1292 passed.
- Note discovered while fixing: embed handling had THREE behaviours (kept verbatim / dropped /
  down-converted to a `(Bildreferenz: …)` text note) — the new strict rule pins all runs to
  the first behaviour. Changed notes.md files will be re-embedded automatically on next bot
  start (index load validates source-file hashes), no manual re-index needed.

## 7. Verification trail

- Compared raw vs merged for 10-28 in full (134-line raw, 10,281-char merged) — coverage ~100%,
  1 dropped embed, 1 jargon mis-fix ("kritisiert"), nothing invented.
- Spot-checked every 10-20 recap bullet against the raw (Baladin, Clowns-Zelt, Pudding, Essenz,
  Eulenbär/Hunting Traps, Titan and Eva, Beast Burrow) — all sourced.
- dev.log: recap/merge request lines present for all four sessions; zero merge failures since
  2026-09-30 (old-model reasoning-budget issue, already fixed).
- Raw uploads confirmed retained in `.attachments/` → historical re-merge safe via
  `data/rebuild_merged_log.py` (e.g. re-merge 10-28 to recover the map embed).
