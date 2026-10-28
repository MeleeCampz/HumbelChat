# HANDOFF — Image pipeline (#42) state, 2026-10-28 evening

Written for a clean-context takeover. Everything below is verified unless marked otherwise.

## 1. Where information lives (final layout — do not re-hardcode)

| Layer | Location | Edit via | Applies |
|---|---|---|---|
| Per-character facts: species (renderable English + in-world name), class, appearance, **Visual Anchor blocks** (~30 words, appearance-only, to be reused VERBATIM for named characters), pronouns in each header line, rendering-style notes (context only) | Vault `data/knowledge/humblewood/HumbleWood/Player Characters.md` | Obsidian (two-way sync) | next image, no rebuild (bind-mounted) |
| Cross-cutting image rules no single sheet can express — currently ONLY the googly-eyes rule (Column Bidae's alone; nobody else ever gets them; absent → don't mention) | Vault `HumbleWood/Image Generation Policy.md` | Obsidian | next image, no rebuild |
| Generic rewrite policy: world anchoring (Humblewood/D&D 5e), base style sentence ("Semi-realistic fantasy illustration — detailed fur and feather texture, natural light, storybook charm"), fixed template `[STYLE] . [SCENE] . [CHARACTERS] . [LIGHTING]`, budgets (scene ≤45 words, pose ≤12, unanchored chars ≤25 words, max 5 chars, <1300 chars solo / <2000 for 3+), no-negations-in-positive rule, user-requested-text-verbatim rule, KB-grounding + "read Player Characters ONCE" + never-drop-characters rules | Repo `prompts/image_rewrite_system.md` (baked into image; human-facing header stripped before injection) | editor + git | next REBUILD+deploy |
| Loader `_load_rewrite_system()` in `bot_core/images.py`: repo file + vault note injected at `{{CHARACTER_RULES}}`, re-read EVERY generation; missing files → logged built-in fallbacks (`_FALLBACK_POLICY`, `_DEFAULT_CHARACTER_RULES` = googly-eyes only) | `bot_core/images.py` | — | — |
| Negative prompt (generic guards only: humans/watermark/modern objects/quality) | `.env` → `IMAGE_NEGATIVE_PROMPT` (code default in `config/settings.py` is safety net) | .env | bot restart |
| Mechanics: model/GGUF/GPU/sizes/timeouts, TTS | `.env` (`IMAGE_*`, `TTS_*`) | .env | bot restart |

Party canon: Trixy Smoldersome (raccoon/Mapach artificer, she/her), Seta Onix (quokka-like mammal, in-world name "Jerbeen", ranger, she/her), Romi Stachelpfote (hedgehog monk, he/him), Bubo Krallensturm (owl cleric/archmage, he/him), Column Bidae a.k.a. Luma (blue-tit warlock, he/him, ONLY character with googly eyes). Marvin = Trixy's Steel Defender companion (no sheet yet).

## 2. ACTIVE ISSUE — group-scene rewrites sometimes emit invented generic animals

User report: the 13:33 live run of `/generate_image` "Die komplette Party an
Spieler:innen sitzt am Lagerfeuer und brät sich Marshmallows" painted FOUR
INVENTED animals (brown badger, grey wolf-folk, red fox, owl with spectacles)
— none of the party. Pipeline itself was healthy (load 24 s, gen 25.9 s).

Root cause traced via instrumented repro (`execute_tool` wrapper printing
call+result):
1. Model's first call: `read_section(file='Player Characters')` →
   **"No chunk found for file='Player Characters'"** (43-char error). The bare
   filename without path/`.md` does not resolve — the tool expects keys like
   `humblewood/HumbleWood/Player Characters.md`.
2. Fallback `search_kb(...)` WORKS: 8 hits incl. all five Visual Anchor sections
   (3715 chars returned).
3. BUT then "final answer was a tool call; nudging once for plain text" — the
   forced-final nudge path fired, and the model SOMETIMES drops the gathered
   context and writes from memory → generic animals. NON-DETERMINISTIC: the
   same request in the trace run produced the correct party answer.

Two fix directions (neither implemented yet):
- **(A) Fuzzy filename resolution in `read_section`** (`kb/tools.py`): accept
  bare stems ("Player Characters") case/extension-insensitively → first call
  succeeds, model gets the whole sheet incl. all anchors in one shot. Best fix.
- **(B) Harden the nudge/forced-final path** (`bot_core/agent_loop.py`
  `_plain_final` + decision-turn nudge): re-append `meta.gathered_context` to
  the "answer in plain text" message so the model cannot drop it.

After fixing: verify with the exact German party request (trace script pattern
in §5), then a live `/generate_image`, then regenerate the user's campfire scene.

## 3. Session work already done + deployed (all stamp-verified)

1. **Build stamp** (`955eeb2`): Dockerfile bakes `GIT_SHA` → `/app/.build_sha`;
   bot logs "Bot build sha: …" at startup; deploys verify running container ==
   repo tip. This ended the stale-image incident (a pre-fix image once served
   while looking healthy).
2. **Fragment guard unicode fix** (`ebb6604`): `looks_like_tool_call_fragment`
   now normalizes fullwidth brackets (U+FF1C/FF1E, U+2039/203A), strips
   zero-width chars (U+200B/200C/200D/FEFF), space-collapses before marker
   match; markers include ASCII + wide-slash (U+2044) variants. The 10:20
   "painted tool call" incident's string passed the OLD guard (width variants).
3. **Prompt-generation optimization** (several commits): Visual Anchors added to
   all 5 sheets; fixed template + budgets in policy; negations moved OUT of the
   positive prompt into the negative one; rewrite `max_tokens` 2048→4096
   (party prompts truncated mid-sentence at 671 chars before); ban invented
   species words + enforce sheet pronouns; group scenes read the party sheet ONCE
   (iteration-budget exhaustion made "whole party" come back with 3 of 5).
   Known soft limit: model still compresses anchors to ~50–80% under the cap
   even when told verbatim — identity-critical features survive every run.
4. **Policy externalization** (`80fa968`): see §1 layout. +4 loader tests.
5. **Per-character facts moved to sheets**: all 5 headers carry renderable
   species + explicit pronouns (Seta: "Quokka-like mammal (in-world name:
   Jerbeen)" — bare "Jerbeen" was the unrenderable token that triggered the
   old "cochran" hallucination; Column gained "(blue tit)"). Policy note shrank
   to the cross-cutting googly-eyes rule; code fallback mirrors only that.
6. **Mask line REMOVED** from negative prompt (`.env` + settings default):
   added earlier on a FALSE premise — no generated image ever showed Trixy's
   markings as a mask; the original confusion came from a CAPTION model
   describing a photo ("Banditmaske"). User corrected; unfounded config gone.

Test suite: ~1181 passed / 6 skipped at last full run (host venv).

## 4. Ops recipes

- **Deploy**: `docker compose build --build-arg GIT_SHA=$(git rev-parse HEAD) bot && docker compose up -d bot`, then verify: `diff <(MSYS_NO_PATHCONV=1 docker exec humbelchat-bot cat //app/.build_sha) <(git rev-parse HEAD)`.
- **Container paths**: prefix `MSYS_NO_PATHCONV=1` on docker exec/cp; container paths like `/app/...` (Git Bash mangles them otherwise).
- **Tests**: `C:/Users/Tobias/.cache/hc-test-venv/Scripts/python.exe -m pytest tests/ -q` (hermetic; conftest pins env before config import).
- **Repro rewrites in-container** (no Discord needed): script in `/app` calling
  `bot_core.images.prepare_prompt("…")`; patch `bot_core.agent_loop.execute_tool`
  to trace tool calls/results. Its logs do NOT go to dev.log (separate process) — capture stdout.
- **Live prompt trail**: `logs/dev.log` → grep "final image prompt" / "image negative prompt" / "prompt rewrite done" (shows iterations, tools, fallback).
- **Shared backend**: pi-session chat, bot completions and image gen share one model/engine. Stay quiet during batch image runs; contention causes load stalls/500s.
- **Vault sync**: two-way git push/pull every ~300 s; `data/` is gitignored by the repo (vault has its own git). After editing sheets, KB auto re-embeds changed files on next index tick (content-hash diff) — no manual reindex needed.
- **Branch/push**: work on `feat/tool-calling`. Pushes FROM THIS ENVIRONMENT 403 (its git credentials are read-only — sandbox token in GCM, SSH key is pi-remote-access only); the user pushes from their own machine. As of end of 2026-10-28 `feat/tool-calling` is fully pushed (tip `08ed8b5`). Never push master without explicit intent. `.env` never committed.

## 5. Immediate next steps

1. Implement fix (A) fuzzy `read_section` filename resolution in `kb/tools.py` (+ tests: bare stem, case-insensitive, extension-optional; no false matches across similarly-named files).
2. Implement fix (B) gathered-context re-append in the nudge/forced-final messages (`bot_core/agent_loop.py`).
3. Verify with the exact German party request via the in-container trace pattern (§4); expect 5 named characters with anchors, zero invented animals.
4. Rebuild+deploy (stamped), user tests `/generate_image` live; regenerate the campfire scene they're comparing against.
5. Backlog: #42 tuning/live validation ongoing; #47 blocked on user decisions; #50 depends on #42; #65 prompt-cleanup deferred; #67 has a capture recipe if the silence gap recurs. (Pushes from this environment 403 — user pushes from their machine.)

## 6. Status — evening of 2026-10-28, after clean-context takeover (§5 steps 1–3 DONE)

- **Fix (A) + (B) implemented, deployed (stamped), verified** — commits
  `ba9e942` (kb fuzzy stem + full-file reads), `15183fb` (gathered-context
  re-append in forced-final/nudge/fallback messages), `c44ed2d`
  (trace script). Running container stamp-verified on tip; backend probe 200.
- **In-container verification GREEN** (`scripts/image_rewrite_trace.py`, the
  exact German party request, 3 runs): first call
  `read_section(file="Player Characters")` now succeeds with the FULL sheet
  (34.5k chars, all 49 sections); every run paints all 5 named characters
  with anchors and **zero invented animals**; `fallback=False` each time.
  Run 2 compressed some anchors to ~50% — the KNOWN SOFT LIMIT from §3.3,
  not a regression (identity-critical features survived; run 3 fully verbatim).
- **Still open:** user live-tests `/generate_image` and regenerates the
  campfire scene they're comparing against (§5 step 4) → then #42 can close.
  Backlog #42 carries the full completion note.

## 7. Second fix same day — party+bandit prompt dropped 4 of 5 characters

- **Symptom:** user's "whole party + bandit sneaking up" request rendered
  ONLY Column (dev.log 14:52, `fallback=False`, tools fine). Root cause:
  policy conflict — "At most 5 characters total" vs "never silently drop any
  of them"; with 6 entities the model dropped 4 party members to fit the
  2000-char cap.
- **Fix (repo, `de4f05d`, deployed stamped):** inclusion now explicitly BEATS
  the char cap; >5 characters compress anchors to identity-critical cores
  (species + top three features + signature item) instead of dropping any.
  Never-drop rule also mirrored into vault `Image Generation Policy.md`
  (bind-mounted — applies to the next image without a rebuild).
- **Sheet bloat cleanup (user decision):** `Player Characters.md` condensed IN
  PLACE (vault commit `dd2e2cf`) — per character: header line + merged
  Appearance block (~70 words, identity-critical details) + verbatim Visual
  Anchor + one Rendering line; detailed sections and the duplicate Reusable
  Prompt Blocks deleted. 28k → ~7.5k chars; anchors byte-identical (googly-
  eyes rule unaffected). Repo policy's rendering-style reference updated to
  the new structure in `de4f05d`.
- **Verified** in-container with the user's exact failing prompt: all 5 party
  members + bandit present, zero drops, anchors compressed per the new rule.
