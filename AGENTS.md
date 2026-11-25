## LLM backend topology (read before touching models — GPU split, 2026-11)

Two physical cards, two backends — **never mix them up**:

- **Chat (the bot AND this agent): Strata on the 5070 Ti (GPU 1)** — `INFER_URL`
  in `.env`. The 5070 is chat-only: image gen and the RAG sidecars must never
  load onto it.
- **Image generation: Unsloth Studio on the 4070 Ti SUPER (GPU 0)** —
  `IMAGE_INFER_URL` (default falls back to `INFER_URL`; production sets it to
  `http://host.docker.internal:8888/v1`, Studio pinned to GPU 0 only).
  Image model: `unsloth/Qwen-Image-2.1-Turbo-GGUF` Q4_K_M, measured resident
  ~12.5 GiB (`resident_mib=12456` in dev.log).
- **The 4070 runs ONE mode at a time**, owned by the `gpu-switch` compose
  service (`scripts/gpu_switch_helper.py`): `rag` mode (reranker-llama +
  embedder-llama, default steady state — both pinned `CUDA_VISIBLE_DEVICES=0`)
  ⇄ `image` mode (those stopped, diffusion model loaded). The bot switches via
  `GPU_SWITCH_URL` around every painting window (`IMAGE_MODE_HOLD_S` defers the
  switch-back — only for windows the BOT switched; a manual-mode window restores
  rag immediately) — see `bot_core/images.py`; manual CLI twin:
  `./scripts/gpu_switch.sh status|image|rag`. The helper derives its mode from
  live container state and REFUSES an image switch while Studio is unreachable
  (no half-switched card).
- **Prompt shape is enforced as code, not as prose rules** (analysis §22): a scene
  sentence over the word cap is cut at CLAUSE boundaries from the tail (never the
  opening setting clause, never a description hanging off a cast name — the old
  `split()[:45]` cut mid-clause and shipped `…grips a gear, Marvin.`, a name with no
  verb, which §17/§18 measure as the shape the renderer skips); a scene sentence the
  MODEL wrote that way is reported as the fixable finding `scene_dangling_name`; and
  `IMAGE_PROMPT_MAX_CLAUSES_PER_FIGURE` (profile `budget: clauses=N`) is the equal-
  treatment knob — no figure gets more description clauses than the cap, identity
  clauses never cut — **default 0/off: it is the only mechanism here that deliberately
  shortens figure blocks, and short figure blocks are what §18/§22.1 correlate with
  absence. (Painted, the capped arm lost a party member — but an UNCAPPED re-run of the
  same request lost one too, see the variance note below: single renders decide nothing.)**
- **The renderer is stochastic and the rewrite is stochastic**: request 01 painted 7/7
  and, an hour later with no change of shape or cap, 5/6. Never conclude "shape X loses
  figures" from one render of one request — repeat the arms or the finding is noise.
- **What actually predicts a skipped figure is the thinness of ITS block, not the
  length of the prompt** (§22.1: 845 tokens lost nothing, 840 tokens lost their
  antagonist, and the lost block was 43 tokens against a 99-token median). The composer
  reports it as the fixable finding `figure_thin_line` — under 60 % of the median
  figure block in an image of 3+ figures, measured AFTER the budget ladder has cut.
  Reporting it is NOT a fix, measured four ways on the prompt that loses Boss Frey: a
  line the correction turn enriched, a block moved to the front row, a block moved to
  LEAD (both with the word multiset held identical), and a block rewritten with the
  identity opener every painted figure carries — absent in all four. Position is not it.
  Do not re-derive a positional or clause-order fix for that request: the difference the
  batch shows is a character the vault describes as a BODY being painted, and one it
  describes only as clothing not being painted — a vault line, not composer logic.
- **An NPC appearance is PAINTED unless the notes about THAT character deny it** (the table's
  own species ruling, applied to appearances on 2026-11-25 after §24 measured what refusing it
  bought: nothing). `look_contradiction` cuts ONLY the clauses carrying an animal that the
  retrieved material about that character contradicts, and `_name_evidence` decides what counts
  as being about that character — a heading-owned section, or one step across `:` / `,`, never
  the whole search blob (§19's Boss Frey mess: his species came out as owl, reh and tit from
  Bubo's sheet, Colum's anchor and a Session 5 bounty). Everything else the model wrote is
  painted and reported as `look_invented`; `Composed.invented` names those figures and they are
  deliberately NOT in `Composed.ungrounded`, because a guess must not stall the GPU card
  (`_VAULT_FIX_CODES` holds no appearance codes any more). The two measured failures this
  replaces: 2026-11-24 19:11 — a `look` citing a file that had only reached the loop as a search
  snippet, 12 of 19 distinctive words missed, the WHOLE description cut including the crew
  identity the search had matched, leaving `Boss Frey: planting herself at the brink, hand raised
  in command.` (98 chars against a 447 median) that the renderer left out; §24's request 02 —
  painting the name alone let the renderer invent "a grey-furred wolf in a fine coat with a
  pistol" from the WORD "Grey", which is the invention the refusal was supposed to prevent.
  `backed_clauses` still runs and still names which clauses were retrieved, which is what makes
  a `look_invented` report honest. A correction pass still may not win a tie by thinning a
  figure: `thinnest_block` keeps the first plan when the fixable-finding count is unchanged.
- **Prompt rewriting ALWAYS happens before the switch** (the rewrite's KB agent
  tools include the reranker, which the switch takes offline).
- **…but a rewrite is not locked to a painting window**, so back-to-back `/image`
  calls land a rewrite INSIDE the previous generation's window, where dense
  retrieval is skipped and the reranker is stopped. A correction/lookup pass whose
  fix is reading the vault queues behind `_gen_lock` and runs once rag mode is
  back (`bot_core/images.py::_LookupWindow`, `IMAGE_REWRITE_LOOKUP_WAIT_S`);
  findings the model fixes from its own words never wait. Do not "simplify" this
  by moving the rewrite under `_gen_lock` — that would put the model's thinking
  time inside the exclusive GPU window.
- **Two separate clocks, and the wait consumes neither** (`run_agent_loop`
  `budget_seconds`, analysis §21): the wait is bounded by
  `IMAGE_REWRITE_LOOKUP_WAIT_S` (180 s — a person watching one Discord message,
  not a card timeout; the measured wait for a real back-to-back pair was 3 s),
  and the loop that runs afterwards starts a FRESH `deadline`, so lowering one
  must never "explain" the other. An image rewrite runs on
  `IMAGE_REWRITE_AGENT_BUDGET_SECONDS` (120 s), NOT the chat loop's
  `RAG_AGENT_BUDGET_SECONDS` (45 s): the chat cap was tuned for decision turns
  that only PICK tools, while a rewrite's last turn WRITES a plan at
  `IMAGE_REASONING_EFFORT` (~52 s median at medium) after a 5-6.5 s search round
  trip — sharing the cap made a live rewrite end in `decision turn timed out`
  with nothing slow but the model's own thinking. `run_agent_loop` also takes
  `max_iterations`/`max_tool_calls`, and it feeds those SAME numbers into the
  system addendum that announces the limits to the model: never override one
  without the other.
- **Everything runs in Docker on the host**, not in this sandbox: the bot
  container, the RAG sidecars, the gpu-switch helper *and* the Unsloth Studio
  container. The sandbox can only reach them over the network
  (`host.docker.internal`) — it cannot `docker compose` anything itself. Ask
  the user to run host-side commands (`docker compose up -d --build`, GPU
  exposure, …).

## Backlog workflow

- **`.pi/backlog.md` is the single task list.** The user drives work with "Backlog reference —
  work on these item(s)" prompts that name one or more items already marked `[working]`.
- When an item is finished, mark it `[x]` and append a completion summary (what was done, where
  it lives, verification results) — the existing entries show the expected level of detail.
- Items may be **awaiting a user decision** (e.g. #39's prioritized plan in
  `docs/internal/RAG_ARCHITECTURE_RESEARCH.md` §4). Do not implement those until the user picks.

## Data pipelines (not covered by docs/ — read the PIPELINE.md files)

- **D&D book data** (knowledge base sources): pipeline + history in
  `data/book_collection/PIPELINE.md`. Regeneration goes through `collect_books.py`; verify entry
  counts and diffs against a pre-regen backup before regenerating.
- **Magic item collection**: pipeline in `data/item_collection/` (its own PIPELINE.md). Built
  from cookie-free public sources only (DDB SSR listings, dnd-wiki.org API, Basic Rules A-Z,
  per-book gap lists). `data/items/` holds ONLY the live bot tables.
- **Known gap (user decision 2026-10-19):** ~394 header-only magic-item stubs + 6 Arcana
  Unleashed monsters whose D&D Beyond URLs redirect to the paid marketplace never got content —
  they stay as name+rarity stubs unless a logged-in/headful DDB harvest is done.

## Image prompt pipeline (read before touching `/image` or the vault's Visual Anchors)

Two paths, one outcome (`bot_core/images.py::rewrite_prompt`):

- **Primary — plan + composer.** The model answers with an `<image_plan>` JSON
  block (`prompts/image_plan_system.md`) and **`bot_core/image_prompt.py` writes the
  prompt**. Style sentence, figure-count line, identity-first clause order,
  foreground/background split, budget ladder, negation scrubbing and NPC-look
  provenance (opened file / search snippet / the model's own invention) are
  guarantees of that module, not requests to the model. Anything the composer can
  only *notice* — a collective subject, a bare repeated name, a meta aside, an
  abbreviation of a sheet name, a named character left in `entities`, a description
  stranded in the scene sentence — is a `Finding` said back to the model, never an
  edit of its prose (`FIXABLE_CODES` is that list). A named character with no verified
  appearance triggers one **enforced lookup pass** before composing again, and the
  correction pass is only painted when it composes with FEWER open findings; the
  agent loop refuses a repeated identical tool call instead of re-running it.
- **Fallback — prose.** `prompts/image_rewrite_system.md` (the pre-#84 path) runs
  only when the plan path fails.

Vault coupling: `Characters/Player Characters.md` holds each character's Visual
Anchor as **typed field lines** (`> species:`, `> body:`, `> face:`, `> outfit:`,
`> signature:`, `> quirks:`). The composer paints those verbatim; a plan's
`modify` replaces one field (an empty value removes it, naming nothing — never
write "no cloak" into a prompt). A legacy single-paragraph anchor still parses (as
an opaque blob). Keep the field labels when editing the sheet — analysis and
measurement in `docs/internal/IMAGE_PROMPT_PIPELINE_ANALYSIS_2026-11.md`.

Budgets are settings, not guesses: `IMAGE_PROMPT_BASE_CHARS`,
`IMAGE_PROMPT_CHARS_PER_FIGURE`, `IMAGE_PROMPT_MAX_CHARS`,
`IMAGE_PROMPT_FOREGROUND_FIGURES`.

Reasoning effort on the rewrite is `IMAGE_REASONING_EFFORT` (default `medium`):
applied to **every** rewrite turn (tool-picking, plan answer, lookup, prose
fallback) — measured on the live backend at `low` 42.7 s median / `medium`
52.2 s / `high` 152.8 s, with **identical** plan correctness at all three and
`medium` writing visibly more paintable detail (§12). `IMAGE_REWRITE_MAX_TOKENS`
tracks it — a thinking model spends part of the output cap on thinking and will
otherwise truncate the plan. Do not gate the effort on `agent_loop`'s per-turn
`thinking` flag: a plan answer comes out of a decision turn, so gating it there
makes the setting inert (that was the state until it was measured).

The rewrite's ANSWER SHAPE is said by code, per loop, and the image path must not
inherit the chat path's wording (`tool_prompt_block(answer_instruction=…,
mention_attached_context=False)` in `kb/tools.py`, threaded through
`run_agent_loop`). Two contradictions were measured in the shipped prompt: the
addendum demanded "plain text (no JSON)" directly under a contract demanding a
`<image_plan>` block, and it called tool results "attached user messages" (a
chat-path fact). Likewise the vault note's human-facing "Edit freely — changes
apply…" header is stripped before injection (`_strip_editor_preamble`), and the
contract's worked example uses invented names (Pell Wicker / Orin Vek / Sable Rin)
— with real campaign names in it, the model copied the example's invented NPC look
verbatim into a live plan.

**Campaign wording is DATA, never code** (user decision 2026-11-22, §19–§20). The
composer used to carry this campaign's sentences — a style line, an animal-folk floor
line, a party-sheet path, a vocabulary of species words — and to read FACTS out of
retrieved prose in order to decide whether the plan's species claims were backed. That
machinery is deleted: ~600 lines and 30 tests, because reading a fact out of a search
blob is the one kind of judgment this pipeline kept mis-executing — a `search_kb` hit
arrives as ONE JSON line, so "the block that names Boss Frey" was the whole blob and
his species came out as owl, reh and tit from Bubo's sheet, Colum's anchor and a
Session 5 bounty; `verify_look` then answered that "contradiction" by dropping a look
the rewrite had genuinely opened the file for, which is the figure-with-no-identity
shape §18 measures the renderer dropping.

What survived is the Tier A layer — string comparison against named sources, no prose
reading: paths the rewrite actually opened, sheet spellings, whether the material
writes a name bare. Name spelling is REPORTED, not rewritten (`name_spelling`,
`name_form`): the model that read the request writes the campaign's own form on the
correction turn. `entity_named_not_painted` fires only when the entity ENTRY leads with
the requested name, so "the weathered mossy stone arch of the Old Bridge" is not a
missing character.

The campaign's single sentence about what its beings ARE is `world_rule` in the
Campaign Prompt Profile (`docs/campaign-profile.md`; `IMAGE_WORLD_RULE` overrides it,
`IMAGE_ANIMAL_FLOOR` is still read as the old name): injected verbatim into the style
sentence, never budget-cut, and **never interpreted by code**. Per-character species
wording lives in the vault — its *Image Generation Policy* note and the notes about
each NPC. Do NOT reintroduce a check that decides whether a species claim is true:
where the vault is silent the model's guess is painted, by ruling. What is still not
checked at all (§15–§16): nothing compares the actors the REQUEST names against the
cast the plan paints.

`IMAGE_CODE_CORRECTIONS` (`exp/no-code-corrections`, analysis §24) is the measurement
switch, not a feature: `off` disables every composer path that CORRECTS a plan (the
unverified-`look` refusal and its clause rescue, the foreground reorder, the
sheet-spelling substitution, negation scrubbing, the label strip) and the plan-vs-plan
arbitration in `_correction_pass`, leaving only size limits (budget ladder, scene word
cap) plus the findings report the refinement turn works from. Keep it ON in production.
What §24 measured and must be remembered when touching this code: on a 20-prompt corpus
**15 prompts come out byte-identical with corrections off** — the live footprint is 4
reorders and 1 refused look — the **reorder changes content** (moving a figure reorders
which clauses the budget ladder cuts: 112 chars vs 79 for the same plan), and a refused
look does not remove the invention, it moves it into the renderer, which fills the gap
from the only word it has ("Boss Grey" name-only → a grey-furred wolf with a pistol).
One render per arm decided nothing about which arm paints better; 09 reproduced §22.1 —
leading and thick did not save the figure that last and thinner did.
