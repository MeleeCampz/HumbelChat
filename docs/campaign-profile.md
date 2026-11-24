# Campaign Prompt Profile — the file that makes the bot *your* campaign

`/image` is built in three layers, and only one of them is allowed to contain your
campaign's wording:

| Layer | Where | What it may say |
|---|---|---|
| **Campaign wording** | `data/knowledge/Campaign Prompt Profile.md` | your sentences, your numbers |
| **Composer** | `bot_core/image_prompt.py` | structure only: ordering, dedupe, figure-count line, foreground split, budget ladder, reports |
| **Rewrite model** | `prompts/image_plan_system.md` + the vault's *Image Generation Policy* note | the plan it writes, and the correction turn that fixes what the composer reports |

The composer holds **no campaign text of its own**: no style sentence, no sentence
about what kind of beings exist, no party sheet path, no vocabulary of species words.
Those all come from this file. A second campaign setting is a different vault and a
different profile, not a code change.

## The file

```
<KB_PATH>/Campaign Prompt Profile.md          # or IMAGE_PROMPT_PROFILE=/abs/path
```

Typed lines, the same shape as a Visual Anchor (plain `key: value` lines also work):

```
> style: Semi-realistic fantasy illustration — detailed fur and feather texture…
> world_rule: Nobody in this world is a human being: every figure is an…
> negative: humans, watermark, signature, modern objects, extra limbs
> party_sheet: humblewood/HumbleWood/Characters/Player Characters.md
> budget: base=1100 per_figure=420 max=3600 foreground=4 clauses=0
```

| Field | What it does | Env override |
|---|---|---|
| `style` | Opens every prompt; also the structural test that a rewrite is a prompt, not narration | `IMAGE_PROMPT_STYLE` |
| `world_rule` | One sentence about what kind of beings this world has. Said verbatim, in the style sentence, where the budget ladder never cuts it. The composer neither reads it nor reasons about it | `IMAGE_WORLD_RULE` |
| `negative` | The tail every prompt ends with | `IMAGE_NEGATIVE_PROMPT` |
| `party_sheet` | The sheet whose `Visual Anchor` field lines are painted verbatim, and the roster injected into the rewrite's system prompt. Empty = no anchors: the model describes everyone it names | `IMAGE_PARTY_SHEET` |
| `budget` | `base`, `per_figure`, `max`, `foreground`, `clauses` (per-figure description cap, 0 = off) — see [configuration.md](configuration.md) | the matching `IMAGE_PROMPT_*` numbers |

**Precedence:** an environment variable that is actually set → the profile file → the
neutral defaults (`docs/configuration.md`). A missing line gets no sentence; there is
no fallback wording in code for a campaign the code does not know.

**Live editing:** `load()` re-reads the file on every rewrite, exactly like the party
sheet. An edit in Obsidian applies to the next image, not to the next restart. An
unknown field is logged as ignored, never guessed at.

## Writing `world_rule` well

It is the only place a campaign says what its beings ARE, and it is painted next to
your Visual Anchors, so it must not contradict them:

- this campaign: `Nobody in this world is a human being: every figure is an
  anthropomorphic animal-folk character, or a construct painted as the machine it is`
  — Marvin's anchor says "brass-and-steel automaton", so the sentence names a
  construct instead of claiming every figure is animal-folk.
- a human-world campaign: write your own sentence, or delete the line and the prompt
  simply gets no such claim.

Until 2026-11-20 the profile had `animal_floor` **and** `animal_floor_living` lines and
the composer chose between them at runtime depending on whether a figure in the frame
was described as a machine. Both names are still read (`animal_floor` wins) so an old
profile keeps working; the runtime choice is gone, so the sentence has to cover that
case in its own words.

## Moving to a different campaign

1. Put the new vault under `KB_PATH` (`docs/rag.md`, `docs/obsidian-vault.md`).
2. Rewrite this file: `style`, `world_rule`, `party_sheet`, `budget`.
3. Give the new vault its own *Image Generation Policy* note — per-character rules
   (what a character always wears, what must never be painted on them) belong there,
   not in code (`docs/commands.md`, `/image`).
4. Party sheet in the same block shape: `# Name` → a free line → `## Visual Anchor`
   with `> species:`, `> body:`, `> face:`, `> outfit:`, `> signature:`, `> quirks:`
   field lines. Field names are yours; the composer only cares that they are typed
   lines it can replace one at a time.
5. `characters.json` for the Discord persona(s) — `docs/characters.md`.

Nothing in `bot_core/` needs to change, and nothing in `bot_core/` decides what your
world's inhabitants are.
