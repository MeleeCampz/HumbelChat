# Image-prompt pipeline analysis — is the design wrong, or is the prompting weak? (2026-11-18)

Trigger: after the grounding fixes, three group images still failed at the one thing
the whole rewrite layer exists for — **the right characters in the frame**:

| render | request | what the user saw |
|---|---|---|
| `01_black_pudding` | whole party vs a Black Pudding + a watcher | **Colum drawn a second time** in the background |
| `02_bandit_ambush` | whole party + Boss Frey + bandits + Alisa | **Colum missing entirely** |
| `02b_bandit_ambush` (after the anti-collective fix) | same | **Bubo missing entirely** |
| `03_forge_and_lizard` | Trixy + Marvin + Spleck | good |

Short answer: the *architecture* (rewrite on the chat backend while RAG stays live →
one GPU window → paint → restore) is sound and should stay. The *prompt layer* is a
free-form essay written by a 3-bit quantized chat model under a 45 s / 3-turn budget,
and it is carrying identity in the one way this image model binds worst: long shared
attribute lists. Meanwhile the image backend offers identity conditioning we never
asked for — and the vault data that would make it work is broken.

---

## 1. What actually runs today

`bot_core/images.py`:

1. **Rewrite** (`rewrite_prompt`) on the **chat** backend (`INFER_URL`, `:8890`,
   `MODEL_NAME=qwen3.8-flash-next-iq3_xxs-SingleGPU`), through the shared
   `agent_loop`: `RAG_AGENT_MAX_ITERATIONS=3`, `RAG_AGENT_MAX_TOOL_CALLS=6`,
   `RAG_AGENT_BUDGET_SECONDS=45`, `max_tokens=4096`, reasoning off.
2. **System prompt** = three layers re-read every generation:

   | layer | source | size |
   |---|---|---|
   | policy | `prompts/image_rewrite_system.md` | 6 990 chars (~1 150 words) |
   | character rules | vault `HumbleWood/Image Generation Policy.md` | 2 145 chars |
   | party dossier | vault `Characters/Player Characters.md` (Visual Anchors) | 4 412 chars |
   | **assembled** | | **12 564 chars (~2 032 words)** |

   The six Visual Anchors alone are **2 757 characters** (355–538 each).
3. **Shape guard** (`looks_like_image_prompt`): accepts only answers that open with the
   base style sentence; one corrective retry, else paint the raw request.
4. **Switch** the 4070 to `image` mode (reranker+embedder stop, ~12.5 GiB diffusion
   model loads; measured 12–22 s).
5. **Paint** (`_generate_one`) through the **OpenAI-compatible** endpoint
   (`IMAGE_INFER_URL/v1/images/generations`), sending exactly: `model`, `prompt`,
   `n=1`, `size=1024x1024`, `response_format=b64_json`, `negative_prompt`.
   **No seed. No steps. No guidance. No reference images. No workflow. No candidates.**
6. Batch (`generate_images`) paints N prompts in one window (~12 s/image measured at
   1024), then restores `rag` mode once.

Measured cost: rewrite 17.9–43.2 s per request; a 3-image batch = 81 s end to end.

## 2. Evidence from the live runs

| run | rewrite | structure compliance | render verdict |
|---|---|---|---|
| group, 6 figures | 34.8 s, 0 tools, 3 389 chars | anchors VERBATIM | Colum duplicated in background |
| group + 3 NPCs | 27.3 s, 5 tools, 2 157 chars | anchors COMPRESSED | Colum missing |
| re-run after anti-collective rule | 39.0 s, 3 tools, 1 958 chars | collective still used; headings absent | Bubo missing |
| re-run (Black Pudding) | ~35 s, 1 tool, ~3 300 chars | policy FORBIDS headings → headings emitted | pending |
| Trixy + Marvin + Spleck | 17.9 s, 3 tools, 1 619 chars | compliant | good |

So the same policy produces verbatim-or-compressed anchors, headings or no headings,
collective or individual subjects — **from run to run, on the same request**. That is
not a wording problem any more.

## 3. Why it fails (each mechanism measured)

1. **Identity is carried only by text, in long attribute lists.** 460–540 chars of
   comma-separated descriptors per character. A flow model at `steps=8`,
   `guidance=1.0` (this model's generation defaults — effectively no CFG sharpening)
   binds attributes loosely; long lists smear across entities.
2. **The party's palette vocabulary collides — 34 collisions.** Words shared by two or
   more characters' anchors: `gold`/`golden` (3), `blue` (3), `leather` (3), `grey` (3),
   `brass` (3), `purple` (3), `white` (3), `black` (3); Seta and Romi share `olive`,
   `brown`, `green`; Colum and Bubo share `indigo`; Trixy and Marvin share `steel`.
   Colour/material vocabulary is the strongest identity signal a diffusion model has —
   and we hand four of six characters the same one.
3. **Entity count is unbounded and the policy demands it stay unbounded.** The policy
   says include everyone, "including every requested character beats the cap". Our own
   renders: 6 named figures already fails, 11 named figures fails. Nothing in the
   pipeline caps the count or plans a composition (foreground vs background, hero +
   companions), so the model silently drops or duplicates someone every time.
4. **Two sources of truth contradict each other.** Repo policy: compress anchors when
   the scene has **more than five** characters. Vault note: compress when **more than
   six**. Both cite the same example ("the party plus a bandit" = 7). Guaranteed drift.
5. **The strongest available lever is unused — and blocked by broken data.** Studio
   reports for `unsloth/Qwen-Image-2.1-Turbo-GGUF`: `workflows: txt2img, reference,
   edit`, `conditioning.max_condition_images: 10`, `reference_resolutions: 512/1024/2048`,
   `supports_lora: true`. The vault *nominally* has canonical portraits
   (`Colum.webp`, `Seta.webp`, `Romi.webp`, `Bubo.webp`, `Marvin.webp`, `Portrait_*.webp`)
   — but **24 of 27 vault images are 131-byte Git-LFS pointer stubs**, not image bytes
   (`version https://git-lfs.github.com/spec/v1`, `oid sha256:7f551e…`, `size 417680`).
   Only the three `Session_11_moment-*.png` files are real (1024×1024 RGBA).
   Character consistency by reference conditioning is currently impossible **for a data
   reason, not a design reason**.
6. **No determinism.** We never send `seed`, although the endpoint we already call
   accepts it. Consequences: prompt changes can't be A/B'd (composition and identity
   change together), a good render can't be reproduced or varied deliberately, and the
   user has no "same scene, try again" path.
7. **The loop is open.** Nothing verifies that the requested characters appear. The
   rewriting model never sees the image; the chat backend is text-only; even this agent
   cannot view images. Today the only verifier is the user's eyes — which is exactly
   how these three failures were found.
8. **The negative prompt is the weakest possible lever here.** Studio:
   `supports_negative_prompt: true`, `negative_prompt_min_guidance: 1.0`, and this
   model's default guidance **is** 1.0 with 8 steps. Adding "duplicate of the same
   character, identical clone…" pushes against a model with almost no CFG headroom.
9. **The instruction load is too big for the rewriting model.** 12 564 chars of system
   prompt (~2 032 words), from a 3-bit quantized model, with 3 decision turns and 45 s,
   which must then also *write* a 3 000+ char answer. Compliance drift (headings,
   collectives, compression) is the expected behaviour of that setup, not an anomaly.

## 4. Backend capabilities we pay for and never use

From `/api/inference/images/status` with the model resident:

| capability | state | used by us |
|---|---|---|
| `workflow: reference` + `reference_images` (max **10**, res 512/1024/2048) | supported | ✗ |
| `workflow: edit`, `localized_edit` (annotate/paint/mask), `init_image`, `mask_image` | supported (`unified_edit: true`, `multi_image_edit: false`) | ✗ |
| `seed`, `seeds[]`, `prompts[]` (batch, up to 32) | supported | ✗ |
| `steps` (default 8), `guidance` (default 1.0), `sampler` (none exposed) | supported | defaults only |
| `loras[]` | `supports_lora: true` | ✗ |
| `detailer` (face fix) | needs inpaint pipeline + SAM 3; `supports_inpaint_only_masked: false` for this family → **must probe** | ✗ |
| `controlnet` | **`supports_controlnet: false`** → no layout/pose control | n/a |
| `hires` fix | unavailable for this family | n/a |
| prompt upsampler | not applicable (`supports_prompt_upsampler: false`) — our LLM rewrite is the only rewriting | ✓ (ours) |
| `negative_prompt` | supported at guidance ≥ 1.0 | ✓ |

Note: `prompts_hint` from the backend itself — *"Describe the image, or give an edit
instruction over the input images and refer to them as Image 1, Image 2."* — is the
style the model actually wants, and it is a shorter, reference-first style than our essay.

## 5. Data blockers (vault side, independent of code)

- **24 broken image files** (LFS pointers). Their declared sizes are recoverable
  metadata (e.g. `Colum.webp` → sha256 `7f551e30…`, 417 680 bytes): the bytes exist in
  whatever LFS repo they were copied from. Until they are real files, no portrait can be
  used as conditioning and Obsidian shows them broken too.
- **No NPC appearance data anywhere.** `Lexikon: Personen` records roles/bounties
  ("Benner Seridan: Vorgesetzter eines Banditentyps", "Godfather: ein Luma, 400 Gold"),
  never species or colour → Frey rendered fox-folk then squirrel-folk; Alisa hare-folk
  then fox-folk. Species invention is *forced* by missing data.
- **Premises the vault does not support** get invented confidently (a forge in
  Winnowing Reach; the named smithy `Geweih und Amboss` is in the Roots of Alderheart).

## 6. Options

> **Status update before reading the table:** option **C** (reference conditioning)
> measured badly against this machine's constraints and against the variant use case —
> see §9. B + E remain the recommendation.

| | change | kills | cost | risk |
|---|---|---|---|---|
| **A** | rails on the current essay: figure cap (≤4 named), one threshold for compression (fix the 5-vs-6 contradiction), guard rejects headings + wrong entity count | compression drift, contradictory rule | small | identity failures remain (measured root cause untouched) |
| **B** | **deterministic composer**: LLM returns structured parts (scene ≤45 words, one pose ≤12 words per character, NPC lookup findings, palette-disambiguation choices); **code assembles** the final prompt from the vault anchors — verbatim guarantee becomes a property of code, not model behaviour | verbatim/compress drift, party invention, headings, style drift, count line, ordering, budget arithmetic | medium (rewrite output contract + composer + tests) | NPC grounding still needs lookups; identity blending reduced but still text-only |
| **C** | **identity by reference conditioning**: real per-character portraits → `workflow: reference` + `reference_images`, prompt refers to "Image 1 = Colum Bidae…"; anchors shrink to species + 3 features | identity blending/duplication at the source | needs (i) restored portraits, (ii) native Studio client instead of the OpenAI-compat call | must probe: the API doc states explicit workflows need `init_image`; encode time/VRAM with N condition images unmeasured |
| **D** | **candidates + repair**: `seeds[]`/`prompts[]` to paint 2–3 variants per moment; `/image_fix`-style follow-up using `workflow: edit` + `localized_edit` on the previous image ("add Bubo at the left, keep the rest") | the "user is the verifier" loop, at the repair step | medium, Discord UX (3 attachments) | edit workflow fidelity unproven on this GGUF |
| **E** | **composition policy**: >4 named figures → split into 2 images (batch path already supports it, ~12 s each) or hero + background-silhouette framing | over-capacity renders, the main observed failure | small-medium | changes what `/generate_image` means for group requests |
| **F** | per-character **LoRA** trained on restored portraits | identity for good | large (training pipeline on the 4070) | heavy; only worth it if C under-delivers |

## 7. What must be probed before committing

1. `reference_images` **without** `init_image` on this build (doc says every explicit
   workflow needs `init_image`), and with `workflow:"reference"` + a neutral base —
   plus encode-time/VRAM cost with 1, 3 and 6 condition images at 1024.
2. `detailer` acceptance on this family (`supports_inpaint_only_masked: false`).
3. `edit` workflow fidelity for "add one named figure, keep the rest".
4. Whether `negative_prompt` measurably changes anything at `guidance=1.0` (fixed-seed
   A/B) — if not, the negative prompt should shrink to humans/modern objects only.
5. Fixed-seed A/B of prompt shapes on ONE 6-figure request: current essay vs
   composed-short vs reference-conditioned. Same seed → only the prompt differs.

## 8. Recommendation

Stop iterating the essay. Do **B** (the composer) now — it is the change that makes the
guarantees we currently *ask* for into guarantees we can *test* — and treat **C** as the
real fix for identity, gated on restoring the vault portraits, which is a data
recovery task the repo cannot solve by itself. Add **E**'s figure cap as part of B (it
is the cheapest way to stop the observed failure today). Keep A's threshold fix as a
drive-by. Skip F until C is measured.

**§9 revises this:** C is dropped for now; B gains an override semantics requirement.

## 9. Same-day follow-up: VRAM headroom and the "but wearing metal armour" case

Two objections from the user, both measured, both correct.

### 9.1 Reference conditioning costs VRAM on a card that has ~3.8 GB spare

`GET /api/system?refresh_memory=true` (Studio, live):

| | total | used in `rag` mode | free |
|---|---|---|---|
| GPU 0 — RTX 4070 Ti SUPER (image + RAG sidecars) | **15.99 GB** | 8.00 GB (reranker + embedder) | 7.99 GB |
| GPU 1 — RTX 5070 Ti (Strata chat) | 15.92 GB | 13.95 GB | 1.97 GB |

During an image window the sidecars stop, so the diffusion model lands on a card that is
otherwise empty: it reports `resident_mib: 12456` (12.2 GB), which leaves roughly
**3.8 GB** for activations, VAE decode and any conditioning images. A condition image is
not free — it is VAE-encoded and appended to the sequence as additional tokens, so it
costs attention and memory the same way extra prompt content does.

What is usable as the knob: `reference_resolution` (512 / 1024 / 2048) resizes each
condition image **by area**, so 512 costs a quarter of 1024 per image — six references at
512 are about the token cost of 1.5 at 1024. `memory_mode: balanced | low_vram`,
`vae_tiling` and `cpu_offload` exist as fallbacks, at a speed cost. `max_condition_images:
10` is the family's ceiling, not a budget this card can afford.

Conclusion: C is not a free feature. Any move on it starts with a measured probe (1 and 3
references at 512, then 1024) reading `/api/system` VRAM before and after each render.

### 9.2 The verbatim rule breaks on a MODIFIED character — and today it breaks badly

Request: "Seta Onix, aber sie trägt eine glänzende Metallrüstung statt ihres Umhangs".
The model did **not** copy the anchor verbatim (good instinct): it deleted the cloak,
tunic, bracers and satchel clauses. But it expressed the substitution as narration and
negation, which a diffusion model cannot execute:

- B_armour: "…adjusting shiny metal plate armor **replacing her usual cloak**" — "her
  usual cloak" refers to an outfit the renderer has no memory of.
- C_no_cape ("Colum ohne seinen Umhang"): "wearing ONLY a simple dark tunic **with NO
  cape or cloak**" — a negation written into the positive prompt, which the policy itself
  forbids, and negations in the positive prompt routinely *add* the thing.
- D_new_eyes ("Trixy ohne ihre Brille"): "brass steampunk goggles **removed from**
  forehead, revealing her bright intelligent dark eyes" — same class of instruction.

So the current design has no way to say "same character, different outfit field": the
anchor is one opaque blob, so a deliberate variant can only be phrased as a
contradiction of the blob.

**Design requirement this adds to option B:** the anchor must become **typed fields**
(species/body, palette, face features, default outfit, signature items, quirks) and the
composer must support **overrides**: the request replaces one field with a positive
description and the composer *omits* the replaced field, inheriting everything else.
"Seta in metal armour" then renders as *quokka-like ranger, sandy-tan fur, olive tunic
replaced by polished steel plate armour, recurve bow with leaf pendant* — no negation,
no "instead of", no self-contradiction. Signature items (the bow, the googly eyes, the
brass horn, Marvin's teacup body) are a separate class that a clothing override must not
touch.

This also means the **vault sheets must change shape**: one `>` blockquote paragraph
today → field lines under `## Visual Anchor`. That is an Obsidian editing decision, and
the parser should accept both shapes (old quote = one blob, treated as today).

And it is a second, independent argument against C: a reference image pins the *whole*
character including the outfit, so it makes deliberate variants harder, not easier.

### 9.3 Revised recommendation

1. **B (composer) with typed anchors and override semantics** — costs no VRAM at render
   time, and it is the only option that fixes identity drift, verbatim-vs-variant,
   headings, count and budget in one move.
2. **E (figure cap + split)** inside B: ≤4 named figures per render, bigger requests
   become two images in the same GPU window (~12 s each) — more time, not more VRAM.
3. **Seeds** (`seed`, and `seeds[]` for 2-3 candidates) — free to send, and the only way
   to A/B prompt changes or reproduce a good render. Candidates cost generations, not
   memory.
4. **C (reference conditioning): dropped for now.** Blocked by the LFS-stub portraits,
   costs the card's last 3.8 GB, and fights the variant use case. Revisit only after
   §9.1 is measured and portraits exist.
5. Drive-bys: one compression threshold (fix five-vs-six), shrink the negative prompt if
   a fixed-seed A/B shows it does nothing at `guidance 1.0`.


---

## 10. What was built (2026-11-18, same day as §1–§9)

§9's Option B, with the typed-field and override parts the measurement demanded.

**`bot_core/image_prompt.py`** — the deterministic composer (the module §5.1 said
was missing). The model now plans; the code writes the prompt.

**`prompts/image_plan_system.md`** — the plan contract: the model answers with one
`<image_plan>` JSON block (`scene`, `figures[{name,pose,modify,look,source}]`,
`entities`, `lighting`, `palette`, `mood`) and nothing else. It is asked for that
block twice at most, then the pipeline falls back to the older prose path
(`image_rewrite_system.md`, kept as the fallback and retitled as such).

**The vault party sheet became typed** (`Characters/Player Characters.md`): each
Visual Anchor is now `> species:` / `> body:` / `> face:` / `> outfit:` /
`> signature:` / `> quirks:` field lines. The parser accepts the old single
paragraph too (→ an opaque blob painted verbatim), so nothing else in the vault has
to change. `Image Generation Policy.md` gained a note saying the party sheet is
composed from and that its group-scene wording now describes the composer.

### How each measured failure from §4 is now answered by code

| # | Failure | Fix in code |
|---|---|---|
| 1 | Named NPCs invented / dropped | An NPC's `look` is painted **only** if its `source` names a file this rewrite actually read (`source_matches`, suffix-segment rule — the rule `kb/tools.py:193` got wrong). Otherwise the appearance is dropped and logged. Caught live: the model cited `SESSION_05/notes.md` for Boss Frey after making **no tool call at all**. |
| 2 | Anchor text rewritten/shortened | Anchors are parsed into fields and emitted verbatim; the model never re-emits them. `match_anchor` maps plan names to sheets. |
| 3 | Group images lose outfits | Foreground/background split written by the composer (`Exactly N named figures, F in the foreground…`), so the cast fits the budget instead of being improvised away. |
| 4 | Duplicate figures | `dedupe_figures` keys on the matched anchor name, drops parentheticals, treats a bare prefix ("Seta" / "Seta Onix") as one figure, and merges poses + overrides. |
| 5 | Prompt-length guessing | `prompt_budget(figures) = min(IMAGE_PROMPT_MAX_CHARS, base + per_figure × figures)`, and the compression score penalises **background** figures so the "smaller and softer" ones lose detail before the foreground does. Settings: `IMAGE_PROMPT_BASE_CHARS` / `_CHARS_PER_FIGURE` / `_MAX_CHARS` / `_FOREGROUND_FIGURES`. |
| 6 | Crowd nouns painted as extra bodies | `replace_collectives` swaps a determiner-anchored crowd noun for the cast's names ("die ganze Gruppe" → the nine names), or reports when it cannot swap safely. |

### Override semantics (the variant use case §9.2 measured)

`modify: {"outfit": "polished steel plate armour"}` replaces that field's clauses
and nothing else — the bow, the goggles, the face features, the size all survive.
`modify: {"outfit": ""}` or `remove_fields: ["outfit"]` paints the character
**without** that field, and says nothing about the removal (§9.3's negation lesson).
A legacy blob anchor cannot be replaced exactly; filtering approximates it and the
composer says so in the log. Unknown field names are refused with a warning.

### Guarantees that are properties of the composer, not requests to a model

Style sentence first · each named figure described exactly once · figure-count line
written by code · party members hold the foreground (a plan that listed bandits
first is reordered, and the reorder is logged) · no negation anywhere except the
one fixed `No text, lettering, logos or watermarks.` tail · budget trimmed in a
declared order (mood → palette → quirks/extra → entities → least-distinctive
clause by distinctiveness score → lighting → hard truncate) with **every cut
reported** · identity clauses (`species`, an NPC's first `look` clause) cut last.

Negation handling needed three passes, each learned from a real string:
`(no hair on the head)` must be cut **inside** its clause ("fur coloration only"
survives — that clause is a positive species marker), `NOT an animal … but a
machine` resolves to `construct — a machine`, and comma-level splitting keeps
"2 feet tall" while dropping "no visible pupils". `has_negation` had to become
word-boundary regexes: a space-anchored search misses `(no hair…)`.

### Live dry run (chat backend live, rag mode, no GPU switch, no image generated)

| Request | Result |
|---|---|
| "Die ganze Gruppe in einer Kampfszene … mit Boss Frey und zwei bewaffneten Banditen" | 9 figures, 4 foreground / 5 behind, 3584 chars, 3 clauses compressed; collective noun replaced; the three NPCs' invented looks **dropped** with a warning each |
| "Seta in polierter Metallrüstung statt ihres Umhangs, sie spannt ihren Bogen" | 875 chars, outfit replaced, cloak absent, bow + leaf-charm pendant intact, no negation |
| "Trixy ohne ihre Brille, sie lötet an Marvin" | 1416 chars, goggles gone from the outfit line, no "without" anywhere, Marvin's "construct — a machine" kept |

6-figure party + scene composes at 3308 chars under the 3600 cap with **zero**
cuts — measured, so the policy's old "six figures: identity-critical core only"
guess can go (it already did in the vault note).

**Tests**: `tests/test_image_prompt.py` (43 tests: parsing, overrides, budget
ladder, negation scrubbing, dedupe, collective replacement, source verification,
plan extraction) plus `tests/test_images.py` adapted to the two paths. Full suite
**1456 passed**.

### Still open
- Painted images in a real GPU window for the three dry-run requests (§9.1's
  before/after comparison).
- NPC Visual Anchors in the vault (`Characters/NPC Visual Anchors.md`) — until
  then grounded NPCs like Frey are painted by name, which is honest but plain.
- Whether Qwen-Image-Turbo's *positive* prompt really ignores negations (§9.3's
  A/B) — the composer no longer emits them, so this is now a vault-negative-prompt
  tuning question only.


## 11. Second review pass (same day, after the enforcement landed)

Four more measured failures, all found by running the real rewrite path rather
than reading the code.

**11.1 Invention through a file that was really opened.** The bandit rewrite
wrote "heavy-set badger-folk bandit chief, flat snout, cutlass at his hip" for
Boss Frey and cited `SESSION_01_DerAnfang` — which it *had* read, and which
mentions him by name and says nothing about his looks. "Opened ⇒ believed" was
the weak point. `verify_look` now separates the two claims: a creature noun must
appear in the retrieved text **about that character** (`evidence_about`: the
sections of the retrieved material that name them), in English or German. The
first version of that gate was itself fooled — the model searched *"Frey bandit
badger wolf fox species appearance"* and the vault answered with a D&D
magic-item entry about a badger, so the species was present in the context but
not about Boss Frey. Section-scoped evidence closes that. Re-measured: the same
request now paints Frey and Alisa with real actions and **no species at all**,
which is what the vault actually supports.

**11.2 A character hidden in `entities` was invisible to every rule.** The plan
put "Boss Frey and his bandit crew" and "Alisa Pennygleam" in the scene text.
They skipped grounding, skipped the lookup pass, and the count line said *six*
named figures while eight were named in the same prompt — the exact shape that
paints an extra crowd. A name in `entities` that the scene treats as an actor is
now a figure (pose taken from the scene's own words, group left as a group); a
name the scene only ever says after "at/of/in" stays a place.

**11.3 Duplicate tool calls, not tool count, were the cost.** The 145 s rewrite
of the ambush request spent **4 s** in tools. The rest was model turns, and the
model had re-issued the same `read_section` three times. `agent_loop` now
refuses the second identical call in a loop with a pointer to the result it
already has. The same request went 145.7 s → 21.6 s with that and the retrieval
fixes together.

**11.4 Thinking: measured, not assumed.** Same hard request, same code, only the
reasoning effort changed:

| every rewrite turn | wall time | tool time | tool calls |
|---|---|---|---|
| `low` (today) | 21.6 s | 13 s | 6 |
| `medium` | 27.7 s | 8 s | 6 |

`medium` wrote marginally better *queries* and marginally nicer nouns, and bought
nothing that grounding rules would not have caught anyway. That pair was a single
run each and is **superseded by §12**, which repeats it.

**11.5 Considered and rejected: cutting a leading roll call.** A scene sentence
that opens "Trixy, Seta, Colum, Romi, Bubo, Marvin, Benner Seridan battles a
black pudding" wastes ~110 characters and re-lists the cast. Cutting the list
would leave a verb without its actors, and the actors are exactly what the figure
lines then describe — the fix would have quietly changed *who* the scene says is
fighting. Left alone on purpose.

**Still open after this pass.** The plan can also *lose* something the user
asked for: the ambush request said "Boss Frey und seinen Banditen" and the final
plan named only their boss. Nothing in the composer compares the request's own
actors against what the plan decided to paint — a completeness check there is the
next obvious guarantee, and it is not yet built.


## 12. Reasoning effort, measured with repeats (2026-11-18)

§11's low-vs-medium pair was one run each — noise. This is 12 runs: 2 requests ×
3 efforts × 2 rounds, the second round in reverse order so backend warm-up cannot
favour one level. `/tmp/reasoning_ab.py`, results `/tmp/reasoning_ab.json`.

| effort | request A (8 named beings) | request B (7 figures) | overall median | model turns as share of wall time |
|---|---|---|---|---|
| `low` | 42.8 / 57.0 s | 23.7 / 42.6 s | **42.7 s** | 87 % |
| `medium` | 33.5 / 58.1 s | 46.3 / 84.9 s | **52.2 s** | 89 % |
| `high` | 164.5 / 167.6 s | 82.6 / 141.1 s | **152.8 s** | 97 % |

`low` and `medium` are inside each other's spread; `high` is 3.6× and consistent
about it. Tools cost 2–10 s in every run — the time being bought is model turns.

**What effort levels did not change** (identical at `low`, `medium` and `high`):
invented species for NPCs (the composer caught 4 / 5 / 3 attempts), a named
character hiding in `entities`, the number of figures painted, how many clauses
the budget cut, and whether the enforced lookup pass had to run. Those are
guarded by code, and thinking does not substitute for them.

**What `medium` did buy** — six override requests run at both levels
(`/tmp/override_ab.py`), correctness checked from the composed text:

| case | `low` | `medium` |
|---|---|---|
| clothing swap, signature item must survive (Seta's bow) | correct — "thick fur-lined heavy winter coat in warm russet-brown with cream fur trim" | correct — "…with a high standing collar, heavy quilted seams and a plush pale-fur lining visible at the cuffs and hem" |
| removal the user asked for (Trixy's goggles → three-lens spectacles) | correct — "brass three-lens spectacles" | correct — "brass spectacles with three round lenses **set in a triangular frame**", "adjusting a tiny gear **between her claws**" |
| costume swap that must not change the species | correct | correct |

So: same decisions, more **paintable specificity** — the kind of detail a diffusion
model can actually act on. Cost: ~+9 s on a hard group rewrite.

**Decision taken.** `IMAGE_REASONING_EFFORT` (allowlist as the other reasoning
families) defaults to **`medium`** for the plan/lookup/prose turns; the routing
turn stays thinking-off, since it only picks tools. `IMAGE_REWRITE_MAX_TOKENS`
tracks it — 4096 at off/low, 6144 above — because a thinking model spends part of
the cap on its thinking phase and had previously truncated a party prompt (2048 →
671 chars). `high` stays available but is not the default: 3.6× for no measured
gain.


## 13. The replacement itself invented a duplicate (found by running `medium` live)

First live rewrite after the setting landed, 24.7 s, all guarantees fired — and the
scene sentence read:

> `Seta Onix, Bubo Krallensturm, Romi Stachelpfote, Trixy Smoldersome, Marvin, Colum
> Bidae, Boss Frey and Alisa Pennygleam confronts Boss Frey and his bandits…`

Two defects in one substitution, both introduced by the collective replacement this
pass exists to make safe:

1. **It re-named characters the sentence already named.** Boss Frey appears as both
   subject and object — the composer itself produced the "paint him twice" shape it
   was built to prevent. Names already present in the sentence are now excluded from
   the inserted list.
2. **`confronts` stayed singular.** The verb-agreement map was hand-written and
   therefore always missing the verb this particular request happened to use. It is
   now generated from ~70 verb stems (`_sforms` handles `-es`/`-ies`), keeping the
   curation principle (an unknown word after the list is left alone) without hand-
   listing every s-form.

Re-run of the same request: *"…Seta Onix, Romi Stachelpfote, Bubo Krallensturm and
Trixy Smoldersome, with Marvin beside her, **confront** Boss Frey across a blocked
path while Alisa Pennygleam clutches her coin bag."* One mention, plural verb.

§11.5's open item stands unchanged: nothing yet compares the **request's** own actors
("Boss Frey und seinen Banditen") against the figures the plan chose to paint.


## 14. The knob was inert until it was measured against the turn map

An observer run (no patching, just recording the effort each completion actually
used) showed a whole rewrite — plan, no tools — running in **one turn at `off`**.

`agent_loop` runs its tool-picking turns with `thinking=False` by design, and a plan
answer comes out of exactly such a turn. The completer had been gating the effort on
that flag (`"low" if thinking else "off"`, and then `"off" if not thinking else …`),
so the configured level reached only fallback / forced-answer / nudge turns — the rare
path. The setting was inert on the common path, and §12's numbers described a
configuration the code did not have (every turn raised together).

**Corrected:** `IMAGE_REASONING_EFFORT` now applies to every rewrite turn alike —
tool-picking, plan answer, lookup pass, prose fallback — which is what §12 varied.

Measured under the shipped setting, efforts logged per turn:

| request | before (`off` on decision turns) | after (`medium` everywhere) |
|---|---|---|
| single figure, an override ("Trixy ohne Goggle…") | 2.7 s, 1 turn | **8.2 s**, 1 turn — "leaning close over a half-assembled clockwork mechanism and turning tiny lenses into place with brass tweezers" |
| group, 8 named beings, lookup pass triggered | ~43–57 s | **43.7 s**, 7 turns (all medium), 10 tool calls costing 17.8 s |

The group rewrite is dominated by tool turns and the enforced lookup pass, not by the
reasoning level; the single-figure case is where the level shows up as latency — and as
the extra concrete action in the scene sentence.

**Loop bounds that shape the turn map** (`RAG_AGENT_*`): code defaults 2 iterations /
6 tool calls / 45 s per loop; production `.env` sets `RAG_AGENT_MAX_ITERATIONS=3`. When
iterations run out the loop adds one **forced final** completion over the gathered
transcript. The enforced lookup pass is a SECOND loop with its own bounds — which is why
one rewrite can log seven completions.

---

## 15. The rewrite PROMPT itself, measured against the prompt it replaced (2026-11-19)

§1–§14 fixed the code around the model. This pass went at the prompt the model reads —
and the honest headline is that **the A/B changed nothing about coverage**, while
teaching three things that turned out to be missing from the code.

### 15.1 What the prompt is made of (measured as sent)

| component | before | after |
|---|---|---|
| `prompts/image_plan_system.md` (the JSON contract + rules) | 4,635 | 5,245 |
| campaign rules injected from the vault note | 1,658 | 1,977 |
| party roster rendered from the sheet | 3,359 | 3,359 |
| tool addendum (`tool_prompt_block`) | 3,427 | 3,094 |
| the request as delivered | 139 | 160 |
| **system + user total** | **13,184** | **13,801** (+4.7 %) |

The vault note grew (the beast-folk rule was rewritten: 480 → 802 chars) and its
**611-char human-facing header stopped being injected** — "Edit freely — changes apply
to the next image an image request agent generates from this vault. The bot re-reads
this file before every image request." is guidance for the person holding the vault,
not for the model, and it had been going into the system prompt on every rewrite. The
addendum shrank because the image loop no longer inherits the chat path's paragraphs.

**The contradiction that was removed**: the addendum said *"When you have enough
information, answer in plain text (no JSON) and the result is delivered to the user
instead."* — directly under the contract demanding a `<image_plan>` JSON block, and
under a paragraph claiming tool results are *"already attached to this conversation as
user messages"* (a chat-path fact: in the agent loop they arrive as `tool` messages).
Answer shape is now said by code, per loop: `tool_prompt_block(answer_instruction=…)`
with the image path passing its own instruction, and `mention_attached_context=False`.

Three more prompt changes, each from a failure seen in a plan dump:

- **`<request>…</request>` tagging** (§7's instruction-following finding: a labelled
  zone earns higher compliance than raw text).
- **The contract's example no longer names the campaign.** It described `Frey` (an
  invented NPC look) and *used a real party member, Trixy, as the worked override
  example*. Rewritten with fictional names — Pell Wicker, Orin Vek, Sable Rin — under
  a header that says the names are invented and none of them may be copied.
- **`<previous_image_plan>` memory**: a follow-up in the same channel ("same one, but
  with metal armour") previously saw only its own sentence. The last plan composed for
  that channel is replayed (≤512 chars, 12 channels, in-process, tool-free — same
  shape as `session_images`'s session hint). Proven live: the second request named no
  character and the composer still painted 7 figures, Seta Onix among them.

### 15.2 The A/B: 12 rewrites, both prompts, order reversed in round 2

| request | new prompt | old prompt |
|---|---|---|
| bandit ambush (8 names + a "70 gold" amount) | figs 8, 8 · chars 3587, 3596 · tools 7, 2 | figs 8, 8 · chars 3551, 3575 · tools 7, 11 |
| black pudding (2 NPCs + creature) | figs 7, 7 · chars 3591, 3577 · tools 5, 5 | figs 7, 7 · chars 3555, 3580 · tools 6, 5 |
| single figure + outfit override | figs 1, 1 · chars 1194, 1185 · tools 0, 0 | figs 1, 1 · chars 1403, 1167 · tools 0, 0 |
| **aggregates (n=6 each)** | median 49.7 s · 6.5 turns · 3.5 tools · 3582 chars · 5 budget cuts | median 79.1 s · 7.5 turns · 5.5 tools · 3553 chars · 5 budget cuts |

**Figure counts, budget cuts and collective handling are identical between variants in
every pair.** The wall-time gap is a half-second-per-turn difference in tool turns
under a shared GPU-offline queue and is not a claim: the spread inside each variant
(10.8–130.2 s) is larger than the gap between them. What the prompt shape changed is
*what the model writes into the plan*, measured next.

### 15.3 The one behavioural difference measured: the old example was a ready-made answer

Same request ("…die befreite Händlerin Alisa Pennygleam … Boss Frey und seinen
Banditen"), plans dumped from both variants:

- **old prompt**: `"name": "Boss Frey", "look": "heavy-set badger-folk bandit leader,
  dark grey fur with pale shoulder stripes, flat snout, pointed ears, scarred brow,
  patched black coat, serrated cleaver", "source": "…/SESSION_01_DerAnfang/notes.md"`,
  plus `"Alisa Pennygleam": "plump squirrel-folk peddler, russet fur, cream belly, big
  amber eyes, round wire spectacles, patchwork travelling coat, bushy striped tail"`.
  The first six words are copied out of the contract's own example. Neither vault
  passage describes either character.
- **new prompt**: `"name": "Frey", "look": ""`, with Alisa and the bandit crew pushed
  into `entities` — no invented appearance anywhere in the plan.

Both variants' *final prompts* were clean, because the species gate caught both
inventions — the fix chain holds. But the prompt was pointing at the failure: an
example that supplies a plausible NPC look is an example the model reuses for a live
NPC the vault never describes.

### 15.4 What the run exposed that no code enforced (all three now enforced)

1. **Species claims about a CROWD were unpoliced.** The gate checked `look` clauses
   and figure `species` fields; a scene sentence saying `"two lanky weasel-folk
   bandits with pointed ears, long snouts and russet fur, wearing patched leather
   coats and carrying jagged cleavers"` and an entity line `"two rat-folk bandit
   lackeys with jagged cleavers and iron-shod boots"` sailed through. Same rule, now
   applied to the scene sentence and the entity line: a `-folk` compound must be
   traceable to retrieved text *that names these characters*, and the request's own
   words are always allowed (the composer exists to stop the model inventing campaign
   facts, not to argue with the person running the table). Live after: no species word
   survives anywhere in either composed prompt; "bandit" survives 4×.
2. **An NPC's name drifted.** The plan wrote `"name": "Frey"` for the character the
   request and every session log call **Boss Frey**, so the prompt painted `Frey:`.
   Renamed when three facts agree: the request names the character, the material the
   loop read attaches a word to that name, **and the vault never writes the plan's form
   bare**. That third test is what the first live run got wrong — it renamed "Alisa
   Pennygleam" to **"Händlerin Alisa Pennygleam"**, because a session note says "die
   befreite Händlerin Alisa Pennygleam" (German capitalises every noun; a merchant is
   not part of her name). The vault writes her name bare a dozen times, so the rename
   is now refused.
3. **Party members were abbreviated in the scene sentence.** One composed prompt had
   figure lines reading `Seta Onix … Romi Stachelpfote … Trixy Smoldersome` and a scene
   sentence reading `Seta, Bubo, Romi, Colum, Trixy` — the same §4 mechanism (a name the
   renderer has not been told about is a name it paints on its own). Now the sheet's
   spelling is applied wherever the name appears — scene, entity lines, and poses —
   including a leading-token abbreviation, but only when no two sheet entries share
   that prefix.

Live composed scene after all three: *"In a dim misty forest clearing, Seta Onix, Romi
Stachelpfote, Bubo Krallensturm, Colum Bidae, Trixy Smoldersome and Marvin stand
confronting boss Frey; Alisa Pennygleam clutches a heavy leather coin pouch close to
her chest."* — with `plan abbreviated 'Bubo' for 'Bubo Krallensturm', …` logged as a
warning, so the user can see the guarantee firing.

### 15.5 Still not enforced (the completeness gap, restated with data)

Across the 12 A/B rewrites and the live verification, the request's own crowd —
"Boss Frey und seinen Banditen" — reached the prompt as *a phrase in the scene
sentence* every time, and as *named figures* only when the model happened to enumerate
them (one run produced `"First Bandit"`, `"Second Bandit"` and 10 figures). Nothing in
the pipeline compares the actors the REQUEST names against the cast the plan chose.
That guarantee needs a reading of the request's own nouns, which is the part of this
problem that has not been built. Also open: the enforced lookup pass returned no plan
once (`keeping the first (Frey, Alisa Pennygleam painted by name)`) — the fallback
works, but those two stay ungrounded.

Tests: 1477 → **1502** (14 prompt-shape tests, 11 species/name/spelling tests).

---

## 16. The species rule was pointed the wrong way (table owner's ruling, 2026-11-19)

Asked what to do with the name/species work, the answer came back as a rule about the
world rather than about the pipeline: **"we only want animals in the final output. So if
there are 'bandits' I want them in the final picture. For those unspecified characters,
inventing a species that makes sense in the setting is OK."**

That reverses the direction of the gate built in §15. It policed *invention*; the failure
it prevented (an undescribed bandit painted with no species) is the failure the renderer
commits by painting **people**.

### 16.1 What the gate decides now

`species_verdict()` returns two lists instead of one:

| case | before (§15) | now |
|---|---|---|
| vault says these characters are rat-folk, plan says weasel-folk | stripped | **stripped** (unchanged — the vault's canon wins) |
| nothing describes them, plan says weasel-folk | stripped | **kept**, and logged as the bot's own guess |
| species found only in a block about someone else (a badger-shaped magic sword) | stripped | still not allowed to *name* them; if the block about **them** names a different animal, that one wins |
| the request itself says "as a badger-folk" | kept | kept (the composer does not argue with the table) |

Proximity still decides *whose* evidence counts (`evidence_about`, `_named_blocks`) — only
inverted: an absence is now permission rather than refusal. A contradicted look is still
dropped whole, so a wrong species cannot smuggle the rest of an invented description in.

Reported rather than silent: `Boss Frey: given a species (badger) the vault does not name
— kept, because an animal-folk guess is what keeps humans out of the picture`, for scene
sentences, entity lines and figure looks alike. An invented species is canon the table
should know the bot invented.

### 16.2 The animal-folk floor

`IMAGE_ANIMAL_FLOOR` — one sentence, appended to the **style** line, which the budget
ladder is not allowed to cut: *"Every figure in this image is an anthropomorphic
animal-folk character of this world."*

Unconditional on purpose. Deciding which entries in `entities` are beings and which are
wagons is the shape-guessing this pipeline has already refused twice (§15), and the world
rule is unconditional in the vault too. Cost measured: 80 characters, ~7 % of a
single-figure prompt's 1,194, and it survives a hard cap of 320 characters while mood,
palette and quirks are cut around it.

### 16.3 What got looser, stated plainly

Test `test_a_species_claim_needs_backing_even_when_the_file_was_opened` (§10's drift
catch) had to be rewritten: Boss Frey's `"heavy-set badger-folk bandit chief with a flat
snout and a cutlass at his hip"`, cited to a session note the rewrite really did open, is
now **painted**. The species claim no longer blocks it, and an opened file is still
believed outright. What still protects the campaign is the ordering: the vault's own
species wins, a contradicted look is dropped whole, a figure with no citation at all is
painted by name only, and every invention is logged.

Vault note (`Image Generation Policy.md`) and the plan contract
(`prompts/image_plan_system.md`) were rewritten to the same rule — "give every creature
you describe a species; the vault's wins; a bare role word paints a person" — because the
prompt and the note previously told the model the opposite of what the code now allows.

Tests: 1502 → **1508** (net; 6 rewritten to the new policy, 6 added: contradiction-wins,
guess-kept for a crowd, guess-kept for an undescribed NPC, unrelated-chunk guard,
floor-present, floor-survives-budget, floor-switchable-off, vault-backed-not-reported).

**Still not built**: the floor guarantees that whatever is painted is animal-folk. It does
not guarantee that the actors the *request* names are in the frame — "seinen Banditen"
reaching the image is still the model's choice, not a code property.

## 17. Ten evaluation rounds, and why the scripted language rules are being retired (2026-11-19)

`/tmp/d_run_rewrite.py` runs the same five live requests through the full rewrite
+ composer stack and re-evaluates the output after every code change. Rounds 1–10,
`/tmp/d_run/round{1,2,7,8,9,10}.json`.

| round | what the prompts got wrong | what was changed |
|---|---|---|
| 1 | a negation survived the tail; the name-mentioned-twice check counted any repeat; a collective subject slipped through; the entity crowd was budget-cut out | entity protection, check() split into scene-vs-figure accounting |
| 2 | "the party" still reached a pose line | pose-line collective check |
| 4–5 | **frog-folk bandits** — a D&D familiar-list chunk answered a crowd question | bare animal words only count where they modify a crowd word |
| 6 | **lizard** answering a bandit question via "lizard merchant Spleck" | `_stray_name`: species glued to a name belongs to that name |
| 7 | "birdfolk-folk"; hedgehog leaking from Romi's anchor into the bandits | double-suffix guard; `_frame_species()` exclusion |
| 8 | the agent's OWN search query ("Frey species badger rat lizard") read back as vault evidence | `images._vault_evidence()` strips tool-call echoes |
| 9 | a chief and his crew flattened to one animal; choice flipped between runs | number agreement + attestation counting |
| 10 | **bear-folk** from "regal **bear**ing" (a window sliced a word in half); the merchant's lizard stamped on the bandits in the same sentence | whole-word window growth; `_claim_supported` (per-claim gate, not yet wired) |

### 17.1 The measurement that matters: none of these were model failures

Every defect above was a **text-matching** defect in code: a window boundary, a stem
match across languages, a capital-as-name detector (German capitalises every noun), a
tool-call echo, an adjacency window reaching the wrong sentence. Ten rounds produced
ten patches, and each patch fixed one live case while leaving the class intact. The
vocabulary itself had to absorb `dame`, `mann`, `leute`, `wesen`, `frau`, `weib` —
Humblewood's German compounds — which is not a guarantee, it is a campaign.

### 17.2 Separating the guarantees that are arithmetic from the ones that are reading

| class | guarantee | mechanism | keep? |
|---|---|---|---|
| structure | plan is JSON; figure-count line; style sentence first; foreground ordering; per-figure budget ladder; typed anchor fields + overrides; our own layers write no negations; no lettering tail | schema + arithmetic | **yes** |
| provenance | this look cites a file the loop opened; this claim came only from a search snippet; this figure cites nothing | filesystem facts | **yes** |
| language judgement | which species a crew is, whether a crowd noun names the cast, whether a name is abbreviated, whether a parenthetical is pipeline commentary, whether a mention is a duplicate | ~40 regexes + word lists | **no — becomes a lint report** |
| campaign text in code | `_ANIMAL_FLOOR`, the scale sentence, `_BASE_STYLE`, `_COLLECTIVE_PATTERNS`, `_GENERIC_SPECIES`, the party-sheet path | constants | **no — moves to a profile** |

### 17.3 The decision (owner's ruling, 2026-11-19)

> "I am not a big fan of the scripted enforcement. Using code to judge natural
> language is always a possible pain point. I want this bot to be usable in another
> campaign setting; currently we do a lot of overfitting."

Chosen architecture:

1. **Campaign profile (data)** — style sentence, animal-folk floor, negative tail,
   budget numbers, foreground count, party-sheet path. Read from the vault / profile
   file, not from `bot_core/`. A new campaign is a new file, not a code change.
2. **Composer (code)** — assembly and structure only: counts, ordering, budget,
   overrides, provenance verification. It **never rewrites prose**. It emits findings.
3. **Rewrite model (LLM)** — owns species, appearance consistency, name spelling and
   phrasing, with the vault in front of it, and gets the findings back as **one
   correction turn** before the prompt is finalised.

What is deleted rather than kept-and-gated: species strip/substitute,
`replace_collectives`, `drop_collectives`, `drop_bare_repeat_mentions`,
`campaign_names`/`request_spelling`, `strip_meta_asides`. They survive as *finding
codes* the model is asked to fix.

What this costs honestly: species correctness now depends on the model plus the
vault instead of on regex, and a finding costs one extra model turn (~10–60 s) only
when something is flagged. What it buys: the composer stops silently editing the
table's canon in ways the table never asked for, and the same code runs against a
DarkSun or Greyhawk vault without adding that setting's vocabulary to Python.

## 17.4 — After the deletion: what the three layers actually do (live re-runs, 2026-11-19)

The scripted layer is gone. The same four hard requests, live model, no GPU, after
the campaign profile landed and the findings turn was wired:

| request | what came out | cost |
|---|---|---|
| group + named NPC | 6 figures, 4+3, every name in its own sentence, Boss Frey as an entity line with a **grounded** look, his two thugs badger-folk | 54–63 s |
| override + group | 7 figures: Seta in crimson plate cuirass (override), Romi bare-backed, Bubo with a lantern, Spleck with her wagon **as an object** and her lizard-folk look, Marvin as an automaton | 60–87 s |
| machine + crowd | 2–3 figures, machine floor line in force, cart as an object, the two thieves kept (a look citing a **search snippet only** was dropped and reported) | 34–52 s |
| species conflict | 8–9 figures, 4+4/5, Boss Frey **lizard-folk** (the user's own word — exempt), two bandits promoted by the model into figure lines | 50–54 s |

No crowd noun anywhere; no invented retag of a party member; the count line matches
the figure lines every time. **The rewrites cost the same as before the deletion**
(34–87 s, one or two model turns) — so the scripted layer was not buying speed either.

What the composer still guarantees by code: layout, the count line, dedupe, no
negation in clauses it assembles, identity spelling on the figure lines, provenance
of an NPC look, foreground ordering, the budget ladder and its report.

What it now only *reports*: species claims, collective subjects, name spellings,
meta-notes, repeated mentions, a character filed as an object. Those go back to the
model as one turn (`finding_feedback`), and the second plan is what gets painted.

Open measurement for the next visit: with the anchors consolidated the prompts sit at
3,574–3,597 chars against the profile's `max=3600`, so the ladder is now compressing
13 clauses per image. **`max=` is a profile edit** — no code change needed to find out
whether the model holds 9 figures at 4,200.
