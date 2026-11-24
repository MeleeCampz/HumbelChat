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

## 18. The renderer reads a figure block in order, and the order was wrong (2026-11-19)

The complaint that started this: **"bubo is not visible in the first image"** and
**"in the second one he and trixy merged into one character"** — Round 13, painted
with the three-layer design working correctly. Both prompts named Bubo Krallensturm,
both gave his Visual Anchor in full, both had 8 and 7 figure lines. Nothing was
missing from the plan.

### 18.1 It is not a token ceiling

Prompts were measured with the renderer's own tokenizer (`Qwen/Qwen3-VL-8B-Instruct`,
the text encoder Qwen-Image 2.1 ships), and a ladder was painted at 2/4/6/8/10/12
figures (`2026-11-19_ceiling/`):

| figures | tokens | painted |
|---|---|---|
| 2, 4, 6, 8 | 184 → 587 | every one ✔ |
| 8 (Round 13 prompt) | 855 | **7 — Bubo absent** |
| 10 (descriptor-first) | 701 | 9 — Marvin absent |
| 12 (descriptor-first) | 810 | **all 12** ✔ |
| 12 (shipped shape) | 921 | 13 — one character painted twice |

Eight figures at 587 tokens is clean; eight at 855 loses one; twelve at 810 is clean.
**Length does not predict which figure disappears.** Qwen-Image 2.1 is also not
truncated at 1024 tokens the way Qwen-Image 1.0 was.

### 18.2 It is not the amount of detail

One real plan (request 01, 8 figures), the six Visual Anchors complete, the **budget
ladder switched off so nothing was cut** (`2026-11-19_shape_ab/SHAPES.md`):

| variant | tokens | what differs | result |
|---|---|---|---|
| shipped, ladder ON (control) | 859 | 12 clauses cut | 7/8 — Bubo absent |
| **P1** — same shape, nothing cut | 925 | nothing | **7/8 — Bubo STILL absent** |
| P2 — identity bound to the name, name-free scene | 889 | two shape changes at once | 8/8 |
| P3 — every character name removed | 846 | names only | 7/8 — Bubo lost **and his star robe migrated onto Romi** |
| P4 — P2 without the row-lead phrases | 869 | row leads only | 6/8 — Marvin and Alisa absent |

Cutting nothing did not bring Bubo back. Removing names made things worse and moved
his clothing onto somebody else. More detail is not the lever.

### 18.3 Six runs, one change at a time

The two candidates were run twice each (the renderer is not seeded) and the two
sub-changes they combine were pulled apart (`REPEAT.md`):

| run | shape | Bubo | all figures |
|---|---|---|---|
| R1a, R1b | shipped shape, nothing cut | **absent in both** | 7/8, 7/8 |
| R2a, R2b | identity bound to name + name-free scene + new row lead | present in both | 8/8, 7/8 |
| **R5** | shipped scene, shipped row lead, **species moved before the pose** | present | **8/8** |
| R6 | shipped blocks, scene stops naming the cast | **absent** | 6/8 |

R5 keeps the shipped name-list scene **and** the shipped `Behind them, smaller and
softer,` lead and paints the whole cast — so neither of those is the culprit, and a
name-free scene alone (R6) fixes nothing.

### 18.4 Confirmation: the same words, a different order

`confirm_paint.py` takes **`compose_prompt`'s own output** and moves the identity
clause to the front of each block, asserting the two prompts have the **same word
multiset** (`sorted(re.findall(r"[\w-]+", …))`) and the same length — 3868 chars /
925 tokens in both arms for request 01, 3691 / 868 for request 02:

```
shipped   : Bubo Krallensturm: staff planted, crescent finial flaring light,
            a small owl archmage-cleric, robed from the neck down, …
candidate : Bubo Krallensturm: a small owl archmage-cleric, robed from the
            neck down, staff planted, crescent finial flaring light, …
```

| arm | runs | whole cast painted | Bubo |
|---|---|---|---|
| shipped shape | Round 13 01, P1, R1a, R1b, R6 | **0 of 5** | absent in all 5 |
| identity first | R5, C1, C2, C3 | **4 of 4** | present in all 4 (staff, crescent-and-orb finial, indigo star robe, crescent pendant) |

Fisher exact on 4-of-4 against 0-of-5: **p = 1/126 = 0.008**. The second request
(black pudding) points the same way: its control lost the owl entirely and painted
his staff into Trixy's hands, while `D2_identity_first_02` has Bubo holding his own
crescent staff.

What the experiment also settled, by accident (the first harness attempt introduced
every character twice, and **still did not paint two bodies**): a character named
twice in a prompt is not automatically painted twice — the risk is that the second
introduction **supresses** the first. That is why "one introduction per character"
survives as a report and not as a rewrite.

### 18.5 The change

`bot_core/image_prompt.py::_figure_text` now assembles a block as
**identity → pose → the rest**, where identity is the `species` field or, for a
figure with no anchor, the verified `look`:

```python
identity_field = next((f for f in _IDENTITY_FIELDS if fr.fields.get(f)), "")
clauses = list(fr.fields.get(identity_field, []))
if fr.pose:
    clauses.append(fr.pose)
for f in (*ANCHOR_FIELDS, *(k for k in fr.fields if k not in ANCHOR_FIELDS)):
    if f != identity_field:
        clauses.extend(fr.fields.get(f, []))
```

The budget ladder is untouched — it decides *which* clauses survive, this decides the
order the survivors are said in. A `species` override inherits the position
(`test_an_identity_override_leads_the_block_like_the_field_it_replaces`), a figure
with no identity at all keeps its pose first rather than being given a guess
(`test_a_figure_with_no_identity_at_all_still_leads_with_its_pose`), and one test
reads the composed prompt back into blocks and asserts the sheet's own species line
is the first thing said after every name.

Live rewrite re-run (same five requests, no GPU): every block of every prompt now
opens with its identity — `Bubo Krallensturm: A small owl archmage-cleric, robed from
the neck down, raising the crescent staff at the doorway, …`. Zero extra tokens, zero
extra model turns.

### 18.6 What that re-run then exposed, and the three fixes it forced

The composer's *reports* were next to fail, twice:

1. **A finding that contradicted its own evidence.** A scene finding said *"the
   material about those beings says nothing about their species"* while the verdict
   that produced it had read a span containing "birdfolk": `lint_plan` ran
   `crowd_evidence_spans(…, names + anchors)` for the ruling and
   `vault_animals(…, retrieved)` — a different call, different filtering — for the
   quotation. One string is now computed per claim and read by both sides
   (`_said_animals(evidence)` takes the evidence itself).
2. **A description stranded in the scene sentence.** Boss Frey is not in the party
   sheet, the model wrote `"look": ""`, and the prompt said
   `Boss Frey: raising a heavy battle axe overhead` — while its scene sentence said
   *"Boss Frey, a badger-folk bandit chief, stands alone …"*. Carrying that
   description across would be editing the model's language, so it is reported:
   `identity_left_in_scene` quotes the apposition back (`_scene_description`, bracket
   walking: name, comma, text, comma) and asks for it in the figure's own `look`.
3. **A correction pass that made the plan worse.** Asked to move that description,
   the model answered with a plan containing a meta aside the renderer would have
   painted — `(species not stated in the vault)` — plus a `not`, plus a species
   finding: two problems answered with four. `_rewrite_composed` now composes both
   plans and paints the one with **fewer open findings**, and logs the substitution.

Suite after all of it: **1536 passed**.

### 18.7 What this did not fix

* `D2_identity_first_02` has Bubo correct but paints **Seta and Romi as one figure**
  (one body carrying the bow, the quiver, the rose and the horn). The identity-first
  order reduces introductions competing for one body; it does not make them impossible.
* Nothing yet compares **the actors the request names** against **the cast the plan
  paints** (§15.5 remains the open gap).
* Boss Frey, Alisa Pennygleam and Benner Seridan have **no appearance in the vault**.
  The composer reports them every time; the fix is typing `> look:` lines in
  `Player Characters.md`, which is the table owner's edit, not code.
* The prompt still sits at 3,580–3,600 chars against the profile's `max=3600`, so the
  ladder cuts 4–18 clauses per image. `max=` is a profile edit, still untested at 4,200.

## 19. The evidence was attributed to the wrong character (live dump, 2026-11-20)

§18.7 said the remaining defects were data gaps. They were not all data gaps. The
deferred-lookup probes (§18.6) kept reporting `figure_ungrounded [Boss Frey]` even
when the rewrite had opened files for him, and the explanation turned out to be
**evidence attribution** — the composer handing the species check text that was not
about the character it named.

`/tmp/evidence_audit.py` captures the exact `retrieved` string the composer is given
and runs the composer's own functions over it. One live rewrite of *"Boss Frey stellt
sich der Gruppe auf der Old Bridge in BURGGRUND…"*:

| `retrieved` kept "about Boss Frey" | value |
|---|---|
| characters | 6,453 (47 lines) |
| characters actually naming him | 460 |
| his name appearing in the block | 23 times |
| other names inside the same block | 9 (Alisa Pennygleam, Bubo, Benner Seridan, Seta, Romi, Trixy, Marvin, Luma, Meadowin) |
| species words harvested from it | **owl, reh, tit** |

None of those three animals belong to him. `owl` came from Bubo Krallensturm's own
Appearance section, `tit` from Colum Bidae's Visual Anchor, `reh` from a bounty posted
in Session 5 (`Godfather: Ein Luma…`). A search result arrives as **one line** —
`[{"file": …, "snippet": …}, {"file": …}]` — and `evidence_about` splits on blank
lines, so a JSON array of six hits is one block. "the block that names them" is then
just "the blob", and the blob is the whole search.

**What it destroyed.** `verify_look`'s contradiction gate reads
`contradicted_species(look, evidence_about(name, retrieved))`, so a rewrite that had
read about Boss Frey and written `look: "stocky badger-folk bandit chief"` lost its
look: `look_unverified … material about THIS character says birdfolk, reh, wolf — that
look names their species`. The figure then reached the prompt with nothing but its
name, which is exactly the shape §18 measures the renderer dropping or merging into a
neighbour. One false species attribution bought a second, worse defect.

**The fix** (`species_evidence_about`): a species fact about a character has to sit in
the same **phrase** as their name.

* a section whose HEADING names them counts, heading to bottom, sub-headings included;
* otherwise only the phrase around each occurrence — cut at `. ; ! ?` and quotes,
  extended across one `:` or `,` because that is how a description attaches to a name
  (`- **Frey**: rat-folk bandit chief`, `Boss Frey, a badger-folk bandit chief`);
* the period is a wall: "…Frey bandits, Alisa Pennygleam rescue. A wolf track leads
  into the forest, and at the wood edge the party heals a wounded wolf" no longer says
  Frey is a wolf.

Same rewrite, same dump:

| character | old species words | new species words |
|---|---|---|
| Boss Frey | owl, reh, tit | — (the vault never states his species) |
| Alisa Pennygleam | owl, tit | — |
| Benner Seridan | owl, reh, tit | — |
| Bubo Krallensturm | owl, tit | **owl** (his own sheet) |

`evidence_about` stays what it was for `figure_ungrounded` and
`crowd_evidence_spans`, where "near him" is the right question and no fact is being
read out of it.

### 19.1 A species inside a figure NAME was using the wrong verdict

The live dump named two figures `badger-folk bandit 1` and `badger-folk bandit 2` and
the old code reported `species_contradicted … the material does not back it, a `-folk`
is a claim about their kind and this world has no humans`. That is the ALLOWED case
wearing the refused case's label: §16 allows an animal-folk guess for a being the vault
never describes, and the composer's own rule for a `look` says the same. The label
matters because it is what the correction turn is told to fix, so the model was being
asked to rename figures out of the guess the animal floor exists to force. A name claim
is now read by the same `species_verdict` a `look` gets, with the claim's ROLE sentence
as extra evidence — `crow-folk bandit` is contradicted when the vault calls that bandit
crew `rat-folk` (the vault describes a character through their crew), and reported as
`species_unbacked` when the vault never discusses them.

### 19.2 A place the user named became a character the plan "left unpainted"

Live: `entity_named_not_painted [the weathered mossy stone arch of the Old Bridge] —
the user asked for 'old bridge' and this files them under objects`. The entity line
DRESSED the scene with the bridge; `_request_names` saw the user's own words and
`e.split(",", 1)[0] in asked` matched a substring inside a description, so the composer
demanded that a stone arch be given a pose. The test is position: the plan uses the name
as the entry (`Boss Frey and his bandit crew`) or only mentions it inside something
else's description. Fixed; the character case is unchanged.

Residual, stated rather than hidden: an entity entry that is a **bare** requested name
(`Old Bridge` alone) is still reported, because nothing in the entry distinguishes a
place from a person. It costs a correction turn, never an image.

### 19.3 What the live re-run says after the fix

Same request, same files, one rewrite: `species_contradicted` and `look_unverified` are
gone, and the identical claim is reported truthfully —

> `species_unbacked [scene]`: badger-folk is claimed in the scene sentence and the
> vault states no species for those beings — kept, because a species guess is what
> keeps humans out of the frame, but it is canon the bot invented

Still open from this run: `identity_left_in_scene [Boss Frey]` — the model again put his
description in the scene sentence instead of the figure's `look`, so his block is
painted with a pose and no identity. Reporting it has not made the model stop doing it.

## 20. The species layer was deleted, not repaired (2026-11-22)

§19 fixed evidence attribution and the layer still did not earn its keep. The table's
ruling: **the bot is a general-purpose tool, so no part of it may decide what another
campaign's world contains.** A human-world campaign would have needed the code changed
in four places before a single image could be painted.

### 20.1 The two tiers, and which one survived

Everything the composer checked fell into two classes, and the split is the whole
decision:

| | What it compares | Verdict |
|---|---|---|
| **Tier A — string facts against a named source** | did this rewrite open this path (`read_paths`); does a search snippet contain the phrase; does the material write this name bare or only inside another name; does the entity entry LEAD with the requested name; is there a negation in a paintable clause; is the crowd noun in the scene the one the request used | **kept.** A path, a spelling and a position are checkable and were never wrong |
| **Tier B — a fact read out of retrieved prose** | is this species backed for THIS character; does the vault deny it; is this animal word about the crowd or about somebody named three lines away | **deleted.** §19 is the measurement of how often this misfires, and the fix for one case (heading-owned sections, phrase adjacency) just moves the failure to the next phrasing |

Tier B is not one check with a bug; it is a reading-comprehension task, and the pipeline
already has a reader for it — the rewrite model, in the turn where it holds the vault's
text and can be told to fix what the composer reports.

### 20.2 What was removed

In `bot_core/image_prompt.py` (734 lines out, 44 module-level names): the species
vocabulary (`_SPECIES`, `_species_variants`, `_all_species`, `_GENERIC_SPECIES`,
`_canonical_species`, `_SPECIES_COMPOUND`, `_SPECIES_COMPOUND_SUFFIXES`, `_claim_nouns`,
`_FUNCTION_WORDS`), the attribution machinery from §19 (`species_evidence_about`,
`evidence_about`, `_name_claim_evidence`, `_unit`, `_UNIT_BREAK`, `_adjacent_spans`,
`_segment_keys`, `_name_keys`, `_blocks`, `_named_blocks`, `_clause_start`,
`_clause_end`, `_CLAUSE_BREAK`, `_SEGMENT_STOP`), the verdicts (`species_verdict`,
`_species_named_in`, `_species_near`, `_claimed_species`, `_stray_name`,
`contradicted_species`, `unbacked_species`, `invented_species`, `species_guesses`,
`request_species`), the crowd half (`crowd_evidence_spans`, `crowd_evidence`,
`crowd_adjacency`, `crowd_keys`, `_crowd_spans`, `vault_animals`, `_said_animals`), the
machine detection that picked between two campaign sentences (`is_machine`,
`_MACHINE_WORDS`, `_living_floor`), and the species half of `verify_look`. In
`bot_core/images.py`: the species sentences in the party dossier and the campaign
wording in the built-in fallbacks (§20.4). Three finding codes — `species_contradicted`,
`species_unbacked`, `species_conflict` — are gone from `FIXABLE_CODES`, so the correction
turn is never told to argue about a species. 34 tests retired with them.

What `verify_look` still decides is provenance alone: opened file → search snippet →
the model's own invention (painted, reported as `figure_ungrounded` /
`look_unverified`). That is Tier A.

### 20.3 What replaced it: one line of campaign data

`world_rule:` in the Campaign Prompt Profile — a single sentence, injected verbatim into
the style sentence where the budget ladder never cuts it, and **never read by code**. It
replaces `animal_floor` + `animal_floor_living` (both names are still parsed, the first
wins) and `IMAGE_ANIMAL_FLOOR` (`IMAGE_WORLD_RULE` is the new name, the old one still
works). The old pair existed because the composer detected a construct in the frame and
swapped sentences; the sentence now covers constructs in its own words:

> Nobody in this world is a human being: every figure is an anthropomorphic
> animal-folk character, or a construct painted as the machine it is

This campaign's wording stays editable in Obsidian, and a campaign that has no such
claim deletes the line — the code has no fallback sentence, because a fallback sentence
would be this campaign's claim wearing a default.

### 20.4 The campaign wording that was hiding in fallbacks

`prompts/image_rewrite_system.md` (the prose fallback) opened with a sentence naming
Humblewood and repeating the style sentence, and `images.py` shipped built-in defaults
that were Humblewood canon: "no humans — only animals or humanoid beast-folk", and
"Googly eyes … belong ONLY to Colum Bidae". All of it is out. The fallback now carries
`{{OPENING_STYLE}}`, filled at load from the profile (`style` + `world_rule`,
`images._opening_style`), so both rewrite paths open a prompt with the profile's
sentence and neither has a campaign in the repo. The missing-vault-note default is a
neutral marker.

### 20.5 What is now deliberately unpoliced

- A plan may call the bandits weasel-folk while the notes call them rat-folk, and
  **nothing in the pipeline notices**. The only defence is the vault's own text.
- The vault's *Image Generation Policy* note still tells the model "every creature in
  this world is beast-folk" — that is where the campaign states its facts now, in the
  prompt the model reads, not in a checker.
- A bare role word ("bandits") is still painted as written. The profile's `world_rule`
  sentence is what keeps humans out of the frame, and it is a sentence, not a gate.
- `identity_left_in_scene` (§19.3) is unchanged and still the open behaviour problem:
  reporting that a description is stranded in the scene sentence has not stopped the
  model doing it.

The accepted trade, stated plainly: the checks Tier B removed caught a handful of real
mislabels over the D-run, and produced at least three wrong verdicts that dropped a
verified look or burned a lookup turn (§19). Removing them costs the mislabels and buys
back the wrong verdicts, the campaign coupling, and ~700 lines of reading prose in a
module whose job is assembling text.

## 21. Two clocks: the lookup wait, and the loop that runs after it (2026-11-22)

§14 fixed WHERE a mid-painting lookup runs. Two probes then measured how long it may
take, and the numbers disagreed with the knobs as written.

### 21.1 What the probes measured

Live against the real chat model, the real KB tools and the running sidecars; only the
GPU switch simulated (`images._switcher_enabled = lambda: False`), so the code under
test still believed the card was painting:

| measurement | value |
|---|---|
| wait a real back-to-back pair actually needed | **3 s** (window opened 56 s in, closed 59 s) |
| `search_kb` round trip in rag mode | 4.95 / 5.80 / 6.53 s |
| `search_kb` round trip during image mode (lexical-only) | 0.96 / 1.00 s |
| a rewrite that hit the agent loop's cap | `agent fallback (decision turn timed out)` at **45 s**, with zero tool calls pending |

The last row is the finding. The cap is `RAG_AGENT_BUDGET_SECONDS=45`, which is the
CHAT loop's budget — tuned for decision turns that only pick tools. An image rewrite's
last turn is different work: it WRITES a plan at `IMAGE_REASONING_EFFORT` (measured
~52 s median at medium, §12). A 45 s cap can be consumed entirely by one answer the
model is thinking, and the loop then answers from a single-turn summary of a
conversation it barely had. The tools were never the slow part: 1–6.5 s per round trip
against a 45 s budget.

### 21.2 The two knobs

- **`IMAGE_REWRITE_LOOKUP_WAIT_S`: 900 s → 180 s.** The first draft bounded the wait by
  "as long as a card could plausibly be painting". The wait is paid by a person watching
  one Discord message, and the measured need was 3 s. Past the cap the lookup runs
  lexical-only, which still finds the sheet — what it loses is the reranker's ordering.
  A lookup that needs minutes of waiting is a lookup the vault should have answered with
  an appearance line (§16).
- **`IMAGE_REWRITE_AGENT_BUDGET_SECONDS` (new, 120 s)**, passed as `run_agent_loop`'s
  `budget_seconds` at all three image-rewrite call sites (plan pass, correction/lookup
  pass, prose fallback). Chat RAG keeps its 45 s. `run_agent_loop` also gained
  `max_iterations` / `max_tool_calls` overrides, and those flow into the system
  addendum that ANNOUNCES the limits — an override that moved the applied cap but not
  the announced one would make the harness break a rule it had just told the model.

The wait consumes neither budget: `_LookupWindow` closes before `run_agent_loop`
computes its deadline, so a rewrite that waited 180 s still gets its full 120 s to
think. That is deliberate — a shared clock would make the wait cap silently shrink the
lookup's own budget, and the two are unrelated waits.

Worst case for one `/image`, stated so nobody has to derive it:
120 (rewrite) + 180 (wait) + 120 (lookup) + 600 (`IMAGE_GEN_TIMEOUT_S`) = 1020 s, and
the progress message narrates the waiting part.

## 22. The two live prompts that dropped figures, read as STRUCTURE (2026-11-24)

Two `/image` calls on 2026-11-24 (dev.log 13:47 and 13:56) came back with characters
missing, and the images are not kept on disk — so the prompts were recovered from the
log and re-run offline:

| prompt | composed | figures | clause spread | per-figure token spread |
|---|---|---|---|---|
| `die ganze Gruppe auf der old Bridge gegen Boss Grey` | 3599 chars / 837 tokens | 6 | 10–17 | 93–124 |
| `Boss Grey, der Anführer der Banditen auf der Old Bridge` | 3591 / 833 | 7 | 4–15 | 43–119 |

`/tmp/live_repro.py` rebuilds both plans from the composed text and re-composes them
**byte-identical** to what was sent (the ladder's control arms assert it every run), so
everything below is a property of the prompt that was actually painted, not of a guess
about it.

**A correction to my own first reading.** The first structural pass over these prompts
mis-parsed figure blocks and reported "12 named figures" with named NPCs (Hexe Susan,
Lucky, Fisher Quickfingers…) that are not in either prompt — the count line says
`Exactly 6` and `Exactly 7`, and none of those names appears in the text. Every number
in this section was re-derived from the prompt strings themselves.

### What the two prompts actually do wrong

1. **A cast name with nothing said about it.** Prompt 02's scene sentence ends
   `…Trixy Smoldersome grips a gear, Marvin.` — the name is the last word of the
   sentence and no verb belongs to it. The model wrote that itself (45 words, exactly
   at `_MAX_SCENE_WORDS`, so no trimming was involved).
2. **A second way to make the same shape, and this one is ours.** The budget trim was
   `scene.split()[:45]` — keep the first 45 words. Cut mid-clause, that keeps a name
   and drops its verb, which is defect 1 manufactured by the composer.
3. **The thinnest figure is also the worst-shaped one.** `Boss Grey: planting himself
   at the far end of the bridge, blocking the crossing.` — 43 tokens, the only block in
   either prompt with **no identity clause**, and the **last** block. §18 measured that
   a block whose identity clause does not lead loses its figure (5/5 shipped-shape runs
   lost one, 4/4 identity-first runs lost none, Fisher p = 0.0079), and the 22-run
   corpus adds ~15 % drop rate for the final block. Boss Grey had both.
4. **Unequal treatment.** Every party member arrives with 10–17 clauses and 93–124
   tokens; the one figure the vault has nothing about gets 2 clauses. The clean runs in
   §17 describe every character at the same depth (~50–70 tokens each) and painted 8/8
   at 587 tokens and 12/12 at 810.

### Landed (Tier A — structural, campaign-neutral, tested)

- **`_trim_scene`** (`bot_core/image_prompt.py`): a scene over the word cap is cut at
  *clause* boundaries from the tail inwards, and never takes the opening clause (the
  setting — figure lines repeat the cast, never the place) or a clause hanging off a
  cast name (an apposition is an unanchored character's only appearance). The sentence
  is re-joined with the model's own punctuation, and every dropped clause is named in
  the warning. Commit `f4df3c7`.
- **`scene_dangling_name`**, a fixable finding for defect 1: the composer does not edit
  the model's prose, and a finding is what the correction pass is built to receive.
  Kept distinct in code and in tests from `identity_left_in_scene` (a figure with no
  appearance line that *had* a description in the scene) — two defects, two codes.
  Commit `02a19db`.
- **`_strip_pipeline_label`**: a Visual Anchor written as `> look: scarred snout` — the
  word `look` is this pipeline's own field name, so it was being painted as a word
  inside the image prompt. The content stays, the pipeline word is dropped, and the
  swap is reported. A label the pipeline has never heard of (`> wheels: …`) is the
  sheet's own vocabulary and stays verbatim.
- Suite: **1541 passed**.

### Built, measured offline, and deliberately left OFF

`IMAGE_PROMPT_MAX_CLAUSES_PER_FIGURE` (profile: `budget: … clauses=N`), default **0 =
off**. It is *equal treatment*, applied before the budget ladder and independently of
it: while a figure holds more clauses than the cap, its least distinctive clause comes
out, even when the total budget is already met. Identity clauses are untouchable — a
cap that reached into them would fix a spread by deleting a character.

On the two live prompts (control = the prompt that was painted):

| arm | chars | tokens | clauses/figure | per-figure tokens |
|---|---|---|---|---|
| 02 control | 3591 | 833 | 4–15 | 43–119 |
| 02 cap 8 | 3133 | 737 | 4–11 | 43–88 |
| 02 cap 6 | 2850 | 673 | 4–9 | 43–75 |
| 01 control | 3599 | 837 | 10–17 | 93–124 |
| 01 cap 8 | 2977 | 699 | 9–11 | 77–101 |

The cap does what it claims to the TEXT: the spread narrows, the thinnest figure keeps
everything it had, and the total drops ~12–17 % while the figure count never changes.
**It is not shipped on** because the claim it needs is visual, and the last time a
plausible prompt theory was shipped on text evidence alone it was wrong (§17's colour-
collision theory). The harness is ready and its control arms are byte-identical to the
failures: `data/image_probes/2026-11-24_structure_ladder/manifest.json`, runner
`/tmp/structure_ladder.py --paint` (4 arms for prompt 02, one GPU window, ~9 min).

### What no code can fix

Boss Grey has no appearance in the vault (§16), so his line can only ever be pose plus
whatever the rewrite found in a file it actually opened. `identity_left_in_scene` and
`figure_ungrounded` both fire on exactly this shape, and §21's lookup window is what
gives that lookup a card to run on. The remaining fix is a line in the vault.

### 22.1 The same requests painted, and the cap measured against paint (2026-11-25)

Nine requests were run through the LIVE rewrite path (chat model + KB tools + vault,
rag mode live, 335 s) and their 11 prompts painted in ONE GPU window (258 s, ~19 s per
image): the two 2026-11-24 Discord requests, the clause-cap arms for the two densest
ones, and a spread — close-up, two-figure interaction, two NPCs the party sheet says
nothing about, a collective request, a creature-only request, NPC-against-the-party.
Run dir `data/image_probes/2026-11-25_sample_range/` (PNGs + `REVIEW.md` with the
per-image verdicts).

| prompt | composed | figures | what came back |
|---|---|---|---|
| 01 `die ganze Gruppe … gegen Boss Grey` | 3590 ch / 845 tok | 6 + Boss Grey as an entity | **7 bodies**: all 6 PCs, each with its own signature items, plus a large anonymous brute |
| 01 cap 8 | 2909 / 691 | 6 | **Colum Bidae missing** — 5 PCs + the brute |
| 09 `Boss Frey stellt sich der Gruppe …` | 3594 / 840 | 7 | **6/7 — Boss Frey missing** (43-token block against a 99-token median, and LAST) |
| 09 cap 8 | 3138 / 744 | 7 | **6/7 — Boss Frey missing** (he was already under the cap; it cut the others) |
| 02 `Boss Grey …` | 996 / 215 | 1 | 1/1 ✓ — but painted human-sized, against the profile's scale sentence |
| 03 Seta close-up | 1414 / 325 | 1 | 1/1 ✓ bow, quiver, hooded cloak, bracers, satchel all present |
| 04 Romi + Marvin at the oven | 1630 / 368 | 2 | 2/2 ✓ including the interaction (the pliers change hands) |
| 05 Fisher Quickfingers | 998 / 223 | 1 | 1/1 ✓ painted from the lookup's words (`mapach … raccoon-like beast-folk`) |
| 06 Hexe Susan | 1048 / 250 | 1 + an entity | 1/1 ✓, and the vault's phrase "heavy bosom" came back as a bare-chested figure |
| 07 the group fleeing a coal shaft | 3587 / 832 | 6 | 6/6 ✓ and the flight reads as motion |
| 08 black-pudding octopus | 884 / 190 | 0 named | creature painted ✓ (asked for torchlight, got cold window light) |

**The cap is off, and this is why.** Its claim was textual (spread 93–124 / 43–119 →
43–88). Painted, the capped arm of 01 lost a party member and the capped arm of 09
changed nothing — and the figure that vanished in both 09 arms was already *under* the
cap. Kept as a knob (`IMAGE_PROMPT_MAX_CLAUSES_PER_FIGURE`, default 0) with that result
recorded, so nobody re-derives the text argument and ships it.

**The variance caveat, which this batch proved the hard way.** Request 01 painted 7/7 in
its first run (`01_discord_group_bridge.png`) and **5/6 when the same request was
rewritten and painted again** an hour later (`12_group_vs_grey_v2.png`: Seta Onix gone,
the antagonist gone, a second Trixy-like bandit in her place) — with no cap and no shape
change at all. Single renders of a single request therefore cannot settle a "this shape
loses figures" claim on their own; only repeated arms can. That weakens the cap verdict
from "the cap removed a party member" to "the capped arm lost one and the uncapped re-run
of the same request lost one too". The cap stays off for the stronger reason: it is the
only mechanism here that deliberately shortens figure blocks, and shortening figure
blocks is what §18 and §22 correlate with absence.

**The length of the prompt is not the variable.** 01 (845 tokens) lost nothing; 09
(840 tokens) lost its antagonist. The variable is the shape of that one block.

**New report, no new enforcement**: `figure_thin_line`. After the budget ladder has
done its cuts the composer measures each painted figure block, and when a figure holds
under 60 % of the median block length in an image with three or more figures it reports

> `figure_thin_line [Boss Frey]: this figure gets 211 characters of description while
> the others in the same image average 421; in a crowded image the thinnest figure is
> the one the renderer leaves out — give it the same weight, saying only what the notes
> you opened actually say`

It is **fixable** (the correction turn is the only thing that can add description,
because only the rewrite model has read anything about the character) and it is **not**
in `_VAULT_FIX_CODES` — it asks for more of what was already read, not another lookup.
In this batch it fired on 09 and on 09 alone: the one prompt whose figure was missing,
and no clean prompt.

**That next measurement, run the same evening, came back negative.** Four further arms of
the same request (`data/image_probes/2026-11-25_sample_range/`, files `10_`, `13_pos_*`,
`14_opener.png`):

| arm | what it changed | Boss Frey painted? |
|---|---|---|
| `10_boss_frey_v2` | fresh rewrite with the report live — the correction turn read it and enriched his line 175 → 209 chars | no |
| `13_pos_control` | control repeat of that prompt | no |
| `13_posA_front3` | his block moved into the FRONT group, demotion phrase handed to the last block; word multiset asserted identical | no |
| `13_posB_lead` | his block leads every figure line; word multiset asserted identical | no — a second Trixy-like bandit appeared in his place |
| `14_opener` | his line rewritten into the shape every painted figure here has: `Boss Frey: A small fox-folk bandit chief, …`, identity-first | no |

Position is ruled out for this prompt (2/2 arms with the words held identical), the
identity opener is ruled out, and enrichment through the report did not save him: four
absences, with all six anchored party members present in every one. What the batch did
paint was Boss Grey, twice, and he is described with a BODY — a badger-folk brute, a
fox-folk in a long coat — where Frey's whole description is clothing and posture. The
sample is small and the mechanism is not proven; what is measured is that the composer
cannot talk the renderer around a character the vault never describes. `figure_thin_line`
stays as a report, with this written into its docstring: naming the figure at risk is not
the same as being able to save it.

**Two vault-wording findings, deliberately not fixed in code.** The scale sentence in
the Campaign Profile painted nothing on its own (02 came back human-sized) — the size
that lands is the size the character's own anchor states, which is why six anchored PCs
looked right and an unanchored chief did not. And 06 proves the other half of the same
rule: the vault's words are painted, literally, including "heavy bosom". Both are
vault-line fixes.

---

## 23. Two live requests came back with the NPCs wrong (2026-11-24)

> Correction of the first draft of this section: an earlier version of §23 quoted two
> request texts ("…wenn Boss Frey und seine Schergen angreifen", "…wenn Boss Frey
> angreift") that do NOT appear in the logs, and attributed the party-sheet citation to the
> session notes. The reconstruction below is read line by line out of `logs/dev.log` and
> every quoted string is in those lines.

Two `/image` calls ran on the evening of 2026-11-24. Neither lost a figure COUNT — both
final prompts carry "Exactly 7 named figures". Both lost the NPC's *appearance*, by two
different mechanisms, and only one of the two losses was caused by the composer's own
verification code.

**Request A — 19:07:03, `Boss Grey stellt sich der Gruppe auf der old Bridge`**
(dev.log 32282 → final prompt 32345, 3595 chars, 7 figures).

- First composition (19:07:46, 32300/32302): `identity_left_in_scene [Boss Grey]` — the
  rewrite described him as "a grey-furred humblefolk bandit chief" **in the scene
  sentence** while his figure line carried no `look`; and `figure_thin_line [Boss Grey]`
  at 107 characters against a 445-character median.
- The lookup pass ran 3 tool calls and fixed it. Final block (19:09:29):
  *"Boss Grey: a fox-folk bandit boss, a vixen well dressed as if she had influence,
  commanding toll bandits, planting herself mid-bridge, arms spread to block the crossing."*
- **Nothing was removed here.** The thin-block finding was still open at the end (202
  against 450, 32326) but the character was painted with a description. This is the shape
  §18 fixes, and the fix worked.

**Request B — 19:11:45, `Die Gruppe wird von Banditen überfallen als sie gerade vor einem
Abgrund stehen`** (previous plan injected; dev.log 32365 → final prompt 32398, 3595 chars).

- The request names no NPC. "Boss Frey" arrives through the `<previous_image_plan>` memory
  of request A and the vault's Old Bridge crew.
- First composition (19:12:51): his `look` PASSED verification and was painted — 215
  characters against a 420-character median (`figure_thin_line`, 32382). The budget ladder
  also cut the crew clause *"a hulking fox-folk brute called The Muscle"* from the entity
  line (32384) while compressing party body-clauses elsewhere.
- The correction/lookup pass ran **0 tool calls** (32394) and answered with a plan whose
  `look` cited `Characters/Player Characters.md`. That file reached this rewrite only as a
  **search snippet**, never opened, so `verify_look` applied the word test: 12 of its 19
  distinctive words were not in what had been retrieved (32385). The whole appearance —
  including the parts that WERE in the snippet, "fox-folk bandit" being the crew the
  search matched — was dropped.
- Final block, painted at 19:14:05 (32398):
  *"Boss Frey: planting herself at the brink, hand raised in command."*
  **Pose only.** 98 characters against a 447-character median (32393).

So the answer to "did the code-based verification remove it?" is: **yes, for request B** —
and the removal happened in the composition the *correction pass* produced, which is the
composition that got painted. For request A, no: verification never touched the
description; the earlier problem was placement, and the lookup pass repaired it.

**And the correction pass won with the worse plan.** The comparison shipped in §17.3 keeps
the first plan only when the correction *increases* the fixable-finding count. Here both
compositions carried 2 fixable findings (first: `collective_in_figure` + `figure_thin_line`;
after: `look_unverified` + `figure_thin_line`) while the figure went from 215 characters to
98 — a tie, so "not worse" let the thinner plan through. A tie is not an improvement, and on
this pipeline an unimproved correction that thins a figure is a regression with the volume
turned down.

**The two shipped changes.**

1. **A half-read look keeps the clauses the notes back.** `backed_clauses()` splits the
   look at clause boundaries and judges each one against what was retrieved: the backed
   clauses are painted, the rest are refused and named in the report, and the figure gets
   `look_partially_backed` (fixable, and a vault-fix code — the fix for it is opening the
   file, so a correction that needs it waits for the card, §16). Whole-look verification is
   what cost request B his crew identity: judged whole, a look is believed or thrown away;
   judged clause by clause, the model's half-correct reading stops costing the character
   everything. An entirely unbacked look still goes, unchanged.
2. **A correction that fixes nothing may not thin a figure.** `Composed.block_chars`
   exposes each figure's character count and `thinnest_block()` picks the smallest; when the
   correction pass leaves the fixable-finding count where it was and the smallest block gets
   smaller, the FIRST plan is painted and the swap is logged. A correction that genuinely
   reduces findings still wins even if a block shrinks — the rule only closes the tie.

**What is still open.**

- The entity-line budget cut "a hulking fox-folk brute called The Muscle" from the crew line
  while party clauses were compressed instead, and no rule yet says a named character
  outranks an unnamed crowd clause.
- Request B asked for no NPC and got one, from the previous plan's memory. The injected
  `<previous_image_plan>` is deliberate follow-up memory (§15) and this is its first
  observed side effect: an NPC carried forward into a scene its reader never asked for,
  painted from a description nobody retrieved.
- The composer still cannot write a description it does not have. Both requests ended with
  the NPC's block thinner than the party's; only request A's ended with one the vault
  backed.

**Tests.** `tests/test_image_prompt.py`: a half-read look paints its backed clauses and
reports the rest; a wholly unbacked look still goes whole; `block_chars` /
`thinnest_block` agree with the reported thin figure. `tests/test_images.py::
TestCorrectionKeepsTheFatterPlan`: a correction that only thins is not painted, a correction
that thickens the thin figure is.
