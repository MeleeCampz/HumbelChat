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
