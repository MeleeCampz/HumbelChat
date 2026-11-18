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
