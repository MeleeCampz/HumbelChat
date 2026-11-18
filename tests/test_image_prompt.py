"""Tests for bot_core.image_prompt — the deterministic prompt composer (#84).

Why this module exists (docs/internal/IMAGE_PROMPT_PIPELINE_ANALYSIS_2026-11.md):
six measured image failures were all cases of asking a 3-bit model to obey a
policy and hoping. Each test names the failure it now locks shut:

1. "die ganze Gruppe" painted 2 characters + an unnamed crowd.
2. "Seta in Metallrüstung statt ihres Umhangs" → 'plate armor REPLACING her
   usual cloak' — a contradiction a memoryless renderer cannot resolve.
3. "Colum ohne seinen Umhang" / "Trixy ohne ihre Brille" → negations written
   into a POSITIVE prompt, describing the very thing to leave out.
4. Spleck entered twice (as "Spleck" and "Spleck (the cat-folk)") → two wolves.
5. An 11-figure request squeezed into mush with no declared cuts.
6. A 10-figure request whose scene named the cast collectively while ten anchors
   followed — a crowd noun paints extra unnamed bodies.

The guarantees are code properties: style sentence first, exactly one description
per named figure, a stated figure count, no negation anywhere except the one
fixed lettering guard, and a budget that compresses the LEAST distinctive clauses
first while every requested name survives.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import bot_core.images as images
from bot_core import image_prompt as ip

BASE = images._BASE_STYLE
TAIL = ip.NO_LETTERING_TAIL          # the composer's one fixed negation

_SHEET = """# Party Roster

Use these Visual Anchors in group images.

# Seta Onix
Shoosuva (cattle-tiefling), Fighter. Pronouns: she/her.

## Visual Anchor (group images — reuse VERBATIM for named appearances)

> species: shoosuva — a tall, powerfully built cattle tiefling demon
> body: 6.5 feet tall, deep purple fur covering her entire body, digitigrade cloven hooves
> face: long curved goat-like horns, glowing amber eyes, rectangular dark glasses
> outfit: dark green cloak with a deep emerald hood, studded leather armor beneath
> signature: a longbow with an elegant antler-like design resting on her back
> quirks: calm and quietly vigilant

# Colum Bidae
Birdfolk (blue tit), Warlock. Pronouns: he/him.

## Visual Anchor (group images — reuse VERBATIM for named appearances)

> species: birdfolk — a small blue tit
> body: 2 feet tall, plump round body, fluffy blue-grey feather coloration
> outfit: a patchwork cloak stitched from mismatched fabric scraps
> signature: a small leather spell pouch with bone clasps
> quirks: googly eyes

# Trixy Smoldersome
Jackalwere (chihuahua), Artificer. Pronouns: she/her.

## Visual Anchor (group images — reuse VERBATIM for named appearances)

> species: jackalwere — a chihuahua jackal
> body: tiny, slender delicate frame standing barely 2 feet tall
> outfit: oversized brass goggles pushed up on her forehead, leather apron
> signature: a large brass-bulbed antique lantern horn glowing faintly blue

# Marvin
Steel Defender construct. No pronouns (it/its).

## Visual Anchor (group images — reuse VERBATIM for named appearances)

> species: construct — a brass-and-steel machine, NOT an animal
> body: teacup-shaped automaton on four small wheels
> face: glowing blue eyes
> signature: a brass horn mounted on its back

# Romi Stachelpfote
Hedge (hedgehog), Monk. Pronouns: he/him.

## Visual Anchor (group images — reuse VERBATIM for named appearances)

> A small round hedgehog with soft brown fur, wide alert amber eyes, quills
> folded forward, wearing a simple monk's robe of undyed linen.

## Notes
Not a Visual Anchor line.
"""

_NPCS = [("Benner Seridan", "watching", "old badger-folk priest in ink-dark vestments"),
         ("Lucky", "rolling dice", "weasel-folk gambler in a long green coat"),
         ("Hexe Susan", "stirring a pot", "hedgehog-folk hedge witch with a shawl")]


def _anchors():
    return {a.name: a for a in ip.parse_party_sheet(_SHEET)}


def _plan(scene, figures, **kw):
    data = {"scene": scene, "figures": figures}
    data.update(kw)
    plan = ip.extract_plan("<image_plan>\n" + json.dumps(data) + "\n</image_plan>")
    assert plan is not None
    return plan


def _compose(plan, anchors=None, *, budget=100000, foreground=4):
    return ip.compose_prompt(plan, anchors or _anchors(), BASE,
                             foreground=foreground,
                             budget_kwargs={"base": 1100, "per_figure": 420,
                                            "hard_cap": budget})


def _content(c):
    """The prompt minus the composer's one fixed lettering guard."""
    return c.prompt.replace(TAIL, "")


# ── Parsing the two vault shapes ─────────────────────────────────────────


def test_typed_fields_and_legacy_paragraph_both_parse():
    anchors = _anchors()
    assert set(anchors) == {"Seta Onix", "Colum Bidae", "Trixy Smoldersome",
                            "Marvin", "Romi Stachelpfote"}
    seta = anchors["Seta Onix"]
    assert seta.typed and seta.fields["signature"].startswith("a longbow")
    assert seta.desc.startswith("Shoosuva")
    # The failure #1 class, one layer down: an anchor heading is never mistaken
    # for a field, and a later section never leaks into the anchor.
    assert not any("reuse VERBATIM" in v for v in seta.fields.values())
    assert not any("Not a Visual Anchor" in v for v in seta.fields.values())
    legacy = anchors["Romi Stachelpfote"]
    assert not legacy.typed
    assert legacy.blob.startswith("A small round hedgehog")
    assert legacy.blob.endswith("undyed linen.")


def test_unknown_field_names_are_kept_not_dropped():
    """A field the composer doesn't know is shown to the model and painted —
    editing a sheet in Obsidian must never lose content."""
    raw = ("# Neue Figur\nKit-folk.\n\n## Visual Anchor\n\n"
           "> species: kit-folk\n> wheels: two brass bicycle wheels\n")
    a = ip.parse_party_sheet(raw)[0]
    assert a.fields["species"] == "kit-folk"
    assert a.extra == ["wheels: two brass bicycle wheels"]
    c = _compose(_plan("A figure stands", [{"name": "Neue Figur", "pose": "standing"}]),
                 {"Neue Figur": a})
    assert "two brass bicycle wheels" in c.prompt


def test_continuation_lines_join_their_field():
    raw = ("# A\nKit.\n\n## Visual Anchor\n\n"
           "> body: a tall figure with\n> long arms and\n> species: kit\n")
    a = ip.parse_party_sheet(raw)[0]
    assert a.fields["body"] == "a tall figure with long arms and"
    assert a.fields["species"] == "kit"


# ── The composition guarantees (failures 1, 5, 6) ────────────────────────


def test_structure_count_line_and_one_description_per_figure():
    plan = _plan("Seta draws her bow while Trixy aims a device on a forest path",
                 [{"name": "Seta", "pose": "drawing her longbow"},
                  {"name": "Trixy", "pose": "aiming a device"}],
                 lighting="cold dusk", palette="moss green", mood="tense")
    c = _compose(plan)
    assert c.prompt.startswith(BASE + ". ")
    assert "Exactly 2 named figures in this image" in c.prompt
    # The vault spelling is used even when the plan abbreviates the name — a
    # nickname must not drift onto a different character.
    assert c.prompt.count("Seta Onix:") == 1
    assert c.prompt.count("Trixy Smoldersome:") == 1
    # Appearance is the vault's, pose is the model's.
    assert "deep purple fur" in c.prompt and "antler-like" in c.prompt
    assert "aiming a device" in c.prompt
    assert "brass-bulbed antique lantern horn" in c.prompt
    # Order: style, scene, count, then the figures.
    assert c.prompt.index("forest path") < c.prompt.index("Exactly 2")
    assert c.prompt.index("Exactly 2") < c.prompt.index("Seta Onix:")
    assert c.prompt.endswith(f"Mood: tense. {TAIL}")
    assert c.cuts == []


def test_clauses_whose_only_colours_are_shared_move_last():
    """§3: with six anchors sharing 34 colour tokens, 'gold' identifies nobody.
    A clause whose colours every other figure in the frame also wears is the
    cheapest thing to say, so it goes last and the identity clauses lead."""
    out = ip._unique_first(["glowing amber eyes", "rectangular dark glasses"],
                           shared={"amber", "blue"})
    assert out == ["rectangular dark glasses", "glowing amber eyes"]
    # A clause mixing a shared and a unique colour still leads.
    out2 = ip._unique_first(["amber and indigo fur", "amber eyes"],
                            shared={"amber", "indigo", "blue"})
    assert out2 == ["amber and indigo fur", "amber eyes"]


def test_foreground_split_keeps_the_requested_order():
    plan = _plan("Five party members hold a muddy ford",
                 [{"name": n, "pose": "holding the line"} for n in
                  ("Seta Onix", "Colum Bidae", "Trixy Smoldersome", "Marvin",
                   "Romi Stachelpfote")])
    c = _compose(plan, foreground=4)
    assert c.foreground == 4
    assert "4 in the foreground and 1 behind them" in c.prompt
    assert c.prompt.count("Behind them, smaller and softer, ") == 1
    # The first four named figures are the foreground ones.
    assert c.prompt.index("Seta Onix:") < c.prompt.index("Marvin:")
    assert c.prompt.index("Behind them") > c.prompt.index("Marvin:")


def test_duplicate_figure_entries_are_merged_once():
    """Failure 4: one name entered twice → two bodies painted."""
    look = "lean wolf-folk scout in a moss-grey cloak"
    plan = _plan("Spleck stalks a merchant",
                 [{"name": "Spleck", "pose": "stalking", "look": look,
                   "source": "SESSION_05/notes.md"},
                  {"name": "Spleck (the wolf-folk)", "pose": "crouching",
                   "look": look, "source": "SESSION_05/notes.md"}])
    c = _compose(plan)
    assert c.figures == 1
    assert "Exactly 1 named figure" in c.prompt
    assert c.prompt.count(look) == 1
    assert "stalking" in c.prompt and "crouching" in c.prompt, \
        "merging keeps both actions, not the duplicate entry"


def test_a_party_member_named_two_way_is_one_figure():
    plan = _plan("Seta helps Seta Onix", [{"name": "Seta", "pose": "helping"},
                                          {"name": "Seta Onix", "pose": "standing"}])
    c = _compose(plan)
    assert c.figures == 1
    assert "Exactly 1 named figure" in c.prompt


def test_collective_subject_is_reported_and_never_painted():
    """Failure 6 (measured twice): the scene named the cast AND the anchors
    followed, which is how an unnamed crowd gets painted."""
    plan = _plan("the whole upright party squares off against a badger-folk bandit",
                 [{"name": "Seta Onix", "pose": "nocking an arrow"}])
    c = _compose(plan)
    assert any("collectively" in w for w in c.warnings)
    assert "party" not in _content(c).lower().replace("cattle tiefling", "")
    assert "Exactly 1 named figure" in c.prompt


def test_collective_detection_covers_english_and_german():
    for bad in ("the party charges", "die ganze Gruppe greift an",
                "the group surrounds the wagon", "several adventurers bargain",
                "all of the companions cheer"):
        assert ip.is_collective(bad), bad
    for good in ("Seta draws her bow", "Marvin guards the camp",
                 "a badger-folk chieftain watches"):
        assert not ip.is_collective(good), good


# ── Overrides (failures 2 and 3) ─────────────────────────────────────────


def test_override_replaces_one_field_and_inherits_the_rest():
    """Failure 2: 'plate armor replacing her usual cloak' became a contradiction.
    An override now REPLACES the field and nothing about the old value is said."""
    plan = _plan("Seta stands watch on a rain-wet road",
                 [{"name": "Seta Onix", "pose": "keeping watch",
                   "modify": {"outfit": "polished steel plate armour over a "
                                        "leather harness"}}],
                 lighting="grey rain")
    c = _compose(plan)
    assert "polished steel plate armour" in c.prompt
    assert "dark green cloak" not in c.prompt
    assert "instead of" not in c.prompt and "usual" not in c.prompt
    # Her bow is `signature`, so an `outfit` override cannot reach it.
    assert "longbow" in c.prompt and "antler-like" in c.prompt
    assert "deep purple fur" in c.prompt
    assert not ip.has_negation(_content(c))


def test_removal_is_painted_by_omission():
    """Failure 3: 'NO cape or cloak' and 'goggles removed, revealing…' both
    describe the thing the renderer then paints."""
    c = _compose(_plan("Trixy reads a map by lamplight",
                       [{"name": "Trixy Smoldersome", "pose": "reading a map",
                         "modify": {"outfit": "leather apron"}}]))
    assert "goggle" not in c.prompt.lower()
    assert "leather apron" in c.prompt
    assert not ip.has_negation(_content(c))

    c2 = _compose(_plan("Colum crosses a mossy log",
                        [{"name": "Colum", "pose": "crossing a log",
                          "modify": {"outfit": ""}}]))
    assert "cloak" not in c2.prompt.lower()
    assert "spell pouch" in c2.prompt, "signature survives an outfit removal"
    assert not ip.has_negation(_content(c2))


def test_signature_field_can_be_replaced_but_not_by_an_outfit_override():
    c = _compose(_plan("Seta draws a different weapon",
                       [{"name": "Seta Onix", "pose": "drawing a sword",
                         "modify": {"signature": "a hand crossbow slung at her hip"}}]))
    assert "hand crossbow" in c.prompt
    assert "longbow" not in c.prompt
    assert "dark green cloak" in c.prompt, "the clothing field is untouched"


def test_legacy_blob_override_filters_the_replaced_clause():
    """Old vault format has no fields to replace, so the closest clauses are
    filtered out — and the composer says so instead of silently improvising."""
    c = _compose(_plan("Romi meditates at dawn",
                       [{"name": "Romi Stachelpfote", "pose": "meditating",
                         "modify": {"outfit": "a travelling cloak of oiled wool"}}]))
    assert "oiled wool" in c.prompt
    assert "undyed linen" not in c.prompt, "the replaced clothing clause is gone"
    assert "hedgehog" in c.prompt and "soft brown fur" in c.prompt, \
        "the species and body clauses survive"
    assert any("legacy anchor" in w for w in c.warnings)


def test_an_unanchored_figure_gets_only_the_models_look():
    c = _compose(_plan("Spleck stalks a merchant",
                       [{"name": "Spleck", "pose": "stalking",
                         "look": "lean wolf-folk scout in a moss-grey cloak",
                         "source": "SESSION_05/notes.md"}]))
    assert "lean wolf-folk scout" in c.prompt
    assert not any("no party anchor" in w for w in c.warnings), \
        "a looked-up NPC with a source is exactly what the contract asks for"


def test_an_unanchored_figure_without_a_look_is_reported_not_invented():
    c = _compose(_plan("Seta blocks a road against Frey",
                       [{"name": "Seta Onix", "pose": "blocking the road"},
                        {"name": "Frey", "pose": "brandishing an axe"}]))
    assert any("no party anchor and no `look`" in w for w in c.warnings)
    assert "Frey: brandishing an axe." in c.prompt
    assert not any("look" in w and "source" in w for w in c.warnings)
    assert "Exactly 2 named figures" in c.prompt, "the figure is still counted"


def test_a_look_without_a_source_is_reported():
    c = _compose(_plan("Frey blocks the road",
                       [{"name": "Frey", "pose": "blocking",
                         "look": "heavy-set badger-folk in a torn coat"}]))
    assert any("`look` without a `source`" in w for w in c.warnings)


def test_override_on_an_unknown_field_is_ignored_and_reported():
    plan = ip.ImagePlan(scene="Seta stands", figures=[
        ip.Figure("Seta Onix", "standing", modify={"fur": "bright orange fur"})])
    c = _compose(plan)
    assert "orange" not in c.prompt
    assert "deep purple fur" in c.prompt, "the anchor is not a suggestion"
    assert any("unknown field" in w for w in c.warnings)


# ── Negation scrubbing (failure 3) ───────────────────────────────────────


def test_negated_clause_keeps_its_positive_content():
    a = ip.Anchor("X", fields={"species": "construct — NOT an animal or beast-folk "
                                          "but a brass machine"})
    assert a.clauses_for({})[0] == {"species": ["construct — a brass machine"]}, \
        "the denial goes, the paintable label stays"

    b = ip.Anchor("Y", fields={"face": "glowing blue eyes, no visible pupils"})
    assert b.clauses_for({})[0] == {"face": ["glowing blue eyes"]}

    # A clause that is nothing but a negation carries nothing to paint.
    c = ip.Anchor("Z", fields={"quirks": "no wings, no tail"})
    assert c.clauses_for({})[0] == {}

    # The dropped denial is still REPORTED, so a sheet that argues with itself is
    # visible in the log instead of silently half-painted.
    d = ip.Anchor("W", fields={"outfit": "no cloak at all"})
    assert d.clauses_for({}) == ({}, ["no cloak at all"])


def test_a_negated_parenthetical_is_dropped_not_the_clause_it_hangs_on():
    """Tracy's real anchor line: '…the classic raccoon pattern (fur coloration
    only)' is paintable; a denial inside brackets must not delete the description."""
    a = ip.Anchor("Tracy", fields={"body": "grizzled texture, fluffy blue-grey "
                                           "feather coloration only (no hair on the "
                                           "head)"})
    assert a.clauses_for({})[0] == {"body": ["grizzled texture",
                                             "fluffy blue-grey feather coloration only"]}


def test_split_never_breaks_inside_parentheses():
    """A fragment like 'fur coloration only)' is worse than no fragment: the
    renderer reads it as an object to paint."""
    a = ip.Anchor("Colum", fields={"body": "2 feet tall, short (but sturdy) frame"})
    assert a.clauses_for({})[0] == {"body": ["2 feet tall",
                                             "short (but sturdy) frame"]}


def test_negation_in_a_pose_is_dropped_not_written():
    c = _compose(_plan("Colum stands", [{"name": "Colum Bidae",
                                         "pose": "standing with no cloak on"}]))
    assert "no cloak" not in c.prompt
    assert any("negation in the pose" in w for w in c.warnings)


def test_negation_detection_covers_the_german_requests():
    assert ip.has_negation("wearing ONLY a tunic with NO cape or cloak")
    assert ip.has_negation("goggles removed from her forehead")
    assert ip.has_negation("der Zwerg ohne seinen Umhang")
    assert ip.has_negation("Metallrüstung statt eines Umhangs")
    assert ip.has_negation("a hooded figure, not a beast-folk")
    assert not ip.has_negation("a badger-folk chieftain in a fur-lined cloak")
    assert not ip.has_negation("a watchful party member")


# ── Budget ladder (failure 5) ────────────────────────────────────────────


def test_budget_scales_with_the_cast():
    assert ip.prompt_budget(1) == 1100 + 420
    assert ip.prompt_budget(6) == 3600          # 1100+2520 = 3620 → capped
    assert ip.prompt_budget(8, hard_cap=1200) == 1200


def test_six_real_anchors_fit_without_a_single_cut():
    """Measured: the party's six Visual Anchors + a scene line must fit the real
    budget, so a group image loses nothing (the old policy compressed anchors to
    'identity-critical core' at six — that guess is now a measurement)."""
    anchors = {a.name: a for a in ip.parse_party_sheet(
        images._PARTY_SHEET_FILE.read_text(encoding="utf-8"))}
    assert len(anchors) == 6
    plan = ip.ImagePlan(scene="Six party members hold a muddy ford at dusk",
                        figures=[ip.Figure(n, "holding the line") for n in anchors],
                        lighting="cold dusk", palette="moss green", mood="tense")
    c = ip.compose_prompt(plan, anchors, BASE,
                          budget_kwargs={"base": 1100, "per_figure": 420,
                                         "hard_cap": 3600})
    assert c.cuts == [], c.cuts
    assert len(c.prompt) <= 3600
    assert all(n in c.prompt for n in anchors)


def test_a_shared_colour_clause_is_painted_after_a_distinctive_one():
    """§3 in action: 'amber' worn by both figures identifies neither, so the
    clause that identifies someone moves first and the shared colour trails."""
    anchors = {
        "Aya": ip.Anchor("Aya", fields={"face": "amber eyes, a notched left ear",
                                        "signature": "a copper lantern"}),
        "Bram": ip.Anchor("Bram", fields={"face": "amber eyes, a broken tusk",
                                          "signature": "an iron key"}),
    }
    c = _compose(_plan("Two figures stand at a gate",
                       [{"name": "Aya", "pose": "waiting"},
                        {"name": "Bram", "pose": "knocking"}]), anchors=anchors)
    line = c.prompt[c.prompt.index("Aya:"):c.prompt.index("Bram:")]
    assert line.index("a notched left ear") < line.index("amber eyes")
    assert line.index("amber eyes") < line.index("copper lantern"), \
        "a signature item identifies someone, a shared colour does not"


def test_a_tight_budget_cuts_around_the_cast_never_through_it():
    plan = _plan("Eight figures hold a crowded market square at dusk",
                 [{"name": n, "pose": "holding the line"} for n in
                  ("Seta Onix", "Colum Bidae", "Trixy Smoldersome", "Marvin",
                   "Romi Stachelpfote")]
                 + [{"name": n, "pose": p, "look": l, "source": "NPC Codex.md"}
                    for n, p, l in _NPCS],
                 palette="moss green, brass", mood="bustling")
    c = _compose(plan, budget=1000)
    assert len(c.prompt) <= 1000
    # Including every requested character beats brevity (the repo's own rule).
    for name in ("Seta Onix", "Colum Bidae", "Trixy Smoldersome", "Marvin",
                 "Romi Stachelpfote", "Benner Seridan", "Lucky", "Hexe Susan"):
        assert name in c.prompt
    assert c.cuts and all(w.startswith("budget ") for w in c.cuts)
    # The cut ORDER is declared, not improvised.
    assert "mood line dropped" in c.cuts[0]
    assert "clause(s) compressed" in c.cuts[-1]
    assert "Exactly 8 named figures" in c.prompt


def test_compression_spends_the_least_distinctive_clause_first():
    anchors = {"Colum Bidae": ip.Anchor(
        "Colum Bidae",
        fields={"species": "birdfolk — a small blue tit",
                "body": ", ".join(f"clause number {i} of the body description text"
                                  for i in range(8)),
                "signature": "a small leather spell pouch with bone clasps"})}
    plan = ip.ImagePlan(scene="Colum stands in a hall",
                        figures=[ip.Figure("Colum Bidae", "standing")])
    c = _compose(plan, anchors=anchors, budget=400)
    assert any("body]" in cut for cut in c.cuts)
    assert "clause number 0" not in c.prompt, "the first (least distinctive) goes"
    assert "spell pouch" in c.prompt, "a signature item is the last thing cut"
    assert "Colum Bidae" in c.prompt


def test_a_scene_over_the_hard_cap_is_truncated_and_says_so():
    anchors = {"Colum Bidae": ip.Anchor(
        "Colum Bidae", fields={"body": ", ".join(
            f"a description clause number {i} with several words in it"
            for i in range(40))})}
    plan = ip.ImagePlan(scene="Colum stands in a hall",
                        figures=[ip.Figure("Colum Bidae", "standing")])
    # A budget below the irreducible frame (style + scene + count line + tail):
    # the ladder runs out of compressible parts and says so out loud.
    c = _compose(plan, anchors=anchors, budget=120)
    assert len(c.prompt) <= 120
    assert any("hard-truncated" in w for w in c.warnings)


def test_budget_is_computed_after_dedupe():
    plan = _plan("Seta stands", [{"name": "Seta Onix", "pose": "x"},
                                {"name": "Seta", "pose": "y"}])
    c = _compose(plan, budget=1100 + 420)     # one figure's budget
    assert c.figures == 1
    assert len(c.prompt) <= 1100 + 420


# ── Plan extraction tolerance ────────────────────────────────────────────


def test_extract_plan_handles_fences_prose_and_bare_json():
    for text in (
        '<image_plan>{"scene": "a", "figures": []}</image_plan>',
        'Sure!\n<image_plan>\n{"scene": "b", "figures": []}\n</image_plan>\nEnjoy!',
        '```json\n{"scene": "c", "figures": []}\n```',
        'Here you go: {"scene": "d", "figures": []}',
    ):
        p = ip.extract_plan(text)
        assert p is not None and p.scene in ("a", "b", "c", "d")
    assert ip.extract_plan("") is None
    assert ip.extract_plan("I will search the vault for that.") is None
    assert ip.extract_plan("<image_plan>\nnot json at all\n</image_plan>") is None


def test_extract_plan_maps_the_field_names_and_drops_junk():
    p = ip.extract_plan('<image_plan>{"scene": "s", "figures": [{"name": "A", '
                        '"pose": "p", "modify": {"outfit": "new"}, "junk": 1}], '
                        '"entities": ["a cart", "", 5], "mood": "tense"}</image_plan>')
    assert p.figures[0].pose == "p"
    assert p.figures[0].modify == {"outfit": "new"}
    assert not hasattr(p.figures[0], "junk")
    assert p.entities == ["a cart"]
    assert p.mood == "tense"


def test_extract_plan_accepts_the_shape_variants_a_model_drifts_into():
    p = ip.extract_plan('<image_plan>{"scene": "s", "characters": {"Seta Onix": '
                        '"drawing her bow"}, "figures": [{"name": "Colum", '
                        '"action": "perched", "appearance": "small blue bird", '
                        '"override": ["quirks"]}]}</image_plan>')
    names = [f.name for f in p.figures]
    assert names == ["Colum"] and p.figures[0].pose == "perched"
    assert p.figures[0].look == "small blue bird"
    assert p.figures[0].modify == {"quirks": ""}


def test_extract_plan_allows_a_scene_with_no_cast():
    p = ip.extract_plan('<image_plan>{"scene": "a black pudding oozes down a '
                        'corridor", "entities": ["a black puddling ooze mass"]}'
                        '</image_plan>')
    assert p is not None and p.figures == []
    c = _compose(p)
    assert "Exactly 0 named figures" in c.prompt
    assert "a black puddling ooze mass" in c.prompt


def test_an_empty_plan_is_an_error_not_an_empty_prompt():
    with pytest.raises(ValueError):
        _compose(ip.ImagePlan(scene="", figures=[]))


# ── Cross-module wiring ──────────────────────────────────────────────────


def test_plan_path_rejects_a_prompt_shaped_answer_and_retries(monkeypatch, tmp_path):
    """The plan contract is asked for; a prose answer falls back to the older
    path rather than painting a plan."""
    sheet = tmp_path / "Player Characters.md"
    sheet.write_text(_SHEET, encoding="utf-8")
    monkeypatch.setattr(images, "_PARTY_SHEET_FILE", sheet)

    replies = ["I'll search the knowledge base for the party members.",
               "Let me look that up in the vault first.",
               BASE + ". Seta nocks an arrow on a forest path."]
    calls: list[dict] = []

    async def create(**kwargs):
        # The agent loop spends a turn of its own nudging an answer that was
        # neither a tool call nor usable, so the script is clamped to its last
        # entry rather than running out.
        calls.append(kwargs)
        return MagicMock(choices=[MagicMock(message=MagicMock(
            content=replies[min(len(calls) - 1, len(replies) - 1)]))])

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=create)
    monkeypatch.setattr(images, "_chat_client", lambda: client)

    out = asyncio.run(images.rewrite_prompt("die ganze Gruppe"))
    assert out.startswith(BASE), "the prose fallback prompt is what gets painted"
    assert len(calls) >= 3


def test_the_scene_word_cap_is_enforced():
    """Measured failure 5: an 11-figure request wrote a scene sentence so long no
    anchors fitted. The contract caps it at 45 words; that cap is now code."""
    long_scene = " ".join(f"word{i}" for i in range(90))
    c = _compose(_plan(long_scene, [{"name": "Colum Bidae", "pose": "standing"}]))
    assert any("trimmed" in w for w in c.warnings)
    assert "word89" not in c.prompt and "word44" in c.prompt


def test_the_plan_contract_file_is_loaded_with_the_roster_injected():
    system = images._load_plan_system()
    assert "<image_plan>" in system
    assert "exactly ONE roster field" in system
    if not images._PARTY_SHEET_FILE.exists():
        pytest.skip("vault party sheet not present (data/ is gitignored)")
    # The roster is the REAL vault sheet, with its field labels visible so the
    # model can aim an override — but not the full text it no longer has to copy.
    for name in ("Colum Bidae", "Seta Onix", "Trixy Smoldersome", "Marvin"):
        assert name in system
    assert "outfit:" in system and "signature:" in system
    assert "{{PARTY_ROSTER}}" not in system and "{{CHARACTER_RULES}}" not in system


# ── 7. NPC appearance grounding: a look must trace to a file this rewrite read ─


def test_source_matching_is_suffix_segments_not_substring():
    # The rule kb/tools.py:193 got wrong (analysis §2): "spleck" must not match
    # "Spleck Onix.md", a trailing folder name must.
    assert ip.source_matches("Characters/NPCs.md",
                             ["Humblewood/HumbleWood/Characters/NPCs.md"])
    assert ip.source_matches("NPCs.md", ["Humblewood/HumbleWood/Characters/NPCs.md"])
    assert not ip.source_matches("NPC", ["Humblewood/HumbleWood/Characters/NPCs.md"])
    assert not ip.source_matches("Spleck", ["Humblewood/.../Spleck Onix.md"])
    assert ip.source_matches("HumbleWood\\Characters\\NPCs.md",
                             ["Humblewood/HumbleWood/Characters/NPCs.md"])


def test_an_unverified_npc_look_is_dropped_and_said_so():
    anchors = {a.name: a for a in ip.parse_party_sheet(_SHEET)}
    plan = _plan("Spleck stalks a merchant",
                 [{"name": "Boss Frey", "pose": "blocking the road",
                   "look": "large badger-folk with grey-black striped fur",
                   "source": "Characters/Villains.md"}])
    c = _compose(plan, anchors=anchors)          # no read_paths → old behaviour
    assert "badger-folk" in c.prompt
    c = ip.compose_prompt(plan, anchors, BASE, read_paths=[])
    assert "badger-folk" not in c.prompt
    assert any("did not read" in w for w in c.warnings)
    assert "Boss Frey" in c.prompt, "the figure stays; only the look goes"
    assert "Exactly 1 named figure" in c.prompt


def test_a_look_from_a_file_the_rewrite_read_is_kept():
    anchors = {a.name: a for a in ip.parse_party_sheet(_SHEET)}
    plan = _plan("Spleck stalks a merchant",
                 [{"name": "Boss Frey", "pose": "blocking the road",
                   "look": "large badger-folk with grey-black striped fur",
                   "source": "Characters/Villains.md"}])
    c = ip.compose_prompt(plan, anchors, BASE,
                          read_paths=["Humblewood/HumbleWood/Characters/Villains.md"])
    assert "badger-folk" in c.prompt
    assert not any("did not read" in w for w in c.warnings)


def test_a_citation_named_without_a_lookup_is_reported():
    anchors = {a.name: a for a in ip.parse_party_sheet(_SHEET)}
    plan = _plan("Spleck stalks a merchant",
                 [{"name": "Boss Frey", "pose": "blocking the road",
                   "look": "large badger-folk"}])
    c = ip.compose_prompt(plan, anchors, BASE, read_paths=[])
    assert "badger-folk" not in c.prompt
    assert any("without a `source`" in w for w in c.warnings)


# ── 8. Foreground ordering: the party holds the front row ────────────────────

def test_the_party_holds_the_foreground_even_if_the_plan_listed_npc_first():
    anchors = {a.name: a for a in ip.parse_party_sheet(_SHEET)}
    plan = _plan("A road blocked in the mist",
                 [{"name": "Boss Frey", "pose": "blocking the road",
                   "look": "badger-folk", "source": "Villains.md"},
                  {"name": "Seta Onix", "pose": "nocking an arrow"},
                  {"name": "Colum Bidae", "pose": "casting"},
                  {"name": "Romi Stachelpfote", "pose": "striking"},
                  {"name": "Bubo Krallensturm", "pose": "chanting"},
                  {"name": "Trixy Smoldersome", "pose": "aiming"}])
    c = ip.compose_prompt(plan, anchors, BASE,
                          read_paths=["Humblewood/Characters/Villains.md"])
    assert c.prompt.index("Seta Onix:") < c.prompt.index("Boss Frey:"), \
        "the plan listed the bandit first; the composer puts the party first"
    assert "Behind them, smaller and softer, Boss Frey:" in c.prompt
    assert any("reordered" in w for w in c.warnings)
