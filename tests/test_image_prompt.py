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
from bot_core import prompt_profile

# The composer's style sentence is an argument, so these tests supply their own
# rather than depending on a gitignored campaign profile being present.
BASE = ("Semi-realistic fantasy illustration — detailed fur and feather texture, "
        "natural light, storybook charm")
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


# The campaign sentence is campaign text (it lives in the campaign profile in the real
# bot), so the tests state a campaign of their own instead of reading the repo's.
WORLD_RULE = ("Every figure in this image is an anthropomorphic animal-folk character "
         "of this world")


def _compose(plan, anchors=None, *, budget=100000, foreground=4, rule=WORLD_RULE,
             retrieved="", request_text="", read_paths=None, clause_cap=0):
    return ip.compose_prompt(plan, anchors or _anchors(), BASE,
                             foreground=foreground,
                             max_clauses_per_figure=clause_cap,
                             budget_kwargs={"base": 1100, "per_figure": 420,
                                            "hard_cap": budget},
                             world_rule=rule,
                                                          retrieved=retrieved, request_text=request_text,
                             read_paths=read_paths)


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


def _finding(c, code, subject=None):
    """One finding the composer must have reported (and NOT fixed)."""
    hits = [f for f in c.findings if f.code == code
            and (subject is None or f.subject == subject)]
    assert hits, f"no {code} finding: " + "; ".join(
        f.as_line() for f in c.findings)
    return hits[0]



def test_a_name_used_to_say_where_another_figure_stands_is_not_a_second_figure():
    """Round 12, request 3: 'Trixy Smoldersome stands upright in plate armour while
    Colum Bidae perches on Trixy Smoldersome's shoulder' was reported as Trixy
    painted twice. The genitive says the second mention places COLUM — it does not
    bring Trixy into the frame again, and a false report costs a model turn."""
    c = _compose(_plan("Trixy Smoldersome stands upright while Colum Bidae perches "
                       "on Trixy Smoldersome's shoulder",
                       [{"name": "Trixy Smoldersome", "pose": "standing upright"},
                        {"name": "Colum Bidae", "pose": "perched on her shoulder"}]))
    assert not [f for f in c.findings if f.code == "duplicate_mention"], c.findings
    # The shape the rule was written for still is reported.
    c2 = _compose(_plan("Seta Onix draws her bow; bandits surround Seta Onix, Romi "
                        "Stachelpfote, Marvin",
                        [{"name": "Seta Onix", "pose": "drawing her bow"},
                         {"name": "Romi Stachelpfote", "pose": "advancing"},
                         {"name": "Marvin", "pose": "standing"}]))
    assert {f.subject for f in c2.findings if f.code == "duplicate_mention"} == \
        {"Seta Onix"}, c2.findings


def test_a_negation_in_the_scene_sentence_is_reported_like_one_in_a_pose():
    """Round 12, request 5 reached the renderer with '… in one clawed hand; no one
    else is near.' Poses and entities were policed; the scene line was not, and the
    renderer paints what a sentence names."""
    c = _compose(_plan("Boss Frey stands alone on the path, axe low; no one else "
                       "is near", [{"name": "Boss Frey", "pose": "standing alone"}]))
    f = _finding(c, "negation_clause", "scene")
    assert "no" in f.detail
    assert "no one else is near" in c.prompt, "reported, not deleted"


def test_the_users_own_words_are_matched_as_written_not_as_lowercased_words():
    """Round 12, request 1 reported 'the user asked for gold' for a plan whose
    entity line was 'a bulging leather pouch of seventy gold coins'. German
    capitalizes nouns: `70 Gold` is an amount, not a character. The match is now
    case-sensitive against the request's own spelling."""
    plan = _plan("Seta Onix draws her bow",
                 [{"name": "Seta Onix", "pose": "drawing her bow"}],
                 entities=["a bulging leather pouch of seventy gold coins",
                           "Boss Frey and his bandit crew"])
    c = _compose(plan, request_text="… die befreite Händlerin Alisa Pennygleam "
                                    "klammert den Beutel mit 70 Gold")
    reported = [f.subject for f in c.findings
                if f.code == "entity_named_not_painted"]
    assert reported == [], reported
    c2 = _compose(plan, request_text="Boss Frey stellt sich der Gruppe")
    assert [f.subject for f in c2.findings
            if f.code == "entity_named_not_painted"] == \
        ["Boss Frey and his bandit crew"], c2.findings

def test_the_composer_reports_instead_of_rewriting():
    """The one test that says what the 2026-11-19 redesign IS.

    A plan with three known problems: a collective subject, a negation in a pose
    and a parenthetical note about the bot. Every one of them used to be edited out
    of the plan's own text. Now every one of them reaches the prompt as the model
    wrote it, and every one is named back to the model instead. The scene also
    carries a species the vault disagrees with (weasel-folk vs the notes' rat-folk):
    since 2026-11-22 that is a campaign fact the composer has no way to check, so it
    is painted as written and nothing is reported (§20).
    """
    plan = _plan("the whole party faces two weasel-folk bandits (vault guess)",
                 [{"name": "Seta Onix", "pose": "drawing her bow with no quiver"}],
                 retrieved="")
    c = ip.compose_prompt(
        plan, _anchors(), BASE, world_rule=WORLD_RULE,
        retrieved="### Die Banditen\nBanditen: three rat-folk cutthroats.",
        budget_kwargs={"base": 1100, "per_figure": 420, "hard_cap": 100000})
    codes = {f.code for f in c.findings}
    assert {"collective_subject", "negation_clause", "meta_comment"} <= codes, sorted(codes)
    low = c.prompt.lower()
    for fragment in ("the whole party", "weasel-folk", "no quiver", "vault guess"):
        assert fragment in low, f"{fragment} was edited out of the prompt"
    assert all(f.fixable for f in c.findings if f.code in codes)


def test_a_budget_cut_is_a_fact_not_a_finding_the_model_can_argue_with():
    """`fixable` separates "reword your plan" from "the frame has a size limit"."""
    plan = _plan("Seta Onix draws her bow",
                 [{"name": "Seta Onix", "pose": "drawing her bow"}])
    c = _compose(plan, budget=200)
    assert c.cuts, "the budget ladder must still report what it spent"
    assert not any(f.fixable and "budget" in f.detail for f in c.findings)


def test_the_correction_turn_gets_a_message_it_can_act_on():
    findings = ip.lint_plan(
        _plan("the whole upright party squares off against a badger-folk bandit",
              [{"name": "Seta Onix", "pose": "nocking an arrow"}]), _anchors())
    text = ip.finding_feedback(findings)
    assert "<image_plan>" in text
    assert "collective_subject" in text
    assert "Do not describe the picture in prose" in text
    assert ip.finding_feedback([]) == ""


def test_collective_subject_is_reported_and_never_painted():
    """Failure 6 (measured twice): the scene named the cast AND the anchors
    followed, which is how an unnamed crowd gets painted.

    Retired 2026-11-19 (§17.3): the composer used to swap the crowd noun for a
    generated name list. It now reports the shape and the rewrite model fixes its
    own sentence — a name list invented by a regex was a second sentence nobody
    asked for, and it had its own bugs (§13).
    """
    plan = _plan("the whole upright party squares off against a badger-folk bandit",
                 [{"name": "Seta Onix", "pose": "nocking an arrow"}])
    c = _compose(plan)
    f = _finding(c, "collective_subject", "scene")
    assert "whole upright party" in f.detail, f.detail
    assert "the whole upright party" in c.prompt.lower(), \
        "the model's sentence is left alone"
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
    # No evidence channel here: painted, but said out loud to be unverified.
    assert "badger-folk" in c.prompt
    assert any("has no `source`" in w and "unverified" in w for w in c.warnings)
    # Once the caller verifies, the same look is a rule violation and goes.
    c = ip.compose_prompt(_plan("Frey blocks the road",
                                [{"name": "Frey", "pose": "blocking",
                                  "look": "heavy-set badger-folk in a torn coat"}]),
                          {}, BASE, read_paths=[])
    assert "badger-folk" not in c.prompt
    assert any("without a `source`" in w for w in c.warnings)


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


def test_a_negation_in_a_pose_is_reported_not_silently_dropped():
    """Our own layers still never write a negation. A negation the MODEL wrote is
    no longer deleted on its way past — it comes back as a finding (§17.3)."""
    c = _compose(_plan("Colum stands", [{"name": "Colum Bidae",
                                         "pose": "standing with no cloak on"}]))
    f = _finding(c, "negation_clause", "Colum Bidae")
    assert "no cloak" in c.prompt
    assert "paints the item anyway" in f.detail


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
        prompt_profile.party_sheet_path().read_text(encoding="utf-8"))}
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
    # Budget tight enough to force compression, wide enough that the fixed
    # overhead (style + campaign sentence + count lines) is not what eats it.
    c = _compose(plan, anchors=anchors, budget=560)
    assert any("body]" in cut for cut in c.cuts), c.cuts
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
    monkeypatch.setattr(images, "_party_sheet_file", lambda: sheet)

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
    assert "exactly ONE field" in system
    if not prompt_profile.party_sheet_path().exists():
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
    c = _compose(plan, anchors=anchors)     # no evidence channel: taken at face value
    assert "badger-folk" in c.prompt
    c = ip.compose_prompt(plan, anchors, BASE, read_paths=[])
    assert "badger-folk" not in c.prompt
    assert any("never retrieved" in w for w in c.warnings)
    assert "Boss Frey" in c.prompt, "the figure stays; only the look goes"
    assert "Exactly 1 named figure" in c.prompt


def test_a_look_citing_a_search_snippet_must_be_traceable_to_it():
    """A hit is evidence; a description invented on top of one is not.

    Same citation, two looks: the one whose wording is in what was retrieved is
    painted, the one matching none of it is dropped. This is the difference
    between reading a note and pattern-matching a headline.
    """
    retrieved = ('8 hit(s): [{"file": "humblewood/SessionLogs/SESSION_01/notes.md", '
                 '"snippet": "Spleck Smandra: a lizard merchant who travels by a '
                 'wagon with a blue and yellow canopy"}]')
    snips = ["humblewood/SessionLogs/SESSION_01/notes.md"]
    traceable = _plan("Spleck waits by the wagon",
                      [{"name": "Spleck Smandra", "pose": "waiting by the wagon",
                        "look": "a lizard merchant with a blue and yellow wagon",
                        "source": "SESSION_01/notes.md"}])
    invented = _plan("Spleck waits by the wagon",
                     [{"name": "Spleck Smandra", "pose": "waiting by the wagon",
                       "look": "a tall reptilian warrior in black plate armour "
                               "carrying a two-handed axe",
                       "source": "SESSION_01/notes.md"}])
    keep = ip.compose_prompt(traceable, {}, BASE, read_paths=[],
                             snippet_paths=snips, retrieved=retrieved)
    drop = ip.compose_prompt(invented, {}, BASE, read_paths=[],
                             snippet_paths=snips, retrieved=retrieved)
    assert "blue and yellow wagon" in keep.prompt
    assert "reptilian warrior" not in drop.prompt
    assert any("search snippet" in w for w in drop.warnings)
    assert not any("search snippet" in w for w in keep.warnings)


def test_an_opened_file_is_believed_even_when_the_wording_does_not_match():
    """The vault is half German: a look written in English from a German section
    is a translation, not an invention, so the wording test never applies to a
    file the rewrite actually opened. Provenance is still enforced in code —
    that is a filesystem fact, not a reading of prose."""
    retrieved = ('8 hit(s): [{"file": "humblewood/CampainNotes/07.10.2026.md", '
                 '"snippet": "die Echse (Händler) mit ihrem blau-gelben Wagen"}]')
    plan = _plan("die Echse beside her wagon",
                 [{"name": "Spleck Smandra", "pose": "standing by the wagon",
                   "look": "a lizard merchant with scaled green-grey skin",
                   "source": "CampainNotes/07.10.2026.md"}])
    out = ip.compose_prompt(plan, {}, BASE,
                            read_paths=["CampainNotes/07.10.2026.md"],
                            snippet_paths=[], retrieved=retrieved,
                            world_rule=WORLD_RULE)
    assert "lizard merchant" in out.prompt
    # Nothing is reported either: provenance is a filesystem fact, and a look that
    # traces to an opened file is accepted whole — the composer no longer re-reads
    # it for a species claim it cannot verify (§20).
    assert not any(f.subject == "Spleck Smandra" for f in out.findings), out.findings


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


def test_a_scene_that_re_enumerates_the_cast_is_reported():
    """A name mentioned three times is a character painted twice (§4.2's
    'a second Colum in the background'). The cut is retired: the mention is the
    model's sentence, so it is reported and the correction turn rewrites it."""
    plan = _plan("In a clearing, Seta Onix aims her bow at Boss Frey while Romi "
                 "Stachelpfote advances; bandits surround Seta Onix, Romi "
                 "Stachelpfote, Bubo Krallensturm, Marvin",
                 [{"name": "Seta Onix", "pose": "aiming"},
                  {"name": "Romi Stachelpfote", "pose": "advancing"},
                  {"name": "Bubo Krallensturm", "pose": "raising his staff"},
                  {"name": "Marvin", "pose": "standing"},
                  {"name": "Boss Frey", "pose": "blocking the path"}])
    c = _compose(plan, anchors=_anchors())
    scene = c.prompt.split("Exactly")[0]
    reported = {f.subject for f in c.findings if f.code == "duplicate_mention"}
    assert reported, [f.as_line() for f in c.findings]
    for name in reported:
        assert name in scene and scene.count(name) > 1, \
            f"{name} was reported but the prompt does not repeat it"
    assert "aims her bow" in scene, "the sentence is not touched at all"


RETRIEVED_FREY = ("Boss Frey, der Bandit, überfiel die Gruppe an der Old Bridge und "
                  "verlangte den Zoll für die überfallene Händlerin.")


def test_a_named_character_left_in_entities_is_reported():
    """Measured on the live bandit request: the model filed BOSS FREY and ALISA
    PENNYGLEAM under "Also in the scene" objects, so the count line said fewer
    figures than the prompt named — the shape that paints an extra crowd.

    Retired 2026-11-19 (§17.3): promoting them meant deciding, from a regex, that
    a word is a character rather than a place or a thing. The bot now checks the
    one signal that needs no guessing — the USER named them — and asks for a
    figure line. Meanwhile the count line still counts the figure lines it printed.
    """
    plan = _plan("Boss Frey and Alisa Pennygleam stand before them; the party "
                 "faces them",
                 [{"name": "Seta Onix", "pose": "drawing her bow"},
                  {"name": "Marvin", "pose": "raising a pincer"}],
                 entities=["Boss Frey and his bandit crew", "Alisa Pennygleam"])
    c = _compose(plan, request_text="Boss Frey und Alisa Pennygleam stellen sich "
                                   "der Gruppe")
    reported = [f for f in c.findings if f.code == "entity_named_not_painted"]
    text = " | ".join(f.subject + " :: " + f.detail for f in reported)
    assert "Boss Frey" in text and "Alisa Pennygleam" in text, text
    assert len(reported) == 2, [f.as_line() for f in c.findings]
    # Nothing was promoted and nothing was removed — the plan's own structure is
    # what the renderer sees, and the disagreement is what the model gets back.
    assert "Also in the scene: Boss Frey and his bandit crew, Alisa Pennygleam." \
        in c.prompt, c.prompt
    assert "Boss Frey and Alisa Pennygleam stand before them" in c.prompt, c.prompt
    assert "Exactly 2 named figures" in c.prompt, c.prompt


def test_a_place_name_in_entities_is_not_painted_as_a_character():
    plan = _plan("Trixy Smoldersome works at the forge of Winnowing Reach",
                 [{"name": "Trixy Smoldersome", "pose": "hammering a bracket"}],
                 entities=["Winnowing Reach", "a coal forge"])
    c = _compose(plan)
    assert "Winnowing Reach:" not in c.prompt, c.prompt
    assert "Exactly 1 named figure" in c.prompt, c.prompt
    assert "Winnowing Reach" in c.prompt   # still in the scene sentence


def test_a_place_the_user_named_is_still_a_place_and_not_an_unpainted_person():
    """Live 2026-11-20: the Old Bridge request produced
    `entity_named_not_painted [the weathered mossy stone arch of the Old Bridge] —
    the user asked for 'old bridge' and this files them under objects`, a report
    that asked the correction turn to give a stone arch a pose. Naming a place in
    the request is not naming a character: the test is whether the plan USES the
    name as the entry ("Boss Frey and his bandit crew") or only mentions it inside
    a description of something else."""
    plan = _plan("Boss Frey bars the way across the Old Bridge", [{"name": "Marvin",
                 "pose": "rolling to a halt"}],
                 entities=["the weathered mossy stone arch of the Old Bridge",
                           "sputtering torches on iron brackets"])
    c = _compose(plan, request_text="Boss Frey stellt sich der Gruppe auf der "
                                    "Old Bridge")
    reported = [f.subject for f in c.findings
                if f.code == "entity_named_not_painted"]
    assert reported == [], reported
    # The guard is position, not vocabulary: the same place name filed AS the entry
    # is still reported — with nothing in the entry to tell a place from a person,
    # that residual is the price of keeping the real case (a character lost to the
    # props) caught. It costs a correction turn, never an image.
    c2 = _compose(_plan("the party faces them at the crossing",
                        [{"name": "Marvin", "pose": "rolling to a halt"}],
                        entities=["Old Bridge", "Boss Frey and his bandit crew"]),
                 request_text="Boss Frey auf der Old Bridge")
    assert [f.subject for f in c2.findings
            if f.code == "entity_named_not_painted"] == [
        "Old Bridge", "Boss Frey and his bandit crew"]


# ── 9. A replacement must not re-name someone the sentence already names ─────

# ── 10. The crowd sentences were the last unpoliced place a species could leak ─

_CROWD = "Two lanky weasel-folk bandits block the path while Seta Onix draws her bow"


# ── 11. A name the plan abbreviated is painted with the name the campaign uses ─

def test_a_plan_that_wrote_a_name_short_is_told_the_campaigns_spelling():
    """Measured live 2026-11-19: the plan wrote "name": "Frey" for the character
    the request and the vault both call Boss Frey, so the prompt said "Frey:" —
    a different string from the one the party plays and every earlier image used.

    Report, not rename: the three facts are the request, the retrieved material and
    the plan's own name, and the model that read the request can write the spelling
    itself. Renaming from outside was another rewrite the table never asked for.
    """
    plan = _plan("Frey snarls forward while Seta Onix draws her bow",
                 [{"name": "Seta Onix", "pose": "drawing her bow"},
                  {"name": "Frey", "pose": "snarling forward"}])
    c = ip.compose_prompt(
        plan, _anchors(), BASE, world_rule=WORLD_RULE,
        retrieved="Boss Frey, der Bandit, überfiel die Gruppe an der Old Bridge.",
        request_text="Boss Frey stellt sich der Gruppe in den Weg")
    f = _finding(c, "name_form", "Frey")
    assert "Boss Frey" in f.detail
    assert "Frey: snarling forward" in c.prompt, c.prompt


def test_a_plan_that_listed_one_person_twice_is_told_so():
    """Without an external rename, "Boss Frey" and "Frey" are two figure lines.
    That is reported rather than silently merged: the merge itself used to rewrite
    the plan's cast, and a plan that really means two different bandits must not
    have one deleted by a regex."""
    plan = _plan("Boss Frey snarls forward",
                 [{"name": "Boss Frey", "pose": "snarling forward"},
                  {"name": "Frey", "pose": "blocking the path"}])
    c = ip.compose_prompt(plan, _anchors(), BASE, world_rule=WORLD_RULE,
                          retrieved="Boss Frey, der Bandit.",
                          request_text="Boss Frey blockt den Weg")
    f = _finding(c, "same_person_listed_twice")
    assert "Boss Frey" in f.detail and "Frey" in f.detail
    assert c.figures == 2, "nothing is merged — the model decides who is who"


def test_a_capitalised_german_noun_in_front_of_a_name_is_not_read_as_part_of_it():
    """German capitalises every noun, so "die befreite Händlerin Alisa Pennygleam"
    offers "Händlerin Alisa" as a longer name. It is only accepted when the vault
    writes the name that way too."""
    plan = _plan("Alisa Pennygleam clutches a coin bag",
                 [{"name": "Alisa Pennygleam", "pose": "clutching a coin bag"}])
    c = ip.compose_prompt(
        plan, _anchors(), BASE,
        retrieved="Alisa Pennygleam, eine Händlerin aus Clearwater.",
        request_text="die befreite Händlerin Alisa Pennygleam klammert den Beutel")
    assert "Händlerin Alisa" not in c.prompt, c.prompt
    assert not any("campaign's name" in w for w in c.warnings), c.warnings


def test_no_rename_when_the_plan_already_uses_the_campaign_name():
    plan = _plan("Boss Frey snarls forward",
                 [{"name": "Boss Frey", "pose": "snarling forward"}])
    c = ip.compose_prompt(plan, _anchors(), BASE,
                          retrieved="Boss Frey, der Bandit.",
                          request_text="Boss Frey blockt den Weg")
    assert not any("campaign's name" in w for w in c.warnings), c.warnings


def test_a_party_member_abbreviated_in_the_scene_is_reported():
    """Measured live 2026-11-19 (§15): the figure lines of one prompt said "Seta
    Onix, Romi Stachelpfote" while its scene sentence said "Seta, Bubo, Romi,
    Colum, Trixy". The renderer reads the whole prompt, and a name that looks
    different from the one already described is a name it paints again.

    The FIGURE LINES still take the sheet's spelling (that is identity: "Seta" and
    "Seta Onix" must dedupe to one figure). The sentence is reported."""
    plan = _plan("Seta, Romi and Marvin hold the path while a bandit blocks it",
                 [{"name": "Seta", "pose": "drawing her bow"},
                  {"name": "Romi Stachelpfote", "pose": "planting her horn"},
                  {"name": "Marvin", "pose": "rolling forward beside Romi"}])
    c = _compose(plan)
    assert "Seta, Romi and Marvin hold the path" in c.prompt, c.prompt
    # The figure line takes the SHEET spelling, and its identity clause leads the
    # block with the pose following it (analysis §18).
    assert "Seta Onix: shoosuva" in c.prompt, c.prompt
    assert (c.prompt.index("Seta Onix: shoosuva")
            < c.prompt.index("drawing her bow")), c.prompt
    reported = {(f.subject, f.detail) for f in c.findings if f.code == "name_spelling"}
    assert any("Seta Onix" in d for _, d in reported), reported
    assert c.figures == 3, c.prompt


def test_a_pose_that_named_another_member_shortly_is_reported():
    plan = _plan("Marvin rolls forward while Trixy works the bellows",
                 [{"name": "Marvin", "pose": "standing alert beside Trixy"},
                  {"name": "Trixy", "pose": "working the bellows"}])
    c = _compose(plan)
    f = _finding(c, "name_spelling", "Marvin")
    assert "Trixy Smoldersome" in f.detail
    assert "beside Trixy" in c.prompt, "the pose is the model's wording"


# ── 12. Nobody in this world is a person (table owner's ruling, 2026-11-19) ──

def test_the_world_rule_line_rides_in_the_style_sentence():
    """The ruling: "we only want animals in the final output". A cast member nobody
    describes would otherwise be painted as the wrong kind of being, so the campaign's
    own sentence about what its beings are goes in the style sentence — the one line the budget ladder never
    cuts — rather than in a sentence that can be trimmed away."""
    plan = _plan("Bandits block the path while Seta Onix draws her bow",
                 [{"name": "Seta Onix", "pose": "drawing her bow"}])
    c = _compose(plan)
    assert "Every figure in this image is an anthropomorphic animal-folk character" in c.prompt
    head = c.prompt.split(". Bandits block")[0]
    assert "animal-folk" in head, "the campaign sentence must be style-level, not scene-level"
    assert ".." not in c.prompt, c.prompt


def test_the_world_rule_line_survives_the_tightest_budget():
    plan = _plan("Seta Onix draws her bow at a bandit",
                 [{"name": "Seta Onix", "pose": "drawing her bow"}],
                 mood="tense dusk on the Old Bridge", palette="amber and slate",
                 lighting="low dusk light")
    c = _compose(plan, budget=320)
    assert "anthropomorphic animal-folk" in c.prompt, c.prompt
    assert "Mood:" not in c.prompt and "Palette:" not in c.prompt, c.prompt


def test_the_world_rule_line_can_be_switched_off():
    plan = _plan("Seta Onix draws her bow", [{"name": "Seta Onix",
                                             "pose": "drawing her bow"}])
    c = ip.compose_prompt(plan, _anchors(), BASE, world_rule="",
                          budget_kwargs={"base": 1100, "per_figure": 420,
                                         "hard_cap": 100000})
    assert "Every figure in this image is an anthropomorphic" not in c.prompt, c.prompt




# ── Round 2 of the evaluation run (§17): what the D-set prompts showed.

def test_a_note_about_the_pipeline_is_reported_not_cut_out():
    """A live scene sentence read: 'Boss Frey, a badger-folk bandit chief
    (species from the bot's own guess)'. The contract asks for a report, and the
    model wrote the report where the renderer would paint it.

    Retired 2026-11-19 (§17.3): stripping parentheticals was a regex deciding which
    bracket was commentary and which was a description. The finding says it out loud
    and the model removes its own note."""
    plan = ip.ImagePlan(scene="Boss Frey, a bandit chief (species from the bot's "
                              "own guess), stands alone",
                        figures=[ip.Figure(name="Boss Frey", pose="standing alone")])
    c = ip.compose_prompt(plan, _anchors(), BASE, world_rule=WORLD_RULE)
    assert "bot's own guess" in c.prompt
    f = _finding(c, "meta_comment", "scene")
    assert "bot's own guess" in f.detail


# A search result arrives as ONE line: `[{"file": …, "snippet": …}, {"file": …}]`, so
# the block that "names Boss Frey" is the blob — and the blob holds Bubo's owl, a
# blue tit's own sheet and a deer-folk bounty, none of them about him.
_BLOB = ('[{"file": "humblewood/…/SESSION_01_DerAnfang/notes.md", "section": "Full '
         'Document", "snippet": "Session 01: Ashbarrow fire, Frey bandits, Alisa '
         'Pennygleam rescue. A wolf track leads into the forest, and at the wood edge '
         'the party heals a wounded wolf."}, {"file": "humblewood/…/Player '
         'Characters.md", "section": "Bubo Krallensturm -> Appearance", "snippet": '
         '"An owl, upright and robed, snow-white facial disc, amber eyes."}, '
         '"4 location(s) contain Frey"}, {"file": "…/SESSION_05/notes.md", '
         '"section": "Lexikon: Personen / Banditen", "snippet": "Godfather: Ein Luma, '
         'der 400 Gold Kopfgeld kostet."}]')


def test_the_budget_never_cuts_a_crowd_the_request_itself_named():
    """The D run proved this gap: 'Boss Frey und seinen Banditen' came out with
    no bandits in the frame at all — the entity line was the budget's first
    casualty."""
    long_anchor = ("deep purple fur covering her entire body, digitigrade cloven "
                   "hooves, long curved goat-like horns, glowing amber eyes, "
                   "rectangular dark glasses, dark green cloak with a deep emerald "
                   "hood, studded leather armor beneath, a longbow with an elegant "
                   "antler-like design resting on her back")
    plan = ip.ImagePlan(
        scene="bandits bar the forest path",
        entities=["two rat-folk bandits with crude axes", "a mossy old bridge"],
        figures=[ip.Figure(name="Seta Onix", pose="drawing her bow",
                           look=long_anchor)])
    c = ip.compose_prompt(plan, _anchors(), BASE,
                          budget_kwargs={"base": 300, "per_figure": 120,
                                         "hard_cap": 1000},
                          request_text="Boss Frey und seinen Banditen")
    assert "bandit" in c.prompt, c.prompt
    assert "mossy old bridge" not in c.prompt, c.prompt
    assert any("trimmed to what the request named" in w for w in c.warnings), c.warnings


def test_a_pose_carrying_a_collective_object_is_reported():
    """Round 6 of the evaluation run: 'Boss Frey: snarling forward, sabre leveled
    at the party' — a cast member named twice is a character painted twice (§4.2).
    The composer used to delete the crowd word; the pose is left as written and the
    disagreement is reported instead."""
    plan = _plan("Boss Frey bars the forest path",
                 [{"name": "Boss Frey", "pose": "snarling forward, sabre leveled at "
                                               "the party"},
                  {"name": "Seta Onix", "pose": "drawing her bow"}])
    c = _compose(plan)
    f = _finding(c, "collective_in_figure", "Boss Frey")
    assert "the party" in f.detail
    assert "Boss Frey: snarling forward, sabre leveled at the party" in c.prompt


# ── Whose animal is it? (rounds 4–7 of the evaluation run, §17) ─────────────

RETRIEVED_LIZARD_MERCHANT = (
    "HumbleWood Session 5 Trunk Market: Romi leaves, lizard merchant Spleck "
    "sells potions.")
RETRIEVED_BANDIT_RATS = (
    "SESSION_01: Boss Frey führt seine Banditen: three rat-folk cutthroats an "
    "der Old Bridge.")


def _crowd_case(scene, retrieved, look, request, *, source):
    anchors = _anchors()
    return ip.compose_prompt(
        ip.ImagePlan(scene=scene,
                     figures=[ip.Figure(name="Alisa Pennygleam", look=look,
                                        source=source)]),
        anchors, BASE,
        read_paths=[source], retrieved=retrieved, request_text=request,
        budget_kwargs={"base": 1100, "per_figure": 420, "hard_cap": 100000})


def _blocks(prompt: str, names) -> dict[str, str]:
    """One figure's block text, read out of the composed prompt.

    A block starts at its own `Name: ` label and ends at the next label or at the
    tail sentence. `compose_prompt` writes those labels, so this is reading its
    output, not guessing at it.
    """
    out: dict[str, str] = {}
    for i, n in enumerate(names):
        start = prompt.index(f"{n}: ")
        nxt = [prompt.index(f"{m}: ", start + 1) for m in names[i + 1:]]
        nxt.append(prompt.index(ip.NO_LETTERING_TAIL))
        out[n] = prompt[start:min(nxt)]
    return out


def test_what_a_figure_is_leads_its_block_before_what_it_is_doing():
    """Measured 2026-11-19 (analysis §18): the identity clause goes first.

    Eight characters, their Visual Anchors complete, the same words in prompts of
    EXACTLY the same length — only the identity clause's position differed. Every
    render whose block opened with the pose lost a figure (5 of 5: Bubo absent from
    four prompts that named him, the owl absent outright from a fifth, where his
    staff was painted into the raccoon's hands). Every render whose block opened
    with the identity painted the whole cast (4 of 4; Fisher exact p = 0.008). The
    fix is one clause order, not more detail: cutting nothing at all did NOT bring
    Bubo back.

    A diffusion model decides WHAT to draw before what that thing is busy with. A
    block opening "Bubo Krallensturm: staff planted, crescent finial flaring light,
    a small owl archmage-cleric…" is an instruction about a staff.
    """
    plan = _plan("Sable Rin holds the gate while Pell Wicker ropes the cart", [
        {"name": "Seta Onix", "pose": "drawing her longbow"},
        {"name": "Colum Bidae", "pose": "crackling blue sparks between his claws"},
        {"name": "Marvin", "pose": "rolling forward beside Seta"},
    ])
    c = _compose(plan)
    blocks = _blocks(c.prompt, ["Seta Onix", "Colum Bidae", "Marvin"])
    anchors = _anchors()
    poses = {"Seta Onix": "drawing her longbow",
             "Colum Bidae": "crackling blue sparks between his claws",
             "Marvin": "rolling forward beside Seta"}
    for name, pose in poses.items():
        # The identity leads — the sheet's own species line, not a copy this test
        # retypes by hand (Marvin's "NOT an animal" tail is dropped by the machine
        # rule, so the first clause is what has to lead).
        head = anchors[name].fields["species"].split(",")[0].strip()
        block = blocks[name]
        assert head in block, f"{name}: {block[:150]}"
        assert block.split(": ", 1)[1].startswith(head), block[:150]
        assert block.index(head) < block.index(pose), (
            f"{name}'s pose is said before its identity: {block[:150]}")


def test_an_identity_override_leads_the_block_like_the_field_it_replaces():
    """A `modify` on `species` replaces the identity, so it inherits the position."""
    plan = _plan("Trixy Smoldersome faces the raiders alone", [{
        "name": "Trixy Smoldersome", "pose": "planting her lantern horn",
        "modify": {"species": "a fox-folk courier"}}])
    c = _compose(plan)
    block = _blocks(c.prompt, ["Trixy Smoldersome"])["Trixy Smoldersome"]
    assert block.split(": ", 1)[1].startswith("a fox-folk courier"), block[:160]
    assert "planting her lantern horn" in block
    assert block.index("a fox-folk courier") < block.index("planting her lantern horn")


def test_a_figure_with_no_identity_at_all_still_leads_with_its_pose():
    """The ladder cannot invent an identity to lead with: Boss Frey has none here, so
    his block stays pose-first rather than gaining a guess the vault never made."""
    plan = _plan("Boss Frey blocks the path", [{"name": "Boss Frey",
                                                "pose": "snarling forward"}])
    c = _compose(plan)
    assert "Boss Frey: snarling forward" in c.prompt, c.prompt


def test_a_description_left_in_the_scene_sentence_is_reported_as_out_of_reach():
    """Measured live 2026-11-19 (§18): "Boss Frey, a badger-folk bandit chief, stands
    alone in a mossy forest clearing" arrived together with `"look": ""`, so his figure
    line went out as "Boss Frey: raising a heavy battle axe overhead" — the description
    existed in the prompt, just not where the renderer reads it.

    Carrying it across would be editing the model's language. Pointing at it is not.
    """
    plan = _plan("Boss Frey, a badger-folk bandit chief, stands alone under the moon",
                 [{"name": "Boss Frey", "pose": "raising a heavy battle axe"}])
    c = _compose(plan)
    codes = [f.code for f in c.findings]
    assert "identity_left_in_scene" in codes, c.findings
    f = next(f for f in c.findings if f.code == "identity_left_in_scene")
    assert f.subject == "Boss Frey"
    assert "badger-folk bandit chief" in f.detail, f.detail
    assert f.code in ip.FIXABLE_CODES
    assert "figure_ungrounded" not in codes, (
        "the plan did describe him — in the wrong field, which is a different complaint")


def test_a_name_the_plan_described_nowhere_is_still_reported_plainly():
    plan = _plan("Boss Frey blocks the path", [{"name": "Boss Frey",
                                                "pose": "snarling forward"}])
    c = _compose(plan)
    codes = [f.code for f in c.findings]
    assert codes == ["figure_ungrounded"], c.findings


def test_a_scene_description_is_not_hunted_for_where_there_is_none():
    """_scene_description returns "" rather than a guess: a name followed by a verb is
    not a description, and an apposition that only repeats the name says nothing."""
    assert ip._scene_description("Boss Frey blocks the path", "Boss Frey") == ""
    assert ip._scene_description("Bubo, an owl archmage, raises his staff",
                                 "Bubo") == "an owl archmage"
    assert ip._scene_description("Seta, Seta and Romi hold the gate", "Seta") == ""

def test_the_plan_contract_asks_for_a_species_word_without_promising_a_checker():
    """Two different things the plan prompt used to say in one sentence.

    "Name a SPECIES, a bare role word is painted as a person" is a fact about the
    renderer (§15/§16 measured it) and stays — it is language guidance for the model.
    "A species the vault states always wins and is never replaced; an invented one is
    reported as the bot's own guess" described a CHECKER that §20 deleted. A prompt may
    not promise behaviour the code no longer performs: the table would read the
    guarantee into a log line that never appears.
    """
    raw = images._PLAN_FILE.read_text(encoding="utf-8")
    assert "SPECIES WORD" in raw
    assert "paints a bare role word as a person" in raw
    for retired in ("always wins and is never replaced",
                    "reported to the table as the bot's own guess",
                    "species the vault states"):
        assert retired not in raw, retired


# ── §22: a budget cut must never create the dangling-name shape ──────────

# The live case, 2026-11-24, "/image Boss Grey, der Anführer der Banditen auf der
# Old Bridge": a 53-word scene sentence, a cap of 45. The plan's own words put a
# cast member's NAME at word 45 and its verb past the end, and the composer's trim
# kept exactly the first 45 — shipping "…grips a gear, Marvin." as the end of the
# scene. A name with no action is the shape the renderer skips (§17, §18).
_LIVE_LONG_SCENE = (
    "Boss Grey, a badger-folk bandit chief, stands at the far end of the old wooden "
    "bridge, blocking the crossing while Bubo Krallensturm raises his staff, Seta Onix "
    "crouches low on the planks, Romi Stachelpfote sways in a low stance, Trixy "
    "Smoldersome grips a gear, Marvin stands at the rail with his pot hook")
_LIVE_NAMES = ["Boss Grey", "Bubo Krallensturm", "Seta Onix", "Romi Stachelpfote",
               "Trixy Smoldersome", "Marvin"]


def test_the_old_word_cut_is_the_bug_this_replaces():
    """Keeping the first `limit` words landed mid-clause, on a name."""
    cut = " ".join(_LIVE_LONG_SCENE.split()[:45]).rstrip(" ,;.")
    assert cut.endswith("Marvin"), cut


def test_a_scene_trim_cuts_whole_clauses_and_never_leaves_a_name_without_a_verb():
    kept, gone = ip._trim_scene(_LIVE_LONG_SCENE, _LIVE_NAMES, ip._MAX_SCENE_WORDS)
    assert len(kept.split()) <= ip._MAX_SCENE_WORDS
    assert not any(kept.rstrip().endswith(n) for n in _LIVE_NAMES), kept
    assert gone == ["Marvin stands at the rail with his pot hook"], gone
    # The survivors are still a sentence, punctuation and all.
    assert kept.startswith("Boss Grey, a badger-folk bandit chief, stands")


def test_a_scene_trim_keeps_the_setting_and_a_nameless_characters_only_description():
    """The two clauses a figure line cannot replace.

    The setting ("at the far end of the old wooden bridge") appears nowhere else in
    the prompt, and "a badger-folk bandit chief" is the ONLY appearance Boss Grey
    has — the vault has no notes on him, so that apposition is the whole identity
    claim. A trim that cut either would trade a fact for a repetition.
    """
    kept, _ = ip._trim_scene(_LIVE_LONG_SCENE, _LIVE_NAMES, ip._MAX_SCENE_WORDS)
    assert "at the far end of the old wooden bridge" in kept
    assert "a badger-folk bandit chief" in kept


def test_a_composed_scene_over_the_cap_reports_the_clause_it_dropped():
    plan = _plan(_LIVE_LONG_SCENE,
                 [{"name": n, "pose": "holding the line"} for n in _LIVE_NAMES])
    c = _compose(plan)
    warns = " ".join(c.warnings)
    assert "trimmed to the contract's" in warns and "by dropping the tail clauses" in warns, c.warnings
    assert "Marvin stands at the rail with his pot hook" in warns, c.warnings
    kept, _ = ip._trim_scene(_LIVE_LONG_SCENE, _LIVE_NAMES, ip._MAX_SCENE_WORDS)
    assert kept in c.prompt and "with his pot hook" not in c.prompt


def test_the_clause_cap_empties_the_fattest_block_before_touching_a_thin_one():
    """Equal treatment, measured: 8/8 at 587 tokens and 12/12 at 810 (the ceiling
    ladder) versus a shipped prompt that spent 124 tokens on Bubo and 16 on Benner."""
    fat = ip.Anchor("Colum Bidae", fields={
        "species": "a small blue tit with amber eyes",
        "body": ", ".join(f"a body clause number {i} of nine here" for i in range(9)),
        "outfit": ", ".join(f"an outfit clause number {i}" for i in range(5)),
        "signature": "a small leather spell pouch with bone clasps"})
    thin = ip.Anchor("Marvin", fields={
        "species": "a hedgehog cook", "look": "stocky, flour-dusted",
        "signature": "a battered pot hook"})
    anchors = {"Colum Bidae": fat, "Marvin": thin}
    plan = ip.ImagePlan(scene="A bridge at dusk",
                        figures=[ip.Figure("Colum Bidae", "hovering"),
                                 ip.Figure("Marvin", "cooking")])
    capped = _compose(plan, anchors=anchors, budget=700, clause_cap=4)
    uncapped = _compose(plan, anchors=anchors, budget=700, clause_cap=0)
    fat_cuts = sum(1 for w in capped.cuts if "Colum Bidae" in w)
    thin_cuts = sum(1 for w in uncapped.cuts if "Marvin" in w)
    clause_cuts = [w for w in capped.cuts if "[body]" in w or "[outfit]" in w]
    assert fat_cuts, capped.cuts
    assert clause_cuts and all("Colum Bidae" in w for w in clause_cuts), capped.cuts
    assert thin_cuts, uncapped.cuts       # the old ladder reached into the thin block
    assert "spell pouch" in capped.prompt and "pot hook" in capped.prompt
    for name in ("Colum Bidae", "Marvin"):
        assert name in capped.prompt, "the cap balances blocks, it never drops figures"


def test_the_clause_cap_defaults_to_today_behaviour():
    plan = _plan("A bridge at dusk", [{"name": "Marvin", "pose": "cooking"}])
    assert _compose(plan).prompt == _compose(plan, clause_cap=0).prompt
