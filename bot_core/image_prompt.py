"""The deterministic image-prompt composer (#84, analysis doc §9.3).

The rewrite model no longer writes the prompt the renderer sees — it writes a
small **plan**, and this module writes the prompt. Everything the policy used to
*ask* for (the style sentence first, one description per character, a figure
count, no collective subject, no negations, a character budget) becomes a
property of the code that assembles the text, so it is testable instead of
hoped for.

Why (measured in docs/internal/IMAGE_PROMPT_PIPELINE_ANALYSIS_2026-11.md):
§3 — six Visual Anchors share 34 colour/material tokens, 'gold' and 'blue' each
owned by three of them, so a colour in one character's description is not a
unique handle; §9.2 — a MODIFIED character ("Seta in metal armour instead of her
cloak") could only be phrased as a negation or as "replacing her usual cloak",
which the renderer cannot execute because it has no memory of her default
outfit. Typed anchor FIELDS fix that: an override replaces exactly one field and
the composer omits the field it replaces. Signature items (the bow, the goggles,
Marvin's built-in tools) are a separate class, so a clothing override cannot
reach them.

Two data layers:
- the **vault** holds each Visual Anchor as field lines under `## Visual Anchor`
  (`> outfit: …`); the legacy single-paragraph quote still parses, as one opaque
  `blob`, and paints verbatim as before.
- the **plan** (the rewrite model's reply, inside an `<image_plan>` block) is
  `{scene, figures[], entities[], lighting, palette, mood}` where a figure is
  `{name, pose, modify{field: text}}` for a party member and
  `{name, pose, look, source}` for anyone else.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any

log = logging.getLogger(__name__)

# The typed fields an anchor carries, in the order they are painted: what a
# character IS, then body, face, clothing, carried items, quirks.
ANCHOR_FIELDS: tuple[str, ...] = (
    "species", "body", "face", "outfit", "signature", "quirks",
)

# Vault field labels accepted for each field (lowercased, prefix-matched).
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "species": ("species", "kind", "creature", "race", "ancestry", "type"),
    "body": ("body", "build", "frame", "physique", "size", "height"),
    "face": ("face", "head", "eyes", "nose", "muzzle", "features"),
    "outfit": ("outfit", "clothing", "clothes", "attire", "dress", "wearing",
               "gear", "armour", "armor"),
    "signature": ("signature", "signature item", "signature items", "items",
                  "carries", "carried items", "props", "weapons", "accessories"),
    "quirks": ("quirks", "mannerisms", "marks", "distinguishing", "tells",
               "notes", "symbol", "symbols"),
}

# A clause that NEGATES. Diffusion models paint what a positive prompt NAMES, so
# "NO cape" is worse than silence — the composer's answer to a removal request is
# to OMIT the field and never announce the omission.
_NEGATION_CUES = (
    "no", "not", "none", "without", "instead of", "instead", "rather than",
    "never", "avoid", "omit", "removed", "removing", "lacks", "missing",
    "ohne", "kein", "keine", "statt", "ohne seinen", "ohne ihre", "n't",
)

# Word-boundary matching, not substring matching: "(no hair on the head)" is a
# negation even though a space-anchored search misses it behind the bracket, and
# "no" must not match inside "nostrils" or a name like "Noh".
_NEGATION_RES = tuple(
    re.compile(r"(?<![A-Za-zÀ-ÿ])" + re.escape(cue) + r"(?![A-Za-zÀ-ÿ])")
    for cue in _NEGATION_CUES
)

# Crowd nouns naming the cast as a whole — the mechanism behind the duplicated
# and missing-character renders: a collective subject makes the model paint an
# unnamed crowd ON TOP of the characters listed one by one afterwards (§4).
_COLLECTIVE_PATTERNS = (
    r"\bthe party\b", r"\bparty members\b", r"\bwhole party\b", r"\bparty\b",
    r"\bpartei\b", r"\btruppe\b", r"\bthe group\b", r"\bthe gang\b",
    r"\ba group of\b", r"\bthe companions\b", r"\bthe crew\b",
    r"\bsome adventurers\b", r"\bseveral adventurers\b", r"\ball of the\b",
    r"\bdie gruppe\b", r"\bdie ganze gruppe\b", r"\bdie helden\b",
)

# The subset that is SAFE to substitute with the cast's names: a collective with a
# determiner in front of it ("the whole upright party", "die ganze Gruppe") is
# naming THIS cast. "a party of bandits" is naming somebody else and is only
# reported. Substitution is the fix for measured failure 6 — the scene named the
# cast collectively AND ten anchors followed, and the crowd noun painted extra
# unnamed bodies on top of them.
_COLLECTIVE_REPLACE = (
    r"\b(?:the|die|alle|all of the|ihre|seine|ire)\s+(?:[a-z]+\s+){0,3}?"
    r"(?:party|partei|gruppe|gang|truppe|companions|crew|helden|adventurers|group)"
    r"(?:\s+members)?\b",
)

_PALETTE_WORDS = frozenset((
    "gold", "golden", "silver", "brass", "steel", "copper", "bronze", "olive",
    "brown", "grey", "gray", "black", "white", "cream", "tan", "taupe", "indigo",
    "navy", "blue", "turquoise", "teal", "green", "emerald", "crimson", "red",
    "orange", "amber", "yellow", "purple", "violet", "pink", "ivory", "leather",
    "fur", "feather", "feathered", "silk", "linen", "wool", "beige", "charcoal",
))

# Words that mark a blob clause as belonging to a field (legacy-anchor overrides,
# where there is no field to replace and clauses must be filtered instead).
_SUBJECT_CUES: dict[str, tuple[str, ...]] = {
    "outfit": ("cloak", "cape", "tunic", "robe", "mantle", "hood", "boot",
               "boots", "glove", "gloves", "gauntlet", "bracer", "bracers",
               "belt", "satchel", "strap", "straps", "armor", "armour",
               "jerkin", "vest", "trousers", "pants", "sandals", "hat", "cap",
               "bands", "harness", "pauldron", "scarf", "tattered", "worn"),
    "signature": ("bow", "staff", "horn", "pendant", "goggles", "lantern",
                  "instrument", "sword", "dagger", "axe", "bag", "pouch",
                  "quiver", "tool", "flute", "hammer", "charm", "rose"),
    "face": ("eye", "eyes", "nose", "muzzle", "brow", "cheek", "face", "beak"),
    "body": ("fur", "wings", "tail", "ears", "head", "build", "paws", "claws",
             "horns", "frame", "body", "wingspan", "quills", "limbs"),
    "species": ("mammal", "bird-folk", "beast-folk", "lizard", "automaton",
                "construct", "creature", "folk", "raccoon", "hedgehog", "owl"),
    "quirks": ("always", "habitually", "posture", "recurring", "symbol"),
}

_SIGNATURE_CUES = (
    "bow", "staff", "horn", "pendant", "goggles", "lantern", "instrument",
    "sword", "dagger", "axe", "quiver", "flute", "hammer", "charm", "rose",
    "built into", "table", "hook",
)

# The ONE negation the composer itself writes, and the only one: AI-artifact
# lettering is an artifact the model produces unasked, not an object a denial
# would paint, so the "denials paint the thing" rule (analysis §3) does not apply
# here. Readable lettering a USER asks for is still possible — the scene sentence
# names it — and IMAGE_NEGATIVE_PROMPT deliberately carries no blanket text
# suppression for the same reason.
NO_LETTERING_TAIL = "No text, lettering, logos or watermarks."

# The table owner's ruling, 2026-11-19: "we only want animals in the final output"
# — this campaign's world is animal-folk, so a character the vault never describes
# must still be painted as one rather than as a person. The anchors carry their own
# species; this line is the floor under everyone else, and it rides in the style
# sentence, which the budget ladder is not allowed to cut.
_ANIMAL_FLOOR = ("Every figure in this image is an anthropomorphic animal-folk "
                 "character of this world")
# …with one exception the vault states outright: Marvin is a Steel Defender, a
# machine. "Every figure is animal-folk" next to "brass-and-steel automaton" is a
# contradiction the renderer resolves by painting whichever it reads last, so the
# floor says LIVING whenever someone in the frame is described as a machine. The
# switch is made from the sheet's own wording, never from a guess about a name.
_ANIMAL_FLOOR_LIVING = _ANIMAL_FLOOR.replace("Every figure", "Every living figure")
_MACHINE_WORDS = ("automaton", "construct", "machine", "golem", "robot", "robotic",
                  "clockwork", "defender", "android", "homunculus")


def is_machine(*texts: str) -> bool:
    """Does any of these descriptions say the figure is a made thing, in the
    vault's or the plan's own words?"""
    normed = " ".join(_norm(t) for t in texts)
    return any(re.search(r"\b" + w + r"\b", normed) for w in _MACHINE_WORDS)


def _identity_signal(text: str) -> bool:
    """Does this description already say WHAT the figure is, species-wise?

    A species word (badger, ratte, owl) or a generic identity the vault uses for a
    non-animal character (construct, tiefling, demon, beast-folk).
    """
    if _claimed_species(text):
        return True
    normed = _norm(text)
    return any(re.search(r"\b" + re.escape(w) + r"\b", normed)
               for w in _GENERIC_SPECIES)

# The scene-sentence word cap, straight from the plan contract. A runaway scene
# sentence is measured failure 5 (an 11-figure request became 3,468 characters of
# mush with no anchors left), so the composer enforces the cap rather than asking
# for it.
_MAX_SCENE_WORDS = 45
_MAX_POSE_WORDS = 12      # the plan contract's own numbers, now checked here too
_MAX_LOOK_WORDS = 25

# The budget scales with the cast, because the anchors ARE the content: six full
# Visual Anchors plus a scene line measure 3,566 chars, and the repo's own policy
# says "including every requested character BEATS staying under the char cap".
_BASE_CHARS = 1100        # style + scene + count + tail, measured
_CHARS_PER_FIGURE = 420   # one anchor's worth of identity
_HARD_CAP = 3600          # the size the current pipeline has always held


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_collective(text: str) -> bool:
    """True when *text* names the cast with a crowd noun instead of a person."""
    t = _norm(text)
    return any(re.search(p, t) for p in _COLLECTIVE_PATTERNS)


# A collective inside a FIGURE LINE cannot be swapped for the cast's names: the
# names are already the subject of that very line, and a second mention is what
# painted a duplicate Colum (§4.2). There the phrase is simply removed.
_COLLECTIVEPhrase = re.compile(
    r"\b(?:at|toward|towards|against|upon|before|behind|among|amid|past|round)\s+"
    r"(?:the|his|her|their)\s+(?:party|group|company|band|crowd|mob|rabble)\b"
    r"|\b(?:the|his|her|their)\s+(?:party|group|company|crowd|mob)\b", re.IGNORECASE)


def drop_collectives(text: str) -> tuple[str, list[str]]:
    """Remove a crowd noun from a figure line, keeping the line grammatical."""
    out = text
    hits: list[str] = []
    for m in _COLLECTIVEPhrase.finditer(text or ""):
        hits.append(m.group(0))
    if hits:
        out = _COLLECTIVEPhrase.sub("", out)
        out = re.sub(r"\s+,", ",", out)
        out = re.sub(r"\s{2,}", " ", out)
        # "sabre leveled at" — the preposition lost its object and dangles.
        out = re.sub(r"\s+(?:at|toward|towards|against|upon|before|behind|among|"
                     r"amid|past|round)\s*$", "", out.strip().rstrip(",. "))
    return out.strip(), hits


def replace_collectives(text: str, names: list[str]) -> tuple[str, int]:
    """Swap a determiner-anchored crowd noun for the cast's own names.

    Grammar-safe: a subject is replaced by a subject. This is the part that
    actually stops the extra-figures render; detection without replacement would
    just log a failure that still got painted.

    A name the sentence ALREADY names is not inserted again (measured live:
    "the party confronts Boss Frey and his bandits" became "…Boss Frey and Alisa
    Pennygleam confronts Boss Frey and his bandits" — asking the model to paint
    the same character twice, which is the very defect this pass exists to stop).
    """
    out = text
    hits = 0
    inserted: list[str] = []
    for pat in _COLLECTIVE_REPLACE:
        def sub(m: re.Match, base: str = out) -> str:
            nonlocal hits
            hits += 1
            rest = base[:m.start()] + "\n" + base[m.end():]
            keep = [n for n in names if not _names_it(rest, n)]
            listed = _listed(keep) or _listed(names)
            inserted.append(listed)
            return listed or m.group(0)
        out = re.sub(pat, sub, out, flags=re.I)
    for listed in dict.fromkeys(inserted):
        # "the party faces them" → "Seta Onix, Marvin and Boss Frey faces them"
        # is what a literal substitution leaves behind. The verb was agreeing
        # with a collective noun; it has to agree with a list.
        esc = re.escape(listed)
        for bad, good in _VERB_AFTER_LIST.items():
            out = re.sub(rf"\b{esc}\s+{bad}\b", f"{listed} {good}",
                         out, flags=re.I)
            out = re.sub(rf"\b{esc} and\s+{bad}\b", f"{listed} {good}",
                         out, flags=re.I)
    return out, hits


def _listed(names: list[str]) -> str:
    if len(names) >= 2:
        return ", ".join(names[:-1]) + " and " + names[-1]
    return ", ".join(names)


def _names_it(text: str, name: str) -> bool:
    """Is this character already named in this sentence (including "Marvin's")?"""
    return bool(re.search(r"(?<![A-Za-z\u00c0-\u00ff])" + re.escape(name)
                          + r"(?![A-Za-z\u00c0-\u00ff])", text, re.I))


# Only curated verbs: an unknown word after the list is left alone rather than
# guessed at. Generated from stems so an s-form cannot be missed — the live run
# that produced "…and Alisa Pennygleam confronts Boss Frey" proved that a
# hand-written list always lacks the verb this request happened to use.
_VERB_STEMS = (
    "confront", "face", "battle", "fight", "attack", "assault", "charge",
    "advance", "circle", "surround", "encircle", "block", "watch", "move",
    "wait", "strike", "loom", "gather", "stand", "hold", "keep", "meet",
    "close", "point", "raise", "fire", "shoot", "draw", "ready", "nock",
    "leap", "run", "rush", "grab", "seize", "tackle", "restrain", "corner",
    "ambush", "defend", "guard", "shield", "support", "cover", "follow",
    "lead", "approach", "arrive", "hesitate", "freeze", "stare", "glare",
    "look", "turn", "step", "nod", "shout", "call", "speak", "sing",
    "crouch", "kneel", "hover", "float", "fly", "swing", "slash", "throw",
    "catch", "grip", "clutch", "aim", "bar", "hunt", "chase", "outnumber",
    "trap",
)


def _sforms(stem: str) -> list[str]:
    if stem.endswith(("s", "x", "z", "ch", "sh", "o")):
        return [stem + "es"]
    if stem.endswith("y") and stem[-2] not in "aeiou":
        return [stem + "ies"]
    return [stem + "s"]


_VERB_AFTER_LIST: dict[str, str] = {
    form: stem for stem in _VERB_STEMS for form in _sforms(stem)
}
_VERB_AFTER_LIST.update({"is": "are", "was": "were", "has": "have",
                         "does": "do"})


def has_negation(clause: str) -> bool:
    return any(r.search(_norm(clause)) for r in _NEGATION_RES)


def _scrub(clause: str) -> str:
    """Strip a negation out of a clause, keeping whatever positive content it
    also carries: "NOT an animal or beast-folk but a machine" → "a machine".

    Marvin's species line is the clearest case — it is written as a denial
    because the world is full of beast-folk, and "a machine" is the part the
    renderer can actually draw. Three passes, most content-preserving first:
    drop a negated parenthetical, resolve an "not X but Y", then keep the
    comma-level clauses that were never negated. A clause that is NOTHING but a
    negation returns empty and is left out of the prompt entirely.
    """
    clause = clause.strip(" .")
    if not has_negation(clause):
        return clause

    # 1. "fluffy blue-grey feather coloration only (no hair on the head)" — the
    #    denial is a side remark; the description it hangs on is paintable.
    def repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        return f"({inner})" if inner and not has_negation(inner) else ""

    without_paren = re.sub(r"\(([^()]*)\)", repl, clause).strip(" ,.")
    if without_paren and not has_negation(without_paren):
        return without_paren

    # 2. "construct — NOT an animal or beast-folk but a machine" keeps its leading
    #    label: "construct" is the paintable part the sheet leads with, and dropping
    #    it would leave only "a machine".
    if "—" in clause and "—" not in clause.split("—", 1)[-1]:
        head, _, tail = clause.partition("—")
        if head.strip() and not has_negation(head):
            sub = _scrub(tail)
            if sub:
                return f"{head.strip()} — {sub}"
            return head.strip()

    # 2. "NOT an animal but a machine" → "a machine".
    tail = clause.rsplit(" but ", 1)[-1].strip(" .,")
    if tail and not has_negation(tail):
        return tail

    # 3. "glowing blue eyes, no visible pupils" → "glowing blue eyes".
    keep = [c for c in _split(clause) if c and not has_negation(c)]
    return ", ".join(keep)


def _split(text: str) -> list[str]:
    """Split a field value into clauses at TOP level only, unsanitised.

    "…the classic raccoon pattern (fur coloration only), grizzled salt-and-
    pepper texture…" must not split inside the parenthesis: a fragment like
    "fur coloration only)" is worse than no fragment, because the renderer reads
    it as an object to paint. Clauses come back unsanitised — _scrub() is what
    removes a negation, and it runs per field in Anchor.clauses_for().
    """
    out: list[str] = []
    buf: list[str] = []
    depth = 0

    def emit(piece: str) -> None:
        piece = piece.strip(" .")
        if piece:
            out.append(piece)

    for ch in text or "":
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch in ";," and depth == 0:
            emit("".join(buf))
            buf = []
            continue
        buf.append(ch)
    emit("".join(buf))
    return out


def _clauses(text: str) -> list[str]:
    """The painted clauses of a field: split, then negations scrubbed away."""
    return [c for c in (_scrub(p) for p in _split(text)) if c]


_FIELD_WEIGHT = {"signature": 12, "face": 8, "species": 6, "outfit": 4,
                 "body": 2, "quirks": 0, "look": 10}

# The clauses that say WHO a figure is — the first clause of an anchor's species
# line, or of an unanchored figure's looked-up `look`. Compression may spend
# everything else first, because a character reduced to "a figure holding a bow"
# is a character the renderer invents (measured failure 1).
_IDENTITY_FIELDS = ("species", "look")


_GENERIC_LOOK_WORDS = frozenset((
    "with from over under above below beside behind within without the a an and or of"
    " his her its their in on at by is are as while small large long short dark light"
    " deep soft bright round broad slender narrow fine plain simple warm cool rich"
    " highly giving made wearing dressed clad carrying holding built steel brass"
    " leather fur head face body eyes ears neck chest shoulder hands feet one two"
    " three four five six seven eight nine ten".split()))


def look_tokens(look: str) -> list[str]:
    """The distinctive words of a description — the ones a check can match."""
    return [t for t in re.findall(r"[a-z\u00e4-\u00fc\u00df]{3,}", look.lower())
            if t not in _GENERIC_LOOK_WORDS]


def look_supported_by(look: str, retrieved: str) -> tuple[int, int]:
    """(matched, total) distinctive words of `look` found in retrieved text."""
    toks = look_tokens(look)
    text = retrieved.lower()
    return sum(1 for t in toks if t in text), len(toks)


def _snippet_supported(look: str, retrieved: str) -> bool:
    """A look taken from a SEARCH hit has to be traceable to that hit."""
    hits, total = look_supported_by(look, retrieved)
    if total == 0:
        return True
    if total <= 3:
        return hits == total
    return hits >= max(2, -(-total // 2))


# A species is the most consequential thing a description asserts, and the
# easiest to invent: measured live on the bandit-ambush rewrite, the model wrote
# "heavy-set badger-folk bandit chief, flat snout, cutlass at his hip" for a
# character the vault never describes — and cited a session note it HAD opened,
# which mentions him by name and says nothing about how he looks. "The file was
# opened, so believe the description" is therefore not sufficient on its own.
# A creature noun must appear in what was actually retrieved, in English or in
# German (the vault is half German and the model translates).
_SPECIES = (
    ("badger", "dachs"), ("fox", "fuchs"), ("squirrel", "eichh\u00f6rnchen"),
    ("raccoon", "waschb\u00e4r"), ("hedgehog", "igel"), ("owl", "eule"),
    ("rabbit", "kaninchen", "hase"), ("mouse", "maus"), ("rat", "ratte"),
    ("crow", "kr\u00e4he"), ("raven", "rabe"), ("jay", "h\u00e4her"),
    ("cat", "katze"), ("dog", "hund"), ("bear", "b\u00e4r"), ("wolf", "wolf"),
    ("deer", "reh", "hirsch"), ("boar", "eber"), ("lizard", "echse"),
    ("frog", "frosch"), ("toad", "kr\u00f6te"), ("mole", "maulwurf"),
    ("otter", "otter"), ("beaver", "biber"), ("skunk", "stinktier"),
    ("weasel", "wiesel"), ("horse", "pferd", "pony"), ("donkey", "esel"),
    ("goat", "ziege"), ("sheep", "schaf"), ("bat", "fledermaus"),
    ("turtle", "schildkr\u00f6te", "turtle"), ("snake", "schlange"),
    ("hamster", "hamster"), ("quokka", "quokka"), ("mink", "mink"),
    # Humblewood's mustelid crowd, which the vault's own prose keeps using: the
    # first live run after the policy change wrote "two stoat-folk bandits" and the
    # gate let it through untouched because the word was not in its vocabulary.
    ("stoat", "hermelin"), ("ermine", "hermelin"), ("ferret", "iltis", "polecat"),
    ("marten", "marder"),
    # In-world species names, from the party sheet itself: "Quokka-like mammal
    # (in-world name: Jerbeen)", "Strig (owl)", "Mapach (raccoon)". A vault that
    # calls someone a strig must not be reported as an invented species.
    ("quokka", "jerbeen"), ("strig", "strig"), ("mapach", "mapach"),
    ("hedge", "hedge"), ("birdfolk", "birdfolk", "bird-folk"),
    # Colum's sheet line says "Birdfolk (blue tit)": the conflict lint flagged that
    # the one real species the vault uses for him was a word the gate could not see.
    ("tit", "meise", "titmouse"),
)
_species_variants = {w: frozenset(variants) for variants in _SPECIES for w in variants}
_all_species = frozenset(_species_variants)


def _name_keys(name: str) -> list[str]:
    return [k for k in {name,
                        name.split()[-1] if " " in name else "",
                        name.split()[0]} if len(k) >= 4]


def _named_blocks(retrieved: str, keys: Sequence[str]) -> list[str]:
    """The retrieved blocks that contain at least one of ``keys``.

    Sections, not just paragraphs: a heading stays with its body, so a character's
    own section counts and the next section (another character, or an unrelated
    D&D entry) does not.
    """
    if not retrieved:
        return []
    keys = [k for k in keys if len(k) >= 4]
    if not keys:
        return [retrieved]
    blocks: list[str] = []
    for part in re.split(r"\n\s*\n", retrieved):
        blocks.extend(re.split(r"\n(?=#{1,4} )", part))
    return [b for b in blocks if any(k.lower() in b.lower() for k in keys)]


def evidence_about(name: str, retrieved: str) -> str:
    """The retrieved material that is ABOUT this character.

    A species has to be retrieved about the character, not merely somewhere in
    the loop's context: measured on the bandit request, the rewrite searched
    "Frey bandit badger wolf fox species appearance" and the vault answered with
    a D&D magic-item chunk about a badger — which made an invented badger-folk
    look look backed. Retrieved material arrives in blocks (one search hit or
    one file read per block); only the blocks naming the character count.
    """
    return "\n".join(_named_blocks(retrieved, _name_keys(name)))


# Words that name no species and are therefore never a fact claim.
_GENERIC_SPECIES = frozenset(
    """beast beastfolk animal animals anthropomorphic creature creatures humanoid
    human humanoids construct constructs tiefling demon devil spirit monster
    folk folks kin people volk race races kind kinds""".split())


def _claimed_species(text: str) -> list[str]:
    """Species animals a description claims, in the order they appear.

    Both shapes: a "-folk" compound ("weasel-folk bandits") and a bare creature
    word used as an identity ("a brass-and-steel construct", "a plump peddler
    squirrel"). Generic words are not claims.
    """
    out: list[str] = []
    for m in _SPECIES_COMPOUND.finditer(text or ""):
        t = m.group(1).lower()
        if t not in _GENERIC_SPECIES and t not in out:
            out.append(t)
    for tok in re.findall(r"[a-z\u00e4\u00f6\u00fc]+", _norm(text or "")):
        if tok in _all_species and tok not in _GENERIC_SPECIES and tok not in out:
            out.append(tok)
    return out


def _species_named_in(evidence: str) -> set[str]:
    """Which species the material itself names (English or German spellings)."""
    found: set[str] = {m.group(1).lower()
                       for m in _SPECIES_COMPOUND.finditer(evidence or "")}
    for tok in re.findall(r"[a-z\u00e4\u00f6\u00fc]+", _norm(evidence or "")):
        if tok in _all_species and tok not in _GENERIC_SPECIES:
            found.add(tok)
    return found


def _species_near(keys: Sequence[str], evidence: str,
                  window: int = 90) -> set[str]:
    """Species named NEAR the words that identify the beings being described.

    Chunk membership is not evidence. A live run retagged "two weasel-folk
    bandits" as "lizard-folk bandits" because the retrieved material contained
    the PHB's find-familiar list — "Bat, Cat, Frog, Hawk, Lizard, Octopus, Owl,
    Rat, Raven, Spider, Weasel" — a rules paragraph about familiars that merely
    shared a chunk with the ambush. Proximity to the crowd word is what makes a
    species statement ABOUT that crowd.
    """
    hay = evidence or ""
    out: set[str] = set()
    if not keys:
        return out
    low = hay.lower()
    for k in keys:
        k = _norm(k)
        if len(k) < 4:
            continue
        # Stem matching: the vault and the plan need not spell the crowd alike.
        # "bandits" has to answer "Banditen" before the vault's own animal counts
        # as a statement ABOUT that crowd rather than about somebody else.
        stem = k[:5]
        for m in re.finditer(re.escape(stem), low, re.I):
            lo, hi = max(0, m.start() - window), min(len(hay), m.end() + window)
            out |= _species_named_in(hay[lo:hi])
    return out


def species_verdict(text: str, evidence: str,
                    allow: Iterable[str] = (),
                    near: Sequence[str] = ()) -> tuple[list[str], list[str]]:
    """(must_strip, allowed_guess) for the species a description claims.

    The policy is the table owner's (2026-11-19): **this world has no humans**, so
    a crowd or NPC the vault never describes must still come out as animal-folk
    rather than as people — inventing a species that fits the setting is allowed,
    and it is what puts animals in the final picture. What is refused is a claim
    that CONTRADICTS the material about those particular characters: if the vault
    says the crew are rats, they are not painted as weasels, and a species found
    in a block about someone else (a badger-shaped magic sword) never authorises a
    claim about bandits either.

    So the gate inverted: backing used to be required, now only agreement with what
    IS said is required. `allow` carries words the caller vouches for — species the
    user typed themselves, the party's own anchors.
    """
    claims = _claimed_species(text)
    if not claims:
        return [], []
    allowed = {_norm(a) for a in allow}
    # Proximity when the caller says WHICH beings are being described; whole-body
    # reading when the evidence is already narrowed to them (a named character).
    named = _species_near(near, evidence) if near else _species_named_in(evidence)
    hay = _norm(evidence)
    strip: list[str] = []
    guess: list[str] = []
    for token in claims:
        if token in allowed:
            continue
        if any(re.search(r"\b" + re.escape(v) + r"\b", hay)
               for v in _species_variants.get(token, (token,))):
            continue                     # the material says exactly this
        if not named:
            guess.append(token)          # nothing describes them: gap-filling ok
            continue
        strip.append(token)              # it describes them, and not this way
    return strip, guess


def contradicted_species(text: str, evidence: str,
                         allow: Iterable[str] = ()) -> list[str]:
    return species_verdict(text, evidence, allow)[0]


def _canonical_species(token: str) -> str:
    """The English spelling of a species the material named in German.

    SESSION_01 calls the ambush boss a "Fuchsdame": when the plan says
    badger-folk, the vault does not merely object — it offers an animal, and
    "the vault's species wins" is only real if THAT animal reaches the prompt.
    """
    variants = frozenset(_species_variants.get(token) or (token,))
    for group in _SPECIES:
        if variants and variants <= set(group) or token in group:
            return group[0]
    return token


def species_replacement(text: str, evidence: str,
                        allow: Iterable[str] = (),
                        exclude: Iterable[str] = (),
                        near: Sequence[str] = ()) -> str | None:
    """The species the MATERIAL names when it contradicts what *text* claims.

    `exclude` is the species of the characters already standing in the frame.
    Measured live (§17): a bandit crowd was retagged "hedgehog-folk" because the
    retrieved passages about the ambush describe Romi, who IS a hedgehog — the
    animal of one side of a standoff is not evidence about the other side.
    """
    strip, _ = species_verdict(text, evidence, allow, near=near)
    if not strip:
        return None
    banned = {_norm(e) for e in exclude}
    named = {t for t in (_species_near(near, evidence) if near
                         else _species_named_in(evidence))
             if t not in banned and _canonical_species(t) not in
             {b.split("-")[0] for b in banned}}
    # The MOST ATTESTED animal, not the first one found. When the vault says more
    # than one thing about a crew (round 9 of the evaluation run: fox for the chief
    # from "Fuchsdame", lizard from one market line), a prompt that gives one crowd
    # two different animals is worse than either answer on its own.
    # Which animal fits THIS claim? Number first: the vault describes the ambush
    # boss as "eine Fuchsdame" (one fox-folk) and the crew as "three rat-folk
    # cutthroats", so a chief and his lackeys get different answers from the same
    # material — which is what the campaign actually says. Only when the number
    # does not discriminate does attestation decide, then alphabetical order, so
    # the choice is reproducible rather than a set-iteration accident.
    want = _claim_number(text)

    def fit(t: str) -> tuple[int, int, int, str]:
        tnum = _evidence_number(evidence, t)
        agree = 2 if tnum == want or tnum == "any" or want == "any" else 0
        return (agree, _attested(evidence, t), len(t), t)

    for token in sorted(named, key=fit, reverse=True):
        if token not in _GENERIC_SPECIES:
            return _canonical_species(token)
    return None


_PLURAL_WORDS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|drei|vier|"
    r"fünf|sechs|sieben|acht|neun|zehn|\d{1,2})\b", re.I)
_SINGULAR_WORDS = re.compile(r"\b(eine|ein|a|an|one)\b", re.I)

# A collective head noun is plural for this purpose: "bandit crew", "the party".
_COLLECTIVE_HEADS = frozenset(
    "party gang crew group band troop company mob crowd horde pack".split())


def _claim_number(text: str) -> str:
    """Singular or plural: what the claim's own noun says."""
    for m in _SPECIES_COMPOUND.finditer(text or ""):
        nxt = re.findall(r"[A-Za-zÄÖÜäöüß]+", text[m.end():m.end() + 30])
        if not nxt:
            continue
        # "two weasel-folk bandit lackeys" and "bandits bar the path": whichever of
        # the next two words is marked plural decides, because in English the head
        # noun can sit either side of a modifier ("bandit lackeys" / "bandits bar").
        words = [w.lower() for w in nxt[:2] if w.lower() not in _all_species]
        if any(w.endswith("s") or w.endswith("en") or w in _COLLECTIVE_HEADS
               for w in words):
            return "plural"
        return "singular"
    return "any"


def _evidence_number(evidence: str, token: str) -> str:
    """What the material says about that animal: a count, or one being."""
    low = _norm(evidence)
    seen_any = False
    words = [token] + [v for v in _species_variants.get(token, ()) if v != token]
    for m in re.finditer("|".join(re.escape(w) for w in dict.fromkeys(words)), low):
        back = low[max(0, m.start() - 28):m.start()]
        if _PLURAL_WORDS.search(back):
            return "plural"
        if _SINGULAR_WORDS.search(back):
            seen_any = True
    return "singular" if seen_any else "any"


def _attested(evidence: str, token: str) -> int:
    """How often the material really says this animal.

    Across spellings and languages: the vault writes "Fuchsdame" where the plan
    would write "fox", and a count on the English word alone reads as silence.
    """
    low = _norm(evidence)
    words = [token] + [v for v in _species_variants.get(token, ()) if v != token]
    words += [w for w in re.findall(r"[a-z\u00e4\u00f6\u00fc]+", low)
              if _canonical_species(w) == token]
    return sum(len(re.findall(r"(?<![a-z\u00e4\u00f6\u00fc])" + re.escape(w)
                              + r"(?![a-z\u00e4\u00f6\u00fc])", low))
               for w in dict.fromkeys(words))


def crowd_species_choice(claims: Sequence[str], retrieved: str,
                         ours: Sequence[str] = (),
                         exclude: Iterable[str] = ()) -> str | None:
    """One animal for a GROUP of claims, when they describe the same beings.

    Kept available because a prompt that gives one crew two animals is a real
    failure mode (round 9: fox-folk chief, lizard-folk bandits). It is NOT wired
    into compose_prompt: a chief and his lackeys are separate claims, and the vault
    genuinely describes them differently — "eine Fuchsdame" for the boss, "three
    rat-folk cutthroats" for the crew (test_the_vaults_own_species_replaces_the_one_
    the_plan_invented). A global average would flatten that into a lie.
    """
    text = " ".join(c for c in claims if c)
    if not text or not retrieved:
        return None
    ev = crowd_evidence_spans(text, retrieved, ours)
    if not ev:
        return None
    return species_replacement(text, ev, exclude=exclude)


def _claim_supported(species: str, tail: str, evidence: str) -> bool:
    """Does the material put this animal next to the noun THIS claim is about?

    Round 10 of the evaluation run (§17): the request named the freed merchant
    Alisa Pennygleam AND Boss Frey's bandits in one sentence. The vault's only
    lizard is the Trunk Market merchant, and a substitution that stamps one animal
    over every species claim in the sentence retagged the bandits with it.
    """
    if not evidence:
        return True
    nouns = [w for w in re.findall(r"[A-Za-zÄÖÜäöüß]+", tail)[:3]
             if _norm(w) not in _SEGMENT_STOP and _norm(w) not in _all_species]
    if not nouns:
        return True
    return bool({t for t in _species_near(nouns, evidence, window=60)
                 if _canonical_species(t) == species})


def substitute_species(text: str, tokens: Sequence[str], replacement: str,
                       evidence: str = "") -> str:
    """Swap the claimed animals for the one the vault uses, keeping the sentence."""
    out = text
    # The vault writes "birdfolk" as one word: appending "-folk" again gives the
    # nonsense "birdfolk-folk" that a live 04 run actually produced.
    target = (replacement if any(replacement.lower().endswith(sfx)
                                for sfx in _SPECIES_COMPOUND_SUFFIXES)
        else f"{replacement}-folk")
    for token in tokens:
        variants = sorted(_species_variants.get(token, (token,)),
                          key=len, reverse=True)
        pat = re.compile(
            r"\b(?:" + "|".join(re.escape(v) for v in variants) + r")"
            r"[\-\u2011 ]?(?:" + "|".join(_SPECIES_COMPOUND_SUFFIXES) + r")\b",
            re.IGNORECASE)
        def _swap(m: re.Match) -> str:
            tail = text[m.end():m.end() + 30]
            return target if _claim_supported(replacement, tail, evidence) \
                else m.group(0)

        out = pat.sub(_swap, out)
    return out


def crowd_keys(text: str) -> list[str]:
    """The words that identify WHICH beings a description is about."""
    return _segment_keys(text)


def crowd_evidence(text: str, retrieved: str) -> str:
    """The retrieved blocks that are about the beings THIS sentence names."""
    return "\n".join(_named_blocks(retrieved, _segment_keys(text)))


# The old three-argument call shape, kept so an older call site still reads.
_crowd_spans = lambda text, retrieved, evidence=(), ours=(): crowd_evidence_spans(
    text, retrieved, ours if ours is not evidence else ())


_CLAUSE_BREAK = ",:!?\u2014\u2013\"\u201e()"


def _clause_start(span: str, i: int) -> int:
    for j in range(i - 1, max(i - 41, -1), -1):
        if span[j] in _CLAUSE_BREAK:
            return j + 1
    return max(i - 40, 0)


def _clause_end(span: str, i: int) -> int:
    for j in range(i, min(i + 41, len(span))):
        if span[j] in _CLAUSE_BREAK:
            return j
    return min(i + 40, len(span))


def _claim_nouns(text: str) -> list[str]:
    """The role nouns a species claim is actually attached to.

    "two squirrel-folk bandits bar the path" is a claim about BANDITS, so only a
    vault line that puts an animal next to the word "bandit" may answer it. Round
    5 of the evaluation run (§17) showed what happens otherwise: the sentence
    "…at Winnowing Reach" handed the substitution a key ("reach"), the vault's
    bandit line matched it, and Alisa Pennygleam the merchant came out rat-folk.
    """
    nouns: list[str] = []
    for m in _SPECIES_COMPOUND.finditer(text or ""):
        tail = re.findall(r"[a-z\u00e4\u00f6\u00fc]{4,}", _norm(text[m.end():m.end() + 30]))
        head = re.findall(r"[a-z\u00e4\u00f6\u00fc]{4,}",
                          _norm(text[max(0, m.start() - 30):m.start()]))
        for w in (tail[:2] or head[-2:]):
            if w not in _SEGMENT_STOP and w not in _all_species:
                nouns.append(w[:-1] if w.endswith("s") and len(w) > 4 else w)
    return list(dict.fromkeys(nouns))[:6]


_FUNCTION_WORDS = frozenset("""a an the der die das den dem des ein eine einer einem of at in on
und or and but mit bei zu von zum zur for with near into from to by as like then they their
this that these those there here all any some each every other such""".split())


def crowd_evidence_spans(text: str, retrieved: str,
                         ours: Sequence[str] = ()) -> str:
    """What the material says about THE CROWD *text* describes — no more.

    Two filters, both learned from the evaluation run (§17):

    * only spans where a crowd word and an animal SIT TOGETHER count as a species
      statement about that crowd, so a D&D familiar list sharing a chunk with the
      ambush ("Bat, Cat, Frog, Hawk, Lizard … Rat, Raven") never answers it, and
    * a species claim belonging to a NAMED individual is blanked, so Spleck the
      lizard merchant is not evidence about Alisa Pennygleam the merchant.

    This is the evidence both the contradiction test and the substitution read,
    which is why one function serves both: a rule that only policed the
    substitution still let the contradiction strip a character on somebody else's
    description.
    """
    """Crowd-near spans with another character's species blanked out.

    A span is not rejected wholesale for mentioning someone (the vault names its
    places and its NPCs constantly); the SPECIES claim that belongs to a named
    individual is blanked, so it cannot be borrowed for our crowd.
    """
    nouns = _claim_nouns(text) or _segment_keys(text)
    # The vault's OWN spelling of the crowd ("Banditen") is what a name gets glued
    # to — a German session log names its bandits just as plainly as English does.
    nouns = nouns + [k for k in _segment_keys(text) if k not in nouns][:4]
    # A word that is part of a name we are painting is not a role word: "merchant"
    # in "the hedgehog-folk merchant Alisa Pennygleam" describes Alisa, and the
    # vault's lizard merchant Spleck has nothing to do with her.
    ours_low = [_norm(o) for o in ours]
    nouns = [n for n in nouns if not any(_norm(n) in o for o in ours_low)]
    # …nor may the claim's OWN species be a crowd key: searching the vault with
    # "weasel" finds the D&D familiar list, which then "contradicts" the plan with
    # the very sentence it is being asked about.
    claim = {_norm(w) for w in _claimed_species(text)}
    nouns = [n for n in nouns if _norm(n) not in claim]
    out: list[str] = []
    for span in crowd_adjacency(" ".join(nouns), retrieved).split("\n"):
        keep = list(span)
        low = span.lower()
        # Both claim shapes: a "rat-folk" compound and a bare animal word used as
        # an identity ("lizard merchant Spleck"). The bare shape is what got
        # Alisa Pennygleam painted as a lizard.
        hits = [m.span() for m in _SPECIES_COMPOUND.finditer(span)]
        # A bare animal word is a species claim only where it MODIFIES a crowd
        # word: "lizard merchant", "Frosch Piraten". Round 8 of the evaluation run
        # (§17) shows why distance is not enough — the D&D familiar list
        # ("…options: Bat, Cat, Frog, Hawk, Lizard…") sits sixty characters from
        # "Banditen" and answers a question nobody asked it.
        for m in re.finditer(r"[a-z\u00e4\u00f6\u00fc]+", low):
            w = m.group(0)
            if w not in _all_species or w in _GENERIC_SPECIES:
                continue
            following = re.findall(r"[a-z\u00e4\u00f6\u00fc]+", low[m.end():m.end() + 24])[:1]
            if not any(f and (f == n or f.startswith(n[:5]) or n.startswith(f[:5]))
                       for f in following for n in nouns):
                continue
            hits.append(m.span())
        for lo0, hi0 in hits:
            lo, hi = max(0, lo0 - 40), min(len(span), hi0 + 40)
            near = span[lo:hi]
            # …and only when the name is attached to it, not merely in the same
            # breath: "Winnowing Reach: three rat-folk cutthroats" names a place,
            # and the colon says so. Breaking on punctuation keeps the vault's
            # good evidence ("three rat-folk cutthroats") intact.
            # "Attached" means close and in the same breath. "lizard merchant
            # Spleck" is a species belonging to Spleck; "Boss Frey führt seine
            # Banditen: two rat-folk cutthroats" is a species belonging to the
            # bandits, with a name merely further up the line.
            # The whole breath around the claim, not just a few characters:
            # "Session 5 Trunk Market: Romi leaves, lizard merchant Spleck sells
            # potions" — Spleck is the vault's OWN lizard merchant, and his species
            # is not Alisa Pennygleam's. A species may be borrowed for a crowd only
            # when everyone the sentence names is a character we are painting.
            lo_s = _clause_start(span, lo0)
            hi_s = _clause_end(span, hi0)
            for _ in range(2):          # widen once: names sit a clause further out
                lo_s = _clause_start(span, lo_s)
                hi_s = _clause_end(span, hi_s)
            if _stray_name(span[lo_s:hi_s], ours, nouns):
                keep[lo0:hi0] = " " * (hi0 - lo0)
        out.append("".join(keep))
    return "\n".join(out)


def crowd_adjacency(text: str, retrieved: str, window: int = 60) -> str:
    """The spans of retrieved text where a crowd word and an animal SIT TOGETHER.

    Substitution is the risky operation: picking a species for a nameless crew
    invents campaign canon. Round 4 of the evaluation run (§17) showed what
    loose evidence does — a badger-folk bandit chief retagged FROG-folk (from
    "Frosch Piraten" in an unrelated session note) and birdfolk-folk (from a
    calendar entry about a festival). So the species has to sit next to the crowd
    word on the same line to count as a statement about that crowd.
    """
    keys = [k for k in _segment_keys(text) if len(k) >= 4]
    if not keys or not retrieved:
        return ""
    spans: list[str] = []
    for line in retrieved.splitlines():
        low = line.lower()
        for k in keys:
            for m in re.finditer(re.escape(k), low):
                lo, hi = max(0, m.start() - window), min(len(line), m.end() + window)
                # Grow the window to whole words: a slice that cut "regal bearing"
                # down to "bear" handed the substitution an animal the campaign
                # never named (round 10 of the evaluation run, §17).
                while lo > 0 and (line[lo - 1].isalnum() or line[lo - 1] in "äöüß-_"):
                    lo -= 1
                while hi < len(line) and (line[hi].isalnum() or line[hi] in "äöüß-_"):
                    hi += 1
                spans.append(line[lo:hi])
    return "\n".join(spans)


def _stray_name(span: str, ours: Iterable[str],
                nouns: Sequence[str] = ()) -> str | None:
    """A character the vault NAMED together with one of these role words.

    The vault does have a lizard merchant — Spleck, at the Trunk Market in
    Alderheart. Round 5 of the evaluation run (§17) borrowed his species for
    Alisa Pennygleam, because the request said "merchant" and the vault said
    "lizard merchant" somewhere nearby. A species glued to a NAME is evidence
    about that individual, not about the crowd the plan is describing.

    The test is the GLUE, not capitalisation: "lizard merchant Spleck" and "die
    Hexe Susan" name somebody, so the species belongs to them. "an der Old
    Bridge" and "bei Winnowing Reach" name a place, and a place must not outvote
    the vault's species statement — which is why scanning for anything capitalised
    (German capitalises every noun) was the wrong detector.
    """
    allowed = {_norm(o) for o in ours}
    role = sorted({re.escape(_norm(n)) for n in nouns if len(_norm(n)) >= 4})
    if not role:
        return None
    names = "|".join(sorted(_all_species))
    cap = r"[A-ZÄÖÜ][a-zäöüß\u00df]{3,}"
    # Two separate scans, not one alternation: "lizard merchant Spleck" matches the
    # name-before-role shape first ("lizard" before "merchant"), and an alternation
    # would have CONSUMED "merchant" there, so the name glued after it — the one
    # that actually owns the species — was never seen.
    after = re.compile(r"(?:" + "|".join(role) + r")\s+(?!(?:" + names + r")\b)("
                       + cap + r")", re.IGNORECASE)
    before = re.compile(r"(" + cap + r")\s+(?:" + "|".join(role) + r")\b",
                        re.IGNORECASE)
    for m in list(after.finditer(span)) + list(before.finditer(span)):
        raw = m.group(1) or m.group(2)
        # The role words match in any case ("Banditen"), but the NAME side has to
        # actually be a name: with IGNORECASE on, "three rat-folk cutthroats" looked
        # like a character called Three and blanked the vault's own species.
        if not raw[0].isupper():
            continue
        tok = _norm(raw)
        if tok in allowed or tok in _all_species or tok in _FUNCTION_WORDS:
            continue
        if any(tok in _norm(o) for o in allowed):
            continue
        return raw
    return None


def invented_species(look: str, retrieved: str,
                     allow: Iterable[str] = ()) -> list[str]:
    """Kept name for the contradiction test on a figure's own `look`."""
    return contradicted_species(look, retrieved, allow)


# A species claim was policed only inside a figure's `look`. Measured live
# (analysis §15): the same ambush request that produced an invented badger-folk
# Boss Frey also produced "two lanky weasel-folk bandits with pointed ears, long
# snouts and russet fur" and "two rat-folk bandit lackeys" in `entities` — the
# crowd descriptions were never checked, and a small scene keeps the entity line
# instead of budget-dropping it. Only the `-folk` compounds are policed: "a giant
# rat blocking the corridor" names a creature, which is what `entities` is FOR,
# while "weasel-folk bandits" is a claim about WHO they are.
# A species compounded onto a being-word is how the vault's German says
# beast-folk: "eine Fuchsdame" (a fox-woman) is the ambush boss's own species
# in SESSION_01. Without these heads the gate reads "Fuchsdame" as an unknown
# word and never notices that the vault HAS an animal for that character.
_SPECIES_COMPOUND_SUFFIXES = ("folk", "folks", "kin", "people", "volk",
                              "dame", "mann", "leute", "wesen", "frau", "weib")
_SPECIES_COMPOUND = re.compile(
    r"\b(" + "|".join(sorted(_all_species, key=len, reverse=True)) + r")"
    r"[\-\u2011 ]?(?:" + "|".join(_SPECIES_COMPOUND_SUFFIXES) + r")\b",
    re.IGNORECASE)

# Words too general to locate the passage a claim came from.
_SEGMENT_STOP = frozenset(
    """scene scenes image images picture pictures character characters figure
    figures people person persons group groups party several many some other
    such about inside outside behind before during while through towards toward
    wearing holding carrying standing walking with them themselves every all
    foreground background lighting palette mood styleRequest request""".split())


def _segment_keys(text: str) -> list[str]:
    """The distinctive nouns of a sentence, used to find where it came from.

    Singular stems are included so "bandits" finds a vault passage that says
    "bandit crew". A vault that names the crowd only in German ("Banditen") is
    not found, and the species claim is then STRIPPED — the safe direction: the
    crowd is still painted, the composer just refuses to decide what species it
    is.
    """
    out: set[str] = set()
    for w in re.findall(r"[a-z\u00e4\u00f6\u00fc]{4,}", _norm(text)):
        if w in _SEGMENT_STOP or w in _all_species:
            continue
        out.add(w)
        if w.endswith("s") and len(w) > 4:
            out.add(w[:-1])
    return sorted(out)[:12]


def _frame_species(resolved) -> set[str]:
    """Species the vault states for the characters already in the frame."""
    out: set[str] = set()
    for f, a in resolved:
        if a is None:
            continue
        hay = " ".join([a.name, a.desc, a.blob, *a.fields.values()])
        out |= _species_named_in(hay)
    return out


def _named_in_request(entry: str, request_text: str) -> bool:
    """Does this entity entry name something the user's own words named?

    Word overlap, not name guessing: "zwei Banditen mit Äxten" matches a request
    that says "seinen Banditen" because BANDIT is in both, and no classifier has
    to decide whether a bandit is a character.
    """
    def stems(text: str) -> set[str]:
        # Stems, not words: the request says "seinen Banditen", the plan says
        # "bandits". Exact word equality would call that no overlap and cut the
        # user's crowd out of their own image.
        return {w[:6] for w in re.findall(r"[a-z\u00e4\u00f6\u00fc]{4,}", _norm(text))}

    return bool(stems(request_text) & stems(entry))


def request_species(text: str) -> set[str]:
    """Species the USER named. Their own campaign is their own authority.

    "Paint Boss Frey as a badger-folk" is not the model inventing a species, and
    dropping it would be the composer arguing with the person holding the reins.
    """
    out: set[str] = set()
    for token in re.findall(r"[a-z\u00e4\u00f6\u00fc]+", _norm(text)):
        if token in _all_species:
            out.add(token)
    return out


def unbacked_species(text: str, retrieved: str,
                     request_text: str = "") -> list[str]:
    """Species a sentence claims that the material about THOSE beings denies."""
    strip, _ = species_verdict(text, crowd_evidence(text, retrieved),
                               request_species(request_text), near=crowd_keys(text))
    return strip


def species_guesses(text: str, retrieved: str,
                    request_text: str = "") -> list[str]:
    """Species a sentence claims that nothing describes either way.

    Kept by policy, but reported: an invented species in a painted image is the
    composer deciding campaign canon, and the person running the table should see
    it and put the real species in the vault if they have one.
    """
    _, guess = species_verdict(text, crowd_evidence(text, retrieved),
                               request_species(request_text), near=crowd_keys(text))
    return guess


_META_ASIDE = re.compile(
    r"\((?=[^)]*?\b(bot|vault|guess|guessed|species|invent|plan|look-?up|looked up|"
    r"retrieved|notes|not stated|unverified|assumption|unclear|no description)\b)"
    r"[^)]*\)", re.IGNORECASE)


def strip_meta_asides(text: str) -> tuple[str, list[str]]:
    """Cut parenthetical notes ABOUT the pipeline out of paintable text.

    A plan that writes "(species from the bot's own guess)" has answered the
    contract's reporting rule in the wrong place: the parenthesis is a comment
    about the plan, not a description of the picture, and a diffusion model
    paints comments too.
    """
    out = _META_ASIDE.sub("", text or "")
    out = re.sub(r"\s+", " ", out).strip()
    return out, [m.group(0) for m in _META_ASIDE.finditer(text or "")]


def strip_species(text: str, tokens: Sequence[str]) -> str:
    """Remove the claimed species compounds without breaking the sentence."""
    out = text
    for token in tokens:
        variants = sorted(_species_variants.get(token, (token,)),
                          key=len, reverse=True)
        pat = re.compile(
            r"\b(?:" + "|".join(re.escape(v) for v in variants) + r")"
            r"[\-\u2011 ]?(?:" + "|".join(_SPECIES_COMPOUND_SUFFIXES) + r")\b"
            r"[ ,]*", re.IGNORECASE)
        out = pat.sub("", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,;.!])", r"\1", out)
    # "the weasel-folk" alone would leave "the" attached to nothing.
    out = re.sub(r"\s*\b(?:the|a|an|some|his|her|their)\s*(?=[,;.!?]|$)", "", out,
                 flags=re.IGNORECASE)
    return out.strip(" ,;")


def verify_look(figure: Figure, read_paths: Sequence[str] | None,
                snippet_paths: Iterable[str] | None,
                retrieved: str, request_text: str = "") -> tuple[bool, str]:
    """May this NPC's `look` be painted? Three grounds, not equally strong.

    * The rewrite OPENED the file it cites: it had the section in front of it and
      may describe it in its own words (the vault is half German, so wording
      overlap is not the test there).
    * The file came back only as a SEARCH snippet: then the wording must be
      traceable to that snippet. A ten-word description matching none of a
      hundred-character snippet was not read from it — §4.2 measured a "large
      badger-folk with coarse grey-brown fur and a red bandana" for a character
      the vault never describes, written after no lookup at all.
    * It cites nothing this rewrite touched: invention, dropped loudly.

    With no evidence channel at all (`read_paths=None`, no snippets) there is
    nothing to check against and the plan is taken at its word — that is the
    caller's choice, not this function's.
    """
    if not figure.look:
        return True, ""
    if not figure.source:
        if read_paths is None and not snippet_paths:
            # No evidence channel at all (a unit test, a caller that trusts the
            # plan): paint it, but say out loud that nothing backs it.
            return True, ("`look` has no `source` and this caller verifies "
                          "nothing — painted unverified")
        return False, ("`look` without a `source` — the vault records NPC roles, "
                       "never appearances")
    bad = invented_species(figure.look, evidence_about(figure.name, retrieved),
                           allow=request_species(request_text))
    if bad:
        says = (", ".join(sorted(_species_named_in(
            evidence_about(figure.name, retrieved)))) or "something else")
        return False, ("describes them as " + "/".join(bad) +
                       "-folk while the material about THIS character says " +
                       says + " — the invented species is dropped and the "
                       "figure is painted by name")
    if read_paths is None and not snippet_paths:
        return True, ""
    if read_paths is not None and source_matches(figure.source, read_paths):
        return True, ""
    if snippet_paths and source_matches(figure.source, snippet_paths):
        if _snippet_supported(figure.look, retrieved):
            return True, ""
        hits, total = look_supported_by(figure.look, retrieved)
        return False, (f"`look` cites {figure.source!r} only as a search snippet, "
                       f"and {total - hits} of its {total} descriptive words are "
                       "not in what was retrieved — the appearance is dropped and "
                       "the figure is painted by name")
    return False, (f"`look` cites {figure.source!r}, which this rewrite never "
                   "retrieved — the appearance is dropped and the figure is painted"
                   " by name (measured drift, analysis §10)")


def source_matches(source: str, read_paths: Sequence[str]) -> bool:
    """Does *source* name a file in *read_paths*?

    Suffix-segment equality, not substring: a plan citing
    "Characters/NPCs.md" matches the read path
    "Humblewood/HumbleWood/Characters/NPCs.md", and "NPCs.md" matches too, while
    "Spleck.md" does NOT match "Spleck Onix.md" — the rule that kb/tools.py:193
    got wrong and that made a real character file unreadable (analysis §2).
    """
    def segs(p: str) -> list[str]:
        return [s for s in (p or "").replace("\\", "/").split("/") if s and s != "."]

    s = [x.lower() for x in segs(source)]
    if not s:
        return False
    for rp in read_paths:
        r = [x.lower() for x in segs(rp)]
        if not r:
            continue
        n = min(len(s), len(r))
        if r[-n:] == s[-n:]:
            return True
    return False


def drop_bare_repeat_mentions(scene: str, names: Sequence[str]) -> tuple[str, list[str]]:
    """Cut second mentions of a name that stand alone in a comma segment.

    The safe shape to cut is a segment that is nothing but a name — the tail of
    an enumeration ("…as bandits surround Seta Onix, Romi Stachelpfote, Bubo
    Krallensturm"). A mention welded to the character's own action ("Seta Onix
    aims her bow") is information about the scene and is never touched.
    """
    known = [n for n in names if n]
    segments: list[str] = []
    current = ""
    depth = 0
    for ch in scene:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            segments.append(current)
            current = ""
        else:
            current += ch
    segments.append(current)

    def bare_name(seg: str) -> str | None:
        core = seg.strip().strip(".").strip()
        for n in known:
            if core.lower() == n.lower():
                return n
        return None

    seen: set[str] = set()
    kept: list[str] = []
    dropped: list[str] = []
    for seg in segments:
        n = bare_name(seg)
        if n is not None and n.lower() in {k.lower() for k in seen}:
            dropped.append(n)
            continue
        for name in known:
            if re.search(r"\b" + re.escape(name) + r"\b", seg, re.I):
                seen.add(name)
        kept.append(seg)
    out = ", ".join(seg for seg in kept if seg.strip())
    # Tidy what the cut left behind: no dangling comma, no double punctuation.
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",\s*(?=[.,;])", "", out)
    out = re.sub(r"[.,;]+$", "", out)
    out = re.sub(r"\s{2,}", " ", out).strip().rstrip(",;")
    return out, dropped


def prompt_budget(figures: int, *, base: int = _BASE_CHARS,
                  per_figure: int = _CHARS_PER_FIGURE,
                  hard_cap: int = _HARD_CAP) -> int:
    """How much prompt one scene of *figures* named characters gets."""
    return min(hard_cap, base + per_figure * max(1, figures))


def _palette_of(text: str) -> set[str]:
    words = set(re.findall(r"[a-zäöü]+", _norm(text)))
    return {w for w in words if w in _PALETTE_WORDS}


def _unique_first(clauses: list[str], shared: set[str]) -> list[str]:
    """Clauses carrying a colour no other figure here wears lead the field.

    Reordering happens WITHIN one field, never across fields — what a character
    IS must lead their description. What trails is then exactly the shared
    palette ('gold' is worn by three of the six), which a budget trim should
    reach first (§3).
    """
    own: list[str] = []
    shared_c: list[str] = []
    for c in clauses:
        pal = _palette_of(c)
        (shared_c if (pal and pal <= shared) else own).append(c)
    return own + shared_c


# ─────────────────────────── anchors ──────────────────────────────────────

@dataclass
class Anchor:
    """One character's Visual Anchor — typed when the sheet carries fields.

    `fields` maps ANCHOR_FIELDS keys → text. A LEGACY anchor (one paragraph)
    lands in `blob` instead: it paints verbatim as always, and an override on it
    filters clauses instead of replacing one.
    """
    name: str
    desc: str = ""
    fields: dict[str, str] = field(default_factory=dict)
    blob: str = ""
    extra: list[str] = field(default_factory=list)   # unknown-labelled fields

    @property
    def typed(self) -> bool:
        return bool(self.fields)

    def clauses_for(
        self, override: dict[str, str],
    ) -> tuple[dict[str, list[str]], list[str]]:
        """Per-field clauses to paint, with *override* fields replaced.

        An override REPLACES its field — the replaced field is not painted at
        all, so "metal armour instead of the cloak" cannot contradict itself, and
        an EMPTY override means "paint it by omission" (nothing is written).
        For a legacy blob there is no field to replace, so clauses are filtered:
        anything reading like the overridden subject is dropped, signature
        clauses survive, and the replacement is appended. Dropping an item is far
        less harmful than painting two contradictory outfits.
        """
        out: dict[str, list[str]] = {}
        dropped = [c for c in _split(
            " ; ".join(override.values() if not self.typed else
                       [override.get(f, "") for f in ANCHOR_FIELDS])
        ) if has_negation(c)]
        dropped += [c for c in _split(
            " ; ".join(self.fields.values()) if self.typed else [self.blob]
        ) if has_negation(c)]
        if self.typed:
            for f in ANCHOR_FIELDS:
                src = override[f] if f in override else self.fields.get(f, "")
                clauses = _clauses(src)
                if clauses:
                    out[f] = clauses
            return out, dropped

        kept: list[str] = []
        for clause in _clauses(self.blob):
            drop = False
            for f in override:
                if f == "signature":
                    continue
                n = _norm(clause)
                if any(cue in n for cue in _SUBJECT_CUES.get(f, ())) \
                        and not any(cue in n for cue in _SIGNATURE_CUES):
                    drop = True     # this clause IS the field being replaced
            if not drop:
                kept.append(clause)
        for repl in override.values():
            if repl:
                kept.extend(_clauses(repl))
        return {"outfit": kept}, dropped


def parse_anchor(name: str, desc: str, block_lines: list[str]) -> Anchor:
    """Build an Anchor from the lines of its `## Visual Anchor` blockquote.

    Field form (at least one line matching `known-label: value`) → typed.
    Otherwise the whole block is one legacy blob, painted verbatim. A label the
    repo doesn't know ("fur markings: …") is kept verbatim in `extra` rather
    than silently dropped, so a sheet edit never loses content.
    """
    alias_to_field = {a: f for f, aliases in _FIELD_ALIASES.items()
                      for a in aliases}
    fields: dict[str, str] = {}
    extra: list[str] = []
    plain: list[str] = []
    current: str | None = None
    typed = False

    for raw in block_lines:
        line = raw.lstrip(">").strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z][A-Za-z /_-]{1,24}):\s*(.+)$", line)
        key = _norm(m.group(1)) if m else ""
        if m and key in alias_to_field:
            f = alias_to_field[key]
            typed = True
            current = f
            fields[f] = (fields.get(f, "") + " " + m.group(2).strip()).strip()
        elif m:
            # A label the repo doesn't know ("wheels: …") is kept verbatim rather
            # than silently dropped or glued onto the previous field, so a sheet
            # edit in Obsidian never loses content — and a continuation line of a
            # field value that merely contains a colon still joins its field.
            typed = True
            extra.append(f"{m.group(1).strip()}: {m.group(2).strip()}")
        elif current:
            fields[current] = f"{fields[current]} {line}".strip()
        else:
            plain.append(line)

    if typed:
        return Anchor(name=name, desc=desc, fields=fields, extra=extra)
    return Anchor(name=name, desc=desc, blob=" ".join(plain).strip())


# ─────────────────────────── the plan ─────────────────────────────────────

@dataclass
class Figure:
    """One named figure the plan wants painted."""
    name: str
    pose: str = ""
    # field -> replacement text; an EMPTY value means "omit this field entirely".
    modify: dict[str, str] = field(default_factory=dict)
    look: str = ""     # non-party characters: what the campaign notes say
    source: str = ""   # …and which file said so


@dataclass
class ImagePlan:
    scene: str = ""
    figures: list[Figure] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)   # monsters/objects, not cast
    lighting: str = ""
    palette: str = ""
    mood: str = ""


def _find_json_object(text: str) -> str | None:
    """The last balanced top-level {...} in *text* (a model that skipped the tag)."""
    starts = [i for i, ch in enumerate(text or "") if ch == "{"]
    for start in starts:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        data = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(data, dict) and ("scene" in data
                                                   or "figures" in data
                                                   or "characters" in data):
                        return candidate
                    break
    return None


def extract_plan(text: str) -> ImagePlan | None:
    """Pull the `<image_plan>…</image_plan>` JSON out of a model reply.

    Tolerant on purpose: a fenced block, a stray line before the tag or prose
    after it are all fine — a chatty model that produced the JSON beats a rewrite
    that has to be thrown away. No block (or unparseable JSON) returns None and
    the caller falls back to the prose path.
    """
    blocks = re.findall(r"<image_plan>(.*?)</image_plan>", text or "", re.S)
    if not blocks:
        blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.S)
    if not blocks:
        bare = _find_json_object(text or "")
        blocks = [bare] if bare else []
    for block in reversed(blocks):
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            plan = _plan_from_json(data)
            if plan is not None:
                return plan
    return None


def _plan_from_json(data: dict[str, Any]) -> ImagePlan | None:
    raw_figures = data.get("figures") or data.get("characters") or []
    if isinstance(raw_figures, dict):        # {"Seta Onix": "drawing her bow"}
        raw_figures = [{"name": k, "pose": v} for k, v in raw_figures.items()]
    if not isinstance(raw_figures, list):
        return None

    figures: list[Figure] = []
    for item in raw_figures:
        if isinstance(item, str):
            figures.append(Figure(name=item))
            continue
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("character") or "").strip()
        if not name:
            continue
        modify: dict[str, str] = {}
        raw_mod = item.get("modify") or item.get("override") or {}
        if isinstance(raw_mod, dict):
            for k, v in raw_mod.items():
                key = _canonical_field(k)
                if key:
                    modify[key] = str(v).strip() if v else ""
        elif isinstance(raw_mod, (list, tuple)):    # ["outfit"] = remove it
            for k in raw_mod:
                key = _canonical_field(str(k))
                if key:
                    modify[key] = ""
        figures.append(Figure(
            name=name,
            pose=str(item.get("pose") or item.get("action") or "").strip(),
            modify=modify,
            look=str(item.get("look") or item.get("appearance") or "").strip(),
            source=str(item.get("source") or "").strip(),
        ))
    # A scene with NO named cast is a legitimate plan ("a black pudding oozes
    # down a corridor"), but a plan with neither cast nor scene is nothing.
    if not figures and not str(data.get("scene") or "").strip():
        return None

    ents = data.get("entities") or data.get("creatures") or []
    if isinstance(ents, str):
        ents = [ents]
    # Non-strings are counts or stray numbers ("entities": ["a cart", 5]) — a bare
    # number in a prompt is a thing the renderer may try to paint.
    return ImagePlan(
        scene=str(data.get("scene") or "").strip(),
        figures=figures,
        entities=[e.strip() for e in ents if isinstance(e, str) and e.strip()],
        lighting=str(data.get("lighting") or "").strip(),
        palette=str(data.get("palette") or "").strip(),
        mood=str(data.get("mood") or "").strip(),
    )


def _canonical_field(key: str) -> str | None:
    k = _norm(key)
    if k in ANCHOR_FIELDS:
        return k
    for f, aliases in _FIELD_ALIASES.items():
        if any(k == a or k.startswith(a + " ") for a in aliases):
            return f
    return None


def match_anchor(name: str, anchors: dict[str, Anchor]) -> Anchor | None:
    """Resolve a plan name to an anchor: exact, then a UNIQUE prefix ('Seta')."""
    n = _norm(name)
    for a in anchors.values():
        if _norm(a.name) == n:
            return a
    hits = [a for a in anchors.values()
            if _norm(a.name).startswith(n) or n in _norm(a.name).split()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        log.warning("plan name %r matches %s — no anchor chosen", name,
                    ", ".join(h.name for h in hits))
    return None


def _as_figure(item: Any) -> Figure | None:
    """Accept a Figure or a plain dict (a hand-built plan, a future tool path)."""
    if isinstance(item, Figure):
        return item
    if isinstance(item, str):
        return Figure(name=item)
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or item.get("character") or "").strip()
    if not name:
        return None
    return Figure(
        name=name,
        pose=str(item.get("pose") or item.get("action") or "").strip(),
        modify={k: v for k, v in (item.get("modify") or {}).items()
                if isinstance(k, str)},
        look=str(item.get("look") or item.get("appearance") or "").strip(),
        source=str(item.get("source") or "").strip(),
    )


def dedupe_figures(
    pairs: list[tuple[Figure, Anchor | None]],
) -> list[tuple[Figure, Anchor | None]]:
    """One entry per named character — the count line promises an exact total.

    Takes the (figure, anchor) pairs compose_prompt has already resolved, and keys
    on the ANCHOR name when one matched. Measured failure 4 was the same character
    entered twice under two spellings ("Spleck" and "Spleck (the cat-folk)") —
    which painted two bodies — so unanchored names are keyed on the name with any
    parenthetical qualifier dropped, and a bare prefix ("Seta" / "Seta Onix")
    counts as the same figure. Merging keeps both poses and every override.
    """
    seen: dict[str, tuple[Figure, Anchor | None]] = {}
    order: list[str] = []
    merged: list[str] = []

    def key_of(f: Figure, a: Anchor | None) -> str:
        if a is not None:
            return _norm(a.name)
        return _norm(re.sub(r"\([^)]*\)", "", f.name))

    for f, a in pairs:
        key = key_of(f, a)
        if not key:
            continue
        # A prefix of an existing name is the same figure under a longer spelling.
        bucket = key
        for existing in order:
            if existing == key or existing.startswith(key + " ") \
                    or key.startswith(existing + " "):
                bucket = existing
                break
        if bucket in seen:
            merged.append(f.name)
            old, _ = seen[bucket]
            old.pose = " ".join(p for p in (old.pose, f.pose) if p)
            old.modify.update(f.modify)
            old.look = old.look or f.look
            continue
        seen[bucket] = (f, a)
        order.append(bucket)
    if merged:
        log.warning("plan listed %d duplicate figure entr(s) (%s) — merged into "
                    "one description", len(merged), ", ".join(merged))
    return [seen[k] for k in order]


# ─────────────────────────── composition ──────────────────────────────────

@dataclass
class _FigureRender:
    name: str
    prefix: str                        # "" or "Behind them, smaller and softer, "
    pose: str
    fields: dict[str, list[str]]      # field -> clauses, override applied
    extra: list[str]


@dataclass
class _Render:
    """The parts of a render, so a budget trim REMOVES a part instead of
    mutating a sentence — the old failure was the model improvising its own
    cuts when it ran out of room."""
    scene_line: str
    entity_line: str
    figures: list[_FigureRender]
    protect: set[str]
    lighting: str = ""
    palette: str = ""
    mood: str = ""


@dataclass
class Composed:
    prompt: str
    figures: int
    foreground: int
    cuts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # The figures actually painted, in prompt order. `figures` is a COUNT, so a
    # caller that needs to know WHICH characters ended up in the frame (images.py
    # asks for the ones with no grounded appearance) must not re-derive it from
    # the prompt text.
    cast: list["Figure"] = field(default_factory=list)
    # Names painted by name alone — no anchor, no verified appearance. The
    # caller (images.py) uses this to run one lookup pass instead of trusting a
    # rule the model may simply not have followed.
    ungrounded: list[str] = field(default_factory=list)


def _protect_names(anchors: dict[str, Anchor], figures: list[Figure]) -> set[str]:
    """Tokens a clause must not be lowercased when it lands mid-sentence."""
    out: set[str] = set()
    for name in list(anchors) + [f.name for f in figures]:
        for w in re.findall(r"[A-ZÄÖÜ][A-Za-zäöü]+", name):
            out.add(w.lower())
    return out


def _join(clauses: list[str], protect: set[str]) -> str:
    """One flowing sentence per character; only the leading clause keeps its case."""
    out: list[str] = []
    for i, c in enumerate(c for c in clauses if c):
        if i and c[0].isupper():
            first = re.match(r"[A-ZÄÖÜ][A-Za-zäöü]*", c)
            if first and first.group(0).lower() not in protect:
                c = c[0].lower() + c[1:]
        out.append(c)
    return ", ".join(out)


def _figure_text(fr: _FigureRender, protect: set[str]) -> str:
    clauses: list[str] = [fr.pose] if fr.pose else []
    # ANCHOR_FIELDS order for the party six (face before outfit etc.), then any
    # other bucket — an unanchored figure's looked-up `look` lives there.
    for f in (*ANCHOR_FIELDS, *(k for k in fr.fields if k not in ANCHOR_FIELDS)):
        clauses.extend(fr.fields.get(f, []))
    clauses.extend(fr.extra)
    desc = _join(clauses, protect)
    return f"{fr.prefix}{fr.name}: {desc}." if desc else f"{fr.prefix}{fr.name}."


def _render_text(r: _Render) -> str:
    lines = [r.scene_line]
    if r.entity_line:
        lines.append(r.entity_line)
    lines.extend(_figure_text(fr, r.protect) for fr in r.figures)
    tail_bits = [t for t in (
        f"Lighting: {r.lighting}." if r.lighting else "",
        f"Palette: {r.palette}." if r.palette else "",
        f"Mood: {r.mood}." if r.mood else "",
        NO_LETTERING_TAIL,
    ) if t]
    lines.append(" ".join(tail_bits))
    return " ".join(lines)


# A name the plan hid in `entities` is still a character in the frame.
_NAME_SHAPE = re.compile(
    r"^\s*((?:[A-ZÄÖÜ][A-Za-zÀ-ÿ'’-]+[, ]+){1,3}[A-ZÄÖÜ][A-Za-zÀ-ÿ'’-]+)")
# A name the scene only ever mentions after one of these is a place, not an actor.
_PLACE_PREPOSITIONS = frozenset(
    "at in of to from near inside through toward über bei von aus nach zu auf vor".split())


_POSE_ING = {
    "stand": "standing", "sit": "sitting", "lie": "lying", "crouch": "crouching",
    "kneel": "kneeling", "face": "facing", "battle": "battling",
    "attack": "attacking", "fight": "fighting", "charge": "charging",
    "advance": "advancing", "watch": "watching", "wait": "waiting",
    "strike": "striking", "move": "moving", "block": "blocking",
    "circle": "circling", "surround": "surrounding", "run": "running",
    "leap": "leaping", "dash": "dashing", "turn": "turning", "look": "looking",
    "stare": "staring", "glare": "glaring", "hold": "holding", "carry": "carrying",
    "raise": "raising", "aim": "aiming", "shoot": "shooting", "cast": "casting",
    "hammer": "hammering", "work": "working", "assist": "assisting",
    "clutch": "clutching", "brandish": "brandishing", "snarl": "snarling",
    "loom": "looming", "gather": "gathering", "point": "pointing",
}


def _pose_from_scene(scene: str, name: str) -> str:
    """What the scene sentence says this character is doing."""
    for seg in re.split(r"[;,]", scene):
        if not re.search(r"\b" + re.escape(name) + r"\b", seg, re.I):
            continue
        rest = re.sub(r"\b(?:and|with|while|as)\b", ",", seg)
        # Every name-shaped run goes; what is left is the action.
        rest = re.sub(r"\b[A-ZÄÖÜ][A-Za-zÀ-ÿ'’-]+(?:\s+[A-ZÄÖÜ][A-Za-zÀ-ÿ'’-]+){0,2}",
                      ",", rest)
        parts = [w.strip() for w in rest.split(",") if w.strip()]
        action = [p for p in parts if p.lower() not in _PLACE_PREPOSITIONS
                  and not re.match(r"^[A-ZÄÖÜ]", p)]
        if action:
            # "stand before them" is what the scene says; as a pose line it reads
            # as a person's verb. A pose is an -ing phrase.
            first = action[0].split()[0]
            action[0] = _POSE_ING.get(first.lower(), first) + action[0][len(first):]
            return " ".join(action).strip(" ,.")
        break
    return ""


def promote_named_entities(plan: ImagePlan) -> tuple[ImagePlan, list[str]]:
    """Move a named character out of `entities` and into the cast.

    `entities` is for monsters and objects; `figures` is for the cast. Measured on
    the live bandit request the model wrote "Boss Frey and his bandit crew" into
    the scene text only, which did two things at once: the two named characters
    escaped the grounding rules entirely, and the count line read "Exactly 6 named
    figures" while eight named beings were named in the same prompt — the exact
    shape that makes a diffusion model paint an extra unnamed crowd.

    A name counts as an actor when the scene says it and does not say it as a
    location ("at the forge of Winnowing Reach" stays a place).
    """
    figures = list(plan.figures)
    have = {str(getattr(f, "name", "")).lower() for f in figures}
    entities: list[str] = []
    notes: list[str] = []
    scene_low = _norm(plan.scene)
    # A remainder can name another character ("Boss Frey and Alisa Pennygleam"),
    # so the leftovers go back through the same test.
    queue = list(plan.entities)
    while queue:
        e = queue.pop(0)
        m = _NAME_SHAPE.match(e)
        name = m.group(1).strip(" ,&") if m else ""
        if not name or name.lower() in have:
            entities.append(e)
            continue
        actor = False
        for occ in re.finditer(re.escape(name), plan.scene, re.I):
            before = plan.scene[:occ.start()].rstrip().split()
            if before and before[-1].lower().strip(",.") in _PLACE_PREPOSITIONS:
                continue
            actor = True
            break
        if not actor or _norm(name) not in scene_low:
            entities.append(e)
            continue
        # The plan's own sentence already says what they are doing; taking the
        # pose from it is derivation, not invention, and a named figure with an
        # action paints better than a bare name.
        have.add(name.lower())
        figures.append(Figure(name=name, pose=_pose_from_scene(plan.scene, name)))
        rest = re.sub(r"^(?:(?:and|with|plus|his|her|their|the|a|an)\b[ ,]*)+",
                      "", e[m.end():].strip(" ,&"))
        if rest:
            queue.insert(0, rest)
        notes.append(f"entities named {name!r}, a character in the scene — painted "
                     "as a figure and looked up like one")
    if not notes:
        return plan, notes
    return replace(plan, figures=figures, entities=entities), notes


def request_spelling(name: str, request_text: str, backing_text: str = "") -> str:
    """The name the campaign uses, when the plan abbreviated it.

    Measured live (§15): the plan wrote `"name": "Frey"` for a character the
    request and every session log call **Boss Frey**, so the prompt painted
    "Frey:" — a different string from the one the party plays, the vault files,
    and every earlier image used.

    Three facts decide it, no shape-guessing:

    1. the request names this character (so an abbreviated bystander the model
       invented cannot be renamed onto some vault character of the same name),
    2. the material the loop actually read attaches a word to that name —
       "Boss Frey", "Sable Rin",
    3. and it **never writes the plan's form bare**: every occurrence of "Frey" in
       the notes is part of "Boss Frey".

    Test 3 is what separates a name from a title noun. The first live run of this
    guarantee renamed "Alisa Pennygleam" to "Händlerin Alisa Pennygleam", because
    a session note really does say "die befreite Händlerin Alisa Pennygleam" —
    German capitalises every noun, and a merchant is not part of her name. The
    vault writes her name bare a dozen times, so the bare form IS her name and the
    rename was refused.
    """
    if not name or not backing_text:
        return ""
    esc = re.escape(name)
    name_re = r"(?<![A-Za-zÀ-ÿ])" + esc + r"(?![A-Za-zÀ-ÿ])"
    if not re.search(name_re, request_text or "", re.I):
        return ""
    CAPWORD = r"[A-ZÄÖÜ][A-Za-zÀ-ÿ'’-]+"
    total = 0
    attached: list[tuple[str, list[str]]] = []   # (name as written, the words attached)
    for m in re.finditer(name_re, backing_text, re.I):
        total += 1
        tail = re.search(r"(?:" + CAPWORD + r" +){1,2}$", backing_text[:m.start()])
        if tail:
            attached.append((m.group(0),
                             re.findall(CAPWORD, tail.group(0))))
    if not attached or total - len(attached) > 0:
        return ""      # the vault writes the plan's form on its own: no drift
    forms: dict[str, int] = {}
    for written, words in attached:
        for n in (1, 2):
            if len(words) >= n:
                cand = " ".join(words[-n:]) + " " + written
                forms[cand] = forms.get(cand, 0) + 1
    if not forms:
        return ""
    best = sorted(forms.items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]
    return best if len(best) > len(name) else ""


def _apply_names(text: str, renames: dict[str, str]) -> str:
    """Rename mentions in a sentence without double-prefixing the correct ones."""
    for old, new in renames.items():
        # Mentions that ALREADY carry the new name are shielded behind a sentinel,
        # so "Boss Frey" cannot become "Boss Boss Frey".
        shielded = re.subn(
            r"(?<![A-Za-zÀ-ÿ])" + re.escape(new) + r"(?![A-Za-zÀ-ÿ])",
            "\x00", text, flags=re.I)[0]
        text = re.sub(r"(?<![A-Za-zÀ-ÿ])" + re.escape(old) + r"(?![A-Za-zÀ-ÿ])",
                      new, shielded, flags=re.I).replace("\x00", new)
    return text


def campaign_names(plan: ImagePlan, request_text: str, *,
                   anchors: dict[str, Anchor] | None = None,
                   retrieved: str = "") -> tuple[ImagePlan, list[str]]:
    """Rename plan figures to the spelling the campaign actually uses."""
    if not request_text:
        return plan, []
    backing = "\n".join([retrieved or ""] + list((anchors or {}).keys()))
    renames: dict[str, str] = {}
    notes: list[str] = []
    for f in plan.figures:
        nm = str(getattr(f, "name", "") or "").strip()
        if not nm or nm in renames:
            continue
        better = request_spelling(nm, request_text, backing)
        if better:
            renames[nm] = better
            notes.append(f"plan named {nm!r}, the request and the vault name "
                         f"{better!r} — painted with the campaign's name")
    if not renames:
        return plan, notes

    def _apply(text: str) -> str:
        return _apply_names(text, renames)

    figures = [replace(f, name=renames.get(
        str(getattr(f, "name", "") or "").strip(),
        str(getattr(f, "name", "") or "")),
        pose=_apply(str(getattr(f, "pose", "") or ""))) for f in plan.figures]
    return replace(plan, figures=figures,
                   scene=_apply(plan.scene),
                   entities=[_apply(e) for e in plan.entities]), notes


def compose_prompt(
    plan: ImagePlan,
    anchors: dict[str, Anchor],
    base_style: str,
    *,
    max_chars: int | None = None,
    foreground: int = 4,
    budget_kwargs: dict[str, int] | None = None,
    read_paths: Iterable[str] | None = None,
    snippet_paths: Iterable[str] | None = None,
    retrieved: str = "",
    request_text: str = "",
    animal_floor: str = _ANIMAL_FLOOR,
) -> Composed:
    """Assemble the final prompt from the plan + the vault anchors.

    The guarantees are now properties of this function rather than requests to a
    3-bit model: style sentence first, each named character described exactly
    once, a figure-count line written here, a foreground/background split when
    there are more figures than the model can hold, no negation clause anywhere,
    and a hard character budget trimmed in a declared order that reports every
    cut.
    """
    warnings: list[str] = []
    plan, renamed = campaign_names(plan, request_text, anchors=anchors,
                                   retrieved=retrieved)
    warnings.extend(renamed)
    plan, promoted = promote_named_entities(plan)
    warnings.extend(promoted)
    # Meta-commentary first, because everything below treats these strings as
    # descriptions of the picture.
    for label in ("scene", "mood", "palette", "lighting"):
        cleaned, asides = strip_meta_asides(getattr(plan, label))
        if asides:
            setattr(plan, label, cleaned)
            warnings.append(f"the plan's {label} carried a note about the pipeline "
                            f"({', '.join(asides)}) — stripped, because a diffusion "
                            "model paints comments")
    figures_clean: list[Figure] = []
    for f in plan.figures:
        asides: list[str] = []
        pose, a1 = strip_meta_asides(f.pose)
        look, a2 = strip_meta_asides(f.look)
        asides = a1 + a2
        figures_clean.append(replace(f, pose=pose, look=look) if asides else f)
        if asides:
            warnings.append(f"{f.name}: a note about the source or the guess was "
                            f"stripped from its figure line ({', '.join(asides)})")
    plan.figures = figures_clean
    ents_clean = []
    for e in plan.entities:
        cleaned, asides = strip_meta_asides(e)
        if asides:
            warnings.append(f"an entity ({e!r}) carried a note about the pipeline — "
                            "stripped")
        ents_clean.append(cleaned)
    plan.entities = ents_clean
    figures = [f for f in (_as_figure(x) for x in plan.figures) if f is not None]
    # A species claim smuggled into a NAME is still a species claim: a live run
    # produced figures called "crow-folk bandit 1" and "crow-folk bandit 2", and
    # the gate only ever policed the scene sentence and the entity line.
    policed: list[Figure] = []
    for f in figures:
        bad = unbacked_species(f.name, retrieved, request_text)
        if bad and strip_species(f.name, bad):
            swap = species_replacement(
                f.name, crowd_evidence_spans(f.name, retrieved,
                                             [f.name] + list(anchors)),
                request_species(request_text))
            name = (substitute_species(f.name, bad, swap) if swap
                    else strip_species(f.name, bad))
            warnings.append(f"{f.name}: the figure NAME claimed "
                            f"{'/'.join(bad)}-folk"
                            + (f" — painted as {swap}-folk" if swap else " — stripped"))
            policed.append(replace(f, name=name.strip() or f.name))
        else:
            policed.append(f)
    figures = policed
    if not figures and not plan.scene.strip():
        # An empty plan must never paint an empty prompt: the caller's raw-request
        # fallback is a better image than nothing.
        raise ValueError("image plan has neither figures nor a scene")
    # Dedupe AFTER resolving names, so "Seta" and "Seta Onix" are one figure.
    resolved = dedupe_figures([(f, match_anchor(f.name, anchors)) for f in figures])
    # Every name the plan named, before any budget cut: the scene sentence may
    # enumerate a figure the budget later drops, and that mention is still one
    # too many.
    plan_names = [a.name if a is not None else f.name for f, a in resolved]
    keep: list[tuple[Figure, Anchor | None]] = []
    ungrounded: list[str] = []
    for f, a in resolved:
        if a is not None:
            keep.append((f, a))
            continue
        if not f.look:
            warnings.append(f"{f.name}: no party anchor and no `look` — painted "
                            "by name alone (a named NPC must be looked up first)")
            ungrounded.append(f.name)
            keep.append((f, a))
            continue
        if len(f.pose.split()) > _MAX_POSE_WORDS:
            warnings.append(f"{f.name}: pose is {len(f.pose.split())} words, over the "
                            f"contract's {_MAX_POSE_WORDS} — kept whole, because the "
                            "composer only spends a pose, it never rewrites one")
        if len(f.look.split()) > _MAX_LOOK_WORDS:
            warnings.append(f"{f.name}: look is {len(f.look.split())} words, over the "
                            f"contract's {_MAX_LOOK_WORDS} — a look is not cut, so "
                            "shorten it in the plan")
        ok, why = verify_look(f, read_paths, snippet_paths, retrieved,
                              request_text)
        if why:
            warnings.append(f"{f.name}: {why}")
        if ok:
            # A species the vault never states, painted anyway: allowed by the
            # animal-folk ruling, but said out loud — an invented species in a
            # painted image is canon the table should know the bot invented.
            for token in species_verdict(
                    f.look, evidence_about(f.name, retrieved),
                    request_species(request_text))[1]:
                warnings.append(f"{f.name}: given a species ({token}) the vault "
                                "does not name — kept, because an animal-folk guess "
                                "is what keeps humans out of the picture")
            keep.append((f, a))
            continue
        ungrounded.append(f.name)
        # Never mutate the caller's plan: an unverifiable look is dropped on a
        # copy, so the plan the log shows is the plan the model wrote.
        keep.append((replace(f, look=""), None))
    resolved = keep

    # The sheet's spelling of a party member, EVERYWHERE — not only in the figure
    # line. Measured live (§15): the figure lines said "Seta Onix, Romi
    # Stachelpfote" while the scene sentence in the same prompt said "Seta, Bubo,
    # Romi, Colum, Trixy". The renderer reads both, and a name that looks
    # different from the one already described is a name that gets painted again.
    spelling = {f.name: a.name for f, a in resolved
                if a is not None and a.name != f.name}
    # …and the same for a name the SCENE abbreviated on its own: the plan listed
    # "Romi Stachelpfote" correctly but wrote "Romi" in the sentence. A leading
    # token of a sheet name is only substituted when no other sheet entry starts
    # with it, so an ambiguous short name is left alone rather than guessed.
    for f, a in resolved:
        if a is None:
            continue
        toks = a.name.split()
        for n in range(1, len(toks)):
            pref = " ".join(toks[:n])
            if any(o != a.name and _norm(o).startswith(_norm(pref) + " ")
                   for o in anchors):
                continue
            spelling.setdefault(pref, a.name)
    if spelling:
        plan = replace(plan, scene=_apply_names(plan.scene, spelling),
                       entities=[_apply_names(e, spelling) for e in plan.entities])
        resolved = [(replace(f, name=a.name, pose=_apply_names(f.pose, spelling)), a)
                    if a is not None else (replace(f, pose=_apply_names(f.pose,
                                                                        spelling)), a)
                    for f, a in resolved]
        warnings.append("plan abbreviated " + ", ".join(
            f"{o!r} for {spelling[o]!r}" for o in sorted(spelling))
            + " — the sheet's spelling is used wherever the name appears")

    # The foreground is for the characters the party actually plays. A plan that
    # listed bandits before the party (measured in the live dry run) would otherwise
    # paint the monsters large and the player characters "smaller and softer".
    anchored = [p for p in resolved if p[1] is not None]
    unanchored = [p for p in resolved if p[1] is None]
    if anchored and unanchored and resolved != anchored + unanchored:
        moved = [f.name for f, _ in unanchored]
        resolved = anchored + unanchored
        warnings.append(f"plan listed {', '.join(moved)} ahead of the party — "
                        "reordered so the player characters hold the foreground")

    budget = max_chars if max_chars is not None else prompt_budget(
        len(resolved), **(budget_kwargs or {}))
    protect = _protect_names(anchors, [f for f, _ in resolved])

    # The palette worn by two or more figures of THIS image (analysis §3).
    counts: dict[str, int] = {}
    for f, a in resolved:
        texts: list[str] = [f.look]
        if a is not None:
            texts.extend(a.fields.values())
            texts.append(a.blob)
            texts.extend(a.extra)
        for t in texts:
            for w in _palette_of(t):
                counts[w] = counts.get(w, 0) + 1
    shared = {w for w, n in counts.items() if n > 1}

    scene = plan.scene.strip().rstrip(". ")
    if scene:
        scene = scene[0].upper() + scene[1:]
    # The contract's own word cap, enforced: a runaway scene sentence is how an
    # 11-figure request became 3,468 characters of mush with no anchors left (§4).
    if len(scene.split()) > _MAX_SCENE_WORDS:
        words = scene.split()
        scene = " ".join(words[:_MAX_SCENE_WORDS]).rstrip(" ,;.")
        warnings.append(f"scene sentence ran to {len(words)} words — trimmed to the "
                        f"contract's {_MAX_SCENE_WORDS}; the anchor lines are what "
                        "the model paints the characters from")
    # A species claim in the SCENE is policed too (analysis §15), but since the
    # table owner's ruling of 2026-11-19 it polices CONTRADICTION rather than
    # invention: this world has no humans, so a bandit crew the vault never
    # describes must still come out animal-folk. A claim that disagrees with what
    # the material says about those beings is stripped; a claim about beings nobody
    # describes is kept and reported.
    spans = crowd_evidence_spans(scene, retrieved, plan_names + list(anchors))
    bad_scene = contradicted_species(scene, spans, request_species(request_text))
    if bad_scene and strip_species(scene, bad_scene):
        # "The vault's species wins" is only a rule if the vault's ANIMAL is what
        # gets painted. SESSION_01 calls the ambush boss a Fuchsdame: stripping
        # "badger-folk" leaves "bandits", which is what a renderer paints as
        # people. Substitute the animal the vault itself uses instead.
        swap = species_replacement(
            scene, spans, request_species(request_text),
            exclude=_frame_species(resolved))
        if swap:
            scene = substitute_species(scene, bad_scene, swap)
            warnings.append(f"scene sentence claimed {'/'.join(bad_scene)}-folk; the "
                            f"material about those beings says {swap} — painted as "
                            f"{swap}-folk, because that is what the campaign has")
        else:
            scene = strip_species(scene, bad_scene)
            warnings.append(f"scene sentence claimed {'/'.join(bad_scene)}-folk while "
                            "the material about those beings says otherwise — the "
                            "species is stripped and the crowd is painted by what it "
                            "is called")
    for token in species_guesses(scene, retrieved, request_text):
        warnings.append(f"scene sentence gives the crowd a species ({token}-folk) "
                        "the vault does not name — kept, because an animal-folk "
                        "guess is what keeps humans out of the picture; put the "
                        "real species in the vault to replace it")
    if is_collective(scene):
        names = [a.name if a is not None else f.name for f, a in resolved]
        scene2, hits = replace_collectives(scene, names)
        if hits:
            # Not a rewrite of the sentence — a subject swapped for a subject.
            warnings.append(f"scene sentence named the cast collectively — "
                            f"replaced with their names ({', '.join(names)})")
            scene = scene2
        else:
            # A crowd noun naming somebody else ("a party of bandits") or one that
            # cannot be swapped safely: reported, and the count line is what holds
            # the cast in place.
            warnings.append("scene sentence names the cast collectively — the figure "
                            "count line is what holds the cast in place")

    # A plan that enumerates the cast inside the scene sentence AS WELL as in
    # the figure lines mentions every name three times, and a name mentioned
    # three times is what got a second Colum painted in the background (§4.2).
    # Pure-name segments of a trailing enumeration are cut; a mention that
    # carries the character's own action stays.
    scene, cut_names = drop_bare_repeat_mentions(scene, plan_names)
    if cut_names:
        warnings.append("scene sentence re-enumerated the cast — the bare repeat "
                        f"mentions were cut ({', '.join(cut_names)}); the figure "
                        "lines are where each character is described")

    count = len(resolved)
    front_n = min(foreground, count)
    plural = "s" if count != 1 else ""
    count_line = f"Exactly {count} named figure{plural} in this image"
    if count > front_n:
        count_line += f", {front_n} in the foreground and {count - front_n} behind them"
    style = base_style.rstrip(". ")
    # The floor line and a machine in the frame have to agree before either is said.
    # Marvin's sheet line says "steampunk magical construct" and his Visual Anchor
    # says automaton: a fact from the vault, not a guess about what a name looks like.
    if animal_floor and any(is_machine(a.desc if a else "", f.look, f.source,
                                       *(a.fields.values() if a else ()))
                           for f, a in resolved):
        animal_floor = _ANIMAL_FLOOR_LIVING
    # Nobody in this world is a person (the table owner's ruling, 2026-11-19): the
    # floor line rides in the style sentence, which the budget ladder never cuts,
    # so a character or crowd the vault gives no species for is still painted
    # animal-folk instead of human. Unconditional on purpose — deciding which
    # entities are beings and which are wagons is exactly the guesswork this
    # pipeline refuses to do, and 80 characters is what certainty costs here.
    if animal_floor:
        style = f"{style}. {animal_floor.rstrip('. ')}"
    scene_line = (f"{style}. {scene}. {count_line}." if scene
                  else f"{style}. {count_line}.")

    entity_line = ""
    entity_entries: list[str] = []
    if plan.entities:
        ents: list[str] = []
        for e in plan.entities:
            spans_e = crowd_evidence_spans(e, retrieved, plan_names + list(anchors))
            bad_e = contradicted_species(e, spans_e, request_species(request_text))
            if bad_e and strip_species(e, bad_e):
                swap = species_replacement(
                    e, spans_e, request_species(request_text),
                    exclude=_frame_species(resolved))
                if swap:
                    ents.append(substitute_species(e, bad_e, swap).rstrip(". "))
                    warnings.append(f"entities claimed {'/'.join(bad_e)}-folk; the "
                                    f"material says {swap} — painted as {swap}-folk")
                    continue
                warnings.append(f"entities claimed {'/'.join(bad_e)}-folk while the "
                                "material about them says otherwise — the species "
                                "is stripped")
                ents.append(strip_species(e, bad_e).rstrip(". "))
            else:
                ents.append(e.rstrip(". "))
                for token in species_guesses(e, retrieved, request_text):
                    warnings.append(f"entities give {'/'.join(_claimed_species(e))} "
                                    f"a species ({token}-folk) the vault does not "
                                    "name — kept, because an animal-folk guess is "
                                    "what keeps humans out of the picture")
        # A POSITIVE label: "Not characters, but present…" is a negation, and the
        # renderer does not read the distinction.
        entity_line = f"Also in the scene: {', '.join(ents)}."
        entity_entries = ents

    # A collective in a FIGURE LINE is a different case from one in the scene:
    # the names are already the subject of that line, so they cannot be inserted
    # again — the crowd word is removed instead. Round 6: "Boss Frey: snarling
    # forward, sabre leveled at the party".
    cleaned: list[tuple[Figure, Anchor | None]] = []
    for f, a in resolved:
        pose, hits = drop_collectives(f.pose)
        look, hits2 = drop_collectives(f.look)
        if hits or hits2:
            warnings.append(f"{f.name}: the figure line said "
                            f"{(hits or hits2)[0]!r} — a cast member named twice is "
                            "a character painted twice, so the crowd word was "
                            "removed rather than the names added")
            cleaned.append((replace(f, pose=pose, look=look), a))
        else:
            cleaned.append((f, a))
    resolved = cleaned
    ents2: list[str] = []
    for e in plan.entities:
        kept, hits = drop_collectives(e)
        if hits:
            warnings.append(f"an entity ({e!r}) said {hits[0]!r} — the crowd word "
                            "was removed; the figures already name the cast")
        ents2.append(kept)
    plan.entities = ents2

    renders: list[_FigureRender] = []
    for i, (f, a) in enumerate(resolved):
        pose = f.pose.strip().rstrip(". ")
        if pose and has_negation(pose):
            warnings.append(f"{f.name}: negation in the pose ({pose!r}) — dropped; "
                            "a removal is painted by leaving the item out")
            pose = ""
        fields: dict[str, list[str]] = {}
        extra: list[str] = []
        if a is not None:
            unknown = sorted(k for k in f.modify if k not in ANCHOR_FIELDS)
            if unknown:
                warnings.append(f"{f.name}: override names unknown field(s) "
                                f"{', '.join(unknown)} — ignored (the fields are "
                                f"{', '.join(ANCHOR_FIELDS)})")
            raw_fields, dropped = a.clauses_for(f.modify)
            if f.modify and not a.typed:
                warnings.append(f"{f.name}: legacy anchor has no fields to "
                                "replace — the override filtered the clauses that "
                                "read like that subject instead (split the sheet's "
                                "Visual Anchor into field lines for exact "
                                "replacement)")
            for bad in dropped:
                warnings.append(f"{f.name}: negation ({bad!r}) — not written into "
                                "the prompt; a removal is painted by omission")
            for name, clauses in raw_fields.items():
                fields[name] = _unique_first(clauses, shared)
            extra = list(a.extra)
        else:
            for bad in (c for c in _split(f.look) if has_negation(c)):
                warnings.append(f"{f.name}: negation in its look ({bad!r}) — not "
                                "written into the prompt")
            fields["look"] = _unique_first(_clauses(f.look), shared)
        renders.append(_FigureRender(
            # The vault spelling of the name, when an anchor matched: it is the
            # name the sheet and every earlier image used, and a plan abbreviation
            # ("Seta") must not drift into a different character.
            name=a.name if a is not None else f.name,
            prefix="Behind them, smaller and softer, " if i >= front_n else "",
            pose=pose, fields=fields, extra=extra))

    r = _Render(scene_line=scene_line, entity_line=entity_line, figures=renders,
                protect=protect, lighting=plan.lighting.strip().rstrip(". "),
                palette=plan.palette.strip().rstrip(". "),
                mood=plan.mood.strip().rstrip(". "))
    prompt = _render_text(r)

    # ── Budget: remove parts, then clauses, in a declared order. Every cut is
    # reported: the old failure was the model improvising its own cuts when it
    # ran out of room, which is how a Visual Anchor quietly loses its bow.
    # Never the style sentence, never the count line, never a character's name.
    def cut(note: str) -> None:
        nonlocal prompt
        warnings.append(note)
        prompt = _render_text(r)

    if len(prompt) > budget:
        r.mood = ""
        cut(f"budget {budget}: mood line dropped")
    if len(prompt) > budget:
        r.palette = ""
        cut(f"budget {budget}: palette line dropped — the anchors' own colours "
            "already set it")
    if len(prompt) > budget:
        for fr in r.figures:
            fr.fields.pop("quirks", None)
            fr.extra = []
        cut(f"budget {budget}: quirks and extra-label clauses dropped")
    if len(prompt) > budget and r.entity_line:
        # "Paint Boss Frey und seinen Banditen" and the budget deletes the bandits:
        # the user's own crowd is the least cuttable thing in the image. Entries
        # that share a content word with the request survive; the rest go first.
        keep_ents = [e for e in entity_entries if _named_in_request(e, request_text)]
        dropped = [e for e in entity_entries if e not in keep_ents]
        if keep_ents:
            r.entity_line = f"Also in the scene: {', '.join(keep_ents)}."
            cut(f"budget {budget}: entity line trimmed to what the request named "
                f"({', '.join(dropped)}) — the user's own crowd is not what the "
                "budget cuts")
        else:
            r.entity_line = ""
            cut(f"budget {budget}: entity line dropped — a creature the scene "
                "names belongs in the scene sentence")

    # Clause-by-clause compression: the LEAST distinctive clause of the whole
    # image goes first, one at a time. Distinctiveness = colours this figure
    # alone wears, weighted by field (a signature item or a face feature
    # identifies a character; a colour three of them share identifies nobody).
    def score(field: str, clause: str, background: bool) -> int:
        """What a clause is worth in THIS image: content, then the colours only
        this figure wears, then what kind of thing it says.

        A long, detail-rich clause survives even when its colours are shared
        (Bubo's star-embroidered robe: five shared colours, 42 points of content
        — it is still the single most identifying thing about him), while a
        one-word clause like "upright" goes first. Background figures are painted
        "smaller and softer", so detail spent on them buys less.
        """
        pal = _palette_of(clause)
        return (3 * len(clause.split()) + 8 * len(pal - shared)
                - 2 * len(pal & shared) + _FIELD_WEIGHT.get(field, 0)
                - (6 if background else 0))

    dropped_notes: list[str] = []
    # Pass 1 may not touch an identity clause; pass 2 exists so the ladder can
    # still finish instead of hard-truncating a sentence mid-name.
    for allow_identity_cut in (False, True):
        while len(prompt) > budget:
            best: tuple[int, int, str, int] | None = None   # score, fig, field, idx
            for fi, fr in enumerate(r.figures):
                background = fi >= front_n
                for field, clauses in fr.fields.items():
                    if (field in _IDENTITY_FIELDS
                            and not allow_identity_cut):
                        continue
                    for ci, clause in enumerate(clauses):
                        sc = score(field, clause, background)
                        if best is None or sc < best[0]:
                            best = (sc, fi, field, ci)
            if best is None:
                break
            _, fi, field, ci = best
            fr = r.figures[fi]
            gone = fr.fields[field].pop(ci)
            if not fr.fields[field]:
                del fr.fields[field]
            dropped_notes.append(f"{fr.name} [{field}] {gone!r}")
            prompt = _render_text(r)

    if dropped_notes and len(prompt) <= budget:
        cut(f"budget {budget}: {len(dropped_notes)} least-distinctive clause(s) "
            f"compressed — {', '.join(dropped_notes)}")

    if len(prompt) > budget:
        r.lighting = ""
        cut(f"budget {budget}: lighting line dropped")
    if len(prompt) > budget:
        prompt = prompt[:budget].rsplit(" ", 1)[0]
        warnings.append(f"budget {budget}: prompt hard-truncated — this scene "
                        "asks for more figures than one image can hold")

    if shared:
        log.debug("shared palette tokens in this scene: %s", ", ".join(sorted(shared)))
    cuts = [w for w in warnings if w.startswith("budget ")]
    return Composed(prompt=prompt, figures=count, foreground=front_n,
                    cuts=cuts, warnings=warnings,
                    cast=[f for f, _ in resolved if f.name],
                    ungrounded=ungrounded)


# ─────────────────────────── the party sheet ──────────────────────────────

def parse_party_sheet(raw: str) -> list[Anchor]:
    """One Anchor per character block of the vault party sheet.

    Block shape (unchanged convention, one addition): `# Name` → a free line
    (species/class/pronouns) → portraits → `## …` sections → `## Visual Anchor`
    followed by a blockquote. That blockquote is either the old single paragraph
    (→ one blob, painted verbatim) or the field form (`> outfit: …`, → typed).
    Characters with no anchor block are skipped: there is nothing to paint them
    from, and the model's own creature rules apply to them.
    """
    out: list[Anchor] = []
    name = desc = ""
    block: list[str] = []
    in_anchor = False
    in_fence = False

    def flush() -> None:
        if name and block:
            out.append(parse_anchor(name, desc, block))

    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            if in_anchor:
                block.append(s)
            continue
        if in_fence:
            continue        # a '# ' inside a stat block is not a character heading
        if s.startswith("# "):
            flush()
            name, desc, block, in_anchor = s[2:].strip(), "", [], False
        elif name and not desc and s and not s.startswith(("#", ">", "!", "[")):
            desc = s
        elif s.startswith("## "):
            if in_anchor and block:
                flush()
                block = []
            in_anchor = "visual anchor" in s.lower()
        elif in_anchor and s.startswith(">"):
            block.append(s)
    flush()
    return out
