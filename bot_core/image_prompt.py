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

# The scene-sentence word cap, straight from the plan contract. A runaway scene
# sentence is measured failure 5 (an 11-figure request became 3,468 characters of
# mush with no anchors left), so the composer enforces the cap rather than asking
# for it.
_MAX_SCENE_WORDS = 45

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


def replace_collectives(text: str, names: list[str]) -> tuple[str, int]:
    """Swap a determiner-anchored crowd noun for the cast's own names.

    Grammar-safe: a subject is replaced by a subject. This is the part that
    actually stops the extra-figures render; detection without replacement would
    just log a failure that still got painted.
    """
    out = text
    hits = 0
    for pat in _COLLECTIVE_REPLACE:
        def sub(m: re.Match) -> str:
            nonlocal hits
            hits += 1
            return ", ".join(names) if names else m.group(0)
        out = re.sub(pat, sub, out, flags=re.I)
    return out, hits


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
)
_species_variants = {w: frozenset(variants) for variants in _SPECIES for w in variants}
_all_species = frozenset(_species_variants)


def invented_species(look: str, retrieved: str) -> list[str]:
    """Species the description claims that nothing retrieved mentions.

    Generic words (beast-folk, anthropomorphic) name no species and are not
    policed. A specific one is a fact claim, and the vault has its own notions
    of who is a badger-folk.
    """
    if not retrieved:
        return []
    hay = _norm(retrieved)
    bad = []
    for token in set(re.findall(r"[a-z\u00e4\u00f6\u00fc]+", _norm(look))):
        if token not in _all_species:
            continue
        if any(re.search(r"\b" + re.escape(v) + r"\b", hay)
               for v in _species_variants[token]):
            continue
        bad.append(token)
    return sorted(bad)


def verify_look(figure: Figure, read_paths: Sequence[str] | None,
                snippet_paths: Iterable[str] | None,
                retrieved: str) -> tuple[bool, str]:
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
    bad = invented_species(figure.look, retrieved)
    if bad:
        return False, ("describes it as " + "/".join(bad) + "-folk, which nothing "
                       "retrieved says — the vault never gives this character a "
                       "species, so the figure is painted by name")
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
    figures = [f for f in (_as_figure(x) for x in plan.figures) if f is not None]
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
        ok, why = verify_look(f, read_paths, snippet_paths, retrieved)
        if why:
            warnings.append(f"{f.name}: {why}")
        if ok:
            keep.append((f, a))
            continue
        ungrounded.append(f.name)
        # Never mutate the caller's plan: an unverifiable look is dropped on a
        # copy, so the plan the log shows is the plan the model wrote.
        keep.append((replace(f, look=""), None))
    resolved = keep

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
    scene_line = (f"{style}. {scene}. {count_line}." if scene
                  else f"{style}. {count_line}.")

    entity_line = ""
    if plan.entities:
        ents = ", ".join(e.rstrip(". ") for e in plan.entities)
        # A POSITIVE label: "Not characters, but present…" is a negation, and the
        # renderer does not read the distinction.
        entity_line = f"Also in the scene: {ents}."

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
    if len(prompt) > budget:
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
