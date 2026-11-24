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

from bot_core import prompt_profile
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
    # Not a species JUDGMENT (the composer decides no truth about a claim, §20): this
    # list only says which clauses of a LEGACY single-paragraph anchor are the identity
    # clauses, so an override naming one field can find the clauses to replace. Generic
    # kind words first; the animals are here because this vault's legacy blobs name them.
    "species": ("species", "kind", "creature", "being", "race", "ancestry",
                "humanoid", "animal", "mammal", "bird", "reptile", "automaton",
                "construct", "golem", "robot", "folk", "folkish", "-folk",
                "bird-folk", "beast-folk", "lizard", "raccoon", "hedgehog", "owl"),
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




























# A collective head noun is plural for this purpose: "bandit crew", "the party".



































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

    Whether the SPECIES named in a description is the campaign's is no longer asked
    here (retired 2026-11-20, analysis §19–§20): the gate that asked it read species
    words out of retrieved prose and attributed other characters' animals to the
    character being checked, which dropped a look the rewrite had genuinely opened a
    file for. What stays is the question code can answer without interpreting
    language: was this description retrieved, or invented.

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

# ── Reporting: what the composer SEES but no longer rewrites ─────────────
#
# Until 2026-11-19 this module rewrote the plan's own sentences: it stripped species
# claims, swapped in animals it found nearby, replaced collective nouns with names,
# cut repeat mentions, corrected name spellings and deleted parentheticals. Ten
# evaluation rounds (analysis §17) showed every one of them failing in a different
# campaign-shaped way, and the ruling was that a diffusion prompt is not a place for
# a regex to reinterpret the table's words.
#
# So the same detectors survive, and nothing they find is edited. A Finding goes into
# `Composed.findings`; bot_core/images.py hands the fixable ones back to the rewrite
# model ONCE, in its own language, with the vault in front of it.

_META_COMMENT = re.compile(
    r"\(([^()]*\b(?:bot|vault|guess|guessed|guessing|invented|species|unverified|"
    r"unconfirmed|no description|no appearance|not in the vault|the plan|retrieved|"
    r"notes?|assumption|assumed|unclear|unknown|speculation|presumably)\b[^()]*)\)",
    re.IGNORECASE)

#: Findings the rewrite model can act on. The rest are facts about the frame
#: (a figure with no anchor, a budget cut) that no amount of rewording changes.
FIXABLE_CODES = frozenset({
    "meta_comment", "collective_subject", "collective_in_figure",
    "collective_in_entity", "duplicate_mention", "name_spelling",
    "negation_clause", "entity_named_not_painted", "figure_ungrounded",
    "identity_left_in_scene", "look_unverified", "name_form",
    "same_person_listed_twice", "scene_dangling_name", "figure_thin_line",
})


@dataclass(frozen=True)
class Finding:
    """Something the composer noticed and deliberately did NOT fix."""

    code: str
    subject: str          # a figure name, or "scene" / "entities" / "mood"
    detail: str           # one sentence, quoting both sides of the disagreement

    def as_line(self) -> str:
        return f"{self.code} [{self.subject}]: {self.detail}"

    @property
    def fixable(self) -> bool:
        return self.code in FIXABLE_CODES


def _meta_comments(text: str) -> list[str]:
    return [m.group(1).strip() for m in _META_COMMENT.finditer(text or "")]


def _collective_phrase(text: str) -> str:
    """Which crowd phrase the text uses, for the report ("" when there is none).

    The matched phrase plus the qualifying words in front of it, so the finding
    quotes "the whole upright party" instead of the single noun that matched first.
    Display only — detection stays `is_collective`'s patterns, unchanged.
    """
    t = _norm(text)
    best: re.Match | None = None
    for pattern in _COLLECTIVE_PATTERNS:
        m = re.search(pattern, t)
        if m and (best is None or len(m.group(0)) > len(best.group(0))):
            best = m
    if best is None:
        return ""
    phrase = best.group(0).strip()
    head = phrase.split()[-1]
    keep: list[str] = []
    for w in reversed(t[:best.start()].split()):
        if w in {"the", "a", "an", "their", "his", "her", "its", "our", "whole",
                 "entire", "full", "main", "standing", "upright"}:
            keep.insert(0, w)
        else:
            break
    return " ".join(keep + [phrase])


def _scene_description(scene: str, name: str) -> str:
    """The description a plan hung on *name* inside the scene sentence, if any.

    "Boss Frey, a badger-folk bandit chief, stands alone under the moon" →
    "a badger-folk bandit chief". Bracket-walking again: a name, a comma, text, a
    comma. No reading of the text, and it never invents a description — a name with
    no apposition returns "".
    """
    for m in re.finditer(re.escape(name) + r"\s*,", scene):
        i = m.end()
        while i < len(scene) and scene[i] == " ":
            i += 1
        depth = 0
        for j in range(i, len(scene)):
            ch = scene[j]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                desc = scene[i:j].strip()
                if desc and name.lower() not in desc.lower():
                    return desc
                break
    return ""


def _introductions(text: str, name: str) -> int:
    """Mentions of *name* that INTRODUCE the character, not ones that lean on them.
    "…perches on Trixy Smoldersome's shoulder" says where another figure is; it does
    not bring Trixy into the frame a second time. A genitive after the name is the
    mechanical signal for that, and it is arithmetic on punctuation rather than a
    reading of the sentence — which is the line this file holds itself to.
    """
    if not name:
        return 0
    return sum(1 for m in re.finditer(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])",
                                      text)
               if not re.match(r"^\s*(?:'s|’s|s\b)", text[m.end():]))


def _negated_words(text: str) -> tuple[str, ...]:
    return tuple(m.group(0) for m in re.finditer(
        r"(?<![a-z])(?:no|not|without|never|avoid|devoid|instead of)(?![a-z])",
        _norm(text or "")))


def _repeat_mentions(scene: str, names: Sequence[str]) -> list[str]:
    """Names the scene sentence says more than once, where a figure line already
    describes them.

    Counting, not parsing: this is the §4.2 hazard (a name mentioned three times
    painted a second Colum in the background) stated as a FACT about the sentence
    rather than as a decision about which comma-segments are "bare". A repeat that
    carries an action is a sentence the model meant to write — which is exactly why
    it is reported to the model and not cut by a regex.
    """
    if not scene or len(names) < 2:
        return []
    low = _norm(scene)
    out: list[str] = []
    for name in sorted(names, key=lambda n: -len(_norm(n))):
        full = _norm(name)
        if not full:
            continue
        hits = _introductions(low, full)
        # Count each mention ONCE: the full name is blanked before its short form is
        # counted, or "Boss Frey" reads as a second mention of Frey and every party
        # member with a distinctive surname looks over-described.
        rest = re.sub(r"(?<![a-z])" + re.escape(full) + r"(?![a-z])", " @ ", low)
        short = _norm(name.split()[-1])
        if short != full:
            hits += _introductions(rest, short)
        if hits > 1:
            out.append(name)
    return sorted(set(out))


def _spelling_gaps(plan: "ImagePlan", anchors: dict[str, Anchor]) -> list[tuple[str, str, str]]:
    """A sheet name the plan wrote short — reported, not substituted.

    The figure LINES are still normalised to the sheet's spelling (that is identity,
    not prose: "Seta" and "Seta Onix" must dedupe to one figure), but the scene
    sentence is the model's sentence and it gets told rather than edited.
    """
    # Where a sheet name was shortened, and where it was shortened FROM: the scene
    # sentence and another figure's POSE both get read by the renderer, and a name
    # that looks different from the one already described is a name painted twice.
    haystacks = [("scene", plan.scene)] + [(f.name, f.pose) for f in plan.figures
                                           if isinstance(f, Figure)]
    out: list[tuple[str, str, str]] = []
    for name in anchors:
        toks = name.split()
        for n in range(1, len(toks)):
            short = " ".join(toks[:n])
            if any(o != name and _norm(o).startswith(_norm(short) + " ")
                   for o in anchors):
                continue                      # ambiguous short form: not our call
            for subject, text in haystacks:
                if re.search(r"(?<![A-Za-z])" + re.escape(short) + r"(?![A-Za-z])",
                             text or "", re.IGNORECASE) \
                        and _norm(name) not in _norm(text or ""):
                    out.append((short, name, subject))
    return out


def lint_plan(plan: ImagePlan, anchors: dict[str, Anchor], *,
              retrieved: str = "",
              read_paths: Iterable[str] | None = None,
              snippet_paths: Iterable[str] | None = None,
              request_text: str = "") -> list[Finding]:
    """Read the plan the way the composer reads it, and say what is odd about it.

    No text is changed. Every finding is aimed at the model that wrote the plan, in
    the same turn budget the lookup pass already uses.
    """
    findings: list[Finding] = []
    add = findings.append
    names = [f.name for f in plan.figures if isinstance(f, Figure)]

    for label in ("scene", "mood", "palette", "lighting"):
        for note in _meta_comments(getattr(plan, label)):
            add(Finding("meta_comment", label,
                        f"{note!r} is a note about the bot or the vault, not something "
                        "an image can paint — take it out of that field"))
    for f in plan.figures:
        if not isinstance(f, Figure):
            continue
        for note in _meta_comments(f.pose) + _meta_comments(f.look):
            add(Finding("meta_comment", f.name,
                        f"{note!r} is pipeline commentary inside a figure line"))
        if is_collective(f.pose) or is_collective(f.look):
            phrase = _collective_phrase(f.pose) or _collective_phrase(f.look)
            add(Finding("collective_in_figure", f.name,
                        f"the figure line says {phrase!r} — this line is about "
                        f"{f.name}, and a cast member named twice is a character "
                        "painted twice"))
        if has_negation(f.pose):
            add(Finding("negation_clause", f.name,
                        f"the pose {f.pose!r} says what to leave OUT; a renderer "
                        "paints the item anyway — write the pose positively"))
        if not f.look and not f.source and match_anchor(f.name, anchors) is None:
            # A description the plan DID write — in the scene sentence — while the
            # figure's own line went out as a bare pose. Measured live 2026-11-19
            # (§18): the request for Boss Frey alone produced "Boss Frey, a badger-folk
            # bandit chief, stands alone in a mossy forest clearing" together with
            # `"look": ""`, and his figure line was "Boss Frey: raising a heavy battle
            # axe overhead". The composer does not carry a description from one field
            # to another — that is language editing — but it can point at it.
            said = _scene_description(plan.scene, f.name)
            if said:
                add(Finding("identity_left_in_scene", f.name,
                            f"you described {f.name} as {said!r} in the scene sentence "
                            f"but its figure has no `look`, so its own line is painted "
                            f"without that description — put it in this figure's `look` "
                            f"field, with where you found it in `source`"))
            else:
                add(Finding("figure_ungrounded", f.name,
                            "no Visual Anchor and no `look` — this name is painted "
                            "with no appearance at all; look it up or drop the figure"))

    # A name the sentence brings up and then drops. The composer never edits the
    # model's prose, and a TRIM is not allowed to create this shape either (see
    # _trim_scene); when the model wrote it itself, the correction pass is told.
    for name in _dangling_tail_names(plan.scene, [f.name for f in plan.figures]):
        add(Finding("scene_dangling_name", name,
                    f"the scene sentence ends on the bare name {name!r} with nothing "
                    "said about what it does — a label, not a figure. Give it its own "
                    "clause with an action, or leave it out of the sentence: its "
                    "figure line already says what it looks like and what it is doing"))

    for e in plan.entities:
        for note in _meta_comments(e):
            add(Finding("meta_comment", "entities",
                        f"{note!r} inside an entity entry"))
        if is_collective(e):
            add(Finding("collective_in_entity", "entities",
                        f"{e!r} contains {_collective_phrase(e)!r} — the figure lines "
                        "already name the cast"))
        if has_negation(e):
            add(Finding("negation_clause", "entities",
                        f"{e!r} says what an object is NOT; a renderer paints the "
                        "thing it names — describe it positively"))

    if has_negation(plan.scene):
        add(Finding("negation_clause", "scene",
                    f"the scene sentence says what to leave OUT "
                    f"({', '.join(_negated_words(plan.scene))!r}); a renderer paints "
                    "the thing a sentence names — describe the frame positively"))
    if is_collective(plan.scene):
        add(Finding("collective_subject", "scene",
                    f"the scene names the cast collectively ({_collective_phrase(plan.scene)!r}) "
                    "— name them, or let the figure lines do it: a crowd noun on top of "
                    "an enumerated cast is painted as extra unnamed figures"))
    for note in _meta_comments(plan.scene):
        pass                      # already reported above
    for short, full, subject in _spelling_gaps(plan, anchors):
        add(Finding("name_spelling", subject,
                    f"{subject} says {short!r} while the sheet and the figure lines "
                    f"use {full!r} — one character, two spellings, two figures"))
    for name in _repeat_mentions(plan.scene, names):
        add(Finding("duplicate_mention", name,
                    "the scene enumerates this name again, and the figure line already "
                    "describes them — a name said three times is a character painted "
                    "twice"))
    for note in _meta_comments(plan.scene):
        pass                      # already reported above

    for f in plan.figures:
        if not isinstance(f, Figure) or match_anchor(f.name, anchors):
            continue
        # The plan wrote "Frey" for a character the REQUEST and the retrieved
        # material both call "Boss Frey". Three facts, none of them a guess about
        # what a name looks like — and reported rather than renamed, because the
        # model that read the request knows the campaign's spelling and can simply
        # write it (measured live: a prompt said "Frey:" after the request said
        # "Boss Frey", analysis §15).
        short = f.name.strip()
        if not short or " " in short or len(short) < 4:
            continue
        longer = [m.group(0) for m in re.finditer(
            r"\b([A-ZÄÖÜ][\wÄÖÜß-]+\s+" + re.escape(short) + r")\b",
            (request_text or "") + " " + (retrieved or ""))]
        if longer:
            add(Finding("name_form", short,
                        f"the plan names this figure {short!r} but the request and "
                        f"the retrieved material write {longer[0]!r} — use the "
                        "campaign's spelling"))

    listed: dict[str, list[str]] = {}
    for f in plan.figures:
        if isinstance(f, Figure) and not match_anchor(f.name, anchors):
            listed.setdefault(_norm(f.name.split()[-1]), []).append(f.name)
    for _, forms in listed.items():
        if len(forms) > 1:
            add(Finding("same_person_listed_twice", forms[0],
                        f"the plan lists {forms!r} as separate figures — if they are "
                        "one character, one figure line"))

    painted = {match_anchor(f.name, anchors).name if match_anchor(f.name, anchors)
               else f.name for f in plan.figures if isinstance(f, Figure)}
    asked = _request_names(request_text)
    for e in plan.entities:
        if not isinstance(e, str):
            continue
        # The SAME spelling, case included. "70 Gold" is capitalized because German
        # capitalizes nouns, not because the user named a character called Gold.
        hit = next((n for n in asked if n in _norm(e)
                    and _exact_form(request_text, n) in e
                    and _entry_leads_with(e, _exact_form(request_text, n))), None)
        if not hit or any(hit in _norm(f.name) for f in plan.figures
                          if isinstance(f, Figure)):
            continue
        add(Finding("entity_named_not_painted", e,
                    f"the user asked for {hit!r} and this files them under objects — "
                    "a named character needs a figure line with a pose, not a line "
                    "in the scene dressing"))
    return findings


def _entry_leads_with(entry: str, form: str) -> bool:
    """Does the plan use this name AS the entry, or only mention it inside one?

    The position is the whole test, and it needs no guess about what a word means:
    "Alisa Pennygleam" and "Boss Frey and his bandit crew" file a CHARACTER under
    objects, which is the bug this finding exists to catch. "the weathered mossy
    stone arch of the Old Bridge" files a BRIDGE — which is where a bridge belongs,
    and a live run reported it as an unpainted character, asking the correction turn
    to give a stone arch a pose.
    """
    for seg in re.split(r"[,;]|\band\b|\bmit\b|\bwith\b", entry):
        s = seg.strip()
        if s == form or s.startswith(form + " "):
            return True
    return False


def _exact_form(request_text: str, lowered: str) -> str:
    """How the request actually wrote that name ("Boss Frey", not "boss frey").

    Sliced out of the request's own text, so the composer compares spellings
    instead of deciding what a name means.
    """
    if not lowered:
        return ""
    m = re.search(r"(?<![\wÄÖÜß-])" + re.escape(lowered) + r"(?![\wÄÖÜß-])",
                  request_text or "", re.IGNORECASE)
    return request_text[m.start():m.end()] if m else ""


def _request_names(request_text: str) -> list[str]:
    """Names the USER asked for, longest form first ("boss frey" before "boss").

    The most factual signal available: the request is quoted to the model verbatim,
    so we are not interpreting prose — we are comparing two strings we were given.
    Promoting a name into the cast used to need a regex guess that a word is a
    character rather than a place or an item (§17.3); asking for a figure line is
    what replaces that guess.
    """
    out: list[str] = []
    for m in re.finditer(
            r"\b[A-ZÄÖÜ][\wÄÖÜß-]+(?:\s+[A-ZÄÖÜ][\wÄÖÜß-]+){0,2}\b",
            request_text or ""):
        parts = m.group(0).split()
        for n in range(len(parts), 0, -1):
            form = _norm(" ".join(parts[:n]))
            if form not in out:
                out.append(form)
    return out






def finding_feedback(findings: Sequence[Finding]) -> str:
    """The correction turn's user message body: what was seen, and what to do."""
    lines = [f"- {f.as_line()}" for f in findings if f.fixable]
    if not lines:
        return ""
    return ("Your plan was assembled and these things were noticed in it. Fix them "
            "in the plan and answer with the WHOLE corrected <image_plan> again — "
            "same JSON shape, same rules. Do not describe the picture in prose, do "
            "not explain your changes, and do not repeat a lookup that already "
            "happened.\n\n" + "\n".join(lines))


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
    #: What the composer noticed and did NOT fix. `images.py` gives the fixable
    #: ones back to the rewrite model once (analysis §17.3).
    findings: list["Finding"] = field(default_factory=list)
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


def _scene_units(text: str) -> list[tuple[str, str]]:
    """(clause, following separator) at TOP level — the separators kept.

    `_split` throws the punctuation away, which is fine for a Visual Anchor field
    and not fine here: a scene sentence is the model's own prose and a trim has to
    be able to put the sentence back exactly as it wrote it.
    """
    units: list[tuple[str, str]] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch in ";," and depth == 0:
            if "".join(buf).strip():
                units.append(("".join(buf).strip(), ch))
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        units.append((tail.rstrip("."), ""))
    return units


def _names_in(text: str, names: Iterable[str]) -> int:
    """How many cast names appear in this clause (word boundaries, any case)."""
    return sum(1 for n in names
               if re.search(r"(?<![\wÄÖÜäöüß])" + re.escape(n) + r"(?![\wÄÖÜäöüß])",
                            text, re.I))


def _ends_on_a_name(text: str, names: Iterable[str]) -> bool:
    """Does this clause END with a cast name (the shape an apposition opens with)?"""
    words = [w for w in text.replace(",", " ").split() if w]
    return bool(words) and any(_names_in(w, names) for w in words[-2:])


def _trim_scene(scene: str, names: Iterable[str], limit: int) -> tuple[str, list[str]]:
    """Cut a too-long scene sentence at CLAUSE boundaries, never mid-clause.

    The old cut kept the first `limit` words, and that is how a live prompt on
    2026-11-24 ended `"…while Trixy Smoldersome grips a gear, Marvin"`: the cut
    landed inside a clause, the verb went and the NAME stayed. A cast name with no
    action is the shape the renderer skips (§17/§18), so a budget trim must never
    create one. Whole clauses go instead, from the tail inwards, and two kinds of
    clause are never taken:

    - the OPENING clause, which is the setting — the figure lines repeat the cast,
      they never repeat where the shot is; measured live, the old cut deleted
      "stands at the far end of the old wooden bridge" and kept three figures'
      actions, which are already in their own lines;
    - an APPOSITION hanging off a cast name — for a character with no Visual Anchor
      that phrase is the only appearance the image has (§22: Boss Grey).

    Cutting a figure's action out of the scene costs nothing it does not already
    say: every figure in the plan gets its own line, and the ceiling ladder's clean
    8/8 and 12/12 runs had no cast name in the scene sentence at all.
    """
    units = _scene_units(scene)
    names = list(names)

    def words() -> int:
        return len(" ".join(c for c, _ in units).split())

    gone: list[str] = []
    while words() > limit and len(units) > 1:
        victim = None
        for j in range(len(units) - 1, 0, -1):
            if _ends_on_a_name(units[j - 1][0], names):
                continue            # units[j] is someone's description
            victim = j
            break
        if victim is None:
            break                   # everything left is protected
        gone.append(units.pop(victim)[0])
    if words() > limit:             # one clause, longer than the cap
        kept = units[0][0].split()
        while len(kept) > limit:
            kept.pop()
        # A sentence must not END on a bare cast name either.
        while kept and _ends_on_a_name(kept[-1], names):
            kept.pop()
        gone.append(" ".join(units[0][0].split()[len(kept):]))
        units = [(" ".join(kept), "")]
    out = " ".join(c + sep for c, sep in units).rstrip(" ,;")
    return out.rstrip(" ,;"), gone


def _dangling_tail_names(scene: str, names: Iterable[str]) -> list[str]:
    """Cast names in the scene sentence's FINAL clause and nowhere else in it.

    The live 2026-11-24 prompt ended its scene sentence `"…while Trixy Smoldersome
    grips a gear, Marvin."` — a name with no verb anywhere near it. §17 measured a
    dangling name as a strong predictor of a figure the renderer leaves out, and the
    ceiling ladder's clean runs had none. This is the model's own sentence, so it is
    REPORTED (the correction pass can rewrite it) and never rewritten here.
    """
    names = list(names)
    units = _scene_units(scene)
    if not units:
        return []
    tail = units[-1][0]
    others = " ".join(c for c, _ in units[:-1])
    out: list[str] = []
    for n in names:
        if not _names_in(tail, [n]) or _names_in(others, [n]):
            continue                      # named earlier, with what it was doing
        rest = [w.strip(",.") for w in tail.split()
                if w.strip(",.") and w.lower() != "and" and not _names_in(w, [n])]
        if not rest:                      # the final clause is ONLY this name
            out.append(n)
    return out


# A Visual Anchor label the composer does not know is kept verbatim (a sheet edit in
# Obsidian must never lose content) — EXCEPT the pipeline's own field names. A sheet
# that writes `> look: scarred snout` puts the word "look:" into an image prompt, and
# a renderer paints words it is shown; `look` is also the name the plan JSON uses for
# the same thing, so it is the label a vault author reaches for. The content stays;
# the pipeline word goes, and the swap is reported.
_PIPELINE_LABELS = frozenset({
    "look", "looks", "appearance", "description", "desc", "pose", "override",
    "modify", "source", "anchor", "fields", "figure", "figures", "scene",
})


def _strip_pipeline_label(clause: str, who: str, warnings: list[str]) -> str:
    m = re.match(r"^([A-Za-z][A-Za-z _-]{1,24}):\s*(.+)$", clause)
    if not m or _norm(m.group(1)) not in _PIPELINE_LABELS:
        return clause
    warnings.append(f"{who}: sheet field {m.group(1)!r} is the pipeline's own word, "
                    f"not a description — its text is painted without the label")
    return m.group(2).strip()


# A figure block is reported as thin when it holds under this fraction of the
# median block in the SAME image, and only once there are enough figures for a
# comparison to mean anything (two characters is not a crowd).
_THIN_LINE_MIN_FIGURES = 3
_THIN_LINE_RATIO = 0.6


def _thin_line_findings(r: "_Render") -> list[Finding]:
    """Report a figure described far less than the others in the same image."""
    texts = [_figure_text(fr, r.protect) for fr in r.figures]
    if len(texts) < _THIN_LINE_MIN_FIGURES:
        return []
    lengths = sorted(len(t) for t in texts)
    median = lengths[len(lengths) // 2]
    if median <= 0:
        return []
    return [
        Finding("figure_thin_line", fr.name,
                f"this figure gets {len(t)} characters of description while the "
                f"others in the same image average {median}; in a crowded image the "
                "thinnest figure is the one the renderer leaves out — give it the "
                "same weight, saying only what the notes you opened actually say")
        for fr, t in zip(r.figures, texts)
        if len(t) < _THIN_LINE_RATIO * median
    ]


def _figure_text(fr: _FigureRender, protect: set[str]) -> str:
    """One figure's block: what it IS, then what it is DOING, then the rest.

    The order is measured, not styled (analysis §18). The same eight characters,
    the same full Visual Anchors, the same words in a prompt of exactly the same
    length — only the identity clause's position moved — and every render that put
    the pose first lost a character (5 of 5), while every render that put the
    identity first painted all of them (4 of 4, p = 0.008). A diffusion model
    decides WHAT to draw before it decides what the figure is busy with; a block
    that opens with "staff planted, crescent finial flaring light" is an
    instruction about a staff, and the owl it belongs to goes unpainted.

    The budget ladder is untouched by this: it cuts clauses out of `fields`, and
    this function only decides the order the survivors are said in.
    """
    identity_field = next((f for f in _IDENTITY_FIELDS if fr.fields.get(f)), "")
    clauses: list[str] = list(fr.fields.get(identity_field, []))
    if fr.pose:
        clauses.append(fr.pose)
    for f in (*ANCHOR_FIELDS, *(k for k in fr.fields if k not in ANCHOR_FIELDS)):
        if f != identity_field:
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
# A name the scene only ever mentions after one of these is a place, not an actor.


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
    world_rule: str = "",
    max_clauses_per_figure: int = 0,
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
    # What the composer notices about the plan. Reported, never edited: the model
    # that wrote the plan gets these back as one correction turn (see images.py),
    # which is the layer that can tell "no helmet" from "helmet, no crest".
    findings = lint_plan(plan, anchors, retrieved=retrieved,
                         read_paths=read_paths, snippet_paths=snippet_paths,
                         request_text=request_text)
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
            findings.append(Finding("look_unverified", f.name, why))
        if ok:
            keep.append((f, a))
            continue
        ungrounded.append(f.name)
        # Never mutate the caller's plan: an unverifiable look is dropped on a
        # copy, so the plan the log shows is the plan the model wrote.
        keep.append((replace(f, look=""), None))
    resolved = keep

    # A figure LINE is identity, not prose: "Seta" and "Seta Onix" have to be the
    # same figure, so the sheet's spelling wins there and dedupe stays honest. What
    # the model wrote in its SCENE is left alone — `_spelling_gaps` reports it and
    # the correction turn fixes it. (Measured live, §15: figure lines said "Romi
    # Stachelpfote" while the same prompt's sentence said "Romi".)
    relabelled = {f.name: a.name for f, a in resolved
                  if a is not None and a.name != f.name}
    if relabelled:
        resolved = [(replace(f, name=a.name), a) if a is not None else (f, a)
                    for f, a in resolved]
        warnings.append("plan wrote " + ", ".join(
            f"{o!r} for {n!r}" for o, n in sorted(relabelled.items()))
            + " — the figure lines use the sheet's spelling")

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
    scene_words = len(scene.split())
    if scene_words > _MAX_SCENE_WORDS:
        scene, gone = _trim_scene(scene, [f.name for f, _ in resolved],
                                  _MAX_SCENE_WORDS)
        warnings.append(f"scene sentence ran to {scene_words} words — trimmed to the "
                        f"contract's {_MAX_SCENE_WORDS} by dropping the tail clauses "
                        f"{', '.join(repr(g) for g in gone)}; the anchor lines are what "
                        "the model paints the characters from")
    # The scene sentence is the model's sentence. Everything the composer notices
    # about it — a crowd noun naming the cast, a name enumerated twice — is in
    # `findings`, and the correction turn is where it gets fixed. The one thing
    # still enforced here is the contract's own word cap above, because that is a
    # budget and not an interpretation.

    count = len(resolved)
    front_n = min(foreground, count)
    plural = "s" if count != 1 else ""
    count_line = f"Exactly {count} named figure{plural} in this image"
    if count > front_n:
        count_line += f", {front_n} in the foreground and {count - front_n} behind them"
    style = base_style.rstrip(". ")
    # The campaign's world line — what kind of beings exist in this world — is one
    # sentence of CAMPAIGN DATA (`world_rule:` in the Campaign Prompt Profile), said
    # verbatim, and it rides in the style sentence where the budget ladder never
    # cuts it. The composer holds no wording of its own and asks nothing about who
    # is an animal, a human or a machine: a Humblewood profile says everyone is
    # animal-folk, a human-world profile says something else or nothing, and the
    # code is the same either way (retired species logic, analysis §20).
    if world_rule:
        style = f"{style}. {world_rule.rstrip('. ')}"
    scene_line = (f"{style}. {scene}. {count_line}." if scene
                  else f"{style}. {count_line}.")

    entity_line = ""
    entity_entries: list[str] = []
    if plan.entities:
        ents = [e.strip().rstrip(". ") for e in plan.entities]
        entity_line = f"Also in the scene: {', '.join(ents)}."
        entity_entries = ents

    renders: list[_FigureRender] = []
    for i, (f, a) in enumerate(resolved):
        pose = f.pose.strip().rstrip(". ")
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
            extra = [_strip_pipeline_label(x, f.name, warnings) for x in a.extra]
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

    # `max_clauses_per_figure` (0 = off) is EQUAL TREATMENT, applied before the
    # budget ladder and independently of it: while a figure is holding more clauses
    # than the cap, its LEAST DISTINCTIVE clause comes out, even when the total
    # budget is already met. Measured against the clean ceiling-ladder runs (§17),
    # which gave every character the same treatment and painted 8/8 at 587 tokens
    # and 12/12 at 810 (~50-70 tokens per figure), against the two live 2026-11-24
    # prompts that ran 93-124 tokens per figure and both dropped figures (§22).
    #
    # It is a CEILING on description, never on characters: the cap takes clauses out
    # of the fattest blocks and has nothing to say about who is in the image.
    cap = max(0, max_clauses_per_figure)
    cap_notes: list[str] = []
    if cap:
        while True:
            over = [(fi, fr) for fi, fr in enumerate(r.figures)
                    if sum(len(c) for c in fr.fields.values()) > cap]
            if not over:
                break
            best: tuple[int, int, str] | None = None
            for fi, fr in over:
                background = fi >= front_n
                for field, clauses in fr.fields.items():
                    # The same rule as the budget ladder, for the same reason (§18):
                    # the identity clause is what makes a figure exist in the render.
                    # A cap that cut it would fix a spread by deleting a character.
                    if field in _IDENTITY_FIELDS:
                        continue
                    for ci, clause in enumerate(clauses):
                        sc = score(field, clause, background)
                        if best is None or sc < best[0]:
                            best = (sc, fi, field)
            if best is None:
                break
            _, fi, field = best
            fr = r.figures[fi]
            gone = fr.fields[field].pop(0)
            if not fr.fields[field]:
                del fr.fields[field]
            cap_notes.append(f"{fr.name} [{field}] {gone!r}")
            prompt = _render_text(r)
        if cap_notes:
            cut(f"clause cap {cap}: {len(cap_notes)} clause(s) taken out of the most "
                f"heavily described figures before the budget was asked — every "
                f"character is described, none twice as hard — "
                f"{', '.join(cap_notes)}")

    dropped_notes: list[str] = []
    # Pass 1 may not touch an identity clause; pass 2 exists so the ladder can
    # still finish instead of hard-truncating a sentence mid-name.
    #
    # `max_clauses_per_figure` (0 = off) adds a FAIRNESS pass before the global
    # one: while the budget is over and some figure is holding more clauses than
    # the cap, the least distinctive clause comes out of THAT figure. Measured
    # against the ceiling ladder, which gave every character the same treatment
    # and painted 8/8 at 587 tokens and 12/12 at 810, against the shipped
    # prompts' 17-clause block next to a 2-clause one (§22).
    cap = max(0, max_clauses_per_figure)
    for allow_identity_cut in (False, True):
        for over_cap_only in ((True, False) if cap else (False,)):
            while len(prompt) > budget:
                best: tuple[int, int, str, int] | None = None   # score, fig, field, idx
                for fi, fr in enumerate(r.figures):
                    if over_cap_only and sum(len(c) for c in fr.fields.values()) <= cap:
                        continue
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

    # ── Unequal treatment, measured on the shape that actually gets painted.
    # The 2026-11-25 sample batch (analysis §22.1): in the two seven-figure scenes
    # the ONE figure the renderer left out was the block far thinner than its
    # neighbours — the unanchored NPC at 43 tokens against a 99-token median, painted
    # LAST, in both the shipped arm and the clause-cap arm. The fix is not to shave
    # the others (that arm lost a party member too); it is to tell the rewrite model
    # that this line is under-weighted and let it say more, from what it read.
    findings.extend(_thin_line_findings(r))

    if shared:
        log.debug("shared palette tokens in this scene: %s", ", ".join(sorted(shared)))
    cuts = [w for w in warnings
            if w.startswith(("budget ", "clause cap "))]
    return Composed(prompt=prompt, figures=count, foreground=front_n,
                    cuts=cuts, warnings=warnings, findings=findings,
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
