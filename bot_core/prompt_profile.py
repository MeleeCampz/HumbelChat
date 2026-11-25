"""The campaign's look, as data instead of Python (2026-11-19 architecture pass).

Why this module exists
----------------------
The prompt composer used to carry the campaign inside it: the style sentence, the
animal-folk floor line, the negative tail, the path of the party sheet and the
species vocabulary were constants in `bot_core/image_prompt.py` / `bot_core/images.py`.
That meant a second campaign setting needed a code change, and it meant the table's
own words were being edited by the wrong layer. The species vocabulary is gone from
the code outright (2026-11-20, analysis §20) — what a world's beings ARE is campaign
data, and it reaches the prompt as the one sentence below.

Everything here is the table's wording, read from a file the table can edit:

    <KB_PATH>/Campaign Prompt Profile.md        (or IMAGE_PROMPT_PROFILE=<path>)

Typed lines, same shape as a Visual Anchor:

    > style: Semi-realistic fantasy illustration — detailed fur and feather texture…
    > world_rule: Every figure in this image is an anthropomorphic animal-folk…
    > negative: humans, watermark, signature, modern objects, extra limbs
    > party_sheet: humblewood/HumbleWood/Characters/Player Characters.md
    > budget: base=1100 per_figure=420 max=3600 foreground=4

Precedence: an environment variable that is actually set wins over the file, the
file wins over the neutral default. A file with no `world_rule` line simply gets no
such sentence — the code does not hold a fallback sentence for a campaign it does not
know, and it holds no opinion about who is an animal, a human or a machine: a
Humblewood profile and a human-world profile are the same code with different text.

`world_rule` is said verbatim, once, in the style sentence, where the budget ladder
never cuts it. Nothing reads it or reasons about it.

`load()` re-reads the file on every call, for the same reason `_load_anchors()`
re-reads the party sheet: an edit made in Obsidian applies to the next image, not
to the next bot restart.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import settings

log = logging.getLogger(__name__)

#: Field names the profile understands. An unknown one is reported, never ignored.
KNOWN_FIELDS = ("style", "world_rule", "negative", "party_sheet", "budget")

#: The names a profile written before 2026-11-20 uses. `animal_floor` is the same
#: single sentence under its old name; `animal_floor_living` was the machine-aware
#: variant of it, which no longer exists — the two lines are read as one, the
#: `animal_floor` wording winning when both are present, so an old file keeps
#: working untouched instead of silently losing its sentence.
_LEGACY_FIELDS = {"animal_floor": "world_rule", "animal_floor_living": "world_rule"}

_PROFILE_NAME = "Campaign Prompt Profile.md"

#: Structural defaults. Deliberately not campaign prose: no style sentence, no
#: world rule sentence, no party sheet. A campaign that wants them writes them.
_DEFAULTS: dict[str, Any] = {
    "style": "",
    "world_rule": "",
    "party_sheet": "",
    "base_chars": settings.IMAGE_PROMPT_BASE_CHARS,
    "chars_per_figure": settings.IMAGE_PROMPT_CHARS_PER_FIGURE,
    "max_chars": settings.IMAGE_PROMPT_MAX_CHARS,
    "foreground_figures": settings.IMAGE_PROMPT_FOREGROUND_FIGURES,
    "clauses_per_figure": settings.IMAGE_PROMPT_MAX_CLAUSES_PER_FIGURE,
}

# Only the settings that genuinely have two homes are listed; the negative tail is
# already an env-backed setting and keeps working unchanged.
_ENV_FOR_FIELD = {
    "style": "IMAGE_PROMPT_STYLE",
    "world_rule": "IMAGE_WORLD_RULE",
    "negative": "IMAGE_NEGATIVE_PROMPT",
    "party_sheet": "IMAGE_PARTY_SHEET",
    "base_chars": "IMAGE_PROMPT_BASE_CHARS",
    "chars_per_figure": "IMAGE_PROMPT_CHARS_PER_FIGURE",
    "max_chars": "IMAGE_PROMPT_MAX_CHARS",
    "foreground_figures": "IMAGE_PROMPT_FOREGROUND_FIGURES",
    "clauses_per_figure": "IMAGE_PROMPT_MAX_CLAUSES_PER_FIGURE",
}

_BUDGET_KEYS = {"base": "base_chars", "per_figure": "chars_per_figure",
                "max": "max_chars", "foreground": "foreground_figures",
                "clauses": "clauses_per_figure"}

#: Same story for the environment: a deployment that set the old name keeps working.
_LEGACY_ENV = {"world_rule": "IMAGE_ANIMAL_FLOOR"}


@dataclass(frozen=True)
class Profile:
    """The strings and numbers the composer is allowed to take from outside code."""

    style: str
    world_rule: str
    negative_prompt: str
    party_sheet: str          # path relative to KB_PATH, "" = no anchors
    base_chars: int
    chars_per_figure: int
    max_chars: int
    foreground_figures: int
    source: str               # the file it came from, or "<defaults>"
    clauses_per_figure: int = 0   # 0 = off; the budget ladder's fairness cap
    warnings: tuple[str, ...] = ()

    def sheet_path(self) -> Path | None:
        if not self.party_sheet:
            return None
        p = Path(self.party_sheet)
        return p if p.is_absolute() else Path(settings.KB_PATH) / p


def _profile_files() -> list[Path]:
    """Where a profile may live, most specific first."""
    explicit = os.getenv("IMAGE_PROMPT_PROFILE", "").strip()
    if explicit:
        return [Path(explicit)]
    kb = Path(settings.KB_PATH)
    found = [kb / _PROFILE_NAME]
    found += sorted(kb.glob(f"*/{_PROFILE_NAME}"))
    return found


def _read_file(path: Path) -> dict[str, str]:
    """Typed `>` lines (and plain `key: value` lines) from the profile file."""
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            line = line[1:].strip()
        m = re.match(r"([A-Za-z_][A-Za-z_0-9]*)\s*:\s*(.*)$", line)
        if not m:
            continue
        key, value = m.group(1).strip().lower(), m.group(2).strip()
        key = _LEGACY_FIELDS.get(key, key)
        if key in KNOWN_FIELDS and value and key not in out:
            out[key] = value
    return out


def _budget(text: str, current: dict[str, int]) -> tuple[dict[str, int], str | None]:
    """`base=1100 per_figure=420 max=3600 foreground=4` -> the four numbers."""
    out = dict(current)
    warn = ""
    for part in re.split(r"[,;\s]+", text):
        m = re.match(r"\s*([a-z_]+)\s*=\s*(\d+)\s*$", part)
        if not m:
            warn = f"unparsed budget part: {part.strip()!r}"
            continue
        key = _BUDGET_KEYS.get(m.group(1))
        if key is None:
            warn = f"unknown budget name: {m.group(1)!r} " \
                   f"(known: {', '.join(_BUDGET_KEYS)})"
            continue
        out[key] = int(m.group(2))
    return out, warn


def load() -> Profile:
    """Read the campaign profile. Never raises: a missing file means defaults."""
    warnings: list[str] = []
    fields: dict[str, str] = {}
    source = "<defaults>"
    for path in _profile_files():
        if not path.is_file():
            continue
        fields = _read_file(path)
        source = str(path)
        break

    if source == "<defaults>":
        # Silent defaults are how a campaign loses its own wording: the code holds no
        # fallback prose on purpose, so a profile that went missing (it lives under
        # `data/`, which is gitignored) means every prompt simply stops opening with
        # the campaign's sentence — and nothing in the log said so. Say it once per load.
        warnings.append(
            "no campaign profile file found — looked in: "
            + ", ".join(str(p) for p in _profile_files())
            + " (or IMAGE_PROMPT_PROFILE). The composer holds no fallback wording, so "
              "this deployment paints with no style sentence, no world-rule sentence "
              "and no party sheet — docs/campaign-profile.md")

    values: dict[str, Any] = dict(_DEFAULTS)
    for key in ("style", "world_rule", "negative", "party_sheet"):
        if key in fields:
            values[key if key != "negative" else "negative_prompt"] = fields[key]

    if "budget" in fields:
        merged, warn = _budget(fields["budget"], values)
        values.update(merged)
        if warn:
            warnings.append(f"profile budget: {warn}")

    for key, env_name in _ENV_FOR_FIELD.items():
        raw = os.getenv(env_name)
        if raw is None:
            legacy = _LEGACY_ENV.get(key)
            raw = os.getenv(legacy) if legacy else None
            continue
        target = "negative_prompt" if key == "negative" else key
        if target in ("base_chars", "chars_per_figure", "max_chars",
                      "foreground_figures", "clauses_per_figure"):
            try:
                values[target] = int(raw)
            except ValueError:
                warnings.append(f"{env_name} is not a number: {raw!r}")
        else:
            values[target] = raw.strip()

    unknown = [k for k in fields if k not in KNOWN_FIELDS]
    if unknown:
        warnings.append(f"profile fields ignored (unknown): {', '.join(sorted(unknown))}")

    negative = values.get("negative_prompt") or settings.IMAGE_NEGATIVE_PROMPT

    return Profile(
        style=values["style"],
        world_rule=values["world_rule"],
        negative_prompt=negative,
        party_sheet=values["party_sheet"],
        base_chars=int(values["base_chars"]),
        chars_per_figure=int(values["chars_per_figure"]),
        max_chars=int(values["max_chars"]),
        foreground_figures=int(values["foreground_figures"]),
        clauses_per_figure=int(values["clauses_per_figure"]),
        source=source,
        warnings=tuple(warnings),
    )


def party_sheet_path(profile: Profile | None = None) -> Path | None:
    return (profile or load()).sheet_path()
