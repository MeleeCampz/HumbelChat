"""Character/persona loading, validation, and display-name mapping."""
from __future__ import annotations
import json
import pathlib
import logging

log = logging.getLogger("bot.config.characters")


class Character:
    """Immutable character config from characters.json."""
    key: str

    def __init__(
        self,
        key:str,
        display:str,
        model:str,
        system_prompt:str | None = None,
        max_tokens:int | None = None,
        temperature:float | None = None,
    ):
        self.key = key
        self.display = display
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def from_dict(key: str, data: dict) -> Character:
        display = data.get("display", data.get("name", key.title()))
        model = data.get("model", "")  # required — no fallback to .env
        return Character(
            key=key,
            display=display,
            model=model,
            system_prompt=data.get("system_prompt"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
        )

    def __repr__(self):
        return f"Character({self.display!r} → {self.model!r})"


# Default empty choices — populated by load_characters() so __init__.py doesn'            not crash
CHARACTER_CHOICES: list = []


def _load_char_json(path: pathlib.Path) -> tuple[str, list[Character]]:
    """Return (default_key, list of characters)."""
    try:
        raw = path.read_text(encoding="utf-8")
        cfg = json.loads(raw)
    except FileNotFoundError:
        log.warning("characters.json not found — single default character only.")
        return "default", [Character.from_dict("default", {"model": "", "display": "Default"})]
    except json.JSONDecodeError as exc:
        log.error("characters.json JSON parse error: %s", exc)
        return "default", [Character.from_dict("default", {"model": "", "display": "Default"})]

    chars_cfg = cfg.get("characters", {})
    default_key = cfg.get("default", list(chars_cfg)[0] if chars_cfg else "default")

    characters: list[Character] = []
    for key, data in chars_cfg.items():
        characters.append(Character.from_dict(key, data))

    # Safety — ensure default exists
    if not any(c.key == default_key for c in characters):
        log.warning("Default character '%s' not found in characters.json — falling back to first.", default_key)
        default_key = characters[0].key if characters else "default"

    return default_key, characters


# ── Global state (populated on import) ───────

_CHARACTERS: list[Character] = []  # all loaded characters
_DEFAULT_KEY: str = "default"     # internal key of default character
_CHAR_DISPLAY_MAP: dict[str, str] = {}  # display → char (for fast lookup by dropdown name)


def load_characters(path: pathlib.Path = pathlib.Path("characters.json")) -> None:
    """Load characters.json and populate globals."""
    global _CHARACTERS, _DEFAULT_KEY, _CHAR_DISPLAY_MAP, CHARACTER_CHOICES
    _DEFAULT_KEY, _CHARACTERS = _load_char_json(path)
    _CHAR_DISPLAY_MAP = {c.display: c for c in _CHARACTERS}

    # Store raw dicts — converted to Discord Choice objects in main.py at bot startup
    CHARACTER_COMPONENTS = [
        {"name": c.display, "value": c.key}
        for c in _CHARACTERS
    ]
    CHARACTER_CHOICES = CHARACTER_COMPONENTS


def get_character(key_or_display: str) -> Character | None:
    """Look up by internal key OR display name. Returns None if not found."""
    # Direct key match first
    if any(c.key == key_or_display for c in _CHARACTERS):
        return next(c for c in _CHARACTERS if c.key == key_or_display)
    # Display match (case-imsensitive)
    for name, char in _CHAR_DISPLAY_MAP.items():
        if name.lower() == key_or_display.lower():
            return char
    return None


def default_character() -> Character:
    try:
        return next((c for c in _CHARACTERS if c.key == _DEFAULT_KEY), _CHARACTERS[0])
    except (IndexError, AttributeError):
        # Fallback to a dummy if everything failed
        return Character.from_dict("default", {"model": ""})
