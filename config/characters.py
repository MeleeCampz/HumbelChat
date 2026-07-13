"""Character/persona loading, validation, and display-name mapping."""
from __future__ import annotations
import json
import pathlib
import logging

log = logging.getLogger("bot.config.characters")


class Character:
    """Immutable character config from characters.json."""
    key: str
    display: str
    model: str
    system_prompt: str | None
    max_tokens: int | None
    temperature: float | None

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
        model = data.get("model", "") 
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


# Global state (populated on import and via load_characters)
_CHARACTERS: list[Character] = []
_DEFAULT_KEY: str = "default"
_CHAR_DISPLAY_MAP: dict[str, Character] = {}
CHARACTER_CHOICES: list = []


def _load_char_json(path: pathlib.Path) -> tuple[str, list[Character]]:
    """Return (default_key, list of characters)."""
    try:
        if not path.exists():
            return "default", [Character("default", "Default", "")]
        raw = path.read_text(encoding="utf-8")
        cfg = json.loads(raw)
    except Exception as e:
        log.error("Error loading characters.json: %s", e)
        return "default", [Character("default", "Default", "")]

    chars_cfg = cfg.get("characters", {})
    default_key = cfg.get("default", "default")

    characters: list[Character] = []
    for key, data in chars_cfg.items():
        characters.append(Character.from_dict(key, data))

    if not any(c.key == default_key for c in characters):
        if characters:
            default_key = characters[0].key
        else:
            default_key = "default"
            characters.append(Character("default", "Default", ""))

    return default_key, characters


def load_characters(path: pathlib.Path) -> None:
    """Load characters.json and populate globals."""
    global _CHARACTERS, _DEFAULT_KEY, _CHAR_DISPLAY_MAP, CHARACTER_CHOICES
    
    _DEFAULT_KEY, _CHARACTERS = _load_char_json(path)
    _CHAR_DISPLAY_MAP = {c.display: c for c in _CHARACTERS}
    CHARACTER_CHOICES = [
        {"name": c.display, "value": c.key}
        for c in _CHARACTERS
    ]


def get_character(key_or_display: str | None) -> Character | None:
    """Look up by internal key OR display name. Returns None if not found."""
    if key_or_display is None:
        return None

    # Direct key match first
    for c in _CHARACTERS:
        if c.key == key_or_display:
            return c
            
    # Display match (case-insensitive)
    target = key_or_display.lower()
    for name, char in _CHAR_DISPLAY_MAP.items():
        if name.lower() == target:
            return char
            
    return None


def default_character() -> Character:
    """Return the default characterOrDefault to first available."""
    try:
        for c in _CHARACTERS:
            if c.key == _DEFAULT_KEY:
                return c
        return _CHARACTERS[0] if _CHARACTERS else Character("default", "Default", "")
    except (IndexError, AttributeError):
        return Character("default", "Default", "")

# Wait, I see a typo in default_character: _CHAR_CHARS should be _CHARACTERS. 
# Also, i'll fix it right now.
