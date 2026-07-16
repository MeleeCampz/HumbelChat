"""Application settings — all environment variables with defaults, typed."""
from __future__ import annotations
import os
import pathlib

# ═══ Helper functions (must be before their usage) ══════════════════════

def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value else default
    except ValueError:
        return default


def _or_clear(value: str | None):
    """Return "clear" once if value equals that string, then None."""
    if not value:
        return None
    val = value.strip().lower()
    return "clear" if val == "clear" else None


# ════════════════════════════════════
#  DISCORD
# ════════════════════════════════════
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

# ════════════════════════════════════
#  AI PROVIDER (any OpenAI-compatible)
# ════════════════════════════════════
INFER_URL     = os.getenv("INFER_URL", "http://127.0.0.1:11434/v1")
INFER_API_KEY = os.getenv("INFER_API_KEY", "")  # sometimes empty for local

# ════════════════════════════════════
#  CHARACTER defaults (per-char in characters.json)
# ════════════════════════════════════
DEFAULT_MODEL       = ""          # chars in JSON override these
DEFAULT_SYSTEM_PROMPT = ""
CONTEXT_WINDOW      = _safe_int(os.getenv("CONTEXT_WINDOW"), 10)
REQUEST_TIMEOUT     = _safe_int(os.getenv("AI_REQUEST_TIMEOUT"), 120)
MAX_TOKENS          = _safe_int(os.getenv("MAX_TOKENS"), 2000)

# ════════════════════════════════════
#  BOT BEHAVIOUR
# ════════════════════════════════════
BOT_PREFIX          = os.getenv("BOT_PREFIX", "!ai")
CHAT_HISTORY_RESET: str | None = _or_clear(os.getenv("CHAT_HISTORY_RESET"))

# ════════════════════════════════════
#  KNOWLEDGE BASE
# ════════════════════════════════════
KB_PATH             = pathlib.Path(os.getenv("KB_PATH", "data/knowledge"))
DEFAULT_KB_NAME     = os.getenv("KB_DEFAULT_KB", "humblewood").lower()
CHUNK_TARGET        = _safe_int(os.getenv("CHUNK_SIZE"), 2000)

# ════════════════════════════════════
#  LEGACY COMPAT (read but don't use in new code)
# ════════════════════════════════════
_OPENWEBUI_KEY      = os.getenv("OPENWEBUI_API_KEY", "")
_KB_KNOWLEDGE_BASE  = os.getenv("KB_KNOWLEDGE_BASE", "HumbleWood")


# ════════════════════════════════════
#  Singleton — populated at import time
# ════════════════════════════════════

class _Settings:
    """Frozen-style singleton — attributes are set-once after import."""
    DISCORD_TOKEN: str
    INFER_URL: str
    INFER_API_KEY: str
    DEFAULT_MODEL: str
    DEFAULT_SYSTEM_PROMPT: str
    CONTEXT_WINDOW: int
    REQUEST_TIMEOUT: int
    MAX_TOKENS: int
    BOT_PREFIX: str
    CHAT_HISTORY_RESET: str | None
    KB_PATH: pathlib.Path
    DEFAULT_KB_NAME: str
    CHUNK_TARGET: int
    OPENWEBUI_API_KEY: str  # legacy — kept for reference

    def __repr__(self):
        keys = sorted(vars(self))
        return f"Settings({', '.join(f'{k}={getattr(self, k)!r}' for k in keys)})"


settings = _Settings()

# Populate singleton
_INIT_ATTRS = (
    "DISCORD_TOKEN", "INFER_URL", "INFER_API_KEY", "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT", "CONTEXT_WINDOW", "REQUEST_TIMEOUT",
    "MAX_TOKENS", "BOT_PREFIX", "CHAT_HISTORY_RESET", "KB_PATH", "DEFAULT_KB_NAME",
    "CHUNK_TARGET", "OPENWEBUI_API_KEY",
)

_INIT_VALUES = {
    "DISCORD_TOKEN": DISCORD_TOKEN,
    "INFER_URL": INFER_URL,
    "INFER_API_KEY": INFER_API_KEY,
    "DEFAULT_MODEL": DEFAULT_MODEL,
    "DEFAULT_SYSTEM_PROMPT": DEFAULT_SYSTEM_PROMPT,
    "CONTEXT_WINDOW": CONTEXT_WINDOW,
    "REQUEST_TIMEOUT": REQUEST_TIMEOUT,
    "MAX_TOKENS": MAX_TOKENS,
    "BOT_PREFIX": BOT_PREFIX,
    "CHAT_HISTORY_RESET": CHAT_HISTORY_RESET,
    "KB_PATH": KB_PATH,
    "DEFAULT_KB_NAME": DEFAULT_KB_NAME,
    "CHUNK_TARGET": CHUNK_TARGET,
    "OPENWEBUI_API_KEY": _OPENWEBUI_KEY,
}

for _attr in sorted(_INIT_ATTRS):
    setattr(settings, _attr, _INIT_VALUES[_attr])

del (DISCORD_TOKEN, INFER_URL, INFER_API_KEY, DEFAULT_MODEL)  # noqa: F821

