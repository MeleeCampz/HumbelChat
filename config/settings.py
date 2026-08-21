"""Application settings — all environment variables with defaults, typed."""
from __future__ import annotations

import os
import pathlib

# ═══ Helper functions ════════════════════════════════════════════════════

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
DISCORD_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")

# ════════════════════════════════════
#  AI PROVIDER (any OpenAI-compatible)
# ════════════════════════════════════
INFER_URL: str = os.getenv("INFER_URL", "http://127.0.0.1:11434/v1")
INFER_API_KEY: str = os.getenv("INFER_API_KEY", "")  # sometimes empty for local

# ════════════════════════════════════
#  CHARACTER defaults (per-char in characters.json)
# ════════════════════════════════════
DEFAULT_MODEL: str | None = os.getenv("MODEL_NAME")
DEFAULT_SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT", "")
CONTEXT_WINDOW: int = _safe_int(os.getenv("CONTEXT_WINDOW"), 10)
REQUEST_TIMEOUT: int = _safe_int(os.getenv("AI_REQUEST_TIMEOUT"), 120)
MAX_TOKENS: int = _safe_int(os.getenv("MAX_TOKENS"), 2000)
MAX_TOKENS_HARD_CAP: int = _safe_int(os.getenv("MAX_TOKENS_HARD_CAP"), 4096)

# Fallback models tried (in order) when DEFAULT_MODEL fails during summarize/translate.
# Comma-separated list of model slugs.
FALLBACK_MODELS: list[str] = [
    m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()
]

# ════════════════════════════════════
#  BOT BEHAVIOUR
# ════════════════════════════════════
BOT_PREFIX: str = os.getenv("BOT_PREFIX", "!ai")
CHAT_HISTORY_RESET: str | None = _or_clear(os.getenv("CHAT_HISTORY_RESET"))

def _or_default(value: str | None, default: str) -> str:
    """Return value if non-empty, else default (for optional path overrides)."""
    return value if value else default


# ════════════════════════════════════
#  KNOWLEDGE BASE
# ════════════════════════════════════
# Repo root, resolved from this file so all paths work regardless of CWD.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

KB_PATH: pathlib.Path = pathlib.Path(
    _or_default(os.getenv("KB_PATH"), str(_REPO_ROOT / "data" / "knowledge"))
)
CHARACTERS_FILE: pathlib.Path = pathlib.Path(
    _or_default(os.getenv("CHARACTERS_FILE"), str(_REPO_ROOT / "characters.json"))
)

DEFAULT_KB_NAME: str = os.getenv("KB_DEFAULT_KB", "humblewood").lower()
CHUNK_TARGET: int = _safe_int(os.getenv("CHUNK_SIZE"), 2000)
RAG_MAX_DOCS: int = _safe_int(os.getenv("RAG_MAX_DOCS"), 4)
RAG_RETRIEVAL_METHOD: str = os.getenv("RAG_RETRIEVAL_METHOD", "vector").lower()
RAG_MAX_CHARS: int = _safe_int(os.getenv("RAG_MAX_CHARS"), 24000)
RAG_WINDOW_LINES: int = _safe_int(os.getenv("RAG_WINDOW_LINES"), 80)

# ════════════════════════════════════
#  LEGACY COMPAT (read but don't use in new code)
# ════════════════════════════════════
_OPENWEBUI_KEY: str = os.getenv("OPENWEBUI_API_KEY", "")
_KB_KNOWLEDGE_BASE: str = os.getenv("KB_KNOWLEDGE_BASE", "HumbleWood")
