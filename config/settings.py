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


def _history_reset_flag(value: str | None) -> bool:
    """Return True if *value* is the sentinel string "clear" (case-insensitive).

    Used to read CHAT_HISTORY_RESET, which is a boolean-ish env var:
    "clear" (or "1"/"true") means "clear history on startup", anything
    else means "keep it". Previously this was a ``str | None`` helper whose
    only meaningful return value was the string ``"clear"`` — confusing and
    over-typed (see code review §3.6).
    """
    if not value:
        return False
    return value.strip().lower() in ("clear", "1", "true", "yes")


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
# §3.9: lightweight backend liveness probe. 0 = only probe once on startup;
# a positive value starts a periodic background check at that interval.
AI_HEALTH_CHECK_INTERVAL: int = _safe_int(os.getenv("AI_HEALTH_CHECK_INTERVAL"), 0)
AI_HEALTH_CHECK_TIMEOUT: int = _safe_int(os.getenv("AI_HEALTH_CHECK_TIMEOUT"), 5)
MAX_TOKENS: int = _safe_int(os.getenv("MAX_TOKENS"), 2000)
MAX_TOKENS_HARD_CAP: int = _safe_int(os.getenv("MAX_TOKENS_HARD_CAP"), 4096)

# Fallback models tried (in order) when DEFAULT_MODEL fails during summarize/translate.
# Comma-separated list of model slugs.
FALLBACK_MODELS: list[str] = [
    m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()
]

# ── Session overview prompt (customizable) ───────────────────────────────
#: Default system prompt used by /end_session to write the session overview.
DEFAULT_SESSION_SUMMARY_PROMPT: str = (
    "You write concise session overviews. Given the notes and recent chat "
    "of a work session, produce a short overview (max ~250 words) with: "
    "what was done, key points/decisions, and open items or follow-ups for "
    "the next session. Use markdown bullet points. Return ONLY the overview."
)

#: Override for SESSION_SUMMARY_PROMPT — set in .env to customize how the
#: /end_session AI overview is written. Empty = use DEFAULT_SESSION_SUMMARY_PROMPT.
SESSION_SUMMARY_PROMPT: str = os.getenv("SESSION_SUMMARY_PROMPT", "")

# ════════════════════════════════════
#  BOT BEHAVIOUR
# ════════════════════════════════════
BOT_PREFIX: str = os.getenv("BOT_PREFIX", "!ai")
# §3.6: boolean flag instead of the old ``"clear" | None`` sentinel string.
CHAT_HISTORY_RESET: bool = _history_reset_flag(os.getenv("CHAT_HISTORY_RESET"))

# Beyond20-style embed rendering for /ai replies (non-streaming path only).
# Structured replies (headings, tables, lists) become a discord.Embed with a
# title, description and inline fields — the way the Beyond20 bot formats
# rolls. Plain prose still works; tiny/empty replies fall back to text.
# Set EMBED_FORMAT=0 in .env to restore classic plain-text delivery.
EMBED_FORMAT: bool = os.getenv("EMBED_FORMAT", "1") not in ("0", "false", "no")

def _or_default(value: str | None, default: str) -> str:
    """Return value if non-empty, else default (for optional path overrides)."""
    return value if value else default


# ════════════════════════════════════
#  EMBEDDING MODEL
# ════════════════════════════════════
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

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

CHUNK_TARGET: int = _safe_int(os.getenv("CHUNK_SIZE"), 2000)
RAG_MAX_DOCS: int = _safe_int(os.getenv("RAG_MAX_DOCS"), 4)
RAG_RETRIEVAL_METHOD: str = os.getenv("RAG_RETRIEVAL_METHOD", "vector").lower()
RAG_MAX_CHARS: int = _safe_int(os.getenv("RAG_MAX_CHARS"), 24000)
RAG_WINDOW_LINES: int = _safe_int(os.getenv("RAG_WINDOW_LINES"), 80)

# ════════════════════════════════════
#  INPUT VALIDATION
# ════════════════════════════════════
# Hard cap on user prompt length (chars).  Prevents a user from pasting
# 100 K+ chars which would blow past any context window once RAG is added.
MAX_INPUT_CHARS: int = _safe_int(os.getenv("MAX_INPUT_CHARS"), 50_000)

# ════════════════════════════════════
#  RATE LIMITING
# ════════════════════════════════════
# Max AI requests per user in a sliding window.
AI_RATE_LIMIT_MAX: int = _safe_int(os.getenv("AI_RATE_LIMIT_MAX"), 5)
AI_RATE_LIMIT_WINDOW: int = _safe_int(os.getenv("AI_RATE_LIMIT_WINDOW"), 60)


