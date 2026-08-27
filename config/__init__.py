"""Configuration package."""
from .settings import (
    BOT_PREFIX,
    CHAT_HISTORY_RESET,
    CHUNK_TARGET,
    CONTEXT_WINDOW,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DISCORD_TOKEN,
    FALLBACK_MODELS,
    INFER_API_KEY,
    INFER_URL,
    KB_PATH,
    MAX_TOKENS,
    RAG_MAX_CHARS,
    RAG_MAX_DOCS,
    RAG_RETRIEVAL_METHOD,
    RAG_WINDOW_LINES,
    REQUEST_TIMEOUT,
)
# NOTE: the character registry is exposed through accessor functions only.
# Re-exporting the private ``_CHARACTERS`` list here used to invite the
# import-by-value trap (callers binding the pre-load empty list).
from .characters import load_characters, get_character, default_character, get_character_choices

__all__ = [
    "BOT_PREFIX",
    "CHAT_HISTORY_RESET",
    "CHUNK_TARGET",
    "CONTEXT_WINDOW",
    "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "DISCORD_TOKEN",
    "FALLBACK_MODELS",
    "INFER_API_KEY",
    "INFER_URL",
    "KB_PATH",
    "MAX_TOKENS",
    "RAG_MAX_CHARS",
    "RAG_MAX_DOCS",
    "RAG_RETRIEVAL_METHOD",
    "RAG_WINDOW_LINES",
    "REQUEST_TIMEOUT",
    "load_characters",
    "get_character",
    "default_character",
    "get_character_choices",
]
