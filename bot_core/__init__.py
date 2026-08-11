"""Bot core — AI client + conversation history."""
from .ai_client import ask_ai, get_current_message_count
from .history import (
    clear_history,
    ensure_history,
    get_active_char_key,
    get_history,
    get_message_count,
    set_active_char_key,
    set_history,
)

__all__ = [
    "ask_ai",
    "clear_history",
    "ensure_history",
    "get_active_char_key",
    "get_current_message_count",
    "get_history",
    "get_message_count",
    "set_active_char_key",
    "set_history",
]

# Backward-compat alias
_chat_history = {}  # type: ignore[assignment]  # deprecated: use get_history()
