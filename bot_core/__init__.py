"""Bot core — AI client + conversation history."""
from .ai_client import (
    ask_ai,
    RateLimitError,
)
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
    "get_history",
    "get_message_count",
    "RateLimitError",
    "set_active_char_key",
    "set_history",
]

# Backward-compat alias (deprecated: use get_history()/set_history()).
# Mirrors the runtime shape of bot_core.history._chat_history.
_chat_history: dict[int | None, dict[int | None, list[dict[str, str]]]] = {}
