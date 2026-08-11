"""Per-channel conversation history store (singleton)."""
from __future__ import annotations

# guild_id -> channel_id -> [messages]
_chat_history: dict[int, dict[int, list[dict]]] = {}

# Per-guild / per-channel active character key map
_active_characters: dict[tuple[int, int], str] = {}


def ensure_history(guild_id: int, channel_id: int) -> None:
    _chat_history.setdefault(guild_id, {})
    _chat_history[guild_id].setdefault(channel_id, [])


def get_history(guild_id: int, channel_id: int) -> list[dict]:
    return _chat_history.get(guild_id, {}).get(channel_id, [])


def set_history(guild_id: int, channel_id: int, messages: list[dict]) -> None:
    _chat_history.setdefault(guild_id, {})[channel_id] = messages


def clear_history(guild_id: int, channel_id: int) -> None:
    _chat_history.get(guild_id, {}).pop(channel_id, None)


def get_message_count(guild_id: int, channel_id: int) -> int:
    return len(_chat_history.get(guild_id, {}).get(channel_id, []))


def get_active_char_key(guild_id: int | None, channel_id: int) -> str:
    from config.characters import default_character
    if guild_id is not None and (guild_id, channel_id) in _active_characters:
        return _active_characters[(guild_id, channel_id)]
    return default_character().key


def set_active_char_key(guild_id: int | None, channel_id: int, char_key: str) -> None:
    if guild_id is not None:
        _active_characters[(guild_id, channel_id)] = char_key
