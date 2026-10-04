"""Per-channel conversation history store (singleton) with optional disk persistence.

History (and the per-channel active-character selection) survive restarts
by being mirrored to a JSON file. All public functions keep the same
signatures as before; persistence failures are logged, never raised.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import threading

log = logging.getLogger("bot.history")

# guild_id -> channel_id -> [messages]  (each message: {"role": str, "content": str})
# NOTE: ids may be None for DM turns — that has always happened at runtime
# (persisted under the string key "None"), so the types say so too.
_chat_history: dict[int | None, dict[int | None, list[dict[str, str]]]] = {}

# Per-guild / per-channel active character key map
_active_characters: dict[tuple[int | None, int | None], str] = {}

# --- Persistence -----------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_PERSIST_PATH = _REPO_ROOT / "data" / "history.json"

_persist_path: pathlib.Path | None = None
_persist_lock = threading.Lock()
_persist_warned = False


def _get_persist_path() -> pathlib.Path | None:
    """Resolve the persistence file path.

    Honours $HISTORY_PERSIST_FILE (set it to an empty string to disable
    persistence entirely, e.g. in tests).
    """
    global _persist_path
    if _persist_path is not None:
        return _persist_path if _persist_path else None
    env = os.environ.get("HISTORY_PERSIST_FILE")
    if env is not None:
        _persist_path = pathlib.Path(env) if env else None
        return _persist_path if _persist_path else None
    return _DEFAULT_PERSIST_PATH


def _save_to_disk() -> None:
    """Atomically write in-memory state to the persistence file."""
    path = _get_persist_path()
    if path is None:
        return
    with _persist_lock:
        payload = {
            "history": {
                str(g): {str(c): msgs for c, msgs in ch.items()}
                for g, ch in _chat_history.items()
            },
            "active_characters": {
                f"{g}:{c}": k for (g, c), k in _active_characters.items()
            },
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as e:
            log.warning("Failed to persist history to %s: %s", path, e)


def _parse_id_key(raw: str) -> int | None:
    """Inverse of the ``str(g)``/``str(c)`` keys written by ``_save_to_disk``.

    DM turns have ``guild_id=None`` (and theoretically a ``None`` channel),
    which persist as the literal string ``"None"``. The old loader called
    ``int("None")``, raised, and — caught only by the outer handler — aborted
    the WHOLE load: remaining channels AND the entire active-character map
    were silently dropped. This parses every key shape symmetrically with
    how it was written.
    """
    if raw == "None":
        return None
    return int(raw)


def load_persisted() -> None:
    """Load persisted state (if any) into memory. Safe to call repeatedly."""
    global _persist_warned
    path = _get_persist_path()
    if path is None:
        return
    if not path.exists():
        return
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception as e:
        _persist_warned = True
        log.warning("Could not load persisted history from %s: %s", path, e)
        return
    loaded_channels = 0
    # Per-entry guards: one malformed key must never abort the rest of the
    # load (see _parse_id_key docstring for the historical bug).
    for g_str, ch in payload.get("history", {}).items():
        try:
            g, c = _parse_id_key(g_str), None
        except ValueError:
            log.warning("Skipping history entry with bad guild key %r", g_str)
            continue
        for c_str, msgs in ch.items():
            try:
                c = _parse_id_key(c_str)
            except ValueError:
                log.warning("Skipping history entry with bad channel key %r", c_str)
                continue
            _chat_history.setdefault(g, {})[c] = msgs
            loaded_channels += 1
    for key, char_key in payload.get("active_characters", {}).items():
        try:
            g_str, c_str = key.rsplit(":", 1)
            g, c = _parse_id_key(g_str), _parse_id_key(c_str)
        except (ValueError):
            log.warning("Skipping active-character entry with bad key %r", key)
            continue
        _active_characters[(g, c)] = char_key
    log.info("Loaded persisted history from %s (%d channel(s))", path, loaded_channels)


# --- Public API (unchanged signatures) -------------------------------------

def ensure_history(guild_id: int | None, channel_id: int | None) -> None:
    _chat_history.setdefault(guild_id, {})
    _chat_history[guild_id].setdefault(channel_id, [])


def get_history(guild_id: int | None, channel_id: int | None) -> list[dict[str, str]]:
    return _chat_history.get(guild_id, {}).get(channel_id, [])


def set_history(guild_id: int | None, channel_id: int | None,
                messages: list[dict[str, str]]) -> None:
    _chat_history.setdefault(guild_id, {})[channel_id] = messages
    _save_to_disk()


def clear_history(guild_id: int | None, channel_id: int | None) -> None:
    _chat_history.get(guild_id, {}).pop(channel_id, None)
    _active_characters.pop((guild_id, channel_id), None)
    _save_to_disk()


def reset_all_history() -> None:
    """Wipe every channel's history and active-character selection.

    Called at startup when ``CHAT_HISTORY_RESET`` is set (see main.py).
    Persists the emptied state so the wipe survives subsequent restarts.
    """
    _chat_history.clear()
    _active_characters.clear()
    _save_to_disk()


def get_message_count(guild_id: int | None, channel_id: int | None) -> int:
    return len(_chat_history.get(guild_id, {}).get(channel_id, []))


def get_active_char_key(guild_id: int | None, channel_id: int | None) -> str:
    from config.characters import default_character
    if guild_id is not None and (guild_id, channel_id) in _active_characters:
        return _active_characters[(guild_id, channel_id)]
    return default_character().key


def set_active_char_key(guild_id: int | None, channel_id: int | None,
                        char_key: str) -> None:
    if guild_id is not None:
        _active_characters[(guild_id, channel_id)] = char_key
        _save_to_disk()
