"""Persistent no-repeat pool state for /roll_items.

Tracks which items have already been rolled out so an item never comes up
twice until the pool is reset (``fresh:true`` or ``/reset_rolls``). State
lives in a hidden JSON file next to the tables — ``<ITEMS_DIR>/
.rolled_state.json`` — shaped as ``{"<table>": ["Item Name", ...]}``.

Identity is **(table, name)**: items keep their table of origin (see
:class:`bot_core.item_tables.Item`), so an all-tables roll records each item
under the table it came from and a later single-table roll still excludes it.

All read-modify-write cycles run under a process-wide lock (rolls happen in
``asyncio.to_thread``) and writes are atomic (temp file + ``os.replace``).
A missing or corrupt state file degrades to "nothing consumed" with a
warning — it must never break rolling.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path

from bot_core.item_tables import Item

log = logging.getLogger("bot.item_state")

#: Hidden filename (dot-prefixed, not ``.csv``) so table discovery, seeding,
#: and the "no tables" check never see it.
STATE_FILENAME = ".rolled_state.json"

_lock = threading.Lock()


def _state_path() -> Path:
    # Read at call time (not import time) so tests can monkeypatch settings.
    from config import settings
    return Path(settings.ITEMS_DIR) / STATE_FILENAME


def identity(item: Item) -> tuple[str, str]:
    """Pool identity of an item: ``(table, name)``."""
    return (item.table, item.name)


def load_state() -> dict[str, list[str]]:
    """Load consumed-item state. Missing/corrupt file → ``{}`` (corrupt warns)."""
    path = _state_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("item_state: unreadable state file %s (%s) — starting fresh", path, exc)
        return {}
    if not isinstance(raw, dict):
        log.warning("item_state: state file %s is not an object — starting fresh", path)
        return {}
    # Tolerate slightly malformed shapes; keep only string lists.
    return {
        table: [n for n in names if isinstance(n, str)]
        for table, names in raw.items()
        if isinstance(table, str) and isinstance(names, list)
    }


def _save_state(state: dict[str, list[str]]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=1)
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def exclude_consumed(items: list[Item]) -> tuple[list[Item], int]:
    """Split ``items`` into ``(available, total_in_scope)``.

    An item is *consumed* when its (table, name) identity is recorded in the
    state file. Stale identities (rows deleted from the CSVs) are simply
    never matched — :func:`prune` removes them on the next roll.
    """
    with _lock:
        state = load_state()
    consumed: set[tuple[str, str]] = {
        (table, name) for table, names in state.items() for name in names
    }
    available = [i for i in items if identity(i) not in consumed]
    return available, len(items)


def consume(items: list[Item]) -> None:
    """Record ``items`` as rolled out (idempotent; stale entries via :func:`prune`)."""
    with _lock:
        state = load_state()
        changed = False
        for item in items:
            table, name = identity(item)
            bucket = state.setdefault(table, [])
            if name not in bucket:
                bucket.append(name)
                changed = True
        if changed:
            _save_state(state)


def prune(items: list[Item]) -> bool:
    """Drop state entries whose (table, name) is no longer in ``items``.

    Only the buckets of tables present in ``items`` are touched, so a
    single-table roll never disturbs other tables' state. Call after loading
    the pool so deleted CSV rows don't linger forever. Returns True when
    anything was removed (and the file was rewritten).
    """
    with _lock:
        state = load_state()
        live = {identity(i) for i in items}
        present_tables = {t for t, _ in live}
        changed = False
        kept: dict[str, list[str]] = {}
        for table, names in state.items():
            if table not in present_tables:
                kept[table] = names  # outside this pool view — leave alone
                continue
            pruned = [n for n in names if (table, n) in live]
            if pruned != names:
                changed = True
            if pruned:
                kept[table] = pruned
        if not changed:
            return False
        _save_state(kept)
        return True


def reset(table: str | None = None) -> int:
    """Clear consumed entries for one table (case-insensitive) or all.

    Returns how many items were released.
    """
    with _lock:
        state = load_state()
        if table is None:
            released = sum(len(v) for v in state.values())
            if released:
                _save_state({})
            return released
        wanted = str(table).strip().lower().removesuffix(".csv")
        released = 0
        for t in [t for t in state if t.lower() == wanted]:
            released += len(state.pop(t))
        if released:
            _save_state(state)
        return released
