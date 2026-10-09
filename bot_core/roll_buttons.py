"""Per-item toggle buttons attached to roll (and item-search) result messages.

Every rolled/searched item gets one button carrying the whole interaction:

* 🚫  (red)   — item is *not* consumed → click marks it used
* ↩️  (green) — item *is* consumed    → click puts it back into the pool

Buttons use stable ``custom_id`` values of the form ``ri1:<table>|<name>`` so a
fresh view can be rebuilt for any message after a bot restart; messages that
carry buttons are recorded in ``<ITEMS_DIR>/.roll_messages.json`` and their
views are re-attached on startup (see :func:`rehydrate`).

Dispatch model (discord.py 2.7.x): the library dispatches component clicks by
calling ``item.callback(interaction)`` — there is no ``Button.click`` override
hook in this version. Each button therefore gets an *instance* callback that
shadows the class-level no-op, which keeps the normal ViewStore path
(send-time registration + ``add_view`` rehydration) fully functional.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from typing import Any, Optional, Sequence, Tuple

import discord
from discord import app_commands

from . import item_state

log = logging.getLogger(__name__)

# Bump if the custom_id format ever changes.
_CUSTOM_ID_PREFIX = "ri1"

# (table, name) pair identifying one rolled/searched item.
Entry = Tuple[str, str]

# How many message records to keep; oldest are dropped first.
_MAX_RECORDS = 200


def _records_path() -> str:
    # Read at call time (not import time) so tests can monkeypatch settings.
    from config import settings
    return os.path.join(settings.ITEMS_DIR, ".roll_messages.json")


# ---------------------------------------------------------------------------
# custom_id encoding
# ---------------------------------------------------------------------------

def encode_custom_id(table: str, name: str) -> str:
    """Encode (table, name) into a button ``custom_id`` (max 100 chars)."""
    cid = f"{_CUSTOM_ID_PREFIX}:{table}|{name}"
    if len(cid) > 100:
        # Keep the prefix + table intact; truncate the name.
        budget = 100 - len(f"{_CUSTOM_ID_PREFIX}:{table}|")
        cid = f"{_CUSTOM_ID_PREFIX}:{table}|{name[:budget]}"
    return cid


def decode_custom_id(custom_id: Optional[str]) -> Optional[Tuple[str, str]]:
    """Inverse of :func:`encode_custom_id`; ``None`` when not one of ours."""
    if not custom_id or not custom_id.startswith(f"{_CUSTOM_ID_PREFIX}:"):
        return None
    rest = custom_id[len(_CUSTOM_ID_PREFIX) + 1:]
    table, sep, name = rest.partition("|")
    if not sep or not name or not table:
        return None
    return (table, name)


def _is_ours(custom_id: Optional[str]) -> bool:
    return decode_custom_id(custom_id or "") is not None


# ---------------------------------------------------------------------------
# Buttons / view
# ---------------------------------------------------------------------------

class _RollItemButton(discord.ui.Button[discord.ui.View]):
    """One toggle button for a single (table, name) item."""

    def __init__(self, table: str, name: str, consumed: bool) -> None:
        label = f"{'↩️' if consumed else '🚫'} {name}"[:80]
        super().__init__(
            style=discord.ButtonStyle.success if consumed else discord.ButtonStyle.danger,
            label=label,
            custom_id=encode_custom_id(table, name),
        )
        self._table = table
        self._name = name
        # Instance attribute shadows the class-level no-op `callback` that
        # discord.py's ViewStore dispatch invokes. (This version of the
        # library has no Button.click override hook.)
        self.callback = self._on_click  # type: ignore[method-assign]

    async def _on_click(self, interaction: discord.Interaction) -> None:
        await _handle_toggle(interaction, self._table, self._name)


def build_view(
    entries: Sequence[Entry],
    consumed_override: Optional[set[Tuple[str, str]]] = None,
) -> discord.ui.View:
    """Build a persistent view with one toggle button per (table, name).

    ``consumed_override``, when given, is used *instead of* the state file —
    the roll command passes the freshly rolled items so their buttons show as
    consumed even though the state write happens after the send lands.

    Must be called from a thread with a running event loop (the bot's async
    handlers qualify). In this discord.py version a View constructed without
    a running loop is never dispatched — clicks on it are silently dropped.
    """
    state: dict[str, list[str]] = {} if consumed_override is not None else item_state.load_state()

    def _consumed(table: str, name: str) -> bool:
        if consumed_override is not None:
            return (table, name) in consumed_override
        return name in state.get(table, [])

    view = discord.ui.View(timeout=None)
    for table, name in entries:
        view.add_item(_RollItemButton(table, name, _consumed(table, name)))
    return view


def build_roll_view(table: str, names: Sequence[str]) -> discord.ui.View:
    """View for a roll result: every rolled item shown as consumed."""
    pairs = [(table, n) for n in names]
    return build_view(pairs, consumed_override=set(pairs))


# ---------------------------------------------------------------------------
# Message records (persisted so views survive restarts)
# ---------------------------------------------------------------------------

_records_lock = threading.Lock()
_record_cache: Optional[dict[int, dict[str, Any]]] = None


def _normalize_record(rec: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return {"channel": int, "message": int, "items": [[table, name], ...]}
    or None if the record is unusable. Accepts the legacy single-table shape."""
    try:
        channel = int(rec["channel"])
        message = int(rec["message"])
    except (KeyError, TypeError, ValueError):
        return None
    raw_items = rec.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        return None
    items: list[list[str]] = []
    legacy_table = rec.get("table", "")
    for entry in raw_items:
        if isinstance(entry, str):  # legacy shape: names under one table
            items.append([legacy_table, entry])
        elif (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and all(isinstance(x, str) for x in entry)
        ):
            items.append([entry[0], entry[1]])
    if not items:
        return None
    return {"channel": channel, "message": message, "items": items}


def _load_cache_locked() -> dict[int, dict[str, Any]]:
    global _record_cache
    if _record_cache is not None:
        return _record_cache
    path = _records_path()
    data: list[dict[str, Any]] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
        except (OSError, json.JSONDecodeError) as e:
            log.warning("roll_buttons: could not read %s (%s); starting empty", path, e)
    cache: dict[int, dict[str, Any]] = {}
    for rec in data:
        if isinstance(rec, dict):
            norm = _normalize_record(rec)
            if norm is not None:
                cache[norm["message"]] = norm
    _record_cache = cache
    return cache


def _save_records_locked() -> None:
    """Persist the record cache (caller must hold ``_records_lock``)."""
    records = [
        {"channel": rec["channel"], "message": mid, "items": rec["items"]}
        for mid, rec in _load_cache_locked().items()
    ]
    path = _records_path()
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record_roll_message(
    channel_id: int, message_id: int, entries: Sequence[Entry]
) -> None:
    """Remember that ``message_id`` in ``channel_id`` carries these buttons."""
    with _records_lock:
        cache = _load_cache_locked()
        cache[int(message_id)] = {
            "channel": int(channel_id),
            "items": [[t, n] for t, n in entries],
        }
        # Drop oldest records beyond the cap (insertion order = recency).
        while len(cache) > _MAX_RECORDS:
            cache.pop(next(iter(cache)))
        try:
            _save_records_locked()
        except OSError as e:
            log.error("roll_buttons: could not save message records: %s", e)


def forget_record(message_id: int) -> None:
    with _records_lock:
        if _load_cache_locked().pop(int(message_id), None) is not None:
            try:
                _save_records_locked()
            except OSError as e:
                log.error("roll_buttons: could not save message records: %s", e)


def reset_records_for_tests() -> None:
    """Clear the in-memory record cache (tests only)."""
    global _record_cache
    with _records_lock:
        _record_cache = None


def _message_pairs(message_id: int) -> Optional[list[Entry]]:
    with _records_lock:
        rec = _load_cache_locked().get(int(message_id))
        return [tuple(x) for x in rec["items"]] if rec else None


def _sibling_pairs(message: discord.Message) -> list[Entry]:
    """(table, name) pairs of our buttons as currently rendered on the message."""
    pairs: list[Entry] = []
    for row in getattr(message.components, "rows", []):
        for comp in getattr(row, "components", []):
            decoded = decode_custom_id(getattr(comp, "custom_id", None))
            if decoded is not None:
                pairs.append(decoded)
    return pairs


async def rehydrate(bot: discord.Client) -> None:
    """Re-attach persistent views to recorded messages after a restart."""
    with _records_lock:
        records = list(_load_cache_locked().items())
    alive = 0
    for message_id, rec in records:
        try:
            channel = bot.get_channel(rec["channel"])
            if channel is None or not hasattr(channel, "fetch_message"):
                continue
            await channel.fetch_message(message_id)
            # Rebuild from the live state file so labels are current.
            bot.add_view(build_view([tuple(x) for x in rec["items"]]))
            alive += 1
        except discord.HTTPException:
            forget_record(message_id)  # deleted / inaccessible
        except Exception as e:  # noqa: BLE001 - one bad record must not kill the rest
            log.warning("roll_buttons: rehydrate failed for message %s: %s", message_id, e)
    if records:
        log.info("roll_buttons: rehydrated %d/%d roll messages", alive, len(records))


# ---------------------------------------------------------------------------
# Click handling
# ---------------------------------------------------------------------------

async def _handle_toggle(interaction: discord.Interaction, table: str, name: str) -> None:
    """Toggle consumption for (table, name) and refresh the message's buttons."""
    msg = interaction.message
    if msg is None:
        return

    # The consumed list is shared guild state (like /item_exclude), so any
    # member may toggle — no per-user ownership check.
    pairs = _sibling_pairs(msg) or _message_pairs(msg.id)
    if pairs is None:
        # Message components were stripped but we still know the items.
        pairs = [(table, name)]

    consumed = item_state.toggle_item(table, name)
    new_view = build_view(pairs)  # labels reflect the state *after* the toggle
    try:
        await msg.edit(view=new_view)
    except discord.HTTPException as e:
        item_state.toggle_item(table, name)  # roll back — edit failed
        await _respond(interaction, f"⚠️ Could not update the message: {e}", ephemeral=True)
        return
    text = f"✅ Marked as used: **{name}**" if consumed else f"↩️ Put back: **{name}**"
    try:
        await interaction.response.send_message(text, ephemeral=True)
    except discord.HTTPException:
        pass  # cosmetic only; the state change already happened

    with _records_lock:
        cache = _load_cache_locked()
        if msg.id in cache:
            cache[msg.id]["items"] = [[t, n] for t, n in pairs]
            try:
                _save_records_locked()
            except OSError as e:
                log.error("roll_buttons: could not save message records: %s", e)


async def _respond(
    interaction: discord.Interaction, content: str, *, ephemeral: bool = False
) -> None:
    """Respond to the interaction, degrading gracefully if it's too late."""
    try:
        await interaction.response.send_message(content, ephemeral=ephemeral)
    except app_commands.AppCommandError:
        try:
            await interaction.followup.send(content, ephemeral=ephemeral)
        except discord.HTTPException:
            pass
