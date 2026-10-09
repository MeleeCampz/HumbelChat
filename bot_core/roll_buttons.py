"""Per-item toggle buttons on /roll_items results.

Every roll reply carries one button per rolled item (classic message
components, up to 5 rows x 5 buttons = 25 — the same cap as the ``count``
argument). Clicking a button toggles that single item in the consumed list:

* not consumed  → button shows ``🚫 <name>`` (danger); click marks it used
* consumed      → button shows ``↩️ <name>`` (success); click puts it back

The button label/style on the message itself is the persistent state
indicator, so a scroll-back months later still shows which items are out.

Buttons keep working after a bot restart: each roll records
``(channel, message, table, item names)`` in ``<ITEMS_DIR>/.roll_messages.json``
and :func:`rehydrate` re-registers the views on startup (persistent-view
pattern — ``timeout=None`` + ``custom_id``s + ``client.add_view``). The
``custom_id`` encodes the full ``(table, name)`` identity, so a click never
needs per-message state beyond "which items were on this message" (taken
from the record, with the rendered sibling buttons as fallback).

All state mutations go through :mod:`bot_core.item_state` (locked + atomic),
so button clicks and slash commands are safe to interleave. A click that
cannot be answered edits the message with a short warning instead of dying
silently.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Sequence

import discord

from bot_core import item_state, item_tables

log = logging.getLogger("bot.roll_buttons")

#: Prefix identifying our buttons (versioned so a future format change can
#: ignore legacy custom_ids instead of misparsing them).
CUSTOM_ID_PREFIX = "ri1:"
_SEP = "|"
#: Discord hard limits: custom_id 100 chars, button label 80 chars.
_MAX_CUSTOM_ID = 100
_MAX_LABEL = 80
#: Max buttons rendered per message (5 action rows x 5 buttons).
_MAX_BUTTONS = 25
#: How many roll messages we keep rehydratable across restarts.
_MAX_RECORDS = 200

EMOJI_EXCLUDE = "🚫"   # click → mark used
EMOJI_INCLUDE = "↩️"   # click → put back


# ── custom_id encoding ─────────────────────────────────────────────────────

def encode_custom_id(table: str, name: str) -> str | None:
    """``ri1:<table>|<name>`` — None when it would exceed Discord's 100 chars.

    Decoding splits on the FIRST separator only, so item names may contain
    ``|`` (table names come from filenames and never do).
    """
    cid = f"{CUSTOM_ID_PREFIX}{table}{_SEP}{name}"
    if len(cid) > _MAX_CUSTOM_ID:
        return None
    return cid


def decode_custom_id(custom_id: str) -> tuple[str, str] | None:
    """Inverse of :func:`encode_custom_id`; None for foreign/legacy ids."""
    if not custom_id.startswith(CUSTOM_ID_PREFIX):
        return None
    body = custom_id[len(CUSTOM_ID_PREFIX):]
    table, sep, name = body.partition(_SEP)
    if not sep or not table or not name:
        return None
    return table, name


# ── view construction ─────────────────────────────────────────────────────

def _label(name: str, consumed: bool) -> str:
    emoji = EMOJI_INCLUDE if consumed else EMOJI_EXCLUDE
    budget = _MAX_LABEL - len(emoji) - 1  # emoji + one space
    if len(name) > budget:
        name = name[: max(budget - 1, 1)] + "…"
    return f"{emoji} {name}"


class _RollItemButton(discord.ui.Button):  # type: ignore[misc]  # discord.py ships no stubs for ui.Button
    """One rolled item; click toggles its consumed state."""

    def __init__(self, table: str, name: str, consumed: bool) -> None:
        super().__init__(
            style=discord.ButtonStyle.success if consumed else discord.ButtonStyle.danger,
            label=_label(name, consumed),
            custom_id=f"{CUSTOM_ID_PREFIX}{table}{_SEP}{name}",
        )
        self._table = table
        self._name = name

    async def click(self, interaction: discord.Interaction) -> None:
        await _handle_toggle(interaction, self._table, self._name)


def build_view(
    table: str,
    names: Sequence[str],
    consumed_override: set[str] | None = None,
) -> discord.ui.View:
    """A persistent view with one toggle button per name (max 25).

    Button styles reflect the consumed state at build time; rehydrate() and
    every click rebuild the view, so labels always match the state file.
    ``consumed_override`` forces a specific consumed set (used right after a
    consume roll, before the state write has landed). Names without a valid
    custom_id (>100 chars) are skipped with a warning.
    """
    consumed = (
        set(consumed_override)
        if consumed_override is not None
        else set(item_state.load_state().get(table, []))
    )
    view = discord.ui.View(timeout=None)
    # A plain View auto-chunks added items into action rows of five; add up
    # to 25 buttons (Discord's top-level component cap for legacy messages).
    for name in names[:_MAX_BUTTONS]:
        if encode_custom_id(table, name) is None:
            log.warning("roll_buttons: skipping %r in %s — custom_id too long", name, table)
            continue
        view.add_item(_RollItemButton(table, name, name in consumed))
    if len(names) > _MAX_BUTTONS:
        log.warning("roll_buttons: %d items in %s exceed %d — only the first %d get buttons",
                    len(names), table, _MAX_BUTTONS, _MAX_BUTTONS)
    return view


# ── click handling ─────────────────────────────────────────────────────────

def _message_names(message_id: int, table: str) -> list[str] | None:
    """Items rendered on a roll message: record first, sibling buttons second."""
    with _records_lock:
        rec = _record_cache_locked().get(message_id)
    if rec is not None and rec.get("table") == table:
        return list(rec.get("items", []))
    return None


def _sibling_names(message: discord.Message | None) -> list[str] | None:
    """Parse our custom_ids off the message's rendered buttons (fallback)."""
    if message is None:
        return None
    names: list[str] = []
    for component in getattr(message, "components", []) or []:
        for child in getattr(component, "components", []) or []:
            decoded = decode_custom_id(getattr(child, "custom_id", "") or "")
            if decoded is not None:
                names.append(decoded[1])
    return names or None


def _toggle(table: str, name: str) -> dict[str, Any]:
    """Flip one item's consumed state; returns a plain result dict.

    Identity-based (no live CSV row required): marking a row that was later
    deleted just records stale state, which the next roll prunes. Each
    mutation runs under item_state's own lock; the read-decide window is
    tiny and both operations are idempotent, so no outer lock (the lock is
    not reentrant).
    """
    consumed = set(item_state.load_state().get(table, []))
    if name in consumed:
        released = item_state.release_item(table, name)
        if not released:
            return {"ok": False, "error": f"⚠️ Couldn't put **{name}** back — check the bot logs."}
        return {
            "ok": True,
            "consumed": False,
            "text": f"✅ **{name}** is back in the pool — it can come up in consume rolls again.",
        }
    item = item_tables.Item(name=name, table=table)
    item_state.consume([item])
    return {
        "ok": True,
        "consumed": True,
        "text": f"🚫 **{name}** is marked as used — it won't come up in consume rolls until put back.",
    }


async def _handle_toggle(interaction: discord.Interaction, table: str, name: str) -> None:
    """Button-click body: toggle state, rebuild the row, confirm."""
    try:
        result = await asyncio.to_thread(_toggle, table, name)
        if not result["ok"]:
            await interaction.response.edit_message(content=result["error"])
            return
        names = _message_names(interaction.message.id, table) or _sibling_names(interaction.message)
        if not names:
            # No record and no parsable siblings (shouldn't happen) — the
            # state still toggled; just confirm without touching components.
            await interaction.response.edit_message(content=result["text"])
            return
        new_view = await asyncio.to_thread(build_view, table, names)
        await interaction.response.edit_message(view=new_view, content=result["text"])
    except discord.HTTPException:
        # Message vanished / edit race — state is already consistent.
        log.warning("roll_buttons: could not update message %s after toggle",
                    getattr(interaction.message, "id", "?"), exc_info=True)
    except Exception:
        log.exception("roll_buttons: unexpected error toggling %s/%s", table, name)


# ── roll-message records (restart rehydration) ─────────────────────────────

RECORDS_FILENAME = ".roll_messages.json"
_records_lock = threading.Lock()
_cache: dict[int, dict[str, Any]] | None = None  # message_id → record
#: Messages whose view is (or was) registered in this process — on_ready
#: fires on every gateway reconnect, so rehydrate must not refetch/re-add.
_live_messages: set[int] = set()


def _records_path() -> Path:
    from config import settings
    return Path(settings.ITEMS_DIR) / RECORDS_FILENAME


def _load_cache() -> dict[int, dict[str, Any]]:
    """Read the records file into a fresh dict (no locking — caller holds it)."""
    path = _records_path()
    raw: list[dict[str, Any]] = []
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                raw = [r for r in loaded if isinstance(r, dict)]
        except (OSError, ValueError) as exc:
            log.warning("roll_buttons: unreadable %s (%s) — starting fresh", path, exc)
    return {
        int(r["message"]): r
        for r in raw
        if isinstance(r.get("message"), int) and isinstance(r.get("table"), str)
    }


def _record_cache_locked() -> dict[int, dict[str, Any]]:
    """In-memory records, loaded on first use. Caller must hold ``_records_lock``."""
    global _cache
    if _cache is None:
        _cache = _load_cache()
    return _cache


def _save_records(cache: dict[int, dict[str, Any]]) -> None:
    """Persist the given records (oldest-first by insertion, capped)."""
    records = list(cache.values())[-_MAX_RECORDS:]
    path = _records_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=1)
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def record_roll_message(channel_id: int, message_id: int, table: str, names: list[str]) -> None:
    """Remember a roll message so its buttons survive a restart."""
    _live_messages.add(message_id)
    with _records_lock:
        cache = _record_cache_locked()
        cache[message_id] = {
            "channel": int(channel_id),
            "message": int(message_id),
            "table": table,
            "items": [str(n) for n in names],
        }
        # Rebuild the file from insertion order (dict preserves it).
        _save_records(cache)


def forget_record(message_id: int) -> None:
    with _records_lock:
        cache = _record_cache_locked()
        if message_id in cache:
            del cache[message_id]
            _save_records(cache)


def reset_records_for_tests() -> None:
    """Clear in-memory state (tests point ITEMS_DIR at fresh temp dirs)."""
    global _cache
    with _records_lock:
        _cache = None
    _live_messages.clear()


# ── startup rehydration ────────────────────────────────────────────────────

async def rehydrate(bot: discord.Client) -> int:
    """Re-register button views for roll messages from before a restart.

    Called from ``on_ready``. Fetches each recorded message (skipping
    channels that are gone, deleting records whose message vanished) and
    calls ``bot.add_view`` with a fresh view. Returns how many were restored.
    Never raises — button persistence must not break startup.
    """
    try:
        with _records_lock:
            records = [
                r for r in _record_cache_locked().values()
                if int(r["message"]) not in _live_messages
            ]
        restored = 0
        for rec in records:
            message_id = int(rec["message"])
            channel = bot.get_channel(rec["channel"])
            if channel is None:
                try:
                    channel = await bot.fetch_channel(rec["channel"])
                except discord.HTTPException:
                    channel = None
            if channel is None:
                continue  # not cached yet / no access — retry next startup
            try:
                await channel.fetch_message(message_id)  # raises when gone
            except discord.NotFound:
                forget_record(message_id)
                continue
            except discord.HTTPException:
                continue  # transient — retry next startup
            try:
                bot.add_view(build_view(rec["table"], rec["items"]))
                _live_messages.add(message_id)
                restored += 1
            except Exception:
                log.exception("roll_buttons: failed to rebuild view for message %s", message_id)
        if records:
            log.info("roll_buttons: rehydrated buttons on %d/%d roll message(s)",
                     restored, len(records))
        return restored
    except Exception:
        log.exception("roll_buttons: rehydrate failed (buttons on old messages stay dead until next restart)")
        return 0
