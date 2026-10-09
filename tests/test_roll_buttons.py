"""Tests for per-item roll buttons (bot_core/roll_buttons).

Covers custom_id encoding, view construction (rows/labels/styles), the
toggle state machine against the real item_state store, roll-message record
persistence/cap, and startup rehydration with a fake bot.
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot_core import item_state, item_tables, roll_buttons


@pytest.fixture(autouse=True)
def items_dir(tmp_path, monkeypatch):
    """Fresh ITEMS_DIR per test (state + records files live there)."""
    d = tmp_path / "items"
    d.mkdir()
    from config import settings
    monkeypatch.setattr(settings, "ITEMS_DIR", d)
    roll_buttons.reset_records_for_tests()
    yield


# ── custom_id encoding ─────────────────────────────────────────────────────

class TestCustomId:
    def test_roundtrip(self):
        cid = roll_buttons.encode_custom_id("magic_items", "Bag of Holding")
        assert cid == "ri1:magic_items|Bag of Holding"
        assert roll_buttons.decode_custom_id(cid) == ("magic_items", "Bag of Holding")

    def test_name_with_separator(self):
        cid = roll_buttons.encode_custom_id("t", "a|b|c")
        assert roll_buttons.decode_custom_id(cid) == ("t", "a|b|c")

    def test_over_100_chars_rejected(self):
        assert roll_buttons.encode_custom_id("t", "x" * 200) is None

    def test_foreign_prefix_ignored(self):
        assert roll_buttons.decode_custom_id("other:thing") is None
        assert roll_buttons.decode_custom_id(roll_buttons.CUSTOM_ID_PREFIX) is None
        assert roll_buttons.decode_custom_id(f"{roll_buttons.CUSTOM_ID_PREFIX}onlytable") is None


# ── view construction ─────────────────────────────────────────────────────

class TestBuildView:
    def test_one_button_per_item(self):
        view = roll_buttons.build_view("t", ["A", "B", "C"])
        assert len(view.children) == 3
        e = roll_buttons.EMOJI_EXCLUDE
        labels = [c.label for c in view.children]
        assert labels == [f"{e} A", f"{e} B", f"{e} C"]
        assert all(c.style == discord.ButtonStyle.danger for c in view.children)

    def test_consumed_items_show_include(self):
        item_state.consume([item_tables.Item(name="A", table="t")])
        view = roll_buttons.build_view("t", ["A", "B"])
        assert view.children[0].label == f"{roll_buttons.EMOJI_INCLUDE} A"
        assert view.children[0].style == discord.ButtonStyle.success
        assert view.children[1].label == f"{roll_buttons.EMOJI_EXCLUDE} B"

    def test_consumed_override(self):
        view = roll_buttons.build_view("t", ["A", "B"], consumed_override={"A"})
        assert view.children[0].label == f"{roll_buttons.EMOJI_INCLUDE} A"
        assert view.children[1].label == f"{roll_buttons.EMOJI_EXCLUDE} B"

    def test_chunked_into_rows_of_five(self):
        view = roll_buttons.build_view("t", [f"I{i}" for i in range(12)])
        assert len(view.children) == 12
        rows = view.to_components()
        assert [len(r["components"]) for r in rows] == [5, 5, 2]

    def test_cap_at_25(self):
        view = roll_buttons.build_view("t", [f"I{i}" for i in range(30)])
        assert len(view.children) == 25

    def test_label_truncated_to_80(self):
        # 90 chars: fits the 100-char custom_id but overflows the 80-char label.
        long_name = "x" * 90
        view = roll_buttons.build_view("t", [long_name])
        assert len(view.children[0].label) <= 80
        assert view.children[0].label.endswith("…")

    def test_timeout_none_for_persistence(self):
        view = roll_buttons.build_view("t", ["A"])
        assert view.timeout is None


# ── toggle state machine ───────────────────────────────────────────────────

def _make_ix(message_id: int = 123) -> MagicMock:
    ix = MagicMock()
    ix.message.id = message_id
    ix.response.edit_message = AsyncMock()
    return ix


class TestToggle:
    async def test_mark_used_then_put_back(self):
        roll_buttons.record_roll_message(1, 123, "t", ["Sword", "Potion"])

        ix = _make_ix()
        await roll_buttons._handle_toggle(ix, "t", "Sword")
        assert item_state.load_state().get("t") == ["Sword"]
        kwargs = ix.response.edit_message.await_args.kwargs
        assert "view" in kwargs and "content" in kwargs
        assert "back in the pool" not in kwargs["content"]

        ix2 = _make_ix()
        await roll_buttons._handle_toggle(ix2, "t", "Sword")
        assert item_state.load_state().get("t", []) == []
        assert "back in the pool" in ix2.response.edit_message.await_args.kwargs["content"]

    async def test_button_click_dispatch(self):
        """The Button subclass routes clicks into the same handler."""
        roll_buttons.record_roll_message(1, 123, "t", ["Sword"])
        view = roll_buttons.build_view("t", ["Sword"])
        ix = _make_ix()
        await view.children[0].click(ix)
        assert item_state.load_state().get("t") == ["Sword"]

    async def test_no_record_falls_back_to_siblings(self):
        message = MagicMock()
        btn = MagicMock(custom_id=roll_buttons.encode_custom_id("t", "Sword"))
        row = MagicMock(components=[btn])
        message.components = [row]
        ix = _make_ix()
        ix.message = MagicMock(id=999, components=[row])
        await roll_buttons._handle_toggle(ix, "t", "Sword")
        assert item_state.load_state().get("t") == ["Sword"]

    async def test_missing_message_still_toggles_with_text_only(self):
        ix = _make_ix(message_id=42)  # no record, no siblings
        await roll_buttons._handle_toggle(ix, "t", "Ghost")
        assert item_state.load_state().get("t") == ["Ghost"]
        kwargs = ix.response.edit_message.await_args.kwargs
        assert "view" not in kwargs

    async def test_http_error_swallowed(self):
        roll_buttons.record_roll_message(1, 123, "t", ["Sword"])
        ix = _make_ix()
        ix.response.edit_message.side_effect = discord.HTTPException(MagicMock(), "boom")
        await roll_buttons._handle_toggle(ix, "t", "Sword")  # must not raise
        assert item_state.load_state().get("t") == ["Sword"]


# ── records persistence ────────────────────────────────────────────────────

class TestRecords:
    def test_roundtrip_through_file(self):
        roll_buttons.record_roll_message(1, 111, "t", ["A", "B"])
        from config import settings
        raw = json.loads((pathlib.Path(settings.ITEMS_DIR) / roll_buttons.RECORDS_FILENAME).read_text())
        assert raw == [{"channel": 1, "message": 111, "table": "t", "items": ["A", "B"]}]

    def test_cache_reload_from_file(self):
        roll_buttons.record_roll_message(1, 111, "t", ["A"])
        roll_buttons.reset_records_for_tests()
        with roll_buttons._records_lock:
            assert roll_buttons._record_cache_locked()[111]["table"] == "t"

    def test_cap_at_max_records(self):
        for i in range(roll_buttons._MAX_RECORDS + 5):
            roll_buttons.record_roll_message(1, i, "t", ["A"])
        from config import settings
        raw = json.loads((pathlib.Path(settings.ITEMS_DIR) / roll_buttons.RECORDS_FILENAME).read_text())
        assert len(raw) == roll_buttons._MAX_RECORDS
        # oldest evicted, newest kept
        assert all(r["message"] >= 5 for r in raw)

    def test_forget_record(self):
        roll_buttons.record_roll_message(1, 111, "t", ["A"])
        roll_buttons.forget_record(111)
        with roll_buttons._records_lock:
            assert 111 not in roll_buttons._record_cache_locked()


# ── rehydration ────────────────────────────────────────────────────────────

class TestRehydrate:
    async def test_restores_views_and_skips_live(self):
        roll_buttons.record_roll_message(7, 500, "t", ["A"])  # live (sent this process)
        probe = MagicMock()
        probe.get_channel.return_value = None
        probe.fetch_channel = AsyncMock(side_effect=discord.NotFound(MagicMock(), "x"))
        assert await roll_buttons.rehydrate(probe) == 0  # live → skipped
        roll_buttons._live_messages.discard(500)               # simulate restart
        channel = MagicMock()
        channel.fetch_message = AsyncMock(return_value=MagicMock())
        bot = MagicMock()
        bot.get_channel.return_value = None
        bot.fetch_channel = AsyncMock(return_value=channel)
        restored = await roll_buttons.rehydrate(bot)
        assert restored == 1
        bot.add_view.assert_called_once()
        view = bot.add_view.call_args.args[0]
        assert len(view.children) == 1
        # now marked live — a reconnect re-run does nothing
        assert await roll_buttons.rehydrate(bot) == 0

    async def test_vanished_message_forgotten(self):
        roll_buttons.record_roll_message(7, 500, "t", ["A"])
        roll_buttons._live_messages.discard(500)  # simulate restart
        channel = MagicMock()
        channel.fetch_message = AsyncMock(side_effect=discord.NotFound(MagicMock(), "gone"))
        bot = MagicMock()
        bot.get_channel.return_value = channel
        restored = await roll_buttons.rehydrate(bot)
        assert restored == 0
        with roll_buttons._records_lock:
            assert 500 not in roll_buttons._record_cache_locked()

    async def test_uncached_channel_skipped_not_forgotten(self):
        roll_buttons.record_roll_message(7, 500, "t", ["A"])
        roll_buttons._live_messages.discard(500)  # simulate restart
        bot = MagicMock()
        bot.get_channel.return_value = None
        bot.fetch_channel = AsyncMock(side_effect=discord.NotFound(MagicMock(), "nope"))
        restored = await roll_buttons.rehydrate(bot)
        assert restored == 0
        with roll_buttons._records_lock:
            assert 500 in roll_buttons._record_cache_locked()
