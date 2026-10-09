"""Tests for bot_core.roll_buttons — per-item toggle buttons on roll messages."""
from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot_core import item_state, roll_buttons as rb
from config import settings


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Point item state and message records at a temp dir; reset the cache."""
    monkeypatch.setattr(settings, "ITEMS_DIR", str(tmp_path))
    rb.reset_records_for_tests()
    yield
    rb.reset_records_for_tests()


# ---------------------------------------------------------------------------
# custom_id encoding
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_roundtrip(self):
        assert rb.decode_custom_id(rb.encode_custom_id("weapons", "Longsword")) == (
            "weapons",
            "Longsword",
        )

    def test_roundtrip_name_with_pipe(self):
        # Only the FIRST pipe separates table from name.
        assert rb.decode_custom_id(rb.encode_custom_id("t", "a|b|c")) == ("t", "a|b|c")

    def test_decode_rejects_foreign_ids(self):
        assert rb.decode_custom_id("other:stuff") is None
        assert rb.decode_custom_id("ri1:nopipe") is None
        empty_table = rb.encode_custom_id("", "x")  # "ri1:|x"
        assert rb.decode_custom_id(empty_table) is None

    def test_long_names_truncated_to_100(self):
        cid = rb.encode_custom_id("a_very_long_table_name", "x" * 500)
        assert len(cid) <= 100
        table, name = rb.decode_custom_id(cid)
        assert table == "a_very_long_table_name"
        assert name.startswith("x") and len(name) < 500


# ---------------------------------------------------------------------------
# View construction
# ---------------------------------------------------------------------------

def _buttons(view):
    # Classic views hold flat item lists; rows are derived at render time.
    return list(view.children)


class TestBuildView:
    def test_one_button_per_entry(self):
        view = rb.build_view([("weapons", "A"), ("weapons", "B"), ("armor", "C")])
        assert len(_buttons(view)) == 3

    def test_labels_and_styles_from_state(self, monkeypatch):
        state_file = os.path.join(settings.ITEMS_DIR, ".rolled_state.json")
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({"weapons": ["Used Sword"]}, f)

        import discord as _d

        view = rb.build_view([("weapons", "Used Sword"), ("weapons", "Fresh Sword")])
        used, fresh = _buttons(view)
        assert used.label.startswith("↩️") and used.style is _d.ButtonStyle.success
        assert fresh.label.startswith("🚫") and fresh.style is _d.ButtonStyle.danger

    def test_consumed_override_wins_over_state(self, monkeypatch):
        state_file = os.path.join(settings.ITEMS_DIR, ".rolled_state.json")
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({"weapons": []}, f)

        # Consume-roll path: items not yet in the state file must still show ↩️.
        view = rb.build_roll_view("weapons", ["Sword"])
        btn = _buttons(view)[0]
        assert btn.label.startswith("↩️")

    def test_long_label_truncated(self):
        view = rb.build_view([("t", "y" * 300)])
        assert len(_buttons(view)[0].label) <= 80

    def test_persistent_timeout_none(self):
        assert rb.build_view([("t", "n")]).timeout is None


# ---------------------------------------------------------------------------
# Click handling — dispatched the way discord.py actually does it:
# ViewStore calls `item.callback(interaction)` on the registered item.
# ---------------------------------------------------------------------------

class _FakeInteraction:
    def __init__(self, message):
        self.message = message
        self.user = MagicMock()
        self.response = MagicMock()
        self.response.send_message = AsyncMock()
        self.followup = MagicMock()
        self.followup.send = AsyncMock()

    async def defer(self):
        pass


def _fake_message(buttons, author) -> MagicMock:
    msg = MagicMock()
    msg.author = author
    row = MagicMock()
    row.components = buttons
    msg.components.rows = [row]
    return msg


class TestToggle:
    async def test_click_marks_consumed_and_edits(self):
        item_state.reset("weapons")
        author = MagicMock()
        msg = _fake_message([], author)
        msg.edit = AsyncMock()
        view = rb.build_view([("weapons", "Sword")])
        btn = _buttons(view)[0]

        ix = _FakeInteraction(msg)
        ix.user = author
        await btn.callback(ix)  # the real dispatch entry point

        state = item_state.load_state()
        assert state.get("weapons") == ["Sword"]
        msg.edit.assert_awaited_once()
        # The edit must carry the rebuilt view with a single consumed button.
        kwargs = msg.edit.await_args.kwargs
        new_view = kwargs["view"]
        assert len(_buttons(new_view)) == 1
        assert _buttons(new_view)[0].label.startswith("↩️")
        ix.response.send_message.assert_awaited_once()

    async def test_click_again_releases(self):
        item_state.consume([item_state.Item(name="Sword", table="weapons")])
        author = MagicMock()
        msg = _fake_message([], author)
        msg.edit = AsyncMock()
        view = rb.build_roll_view("weapons", ["Sword"])
        btn = _buttons(view)[0]

        ix = _FakeInteraction(msg)
        ix.user = author
        await btn.callback(ix)

        assert "Sword" not in item_state.load_state().get("weapons", [])

    async def test_click_rolls_back_when_edit_fails(self):
        import discord as _d

        item_state.reset("weapons")
        author = MagicMock()
        msg = _fake_message([], author)
        msg.edit = AsyncMock(side_effect=_d.HTTPException(MagicMock(), {"message": "nope"}))
        view = rb.build_view([("weapons", "Sword")])
        btn = _buttons(view)[0]

        ix = _FakeInteraction(msg)
        ix.user = author
        await btn.callback(ix)

        assert item_state.load_state().get("weapons") is None  # rolled back
        msg.edit.assert_awaited_once()

    def test_instance_callback_shadows_class_noop(self):
        """Regression: discord.py 2.7.x dispatches via `item.callback`, not a
        `Button.click` override — the instance attribute must exist."""
        view = rb.build_view([("weapons", "Sword")])
        btn = _buttons(view)[0]
        assert "callback" in btn.__dict__
        assert btn.__dict__["callback"] is not type(btn).callback


# ---------------------------------------------------------------------------
# Records persistence
# ---------------------------------------------------------------------------

class TestRecords:
    def test_record_and_reload(self):
        rb.record_roll_message(111, 222, [("weapons", "Sword"), ("armor", "Shield")])
        rb.reset_records_for_tests()  # force reload from disk
        pairs = rb._message_pairs(222)
        assert pairs == [("weapons", "Sword"), ("armor", "Shield")]

    def test_record_file_shape(self):
        rb.record_roll_message(1, 2, [("t", "n")])
        path = rb._records_path()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data == [{"channel": 1, "message": 2, "items": [["t", "n"]]}]

    def test_legacy_single_table_records_migrate(self):
        path = rb._records_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"channel": 5, "message": 6, "table": "weapons", "items": ["A", "B"]}], f)
        assert rb._message_pairs(6) == [("weapons", "A"), ("weapons", "B")]

    def test_cap_drops_oldest(self):
        for i in range(rb._MAX_RECORDS + 20):
            rb.record_roll_message(1, i, [("t", f"n{i}")])
        assert rb._message_pairs(0) is None
        assert rb._message_pairs(19) is None
        assert rb._message_pairs(rb._MAX_RECORDS + 19) == [("t", "n219")]

    def test_forget_record(self):
        rb.record_roll_message(1, 2, [("t", "n")])
        rb.forget_record(2)
        rb.reset_records_for_tests()
        assert rb._message_pairs(2) is None

    def test_corrupt_file_starts_empty(self):
        path = rb._records_path()
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        assert rb._message_pairs(1) is None


# ---------------------------------------------------------------------------
# Rehydration
# ---------------------------------------------------------------------------

class TestRehydrate:
    async def test_reattaches_views_and_drops_deleted(self):
        rb.record_roll_message(100, 200, [("weapons", "Sword")])
        rb.record_roll_message(101, 201, [("armor", "Shield")])

        import discord as _d

        live = MagicMock()
        live.fetch_message = AsyncMock(return_value=MagicMock())
        gone = MagicMock()
        gone.fetch_message = AsyncMock(side_effect=_d.HTTPException(MagicMock(), {"message": "404"}))

        bot = MagicMock()
        bot.get_channel = MagicMock(side_effect={100: live, 101: gone}.get)
        await rb.rehydrate(bot)

        # Live message: view added. Gone message: record dropped.
        assert bot.add_view.call_count == 1
        rb.reset_records_for_tests()
        assert rb._message_pairs(200) is not None
        assert rb._message_pairs(201) is None

    async def test_rehydrated_view_labels_follow_state(self):
        item_state.consume([item_state.Item(name="Sword", table="weapons")])
        rb.record_roll_message(100, 200, [("weapons", "Sword")])


        live = MagicMock()
        live.fetch_message = AsyncMock(return_value=MagicMock())
        bot = MagicMock()
        bot.get_channel = MagicMock(return_value=live)
        await rb.rehydrate(bot)

        view = bot.add_view.call_args.args[0]
        assert _buttons(view)[0].label.startswith("↩️")


# ---------------------------------------------------------------------------
# Sibling parsing (rebuild after component edits)
# ---------------------------------------------------------------------------

class TestSiblingParsing:
    def test_parses_only_our_buttons(self):
        ours = MagicMock()
        ours.custom_id = rb.encode_custom_id("weapons", "Sword")
        foreign = MagicMock()
        foreign.custom_id = "someone-else"
        msg = _fake_message([ours, foreign], author=MagicMock())
        assert rb._sibling_pairs(msg) == [("weapons", "Sword")]

    def test_empty_when_no_components(self):
        msg = _fake_message([], author=MagicMock())
        assert rb._sibling_pairs(msg) == []
