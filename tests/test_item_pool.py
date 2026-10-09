"""Tests for /roll_items: pool state, the consume flag, search, and stats."""
from __future__ import annotations

import csv
import json
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot_core import item_search, item_state, item_tables
from commands.item_commands import (
    _resolve_target,
    handle_item_exclude_command,
    handle_item_include_command,
    handle_item_reset_rolls_command,
    handle_item_search_command,
    handle_item_stats_command,
)
from commands.roll_items_command import handle_roll_items_command


# ── helpers / fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def items_dir(tmp_path, monkeypatch) -> pathlib.Path:
    """Point settings.ITEMS_DIR at an empty temp dir."""
    d = tmp_path / "items"
    d.mkdir()
    from config import settings
    from bot_core import roll_buttons
    monkeypatch.setattr(settings, "ITEMS_DIR", d)
    roll_buttons.reset_records_for_tests()
    return d


def write_table(root: pathlib.Path, name: str, rows: list[list[str]], header=None) -> None:
    header = header or ["name", "url", "rarity", "notes"]
    with (root / name).open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def make_ix() -> MagicMock:
    """Mock Interaction capturing content AND embed kwargs on followup.send."""
    calls: list[dict] = []

    async def on_send(content=None, *, embed=None, view=None, ephemeral=False, **kwargs):
        calls.append({"content": content, "embed": embed, "view": view})
        msg = MagicMock()
        msg.id = 1
        msg.channel.id = 1
        return msg

    ix = MagicMock()
    ix.followup.send.side_effect = on_send
    ix.response.defer = AsyncMock()
    ix._calls = calls
    return ix


def state_file(root: pathlib.Path) -> pathlib.Path:
    return root / item_state.STATE_FILENAME


# ── pool state ─────────────────────────────────────────────────────────────

class TestItemState:
    def _items(self, n: int, table: str = "t", rarity: str = "common") -> list[item_tables.Item]:
        return [
            item_tables.Item(name=f"i{k}", url="", rarity=rarity, notes="", table=table)
            for k in range(n)
        ]

    def test_consume_persists_across_reload(self, items_dir):
        items = self._items(3)
        item_state.consume(items[:2])
        assert item_state.load_state() == {"t": ["i0", "i1"]}
        available, total = item_state.exclude_consumed(items)
        assert [i.name for i in available] == ["i2"] and total == 3

    def test_consume_idempotent(self, items_dir):
        items = self._items(2)
        item_state.consume(items)
        item_state.consume(items)
        assert item_state.load_state() == {"t": ["i0", "i1"]}

    def test_exclude_empty_when_no_state(self, items_dir):
        items = self._items(4)
        available, total = item_state.exclude_consumed(items)
        assert len(available) == 4 and total == 4

    def test_reset_all_and_one_table(self, items_dir):
        a, b = self._items(2, table="a"), self._items(3, table="b")
        item_state.consume(a + b)
        assert item_state.reset("b") == 3
        assert item_state.load_state() == {"a": ["i0", "i1"]}
        assert item_state.reset() == 2
        assert item_state.load_state() == {}

    def test_reset_unknown_table_releases_nothing(self, items_dir):
        item_state.consume(self._items(2))
        assert item_state.reset("nope") == 0

    def test_prune_drops_stale_entries_only_for_present_tables(self, items_dir):
        a, b = self._items(3, table="a"), self._items(2, table="b")
        item_state.consume(a + b)
        # Table "a" lost row i1; pruning with only "a"'s current rows must not
        # touch table "b"'s bucket.
        changed = item_state.prune([a[0], a[2]])
        assert changed is True
        state = item_state.load_state()
        assert state["a"] == ["i0", "i2"] and state["b"] == ["i0", "i1"]

    def test_corrupt_state_file_degrades_to_empty(self, items_dir):
        state_file(items_dir).write_text("{not json", encoding="utf-8")
        assert item_state.load_state() == {}
        available, total = item_state.exclude_consumed(self._items(2))
        assert len(available) == 2 and total == 2


# ── search scoring ─────────────────────────────────────────────────────────

class TestItemSearch:
    def _it(self, name: str, table: str = "t") -> item_tables.Item:
        return item_tables.Item(name=name, url="", rarity="common", notes="", table=table)

    def test_scoring_tiers(self):
        items = [self._it("🎒 Bag of Holding"), self._it("Bag of Tricks"),
                 self._it("Magic Baguette")]
        results = item_search.search_items("bag", items)
        assert [i.name for i, _ in results] == [
            "🎒 Bag of Holding", "Bag of Tricks", "Magic Baguette"]
        assert [s for _, s in results] == [0.9, 0.9, 0.8]
        # Exact match tops everything.
        (exact,) = item_search.search_items("bag of tricks", items)
        assert exact[1] == 1.0

    def test_emote_prefix_invisible(self):
        (match,) = item_search.search_items("bag of holding", [self._it("🎒 Bag of Holding")])
        assert match[1] == 1.0

    def test_fuzzy_ratio_tier(self):
        # Typo: not a substring, so it falls to the difflib tier (ratio 0.89).
        items = [self._it("Bag of Holding"), self._it("Totally Unrelated Sword")]
        results = item_search.search_items("bag of poldin", items)
        assert len(results) == 1 and 0.6 <= results[0][1] < 0.9

    def test_below_threshold_dropped(self):
        assert item_search.search_items("zzz qqq", [self._it("Bag of Holding")]) == []

    def test_empty_query_and_limit(self):
        items = [self._it(f"item {k}") for k in range(5)]
        assert item_search.search_items("", items) == []
        assert len(item_search.search_items("item", items, limit=2)) == 2

    def test_deterministic_tie_break(self):
        items = [self._it("Zeta Ring"), self._it("Alpha Ring")]
        results = item_search.search_items("ring", items)
        assert [i.name for i, _ in results] == ["Alpha Ring", "Zeta Ring"]


# ── /roll_items v2 handler paths ───────────────────────────────────────────

class TestRollItemsV2:
    def test_default_roll_touches_no_state(self, items_dir):
        # consume off (default): pure random from the full table, no tracking.
        write_table(items_dir, "loot.csv", [[f"Item {k}", "", "common", ""] for k in range(5)])
        ix1 = make_ix()
        import asyncio
        asyncio.run(handle_roll_items_command(ix1, table="loot", count=3))
        (call1,) = ix1._calls
        assert call1["embed"] is not None and len(call1["embed"].fields) == 3
        assert not state_file(items_dir).exists()  # nothing marked
        footer = call1["embed"].footer.text if call1["embed"].footer else ""
        assert "remaining" not in footer

    def test_remaining_clause_in_footer(self, items_dir):
        write_table(items_dir, "loot.csv", [[f"Item {k}", "", "common", ""] for k in range(5)])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_roll_items_command(ix, table="loot", count=2, consume=True))
        (call,) = ix._calls
        assert "3 of 5 remaining in loot" in call["embed"].footer.text

    def test_consume_excludes_and_marks(self, items_dir):
        write_table(items_dir, "tiny.csv", [["Only", "", "common", ""]])
        import asyncio
        # A consuming roll marks the item…
        ix1 = make_ix()
        asyncio.run(handle_roll_items_command(ix1, table="tiny", count=1, consume=True))
        (call1,) = ix1._calls
        assert call1["embed"] is not None and "Only" in call1["embed"].fields[0].name
        # …so the next consuming roll finds an exhausted pool…
        ix2 = make_ix()
        asyncio.run(handle_roll_items_command(ix2, table="tiny", count=1, consume=True))
        (call2,) = ix2._calls
        assert call2["embed"] is None and "Pool exhausted" in call2["content"]
        # …but a plain roll ignores the tracking and still returns it.
        ix3 = make_ix()
        asyncio.run(handle_roll_items_command(ix3, table="tiny", count=1))
        (call3,) = ix3._calls
        assert call3["embed"] is not None and "Only" in call3["embed"].fields[0].name

    def test_exhausted_message_points_at_reset(self, items_dir):
        write_table(items_dir, "tiny.csv", [["Only", "", "common", ""]])
        import asyncio
        asyncio.run(handle_roll_items_command(make_ix(), table="tiny", count=1, consume=True))
        ix = make_ix()
        asyncio.run(handle_roll_items_command(ix, table="tiny", count=1, consume=True))
        (call,) = ix._calls
        assert "/reset_rolls" in call["content"]

    def test_rarity_filter(self, items_dir):
        write_table(items_dir, "loot.csv", [
            ["C1", "", "common", ""], ["C2", "", "common", ""],
            ["R1", "", "rare", ""], ["R2", "", "rare", ""],
        ])
        import asyncio
        ix = make_ix()
        asyncio.run(handle_roll_items_command(ix, table="loot", count=2,
                                              rarity="rare", consume=True))
        (call,) = ix._calls
        names = {f.name.split("**")[1] for f in call["embed"].fields}
        assert names <= {"R1", "R2"} and len(names) == 2
        # Remaining clause is scoped to the filtered subset, not the whole table.
        assert "remaining in loot (rare)" in call["embed"].footer.text

    def test_rarity_unknown_lists_available(self, items_dir):
        write_table(items_dir, "loot.csv", [["C1", "", "common", ""], ["V1", "", "very rare", ""]])
        import asyncio
        ix = make_ix()
        asyncio.run(handle_roll_items_command(ix, table="loot", count=1, rarity="legendary"))
        (call,) = ix._calls
        assert call["embed"] is None and "`legendary`" in call["content"]
        assert "common" in call["content"] and "very rare" in call["content"]

    def test_rarity_ignored_with_cr(self, items_dir):
        write_table(items_dir, "loot.csv", [
            ["C1", "", "common", ""], ["R1", "", "rare", ""], ["R2", "", "rare", ""],
        ])
        (items_dir / item_tables.CR_TIERS_FILENAME).write_text(
            "min_cr,rarities\n0,common:0-0;rare:1-2\n", encoding="utf-8")
        import asyncio
        ix = make_ix()
        asyncio.run(handle_roll_items_command(ix, table="loot", cr="1", rarity="common"))
        (call,) = ix._calls
        # cr wins: rare items come up despite the common filter.
        assert any("rare" in f.name for f in call["embed"].fields)


# ── utility command handlers ───────────────────────────────────────────────

class TestUtilityCommands:
    def test_search_embed_lists_matches(self, items_dir):
        write_table(items_dir, "loot.csv", [
            ["🎒 Bag of Holding", "https://x/bag", "uncommon", ""],
            ["Bag of Tricks", "https://x/tricks", "uncommon", ""],
            ["Sword of Fire", "", "rare", ""],
        ])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_search_command(ix, query="bag"))
        (call,) = ix._calls
        names = [f.name for f in call["embed"].fields]
        assert any("Bag of Holding" in n for n in names)
        assert any("Bag of Tricks" in n for n in names)
        assert not any("Sword" in n for n in names)

    def test_search_single_table_named_once(self, items_dir):
        # All matches from one table → named once in the title, no per-item line.
        write_table(items_dir, "loot.csv", [
            ["Bag of Holding", "https://x/bag", "uncommon",
             "Carries up to 500 lb of gear."],
            ["Bag of Tricks", "", "uncommon", ""],
        ])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_search_command(ix, query="bag"))
        (call,) = ix._calls
        assert "in `loot`" in call["embed"].title
        values = [f.value for f in call["embed"].fields]
        # Short description is listed so the user need not open the link.
        assert any("Carries up to 500 lb of gear." in v for v in values)
        for v in values:
            assert "table:" not in v

    def test_search_mixed_tables_keep_per_item_line(self, items_dir):
        # Matches from two tables → per-item table lines stay.
        write_table(items_dir, "loot.csv",
                    [["Bag of Holding", "", "uncommon", "Holds 500 lb."]])
        write_table(items_dir, "arcane.csv", [["Bag of Tricks", "", "uncommon", ""]])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_search_command(ix, query="bag"))
        (call,) = ix._calls
        assert "in all tables" in call["embed"].title
        values = [f.value for f in call["embed"].fields]
        assert any("table: `loot`" in v for v in values)
        assert any("table: `arcane`" in v for v in values)

    def test_search_no_matches_friendly(self, items_dir):
        write_table(items_dir, "loot.csv", [["Sword of Fire", "", "rare", ""]])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_search_command(ix, query="zzz"))
        (call,) = ix._calls
        assert call["embed"] is None and "No items matching" in call["content"]
        assert call["view"] is None  # nothing to toggle

    def test_search_results_get_toggle_buttons(self, items_dir):
        from bot_core import roll_buttons as rb
        write_table(items_dir, "loot.csv", [["Bag of Holding", "", "uncommon", ""]])
        write_table(items_dir, "arcane.csv", [["Bag of Tricks", "", "uncommon", ""]])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_search_command(ix, query="bag"))
        (call,) = ix._calls
        view = call["view"]
        assert view is not None and view.timeout is None  # persistent
        buttons = list(view.children)
        assert len(buttons) == 2
        decoded = {rb.decode_custom_id(b.custom_id) for b in buttons}
        assert decoded == {("loot", "Bag of Holding"), ("arcane", "Bag of Tricks")}
        # Recorded so the view can be re-attached after a restart.
        assert set(rb._message_pairs(1)) == {
            ("loot", "Bag of Holding"),
            ("arcane", "Bag of Tricks"),
        }

    def test_stats_ignores_stale_state_entries(self, items_dir):
        # State for a row that no longer exists in the CSV must not count.
        write_table(items_dir, "loot.csv", [["C1", "", "common", ""]])
        (state_file(items_dir)).write_text(
            json.dumps({"loot": ["C1", "Deleted Row"]}), encoding="utf-8")
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_stats_command(ix))
        (call,) = ix._calls
        value = call["embed"].fields[0].value
        # Stale entry not counted: 1 consumed (not 2), remaining 0 (not negative).
        assert "1 rolled out" in value and "0 left" in value

    def test_stats_counts_include_consumed(self, items_dir):
        write_table(items_dir, "loot.csv", [
            ["C1", "", "common", ""], ["C2", "", "common", ""], ["R1", "", "rare", ""],
        ])
        item_state.consume([item_tables.Item(name="C1", table="loot")])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_stats_command(ix))
        (call,) = ix._calls
        value = call["embed"].fields[0].value
        assert "3 total" in value and "1 rolled out" in value and "2 left" in value
        assert "common 2" in value and "rare 1" in value

    def test_reset_reports_count(self, items_dir):
        write_table(items_dir, "loot.csv", [[f"I{k}", "", "common", ""] for k in range(4)])
        item_state.consume([item_tables.Item(name=f"I{k}", table="loot") for k in range(3)])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_reset_rolls_command(ix, table="loot"))
        (call,) = ix._calls
        assert "3 item(s) returned to loot" in call["content"]
        assert item_state.load_state() == {}

    def test_reset_already_fresh(self, items_dir):
        write_table(items_dir, "loot.csv", [["I0", "", "common", ""]])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_reset_rolls_command(ix, table="loot"))
        (call,) = ix._calls
        assert "already fresh" in call["content"]

    def test_reset_unknown_table(self, items_dir):
        write_table(items_dir, "loot.csv", [["I0", "", "common", ""]])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_reset_rolls_command(ix, table="nope"))
        (call,) = ix._calls
        assert call["embed"] is None and "No table named" in call["content"]


# ── release_item (single-item unmark) ─────────────────────────────────────

class TestReleaseItem:
    def _it(self, name: str, table: str = "t") -> item_tables.Item:
        return item_tables.Item(name=name, url="", rarity="common", notes="", table=table)

    def test_release_persists_across_reload(self, items_dir):
        item_state.consume([self._it("i0"), self._it("i1")])
        assert item_state.release_item("t", "i0") is True
        assert item_state.load_state() == {"t": ["i1"]}

    def test_release_idempotent(self, items_dir):
        item_state.consume([self._it("i0")])
        assert item_state.release_item("t", "i0") is True
        assert item_state.release_item("t", "i0") is False
        assert item_state.load_state() == {}

    def test_release_unknown_table_or_name_noop(self, items_dir):
        item_state.consume([self._it("i0")])
        assert item_state.release_item("nope", "i0") is False
        assert item_state.release_item("t", "zzz") is False
        assert item_state.load_state() == {"t": ["i0"]}

    def test_release_leaves_other_marks_alone(self, items_dir):
        item_state.consume([self._it("i0"), self._it("i1")])
        item_state.release_item("t", "i0")
        available, total = item_state.exclude_consumed([self._it(f"i{k}") for k in range(2)])
        assert [i.name for i in available] == ["i0"] and total == 2

    def test_release_table_case_and_suffix_insensitive(self, items_dir):
        item_state.consume([self._it("i0", table="Loot")])
        assert item_state.release_item("LOOT.csv", "i0") is True
        assert item_state.load_state() == {}

    def test_round_trip_consume_release(self, items_dir):
        it = self._it("Only")
        item_state.consume([it])
        available, _ = item_state.exclude_consumed([it])
        assert available == []
        item_state.release_item("t", "Only")
        available, _ = item_state.exclude_consumed([it])
        assert [i.name for i in available] == ["Only"]


# ── /item_exclude + /item_include ─────────────────────────────────────────

class TestMarkCommands:
    def test_resolve_single_strong_candidate(self):
        items = [
            item_tables.Item(name="🎒 Bag of Holding", table="loot"),
            item_tables.Item(name="Sword of Fire", table="loot"),
        ]
        result = _resolve_target("bag", items)
        assert result["status"] == "single"
        assert result["item"].name == "🎒 Bag of Holding"

    def test_resolve_exact_match(self):
        items = [item_tables.Item(name="Bag of Tricks", table="loot")]
        result = _resolve_target("bag of tricks", items)
        assert result["status"] == "single"

    def test_resolve_multiple_candidates_ambiguous(self):
        items = [
            item_tables.Item(name="Bag of Holding", table="loot"),
            item_tables.Item(name="Bag of Tricks", table="loot"),
        ]
        result = _resolve_target("bag", items)
        assert result["status"] == "ambiguous"
        assert len(result["candidates"]) == 2

    def test_resolve_exact_match_outranks_fuzzy_siblings(self):
        # Real-world regression: the magic-items table has difflib≥0.6
        # neighbours for almost every name, so an exact-name query returns
        # many matches — the unique exact match must still act.
        items = [
            item_tables.Item(name="🎒 Bag of Holding", table="loot"),
            item_tables.Item(name="💍 Ring of Mind Shielding", table="loot"),
            item_tables.Item(name="🧹 Broom of Flying", table="loot"),
            item_tables.Item(name="🕊️ Wings of Flying", table="loot"),
        ]
        result = _resolve_target("Bag of Holding", items)
        assert result["status"] == "single"
        assert result["item"].name == "🎒 Bag of Holding"

    def test_resolve_prefix_outranks_fuzzy_siblings(self):
        items = [
            item_tables.Item(name="Wand of Secrets", table="loot"),
            item_tables.Item(name="Wand of Web", table="loot"),  # fuzzy ~0.86
        ]
        result = _resolve_target("wand of se", items)
        assert result["status"] == "single"
        assert result["item"].name == "Wand of Secrets"

    def test_resolve_weak_fuzzy_never_acts(self):
        # Typo matches only via difflib ratio (< 0.8) → shortlist, no action.
        items = [item_tables.Item(name="Bag of Holding", table="loot")]
        result = _resolve_target("bag of poldin", items)
        assert result["status"] == "ambiguous"

    def test_resolve_no_match(self):
        items = [item_tables.Item(name="Sword of Fire", table="loot")]
        assert _resolve_target("zzz qqq", items)["status"] == "none"

    def test_exclude_marks_item(self, items_dir):
        write_table(items_dir, "loot.csv", [
            ["🎒 Bag of Holding", "", "uncommon", ""],
            ["Sword of Fire", "", "rare", ""],
        ])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_exclude_command(ix, name="bag of holding"))
        (call,) = ix._calls
        assert "Bag of Holding" in call["content"] and "/item_include" in call["content"]
        assert item_state.load_state() == {"loot": ["🎒 Bag of Holding"]}

    def test_exclude_already_marked(self, items_dir):
        write_table(items_dir, "loot.csv", [["Bag of Holding", "", "uncommon", ""]])
        item_state.consume([item_tables.Item(name="Bag of Holding", table="loot")])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_exclude_command(ix, name="bag of holding"))
        (call,) = ix._calls
        assert "already marked as used" in call["content"]

    def test_exclude_ambiguous_lists_candidates(self, items_dir):
        write_table(items_dir, "loot.csv", [
            ["Bag of Holding", "", "uncommon", ""],
            ["Bag of Tricks", "", "uncommon", ""],
        ])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_exclude_command(ix, name="bag"))
        (call,) = ix._calls
        assert "exact name" in call["content"]
        assert "Bag of Holding" in call["content"] and "Bag of Tricks" in call["content"]
        assert item_state.load_state() == {}  # nothing marked

    def test_exclude_no_match(self, items_dir):
        write_table(items_dir, "loot.csv", [["Sword of Fire", "", "rare", ""]])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_exclude_command(ix, name="zzz qqq"))
        (call,) = ix._calls
        assert "No items matching" in call["content"]

    def test_exclude_unknown_table(self, items_dir):
        write_table(items_dir, "loot.csv", [["Bag of Holding", "", "uncommon", ""]])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_exclude_command(ix, name="bag", table="nope"))
        (call,) = ix._calls
        assert "No table named `nope`" in call["content"] and "`loot`" in call["content"]

    def test_include_releases_only_that_item(self, items_dir):
        write_table(items_dir, "loot.csv", [
            ["Bag of Holding", "", "uncommon", ""],
            ["Sword of Fire", "", "rare", ""],
        ])
        item_state.consume([
            item_tables.Item(name="Bag of Holding", table="loot"),
            item_tables.Item(name="Sword of Fire", table="loot"),
        ])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_include_command(ix, name="bag of holding"))
        (call,) = ix._calls
        assert "back into the pool" in call["content"]
        assert item_state.load_state() == {"loot": ["Sword of Fire"]}

    def test_include_unmarked_noop(self, items_dir):
        write_table(items_dir, "loot.csv", [["Bag of Holding", "", "uncommon", ""]])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_item_include_command(ix, name="bag of holding"))
        (call,) = ix._calls
        assert "wasn't marked" in call["content"]

    def test_manual_mark_survives_nothing_else_and_reset_clears(self, items_dir):
        # Manual marks live in the consumed state: /reset_rolls clears them too.
        write_table(items_dir, "loot.csv", [["Bag of Holding", "", "uncommon", ""]])
        import asyncio
        asyncio.run(handle_item_exclude_command(make_ix(), name="bag of holding"))
        ix = make_ix()
        asyncio.run(handle_item_reset_rolls_command(ix, table="loot"))
        (call,) = ix._calls
        assert "1 item(s) returned" in call["content"]
        assert item_state.load_state() == {}


# ── default table resolution ───────────────────────────────────────────────

class TestDefaultTable:
    def test_omitted_table_uses_default(self, items_dir, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "DEFAULT_ITEM_TABLE", "loot")
        write_table(items_dir, "loot.csv", [[f"Item {k}", "", "common", ""] for k in range(5)])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_roll_items_command(ix, count=2, consume=True))
        (call,) = ix._calls
        assert call["embed"] is not None and len(call["embed"].fields) == 2
        assert "remaining in loot" in call["embed"].footer.text

    def test_blank_table_falls_back_to_default(self, items_dir, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "DEFAULT_ITEM_TABLE", "loot")
        write_table(items_dir, "loot.csv", [["A", "", "common", ""]])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_roll_items_command(ix, table="   ", count=1))
        (call,) = ix._calls
        assert call["embed"] is not None and "A" in call["embed"].fields[0].name

    def test_missing_default_lists_available_tables(self, items_dir, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "DEFAULT_ITEM_TABLE", "nope")
        write_table(items_dir, "loot.csv", [["A", "", "common", ""]])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_roll_items_command(ix, count=1))
        (call,) = ix._calls
        assert call["embed"] is None
        assert "No table named `nope`" in call["content"] and "`loot`" in call["content"]

    def test_consume_only_marks_the_rolled_table(self, items_dir):
        write_table(items_dir, "a.csv", [["A0", "", "common", ""], ["A1", "", "common", ""]])
        write_table(items_dir, "b.csv", [["B0", "", "common", ""], ["B1", "", "common", ""]])
        ix = make_ix()
        import asyncio
        asyncio.run(handle_roll_items_command(ix, table="a", count=2, consume=True))
        (call,) = ix._calls
        assert call["embed"] is not None and len(call["embed"].fields) == 2
        state = item_state.load_state()
        assert set(state.get("a", [])) == {"A0", "A1"}  # rolled table marked
        assert "b" not in state  # other tables untouched
