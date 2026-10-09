"""Tests for /roll_items: CSV parsing, CR tiers, rolling, and the command handler."""
from __future__ import annotations

import csv
import pathlib
import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot_core import item_tables
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

    async def on_send(content=None, *, embed=None, view=None, ephemeral=False):
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


# ── CSV parsing ────────────────────────────────────────────────────────────

class TestCsvParsing:
    def test_normal_rows(self, items_dir):
        write_table(items_dir, "t.csv", [
            ["Sword", "https://dndbeyond.com/x", "rare", "shiny"],
            ["Potion", "", "", ""],
        ])
        items = item_tables.load_items("t")
        assert [i.name for i in items] == ["Sword", "Potion"]
        assert items[0].url == "https://dndbeyond.com/x"
        assert items[0].rarity == "rare"
        assert items[0].notes == "shiny"
        assert items[0].table == "t"

    def test_bom_and_crlf(self, items_dir):
        (items_dir / "bom.csv").write_bytes(
            "name,rarity\r\nSceptre,epic\r\n".encode("utf-8-sig")
        )
        items = item_tables.load_items("bom")
        assert len(items) == 1 and items[0].name == "Sceptre"

    def test_missing_optional_columns(self, items_dir):
        write_table(items_dir, "min.csv", [["Just A Name"]], header=["name"])
        items = item_tables.load_items("min")
        assert len(items) == 1
        assert items[0].rarity == "common"  # blank rarity normalizes

    def test_unknown_extra_columns_ignored(self, items_dir):
        write_table(
            items_dir, "x.csv", [["Item", "https://u", "rare", "", "extra"]],
            header=["name", "url", "rarity", "notes", "whatever"],
        )
        assert len(item_tables.load_items("x")) == 1

    def test_row_overflow_beyond_header_tolerated(self, items_dir):
        # Unquoted commas in a cell produce more values than header columns;
        # DictReader parks the overflow under a None key — must not crash.
        (items_dir / "ovf.csv").write_text(
            "name,rarity\nBag of Holding,uncommon, 0 weight, holds 500 lb\n", encoding="utf-8"
        )
        items = item_tables.load_items("ovf")
        assert len(items) == 1 and items[0].name == "Bag of Holding"

    def test_row_without_name_skipped(self, items_dir, caplog):
        write_table(items_dir, "bad.csv", [["", "https://u", "", ""], ["Good", "", "", ""]])
        with caplog.at_level("WARNING"):
            items = item_tables.load_items("bad")
        assert [i.name for i in items] == ["Good"]
        assert any("without name" in r.message for r in caplog.records)

    def test_whitespace_trimmed(self, items_dir):
        (items_dir / "ws.csv").write_text("name, url\n  Spatula  ,  https://u  \n", encoding="utf-8")
        items = item_tables.load_items("ws")
        assert items[0].name == "Spatula" and items[0].url == "https://u"

    def test_missing_dir_empty(self, tmp_path, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "ITEMS_DIR", tmp_path / "nope")
        assert item_tables.list_tables() == []
        assert item_tables.load_items(None) == []


# ── Table resolution ───────────────────────────────────────────────────────

class TestTableResolution:
    def test_case_insensitive_and_suffix(self, items_dir):
        write_table(items_dir, "Magic_Items.csv", [["A", "", "", ""]])
        assert item_tables.list_tables() == ["Magic_Items"]
        assert len(item_tables.load_items("magic_items")) == 1
        assert len(item_tables.load_items("MAGIC_ITEMS.CSV")) == 1

    def test_cr_tiers_excluded_from_tables(self, items_dir):
        (items_dir / item_tables.CR_TIERS_FILENAME).write_text(
            "cr_min,cr_max,count_min,count_max,rarities\n0,2,1,1,common\n", encoding="utf-8"
        )
        write_table(items_dir, "loot.csv", [["A", "", "", ""]])
        assert item_tables.list_tables() == ["loot"]

    def test_unknown_table_raises(self, items_dir):
        write_table(items_dir, "a.csv", [["A", "", "", ""]])
        with pytest.raises(LookupError):
            item_tables.load_items("nope")

    def test_all_tables_combined(self, items_dir):
        write_table(items_dir, "a.csv", [["A1", "", "", ""]])
        write_table(items_dir, "b.csv", [["B1", "", "", ""], ["B2", "", "", ""]])
        assert sorted(i.name for i in item_tables.load_items(None)) == ["A1", "B1", "B2"]


# ── CR parsing & tiers ─────────────────────────────────────────────────────

class TestCrParsing:
    @pytest.mark.parametrize("raw,expected", [
        ("2", 2.0), (" 7 ", 7.0), ("1/2", 0.5), (" 1 / 4 ", 0.25), ("1/8", 0.125), ("0", 0.0),
    ])
    def test_valid(self, raw, expected):
        assert item_tables.parse_cr(raw) == expected

    @pytest.mark.parametrize("raw",
                             ["", "abc", "1/", "/2", "1/0", "-3", "1.5.2",
                              "nan", "inf", "-1/2"])
    def test_invalid(self, raw):
        with pytest.raises(ValueError):
            item_tables.parse_cr(raw)


class TestCrTiers:
    def _write_tiers(self, root: pathlib.Path, rows: list[list[str]]) -> None:
        with (root / item_tables.CR_TIERS_FILENAME).open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["min_cr", "rarities"])
            w.writerows(rows)

    def test_parse_sorted(self, items_dir):
        self._write_tiers(items_dir, [
            ["10", "rare:2-3;epic:1-2"],
            ["0", "common:1-2"],
        ])
        tiers = item_tables.load_cr_tiers()
        assert [t.min_cr for t in tiers] == [0.0, 10.0]
        assert tiers[1].ranges == (
            item_tables.RarityRange("rare", 2, 3),
            item_tables.RarityRange("epic", 1, 2),
        )

    def test_fraction_min_cr(self, items_dir):
        self._write_tiers(items_dir, [["1/4", "common:1-1"]])
        (tier,) = item_tables.load_cr_tiers()
        assert tier.min_cr == 0.25

    def test_single_count_pair(self, items_dir):
        # A range written as a single number is invalid — must be min-max.
        self._write_tiers(items_dir, [["0", "common:2"]])
        assert item_tables.load_cr_tiers() == []

    def test_bad_rows_skipped(self, items_dir, caplog):
        self._write_tiers(items_dir, [
            ["abc", "common:1-2"],        # unparseable min_cr
            ["0", "common"],              # missing ':'
            ["0", "common:2-1"],          # inverted range
            ["0", "rare:x-y"],            # bad numbers
            ["0", ":1-2"],                # blank rarity name
            ["0", ""],                    # no rarities at all
            ["3", "common:3-4;rare:1-2"], # good
        ])
        with caplog.at_level("WARNING"):
            tiers = item_tables.load_cr_tiers()
        assert len(tiers) == 1 and tiers[0].min_cr == 3.0

    def test_missing_file_empty(self, items_dir):
        assert item_tables.load_cr_tiers() == []

    def test_resolve_highest_matching_min_cr(self, items_dir):
        self._write_tiers(items_dir, [
            ["0", "common:1-2"],
            ["3", "common:3-4;rare:1-2"],
            ["14", "epic:2-3;legendary:1-2"],
        ])
        tiers = item_tables.load_cr_tiers()
        assert item_tables.resolve_tier(0.0, tiers).min_cr == 0.0
        assert item_tables.resolve_tier(2.9, tiers).min_cr == 0.0
        assert item_tables.resolve_tier(3.0, tiers).min_cr == 3.0
        assert item_tables.resolve_tier(13.0, tiers).min_cr == 3.0
        assert item_tables.resolve_tier(14.0, tiers).min_cr == 14.0
        assert item_tables.resolve_tier(99.0, tiers) is tiers[-1]  # above last stays last
        assert item_tables.resolve_tier(-1.0, tiers) is tiers[0]   # below first → first

    def test_resolve_empty_raises(self):
        with pytest.raises(LookupError):
            item_tables.resolve_tier(1.0, [])


# ── Rolling ────────────────────────────────────────────────────────────────

def _items(n: int, rarity: str = "common", prefix: str = "i") -> list[item_tables.Item]:
    return [item_tables.Item(name=f"{prefix}{k}", rarity=rarity) for k in range(n)]


class TestRollItems:
    def test_no_duplicates(self):
        rng = random.Random(42)
        rolled, exhausted = item_tables.roll_items(_items(100), 50, rng=rng)
        assert len(rolled) == 50
        assert len({i.name for i in rolled}) == 50
        assert not exhausted

    def test_exhausted_at_cap(self):
        rolled, exhausted = item_tables.roll_items(_items(3), 3, rng=random.Random(1))
        assert exhausted and {i.name for i in rolled} == {"i0", "i1", "i2"}

    def test_count_over_len_clamps(self):
        rolled, exhausted = item_tables.roll_items(_items(2), 10, rng=random.Random(1))
        assert len(rolled) == 2 and exhausted

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError):
            item_tables.roll_items([], 3)

    def test_bad_count_raises(self):
        with pytest.raises(ValueError):
            item_tables.roll_items(_items(3), 0)

    def test_deterministic_with_rng(self):
        a, _ = item_tables.roll_items(_items(10), 4, rng=random.Random(7))
        b, _ = item_tables.roll_items(_items(10), 4, rng=random.Random(7))
        assert [i.name for i in a] == [i.name for i in b]


class TestRollForCr:
    def _tiers(self, items_dir):
        with (items_dir / item_tables.CR_TIERS_FILENAME).open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["min_cr", "rarities"])
            w.writerow(["0", "common:1-2"])
            w.writerow(["3", "common:3-4;rare:1-2"])
        return item_tables.load_cr_tiers()

    def test_only_tier_rarities_roll(self, items_dir):
        tiers = self._tiers(items_dir)
        pool = _items(4, "common") + _items(4, "rare") + _items(4, "legendary")
        rolled, tier, _ = item_tables.roll_for_cr(4.0, pool, tiers, rng=random.Random(3))
        assert {r.rarity for r in tier.ranges} == {"common", "rare"}
        assert all(i.rarity in ("common", "rare") for i in rolled)

    def test_per_rarity_counts_within_ranges(self, items_dir):
        tiers = self._tiers(items_dir)
        pool = _items(20, "common") + _items(20, "rare")
        for seed in range(30):
            rolled, tier, _ = item_tables.roll_for_cr(4.0, pool, tiers, rng=random.Random(seed))
            n_common = sum(1 for i in rolled if i.rarity == "common")
            n_rare = sum(1 for i in rolled if i.rarity == "rare")
            assert 3 <= n_common <= 4
            assert 1 <= n_rare <= 2

    def test_no_duplicates_across_groups(self, items_dir):
        tiers = self._tiers(items_dir)
        pool = _items(10, "common", prefix="c") + _items(10, "rare", prefix="r")
        rolled, _, _ = item_tables.roll_for_cr(4.0, pool, tiers, rng=random.Random(5))
        assert len({i.name for i in rolled}) == len(rolled)

    def test_blank_rarity_rolls_as_common(self, items_dir):
        tiers = self._tiers(items_dir)
        pool = [item_tables.Item(name=f"mundane{k}", rarity="common") for k in range(5)]
        rolled, _, _ = item_tables.roll_for_cr(1.0, pool, tiers, rng=random.Random(3))
        assert 1 <= len(rolled) <= 2 and all(i.rarity == "common" for i in rolled)

    def test_fractional_cr_lands_in_low_tier(self, items_dir):
        tiers = self._tiers(items_dir)
        pool = _items(5, "common")
        rolled, tier, _ = item_tables.roll_for_cr(0.5, pool, tiers, rng=random.Random(3))
        assert tier.min_cr == 0.0

    def test_short_pool_clamps_and_exhausted(self, items_dir):
        tiers = self._tiers(items_dir)
        # Tier wants 3-4 common but only 1 exists → clamped to 1 + exhausted.
        pool = _items(1, "common") + _items(5, "rare")
        rolled, _, exhausted = item_tables.roll_for_cr(4.0, pool, tiers, rng=random.Random(3))
        assert sum(1 for i in rolled if i.rarity == "common") == 1 and exhausted

    def test_missing_rarity_group_skipped(self, items_dir):
        tiers = self._tiers(items_dir)
        # No rares at all → only commons roll, no error.
        pool = _items(6, "common")
        rolled, _, exhausted = item_tables.roll_for_cr(4.0, pool, tiers, rng=random.Random(3))
        assert all(i.rarity == "common" for i in rolled) and not exhausted

    def test_zero_min_range_may_roll_nothing(self, items_dir):
        (items_dir / item_tables.CR_TIERS_FILENAME).write_text(
            "min_cr,rarities\n0,rare:0-1;common:1-1\n", encoding="utf-8"
        )
        tiers = item_tables.load_cr_tiers()
        pool = _items(5, "common") + _items(5, "rare")
        for seed in range(20):
            rolled, _, _ = item_tables.roll_for_cr(1.0, pool, tiers, rng=random.Random(seed))
            n_rare = sum(1 for i in rolled if i.rarity == "rare")
            assert n_rare in (0, 1)

    def test_nothing_to_roll_raises(self, items_dir):
        tiers = self._tiers(items_dir)
        with pytest.raises(ValueError):
            item_tables.roll_for_cr(4.0, _items(5, "legendary"), tiers, rng=random.Random(3))

    def test_no_tiers_raises(self, items_dir):
        with pytest.raises(LookupError):
            item_tables.roll_for_cr(1.0, _items(3), [], rng=random.Random(3))


# ── Command handler ────────────────────────────────────────────────────────

class TestFmtCr:
    @pytest.mark.parametrize("raw,expected", [
        (2.0, "2"), (0.5, "1/2"), (0.25, "1/4"), (0.75, "3/4"),
        (6.5, "6 1/2"), (3.25, "3 1/4"), (0.125, "1/8"),
    ])
    def test_render(self, raw, expected):
        from commands.roll_items_command import _fmt_cr
        assert _fmt_cr(raw) == expected


class TestSeeding:
    """First use copies the built-in samples (item_samples/) into ITEMS_DIR."""

    def test_seed_copies_samples_when_empty(self, tmp_path, monkeypatch):
        from config import settings
        d = tmp_path / "empty"
        monkeypatch.setattr(settings, "ITEMS_DIR", d)
        # The real committed samples must be present and usable after seeding.
        assert item_tables.seed_from_samples() is True
        assert (d / "magic_items.csv").is_file()
        assert (d / item_tables.CR_TIERS_FILENAME).is_file()
        assert item_tables.list_tables() == ["magic_items"]
        items = item_tables.load_items("magic_items")
        assert len(items) >= 20
        assert all(i.url.startswith("http") for i in items)
        assert {"common", "uncommon", "rare", "very rare", "legendary"} <= {
            i.rarity for i in items}
        tiers = item_tables.load_cr_tiers()
        assert tiers and all(t.ranges for t in tiers)

    def test_seed_never_overwrites_existing(self, tmp_path, monkeypatch):
        from config import settings
        d = tmp_path / "items"
        d.mkdir()
        monkeypatch.setattr(settings, "ITEMS_DIR", d)
        # A user-edited tier file must survive seeding.
        (d / item_tables.CR_TIERS_FILENAME).write_text(
            "min_cr,rarities\n0,common:1-1\n", encoding="utf-8")
        assert item_tables.seed_from_samples() is True
        tiers = item_tables.load_cr_tiers()
        assert len(tiers) == 1 and tiers[0].min_cr == 0.0  # user's, not sample's

    def test_seed_skips_when_user_tables_exist(self, items_dir):
        write_table(items_dir, "mine.csv", [["Mine", "https://x", "common", ""]])
        assert item_tables.seed_from_samples() is False
        assert not (items_dir / "magic_items.csv").exists()
        assert item_tables.list_tables() == ["mine"]

    def test_seed_noop_when_no_samples(self, items_dir, monkeypatch, tmp_path):
        monkeypatch.setattr(item_tables, "_samples_dir",
                            lambda: tmp_path / "no-samples")
        assert item_tables.seed_from_samples() is False
        assert item_tables.list_tables() == []


class TestHandler:
    async def test_embed_contains_names_and_urls(self, items_dir):
        write_table(items_dir, "loot.csv", [
            ["Sword", "https://dndbeyond.com/sword", "rare", "shiny"],
            ["Potion", "https://dndbeyond.com/potion", "common", ""],
        ])
        ix = make_ix()
        await handle_roll_items_command(ix, table="loot", count=2)
        (call,) = ix._calls
        assert call["content"] is None and call["embed"] is not None
        embed = call["embed"]
        text = embed.to_dict()
        blob = str(text)
        assert "Sword" in blob and "https://dndbeyond.com/sword" in blob
        assert "Potion" in blob and "https://dndbeyond.com/potion" in blob

    async def test_cr_footer_rendered_and_count_ignored(self, items_dir):
        write_table(items_dir, "loot.csv", [
            ["C1", "", "common", ""], ["C2", "", "common", ""],
            ["R1", "", "rare", ""], ["R2", "", "rare", ""],
        ])
        (items_dir / item_tables.CR_TIERS_FILENAME).write_text(
            "min_cr,rarities\n0,common:2-2;rare:1-1\n", encoding="utf-8"
        )
        ix = make_ix()
        # count=1 must be ignored; the tier demands 2 common + 1 rare.
        await handle_roll_items_command(ix, table="loot", count=1, cr="1")
        (call,) = ix._calls
        embed = call["embed"]
        assert embed.footer is not None
        assert "CR 1" in embed.footer.text and "min CR 0" in embed.footer.text
        assert "common ×2" in embed.footer.text and "rare ×1" in embed.footer.text
        assert len(embed.fields) == 3

    async def test_invalid_cr_friendly(self, items_dir):
        write_table(items_dir, "loot.csv", [["A", "", "common", ""]])
        ix = make_ix()
        await handle_roll_items_command(ix, table="loot", count=1, cr="banana")
        (call,) = ix._calls
        assert call["embed"] is None and "parse CR" in call["content"]

    async def test_unknown_table_lists_available(self, items_dir):
        write_table(items_dir, "loot.csv", [["A", "", "common", ""]])
        ix = make_ix()
        await handle_roll_items_command(ix, table="nope", count=1)
        (call,) = ix._calls
        assert "No table named" in call["content"] and "`loot`" in call["content"]

    async def test_count_clamped_to_minimum(self, items_dir):
        write_table(items_dir, "loot.csv", [["A", "", "common", ""]])
        ix = make_ix()
        await handle_roll_items_command(ix, table="loot", count=0)
        (call,) = ix._calls
        assert call["embed"] is not None and len(call["embed"].fields) == 1

    async def test_fresh_dir_seeds_and_rolls(self, items_dir):
        # Empty dir → the handler seeds the samples in place, then rolls.
        ix = make_ix()
        await handle_roll_items_command(ix, table=None, count=2)
        (call,) = ix._calls
        assert (items_dir / "magic_items.csv").is_file()
        assert call["embed"] is not None and len(call["embed"].fields) == 2

    async def test_no_tables_friendly(self, items_dir, monkeypatch, tmp_path):
        # Seeding impossible (no samples anywhere) → friendly error.
        monkeypatch.setattr(item_tables, "_samples_dir",
                            lambda: tmp_path / "no-samples")
        ix = make_ix()
        await handle_roll_items_command(ix, table=None, count=1)
        (call,) = ix._calls
        assert "data/items/" in call["content"]

    async def test_cr_without_tier_file_friendly(self, items_dir):
        write_table(items_dir, "loot.csv", [["A", "", "common", ""]])
        ix = make_ix()
        await handle_roll_items_command(ix, table="loot", count=1, cr="2")
        (call,) = ix._calls
        assert call["embed"] is None and "cr_tiers.csv" in call["content"]

    async def test_exhausted_notice(self, items_dir):
        write_table(items_dir, "tiny.csv", [["Only", "", "common", ""]])
        ix = make_ix()
        await handle_roll_items_command(ix, table="tiny", count=5)
        (call,) = ix._calls
        assert call["embed"].footer is not None and "only 1 item" in call["embed"].footer.text
