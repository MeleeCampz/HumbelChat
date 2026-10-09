"""Item tables for /roll_items — CSV loading, CR tiers, and random rolling.

Pure, unit-testable: no Discord, no AI. Data lives in ``ITEMS_DIR``
(default ``data/items/``) as one CSV per table plus CR tier files:

- item table columns:  name (required), url, rarity, notes, price_gp (optional)
- cr tier columns:     min_cr, rarities — where ``rarities`` holds
                       per-rarity count ranges, e.g. ``common:3-4;rare:1-2``

Tier files come in two flavors: the shared ``cr_tiers.csv``, and optional
per-table curves ``<table>_tiers.csv`` (e.g. ``gems_tiers.csv``) that take
precedence for that table — useful when a table's "rarities" are value tiers
(``10 gp``, ``250 gp``, …) with their own CR breakpoints. Any file ending in
``_tiers.csv`` is reserved and never offered as a rollable table.

Tier selection is a step function: the tier with the **highest** ``min_cr``
that is ≤ the rolled CR wins (CR below every tier uses the first one). A
single ``min_cr`` per row makes overlaps and gaps impossible by construction.

Parsing is deliberately tolerant: BOM/CRLF OK, whitespace trimmed, missing
optional columns OK, unknown extra columns ignored. Rows without a ``name``
are skipped with a warning log. A blank rarity normalizes to ``"common"`` so
mundane items still participate in CR-scaled rolls.
"""
from __future__ import annotations

import csv
import logging
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("bot.item_tables")

#: Filename of the shared CR tier table (fallback for every table).
CR_TIERS_FILENAME = "cr_tiers.csv"

#: Suffix reserved for per-table CR tier files (``<table>_tiers.csv``).
TIERS_SUFFIX = "_tiers.csv"


class NoCrTiersError(LookupError):
    """A CR-scaled roll was requested but no cr_tiers.csv exists/is valid."""


@dataclass(frozen=True)
class Item:
    """One row of an item-table CSV."""
    name: str
    url: str = ""
    rarity: str = ""
    notes: str = ""
    table: str = ""
    price_gp: int | None = None


@dataclass(frozen=True)
class RarityRange:
    """One ``rarity:min-max`` pair from a tier's rarities column."""
    rarity: str
    min_count: int
    max_count: int


@dataclass(frozen=True)
class CrTier:
    """One row of cr_tiers.csv: minimum CR → per-rarity count ranges."""
    min_cr: float
    ranges: tuple[RarityRange, ...]


# ── CR parsing ─────────────────────────────────────────────────────────────

def parse_cr(value: str) -> float:
    """Parse a Challenge Rating: ``"2"``, ``"1/2"``, ``" 1 / 4 "`` → float.

    Raises ``ValueError`` on anything unparseable or negative (CR 0 is valid;
    D&D has CR 0 monsters, but no negative-CR ones).
    """
    text = str(value).strip()
    if not text:
        raise ValueError("empty CR")
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            numerator, denominator = float(num), float(den)
        except ValueError as exc:
            raise ValueError(f"unparseable CR {value!r}") from exc
        if denominator == 0:
            raise ValueError(f"zero denominator in CR {value!r}")
        result = numerator / denominator
    else:
        try:
            result = float(text)
        except ValueError as exc:
            raise ValueError(f"unparseable CR {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid CR {value!r}")
    return result


# ── Table discovery / item loading ────────────────────────────────────────

def _items_dir() -> Path:
    # Read at call time (not import time) so tests can monkeypatch settings.
    from config import settings
    return Path(settings.ITEMS_DIR)


def _samples_dir() -> Path:
    """Built-in sample tables (committed to the repo), see ``seed_from_samples``."""
    return Path(__file__).resolve().parent.parent / "item_samples"


def _is_tiers_file(name: str) -> bool:
    """True for any CR tier file: ``cr_tiers.csv`` or ``<table>_tiers.csv``."""
    n = name.lower()
    return n == CR_TIERS_FILENAME or n.endswith(TIERS_SUFFIX)


def _has_tables(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any(
        p.is_file() and p.suffix.lower() == ".csv" and not _is_tiers_file(p.name)
        for p in root.iterdir()
    )


def seed_from_samples() -> bool:
    """Copy the built-in sample tables into ``ITEMS_DIR`` on first use.

    Runs only while the user directory holds no rollable tables; existing
    files are never overwritten, so in-place edits survive. Returns True if
    anything was copied. Failures (missing samples, unwritable dir) are
    logged and swallowed — the command then reports "no tables" as usual.
    """
    dest = _items_dir()
    src = _samples_dir()
    if _has_tables(dest) or not src.is_dir():
        return False
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("item_tables: cannot create %s: %s", dest, exc)
        return False
    copied = False
    for f in sorted(src.iterdir()):
        if not (f.is_file() and f.suffix.lower() == ".csv"
                and not (dest / f.name).exists()):
            continue
        try:
            shutil.copy2(f, dest / f.name)
            copied = True
        except OSError as exc:
            log.warning("item_tables: cannot copy %s to %s: %s", f.name, dest, exc)
    if copied:
        log.info("item_tables: seeded sample tables into %s", dest)
    return copied


def list_tables() -> list[str]:
    """Sorted rollable table names (CSV filenames without suffix).

    Tier files (``cr_tiers.csv``, ``*_tiers.csv``) are excluded; a missing
directory yields an empty list.
    """
    root = _items_dir()
    if not root.is_dir():
        return []
    tables = [
        p.stem for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() == ".csv" and not _is_tiers_file(p.name)
    ]
    return tables


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into trimmed string dicts keyed by the header row."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, str]] = []
        for raw in reader:
            if raw is None:
                continue
            # DictReader files overflow columns under a None key as a list — drop it.
            rows.append({
                k.strip().lower(): v.strip()
                for k, v in raw.items()
                if k is not None and isinstance(v, str)
            })
    return rows


def load_items(table: str | None = None) -> list[Item]:
    """Load items from one table (case-insensitive, ``.csv`` optional) or all.

    Raises ``LookupError`` when a named table does not exist. Invalid rows
    (missing name) are skipped with a warning log.
    """
    root = _items_dir()
    if table is None:
        paths = [root / f"{name}.csv" for name in list_tables()]
    else:
        wanted = str(table).strip().lower().removesuffix(".csv")
        match = next(
            (t for t in list_tables() if t.lower() == wanted), None
        )
        if match is None:
            raise LookupError(wanted)
        paths = [root / f"{match}.csv"]

    items: list[Item] = []
    for path in paths:
        table_name = path.stem
        for row in _read_csv_rows(path):
            name = row.get("name", "")
            if not name:
                log.warning("item_tables: skipping row without name in %s: %r", path.name, row)
                continue
            rarity = (row.get("rarity") or "common").lower()
            raw_price = row.get("price_gp", "")
            price_gp = _parse_price(raw_price)
            if raw_price and price_gp is None:
                log.warning(
                    "item_tables: ignoring unparseable price %r for %r in %s",
                    raw_price, name, path.name,
                )
            items.append(
                Item(
                    name=name,
                    url=row.get("url", ""),
                    rarity=rarity,
                    notes=row.get("notes", ""),
                    table=table_name,
                    price_gp=price_gp,
                )
            )
    return items


def _parse_price(raw: str) -> int | None:
    """Parse an optional ``price_gp`` column: ``"100"``, ``"2,500 gp"`` → int.

    Blank → ``None`` (item has no price). Garbage or negative → ``None``
    (the caller logs a warning).
    """
    text = str(raw).strip().replace(",", "").replace("_", "")
    if not text:
        return None
    if text.lower().endswith("gp"):
        text = text[:-2].strip()
    try:
        value = int(float(text))
    except ValueError:
        return None
    return value if value >= 0 else None


# ── CR tiers ───────────────────────────────────────────────────────────────

def _parse_rarity_ranges(raw: str) -> list[RarityRange]:
    """Parse ``common:3-4;rare:1-2`` → RarityRange list.

    Raises ``ValueError`` on any malformed pair (missing ``:``, bad numbers,
    inverted range, blank rarity name).
    """
    ranges: list[RarityRange] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        name, sep, rng = part.partition(":")
        if not sep:
            raise ValueError(f"missing ':' in {part!r}")
        lo_s, dash, hi_s = rng.partition("-")
        if not dash:
            raise ValueError(f"missing '-' in {part!r}")
        try:
            lo = int(float(lo_s))
            hi = int(float(hi_s))
        except ValueError as exc:
            raise ValueError(f"bad count range in {part!r}") from exc
        name = name.strip().lower()
        if not name or lo < 0 or hi < lo:
            raise ValueError(f"invalid pair {part!r}")
        ranges.append(RarityRange(name, lo, hi))
    return ranges


def tier_filename(table: str) -> str:
    """Per-table tier filename for a rollable table (``gems`` → ``gems_tiers.csv``).

    Case-insensitive, trailing ``.csv`` tolerated — mirrors ``load_items``.
    """
    return f"{str(table).strip().lower().removesuffix('.csv')}{TIERS_SUFFIX}"


def load_cr_tiers(table: str | None = None) -> list[CrTier]:
    """Parse CR tiers sorted by ``min_cr``. Missing file → [].

    When ``table`` is given, the per-table curve ``<table>_tiers.csv`` takes
    precedence; when it doesn't exist (or no table is given), the shared
    ``cr_tiers.csv`` is used.
    """
    root = _items_dir()
    if table:
        path = root / tier_filename(table)
        if path.is_file():
            return _parse_cr_tiers(path)
    path = root / CR_TIERS_FILENAME
    if not path.is_file():
        return []
    return _parse_cr_tiers(path)


def _parse_cr_tiers(path: Path) -> list[CrTier]:
    """Parse one tier CSV file into a sorted list of ``CrTier``."""
    tiers: list[CrTier] = []
    for row in _read_csv_rows(path):
        try:
            min_cr = parse_cr(row.get("min_cr", ""))
            ranges = _parse_rarity_ranges(row.get("rarities", ""))
        except ValueError as exc:
            log.warning("item_tables: skipping bad %s row %r: %s", path.name, row, exc)
            continue
        if not ranges:
            log.warning("item_tables: skipping %s row without rarities %r", path.name, row)
            continue
        tiers.append(CrTier(min_cr, tuple(ranges)))
    tiers.sort(key=lambda t: t.min_cr)
    return tiers


def resolve_tier(cr: float, tiers: list[CrTier]) -> CrTier:
    """Highest tier with ``min_cr <= cr``; CR below every tier → first tier.

    Raises ``LookupError`` when ``tiers`` is empty.
    """
    if not tiers:
        raise LookupError("no CR tiers configured")
    chosen = tiers[0]
    for tier in tiers:  # sorted ascending by min_cr
        if tier.min_cr <= cr:
            chosen = tier
    return chosen


# ── Rolling ────────────────────────────────────────────────────────────────

def roll_items(
    items: list[Item],
    count: int,
    rng: random.Random | None = None,
) -> tuple[list[Item], bool]:
    """Sample ``count`` items **without replacement** (rolls never repeat).

    Returns ``(rolled, exhausted)``; ``exhausted`` is True when the table had
    no more than ``count`` items (everything is returned, shuffled).
    Raises ``ValueError`` for an empty pool or non-positive count.
    """
    if not items:
        raise ValueError("no items to roll from")
    if count < 1:
        raise ValueError("count must be >= 1")
    rng = rng or random.Random()
    n = min(count, len(items))
    exhausted = count >= len(items)
    return rng.sample(items, n), exhausted


def roll_for_cr(
    cr: float,
    items: list[Item],
    tiers: list[CrTier],
    rng: random.Random | None = None,
) -> tuple[list[Item], CrTier, bool]:
    """CR-scaled roll: pick tier, then roll each rarity's range independently.

    For every ``rarity:min-max`` pair in the tier, a count is drawn uniformly
    in [min, max] (clamped to the number of available items of that rarity) and
    sampled **without replacement** from that rarity's pool. Rarities with no
    matching items are skipped.

    Returns ``(rolled, tier, exhausted)`` where ``exhausted`` means at least
    one rarity had fewer items than its drawn count. Raises
    ``NoCrTiersError`` when no tiers are configured and ``ValueError`` when
    nothing could be rolled at all.
    """
    if not tiers:
        raise NoCrTiersError("no CR tiers configured")
    rng = rng or random.Random()
    tier = resolve_tier(cr, tiers)
    rolled: list[Item] = []
    exhausted = False
    for rr in tier.ranges:
        pool = [i for i in items if i.rarity.lower() == rr.rarity]
        if not pool:
            continue
        wanted = rng.randint(rr.min_count, rr.max_count)
        if wanted > len(pool):
            exhausted = True
        n = max(0, min(wanted, len(pool)))
        rolled.extend(rng.sample(pool, n))
    if not rolled:
        want = ", ".join(f"{r.rarity} {r.min_count}-{r.max_count}" for r in tier.ranges)
        raise ValueError(f"no available items for this roll (tier wants: {want})")
    return rolled, tier, exhausted
