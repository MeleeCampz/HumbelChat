# Plan: `/roll_items` — random item table rolls with item page links + CR scaling

> **Design change after implementation (user feedback):** `cr_tiers.csv` no longer uses
> `cr_min/cr_max/count_min/count_max/rarities`. Each tier row now has a single `min_cr`
> plus per-rarity count ranges (`common:3-4;rare:1-2`); the highest tier with
> `min_cr ≤ CR` wins, and each rarity range is rolled independently. See
> `docs/commands.md` for the current format.

## Goal (task #7)

A new slash command lets the user roll a list of randomly selected items from
item tables. Each rolled item is returned with a link to its item page
plus any extra metadata. Data lives in **CSV spreadsheets** (user decision).

Additionally, an optional **Challenge Rating (`cr`)** parameter scales the
roll: based on the monster's CR, both the **number of items** and the allowed
**rarities** change, driven by a configurable tier CSV (user decision).

No AI calls involved — deterministic local data + `random.sample`, so rolls are
instant and free.

## Data format

New directory `data/items/` — **one CSV file per table**, filename = table name
(e.g. `magic_items.csv`, `loot.csv`). Adding a new table = adding a file; no
restart needed (files are re-read on every roll).

### Item tables

Columns (header row required):

| Column   | Required | Notes                                          |
|----------|----------|------------------------------------------------|
| `name`   | yes      | Item name; rows without it are skipped + logged |
| `url`    | no       | item page link (rendered as a link)        |
| `rarity` | no       | e.g. Common / Uncommon / Rare / Epic / Legendary; **blank = treated as `common` for CR rolls** |
| `level`  | no       | Minimum level or similar tag                    |
| `notes`  | no       | Free text, shown under the item                 |

Tolerant parsing: UTF-8 with BOM OK, whitespace trimmed, missing optional
columns OK, unknown extra columns ignored.

### CR tier table — `data/items/cr_tiers.csv`

Defines how a monster's CR maps to roll size + allowed rarities. Fully
user-tunable in the spreadsheet; no code changes needed.

Columns:

| Column      | Notes                                                        |
|-------------|--------------------------------------------------------------|
| `cr_min`    | Inclusive lower bound (number or fraction like `1/4`)        |
| `cr_max`    | Inclusive upper bound (number or fraction)                   |
| `count_min` | Minimum items to roll for this tier                          |
| `count_max` | Maximum items to roll (a value is drawn uniformly in range)  |
| `rarities`  | Allowed rarities, **semicolon-separated**, case-insensitive (e.g. `common;uncommon;rare`) |

Shipped defaults (DMG-treasure-inspired, user-editable):

```csv
cr_min,cr_max,count_min,count_max,rarities
0,2,1,2,common;uncommon
3,5,2,3,common;uncommon;rare
6,9,2,4,uncommon;rare
10,13,3,4,rare;epic
14,30,3,5,epic;legendary
```

Fractional CRs (`1/8`, `1/4`, `1/2`) sort correctly (parsed as floats).
CR below the first tier → first tier; above the last → last tier.

### Starter data

- `data/items/magic_items.csv` — ~8–10 real catalog entries spanning
  common→legendary (one canonical item-page URL each) so both
  plain and CR-scaled rolls work out of the box.
- `data/items/cr_tiers.csv` — the defaults above.

## New module: `bot_core/item_tables.py` (pure, unit-testable)

- `@dataclass(frozen=True) Item`: `name`, `url`, `rarity`, `level`, `notes`, `table`.
- `@dataclass(frozen=True) CrTier`: `cr_min: float`, `cr_max: float`, `count_min: int`, `count_max: int`, `rarities: frozenset[str]`.
- `parse_cr(value: str) -> float` — accepts `"2"`, `"1/2"`, `" 1 / 4 "`; raises
  `ValueError` on garbage (handler maps to friendly error).
- `list_tables() -> list[str]` — sorted table names from CSV filenames in `ITEMS_DIR`
  (non-directories, `.csv` suffix only, excluding `cr_tiers.csv`; missing dir → empty list).
- `load_items(table: str | None = None) -> list[Item]` — read the given table or
  **all** tables. Case-insensitive table match. Invalid rows skipped with a
  warning log (`log.warning`). Blank rarity normalized to `"common"`.
- `load_cr_tiers() -> list[CrTier]` — parse `cr_tiers.csv`, sorted by `cr_min`;
  overlapping/gap warnings logged; missing file → empty list (CR rolls then fail with a friendly "no tier table" message).
- `resolve_tier(cr: float, tiers: list[CrTier]) -> CrTier` — first tier whose
  `[cr_min, cr_max]` contains `cr`; clamps to first/last tier outside the range.
- `roll_items(items, count, rng=None) -> tuple[list[Item], bool]` —
  `random.sample` **without replacement** (loot rolls must not repeat);
  returns `(rolled, exhausted)` where `exhausted` is True when `count >= len(items)`
  (all items returned, order shuffled). Deterministic with injected `rng`.
- `roll_for_cr(cr: float, items, tiers, rng=None) -> tuple[list[Item], CrTier, bool]` —
  resolve tier → filter items to allowed rarities (case-insensitive) → draw a
  count uniformly in `[count_min, count_max]` (clamped to available) → `roll_items`.
  Returns the tier used (for the reply footer) and whether fewer items were
  available than requested.

## New command: `commands/roll_items_command.py`

```
/roll_items [table] [count] [cr]
```

- `table`: optional string — case-insensitive match against CSV filenames
  (with or without `.csv`). Omitted → roll across **all** tables combined.
- `count`: int, 1–25 (Discord range), default **3**. **Ignored when `cr` is given.**
- `cr`: optional string — monster Challenge Rating (`"4"`, `"1/2"`, ...). When
  present it drives both item count and allowed rarities via the tier table.
- Handler `handle_roll_items_command(interaction, table, count, cr)`:
  1. Defer thinking (same pattern as other commands), then follow-up.
  2. Load + roll (off the event loop via `asyncio.to_thread`).
  3. Reply with a **Discord embed**: title `🎲 Random items from <table>` (or
     "all tables"), one field per item:
     - field name: `**Name**` (+ ` — Rarity` / `Level X` suffix if present)
     - field value: an item-page link line + notes line (omit blank parts).
     - CR roll footer: `CR {cr} → tier {cr_min}–{cr_max}: {n} items, rarities: a; b`.
  4. Error paths (friendly, no traceback):
     - unknown table → "No table named `x`. Available tables: a, b, c."
     - invalid CR string → "Couldn't parse CR `x` — use e.g. `4` or `1/2`."
     - no `cr_tiers.csv` / unparseable → point to the expected file + format.
     - no CSVs at all / empty table → point to `data/items/` and the expected format.
     - fewer allowed-rarity items than the tier count → roll what exists + note
       "*(only N {rarity} items available in this table)*".

## Wiring

- `config/settings.py`:
  ```python
  ITEMS_DIR: pathlib.Path = pathlib.Path(
      _or_default(os.getenv("ITEMS_DIR"), str(_REPO_ROOT / "data" / "items"))
  )
  ```
  (same `_or_default` pattern as `KB_PATH` / `CHARACTERS_FILE`).
- `main.py`: register `/roll_items` with `@app_commands.describe(...)` and
  delegate to the handler, exactly like the existing command registrations.
  The fingerprint-based command sync (`_ensure_commands_synced`) picks the new
  command up automatically on next start — no manual `/sync` needed.

## Docs

- `docs/commands.md`: new `/roll_items` section (usage, both CSV formats with
  examples, CR tier behavior incl. fractional CRs and blank-rarity=common).
- Root `README.md`: one-liner in the Commands (brief) list.

## Tests: `tests/test_roll_items.py`

Follow existing conventions (see `conftest.py`, monkeypatched settings paths):

1. **CSV parsing** — normal rows; BOM + CRLF file; missing optional columns;
   unknown extra columns ignored; row without `name` skipped (warning logged);
   whitespace trimmed; blank rarity → `"common"`; missing dir → empty list.
2. **Table resolution** — case-insensitive match, with/without `.csv` suffix,
   `cr_tiers.csv` excluded from the table list, unknown table handled by handler.
3. **CR parsing & tiers** — `parse_cr` for ints, fractions (`1/2`, ` 1 / 4 `),
   garbage → `ValueError`; tier CSV parsing (incl. fraction bounds);
   `resolve_tier` boundaries, below-first and above-last clamping; missing
   tier file → empty list.
4. **Rolling** — no duplicates in result (fixed seed, large table); count ==
   len → exhausted flag + all items present; count > len clamps; empty list
   errors cleanly; injected `rng` gives deterministic output.
5. **CR-scaled rolling** — tier filters to allowed rarities only (case-
   insensitive); blank-rarity items roll as common; count drawn within
   `[count_min, count_max]` (seeded); fewer available than requested → all
   returned + exhausted flag; fractional CR `1/2` lands in the 0–2 tier.
6. **Command handler** (discord Interaction mocked, as in other command tests) —
   embed contains each item name and its URL; CR footer rendered with tier info;
   `cr` present → `count` ignored; unknown table lists available tables;
   invalid CR string → friendly parse error; missing tier file → format hint;
   exhausted/short-rarity notice rendered.

## Out of scope (for now)

- Item-database API integration (user chose local CSV).
- Weighted rarity probabilities (tiers use an allowed set, per user decision).
- Per-channel roll history, weighted rarity tiers beyond the tier table.
- Prefix-command variant (`!roll`).

## Files touched

| File | Change |
|---|---|
| `bot_core/item_tables.py` | **new** — data model, CSV loading, CR tiers, rolling |
| `commands/roll_items_command.py` | **new** — handler + embed rendering |
| `config/settings.py` | add `ITEMS_DIR` |
| `main.py` | register `/roll_items` |
| `data/items/magic_items.csv` | **new** — starter example table (common→legendary) |
| `data/items/cr_tiers.csv` | **new** — default CR tier mapping |
| `docs/commands.md` | document `/roll_items` + both CSV formats |
| `README.md` | one-liner in command list |
| `tests/test_roll_items.py` | **new** — full coverage above |
