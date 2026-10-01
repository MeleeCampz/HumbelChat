# Item rolls (`/roll_items`)

How the item tables work: data layout, CSV formats, and rolling mechanics.

## Data layout

| Location | Contents |
|---|---|
| `data/items/*.csv` | Your rollable tables (one CSV per table; gitignored) |
| `data/items/cr_tiers.csv` | CR tier definitions for `/roll_items cr:…` |
| `data/items/.rolled_state.json` | No-repeat pool state (hidden, managed by the bot) |
| `item_samples/` | Built-in starter tables shipped with the repo |

**First run.** A fresh clone works out of the box: the first `/roll_items`
copies the starter tables from `item_samples/` into `data/items/` — only while
the folder holds no tables, and existing files are never overwritten. From
then on, edit the CSVs in place; they are re-read on every roll (no restart).
The directory is overridable via `ITEMS_DIR`, and the table used when `table`
is omitted is set by `DEFAULT_ITEM_TABLE` (default: `magic_items`).

## Item table format

One CSV per table (table name = filename without `.csv`, case-insensitive):

```csv
name,url,rarity,notes
Bag of Holding,https://example.com/items/bag-of-holding,uncommon,"Holds up to 500 lb."
```

- **`name`** — the only required column. An emote prefix (e.g. `🎒 Bag of Holding`) is rendered as-is.
- **`url`** — optional; shown as an "Item page" link.
- **`rarity`** — optional, case-insensitive; blank = `common`. The 5e scale is `common / uncommon / rare / very rare / legendary`; other labels (e.g. `artifact`) never match a CR tier.
- **`notes`** — optional short description, shown in roll embeds and `/item_search`.

## CR tier format (`data/items/cr_tiers.csv`)

```csv
min_cr,rarities
0,common:0-1;uncommon:1-2
3,common:0-1;uncommon:2-3;rare:1-2
6,uncommon:1-2;rare:2-3
10,rare:2-3;very rare:1-2
14,very rare:2-3;legendary:1-2
```

Each row is a tier. The tier with the **highest `min_cr` ≤ the rolled CR**
wins (a CR below every tier uses the first one). `rarities` holds per-rarity
count ranges (`rarity:min-max`, semicolon-separated): for each range a count
is drawn uniformly and that many items are sampled. A range of `0-…` means
"maybe"; rarities with no items in the table are skipped.

## Rolling mechanics

**Plain roll** — `/roll_items [table] [count] [rarity]`: draws `count`
(default 3) items uniformly at random **without replacement** from one table
(the named one, or the default). `rarity:` filters first (plain rolls only);
already-rolled items are excluded. If fewer remain than requested, everything
is returned with a note in the footer.

**CR-scaled roll** — `/roll_items cr:<rating>`: `cr` overrides `count`. The
winning tier's per-rarity ranges decide how many of each rarity come up (e.g.
CR 5 → tier `min_cr 3` → maybe 1 common, 2–3 uncommon, 1–2 rare). The footer
shows the tier and the per-rarity breakdown.

## No-repeat pool

An item comes up **once** until the pool is reset — a long campaign doesn't
hand out the same Bag of Holding twice.

- Consumed items are tracked in the hidden `.rolled_state.json`; every roll's
  footer shows what's left (e.g. `— 361 of 365 remaining in magic_items`).
- An exhausted pool gets a friendly message instead of an empty roll.
- Reset with `fresh:true` (resets the rolled table, then rolls) or
  `/reset_rolls [table]` (one table or all, without rolling).
- `preview:true` with `cr:` shows the tier's ranges and the available count
  per rarity — no roll, nothing consumed.

## Utility commands

- **`/item_search <query> [table] [limit]`** — fuzzy-finds items (emote
  prefixes ignored; default top 10) and lists description, rarity, table, and
  page link for each match.
- **`/item_stats [table]`** — per-table totals: total / rolled out / remaining
  plus a per-rarity breakdown. Read-only.
- **`/reset_rolls [table]`** — returns consumed items to the pool (one table or all).

## Tuning tips

- Bigger loot at low CR? Widen the early ranges (`0,common:1-2;uncommon:2-3`).
- Legendary items earlier? Add a row like `8,rare:1-2;legendary:0-1` — it slots in by `min_cr`.
- Rarities a tier never lists can't come up in CR rolls, but *can* in plain rolls.

## Maintaining the tables

Plain CSVs — add, edit, or delete rows directly; the next roll picks up the
changes immediately. One file per table.
