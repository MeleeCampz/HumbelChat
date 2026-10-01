# Item rolls (`/roll_items`)

How the random item tables work: data layout, CSV formats, and the exact
rolling mechanics.

## Data layout

| Location | Contents | Committed? |
|---|---|---|
| `data/items/*.csv` | Your rollable tables (one CSV per table) | no (gitignored, local-only) |
| `data/items/cr_tiers.csv` | CR tier definitions for `/roll_items cr:…` | no |
| `data/items/.rolled_state.json` | Persistent no-repeat pool state (hidden; managed by the bot) | no |
| `item_samples/` | Built-in starter tables shipped with the repo | yes |

**First run.** `data/items/` is gitignored, but a fresh clone still works out
of the box: on its first run, `/roll_items` copies the two files from
`item_samples/` into `data/items/` (creating the directory if needed). From
then on there is a single data location — edit the CSVs in place or replace
them with your own. The seed only acts while the folder holds no rollable
tables and **never overwrites existing files**, so in-place edits survive.

Files are re-read on every roll — adding a table or adjusting tiers never
requires a restart. The location is overridable via the `ITEMS_DIR` env var.

## Item table format

One CSV per table, header row required:

```csv
name,url,rarity,notes
Bag of Holding,https://example.com/items/bag-of-holding,uncommon,"5-ft cube, 0 weight, holds up to 500 lb."
```

- **`name`** — the only required column. Rows without a name are skipped with
  a warning log. An emote prefix here (e.g. `🎒 Bag of Holding`) is rendered
  as-is in the embed.
- **`url`** — optional; rendered as an "Item page" link above the notes.
- **`rarity`** — optional; case-insensitive. A blank rarity normalizes to
  `common`, so mundane items still participate in CR-scaled rolls. The
  official 5e scale is `common / uncommon / rare / very rare / legendary`;
  any other label (e.g. `artifact`) simply never matches a tier range.
- **`notes`** — optional; shown under the link in the embed.

Parsing is deliberately tolerant: BOM/CRLF OK, header and cell whitespace
trimmed, headers matched case-insensitively, missing optional columns and
unknown extra columns OK. Table names are CSV filenames without the `.csv`
suffix, matched case-insensitively (`magic_items`, `Magic_Items.CSV`, and
`MAGIC_ITEMS` all work).

## CR tier format

`data/items/cr_tiers.csv`, fully user-tunable:

```csv
min_cr,rarities
0,common:0-1;uncommon:1-2
3,common:0-1;uncommon:2-3;rare:1-2
6,uncommon:1-2;rare:2-3
10,rare:2-3;very rare:1-2
14,very rare:2-3;legendary:1-2
```

Each row is one tier: a single `min_cr` (integer or fraction — `1/2`, `1/4`)
plus `rarities`, a semicolon-separated list of **per-rarity count ranges**
(`rarity:min-max`). Because each tier has only a *minimum*, overlaps and gaps
between tiers are impossible by construction:

- Tier selection is a **step function** — the tier with the highest `min_cr`
  that is ≤ the rolled CR wins. A CR below every tier uses the first one; a
  CR above the last tier stays on the last one.
- A range of `0-…` means "maybe" (the roll can yield zero of that rarity).
- Malformed rows are skipped with a warning log; a row whose ranges all fail
  to parse is dropped entirely.

## Rolling mechanics

### Plain roll (`/roll_items [table] [count] [rarity]`)

1. Build the pool: all valid rows of the named table, or of **all** tables
   combined when `table` is omitted (rows keep their table of origin).
2. Optionally filter by `rarity:` (plain rolls only — ignored when `cr` is
   given); an unknown rarity errors with the rarities that do exist in scope.
3. Remove items already rolled out (see [Persistent pool](#persistent-no-repeat-pool)).
4. Draw `count` items **uniformly at random without replacement** — an item
   never repeats within one roll, and every remaining item has equal odds
   regardless of rarity (plain rolls ignore the tier table entirely).
5. If the pool holds fewer items than requested, everything is returned
   (shuffled) and the embed footer notes how many were available.

### CR-scaled roll (`/roll_items cr:<rating>`)

`cr` overrides `count`. The rating accepts integers and fractions
(`4`, `1/2`, `1/4`; whitespace tolerated). Then:

1. **Pick the tier** — highest `min_cr ≤ CR` (step function, above).
2. Remove items already rolled out (see [Persistent pool](#persistent-no-repeat-pool)).
3. **Roll each range independently.** For every `rarity:min-max` pair in the
   tier, a count is drawn *uniformly* from `[min, max]`, then that many items
   are sampled without replacement from the pool of items with that rarity.
4. Rarities the tier lists but the table has **no items for are skipped**
   (not an error). If a rarity's pool is smaller than its drawn count, all
   remaining items of that rarity come up and the footer notes the shortage.
5. If nothing could be rolled at all (every listed rarity empty, or every
   range rolled zero), the command replies with a friendly error naming what
   the tier wanted.

**Worked example — `/roll_items cr:5` with the default tiers:** CR 5 → tier
`min_cr 3` (`common:0-1;uncommon:2-3;rare:1-2`). Suppose the table holds 40
common, 90 uncommon, and 30 rare items:

- common: draw uniform in 0–1 → say **1** → sample 1 of 40
- uncommon: draw uniform in 2–3 → say **3** → sample 3 of 90
- rare: draw uniform in 1–2 → say **1** → sample 1 of 30

Result: 5 items, footer `CR 5 → tier (min CR 3): common ×1, uncommon ×3,
rare ×1`. Note the two independent sources of randomness — the per-rarity
*counts* are random, and *which* items fill them are random.

## Persistent no-repeat pool

By default an item can only come up **once** until the pool is reset — so a
long campaign doesn't hand out the same Bag of Holding twice.

- **State file** — consumed items are recorded in the hidden
  `data/items/.rolled_state.json` as `{table: [names]}`. Identity is
  *(table, name)*, so an all-tables roll records each item under its source
  table and a later single-table roll still excludes it. The file is managed
  by the bot; editing it by hand is possible but unnecessary. A missing or
  corrupt file degrades to "nothing consumed" (rolls just work).
- **Consumption timing** — items are recorded only after the reply has been
  delivered; a failed send loses nothing.
- **Footer** — every real roll ends with a remaining clause, e.g.
  `— 361 of 365 remaining in all tables`. With a `rarity:` filter the scope
  names the subset, e.g. `— 8 of 10 remaining in loot (rare)`.
- **Exhaustion** — when everything in scope is already rolled out, the
  command replies with a friendly message pointing at the reset options
  instead of rolling nothing.
- **Resetting**:
  - `fresh:true` on `/roll_items` — clears the whole pool, then rolls from it.
  - `/reset_rolls [table]` — clears one table (or all) without rolling, and
    reports how many items were returned.
- **Previewing** — `/roll_items cr:<rating> preview:true` shows the tier's
  ranges with the currently available pool size per rarity, e.g.
  `common 0-1 (40 available) · uncommon 2-3 (90 available)` — no roll, no
  consumption. Handy for checking a CR before spending items from the pool.
- **Editing tables** — rows you delete from a CSV are pruned from the state
  on the next roll; stale entries never wedge the pool.

## Utility commands

- **`/item_search <query> [table] [limit]`** — fuzzy-finds items (default top
  10, max 25) and lists them with rarity, table, and item page link. Scoring
  is deterministic: exact match > name starts with the query > substring >
  `difflib` similarity (kept when ≥ 0.6). Emote prefixes are ignored —
  `bag of holding` matches `🎒 Bag of Holding`. Never touches pool state.
- **`/item_stats [table]`** — per table: total, rolled out, remaining, plus a
  per-rarity breakdown; footer with grand totals. Read-only.
- **`/reset_rolls [table]`** — returns consumed items to the pool (see above).

## Tuning tips

- Want bigger loot at low CR? Widen the early ranges (`0,common:1-2;uncommon:2-3`).
- Want legendary items to start earlier? Add a row like `8,rare:1-2;legendary:0-1` —
  it slots in automatically by `min_cr`.
- Rarities a tier never lists can never come up in CR rolls, but *can* come up
  in plain rolls (which sample the whole table).

## Maintaining the tables

The tables are plain CSVs — add, edit, or delete rows directly and the next
roll picks up the changes immediately (no restart needed). One file per
table; see the format above.
