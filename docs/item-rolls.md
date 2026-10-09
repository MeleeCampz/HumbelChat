# Item rolls (`/roll_items`)

How the item tables work: data layout, CSV formats, and rolling mechanics.

## Data layout

| Location | Contents |
|---|---|
| `data/items/*.csv` | Your rollable tables (one CSV per table; gitignored) |
| `data/items/cr_tiers.csv` | CR tier definitions for `/roll_items cr:…` |
| `data/items/.rolled_state.json` | Consumed-item tracking for `consume:true` rolls (hidden, managed by the bot) |
| `data/items/.roll_messages.json` | Which roll messages carry item buttons, so they keep working after a restart (hidden, managed by the bot) |
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
(default 3) items uniformly at random from one table (the named one, or the
default) — no repeats within a single roll. `rarity:` filters first (plain
rolls only). If fewer than requested exist, everything is returned with a note
in the footer.

**CR-scaled roll** — `/roll_items cr:<rating>`: `cr` overrides `count`. The
winning tier's per-rarity ranges decide how many of each rarity come up (e.g.
CR 5 → tier `min_cr 3` → maybe 1 common, 2–3 uncommon, 1–2 rare). The footer
shows the tier and the per-rarity breakdown.

## Consuming items (`consume` flag)

By default rolls are pure random — nothing is tracked, and the same item can
come up again. Turn on `consume:true` to keep track of what you hand out:

- A consuming roll excludes items already marked as rolled, then marks the new
  picks; they stay out until cleared with `/reset_rolls [table]` (one table or all).
- Plain rolls ignore the tracking entirely — handy for quick or throwaway rolls
  that shouldn't touch the campaign pool.
- Every consuming roll's footer shows what's left (e.g. `— 361 of 365 remaining in magic_items`);
an exhausted pool gets a friendly message instead of an empty roll.
- Marks can also be set or cleared by hand: `/item_exclude <name>` marks one
  item as used (e.g. the ones you didn't keep from a batch), and
  `/item_include <name>` releases just that item — no full reset needed.
  Manual marks live in the same state as rolled-out items, so `/reset_rolls`
  clears them too, and `/item_stats` counts them as rolled out.

### Item buttons on roll results

Every roll reply carries one button per rolled item (up to 25). Clicking a
button toggles that single item in the consumed list — no typing names:

- **🚫 <name>** (red) — not consumed; click marks it used.
- **↩️ <name>** (green) — consumed; click puts it back in the pool.

The button on the message *is* the state indicator, so a scroll-back still
shows which items are out. Buttons work on plain rolls too (that's how you
mark "didn't keep" items without `consume:true`). They survive bot restarts:
each roll is remembered in `.roll_messages.json` and the buttons are
re-registered at startup (deleted messages drop their record automatically).
`/item_exclude`, `/item_include` and `/reset_rolls` operate on the same state
— a reset just leaves old buttons showing 🚫 until the message is edited or
gone; clicking one re-marks that single item.

## Utility commands

- **`/item_search <query> [table] [limit]`** — fuzzy-finds items (emote
  prefixes ignored; default top 10) and lists description, rarity, table, and
  page link for each match.
- **`/item_stats [table]`** — per-table totals: total / rolled out / remaining
  plus a per-rarity breakdown. Read-only.
- **`/reset_rolls [table]`** — clears the consumed-item tracking (one table or all), so those items can come up again.
- **`/item_exclude <name> [table]`** — manually mark one item as used (fuzzy name match; ambiguous names get a shortlist instead).
- **`/item_include <name> [table]`** — release one marked/rolled item back into the pool, without resetting the table.

## Tuning tips

- Bigger loot at low CR? Widen the early ranges (`0,common:1-2;uncommon:2-3`).
- Legendary items earlier? Add a row like `8,rare:1-2;legendary:0-1` — it slots in by `min_cr`.
- Rarities a tier never lists can't come up in CR rolls, but *can* in plain rolls.

## Maintaining the tables

Plain CSVs — add, edit, or delete rows directly; the next roll picks up the
changes immediately. One file per table.
