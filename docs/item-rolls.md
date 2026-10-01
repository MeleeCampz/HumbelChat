# Item rolls (`/roll_items`)

How the random item tables work: data layout, CSV formats, and the exact
rolling mechanics — plus how the item collection is maintained.

## Data layout

| Location | Contents | Committed? |
|---|---|---|
| `data/items/*.csv` | Your rollable tables (one CSV per table) | no (gitignored, local-only) |
| `data/items/cr_tiers.csv` | CR tier definitions for `/roll_items cr:…` | no |
| `data/items/.rolled_state.json` | Persistent no-repeat pool state (hidden; managed by the bot) | no |
| `item_samples/` | Built-in starter tables shipped with the repo | yes |
| `data/items/tools/` | Private collection pipeline (scripts + caches) | no |

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
  `— 361 of 365 remaining in all tables`.
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

## Maintaining the item collection

The tables are maintained by a private pipeline in `data/items/tools/`
(also local-only; the scripts are intentionally not committed). No account or
login on the item database is needed — everything comes from public pages:

```
owned_sources.csv ─┐
                   ├─► collect_items.py ─► collected_items.csv ─► build_roll_table.py --all ─► magic_items.csv
gap_book_items.json ┘         (pool)        (one row per owned catalog item)
```

`collect_items.py` assembles the pool from four public sources:

1. **Item-database catalog** — the magic-item browse listing on the item
   database is server-rendered and public (~62 pages). Every entry carries
   name + canonical URL + displayed rarity, so one pass captures the whole
   ~1,200-item catalog. (The site's filter params are client-side only; there
   is no per-book or per-rarity server filtering.)
2. **dnd-wiki.org** — a MediaWiki with a working API at `/w/api.php`. Its
   "5e24 Magic Items" / "5e Magic Items" categories hold canon item pages
   whose wikitext cites the source book(s) via `{{Cite Pub|…}}` templates —
   that is the **book attribution** — and include a short description per item.
3. **Basic Rules A-Z index** — the item database's `br-2024/magic-items-a-z`
   page is also server-rendered; everything on it is Basic Rules by
   definition (names, URLs, rarities, descriptions).
4. **Gap lists** (`tools/gap_book_items.json`) — a few owned books have no
   per-book list on the wiki. For those, small item rosters were researched
   manually (dnd5e.wikidot.com's `wondrous-items:<book>` pages, Fandom wikis)
   and committed as name lists; they are matched against the catalog at run
   time.

Items whose cited books match `owned_sources.csv` are kept, deduped by name
(newer edition wins, a specific book beats the free Basic Rules alias),
deduped again by URL (edition variants of one catalog item collapse to one
row), renamed to their catalog display name, and written to
`collected_items.csv` plus a human-readable report (`tools/last_report.txt`).
Cached pages in `.page_cache/` make re-runs cheap — delete that folder to
force a full refresh.

### Adding a newly purchased book

1. Add its **exact official title** to `owned_sources.csv` (one per line;
   check the title on the item database's library page — fuzzy matching
   tolerates small deviations, but exact is best).
2. Re-run `python data/items/tools/collect_items.py`. Most canon books are
   picked up automatically via the wiki citations — check
   `tools/last_report.txt` for the new book's count.
3. If the book shows **0 items**, it has no per-book list on dnd-wiki:
   research its magic-item roster (try `dnd5e.wikidot.com`'s wondrous-items
   pages, the book's Fandom wiki page, or the item database's table-of-contents
   page for the book), and add an entry to `tools/gap_book_items.json` in the
   format of the existing ones (book title + item names + where you found them
   + date). Re-run the collector.
4. Rebuild the table: `python data/items/tools/build_roll_table.py --all`.
   Existing rows are kept **verbatim** (emotes and notes preserved); new items
   are appended with their short description in `notes`. A `.bak` of the
   previous table is written first.

The builder also enriches every row before writing:

- **Emotes** — a row whose `name` has no emote prefix gets one assigned
  deterministically: a hand-picked override (in `EMOTE_OVERRIDES_RAW`) beats
  ordered keyword rules (sword→⚔️, potion→🧪, ring→💍, staff→🦯, …), and
  anything unmatched falls back to ✨. A leading ✨ is treated as a
  placeholder and re-assigned on the next run once a better rule/override
  exists. Existing non-✨ emotes are never touched.
- **Descriptions** — an empty `notes` cell is filled from
  `tools/descriptions_extra.json` (hand-written/fetched short descriptions
  keyed by normalized name) or, failing that, from the cached Basic Rules A-Z
  page. A note that looks mangled by the wiki-markup stripper (leftover
  template fragments) is replaced the same way. Clean notes are left alone.

Notes on the resulting table:

- `--all` includes every pool item, including rarities the CR tiers never
  reference (`artifact`, `varies`, `unknown`). Those never appear in
  CR-scaled rolls (a tier only draws the rarities it lists), but they *can*
  come up in a plain `/roll_items <count>` roll, which samples the whole
  table. Use `--targets common:N,…` instead if you want a smaller table capped
  per rarity.
- Every row carries an emote prefix and a short description; tweak either
  directly in the `name` / `notes` columns whenever you like (the roll embed
  shows name + rarity only, so notes length doesn't matter).
- Verify after each run: `last_report.txt` lists unmatched names and per-book
  counts; every URL in the output comes from a live catalog page fetched that
  day.

Keep volumes low and personal — the item database's terms prohibit
distributing scrapers, which is why none of this is published.
