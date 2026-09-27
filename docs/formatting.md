# D&D Content Formatting (stat blocks, tables)

How the bot renders D&D stat blocks, armor/item comparisons and other
tabular content on Discord — and why it is built in two layers.

**Design principle:** the system prompt *steers* the model (soft), the
delivery layer *guarantees* sanity (hard). Even if the model misbehaves in a
new way, raw markup or unbounded-width output can never reach Discord.

Discord constraint that drives everything: embeds wrap at **~75 monospace
chars on desktop and ~40 on mobile** — column alignment only survives when
no line wraps. Discord also has **no HTML support** (raw `<table>` renders as
literal markup) and **silently drops embed fields with empty names**.

## Layer 1 — Prompt steering

`STAT_BLOCK_FORMAT_RULES` in `config/settings.py` is a `<stat-block-format>`
block appended to **every character's system prompt** via
`_compose_system_prompt()` in `bot_core/ai_client.py`. This keeps
`characters.json` pure persona.

- **Env-overridable:** set `STAT_BLOCK_FORMAT_RULES` in `.env` to change the
  wording without a code change or rebuild.
- Pinned by `tests/test_rag_cpu_pipeline.py::test_compose_system_prompt_appends_stat_block_rules`.

Current rules (abridged):

1. **Stat blocks** — name line in normal markdown, then a *narrow* fenced
   code block: max ~40 chars per line, at most 3 short columns, space-aligned,
   notable traits on the last line(s). Always close the fence.
2. **Comparisons / multi-column data** (armor, weapons, items, class tables) —
   no code fence, no pipe tables, never HTML. One entry per line: bold name,
   a colon, then labeled values separated by `" · "`. Every line ≤ ~60 chars;
   every value labeled (no header row); empty / not-yet-applicable values
   omitted entirely (no dash placeholders); long text values (feature names)
   get their own labeled line directly under the entry.

Example output:

````
**Awakened Tree** *(Huge Plant, Neutral)* · CR 2
```
AC 13    HP 59 (7d12+14)   Speed 20 ft.
STR 19 (+4)  DEX 6 (-2)   CON 15 (+2)
INT 10 (+0)  WIS 10 (+0)  CHA 7 (-2)
Vuln: Fire · Res: Bludgeoning, Piercing
```

**Hide Armor**: AC 12 + Dex (max 2) · Weight 12 lb. · Cost 10 GP
**Wizard, Level 1**: Prof +2 · Cantrips 3 · Prepared 4 · Slots 2
Features: Spellcasting, Ritual Adept, Arcane Recovery
````

### Rule iteration history

Each step was driven by a concrete failure observed on Discord:

| Commit | Problem seen | Rule added |
|---|---|---|
| `1d7d7a2` | stat block compressed into one semicolon run-on sentence | format rules exist at all; appended to every system prompt |
| `ce92f67` | same, different shape | fenced monospace table with space-aligned columns |
| `36c7b21` | 95-char armor table wrapped mid-column → unreadable | narrow stat blocks (≤~40 chars, 3 cols); comparisons one-entry-per-line |
| `a53309b` | model fell back to raw `<table>` HTML when pipes were forbidden | explicit "NEVER HTML" |
| `ae7bb21` | positional header row → runs of `—` placeholders for empty cells | colon after name, inline labels, omit empty values |
| `9727268` | labeled one-liners ran ~115 chars and wrapped mid-value | ≤~60 chars per line; long values get their own labeled line |

## Layer 2 — Deterministic safety nets

All in the delivery layer, so they apply **regardless of what the model
outputs**.

### Empty embed field names (`utils/embed_formatter.py`)

Discord silently drops fields whose `name` is `""` (no error). The
`add_field` wrapper substitutes a single space `" "`, which renders as an
invisible header. Unheaded code blocks in prose replies go into the
description instead of a field, so a stat table stays glued under its name
line. Regression: `TestNoEmptyFieldNames`.

### Fenced pipe tables (`_fenced_table_payload`)

Models often fence pipe tables to "preserve alignment"; fenced code is
rendered verbatim, so wide lines wrap mid-column. Fenced blocks that are
actually pipe tables (≥ half the lines contain pipes) are detected and
re-routed through `_render_table`, which splits wide tables into
self-contained column-group pieces that fit the device. Genuine code stays
verbatim. Regression: `TestFencedPipeTableSafetyNet`.

### HTML sanitization (`_sanitize_html`)

- Called at the top of `_build_embed_impl` (embed path) **and** inside
  `send_long_response` in `utils/response_splitter.py` (plain-text chunk
  path) — both delivery routes are covered.
- HTML `<table>` regions are converted to pipe tables (→ normal rendering);
  any other stray tags are stripped so raw markup can never ship.
- Regression: `TestHtmlTableSafetyNet` (includes the plain-chunk path, which
  reproduces the 2026-09-27 incident where a garbled `<td>` reply went out
  as `[N/5]` chunks).

## Layer 3 — Tests

| File | Covers |
|---|---|
| `tests/test_embed_formatter.py::TestNoEmptyFieldNames` | empty-name fields survive; code stays in description for prose replies |
| `tests/test_embed_formatter.py::TestFencedPipeTableSafetyNet` | fenced pipe table re-rendered (no line > 80 chars); genuine code verbatim |
| `tests/test_embed_formatter.py::TestHtmlTableSafetyNet` | HTML table → rendered; plain-chunk path sanitizes; stray tags stripped |
| `tests/test_rag_cpu_pipeline.py::test_compose_system_prompt_appends_stat_block_rules` | rules wired into every system prompt |

## Layer 4 — Operational notes

- **History contamination:** the model imitates its own earlier replies. Bad
  tables in persisted chat history keep nudging it for ~10 turns even after a
  prompt fix. For a clean evaluation: set `CHAT_HISTORY_RESET=clear` in
  `.env`, restart, then **restore the flag to empty** so future deploys don't
  wipe history.
- **Deploy & verify:** `docker compose up -d --build`, then confirm the new
  code is actually in the image, e.g.
  `docker compose exec -T bot sh -c 'grep -c "_sanitize_html" /app/utils/embed_formatter.py'`.
- **Prompt-only tweaks** (wording of the rules) need no rebuild if done via
  the `STAT_BLOCK_FORMAT_RULES` env var — just recreate the container.

## Troubleshooting quick reference

| Symptom on Discord | Cause | Fix |
|---|---|---|
| table/code block missing entirely, no error | embed field had an empty name → dropped by Discord | `add_field` space-substitution (in place) |
| monospace table wraps mid-column, alignment broken | line wider than device width | narrow-fence rules + `_fenced_table_payload` re-rendering |
| wall of raw `<table>`/`<td>` markup | model emitted HTML (no HTML support in Discord) | `_sanitize_html` on both delivery paths + "NEVER HTML" rule |
| runs of `— — —` in a list | positional format with empty cells | labeled values, omit empties (rule) |
| lines wrap mid-value | entry longer than ~60 chars | 60-char cap; long values on own line (rule) |
| model keeps repeating an old bad format | imitating its own persisted history | `CHAT_HISTORY_RESET=clear` once, then restore flag |

## Related docs

- [`embeds.md`](embeds.md) — embed block parsing, fallback behavior, `EMBED_FORMAT`.
- [`rag.md`](rag.md) — knowledge-base attachment (carries the detail on demand).
