# Embeds (Beyond20-style delivery)

AI replies are requested non-streaming and delivered as **Discord embeds** —
the same style the Beyond20 bot uses for rolls: a title, a description, and
inline fields. Structured replies (headings, tables, lists) look far cleaner
as an embed than as plain text, because Discord messages only support a small
Markdown subset (no tables).

## How it works

- The full reply is parsed into blocks (`utils/embed_formatter.py`):

  | Block                | Embed treatment |
  |---|---|
  | `# H1`               | embed **title** (first one only) |
  | `## H2`              | non-inline field heading (section divider) |
  | `###+`               | inline field heading |
  | `**Bold-only line**` | treated as an H2 section heading (optionally with a parenthetical annotation, e.g. `**Birdfolk** (Avian features)`) |
  | fenced code ```      | preserved verbatim as a non-inline field |
  | pipe table           | self-contained fenced monospace piece(s); header row becomes the field name; wide tables split into continuation fields with repeated headers |
  | bullet/numbered list | grouped into fields of up to 1024 chars |
  | plain paragraph      | description (first) or overflow text after fields |

- Replies longer than one embed are sent as several embed messages (max 10
  embeds per message). Over-long pieces are split across multiple fields.
- The accent color defaults to the D&D Beyond green (`#96BF6B`); degraded
  parses fall back to Discord blurple.

## Fallback behavior

Embed rendering is pure and never raises. When a reply is too small or
unstructured to benefit from an embed (or a Discord API error occurs), the
bot falls back to classic plain-text delivery — split into ≤2000-char
messages if needed — so the user always gets an answer.

## Disabling

Set `EMBED_FORMAT=0` in `.env` to restore classic plain-text delivery for all
replies. Default: on.
