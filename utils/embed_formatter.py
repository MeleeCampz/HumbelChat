"""Beyond20-style embed rendering for AI replies.

Discord messages only support a small Markdown subset (no tables, no font
sizes), so long structured replies — character sheets, stat blocks, item
comparisons — look messy as plain text.  This module converts an AI reply
into a ``discord.Embed`` the way the Beyond20 bot does: structured content
becomes an embed with a title, a description, and *inline fields*, which is
the only reliable way to get table-like layouts in Discord.

Design goals
------------
* **Pure and safe** — :func:`build_embed` never raises; on any problem it
  returns ``None`` and the caller falls back to plain-text delivery.
* **Deterministic** — output depends only on the reply text, so it is easy
  to test and impossible to corrupt conversation state.
* **Lossless-ish** — code blocks are preserved verbatim; tables become
  aligned monospace fields; headings/bullets keep their Markdown inside
  embed fields (Discord renders ``**bold**`` etc. in field values).

Parsing model
-------------
The reply is split into *blocks*:

===================  =====================================================
Block                Embed treatment
===================  =====================================================
``# H1``             → embed **title** (first one only)
``## H2``            → non-inline field heading (section divider)
``###+``             → inline field heading
``**Bold only line**``→ treated as an H2 section heading (LLMs use bold-only
                       lines as sub-headings; keeping them attached to the
                       list that follows prevents label/content mismatches)
fenced code ```      → preserved verbatim as a non-inline field
pipe table           → self-contained fenced monospace piece(s); header row
                       becomes the field name, wide tables split into
                       continuation fields with repeated headers
bullet/numbered list → grouped into fields of up to ``MAX_FIELD_VALUE``
plain paragraph      → description (first) or overflow text after fields
===================  =====================================================

The first H1 is used as the embed title; if there is no heading at all a
short excerpt of the first line becomes the title.  Everything that does not
fit an embed field (Discord caps values at 1024 chars) degrades gracefully:
over-long pieces are split across multiple fields, and if the whole reply
cannot be represented as an embed (no title material / too little content),
``build_embed`` returns ``None``.
"""
from __future__ import annotations

import logging
import re

import discord

log = logging.getLogger("bot.embed")

# ── Discord embed hard limits (discord.py raises if exceeded) ────────────
MAX_TITLE: int = 256
MAX_DESCRIPTION: int = 4096
MAX_FIELD_NAME: int = 256
MAX_FIELD_VALUE: int = 1024
MAX_FIELDS: int = 25

# Beyond20 uses the D&D Beyond green for its roll embeds; we reuse it as the
# default accent so /ai replies look like part of the same family.
DEFAULT_COLOR: int = 0x96BF6B
FALLBACK_COLOR: int = 0x5865F2  # Discord blurple, used when parsing degrades

# ── Markdown block patterns ──────────────────────────────────────────────
_H1_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$")
_H2_RE = re.compile(r"^##\s+(.+?)\s*#*\s*$")
_H3_RE = re.compile(r"^#{3,}\s+(.+?)\s*#*\s*$")
# A line that is one bold segment optionally followed by a parenthetical
# annotation (``**Birdfolk** (Avian features)``).  LLMs use these as
# sub-headings.  Only whole-line matches count; ``He is **strong** today``
# and ``**Note:** see below`` (trailing prose, no parens) stay paragraphs.
_BOLD_HEADING_RE = re.compile(r"^\*\*(.+?)\*\*(?P<ann>(?:\s*\([^()]*\))*)\s*$")
_FENCE_OPEN_RE = re.compile(r"^```(\w*)\s*$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+")
# GFM header separators: one or more dashes, optional alignment colons
# (``---``, ``:---:``, ``| - |`` — LLMs sometimes emit the short form).
_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")


# ════════════════════════════════════════════════════════════════════════
#  Low-level parsing helpers
# ════════════════════════════════════════════════════════════════════════

def _split_table_row(line: str) -> list[str]:
    """Split a ``| a | b |`` row into stripped cells (outer pipes removed)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator_row(line: str) -> bool:
    """True for GFM header separators like ``| --- | :---: |``."""
    cells = _split_table_row(line)
    if not cells:
        return False
    return all(_SEPARATOR_CELL_RE.match(c) or c == "" for c in cells) and \
        any("-" in c for c in cells)


# A table's rendered lines must fit BOTH the 1024-char field-value cap AND
# Discord's practical per-line width (~80 chars before a monospace block
# wraps and columns stop lining up).  Tables wider than this are split into
# column-group pieces (each repeating the header) instead of shrinking every
# column until words get cut off mid-way.
_MAX_TABLE_ROW_WIDTH: int = 78


def _render_table(header: list[str], rows: list[list[str]]) -> tuple[str, list[str]]:
    """Render a table as (field_name, [fenced monospace pieces]).

    The header row becomes the field name (bolded), and the body is rendered
    as one or more *self-contained* code-fenced blocks — the closest thing to
    a real table Discord can display.  Each piece opens AND closes its own
    fence, so splitting a wide table across several fields can never strand
    an unterminated fence (which would make Discord render raw Markdown).

    Column widths adapt per piece: each column gets its natural width up to
    40 chars.  When the full table is wider than ``_MAX_TABLE_ROW_WIDTH``, it
    is split into *column-group* pieces — each piece shows a subset of the
    columns (repeating its header + separator) so words are never cut off
    mid-way and every piece reads standalone.  Rows that still overflow the
    field cap are hard-split across continuation pieces; nothing is lost.
    """
    ncols = max([len(header)] + [len(r) for r in rows] or [0])
    if ncols == 0:
        return "", []

    # Normalise ragged rows to a fixed width (missing cells → empty).
    def _norm(cells: list[str]) -> list[str]:
        out = list(cells[:ncols])
        out += [""] * (ncols - len(out))
        return [c.replace("\n", " ") for c in out]

    header = _norm(header)
    rows = [_norm(r) for r in rows]

    # Drop junk columns: every cell empty or a dash/placeholder.  LLMs
    # sometimes emit trailing padding columns (``| ... | | | |``) or labelled
    # but contentless columns (e.g. ``Spell Slots`` full of ``—`` for a
    # non-spellcaster); on Discord those make tables unreadable, so we strip
    # them before layout.  A table that is *entirely* junk keeps its shape.
    _PLACEHOLDERS = {"", "—", "–", "-", "N/A", "n/a", "/"}

    def _has_data(idx: int) -> bool:
        return any(row[idx].strip() not in _PLACEHOLDERS for row in rows)

    keep = [i for i in range(ncols) if _has_data(i)]
    if 0 < len(keep) < ncols:
        header = [header[i] for i in keep]
        rows = [[row[i] for i in keep] for row in rows]
        ncols = len(header)

    # Natural column widths, capped at 40 so no single cell dominates.
    natural = [min(max(len(c) for c in col), 40) for col in zip(*([header] + rows))]

    def _wrap_cell(cell: str, width: int) -> list[str]:
        """Wrap *cell* to *width* chars (word-boundary preferred).  Never
        truncates — over-long words are hard-broken so content is preserved."""
        if len(cell) <= width:
            return [cell]
        out: list[str] = []
        rest = cell
        while len(rest) > width:
            cut = rest.rfind(" ", 0, width)
            if cut <= 0:
                cut = width  # no word boundary — hard break (very long token)
            out.append(rest[:cut])
            rest = rest[cut:].lstrip()
        out.append(rest)
        return out

    def _fmt_row(cells: list[str], widths: list[int]) -> list[str]:
        """Format one table row as aligned monospace line(s).

        Cells longer than their column width wrap inside the column (subsequent
        lines pad the other columns), so no cell content is ever truncated.
        """
        wrapped = [_wrap_cell(c, widths[i]) for i, c in enumerate(cells)]
        nlines = max(len(w) for w in wrapped)
        lines: list[str] = []
        for li in range(nlines):
            parts = []
            for i, w in enumerate(widths):
                cell_line = wrapped[i][li] if li < len(wrapped[i]) else ""
                # Only the last line of a cell is padded; earlier lines keep
                # their natural width so wrapping looks tidy.
                parts.append(cell_line.ljust(widths[i]) if li == nlines - 1 else cell_line)
            lines.append(" | ".join(parts))
        return lines

    name = "**" + " | ".join(h for h in header if h) + "**"[:MAX_FIELD_NAME]

    # Group columns into pieces that fit the target row width.  Greedy: keep
    # adding a column while it fits; a column wider than the whole target gets
    # its own group with a capped display width (its content still survives via
    # the row splitting below, so nothing is lost).
    groups: list[list[int]] = []
    cur_idx: list[int] = []
    cur_w = 0
    for i in range(ncols):
        w_i = natural[i]
        if w_i > _MAX_TABLE_ROW_WIDTH:
            if cur_idx:
                groups.append(cur_idx)
                cur_idx, cur_w = [], 0
            groups.append([i])
            continue
        sep = 3 if cur_idx else 0  # " | " joiner between columns
        if cur_idx and cur_w + sep + w_i > _MAX_TABLE_ROW_WIDTH:
            groups.append(cur_idx)
            cur_idx, cur_w = [], 0
            sep = 0
        cur_idx.append(i)
        cur_w += sep + w_i
    if cur_idx:
        groups.append(cur_idx)

    pieces: list[str] = []
    for g in groups:
        widths = [min(natural[i], _MAX_TABLE_ROW_WIDTH) for i in g]

        header_lines = _fmt_row([header[i] for i in g], widths)
        # Separator under the header — same " | " joiner as the data rows so
        # every line has identical width and columns line up perfectly.
        sep_line = " | ".join("-" * w for w in widths)
        head_block = header_lines + [sep_line]
        head_len = sum(len(l) for l in head_block) + len(head_block)  # + newlines

        budget = MAX_FIELD_VALUE - head_len - 8  # fences+newlines overhead
        cur: list[str] = []
        cur_len = 0

        def _flush() -> None:
            nonlocal cur, cur_len
            if cur:
                pieces.append("```\n" + "\n".join(head_block + cur) + "\n```")
                cur, cur_len = [], 0

        for r in rows:
            row_lines = _fmt_row([r[i] for i in g], widths)
            # Pack this row's (possibly wrapped) lines into the current piece;
            # when it no longer fits, start a new one.  A single line longer
            # than the whole budget is hard-cut into slices so nothing is lost.
            for line in row_lines:
                step = max(1, budget)
                slices = [line[i:i + step] for i in range(0, len(line), step)] or [""]
                for s in slices:
                    if cur and cur_len + len(s) + 1 > budget:
                        _flush()
                    cur.append(s)
                    cur_len += len(s) + (1 if cur_len else 0)
        _flush()

    return name, pieces


def _split_code_block(code: str, limit: int = MAX_FIELD_VALUE) -> list[str]:
    """Split a fenced code block into self-contained fenced pieces.

    Each piece opens AND closes its own fence (repeating the language tag)
    so long code can never be split mid-fence and render as raw Markdown.
    Splits prefer line boundaries; a single over-long line is hard-cut.
    """
    lang_m = re.match(r"^```(\w*)\s*$", code.strip().splitlines()[0]) if code.strip() else None
    lang = (lang_m.group(1) if lang_m else "")
    fence_open = f"```{lang}"
    inner = code
    # Strip an existing outer fence so we can re-wrap per piece.
    stripped = code.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        first = lines[0].strip()[3:].strip()  # language tag, if any
        lang = first or lang
        fence_open = f"```{lang}"
        body_lines = lines[1:]
        if body_lines and body_lines[-1].strip().startswith("```"):
            body_lines = body_lines[:-1]
        inner = "\n".join(body_lines)

    # piece = fence_open + "\n" + body + "\n```" → overhead is
    # len(fence_open) + 1 (open NL) + 4 ("\n```") = len(fence_open) + 5.
    budget = limit - len(fence_open) - 5
    lines = inner.splitlines()
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in lines:
        add = len(line) + (1 if cur else 0)
        if cur and cur_len + add > budget:
            pieces.append(fence_open + "\n" + "\n".join(cur) + "\n```")
            cur, cur_len = [], 0
            add = len(line)
        while add > budget:  # single line longer than the whole field
            if cur:
                pieces.append(fence_open + "\n" + "\n".join(cur) + "\n```")
                cur, cur_len = [], 0
            pieces.append(fence_open + "\n" + line[:budget] + "\n```")
            line = line[budget:]
            add = len(line)
        cur.append(line)
        cur_len += add
    if cur:
        pieces.append(fence_open + "\n" + "\n".join(cur) + "\n```")
    return pieces or [fence_open + "\n```"]


def _split_field_value(value: str, limit: int = MAX_FIELD_VALUE) -> list[str]:
    """Split a field value into ≤*limit* pieces, preferring line boundaries."""
    if len(value) <= limit:
        return [value]

    pieces: list[str] = []
    while len(value) > limit:
        window = value[:limit]
        cut = window.rfind("\n")
        if cut < limit // 2:
            cut = limit
        pieces.append(value[:cut].rstrip())
        value = value[cut:].lstrip("\n")
    if value:
        pieces.append(value)
    return [p for p in pieces if p]


def _group_lines(lines: list[str], limit: int = MAX_FIELD_VALUE) -> list[list[str]]:
    """Group lines into chunks whose joined length stays ≤ *limit*."""
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for line in lines:
        add = len(line) + (1 if cur else 0)
        if cur and cur_len + add > limit:
            groups.append(cur)
            cur, cur_len = [], 0
            add = len(line)
        # A single line longer than the limit is hard-split.
        while add > limit:
            if cur:
                groups.append(cur)
                cur, cur_len = [], 0
            cur = [line[:limit]]
            cur_len = min(limit, len(line))
            line = line[limit:]
            add = len(line)
        cur.append(line)
        cur_len += add
    if cur:
        groups.append(cur)
    return groups


# ════════════════════════════════════════════════════════════════════════
#  Block extraction
# ════════════════════════════════════════════════════════════════════════

def _extract_blocks(text: str) -> list[tuple[str, object]]:
    """Split reply text into (kind, payload) blocks.

    Kinds: ``h1``, ``h2``, ``h3``, ``code`` (str), ``table`` (header, rows),
    ``list`` (list[str] of line texts), ``para`` (str).
    """
    lines = text.splitlines()
    blocks: list[tuple[str, object]] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # ── Fenced code block (always wins — content is verbatim) ──────
        m = _FENCE_OPEN_RE.match(stripped)
        if m:
            j = i + 1
            buf: list[str] = []
            while j < n and not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            blocks.append(("code", "\n".join(buf)))
            i = j + 1  # skip closing fence (or run past EOF)
            continue

        if not stripped:
            i += 1
            continue

        # ── Headings ───────────────────────────────────────────────────
        m = _H1_RE.match(stripped)
        if m and not stripped.startswith("##"):
            blocks.append(("h1", m.group(1).strip()))
            i += 1
            continue
        m = _H2_RE.match(stripped)
        if m:
            blocks.append(("h2", m.group(1).strip()))
            i += 1
            continue
        m = _H3_RE.match(stripped)
        if m:
            blocks.append(("h3", m.group(1).strip()))
            i += 1
            continue

        # ── Bold label (optionally + parenthetical) → section heading ──────
        # Without this, ``**Birdfolk Species**`` would be swallowed into the
        # description while its list became a separate unnamed field — the
        # label and its content ended up in different places.  A trailing
        # parenthetical annotation (``**Birdfolk** (Avian features)``) is kept
        # as part of the heading so the label stays descriptive.
        m = _BOLD_HEADING_RE.match(stripped)
        if m:
            label = m.group(1).strip().replace("**", "").strip()
            ann = (m.group("ann") or "").strip()
            heading = f"{label} {ann}".strip() if ann else label
            blocks.append(("h2", heading))
            i += 1
            continue

        # ── Pipe table (header row + separator row) ────────────────────
        if stripped.startswith("|") and i + 1 < n and _is_separator_row(lines[i + 1]):
            header = _split_table_row(line)
            j = i + 2
            rows: list[list[str]] = []
            while j < n and lines[j].strip().startswith("|"):
                rows.append(_split_table_row(lines[j]))
                j += 1
            blocks.append(("table", (header, rows)))
            i = j
            continue

        # ── Bullet / numbered list ─────────────────────────────────────
        if _BULLET_RE.match(line) or _NUMBERED_RE.match(line):
            buf = []
            j = i
            while j < n:
                l2 = lines[j]
                if not l2.strip():
                    break
                if _BULLET_RE.match(l2) or _NUMBERED_RE.match(l2):
                    buf.append(l2.rstrip())
                    j += 1
                elif l2.startswith((" ", "\t")):
                    # continuation of the previous bullet
                    buf.append(l2.rstrip())
                    j += 1
                else:
                    break
            blocks.append(("list", buf))
            i = j
            continue

        # ── Plain paragraph (until blank line / next block) ────────────
        buf = [line]
        j = i + 1
        while j < n:
            l2 = lines[j]
            if not l2.strip():
                break
            s2 = l2.strip()
            if (s2.startswith("#") or s2.startswith("```")
                    or _BULLET_RE.match(l2) or _NUMBERED_RE.match(l2)
                    or (s2.startswith("|") and j + 1 < n
                        and _is_separator_row(lines[j + 1]))):
                break
            buf.append(l2.rstrip())
            j += 1
        blocks.append(("para", "\n".join(buf)))
        i = j

    return blocks


# ════════════════════════════════════════════════════════════════════════
#  Embed assembly
# ════════════════════════════════════════════════════════════════════════

def _clamp_title(text: str) -> str:
    text = text.strip().replace("\n", " ")
    return text[:MAX_TITLE]


def build_embed(
    text: str,
    *,
    title_override: str | None = None,
    color: int | None = None,
) -> discord.Embed | None:
    """Convert AI reply *text* into a ``discord.Embed``.

    Returns ``None`` when the text is empty/too small to benefit from embed
    formatting — callers should then fall back to plain-text delivery.  This
    function never raises.  For replies that may overflow a single embed,
    prefer :func:`build_embeds_for_channel`.
    """
    try:
        result = _build_embed_impl(text, title_override=title_override, color=color)
    except Exception as e:  # defensive — a bad reply must not kill delivery
        log.warning("build_embed failed (%s); falling back to plain text", e)
        return None
    if result is None:
        return None
    embed, tail = result
    if tail:
        log.warning(
            "build_embed: %d chars did not fit one embed — caller should use "
            "build_embeds_for_channel for very long replies", len(tail),
        )
    return embed


def _build_embed_impl(
    text: str,
    *,
    title_override: str | None = None,
    color: int | None = None,
) -> discord.Embed | None:
    if not text or not text.strip():
        return None

    blocks = _extract_blocks(text)
    if not blocks:
        return None

    # ── Title: first H1 > override (e.g. character name) > excerpt ─────
    # The most specific wins: an explicit heading in the reply is more
    # informative than a generic per-character label (Beyond20 titles its
    # embeds with the item/roll name, not a bot label).
    title = ""
    for kind, payload in blocks:
        if kind == "h1" and payload:
            title = _clamp_title(str(payload))
            break
    if not title:
        title = (title_override or "").strip()
    if not title:
        first_line = next(
            (b[1] for b in blocks
             if b[0] in ("para", "list", "code") and str(b[1]).strip()),
            "",
        )
        if isinstance(first_line, list):
            first_line = first_line[0] if first_line else ""
        # Strip leading bullet/heading markers so the excerpt reads cleanly.
        title = _clamp_title(re.sub(r"^[\s#>*+\-•)]+", "", str(first_line)))
        if not title:
            return None  # nothing usable as a title → plain-text fallback

    # ── Walk blocks, building description + fields ─────────────────────
    embed = discord.Embed(title=title)
    desc_parts: list[str] = []
    desc_len = 0
    field_count = 0
    degraded = False
    overflow: list[str] = []  # content that fits neither desc nor fields

    # Structured mode: the reply has real sections (headings/tables), so
    # paragraphs that appear *after* the first section belong to the body as
    # fields — putting them in the description would reorder the document.
    # Prose replies keep everything in the description for a clean read.
    structured = any(k in ("h1", "h2", "h3", "table") for k, _ in blocks)

    def add_field(name: str, value: str, inline: bool) -> None:
        nonlocal field_count, degraded
        if not name and not value:
            return
        if field_count >= MAX_FIELDS:
            # Out of fields — park it for the multi-embed path.
            degraded = True
            overflow.append(f"**{name}**\n{value}" if name else value)
            return
        for piece in _split_field_value(value):
            if field_count >= MAX_FIELDS:
                degraded = True
                overflow.append(f"**{name}**\n{piece}" if name else piece)
                return
            embed.add_field(name=name[:MAX_FIELD_NAME], value=piece, inline=inline)
            field_count += 1

    # A section heading (h2/h3) is *pending* until the block it introduces
    # arrives; then it's folded into that field's name instead of rendering as
    # a separate "label" field above an anonymous content field.  This removes
    # one round-trip of vertical space per section and keeps the label glued
    # to its content even when fields get split.
    pending_heading: list[str] = []

    def _fold_name(name: str) -> str:
        """Merge any pending heading into *name* (or use it as the name)."""
        if not pending_heading:
            return name
        h = pending_heading.pop(0)
        # Strip Markdown bold markers from the content name so the combined
        # label reads as one clean string instead of ``Heading — **bold**``.
        clean = name.replace("**", "").strip()
        if not clean or clean in {" ", "(continued)", "(cont.)"}:
            return h[:MAX_FIELD_NAME]
        combined = f"{h} — {clean}"
        if len(combined) > MAX_FIELD_NAME:
            # Keep the section heading — it identifies the block.
            return h[:MAX_FIELD_NAME]
        return combined

    def _append_desc(piece: str) -> None:
        nonlocal desc_len, degraded
        if not piece.strip():
            return
        room = MAX_DESCRIPTION - desc_len
        if room <= 0:
            degraded = True
            overflow.append(piece)
            return
        if len(piece) > room:
            # Description full — keep the tail for later so nothing is lost.
            overflow.append(piece[room:].strip())
            piece = piece[:room].rstrip()
        if desc_parts:
            desc_len += 2 + len(piece)
        else:
            desc_len += len(piece)
        desc_parts.append(piece)

    for kind, payload in blocks:
        if kind == "h1":
            continue  # already consumed as the title
        if kind in ("h2", "h3"):
            # A heading directly after a heading can't be folded — flush the
            # earlier one as its own label field first.
            while pending_heading:
                add_field(pending_heading.pop(0), " ", False)
            pending_heading.append(str(payload)[:MAX_FIELD_NAME])
            continue
        elif kind == "code":
            code = str(payload)
            # Preserve the verbatim block as self-contained fenced piece(s);
            # a long block splits into continuation fields instead of being
            # cut mid-fence (which would render as raw Markdown).
            for idx, piece in enumerate(_split_code_block(code)):
                if len(piece) > MAX_FIELD_VALUE:
                    degraded = True  # defensive — splitter already bounds it
                nm = _fold_name("" if idx == 0 else "(continued)")
                add_field(nm, piece, False)
        elif kind == "table":
            header, rows = payload  # type: ignore[misc]
            name, pieces = _render_table(list(header), [list(r) for r in rows])
            for idx, value in enumerate(pieces):
                if len(value) > MAX_FIELD_VALUE:
                    degraded = True  # defensive — splitter already bounds it
                nm = _fold_name(name if idx == 0 else f"{name} (cont.)")
                add_field(nm, value, False)
        elif kind == "list":
            lines = list(payload)  # type: ignore[arg-type]
            for gi, group in enumerate(_group_lines(lines)):
                # First group takes the pending heading as its field name so
                # the label sits with its bullets; later groups stay unnamed.
                nm = _fold_name("") if gi == 0 else ""
                add_field(nm, "\n".join(group), len(group) <= 4)
        else:  # para
            if structured and desc_parts:
                add_field(_fold_name(""), str(payload), False)
            else:
                _append_desc(str(payload))

    # A heading with nothing after it (trailing label) still renders — as a
    # plain field, so the text is never lost.
    for h in pending_heading:
        add_field(h, " ", False)

    if desc_parts:
        embed.description = "\n\n".join(desc_parts)[:MAX_DESCRIPTION]

    # ── Colour ─────────────────────────────────────────────────────────
    if color is not None:
        try:
            embed.color = discord.Color(int(color))
        except Exception:
            degraded = True
    elif degraded:
        embed.color = discord.Color(FALLBACK_COLOR)
    else:
        embed.color = discord.Color(DEFAULT_COLOR)

    # ── Usefulness gate: an embed for a 5-word reply is just noise ─────
    body_len = sum(len(p) for p in desc_parts) + sum(
        len(f.value) for f in embed.fields
    )
    if body_len < 80 and not embed.fields:
        log.debug("build_embed: content too small (%d chars) — using plain text",
                  body_len)
        return None  # consistent with the other early-exits

    tail = "\n\n".join(overflow).strip()
    log.info(
        "build_embed: title=%r desc_chars=%d fields=%d degraded=%s overflow=%d",
        embed.title, len(embed.description or ""), len(embed.fields),
        degraded, len(tail),
    )
    return embed, tail


# Target size for one embed's worth of text: comfortably under the 4096-char
# description cap, leaving room for section fields.
_CHUNK_TARGET: int = 3500
_MAX_SPLIT_DEPTH: int = 4


def _split_into_chunks(text: str) -> list[str]:
    """Split *text* into chunks of ≤ ``_CHUNK_TARGET`` chars.

    Prefers H2 section boundaries, then blank-line paragraph boundaries,
    then hard character cuts (for pathological single-paragraph text).
    """
    lines = text.splitlines()
    sections: list[str] = []
    cur: list[str] = []
    for line in lines:
        if _H2_RE.match(line.strip()) and cur:
            sections.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        sections.append("\n".join(cur))

    # Break every section into pieces that individually fit the target.
    pieces: list[str] = []
    for section in sections:
        for para in section.split("\n\n"):
            if len(para) <= _CHUNK_TARGET:
                pieces.append(para)
            else:
                # One absurdly long paragraph/line — hard character cut.
                for i in range(0, len(para), _CHUNK_TARGET):
                    pieces.append(para[i : i + _CHUNK_TARGET])

    # Greedily pack pieces into chunks.
    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = f"{buf}\n\n{piece}".strip() if buf else piece
        if len(candidate) <= _CHUNK_TARGET or not buf:
            buf = candidate
        else:
            chunks.append(buf)
            buf = piece
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def _halve(text: str) -> tuple[str, str]:
    """Split *text* roughly in half at a blank-line boundary (or mid-char)."""
    mid = len(text) // 2
    cut = text.rfind("\n\n", 0, mid)
    if cut < mid // 2:
        cut = text.rfind("\n", 0, mid)
    if cut < mid // 2:
        cut = mid
    return text[:cut].rstrip(), text[cut:].lstrip()


def _embeds_for_chunk(
    chunk: str,
    *,
    depth: int = 0,
    color: int | None = None,
) -> list[discord.Embed]:
    """Build embed(s) for one chunk, halving recursively if it still overflows.

    At ``_MAX_SPLIT_DEPTH`` we accept the (extremely rare) residual overflow
    and log it, so this function always terminates.
    """
    try:
        result = _build_embed_impl(chunk, color=color)
    except Exception as e:  # defensive
        log.warning("_embeds_for_chunk failed (%s); dropping chunk", e)
        return []
    if result is None:
        return []
    embed, tail = result
    if not tail:
        return [embed]
    if depth >= _MAX_SPLIT_DEPTH:
        log.warning(
            "build_embeds: %d chars still overflow after %d splits — "
            "truncating (pathological input)", len(tail), depth,
        )
        return [embed]
    left, right = _halve(chunk)
    out: list[discord.Embed] = []
    if left.strip():
        out.extend(_embeds_for_chunk(left, depth=depth + 1, color=color))
    if right.strip():
        out.extend(_embeds_for_chunk(right, depth=depth + 1, color=color))
    return out or [embed]


def build_embeds_for_channel(
    text: str,
    *,
    title_override: str | None = None,
    color: int | None = None,
) -> list[discord.Embed]:
    """Split *text* into one or more embeds that each fit Discord limits.

    Most replies become a single embed.  When the content overflows one
    embed (very long prose, many sections), the text is split at section /
    paragraph boundaries and rebuilt as several embeds so **no content is
    lost**.  Returns ``[]`` when plain-text delivery should be used instead.
    """
    try:
        result = _build_embed_impl(text, title_override=title_override, color=color)
    except Exception as e:  # defensive — a bad reply must not kill delivery
        log.warning("build_embeds_for_channel failed (%s); plain text", e)
        return []
    if result is None:
        return []
    embed, tail = result
    if not tail:
        return [embed]

    # Overflow — split the whole reply into chunks and rebuild per chunk.
    log.info("build_embeds_for_channel: %d chars overflowed; splitting", len(tail))
    out: list[discord.Embed] = []
    for idx, chunk in enumerate(_split_into_chunks(text)):
        if idx == 0 and title_override:
            # Rebuild the first chunk with the original title preserved.
            try:
                r = _build_embed_impl(chunk, title_override=title_override, color=color)
                if r is not None:
                    e, t2 = r
                    if not t2:
                        out.append(e)
                        continue
            except Exception:  # defensive
                pass
        out.extend(_embeds_for_chunk(chunk, color=color))
    return out or [embed]
