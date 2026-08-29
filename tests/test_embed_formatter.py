"""Tests for Beyond20-style embed formatting (utils/embed_formatter.py).

Covers:
  - block extraction (headings, tables, code, lists, paragraphs)
  - Discord limit compliance (title/desc/field caps, max fields)
  - content preservation across multi-embed splits (no data loss)
  - graceful fallback for tiny / empty / pathological input
  - the send_long_response_embedded delivery helper (batching + fallback)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from utils.embed_formatter import (
    build_embed,
    build_embeds_for_channel,
    _extract_blocks,
    _render_table,
    _split_code_block,
    _split_into_chunks,
    MAX_DESCRIPTION,
    MAX_FIELD_NAME,
    MAX_FIELD_VALUE,
    MAX_FIELDS,
    MAX_TITLE,
)

# ── Shared fixtures ──────────────────────────────────────────────────────

CHARACTER_SHEET = """# Trixy Smoldersome — Level 3 Artificer (Mapach)

## Attributes
| Attribute | Score | Modifier |
| --------- | ----- | -------- |
| STR       | 8     | -1       |
| DEX       | 14    | +2       |
| CON       | 12    | +1       |
| INT       | 16    | +3       |

## Equipment
- **Rusty Longsword** (found in the scrap heap)
- Gotheads Leather Armor, reinforced with chain scraps
- Artificer's Toolkit — 4/8 charges

## Notes
Trixy grew up in Alderheart and *hates* birdfolks. Alignment: **neutral chaotic**.

```python
def scrap_gadget(iron):
    return iron + 3
```
"""


def _all_embed_text(embeds) -> str:
    """Concatenate everything user-visible across a list of embeds."""
    parts = []
    for e in embeds:
        parts.append(e.description or "")
        for f in e.fields:
            parts.append(f.name)
            parts.append(f.value)
    return " ".join(parts)


def _assert_within_limits(embed) -> None:
    d = embed.to_dict()
    assert len(d.get("title", "")) <= MAX_TITLE
    assert len(d.get("description") or "") <= MAX_DESCRIPTION
    assert len(d.get("fields", [])) <= MAX_FIELDS
    for f in d.get("fields", []):
        assert len(f["name"]) <= MAX_FIELD_NAME, f["name"]
        assert len(f["value"]) <= MAX_FIELD_VALUE, f["value"]


# ════════════════════════════════════════════════════════════════════════
#  Block extraction
# ════════════════════════════════════════════════════════════════════════

class TestExtractBlocks:

    def test_headings(self):
        blocks = _extract_blocks("# Title\n\n## Sub\n\n### Deep")
        kinds = [k for k, _ in blocks]
        assert kinds == ["h1", "h2", "h3"]
        assert blocks[0][1] == "Title"

    def test_table_with_separator(self):
        text = "| A | B |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |"
        blocks = _extract_blocks(text)
        assert len(blocks) == 1
        kind, (header, rows) = blocks[0]
        assert kind == "table"
        assert header == ["A", "B"]
        assert rows == [["1", "2"], ["3", "4"]]

    def test_pipe_row_without_separator_is_paragraph(self):
        """Rows without a GFM separator row are NOT tables."""
        blocks = _extract_blocks("| a | b |\n| c | d |")
        assert [k for k, _ in blocks] == ["para"]

    def test_code_block_verbatim(self):
        text = "before\n\n```\na | b\n# not a heading\n```\n\nafter"
        blocks = _extract_blocks(text)
        code = [p for k, p in blocks if k == "code"]
        assert len(code) == 1
        assert "| b" in code[0] and "# not a heading" in code[0]
        # surrounding text is intact
        paras = [p for k, p in blocks if k == "para"]
        assert any("before" in p for p in paras)
        assert any("after" in p for p in paras)

    def test_list_grouping(self):
        text = "- one\n- two\n  continuation line\n- three\n\ntail paragraph"
        blocks = _extract_blocks(text)
        lists = [p for k, p in blocks if k == "list"]
        assert len(lists) == 1
        assert len(lists[0]) == 4  # 3 bullets + continuation

    def test_numbered_list(self):
        text = "1. first\n2. second"
        blocks = _extract_blocks(text)
        assert [k for k, _ in blocks] == ["list"]

    def test_bold_only_line_is_heading(self):
        """LLMs use whole-line ``**Bold**`` as sub-headings — they must become
        section headings so the label stays attached to the list below it."""
        text = "intro\n\n**Birdfolk Species**\n*   Corvum\n*   Strig"
        blocks = _extract_blocks(text)
        assert [k for k, _ in blocks] == ["para", "h2", "list"]
        assert blocks[1][1] == "Birdfolk Species"

    def test_inline_bold_stays_paragraph(self):
        """A bold *segment* inside a sentence is not a heading."""
        text = "He is **very strong** today.\nMore prose here."
        blocks = _extract_blocks(text)
        assert [k for k, _ in blocks] == ["para"]


# ════════════════════════════════════════════════════════════════════════
#  Table rendering
# ════════════════════════════════════════════════════════════════════════

class TestRenderTable:

    @staticmethod
    def _unfence(piece: str) -> list[str]:
        """Strip one outer code fence, returning the interior lines."""
        lines = piece.strip().splitlines()
        assert lines[0].startswith("```") and lines[-1].strip() == "```"
        return lines[1:-1]

    def test_alignment_and_fence(self):
        name, pieces = _render_table(["Attribute", "Score"], [["STR", "8"], ["DEX", "14"]])
        assert "Attribute" in name and "Score" in name
        assert len(pieces) == 1
        # self-contained: the piece opens AND closes its own fence so it can
        # never be stranded unterminated when split across fields
        assert pieces[0].startswith("```") and pieces[0].endswith("```")
        lines = self._unfence(pieces[0])
        # header + separator + 2 data rows
        assert len(lines) == 4
        # every line (incl. trailing padding) is the same width → columns align
        widths = {len(l) for l in lines}
        assert len(widths) == 1, f"unaligned table: {widths}"
        # cells land under the right columns
        assert "STR" in lines[2] and "8" in lines[2]
        assert "DEX" in lines[3] and "14" in lines[3]

    def test_ragged_rows_padded(self):
        name, pieces = _render_table(["A", "B"], [["x"]])
        assert len(pieces) == 1
        assert pieces[0].startswith("```") and pieces[0].endswith("```")

    def test_empty_table(self):
        name, pieces = _render_table([], [])
        assert name == "" and pieces == []

    def test_wide_table_fits_line_width_and_splints(self):
        """Regression: the armor-price table (6 cols, ~120 chars wide) used to
        render at 120-char lines — too wide for Discord, so columns wrapped
        and misaligned, and the >1024-char field was then split mid-fence.
        """
        header = ["Armor", "Armor Class (AC)", "Strength", "Stealth", "Weight", "Cost"]
        rows = [
            ["**Padded Armor**", "11 + Dex modifier", "—", "Disadvantage", "8 lb.", "5 GP"],
            ["**Leather Armor**", "11 + Dex modifier", "—", "—", "10 lb.", "10 GP"],
            ["**Studded Leather Armor**", "12 + Dex modifier", "—", "—", "13 lb.", "45 GP"],
            ["**Hide Armor**", "12 + Dex modifier (max 2)", "—", "—", "12 lb.", "10 GP"],
            ["**Chain Shirt**", "13 + Dex modifier (max 2)", "—", "—", "20 lb.", "50 GP"],
            ["**Scale Mail**", "14 + Dex modifier (max 2)", "—", "Disadvantage", "45 lb.", "50 GP"],
            ["**Breastplate**", "14 + Dex modifier (max 2)", "—", "—", "20 lb.", "400 GP"],
            ["**Half Plate Armor**", "15 + Dex modifier (max 2)", "—", "Disadvantage", "40 lb.", "750 GP"],
            ["**Ring Mail**", "14", "—", "Disadvantage", "40 lb.", "30 GP"],
            ["**Chain Mail**", "16", "Str 13", "Disadvantage", "55 lb.", "75 GP"],
            ["**Splint Armor**", "17", "Str 15", "Disadvantage", "60 lb.", "200 GP"],
            ["**Plate Armor**", "18", "Str 15", "Disadvantage", "65 lb.", "1,500 GP"],
            ["**Shield**", "+2", "—", "—", "6 lb.", "10 GP"],
        ]
        name, pieces = _render_table(header, rows)
        assert len(pieces) >= 2  # 6 columns don't fit one piece → column groups
        for piece in pieces:
            # every piece is a complete fenced block ≤ the field cap
            assert piece.startswith("```") and piece.endswith("```")
            assert len(piece) <= MAX_FIELD_VALUE
            lines = self._unfence(piece)
            # each column-group piece starts with its own header + separator
            # line, so it reads standalone
            assert len(lines) >= 3
            assert set(lines[1].replace(" ", "").replace("|", "")) == {"-"}
            widths = {len(l) for l in lines}
            assert len(widths) == 1, f"unaligned table: {widths}"
            # rows stay within Discord's practical monospace line width
            assert max(len(l) for l in lines) <= 65, [len(l) for l in lines]
        # no row lost across the column-group split
        joined = "\n".join(p for p in pieces)
        for armor in ("Padded Armor", "Plate Armor", "Shield"):
            assert armor in joined

    def test_very_wide_cell_wraps_without_loss(self):
        """A 900-char cell wraps inside its column (never truncated)."""
        header = ["A", "B"]
        rows = [["x" * 900, "y"], ["z" * 900, "w"]]
        name, pieces = _render_table(header, rows)
        for p in pieces:
            assert len(p) <= MAX_FIELD_VALUE
        joined = "\n".join(pieces)
        # every character of both cells survives (wrapped across lines)
        assert joined.count("x") >= 900, f"lost x: {joined.count('x')}"
        assert joined.count("z") >= 900, f"lost z: {joined.count('z')}"

    def test_very_wide_table_hard_split_no_content_lost(self):
        """A cell longer than the whole field budget is hard-split across
        continuation pieces with no content lost."""
        header = ["A", "B"]
        rows = [["x" * 1200, "y"]]
        name, pieces = _render_table(header, rows)
        assert len(pieces) > 1
        for p in pieces:
            assert p.startswith("```") and p.rstrip().endswith("```")
            assert len(p) <= MAX_FIELD_VALUE
        joined = "\n".join(pieces)
        assert joined.count("x") >= 1200, f"lost x: {joined.count('x')}"


# ════════════════════════════════════════════════════════════════════════
#  build_embed — structure
# ════════════════════════════════════════════════════════════════════════

class TestBuildEmbed:

    def test_character_sheet_structure(self):
        e = build_embed(CHARACTER_SHEET)
        assert e is not None
        _assert_within_limits(e)
        # H1 becomes the title
        assert "Trixy Smoldersome" in e.title
        # section headings are folded into the field name of the block they
        # introduce (heading + table/list share one field)
        names = [f.name for f in e.fields]
        assert any(n.startswith("Attributes") for n in names), names
        assert any(n.startswith("Equipment") for n in names), names
        assert any(n.startswith("Notes") for n in names), names
        # table rendered as self-contained fenced monospace block(s)
        table_fields = [f for f in e.fields if "STR" in f.value]
        assert table_fields, "no table field found"
        assert all(f.value.startswith("```") and f.value.endswith("```")
                   for f in table_fields)
        assert "-1" in table_fields[0].value and "+3" in table_fields[0].value
        # code block preserved verbatim in a field
        code_fields = [f for f in e.fields if "scrap_gadget" in f.value]
        assert len(code_fields) == 1
        assert "```python" in code_fields[0].value or "```" in code_fields[0].value

    def test_h1_beats_title_override(self):
        """A content heading is more specific than a character label."""
        e = build_embed("# Golem Stats\n" + "x " * 50, title_override="System")
        assert e.title == "Golem Stats"

    def test_title_override_used_without_h1(self):
        e = build_embed("plain prose " * 20, title_override="System")
        assert e is not None and e.title == "System"

    def test_prose_reply_has_no_fields(self):
        """Plain conversation replies stay a clean description (no fields)."""
        prose = ("Orcus is a legendary demon lord of the undead. He commands "
                 "legions of wights, ghouls, and his iconic hound Mordant. "
                 "His lair lies in the Nine Hells on the plane of the Damned.")
        e = build_embed(prose)
        assert e is not None
        assert e.fields == []
        assert "Orcus" in (e.description or "")

    def test_default_color(self):
        e = build_embed("## S\n- a\n- b\n- c\n- d\n- e\n- f\n- g\n- h\nNote long enough for the gate to open here ok?")
        assert e is not None
        assert e.color.value == 0x96BF6B  # D&D Beyond green

    def test_custom_color(self):
        e = build_embed("## S\n- a\n- b\n- c\n- d\n- e\n- f\n- g\n- h\nNote long enough for the gate to open here ok?", color=0xFF0000)
        assert e is not None and e.color.value == 0xFF0000


# ════════════════════════════════════════════════════════════════════════
#  build_embed — fallbacks & limits
# ════════════════════════════════════════════════════════════════════════

class TestBuildEmbedFallbacks:

    @pytest.mark.parametrize("text", ["", "   ", "\n\n  \n"])
    def test_empty_returns_none(self, text):
        assert build_embed(text) is None
        assert build_embeds_for_channel(text) == []

    def test_tiny_reply_returns_none(self):
        """A 3-word reply doesn't benefit from an embed."""
        assert build_embed("ok.") is None
        assert build_embeds_for_channel("Sure, MASTER.") == []

    def test_never_raises_on_garbage(self):
        for text in ["|" * 500, "#" * 300, "```" * 100, "\t\n\r" * 200]:
            result = build_embed(text)  # must not raise
            assert result is None or hasattr(result, "fields")

    def test_very_long_prose_within_limits(self):
        text = "This is a sentence. " * 300
        e = build_embed(text)
        assert e is not None
        _assert_within_limits(e)

    def test_many_sections_within_field_cap(self):
        """40 sections must not exceed the 25-field cap (single embed)."""
        text = "\n\n".join(f"## Section {i}\n- item a\n- item b\n- item c" for i in range(40))
        e = build_embed(text)
        assert e is not None
        _assert_within_limits(e)


# ════════════════════════════════════════════════════════════════════════
#  Multi-embed splitting — content preservation
# ════════════════════════════════════════════════════════════════════════

class TestMultiEmbed:

    def test_long_prose_no_content_lost(self):
        text = " ".join(f"Sentence number {i} talks about something. " for i in range(200))
        embeds = build_embeds_for_channel(text)
        assert len(embeds) > 1
        joined = _all_embed_text(embeds)
        missing = [i for i in range(200) if f"Sentence number {i} talks" not in joined]
        assert not missing, f"content lost: {missing[:10]}"
        for e in embeds:
            _assert_within_limits(e)

    def test_many_sections_no_content_lost(self):
        text = "\n\n".join(f"## Section {i}\n- item a\n- item b\n- item c" for i in range(40))
        embeds = build_embeds_for_channel(text)
        joined = _all_embed_text(embeds)
        missing = [i for i in range(40) if f"Section {i}" not in joined]
        assert not missing, f"sections lost: {missing[:10]}"
        for e in embeds:
            _assert_within_limits(e)

    ARMOR_TABLE_REPLY = (
        "MASTER, here is the table showing all the prices for different "
        "types of armors from the HumbleWood campaign rules:\n\n"
        "### Armor\n\n"
        "| Armor | Armor Class (AC) | Strength | Stealth | Weight | Cost |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        "| **Padded Armor** | 11 + Dex modifier | — | Disadvantage | 8 lb. | 5 GP |\n"
        "| **Leather Armor** | 11 + Dex modifier | — | — | 10 lb. | 10 GP |\n"
        "| **Studded Leather Armor** | 12 + Dex modifier | — | — | 13 lb. | 45 GP |\n"
        "| **Hide Armor** | 12 + Dex modifier (max 2) | — | — | 12 lb. | 10 GP |\n"
        "| **Chain Shirt** | 13 + Dex modifier (max 2) | — | — | 20 lb. | 50 GP |\n"
        "| **Scale Mail** | 14 + Dex modifier (max 2) | — | Disadvantage | 45 lb. | 50 GP |\n"
        "| **Breastplate** | 14 + Dex modifier (max 2) | — | — | 20 lb. | 400 GP |\n"
        "| **Half Plate Armor** | 15 + Dex modifier (max 2) | — | Disadvantage | 40 lb. | 750 GP |\n"
        "| **Ring Mail** | 14 | — | Disadvantage | 40 lb. | 30 GP |\n"
        "| **Chain Mail** | 16 | Str 13 | Disadvantage | 55 lb. | 75 GP |\n"
        "| **Splint Armor** | 17 | Str 15 | Disadvantage | 60 lb. | 200 GP |\n"
        "| **Plate Armor** | 18 | Str 15 | Disadvantage | 65 lb. | 1,500 GP |\n"
        "| **Shield** | +2 | — | — | 6 lb. | 10 GP |"
    )

    SPECIES_REPLY = (
        "MASTER, here are the specific species for HumbleWood divided by "
        "their folk group:\n\n"
        "**Birdfolk Species**\n"
        "*   Corvum\n"
        "*   Gallus (with subraces: Bright Gallus, Huden Gallus)\n"
        "*   Luma (with variants: Sable Luma, Sera Luma)\n"
        "*   Raptor\n"
        "*   Strig\n\n"
        "**Humblefolk Species**\n"
        "*   Cervan\n"
        "*   Jerbeen\n"
        "*   Hedge\n"
        "*   Vulpin\n"
        "*   Mapach"
    )

    def test_bold_labels_stay_with_their_lists(self):
        """Regression: the birdfolk/humblefolk reply rendered both bold
        labels into the description while the two lists became separate
        unnamed fields — so the headline and contents didn't match up.
        Each label must now be a field name directly above its list.
        """
        embeds = build_embeds_for_channel(self.SPECIES_REPLY, title_override="System")
        assert embeds, "no embed produced"
        for e in embeds:
            _assert_within_limits(e)
        # The labels must be field names, not description text
        all_names = [f.name for e in embeds for f in e.fields]
        assert any("Birdfolk Species" in n for n in all_names), all_names
        assert any("Humblefolk Species" in n for n in all_names), all_names
        desc = " ".join(e.description or "" for e in embeds)
        assert "Birdfolk Species" not in desc
        assert "Humblefolk Species" not in desc
        # Each label is folded into the field name of its own list, and all 5
        # entries sit in that same field's value.
        fields = [f for e in embeds for f in e.fields]
        bird_val = next(f.value for f in fields if f.name.startswith("Birdfolk Species"))
        humble_val = next(f.value for f in fields if f.name.startswith("Humblefolk Species"))
        for sp in ("Corvum", "Gallus", "Luma", "Raptor", "Strig"):
            assert sp in bird_val, f"{sp} missing from birdfolk field"
        for sp in ("Cervan", "Jerbeen", "Hedge", "Vulpin", "Mapach"):
            assert sp in humble_val, f"{sp} missing from humblefolk field"

    # The 01:05 reply used a bold label WITH a trailing parenthetical
    # annotation — the whole line is not bold-only, so the earlier fix did not
    # catch it.  Both labels collapsed into the description and the two lists
    # became anonymous fields (the "wrong order" / mismatched headline bug).
    ANNOTATED_SPECIES_REPLY = (
        "MASTER, here are the two major folk groups in Humblewood and their "
        "respective species:\n\n"
        "**Birdfolk** (Avian features, live in settlements called perches)\n"
        "*   **Corvum**\n"
        "*   **Gallus**\n"
        "*   **Luma**\n"
        "*   **Raptor**\n"
        "*   **Strig**\n\n"
        "**Humblefolk** (Furred forms, live close to the forest floor; no "
        "shared languages/history like Birdfolk)\n"
        "*   **Cervan**\n"
        "*   **Jerbeen**\n"
        "*   **Hedge**\n"
        "*   **Vulpin**\n"
        "*   **Mapach**"
    )

    def test_annotated_bold_labels_stay_with_their_lists(self):
        """A bold label followed by a parenthetical annotation must still be a
        section heading so its list is attached to it — not dumped into the
        description while the list becomes an anonymous field."""
        embeds = build_embeds_for_channel(self.ANNOTATED_SPECIES_REPLY, title_override="System")
        assert embeds, "no embed produced"
        for e in embeds:
            _assert_within_limits(e)
        all_names = [f.name for e in embeds for f in e.fields]
        assert any(n.startswith("Birdfolk") for n in all_names), all_names
        assert any(n.startswith("Humblefolk") for n in all_names), all_names
        # The annotation is preserved in the heading, not dropped.
        bird_name = next(n for n in all_names if n.startswith("Birdfolk"))
        assert "Avian features" in bird_name, bird_name
        desc = " ".join(e.description or "" for e in embeds)
        assert "Birdfolk" not in desc
        assert "Humblefolk" not in desc
        # Each list sits under its own labelled field.
        fields = [f for e in embeds for f in e.fields]
        bird_val = next(f.value for f in fields if f.name.startswith("Birdfolk"))
        humble_val = next(f.value for f in fields if f.name.startswith("Humblefolk"))
        for sp in ("Corvum", "Gallus", "Luma", "Raptor", "Strig"):
            assert sp in bird_val, f"{sp} missing from birdfolk field"
        for sp in ("Cervan", "Jerbeen", "Hedge", "Vulpin", "Mapach"):
            assert sp in humble_val, f"{sp} missing from humblefolk field"

    def test_bold_with_prose_after_stays_paragraph(self):
        """A bold label followed by ordinary prose (no parens) is NOT a heading,
        so we don't over-capture real sentences as section dividers."""
        blocks = _extract_blocks("**Note:** see the rules below for details.")
        assert [k for k, _ in blocks] == ["para"]

    def test_armor_table_reply_renders_cleanly(self):
        """Regression: the real armor-price reply used to produce an
        unterminated fence (piece 1) and a header-less misaligned block
        (piece 2).  Every table field must now be a complete, aligned,
        standalone-readable fenced block with no content lost.
        """
        embeds = build_embeds_for_channel(self.ARMOR_TABLE_REPLY, title_override="System")
        assert embeds, "no embed produced for the armor table reply"
        for e in embeds:
            _assert_within_limits(e)
            for f in e.fields:
                v = f.value
                # a fenced field must close its own fence
                if v.startswith("```"):
                    assert v.rstrip().endswith("```"), f"unterminated fence: {v[:80]!r}"
                # every interior line of a table block is the same width
                if "|" in v and v.startswith("```"):
                    lines = v.strip().splitlines()[1:-1]
                    if len(lines) >= 2:
                        widths = {len(l) for l in lines}
                        assert len(widths) == 1, f"unaligned: {widths}"
        joined = _all_embed_text(embeds)
        for armor in ("Padded Armor", "Studded Leather Armor", "Half Plate Armor",
                      "Ring Mail", "Splint Armor", "Plate Armor", "Shield"):
            assert armor in joined, f"lost row: {armor}"

    ROGUE_TABLE_REPLY = (
        "MASTER!\n\n"
        "Here is the **Rogue Features** table from the character creation rules:\n\n"
        "**Rogue Features**\n\n"
        "| Level | Proficiency Bonus | Class Features | Thieves' Cant (Levels) | Cantrips | Prepared Spells | Spell Slots per Spell Level | | | | | | | |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        "| **1** | +2 | Cunning Action, Thief, Roguish Archetype, Sneak Attack | 0 | 2 | 3 | — | — | — | — | — | — |\n"
        "| **5** | +3 | Fast Rogues, Thieves' Cant (2) | 2 | 2 | 6 | 2 | 2 | 2 | — | — | — |\n"
        "| **12** | +4 | Ability Score Improvement | 3 | 2 | 13 | 2 | 2 | 2 | 1 | 1 | 1 |\n"
        "| **20** | +6 | Master of Shadows | 5 | 2 | 22 | 2 | 2 | 2 | 1 | 1 | 1 |"
    )

    def test_rogue_table_junk_columns_stripped(self):
        """Regression: the Rogue reply's table had 8 trailing padding columns
        (empty header, dash cells) that made it unreadable.  Junk columns must
        be dropped; real data preserved; and the repeated bold label must fold
        into the field name instead of duplicating above it.
        """
        embeds = build_embeds_for_channel(self.ROGUE_TABLE_REPLY, title_override="System")
        assert embeds, "no embed produced for the rogue table reply"
        for e in embeds:
            _assert_within_limits(e)
        joined = _all_embed_text(embeds)
        # real content survives
        for token in ("Cunning Action", "Master of Shadows", "+2", "+6"):
            assert token in joined, f"lost data: {token}"
        # the junk padding columns are gone — a rendered table row should not
        # end with a run of empty cells or lone dashes
        for e in embeds:
            for f in e.fields:
                if not (f.value.startswith("```") and "|" in f.value):
                    continue
                for line in f.value.strip().splitlines()[1:-1]:
                    assert not line.rstrip().endswith("|  |"), f"junk column: {line!r}"
        # the bold label is folded into the table field name, not repeated as
        # a separate anonymous label field
        names = [f.name for e in embeds for f in e.fields]
        assert any(n.startswith("Rogue Features") and "Level" in n for n in names), names
        assert not any(f.value.strip() == " " for e in embeds for f in e.fields)

    def test_pathological_single_line_preserved(self):
        """A 20k-char line with no breaks at all must not lose data."""
        text = "x" * 20000
        embeds = build_embeds_for_channel(text)
        assert len(embeds) > 1
        total = sum(len(f.value) for e in embeds for f in e.fields) + \
                sum(len(e.description or "") for e in embeds)
        assert total >= 20000 - 100  # allow tiny fence overhead variance

    def test_title_override_applies_to_first_embed(self):
        text = " ".join(f"Sentence number {i} talks about something. " for i in range(200))
        embeds = build_embeds_for_channel(text, title_override="My Custom Title")
        assert embeds[0].title == "My Custom Title"

    def test_split_chunks_respect_target(self):
        text = "\n\n".join(f"Section {i} body " * 20 for i in range(50))
        chunks = _split_into_chunks(text)
        assert all(len(c) <= 3500 + 10 for c in chunks), [len(c) for c in chunks]


# ════════════════════════════════════════════════════════════════════════
#  Code-block splitting (self-contained fences)
# ════════════════════════════════════════════════════════════════════════

class TestCodeBlockSplitting:

    def test_short_block_single_piece(self):
        pieces = _split_code_block("def f():\n    return 1")
        assert len(pieces) == 1
        assert pieces[0].startswith("```") and pieces[0].endswith("```")
        assert "return 1" in pieces[0]

    def test_long_block_every_piece_self_contained(self):
        code = "\n".join(f"line_number_{i} = {i}" for i in range(400))
        pieces = _split_code_block(code)
        assert len(pieces) > 1
        for p in pieces:
            assert p.startswith("```") and p.rstrip().endswith("```")
            assert len(p) <= MAX_FIELD_VALUE
        joined = "\n".join(pieces)
        for i in (0, 200, 399):
            assert f"line_number_{i}" in joined, f"lost line {i}"

    def test_language_tag_preserved(self):
        code = "```python\n" + "x = 1\n" * 500 + "```"
        pieces = _split_code_block(code)
        assert len(pieces) > 1
        for p in pieces:
            assert p.startswith("```python") and p.rstrip().endswith("```")
        assert "x = 1" in "\n".join(pieces)

    def test_single_huge_line_hard_split(self):
        code = "z" * 5000
        pieces = _split_code_block(code)
        assert len(pieces) > 1
        for p in pieces:
            assert p.startswith("```") and p.rstrip().endswith("```")
            assert len(p) <= MAX_FIELD_VALUE
        assert "z" * 5000 == "".join(l.strip() for p in pieces for l in p.strip().splitlines()[1:-1])


# ════════════════════════════════════════════════════════════════════════
#  send_long_response_embedded — delivery helper
# ════════════════════════════════════════════════════════════════════════

class _FakeInteraction:
    """Minimal stand-in for discord.Interaction (has followup.send)."""

    def __init__(self):
        self.sent = []
        outer = self

        class _Followup:
            async def send(self, content=None, **kw):
                outer.sent.append((content, kw))
                return MagicMock(id=1)

        self.followup = _Followup()


class _FakeMessage:
    """Minimal stand-in for discord.Message (has reply, NO followup)."""

    def __init__(self):
        self.sent = []

    async def reply(self, content=None, **kw):
        self.sent.append((content, kw))
        return MagicMock(id=2)


class _BoomMessage:
    async def reply(self, content=None, **kw):
        raise RuntimeError("boom")


@pytest.mark.asyncio
class TestSendLongResponseEmbedded:

    async def test_single_embed_uses_embed_kwarg(self):
        from utils.response_splitter import send_long_response_embedded

        src = _FakeInteraction()
        ok = await send_long_response_embedded(src, CHARACTER_SHEET, "System")
        assert ok is True
        assert len(src.sent) == 1
        content, kw = src.sent[0]
        assert content is None
        assert "embed" in kw and "embeds" not in kw
        _assert_within_limits(kw["embed"])

    async def test_long_reply_batches_into_one_message(self):
        from utils.response_splitter import send_long_response_embedded

        text = "\n\n".join(
            f"## Part {i}\nLong paragraph about part {i} with enough words to be substantial here."
            for i in range(1, 40)
        )
        src = _FakeInteraction()
        ok = await send_long_response_embedded(src, text, "System")
        assert ok is True
        # All embeds batched into ONE message (≤10 per message on Discord).
        assert len(src.sent) == 1
        n = len(src.sent[0][1].get("embeds", []))
        assert 1 < n <= 10

    async def test_message_reply_path(self):
        from utils.response_splitter import send_long_response_embedded

        src = _FakeMessage()
        ok = await send_long_response_embedded(src, CHARACTER_SHEET, "System")
        assert ok is True
        content, kw = src.sent[0]
        assert content is None and "embed" in kw

    async def test_small_reply_returns_false_without_sending(self):
        from utils.response_splitter import send_long_response_embedded

        src = _FakeMessage()
        ok = await send_long_response_embedded(src, "ok.", "System")
        assert ok is False
        assert src.sent == []  # caller will fall back to plain text

    async def test_api_error_returns_false_never_raises(self):
        from utils.response_splitter import send_long_response_embedded

        ok = await send_long_response_embedded(_BoomMessage(), CHARACTER_SHEET, "System")
        assert ok is False


# ════════════════════════════════════════════════════════════════════════
#  /ai command wiring (non-streaming path)
# ════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestAICommandEmbedWiring:

    def _make_ix(self):
        ix = MagicMock()
        ix.guild_id = 123456
        ix.channel_id = 789012
        ix.response = MagicMock(is_done=MagicMock(return_value=True))
        ix.channel = MagicMock()
        ix._sent = []

        async def fake_followup(content=None, **kw):
            ix._sent.append((content, kw))
            return MagicMock(id=7)

        ix.followup.send = MagicMock(side_effect=fake_followup)
        return ix

    async def test_embed_enabled_uses_embed_path(self):
        from pathlib import Path
        from commands.ai_command import handle_ai_command
        from config.characters import load_characters
        import config.settings as settings
        from unittest.mock import patch, AsyncMock

        load_characters(Path("characters.json.example"))
        ix = self._make_ix()

        with patch.object(settings, "EMBED_FORMAT", True), \
             patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock,
                   return_value=(CHARACTER_SHEET, {})):
            await handle_ai_command(ix, message="sheet please")

        # Exactly one followup, carrying an embed (no plain text).
        assert len(ix._sent) == 1
        content, kw = ix._sent[0]
        assert content is None
        assert "embed" in kw
        assert "Trixy Smoldersome" in kw["embed"].title

    async def test_embed_disabled_uses_plain_text(self):
        from pathlib import Path
        from commands.ai_command import handle_ai_command
        from config.characters import load_characters
        import config.settings as settings
        from unittest.mock import patch, AsyncMock

        load_characters(Path("characters.json.example"))
        ix = self._make_ix()

        with patch.object(settings, "EMBED_FORMAT", False), \
             patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock,
                   return_value=("A plain text reply.", {})):
            await handle_ai_command(ix, message="hi")

        assert len(ix._sent) == 1
        content, kw = ix._sent[0]
        assert "A plain text reply." in str(content)
        assert "embed" not in kw and "embeds" not in kw

    async def test_embed_fallback_when_reply_too_small(self):
        """Tiny replies can't be embeds — must fall back to plain text so
        the user still gets an answer."""
        from pathlib import Path
        from commands.ai_command import handle_ai_command
        from config.characters import load_characters
        import config.settings as settings
        from unittest.mock import patch, AsyncMock

        load_characters(Path("characters.json.example"))
        ix = self._make_ix()

        with patch.object(settings, "EMBED_FORMAT", True), \
             patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock,
                   return_value=("ok.", {})):
            await handle_ai_command(ix, message="hi")

        # Fallback plain-text chunk was sent (not an embed).
        assert len(ix._sent) == 1
        content, kw = ix._sent[0]
        assert "ok." in str(content)
        assert "embed" not in kw


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
