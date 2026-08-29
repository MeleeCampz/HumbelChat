"""Tests for select_ranked_chunks — the vector path's chunk selection.

Regression context: the Rogue table bug.  The vector index ranked
``classes.md [Rogue Class Features]`` (a self-contained section with the
full features table) at the top, but the old code discarded ranked content
and re-windowed the raw file by keyword anchors — missing the section and
making the model answer "the KB does not contain the Rogue table".

These tests pin the contract: ranked index chunks are served directly,
multiple per file, with per-file caps, for both structured (header-split)
and unstructured ("Full Document", e.g. player session logs) data.
"""
from kb.retrievers import select_ranked_chunks


def _ranked(*items):
    """Build a ranked list from (name, content, score) tuples."""
    return [(n, c, s) for n, c, s in items]


class TestSelectRankedChunks:
    def test_empty_input(self):
        assert select_ranked_chunks([]) == []

    def test_single_chunk_returned_verbatim(self):
        ranked = _ranked(("classes.md [Rogue Class Features]", "### Rogue Class Features\n<table>...</table>", 0.9))
        docs = select_ranked_chunks(ranked, top_n=5)
        assert len(docs) == 1
        name, content = docs[0]
        assert name == "classes.md"
        # Content must be the exact index chunk — no re-windowing.
        assert content == "### Rogue Class Features\n<table>...</table>"

    def test_rogue_scenario_table_chunk_survives(self):
        """The regression: a top-ranked self-contained table section must
        reach the context even when other files also match."""
        ranked = _ranked(
            ("character-creation.md [Full Document]", "Class Overview ... Rogue ... Dexterity", 0.86),
            ("classes.md [Rogue Class Features]", "### Rogue Class Features\n**Rogue Features**\n<table>Level|Proficiency Bonus|Class Features|Sneak Attack</table>", 0.85),
            ("monsters-A-Z.md [Traits]", "**Speed** 30 ft.", 0.70),
            ("equipment.md [Weapons]", "## Weapons\nlongsword", 0.65),
        )
        docs = select_ranked_chunks(ranked, top_n=4)
        by_name = dict(docs)
        assert "classes.md" in by_name
        # The table itself must be present in the context for classes.md.
        assert "Sneak Attack" in by_name["classes.md"]
        # All four files survive (one entry per file).
        assert len(docs) == 4

    def test_multiple_chunks_per_file_in_rank_order(self):
        ranked = _ranked(
            ("classes.md [Rogue Class Features]", "table", 0.9),
            ("other.md [Full Document]", "x", 0.85),
            ("classes.md [Level 1: Sneak Attack]", "sneak detail", 0.8),
            ("classes.md [Becoming a Rogue …]", "becoming text", 0.7),
        )
        docs = dict(select_ranked_chunks(ranked, top_n=5))
        # All three classes.md chunks are kept, in rank order.
        assert docs["classes.md"] == "table\n\nsneak detail\n\nbecoming text"

    def test_per_file_chunk_cap(self):
        ranked = [(f"big.md [Section {i}]", f"chunk{i}" * 10, 0.9 - i * 0.01) for i in range(8)]
        docs = dict(select_ranked_chunks(ranked, top_n=5))
        body = docs["big.md"]
        # Only the first MAX_CHUNKS_PER_FILE (5) chunks are kept.
        assert "chunk0" in body and "chunk4" in body
        assert "chunk5" not in body

    def test_per_file_char_cap(self):
        big = "A" * 20_000
        small = "B" * 100
        ranked = _ranked(
            ("big.md [Section 1]", big, 0.9),
            ("big.md [Section 2]", big, 0.85),   # would exceed per-file cap
            ("big.md [Section 3]", small, 0.8),  # fits under the cap → kept
        )
        docs = dict(select_ranked_chunks(ranked, top_n=5))
        body = docs["big.md"]
        assert body.count("A" * 20_000) == 1
        assert "B" * 100 in body

    def test_unstructured_player_log_full_document(self):
        """Player session logs are stored as a single 'Full Document' chunk.
        Selection must not assume markdown headers or line windows."""
        log = (
            "# Session: The first session\n\n## Notes\n"
            "- (2026-08-29 15:04) We found the secret key to the dungeon. "
            "(by MeleeChan)\n- (2026-08-29 15:17) We are at a place called everdell.\n"
        )
        ranked = _ranked(
            ("2026-08-29_01_The first session.md [Full Document]", log, 0.88),
            ("spells.md [Fireball]", "### Fireball\n3rd-level evocation", 0.60),
        )
        docs = dict(select_ranked_chunks(ranked, top_n=5))
        assert "2026-08-29_01_The first session.md" in docs
        # The whole unstructured log is served verbatim.
        assert docs["2026-08-29_01_The first session.md"] == log

    def test_duplicate_names_deduplicated(self):
        ranked = _ranked(
            ("a.md [Sec]", "one", 0.9),
            ("a.md [Sec]", "one", 0.85),  # same display name twice
        )
        docs = dict(select_ranked_chunks(ranked, top_n=5))
        assert docs["a.md"] == "one"

    def test_top_n_limits_total_files(self):
        ranked = [(f"f{i}.md [Sec]", f"content{i}", 0.9 - i * 0.01) for i in range(10)]
        docs = select_ranked_chunks(ranked, top_n=3)
        # At most top_n files; each file at most its chunk cap.
        assert len(docs) <= 3
