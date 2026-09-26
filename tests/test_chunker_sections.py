"""Unit tests for per-section chunking in kb.chunker (one chunk per section).

Regression coverage for the packing bug where a single tiny section (< MIN_CHUNK_SIZE)
seeded an accumulator that then bundled every subsequent section up to MAX_CHUNK_SIZE —
collapsing e.g. 352 spell headers into ~48 chunks of ~7 spells each, so a lookup like
"Fireball" could only retrieve Fireball *plus* its neighbours as one blob.
"""
from kb.chunker import Chunker

MIN = Chunker.MIN_CHUNK_SIZE  # 80
MAX = Chunker.MAX_CHUNK_SIZE  # 7500


def _spell(name: str, body_len: int = 150) -> str:
    """Build a small `#### Name` section of roughly *body_len* chars of content."""
    head = "Casting Time: 1 action. Range: 150 feet. "
    pad = max(0, body_len - len(head))
    return f"#### {name}\n\n{head}{'x' * pad}"


class TestPerSectionChunking:
    def test_each_section_becomes_its_own_chunk(self):
        names = ["Fireball", "Lightning Bolt", "Misty Step", "Counterspell", "Shield"]
        body = "\n\n".join(_spell(n) for n in names)
        chunks = Chunker._split_by_headers(body, "spells.md", "spells.md")

        # One chunk per spell — no packing.
        assert len(chunks) == len(names)

        # Fireball is isolated: exactly one chunk owns it and it holds no neighbour's name.
        fb = [c for c in chunks if "Fireball" in c.section_path]
        assert len(fb) == 1
        other_names = set(names) - {"Fireball"}
        assert not any(n in fb[0].content for n in other_names)

    def test_no_packing_across_large_sections(self):
        # 10 sections of ~1000 chars each: total far exceeds MAX, yet each must stay its own chunk.
        body = "\n\n".join(_spell(f"Spell{i}", body_len=1000) for i in range(10))
        chunks = Chunker._split_by_headers(body, "spells.md", "spells.md")
        assert len(chunks) == 10
        assert all(len(c.content) <= MAX for c in chunks)

    def test_tiny_fragment_is_folded_not_orphaned(self):
        big_a = _spell("BigAlpha", body_len=300)
        tiny = "#### Tiny\n\nNote."  # content well below MIN after the header
        big_b = _spell("BigBeta", body_len=300)
        body = "\n\n".join([big_a, tiny, big_b])
        chunks = Chunker._split_by_headers(body, "spells.md", "spells.md")

        # No standalone chunk may fall below MIN (the tiny fragment is folded in).
        assert all(len(c.content) >= MIN for c in chunks)
        # The tiny note must survive somewhere.
        assert any("Note." in c.content for c in chunks)

    def test_oversized_single_section_is_hard_split(self):
        huge = "#### Enormous\n\n" + "\n".join(f"line {i} of the enormous section" for i in range(400))
        body = huge  # a single section far larger than MAX
        chunks = Chunker._split_by_headers(body, "spells.md", "spells.md")
        assert len(chunks) >= 2
        assert all(len(c.content) <= MAX for c in chunks)
        # The header line is preserved in the first piece.
        assert "Enormous" in chunks[0].content

    def test_preamble_folded_into_first_section(self):
        preamble = "Intro line under 80 chars."  # below MIN
        body = preamble + "\n\n" + _spell("FirstSpell", body_len=200)
        chunks = Chunker._split_by_headers(body, "spells.md", "spells.md")
        # The preamble is attached to the first spell's chunk, not emitted as an orphan.
        assert "FirstSpell" in chunks[0].content and "Intro line" in chunks[0].content
