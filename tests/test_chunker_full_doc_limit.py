"""Chunker full-document cap (code review §2.3).

The "keep small files intact" strategy used a hardcoded 8000 chars while the
recursive splitter caps chunks at MAX_CHUNK_SIZE=7500 — a 7500-8000 char file
became ONE chunk that silently exceeded the 2048-token embedder window.
FULL_DOC_CHAR_LIMIT is now pinned to MAX_CHUNK_SIZE.
"""
from __future__ import annotations

from kb.chunker import Chunker


def _write(tmp_path, text: str):
    p = tmp_path / "doc.md"
    p.write_text(text, encoding="utf-8")
    return p


class TestFullDocLimit:
    def test_limit_pinned_to_max_chunk_size(self):
        assert Chunker.FULL_DOC_CHAR_LIMIT == Chunker.MAX_CHUNK_SIZE == 7500

    def test_just_under_limit_stays_intact(self, tmp_path):
        text = ("word " * 1400).strip()  # ~7000 chars, no headers
        assert len(text) < Chunker.FULL_DOC_CHAR_LIMIT
        chunks = Chunker.split_file_sync(_write(tmp_path, text))
        assert len(chunks) == 1

    def test_gap_zone_now_splits(self, tmp_path):
        # 7800 chars used to be a single oversized chunk.
        text = ("word " * 1560).strip()  # ~7800 chars
        assert Chunker.MAX_CHUNK_SIZE < len(text) < 8000
        chunks = Chunker.split_file_sync(_write(tmp_path, text))
        assert len(chunks) >= 2
        assert all(len(c.content) <= Chunker.MAX_CHUNK_SIZE for c in chunks)
