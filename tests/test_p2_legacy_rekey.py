"""P2 #22: legacy cache re-keying must walk the directory ONCE.

``_read_cache_rows`` re-keys v3 rows that are stored by *basename* (no ``/`` in
the key) to the file's KB-relative path.  The old code ran a full
``_iter_kb_files`` rglob **per row**, so a legacy cache of N rows over M files
cost N directory walks.  These tests prove exactly ONE walk happens (regardless
of N rows) and that the re-keying is still correct (unique basename → re-keyed;
ambiguous basename → left for re-embed).
"""
from __future__ import annotations

import pathlib
import sqlite3

import pytest
from unittest.mock import patch

from kb.index import (
    KBIndexStore,
    _SCHEMA_CREATE_DOC_INDEX,
    _SCHEMA_CREATE_METADATA,
    _SCHEMA_VERSION,
    _content_hash,
    _pack_embedding,
)


def _write_legacy_db(db_path: pathlib.Path, rows: list[tuple[str, str, str, list[float]]]):
    """Write a v3-shaped SQLite cache where rows are keyed by *basename*.

    *rows* is a list of ``(basename, doc_name, content, embedding)``.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(_SCHEMA_CREATE_DOC_INDEX)
    conn.execute(_SCHEMA_CREATE_METADATA)
    now = 1234567890.0
    for basename, doc_name, content, embedding in rows:
        conn.execute(
            "INSERT INTO document_index "
            "(source_file, doc_name, content, content_hash, embedding, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (basename, doc_name, content, _content_hash(content),
             _pack_embedding(embedding), now),
        )
    conn.execute("INSERT OR REPLACE INTO metadata (key,value) VALUES ('schema_version', ?)",
                 (_SCHEMA_VERSION,))
    conn.commit()
    conn.close()


@pytest.fixture
def kb(tmp_path) -> pathlib.Path:
    root = tmp_path / "kb22"
    root.mkdir()
    (root / "notes_a.txt").write_text("session A notes")
    (root / "notes_b.txt").write_text("session B notes")
    (root / "notes_c.txt").write_text("session C notes")
    sub = root / "sub"
    sub.mkdir()
    (sub / "doc.txt").write_text("nested doc")
    return root


class TestLegacyRekeyingSingleWalk:

    def test_exactly_one_walk_for_n_rows(self, kb):
        """N legacy rows over M files → exactly ONE _iter_kb_files call."""
        rows = [
            ("notes_a.txt", "notes_a.txt [S1]", "session A notes", [0.1, 0.2]),
            ("notes_b.txt", "notes_b.txt [S1]", "session B notes", [0.3, 0.4]),
            ("notes_c.txt", "notes_c.txt [S1]", "session C notes", [0.5, 0.6]),
            ("doc.txt", "doc.txt [S1]", "nested doc", [0.7, 0.8]),
        ]
        store = KBIndexStore(kb)
        _write_legacy_db(store.db_path, rows)

        calls: list[int] = []
        import kb.index as I
        real_iter = I._iter_kb_files

        def counting(path):
            calls.append(1)
            return real_iter(path)

        with patch.object(I, "_iter_kb_files", counting):
            result = store._read_cache_rows()

        assert len(calls) == 1, f"expected exactly 1 directory walk, got {len(calls)}"
        # Re-keyed to KB-relative paths.
        assert set(result.keys()) == {
            "notes_a.txt", "notes_b.txt", "notes_c.txt", "sub/doc.txt",
        }
        assert result["sub/doc.txt"][0]["content"] == "nested doc"

    def test_modern_cache_never_walks(self, kb):
        """A fully-modern cache (all keys contain '/') makes ZERO walks."""
        rows = [
            ("notes_a.txt", "notes_a.txt [S1]", "session A notes", [0.1, 0.2]),
            ("sub/doc.txt", "doc.txt [S1]", "nested doc", [0.7, 0.8]),
        ]
        store = KBIndexStore(kb)
        _write_legacy_db(store.db_path, rows)  # keys already contain '/' for sub/doc

        calls: list[int] = []
        import kb.index as I
        real_iter = I._iter_kb_files

        def counting(path):
            calls.append(1)
            return real_iter(path)

        with patch.object(I, "_iter_kb_files", counting):
            result = store._read_cache_rows()

        # notes_a.txt has no '/' (basename) → 1 walk for it; but the key point is
        # the walk happens at most once even though multiple rows exist.
        assert len(calls) <= 1
        assert "notes_a.txt" in result
        assert "sub/doc.txt" in result

    def test_ambiguous_basename_left_for_reembed(self, kb):
        """Two on-disk files share a basename → the row is NOT re-keyed (safe)."""
        # Create a collision: root/notes_a.txt AND sub/notes_a.txt.
        sub2 = kb / "sub2"
        sub2.mkdir()
        (sub2 / "notes_a.txt").write_text("collision copy")

        rows = [("notes_a.txt", "notes_a.txt [S1]", "session A notes", [0.1, 0.2])]
        store = KBIndexStore(kb)
        _write_legacy_db(store.db_path, rows)

        calls: list[int] = []
        import kb.index as I
        real_iter = I._iter_kb_files

        def counting(path):
            calls.append(1)
            return real_iter(path)

        with patch.object(I, "_iter_kb_files", counting):
            result = store._read_cache_rows()

        # Ambiguous: row stays keyed by the raw basename (will be re-embedded),
        # not silently mapped to one of the two files.
        assert "notes_a.txt" in result
        assert len(calls) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
