"""P2 #19: an unchanged file must NOT be re-chunked on index load.

The old load path re-read and re-chunked *every* file just to compare against
the cache — so a "cache HIT, 0 API calls" still paid the full re-chunk CPU.
Now each source file's SHA-256 is stored in the cache; if it matches disk the
chunks are guaranteed valid (the chunker is deterministic) and re-chunking is
skipped entirely.  These tests prove:
  * a fresh build stores a per-file ``file_hash`` in the DB,
  * an unchanged reload makes ZERO ``Chunker.split_file_sync`` (re-chunk) calls,
  * a modified file is still detected (hash mismatch) and re-chunked.
"""
from __future__ import annotations

import hashlib
import pathlib

import pytest
from unittest.mock import patch

import kb.index as I


class FakeEmbedder:
    """Deterministic drop-in for kb.embedder.Embedder (no network)."""

    def __init__(self, *a, **k):
        self.calls = 0

    async def encode(self, texts):
        self.calls += 1
        # One fixed 4-dim vector per input (content-agnostic; we only need shape).
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _make_store(kb: pathlib.Path):
    store = I.KBIndexStore(kb)
    store._embedder = FakeEmbedder()  # no network; encode returns fixed vectors
    return store


@pytest.fixture
def kb(tmp_path) -> pathlib.Path:
    root = tmp_path / "kb19"
    root.mkdir()
    (root / "one.txt").write_text("first document content, stays the same")
    (root / "two.txt").write_text("second document content, will be edited")
    return root


class TestFileHashSkipRechunk:

    @pytest.mark.asyncio
    async def test_build_stores_file_hash(self, kb):
        """A fresh build must persist a file_hash that matches the file bytes."""
        store = _make_store(kb)
        await store.load()
        import sqlite3
        conn = sqlite3.connect(str(store.db_path))
        fh = {row[0]: row[1] for row in conn.execute(
            "SELECT source_file, file_hash FROM document_index")}
        conn.close()
        expected_one = hashlib.sha256(
            (kb / "one.txt").read_bytes()).hexdigest()
        # Both files present with a correct hash for at least one of them.
        assert fh.get("one.txt") == expected_one
        assert fh.get("two.txt") == hashlib.sha256(
            (kb / "two.txt").read_bytes()).hexdigest()

    @pytest.mark.asyncio
    async def test_unchanged_reload_rechunks_zero(self, kb):
        """Reload of an UNCHANGED KB must make zero re-chunk calls (fast path)."""
        store = _make_store(kb)
        await store.load()  # fresh build → populates cache w/ file_hash

        store2 = _make_store(kb)
        import kb.chunker as C
        real_split = C.Chunker.split_file_sync
        rechunks = []

        def spy(file_path):
            rechunks.append(file_path)
            return real_split(file_path)

        # _rechunk calls Chunker.split_file_sync off-loop; if the fast path
        # (file-hash match) works, it is never reached for an unchanged KB.
        with patch.object(C.Chunker, "split_file_sync", staticmethod(spy)):
            idx = await store2.load()

        assert rechunks == [], (
            f"unchanged KB must not re-chunk; got {len(rechunks)} re-chunk call(s)")
        assert idx is not None and idx.count() >= 2  # chunks came from cache

    @pytest.mark.asyncio
    async def test_modified_file_is_rechunked(self, kb):
        """Changing a file's content must invalidate its cache entry (re-chunk)."""
        store = _make_store(kb)
        await store.load()

        # Edit two.txt → its file hash changes.
        (kb / "two.txt").write_text("second document content, CHANGED now")

        store3 = _make_store(kb)
        import kb.chunker as C
        real_split = C.Chunker.split_file_sync
        rechunks = []

        def spy(file_path):
            rechunks.append(file_path)
            return real_split(file_path)

        with patch.object(C.Chunker, "split_file_sync", staticmethod(spy)):
            await store3.load()

        # Only the changed file should be re-chunked.
        assert [pathlib.Path(p).name for p in rechunks] == ["two.txt"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
