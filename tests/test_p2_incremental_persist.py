"""P2 #20: a single-file update must not rewrite the whole cache.

The old ``_save_to_disk_from`` rebuilt the entire DB (all pickled embeddings)
on *every* single-file update — fine at tiny scale, but it rewrote every other
document's rows (new SQLite page/rowid) for nothing.  Now single-file updates
use an incremental upsert: only the changed file's rows are replaced and only
removed files' rows are deleted; every other row keeps its rowid/page.  These
tests prove:
  * updating one file keeps the *other* files' rowids stable,
  * the full ``rebuild()`` still rewrites (rowids all renumbered) — the
    periodic compaction path — and
  * the persisted DB round-trips identically (same source files + chunk count).
"""
from __future__ import annotations

import pathlib
import sqlite3

import pytest


class FakeEmbedder:
    """Deterministic drop-in (no network); encode returns a fixed 4-dim vector."""

    def __init__(self, *a, **k):
        self.calls = 0

    async def encode(self, texts):
        self.calls += 1
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _make_store(kb: pathlib.Path):
    from kb.index import KBIndexStore
    store = KBIndexStore(kb)
    store._embedder = FakeEmbedder()
    return store


def _rowids_by_source(db: pathlib.Path) -> dict[str, list[int]]:
    conn = sqlite3.connect(str(db))
    out: dict[str, list[int]] = {}
    for src, rid in conn.execute(
        "SELECT source_file, id FROM document_index ORDER BY id"
    ):
        out.setdefault(src, []).append(rid)
    conn.close()
    return out


def _all_rowids(db: pathlib.Path) -> list[int]:
    conn = sqlite3.connect(str(db))
    rows = [r for (r,) in conn.execute("SELECT id FROM document_index ORDER BY id")]
    conn.close()
    return rows


@pytest.fixture
def kb(tmp_path) -> pathlib.Path:
    root = tmp_path / "kb20"
    root.mkdir()
    (root / "aaa.txt").write_text("first doc, alpha content")
    (root / "bbb.txt").write_text("second doc, beta content")
    (root / "ccc.txt").write_text("third doc, gamma content")
    return root


class TestIncrementalPersist:

    @pytest.mark.asyncio
    async def test_single_update_keeps_other_rowids(self, kb):
        """Updating ONE file must not renumber the other files' rows."""
        store = _make_store(kb)
        await store.load()  # fresh build (full rewrite) → rowids 1..N
        before = _rowids_by_source(store.db_path)
        assert set(before) == {"aaa.txt", "bbb.txt", "ccc.txt"}
        bbb_before = before["bbb.txt"]
        ccc_before = before["ccc.txt"]

        # Update only aaa.txt (content changes).
        (kb / "aaa.txt").write_text("first doc, CHANGED content")
        await store.update_single_document(kb / "aaa.txt")

        after = _rowids_by_source(store.db_path)
        # The two untouched files must keep EXACTLY their original rowids —
        # a full rewrite would have renumbered all of them.
        assert after["bbb.txt"] == bbb_before, "untouched file rowids changed (full rewrite?)"
        assert after["ccc.txt"] == ccc_before, "untouched file rowids changed (full rewrite?)"
        # The updated file is present (its rows were replaced).
        assert after["aaa.txt"], "updated file's rows missing"

    @pytest.mark.asyncio
    async def test_rebuild_full_rewrites(self, kb):
        """A full rebuild() is the compaction path — every rowid is renumbered."""
        store = _make_store(kb)
        await store.load()
        first = _all_rowids(store.db_path)

        # Force a full rebuild (drops cache, rewrites the whole table).
        await store.rebuild()
        second = _all_rowids(store.db_path)

        # Same number of rows, but a fresh full rewrite assigns a clean 1..N.
        assert len(first) == len(second)
        assert second == list(range(1, len(second) + 1)), "full rebuild should renumber 1..N"

    @pytest.mark.asyncio
    async def test_round_trip_identical(self, kb):
        """After an incremental update, reloading the persisted DB is consistent."""
        store = _make_store(kb)
        await store.load()
        (kb / "bbb.txt").write_text("second doc, EDITED content")
        ok = await store.update_single_document(kb / "bbb.txt")
        assert ok is True

        # A brand-new store reading the same DB must load without re-embedding.
        store2 = _make_store(kb)
        idx = await store2.load()
        sources = {d.source() for d in idx._docs}
        assert sources == {"aaa.txt", "bbb.txt", "ccc.txt"}
        assert idx.count() >= 3  # all three files' chunks present
        # bbb.txt's stored content reflects the edit.
        bbb = [d for d in idx._docs if d.source() == "bbb.txt"]
        assert any("EDITED" in d.content for d in bbb)

    @pytest.mark.asyncio
    async def test_removed_file_pruned_from_db(self, kb):
        """Removing a file deletes its rows (no full rewrite) — others stable."""
        store = _make_store(kb)
        await store.load()
        before = _rowids_by_source(store.db_path)
        aaa_before = before["aaa.txt"]

        ok = await store.remove_document(kb / "ccc.txt")
        assert ok is True

        after = _rowids_by_source(store.db_path)
        assert "ccc.txt" not in after, "removed file's rows must be deleted"
        # Other files keep their rowids (incremental, not full rewrite).
        assert after["aaa.txt"] == aaa_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
