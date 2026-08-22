"""P2-5: Tests for the vector index disk cache (kb/index.py) and vector retrieval.

Covers:
  * Fresh build with a fake (offline) embedder
  * Cache HIT on reload — zero embedding API calls
  * Partial invalidation — only changed/new files re-embedded
  * Deleted files dropped from cache
  * Graceful degradation when the embedding backend is down
  * Force rebuild
  * update_single_document / remove_document
  * Legacy (pre-v3) cache schema ignored
  * query_with_embeddings (no double embedding)
  * Atomic persistence (no stray .tmp files)
"""
from __future__ import annotations

import asyncio
import hashlib
import pickle
import sqlite3
from unittest.mock import AsyncMock

import pytest

from kb.index import KBIndexStore, _content_hash, _iter_kb_files
from kb.vector_db import KBVectorIndex


# ──────────────────────────── Fake embedder ──────────────────────────────

def _fake_vector(text: str) -> list[float]:
    """Deterministic 8-dim unit-ish vector derived from the text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vec = [b / 255.0 for b in digest[:8]]
    return [round(v, 6) for v in vec]


class FakeEmbedder:
    """Drop-in replacement for kb.embedder.Embedder that counts API calls."""

    def __init__(self, fail: bool = False):
        self.call_count = 0
        self.encoded_texts: list[str] = []
        self.fail = fail

    async def encode(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.encoded_texts.extend(texts)
        if self.fail:
            raise RuntimeError("embedding backend down (simulated)")
        return [_fake_vector(t) for t in texts]


def install_fake_embedder(store: KBIndexStore, fail: bool = False) -> FakeEmbedder:
    """Patch the embedder on the store AND on every KBVectorIndex instance."""
    fake = FakeEmbedder(fail=fail)
    store._embedder = fake
    KBVectorIndex._embedder = fake
    return fake


def make_store(tmp_path, kb_dir) -> KBIndexStore:
    return KBIndexStore(
        kb_dir,
        persist_dir=tmp_path / "index_cache",
    )


def with_fake(index: KBVectorIndex, store: KBIndexStore | None = None) -> FakeEmbedder:
    """Install a fresh fake embedder on the index (and store, if given)."""
    fake = FakeEmbedder()
    index._embedder = fake
    if store is not None:
        store._embedder = fake
    return fake


# ──────────────────────────── KB fixtures ────────────────────────────────

@pytest.fixture
def kb_dir(tmp_path) -> "pathlib.Path":
    import pathlib
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "time_system.md").write_text(
        "# Time System\n\nThe realm of humblewood runs on a unique time system "
        "with nine moon cycles per year, each lasting thirty-one days."
    )
    (kb / "history.txt").write_text(
        "History of the land: founded by the early settlers, the kingdom grew "
        "slowly over three centuries of trade and expansion."
    )
    return kb


@pytest.fixture
def multi_chunk_kb(tmp_path) -> "pathlib.Path":
    import pathlib
    kb = tmp_path / "kb_multi"
    kb.mkdir()
    # >8000 chars with markdown headers → multiple chunks per file
    sections = []
    for i in range(8):
        sections.append(
            f"# Chapter {i}\n\n" + (f"Section body text for chapter {i}. " * 60) + "\n"
        )
    (kb / "big_book.md").write_text("\n".join(sections))
    (kb / "small.md").write_text("# Small\n\nA short single-chunk document about pikes.")
    return kb


# ──────────────────────────── 1. Fresh build ─────────────────────────────

class TestFreshBuild:
    @pytest.mark.asyncio
    async def test_build_indexes_all_files(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        fake = install_fake_embedder(store)
        idx = await store.load()

        assert not idx.is_empty()
        assert fake.call_count == 1  # one batched API call for all chunks
        # Every file's text went through the embedder
        joined = " ".join(fake.encoded_texts)
        assert "time system" in joined.lower()
        assert "history" in joined.lower()
        assert store.db_path.exists()

    @pytest.mark.asyncio
    async def test_multi_chunk_file_all_chunks_cached(self, multi_chunk_kb, tmp_path):
        store = make_store(tmp_path, multi_chunk_kb)
        install_fake_embedder(store)
        idx = await store.load()

        big_chunks = [d for d in idx._docs if d.source() == "big_book.md"]
        small_chunks = [d for d in idx._docs if d.source() == "small.md"]
        assert len(big_chunks) >= 2, "big_book.md should produce multiple chunks"
        assert len(small_chunks) == 1

        # Reload — ALL chunks of the multi-chunk file must be served
        store2 = make_store(tmp_path, multi_chunk_kb)
        fake2 = install_fake_embedder(store2)
        idx2 = await store2.load()
        assert fake2.call_count == 0
        big2 = [d for d in idx2._docs if d.source() == "big_book.md"]
        assert len(big2) == len(big_chunks)

    @pytest.mark.asyncio
    async def test_missing_kb_path_gives_empty_index(self, tmp_path):
        store = make_store(tmp_path, tmp_path / "no_such_kb")
        install_fake_embedder(store)
        idx = await store.load()
        assert idx.is_empty()
        assert store.db_path.exists()


# ──────────────────────────── 2. Cache HIT ───────────────────────────────

class TestCacheHit:
    @pytest.mark.asyncio
    async def test_reload_is_free_and_identical(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        fake = install_fake_embedder(store)
        idx1 = await store.load()
        first = [(d.display_name, d.content, d.embedding) for d in idx1._docs]

        store2 = make_store(tmp_path, kb_dir)
        fake2 = install_fake_embedder(store2)
        idx2 = await store2.load()

        assert fake2.call_count == 0, "cache HIT must not call the embedder"
        second = [(d.display_name, d.content, d.embedding) for d in idx2._docs]
        assert first == second, "reloaded index must be byte-identical"

    @pytest.mark.asyncio
    async def test_persisted_rows_have_correct_hashes(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        await store.load()

        conn = sqlite3.connect(str(store.db_path))
        rows = conn.execute(
            "SELECT doc_name, content, content_hash FROM document_index"
        ).fetchall()
        conn.close()
        assert rows, "cache table must not be empty"
        for doc_name, content, content_hash in rows:
            assert content_hash == _content_hash(content)


# ────────────────────── 3. Partial invalidation ──────────────────────────

class TestPartialInvalidation:
    @pytest.mark.asyncio
    async def test_only_changed_file_re_embedded(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        await store.load()

        # Change one file on disk.
        (kb_dir / "history.txt").write_text(
            "Completely rewritten history: the kingdom now has a new dynasty "
            "and a rebuilt capital."
        )

        store2 = make_store(tmp_path, kb_dir)
        fake2 = install_fake_embedder(store2)
        idx2 = await store2.load()

        assert fake2.call_count == 1
        # Only the changed file's content was sent to the embedder.
        assert any("new dynasty" in t for t in fake2.encoded_texts)
        assert not any("early settlers" in t for t in fake2.encoded_texts)
        # Unchanged file keeps its original (fake) embedding.
        assert any("early settlers" not in d.content for d in idx2._docs)
        assert any("new dynasty" in d.content for d in idx2._docs)

    @pytest.mark.asyncio
    async def test_new_file_does_not_re_embed_existing(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        idx1 = await store.load()
        baseline = len(idx1._docs)

        (kb_dir / "new_notes.md").write_text(
            "# New Notes\n\nFreshly added notes about the fishing guild roster."
        )

        store2 = make_store(tmp_path, kb_dir)
        fake2 = install_fake_embedder(store2)
        idx2 = await store2.load()

        assert fake2.call_count == 1
        assert len(fake2.encoded_texts) == 1  # only the new file (single chunk)
        assert "fishing guild" in fake2.encoded_texts[0]
        assert idx2.count() == baseline + 1

    @pytest.mark.asyncio
    async def test_deleted_file_dropped_from_index_and_cache(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        idx1 = await store.load()
        baseline = idx1.count()

        (kb_dir / "history.txt").unlink()

        store2 = make_store(tmp_path, kb_dir)
        fake2 = install_fake_embedder(store2)
        idx2 = await store2.load()

        assert fake2.call_count == 0, "no re-embedding needed when a file is deleted"
        assert idx2.count() == baseline - 1
        assert all(d.source() != "history.txt" for d in idx2._docs)

        # Deleted file's rows must be gone from the persisted cache too.
        rows = store2._read_cache_rows()
        assert "history.txt" not in rows


# ────────────────────── 4. Graceful degradation ──────────────────────────

class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_backend_down_serves_cached_subset(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        await store.load()

        (kb_dir / "brand_new.md").write_text("# Brand\n\nBrand new unindexed content.")

        store2 = make_store(tmp_path, kb_dir)
        install_fake_embedder(store2, fail=True)
        idx2 = await store2.load()  # must not raise

        assert not idx2.is_empty(), "cached chunks must still be served"
        assert any("time system" in d.content for d in idx2._docs)

        # Backend back — next load picks up the new file.
        store3 = make_store(tmp_path, kb_dir)
        fake3 = install_fake_embedder(store3)
        idx3 = await store3.load()
        assert fake3.call_count == 1
        assert any("Brand new" in d.content for d in idx3._docs)

    @pytest.mark.asyncio
    async def test_fresh_build_with_backend_down_is_empty(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store, fail=True)
        idx = await store.load()  # must not raise
        assert idx.is_empty()


# ──────────────────────────── 5. Rebuild ─────────────────────────────────

class TestRebuild:
    @pytest.mark.asyncio
    async def test_rebuild_re_embeds_everything(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        await store.load()

        fake = FakeEmbedder()
        store._embedder = fake
        idx = await store.rebuild()
        assert fake.call_count == 1
        assert fake.encoded_texts
        assert not idx.is_empty()


# ──────────────── 6. update_single_document / remove ─────────────────────

class TestUpdateAndRemove:
    @pytest.mark.asyncio
    async def test_update_single_document_replaces_only_that_file(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        fake = install_fake_embedder(store)
        idx = await store.load()
        before = idx.count()
        other_contents = {
            d.content for d in idx._docs if d.source() != "time_system.md"
        }

        # Rewrite one document and update it in place.
        (kb_dir / "time_system.md").write_text(
            "# Time System v2\n\nRewritten time rules: ten moon cycles now."
        )
        ok = await store.update_single_document(kb_dir / "time_system.md")
        assert ok

        idx = store.get_index()
        assert idx.count() == before  # 1 chunk in, 1 chunk out
        # Only the updated file went to the embedder.
        assert any("ten moon cycles" in t for t in fake.encoded_texts)
        # Other documents' embeddings are untouched.
        assert {d.content for d in idx._docs if d.source() != "time_system.md"} == other_contents
        assert any("ten moon cycles" in d.content for d in idx._docs)

    @pytest.mark.asyncio
    async def test_update_single_document_new_file_adds(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        idx = await store.load()
        before = idx.count()

        (kb_dir / "added.txt").write_text("Added document: notes on the river trade routes.")
        ok = await store.update_single_document(kb_dir / "added.txt")
        assert ok
        assert store.get_index().count() == before + 1

    @pytest.mark.asyncio
    async def test_update_missing_file_returns_false(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        await store.load()
        assert await store.update_single_document(tmp_path / "ghost.txt") is False

    @pytest.mark.asyncio
    async def test_remove_document(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        idx = await store.load()
        before = idx.count()

        assert await store.remove_document(kb_dir / "history.txt") is True
        assert store.get_index().count() == before - 1
        assert all(d.source() != "history.txt" for d in store.get_index()._docs)

        # Removing again (nothing left) returns False.
        assert await store.remove_document(kb_dir / "history.txt") is False


# ────────────────────── 7. Legacy schema tolerance ───────────────────────

class TestLegacyCache:
    @pytest.mark.asyncio
    async def test_legacy_schema_ignored_and_rebuilt(self, kb_dir, tmp_path):
        # Simulate a pre-v3 cache: no content_hash / source_file columns.
        cache_dir = tmp_path / "index_cache"
        cache_dir.mkdir()
        db = cache_dir / "vector_index.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            """CREATE TABLE document_index (
                   id INTEGER PRIMARY KEY,
                   doc_name TEXT NOT NULL,
                   content TEXT NOT NULL,
                   embedding BLOB,
                   updated_at REAL
               )"""
        )
        conn.execute(
            """CREATE TABLE metadata (
                   key TEXT PRIMARY KEY,
                   value TEXT NOT NULL
               )"""
        )
        conn.execute(
            "INSERT INTO document_index (doc_name, content, embedding, updated_at) "
            "VALUES (?, ?, ?, 0.0)",
            ("time_system.md [Full Document]", "stale legacy content", pickle.dumps([0.1] * 8)),
        )
        conn.commit()
        conn.close()

        store = make_store(tmp_path, kb_dir)
        fake = install_fake_embedder(store)
        idx = await store.load()

        assert not idx.is_empty()
        assert fake.call_count == 1, "legacy cache must be treated as a miss"
        assert not any("stale legacy content" in d.content for d in idx._docs)
        # Cache has been rewritten with the v3 schema.
        cols = {
            r[1]
            for r in sqlite3.connect(str(db)).execute("PRAGMA table_info(document_index)")
        }
        assert "content_hash" in cols and "source_file" in cols


# ──────────────────── 8. query_with_embeddings ───────────────────────────

class TestQueryWithEmbeddings:
    @pytest.mark.asyncio
    async def test_query_returns_matching_doc_and_vector(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        fake = install_fake_embedder(store)
        idx = await store.load()

        fake = with_fake(idx, store)
        results, q_emb = await idx.query_with_embeddings("time system", top_n=3)

        assert fake.call_count == 1, "exactly one embedding call per query"
        assert q_emb == _fake_vector("time system")
        assert results, "expected at least one hit"
        top_name, top_content, top_score = results[0]
        assert "time_system" in top_name
        assert top_score > 0

    @pytest.mark.asyncio
    async def test_query_empty_on_empty_index(self, tmp_path):
        idx = KBVectorIndex()
        results, q_emb = await idx.query_with_embeddings("anything")
        assert results == [] and q_emb == []

    @pytest.mark.asyncio
    async def test_query_with_embeddings_failsoft(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        idx = await store.load()
        install_fake_embedder(store, fail=True)
        results, q_emb = await idx.query_with_embeddings("time system")
        assert results == [] and q_emb == []


# ──────────────────────────── 9. Persistence ─────────────────────────────

class TestPersistence:
    @pytest.mark.asyncio
    async def test_no_temp_file_left_behind(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        await store.load()
        await store.shutdown()
        assert store.db_path.exists()
        assert not store.db_path.with_suffix(".tmp").exists()

    @pytest.mark.asyncio
    async def test_metadata_records_schema_version(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        await store.load()
        conn = sqlite3.connect(str(store.db_path))
        meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        conn.close()
        assert meta["schema_version"] == "3"
        assert str(store.kb_path) == meta["kb_path"]


# ──────────────────────── 10. _iter_kb_files sanity ──────────────────────

class TestIterKBFiles:
    def test_only_indexable_extensions(self, tmp_path):
        kb = tmp_path / "kb_ext"
        kb.mkdir()
        (kb / "a.md").write_text("a")
        (kb / "b.txt").write_text("b")
        (kb / "c.png").write_text("c")
        (kb / "d.py").write_text("d")
        (kb / "secret?.md").write_text("e")
        files = _iter_kb_files(kb)
        names = {p.name for p in files}
        assert names == {"a.md", "b.txt"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
