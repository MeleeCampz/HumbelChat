"""Regression tests for P0 #3 and P0 #4.

P0 #3 — concurrent KB index mutations must not lose chunks.  Before the fix,
       the read → embed → merge → persist sequence in ``KBIndexStore`` spanned
       ``await`` points (embedding calls) with no lock, so two concurrent
       ``/upload_kb`` calls (or ``/upload_kb`` + ``/sync_kb``) both merged
       onto the stale doc list and the last writer won — the first file's
       chunks vanished from memory AND from the on-disk cache.

P0 #4 — blocking KB file IO must not freeze the event loop.  Before the fix
       the keyword retrieval path (rglob + per-file reads) and the
       /upload_kb, /list_kb_docs handlers ran directly on the loop thread.
"""
from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import threading

import pytest


def _fake_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [round(b / 255.0, 6) for b in digest[:8]]


class SlowFakeEmbedder:
    """Embedder that sleeps, so concurrent updates overlap inside the lock."""

    def __init__(self, delay: float = 0.15):
        self.delay = delay
        self.call_count = 0
        self.encoded_texts: list[str] = []

    async def encode(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.encoded_texts.extend(texts)
        await asyncio.sleep(self.delay)  # <- the await point the lock must span
        return [_fake_vector(t) for t in texts]


def _store_for(tmp_path, kb_dir):
    from kb.index import KBIndexStore
    store = KBIndexStore(kb_dir, persist_dir=tmp_path / "index_cache")
    store._embedder = SlowFakeEmbedder()
    return store


def _index_sources(store) -> set[str]:
    idx = store.get_index()
    if idx is None:
        return set()
    return {doc.source().lower() for doc in idx._docs}


def _sqlite_sources(store) -> set[str]:
    if not store.db_path.exists():
        return set()
    conn = sqlite3.connect(str(store.db_path))
    try:
        rows = conn.execute("SELECT source_file FROM document_index").fetchall()
    finally:
        conn.close()
    return {r[0].lower() for r in rows}


# ─────────────────────── P0 #3: concurrent mutation race ───────────────────────

class TestConcurrentIndexMutations:

    @pytest.mark.asyncio
    async def test_two_concurrent_updates_both_survive(self, tmp_path):
        """Two simultaneous /upload_kb calls must NOT drop each other's chunks.

        The embedder sleeps mid-update, guaranteeing the read → embed →
        merge → persist windows overlap.  After the fix (per-store lock),
        both files must be present in the in-memory index AND in SQLite.
        """
        kb = tmp_path / "kb"
        kb.mkdir()

        # Start from an EMPTY index — then two uploads land concurrently,
        # exactly the /upload_kb flow (file written, then auto-indexed).
        store = _store_for(tmp_path, kb)
        await store.load()
        assert _index_sources(store) == set()  # empty KB, nothing embedded yet

        (kb / "alpha.md").write_text("# Alpha\n\nUnique content about alpha systems.")
        (kb / "beta.md").write_text("# Beta\n\nUnique content about beta protocols.")

        results = await asyncio.gather(
            store.update_single_document(kb / "alpha.md"),
            store.update_single_document(kb / "beta.md"),
        )
        assert results == [True, True]

        mem = _index_sources(store)
        assert {"alpha.md", "beta.md"} <= mem, (
            f"lost-update race: in-memory index only has {sorted(mem)}"
        )
        disk = _sqlite_sources(store)
        assert {"alpha.md", "beta.md"} <= disk, (
            f"lost-update race: SQLite cache only has {sorted(disk)}"
        )

    @pytest.mark.asyncio
    async def test_concurrent_sync_and_update_do_not_clobber(self, tmp_path):
        """/sync_kb racing /upload_kb must also keep both files' chunks."""
        kb = tmp_path / "kb"
        kb.mkdir()

        store = _store_for(tmp_path, kb)
        await store.load()
        assert _index_sources(store) == set()

        # Now both files land: /sync_kb scans them while a concurrent
        # /upload_kb indexes delta.
        (kb / "gamma.md").write_text("# Gamma\n\nGamma session notes and tactics.")
        (kb / "delta.md").write_text("# Delta\n\nDelta lore about the delta kingdom.")

        (idx, report), ok = await asyncio.gather(
            store.sync_changes(),
            store.update_single_document(kb / "delta.md"),
        )
        assert ok is True
        assert report["failed"] == []

        mem = _index_sources(store)
        assert {"gamma.md", "delta.md"} <= mem, (
            f"sync+upload race lost chunks: in-memory index has {sorted(mem)}"
        )
        disk = _sqlite_sources(store)
        assert {"gamma.md", "delta.md"} <= disk

    @pytest.mark.asyncio
    async def test_update_serializes_with_lock(self, tmp_path):
        """Updates must run one at a time (lock held across the embed)."""
        kb = tmp_path / "kb"
        kb.mkdir()
        for name in ("one.md", "two.md"):
            (kb / name).write_text(f"# {name}\n\nBody text for {name}.")

        store = _store_for(tmp_path, kb)
        await store.load()

        inside = 0
        peak = 0
        lock = store._mutation_lock()

        original_encode = store._embedder.encode

        async def counting_encode(texts):
            nonlocal inside, peak
            inside += 1
            peak = max(peak, inside)
            try:
                assert lock.locked(), "mutation lock must be held during embedding"
                return await original_encode(texts)
            finally:
                inside -= 1

        store._embedder.encode = counting_encode

        await asyncio.gather(
            store.update_single_document(kb / "one.md"),
            store.update_single_document(kb / "two.md"),
        )
        assert peak == 1, f"embeddings overlapped {peak}x — lock not serializing"
        assert {"one.md", "two.md"} <= _index_sources(store)


# ────────────────── P0 #4: blocking IO must not stall the loop ──────────────────

class TestBlockingIoOffloaded:

    @pytest.mark.asyncio
    async def test_keyword_retrieval_does_not_block_event_loop(self, tmp_path):
        """A slow KB file read must not stop a concurrent heartbeat task.

        Stubs ``Path.read_bytes`` (the hot call inside read_kb_files /
        get_relevant_chunks) to sleep 0.5 s.  If the keyword path still ran
        on the loop, the heartbeat could not tick while the read is in
        flight; with the thread offload it ticks freely.
        """
        import pathlib
        import kb.reader as reader
        from kb import retrievers

        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "sword_guide.txt").write_text(
            "Sword mastery guide. Sword techniques for the rogue class.\n"
            "Sword swings, sword stances, sword training routines.\n"
        )

        read_state = []
        real_read_bytes = pathlib.Path.read_bytes
        # kb.reader uses the same Path class — patch it there explicitly.
        assert reader.pathlib.Path is pathlib.Path

        def slow_read_bytes(self):
            if self.name == "sword_guide.txt":
                read_state.append("start")
                import time
                time.sleep(0.5)  # blocks the WORKER thread, not the loop
                read_state.append("end")
            return real_read_bytes(self)

        reader.pathlib.Path.read_bytes = slow_read_bytes
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            for _ in range(50):
                await asyncio.sleep(0.02)
                ticks += 1

        try:
            hb = asyncio.create_task(heartbeat())
            docs = await retrievers.retrieve_kb_documents(
                "sword techniques", kb, strategy="keyword", top_n=5
            )
            await hb
        finally:
            reader.pathlib.Path.read_bytes = real_read_bytes

        assert read_state[:2] == ["start", "end"], "stubbed read never ran"
        # The read slept 0.5 s; the heartbeat (20 ms ticks, up to 50 = 1 s)
        # must have ticked freely while the read was in flight.
        assert ticks >= 10, (
            f"event loop was blocked: only {ticks} heartbeat ticks "
            "during a 0.5 s blocking read"
        )
        assert docs, "keyword retrieval returned no documents"
        assert any("sword" in name.lower() for name, _ in docs)

    @pytest.mark.asyncio
    async def test_list_kb_files_async_offloads_to_executor(self, tmp_path, monkeypatch):
        """list_kb_files_async must run its scan off the event-loop thread."""
        from kb import storage

        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "doc1.md").write_text("content one")
        (kb / "doc2.md").write_text("content two")

        executed_on = []
        real_list = storage.list_kb_files

        def spy_list(*a, **k):
            executed_on.append(threading.get_ident())
            return real_list(*a, **k)

        async def capture_loop_ident():
            executed_on.append(("loop", threading.get_ident()))

        await capture_loop_ident()
        loop_ident = executed_on[0][1]

        monkeypatch.setattr(storage, "list_kb_files", spy_list)
        docs = await storage.list_kb_files_async(kb)

        assert len(docs) == 2
        work_idents = [i for i in executed_on if isinstance(i, int)]
        assert work_idents, "list_kb_files was never called"
        assert work_idents[0] != loop_ident, (
            "list_kb_files ran on the event-loop thread (blocking IO not offloaded)"
        )

    @pytest.mark.asyncio
    async def test_validate_upload_async_offloads_to_executor(self, tmp_path, monkeypatch):
        """validate_upload_async must write off the event-loop thread."""
        from kb import storage

        kb = tmp_path / "kb"
        kb.mkdir()

        executed_on = []
        real_validate = storage.validate_upload

        def spy_validate(*a, **k):
            executed_on.append(threading.get_ident())
            return real_validate(*a, **k)

        async def capture_loop_ident():
            executed_on.append(("loop", threading.get_ident()))

        await capture_loop_ident()
        loop_ident = executed_on[0][1]

        monkeypatch.setattr(storage, "validate_upload", spy_validate)
        dest, summary = await storage.validate_upload_async(
            b"hello kb", filename="note.txt", kb_path=kb
        )

        assert dest.exists() and summary["size"] == 8
        work_idents = [i for i in executed_on if isinstance(i, int)]
        assert work_idents and work_idents[0] != loop_ident, (
            "validate_upload ran on the event-loop thread (blocking IO not offloaded)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
