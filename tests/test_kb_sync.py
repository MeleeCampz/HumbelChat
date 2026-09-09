"""Tests for /sync_kb: KBIndexStore.sync_changes() + the command handler.

sync_changes() is the fast, diff-based alternative to a full rebuild: it only
re-embeds files that are new, renamed, changed, or were deleted on disk
(covering documents placed into the KB folder outside of /upload_kb).
"""
from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from tests._shared import Interaction


# ──────────────────────────── Fake embedder ────────────────────────────

class FakeEmbedder:
    """Drop-in for kb.embedder.Embedder — deterministic, no network."""

    def __init__(self, model_name: str = "fake", *, batch_size: int = 8) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.calls: list[list[str]] = []

    async def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


def make_store(kb_path: pathlib.Path, embedder: FakeEmbedder | Exception) -> "object":
    """Build a KBIndexStore with the embedding layer stubbed out."""
    from kb.index import KBIndexStore

    if isinstance(embedder, Exception):
        class ExplodingEmbedder:
            def __init__(self, *a, **k): pass
            async def encode(self, texts): raise embedder

        cls = ExplodingEmbedder
    else:
        cls = FakeEmbedder

    with patch("kb.index.Embedder", cls):
        return KBIndexStore(kb_path, model_name="fake")


def _write(kb: pathlib.Path, name: str, content: str) -> None:
    (kb / name).write_text(content, encoding="utf-8")


# ──────────────────────── KBIndexStore.sync_changes ────────────────────────

class TestSyncChanges:

    @pytest.mark.asyncio
    async def test_added_and_renamed_and_removed(self, tmp_path):
        """New file, rename (delete+add same content), and deletion in one pass."""
        kb = tmp_path / "kb"
        kb.mkdir()
        _write(kb, "a.md", "alpha content")
        _write(kb, "b.txt", "beta content")
        _write(kb, "old.md", "gamma content")

        embedder = FakeEmbedder()
        store = make_store(kb, embedder)
        idx = await store.load()
        assert idx.count() == 3  # all files indexed on first load

        # Disk changes (simulating manual file management):
        (kb / "b.txt").unlink()                      # deletion
        (kb / "old.md").unlink()                     # ...+ same content re-added
        _write(kb, "new.md", "gamma content")        # as a new name = rename
        _write(kb, "c.md", "delta content")          # brand-new file

        idx, report = await store.sync_changes()

        assert report["added"] == ["c.md"]
        assert report["renamed"] == [("old.md", "new.md")]
        assert report["changed"] == []
        assert report["removed"] == ["b.txt"]
        assert report["failed"] == []
        assert report["changed_count"] == 2          # c.md + new.md re-embedded
        assert report["ok"] is True
        # a.md (1) + c.md (1) + new.md (1); old.md/b.txt rows are gone
        assert idx.count() == 3

        # No duplicate rows for the renamed file's old name:
        rows = store._read_cache_rows()
        assert set(rows) == {"a.md", "c.md", "new.md"}

    @pytest.mark.asyncio
    async def test_rename_with_trailing_newline_detected(self, tmp_path):
        """Renamed files often gain/lose a trailing newline — still a rename."""
        kb = tmp_path / "kb"
        kb.mkdir()
        _write(kb, "orig.md", "same body content\n")

        store = make_store(kb, FakeEmbedder())
        await store.load()

        (kb / "orig.md").unlink()
        _write(kb, "renamed.md", "same body content")  # no trailing newline

        idx, report = await store.sync_changes()
        assert report["renamed"] == [("orig.md", "renamed.md")]
        assert report["added"] == []
        assert report["changed"] == []
        assert idx.count() == 1
        assert set(store._read_cache_rows()) == {"renamed.md"}

    @pytest.mark.asyncio
    async def test_changed_file_reembedded_once(self, tmp_path):
        """Editing a file in place re-embeds it (single replace, no dupes)."""
        kb = tmp_path / "kb"
        kb.mkdir()
        _write(kb, "doc.md", "first version")

        embedder = FakeEmbedder()
        store = make_store(kb, embedder)
        await store.load()

        _write(kb, "doc.md", "second version — edited content")
        idx, report = await store.sync_changes()

        assert report["changed"] == ["doc.md"]
        assert report["added"] == [] and report["removed"] == []
        assert report["changed_count"] == 1
        assert idx.count() == 1

        rows = store._read_cache_rows()
        assert set(rows) == {"doc.md"}
        assert rows["doc.md"][0]["content"] == "second version — edited content"

    @pytest.mark.asyncio
    async def test_no_changes_is_noop(self, tmp_path):
        """Untouched folder: nothing re-embedded, no extra API calls."""
        kb = tmp_path / "kb"
        kb.mkdir()
        _write(kb, "a.md", "alpha content")

        embedder = FakeEmbedder()
        store = make_store(kb, embedder)
        await store.load()
        calls_before = len(embedder.calls)

        idx, report = await store.sync_changes()

        assert report["changed_count"] == 0
        assert report["ok"] is True
        assert len(embedder.calls) == calls_before  # zero embedding calls
        assert idx.count() == 1

    @pytest.mark.asyncio
    async def test_embedder_down_keeps_old_index_and_cache(self, tmp_path):
        """Backend failure must not wipe previously indexed chunks (see the
        empty-cache incident from a failed full rebuild)."""
        kb = tmp_path / "kb"
        kb.mkdir()
        _write(kb, "a.md", "alpha content")

        store = make_store(kb, FakeEmbedder())
        await store.load()
        assert store.get_index().count() == 1

        # Swap in a failing embedder, then add a new file:
        with patch.object(store, "_embedder", _raise_embedder()):
            _write(kb, "b.md", "beta content")
            idx, report = await store.sync_changes()

        assert report["failed"] == ["b.md"]
        assert report["ok"] is False
        assert report["changed_count"] == 1
        # Old chunks survive in memory AND on disk (cache not emptied):
        assert idx.count() == 1
        assert set(store._read_cache_rows()) == {"a.md"}

    @pytest.mark.asyncio
    async def test_deletion_only_updates_index(self, tmp_path):
        """Deleting a file drops its rows even though nothing needs embedding."""
        kb = tmp_path / "kb"
        kb.mkdir()
        _write(kb, "a.md", "alpha content")
        _write(kb, "b.md", "beta content")

        store = make_store(kb, FakeEmbedder())
        await store.load()
        assert store.get_index().count() == 2

        (kb / "b.md").unlink()
        idx, report = await store.sync_changes()

        assert report["removed"] == ["b.md"]
        assert report["changed_count"] == 0
        assert idx.count() == 1
        assert set(store._read_cache_rows()) == {"a.md"}


def _raise_embedder():
    class _Bad:
        async def encode(self, texts):
            raise RuntimeError("backend down")
    return _Bad()


# ──────────────────────────── /sync_kb handler ────────────────────────────

class TestSyncKbCommand:

    @pytest.mark.asyncio
    async def test_no_changes(self):
        ix = Interaction()
        with patch("kb.retrievers.sync_kb_store", AsyncMock(return_value=(MagicIdx(0), {"changed_count": 0, "ok": True}))):
            from commands.kb_commands import handle_sync_kb
            await handle_sync_kb(ix)
        assert any("Nothing to do" in s for s in ix._sent)

    @pytest.mark.asyncio
    async def test_report_lists_all_categories(self):
        ix = Interaction()
        report = {
            "added": ["c.md"],
            "renamed": [("old.md", "new.md")],
            "changed": ["doc.md"],
            "removed": ["b.txt"],
            "failed": [],
            "changed_count": 3,
            "ok": True,
        }
        with patch("kb.retrievers.sync_kb_store", AsyncMock(return_value=(MagicIdx(42), report))):
            from commands.kb_commands import handle_sync_kb
            await handle_sync_kb(ix)

        msg = next(s for s in ix._sent if "KB sync" in s)
        assert "re-indexed **3**" in msg
        assert "🆕 added: `c.md`" in msg
        assert "🔁 renamed: `old.md` → `new.md`" in msg
        assert "✏️ changed: `doc.md`" in msg
        assert "🗑️ removed from index: `b.txt`" in msg
        assert "**42** chunk(s)" in msg
        assert "backend may be down" not in msg

    @pytest.mark.asyncio
    async def test_failure_warning_shown(self):
        ix = Interaction()
        report = {
            "added": [], "renamed": [], "changed": [], "removed": [],
            "failed": ["x.md"],
            "changed_count": 1,
            "ok": False,
        }
        with patch("kb.retrievers.sync_kb_store", AsyncMock(return_value=(MagicIdx(5), report))):
            from commands.kb_commands import handle_sync_kb
            await handle_sync_kb(ix)
        assert any("backend may be down" in s and "`x.md`" in s for s in ix._sent)

    @pytest.mark.asyncio
    async def test_exception_reported(self):
        ix = Interaction()
        with patch("kb.retrievers.sync_kb_store", AsyncMock(side_effect=RuntimeError("boom"))):
            from commands.kb_commands import handle_sync_kb
            await handle_sync_kb(ix)
        assert any("KB sync failed" in s and "boom" in s for s in ix._sent)

    def test_handler_exists(self):
        from commands.kb_commands import handle_sync_kb
        assert callable(handle_sync_kb)


class MagicIdx:
    """Minimal KBVectorIndex stand-in for handler tests."""

    def __init__(self, count: int):
        self._count = count

    def count(self) -> int:
        return self._count

    def is_empty(self) -> bool:
        return self._count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
