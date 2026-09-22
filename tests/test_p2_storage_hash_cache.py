"""P2 #18: /list_kb_docs must not re-hash unchanged files.

The old ``list_kb_files`` read + SHA-256'd *every* file on every listing, so
a listing cost O(total KB size) even though the user only asked for metadata.
These tests prove the sidecar cache:
  * the first listing populates it (files are read + hashed),
  * a second listing of an unchanged KB makes ZERO ``read_bytes`` calls,
  * returning identical hashes, and
  * a changed file (new size or mtime) is re-hashed exactly once.
"""
from __future__ import annotations

import os
import pathlib

import pytest
from unittest.mock import patch

import kb.storage as S


@pytest.fixture
def kb(tmp_path) -> pathlib.Path:
    root = tmp_path / "kb18"
    root.mkdir()
    (root / "a.txt").write_text("alpha content one")
    (root / "b.md").write_text("# beta\nsecond doc")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("nested doc three")
    return root


class _ReadCounter:
    """Patch ``Path.read_bytes`` to count how many times it is called."""

    def __init__(self):
        self.count = 0
        self._patcher = None

    def __enter__(self):
        # ``counting`` is a function placed on ``Path``, so its first arg is the
        # Path instance — reference the counter through the closure, not `self`.
        counter = self
        orig = pathlib.Path.read_bytes

        def counting(path_self, *a, **k):
            counter.count += 1
            return orig(path_self, *a, **k)

        self._patcher = patch.object(pathlib.Path, "read_bytes", counting)
        self._patcher.start()
        return self

    def __exit__(self, *a):
        if self._patcher is not None:
            self._patcher.stop()


class TestListKBFilesCache:

    def test_second_listing_makes_zero_reads(self, kb):
        first = S.list_kb_files(kb)
        assert len(first) == 3, "3 visible docs (hidden cache file excluded)"
        with _ReadCounter() as rc:
            second = S.list_kb_files(kb)
        assert rc.count == 0, "unchanged KB must make ZERO read_bytes calls"
        # Same metadata, byte-for-byte.
        assert first == second

    def test_hashes_match_reference(self, kb):
        """Cached hashes must equal a fresh, independent hash of the content."""
        listing = {d["filename"]: d["sha256"] for d in S.list_kb_files(kb)}
        for name, content in [
            ("a.txt", "alpha content one"),
            ("b.md", "# beta\nsecond doc"),
            ("c.txt", "nested doc three"),
        ]:
            expected = __import__("hashlib").sha256(content.encode()).hexdigest()[:16]
            assert listing[name] == expected

    def test_changed_file_is_rehashed_once(self, kb):
        """A content change (new size + mtime) triggers exactly one re-read."""
        S.list_kb_files(kb)  # populate cache

        # Modify a.txt: change content AND bump mtime explicitly (mtime
        # resolution can be coarse; a deliberate bump makes this deterministic).
        target = kb / "a.txt"
        target.write_text("alpha content one CHANGED")
        os.utime(target, (2000000000.0, 2000000000.0))

        with _ReadCounter() as rc:
            listing = S.list_kb_files(kb)
        assert rc.count == 1, "only the changed file should be re-read"
        got = {d["filename"]: d["sha256"] for d in listing}
        expected = __import__("hashlib").sha256(
            b"alpha content one CHANGED").hexdigest()[:16]
        assert got["a.txt"] == expected

    def test_deleted_file_is_pruned(self, kb):
        S.list_kb_files(kb)  # populate (3 keys)
        (kb / "b.md").unlink()
        listing = S.list_kb_files(kb)
        assert [d["filename"] for d in listing] == ["a.txt", "c.txt"]

    def test_cache_file_is_hidden_and_not_listed(self, kb):
        """The sidecar cache lives inside the KB root but is never a 'doc'."""
        S.list_kb_files(kb)
        assert (kb / S.SHA256_CACHE_FILENAME).exists()
        names = [d["filename"] for d in S.list_kb_files(kb)]
        assert S.SHA256_CACHE_FILENAME not in names

    def test_subfolder_scan_does_not_prune_siblings(self, kb):
        """A subfolder listing must not drop root-level files from the cache."""
        full = S.list_kb_files(kb)          # populates cache with 3 keys
        sub = S.list_kb_files(kb, subfolder="sub")
        assert [d["filename"] for d in sub] == ["c.txt"]
        # Root files must still be listed afterwards (cache not pruned wrongly).
        again = S.list_kb_files(kb)
        assert {d["filename"] for d in again} == {"a.txt", "b.md", "c.txt"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
