"""Persistent vector index for KB document retrieval.

Caches the in-memory ``KBVectorIndex`` to disk (SQLite) so bot restarts
don't require re-indexing the entire knowledge base — saving seconds of
startup time and avoiding repeated API calls to the embedding backend.

Concurrency (P0 #3)
-------------------
Every public index mutation — ``load`` / ``rebuild`` / ``update_single_document``
/ ``remove_document`` / ``sync_changes`` / ``shutdown`` — is serialized behind
a per-store ``asyncio.Lock`` held for the whole read → embed → merge →
persist sequence.  Without it, two concurrent ``/upload_kb`` (or
``/upload_kb`` + ``/sync_kb``) calls would each read the old doc list, embed,
and merge onto the *stale* read, and the last writer would win — silently
dropping the other file's chunks from memory **and** from the on-disk cache.
Queries (``get_index``) are read-only and run without the lock, so retrieval
latency is unaffected.

Blocking IO (P0 #4)
-------------------
Filesystem walks, SQLite writes and embedding serialization are offloaded to a
worker thread (``asyncio.to_thread``) at every async boundary so a large KB
scan never freezes the event loop (gateway, typing indicators, voice).

Design
------
* Every chunk row stores a SHA-256 hash of its **content**.  On load, any
  chunk whose file is missing on disk or whose content hash changed is
  dropped; the remaining rows are reused as-is (cache HIT — no API calls).
* Files that are absent from the cache (new/changed) are re-chunked and
  re-embedded, then merged into the cache and the in-memory index.
* Embedding failures degrade gracefully: previously cached chunks stay
  usable, only the missing pieces fall back to keyword retrieval.

Usage
-----
    from kb.index import KBIndexStore

    # Default cache dir is <KB_PATH>/.vector_index_cache (pass persist_dir
    # to override).
    store = KBIndexStore("path/to/kb")

    # Load (uses disk cache when valid, else builds & saves)
    idx = await store.load()
    results = await idx.query("time system", top_n=5)

    # Incremental updates
    await store.update_single_document("new_doc.md")   # re-indexes only this file
    await store.remove_document("old_doc.txt")         # removes from index
    idx, report = await store.sync_changes()           # cheap diff-based sync (added/renamed/changed/removed)
    await store.rebuild()                              # full rebuild (force)

"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import array
import pathlib
import re
import sqlite3
import time

from kb.vector_db import KBVectorIndex, _DocEntry
from kb.embedder import Embedder

logger = logging.getLogger("kb.index")

# ──────────────────────────── Constants ────────────────────────────────

# File extensions eligible for KB indexing (must match vector_db.py).
KB_FILE_EXTENSIONS = {".txt", ".md", ".csv", ".html", ".xml", ".rtf"}
_MAX_BYTES_PER_FILE = 1024 * 1024  # 1 MB
# (size guard lives in vector_db.Chunker — keep indexing behavior in one place)

# ──────────────────────────── Schema ────────────────────────────────────

_SCHEMA_CREATE_DOC_INDEX = """\
CREATE TABLE IF NOT EXISTS document_index (
    id          INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,          -- original filename (cache bookkeeping)
    doc_name    TEXT NOT NULL,          -- display name "file [Section]"
    content     TEXT NOT NULL,
    content_hash TEXT NOT NULL,         -- sha256 of content
    file_hash   TEXT,                   -- P2 #19: sha256 of the SOURCE file (skip re-chunk)
    embedding   BLOB,                   -- packed float64 vector (not pickle)
    updated_at  REAL DEFAULT (strftime('%s','now'))
);
"""

_SCHEMA_CREATE_METADATA = """\
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_SCHEMA_VERSION = "5"  # v4 file hash + v5 packed embeddings and model identity


# ──────────────────────────── Helpers ────────────────────────────────────

def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _file_hash_bytes(data: bytes) -> str:
    """P2 #19: SHA-256 of a source file's raw bytes (already in memory).

    Reuses the exact bytes the chunker/chunk-fallback read, so the stored
    "file hash" and the content actually chunked are guaranteed consistent —
    a re-chunk later compares against the same content that produced the chunks.
    """
    return hashlib.sha256(data).hexdigest()


def _read_text_sync(path: pathlib.Path) -> str:
    """Whole-file read + decode (run via ``asyncio.to_thread`` — P0 #4)."""
    return path.read_bytes().decode("utf-8", errors="replace")


def _pack_embedding(values: list[float]) -> bytes:
    """Store a vector as raw float64 bytes. Pickle is not used: a cache file
    must not be able to execute code when it is loaded."""
    buf = array.array("d", (float(v) for v in values))
    return buf.tobytes()


def _unpack_embedding(blob: bytes | memoryview) -> list[float]:
    raw = bytes(blob)
    if not raw or len(raw) % 8:
        raise ValueError("embedding blob is not packed float64")
    buf = array.array("d")
    buf.frombytes(raw)
    return buf.tolist()


def _is_hidden_kb_path(kb_path: pathlib.Path, path: pathlib.Path) -> bool:
    """True when *path* is a dotfile or lives under a dot-directory of the KB."""
    try:
        parts = path.resolve().relative_to(kb_path.resolve()).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") for part in parts)


def _iter_kb_files(kb_path: pathlib.Path) -> list[pathlib.Path]:
    """All indexable files under *kb_path*, sorted for determinism."""
    files: list[pathlib.Path] = []
    for p in sorted(kb_path.rglob("*")):
        if not p.is_file() or "?" in p.name:
            continue
        if _is_hidden_kb_path(kb_path, p):
            continue
        ext = p.suffix.lower()
        if ext not in KB_FILE_EXTENSIONS:
            continue
        try:
            if p.stat().st_size > _MAX_BYTES_PER_FILE:
                continue
        except OSError:
            continue
        files.append(p)
    return files


def _rel_key(kb_path: pathlib.Path, path: pathlib.Path) -> str:
    """KB-relative POSIX path used as the stable identity for a file in the
    vector index.

    The index historically keyed documents by their *basename* alone.  That is
    ambiguous once a KB contains subfolders — the per-session notes folders live
    under ``session_notes/<date>_<idx>/`` and each holds a ``notes.md``, a
    ``transcript_01.md``, etc.  Keying by basename made every session's
    ``notes.md`` (and same-named attachments/transcripts) collapse onto the
    same cache key and silently overwrite one another.

    Keying by the path *relative to the KB root* makes each file unique while
    staying identical to the basename for flat (root-level) KBs, so existing
    caches and behaviour are unchanged for the common case.
    """
    try:
        return path.resolve().relative_to(kb_path.resolve()).as_posix()
    except ValueError:
        # Path is not under the KB root (e.g. a bare filename handed to
        # update_single_document / remove_document) — fall back to the basename
        # so those calls behave exactly as before.
        return pathlib.Path(path).name


# ──────────────────────────── Index store ────────────────────────────────

class KBIndexStore:
    """Persistent vector index with content-hash based SQLite caching.

    Load order on each startup:
      1. If a valid cache exists, reuse rows whose file + content hash
         still match disk (HIT — zero embedding API calls).
      2. Re-embed only files that are new, changed, or whose cached rows
         are stale — one batched API call for all of them.
      3. Persist the merged result for the next run.
    """

    def __init__(
        self,
        kb_path: str | pathlib.Path,
        *,
        persist_dir: str | pathlib.Path | None = None,
        model_name: str = "",  # defaults to EMBEDDING_MODEL env var when empty
    ) -> None:
        self.kb_path = pathlib.Path(kb_path)
        # Default cache lives next to the KB itself (absolute) so the bot
        # works no matter which directory it is launched from. The old
        # CWD-relative "kb/.index_cache" default broke when the bot was
        # started from anywhere other than the repo root.
        if persist_dir is None:
            self.persist_dir = self.kb_path / ".vector_index_cache"
        else:
            self.persist_dir = pathlib.Path(persist_dir)
            if not self.persist_dir.is_absolute():
                self.persist_dir = (pathlib.Path(__file__).resolve().parent.parent / persist_dir)
        # Fall back to the configured embedding model (env var) when none given.
        if not model_name:
            from config.settings import EMBEDDING_MODEL
            model_name = EMBEDDING_MODEL
        self.model_name = model_name

        self._db_path = self.persist_dir / "vector_index.db"
        self._index: KBVectorIndex | None = None
        self._embedder = Embedder(model_name=model_name)
        # P0 #3: serializes every index mutation (read → embed → merge →
        # persist) so concurrent updates can't lose each other's chunks.
        # Created lazily; asyncio primitives resolve the running loop on
        # use, so the one instance is safe to reuse across test loops.
        # (Do NOT recreate it per call — two distinct Lock objects never
        # exclude each other.  Python ≥3.10 locks carry no loop binding.)
        self._mut_lock: asyncio.Lock | None = None

    def _mutation_lock(self) -> asyncio.Lock:
        """Return this store's mutation lock (created once, lazily)."""
        if self._mut_lock is None:
            self._mut_lock = asyncio.Lock()
        return self._mut_lock

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def load(self, force_rebuild: bool = False) -> KBVectorIndex:
        """Load the vector index, using the disk cache when possible."""
        if self._index is not None:
            return self._index

        # P0 #3: building/loading is a full read → embed → persist mutation;
        # serialize it against concurrent updates/syncs on this store.
        async with self._mutation_lock():
            return await self._load_inner(force_rebuild)

    async def _load_inner(self, force_rebuild: bool = False) -> KBVectorIndex:
        """Lock-free load core.

        ``load()`` wraps this in the mutation lock; ``sync_changes`` /
        ``update_single_document`` call it *while already holding* the lock
        (asyncio.Lock is not re-entrant — calling ``load()`` from there
        would deadlock).
        """
        if self._index is not None:
            return self._index

        if not self.kb_path.exists():
            logger.warning("KB path '%s' does not exist — empty index", self.kb_path)
            self._index = KBVectorIndex()
            await self._save_empty_cache()
            return self._index

        if force_rebuild or not self._db_path.exists():
            if force_rebuild:
                # Drop the persisted cache so nothing stale is reused.
                if self._db_path.exists():
                    self._db_path.unlink()
            self._index = await self._build_fresh()
        elif not await asyncio.to_thread(self._cache_identity_ok):
            logger.warning(
                "Vector cache was built for a different schema or embedding model "
                "— rebuilding (%s)", self._db_path,
            )
            self._db_path.unlink(missing_ok=True)
            self._index = await self._build_fresh()
        else:
            self._index = await self._load_incremental()

        return self._index

    async def rebuild(self) -> KBVectorIndex:
        """Force a full rebuild (drops the existing cache)."""
        self._index = None
        return await self.load(force_rebuild=True)

    async def shutdown(self) -> None:
        """Persist the index before shutdown (belt-and-suspenders)."""
        async with self._mutation_lock():
            if self._index is not None and not self._index.is_empty():
                await self._save_to_disk()

    # ── Public updates ──────────────────────────────────────────────────

    async def update_single_document(self, file_path: str | pathlib.Path) -> bool:
        """Re-index a single document (add or replace). Returns True on success."""
        path = pathlib.Path(file_path)

        # P0 #3: hold the mutation lock across the ENTIRE read → embed →
        # merge → persist sequence.  The embed call below awaits for seconds,
        # during which a second concurrent update used to read the same stale
        # doc list and both merges overwrote each other (lost chunks).
        async with self._mutation_lock():
            if not path.is_file():
                logger.warning("File '%s' does not exist; skipping update", file_path)
                return False

            # Ensure we have a working index to merge into.  (Held-lock
            # variant — ``load()`` would deadlock on the non-reentrant lock.)
            if self._index is None or self._index.is_empty():
                self._index = await self._load_inner()

            try:
                entries, embeddings, fh_map = await self._embed_one_file(path)
            except Exception as exc:
                logger.warning("Failed to embed '%s': %s", path.name, exc)
                return False

            old_docs = list(self._index._docs) if self._index is not None else []
            key = _rel_key(self.kb_path, path)
            merged = self._merge_replace(old_docs, key, entries, embeddings,
                                         file_hash=fh_map.get(key))  # P2 #19
            self._index = KBVectorIndex.from_entries(
                self._entries_from_docs(merged),
                [d.embedding for d in merged],
            )

            await self._save_to_disk(changed={key})  # P2 #20: incremental
            logger.info("Updated index with '%s' (%d chunk(s))", key, len(entries))
            return True

    async def remove_document(self, file_path: str | pathlib.Path) -> bool:
        """Remove a document from the index. Returns True if something was removed."""
        async with self._mutation_lock():
            if self._index is None or self._index.is_empty():
                return False

            target = _rel_key(self.kb_path, pathlib.Path(file_path)).lower()
            old_count = self._index.count()
            self._index._docs = [  # type: ignore[union-attr]
                doc for doc in self._index._docs  # type: ignore[union-attr]
                if doc.source().lower() != target
            ]

            removed = old_count - self._index.count()
            if removed > 0:
                # P2 #20: incremental persist — the file's rows are gone from the
                # index, so the upsert path deletes them from the cache.
                await self._save_to_disk(changed=set())
                logger.info("Removed %d chunk(s) for '%s'", removed, target)
                return True

            logger.warning("No matching chunks found to remove for '%s'", file_path)
            return False

    async def sync_changes(self) -> tuple[KBVectorIndex, dict]:
        """Sync the index with files added, renamed, changed, or deleted on disk.

        Cheap counterpart to :meth:`rebuild`: it reuses the disk cache and only
        re-embeds files whose chunks are missing or no longer match disk.  This
        covers documents that were never added through the Discord upload
        command (dropped into the KB folder, renamed, edited, or removed
        externally).

        Returns ``(index, report)`` where *report* contains
        ``added / changed / renamed / removed / failed`` lists (renamed is a
        list of ``(old_name, new_name)`` pairs) plus ``changed_count`` (files
        that needed re-embedding) and ``ok`` (False when the embedding backend
        failed for at least one file).
        """
        # P0 #3: serialize the whole scan → embed → merge → persist run so a
        # concurrent upload/sync cannot interleave and lose chunks.
        async with self._mutation_lock():
            if self._index is None or self._index.is_empty():
                await self._load_inner()  # lock-free core — we already hold it

            files = await self._iter_files()
            rows = await self._read_cache_rows_async() if self._db_path.exists() else {}

            added: list[pathlib.Path] = []
            changed: list[pathlib.Path] = []
            removed: list[str] = []

            for path in files:
                cached = rows.get(_rel_key(self.kb_path, path))
                if cached is None:
                    added.append(path)
                elif not await self._chunks_valid(cached, path):
                    changed.append(path)

            disk_names = {_rel_key(self.kb_path, p).lower() for p in files}
            for name in rows:
                if name.lower() not in disk_names:
                    removed.append(name)

            # A file whose old name disappeared and whose new name is unindexed —
            # with identical content — is a rename, not remove+add.  Pair them by
            # content signature so the report stays honest and the user sees the
            # rename instead of two noisy entries.
            # Signatures are strip-normalized because cached chunk content is
            # stored stripped (chunker output) while raw disk text often keeps a
            # trailing newline — an un-stripped compare would miss plain renames.
            old_sig: dict[str, str] = {}
            for name in removed:
                chunks = rows.get(name) or []
                if chunks:
                    old_sig[name] = "\0".join(c["content"] for c in chunks)

            async def _new_sig(path: pathlib.Path) -> str:
                try:
                    return (await self._read_file_text(path)).strip()
                except OSError:
                    return ""

            used_old: set[str] = set()
            renamed: list[tuple[str, str]] = []
            still_added: list[pathlib.Path] = []
            for path in added:
                sig = await _new_sig(path)
                match = next(
                    (old for old in removed if old not in used_old and old_sig.get(old) == sig),
                    None,
                )
                if match:
                    used_old.add(match)
                    renamed.append((match, _rel_key(self.kb_path, path)))
                    changed.append(path)
                else:
                    still_added.append(path)

            # De-dup: a renamed file lands in both `still_added` (via `changed`)
            # and `changed` — embedding it twice would duplicate its chunks.
            to_embed: list[pathlib.Path] = []
            for p in still_added + changed:
                if p not in to_embed:
                    to_embed.append(p)

            failures: list[str] = []
            if to_embed:
                try:
                    new_entries, new_embeddings, new_fh = await self._embed_files(to_embed)
                except Exception as exc:
                    logger.warning("Sync-embed failed for %d file(s): %s", len(to_embed), exc)
                    failures = [_rel_key(self.kb_path, p) for p in to_embed]
                    new_entries, new_embeddings, new_fh = [], [], {}

            if to_embed or removed:
                old_docs = list(self._index._docs) if self._index is not None else []
                merged = old_docs
                for path in to_embed:
                    # Each replace targets exactly one source_file (relative path),
                    # so only hand it this file's own rows — keeps other files' rows
                    # intact and drops the old rows of a renamed file.
                    key = _rel_key(self.kb_path, path)
                    file_entries = [e for e in new_entries if e[2].lower() == key.lower()]
                    file_embs = [new_embeddings[i] for i, e in enumerate(new_entries) if e[2].lower() == key.lower()]
                    if file_entries:
                        merged = self._merge_replace(merged, key, file_entries, file_embs,
                                                     file_hash=new_fh.get(key))  # P2 #19
                # Drop rows for files that are gone from disk (renamed-away names
                # too, so a rename ends up as replace instead of duplicate).
                stale = {n.lower() for n in removed}
                merged = [d for d in merged if d.source().lower() not in stale]
                self._index = KBVectorIndex.from_entries(
                    self._entries_from_docs(merged),
                    [d.embedding for d in merged],
                    [d.file_hash for d in merged],  # P2 #19
                )
                # Only persist a non-empty index — never wipe a good on-disk cache
                # with an empty one when the embedding backend is down.
                if not self._index.is_empty():
                    # P2 #20: incremental persist — only the embedded files changed;
                    # removed files' rows are pruned by the upsert path.
                    await self._save_to_disk(changed={_rel_key(self.kb_path, p) for p in to_embed})

            report = {
                "added": [_rel_key(self.kb_path, p) for p in still_added],
                "changed": [_rel_key(self.kb_path, p) for p in changed
                            if _rel_key(self.kb_path, p) not in {r[1] for r in renamed}],
                "renamed": renamed,
                "removed": [n for n in removed if n not in used_old],
                "failed": failures,
                "changed_count": len(to_embed),
                "ok": not failures,
            }
            logger.info(
                "KB sync: %d added, %d changed, %d renamed, %d removed, %d failed (%d chunk(s) total)",
                len(report["added"]), len(report["changed"]), len(renamed),
                len(report["removed"]), len(failures),
                self._index.count() if self._index else 0,
            )
            return self._index, report

    # ── Querying ────────────────────────────────────────────────────────

    def get_index(self) -> KBVectorIndex | None:
        """Return the in-memory vector index (or None if not loaded)."""
        return self._index

    @property
    def db_path(self) -> pathlib.Path:
        return self._db_path

    # ── Build paths ─────────────────────────────────────────────────────

    async def _build_fresh(self) -> KBVectorIndex:
        """Build the full index from disk, reusing cached embeddings where possible."""
        files = await self._iter_files()
        if not files:
            logger.warning("No indexable files found in '%s'", self.kb_path)
            idx = KBVectorIndex()
            await self._save_empty_cache()
            return idx

        cached_rows = await self._read_cache_rows_async() if self._db_path.exists() else {}
        to_embed_files: list[pathlib.Path] = []

        entries: list[tuple[str, str, str]] = []   # (display_name, content, source_file)
        embeddings: list[list[float]] = []
        file_hash_by_key: dict[str, str] = {}      # P2 #19

        for path in files:
            key = _rel_key(self.kb_path, path)
            cached = cached_rows.get(key)
            if cached is not None and await self._chunks_valid(cached, path):
                for c in cached:
                    entries.append((c["doc_name"], c["content"], key))
                    embeddings.append(c["embedding"])
                fh = cached[0].get("file_hash")    # P2 #19: carry forward (None on legacy)
                if fh:
                    file_hash_by_key[key] = fh
                continue

            # Not cached, changed, or stale — re-embed the whole file.
            to_embed_files.append(path)

        logger.info(
            "Index build: %d file(s) — %d chunk(s) from cache, %d file(s) need embedding",
            len(files), len(entries), len(to_embed_files),
        )

        if to_embed_files:
            try:
                new_entries, new_embeddings, new_fh = await self._embed_files(to_embed_files)
            except Exception as exc:
                logger.warning(
                    "Fresh index build failed: %s — embedding backend may be down", exc
                )
                new_entries, new_embeddings, new_fh = [], [], {}
            entries.extend(new_entries)
            embeddings.extend(new_embeddings)
            file_hash_by_key.update(new_fh)          # P2 #19

        if not embeddings:
            logger.error("Index build produced no chunks for '%s' — check embedding backend connectivity", self.kb_path)
            await self._save_empty_cache()
            return KBVectorIndex()

        aligned_hashes = [file_hash_by_key.get(e[2]) for e in entries]  # P2 #19
        idx = KBVectorIndex.from_entries(entries, embeddings, aligned_hashes)
        await self._save_to_disk_from(idx)
        logger.info("Index ready: %d chunk(s) from %d file(s), persisted to %s",
                    idx.count(), len(files), self._db_path)
        return idx

    async def _load_incremental(self) -> KBVectorIndex:
        """Load from cache, re-embedding only changed/missing files.

        Never raises: if the embedding backend is down, the previously
        cached chunks are still returned (minus the stale ones) so RAG
        degrades to "cached subset + keyword fallback" instead of failing.
        """
        files = await self._iter_files()
        cached_rows = await self._read_cache_rows_async()

        entries: list[tuple[str, str, str]] = []
        embeddings: list[list[float]] = []
        to_embed_files: list[pathlib.Path] = []
        file_hash_by_key: dict[str, str] = {}      # P2 #19

        for path in files:
            key = _rel_key(self.kb_path, path)
            cached = cached_rows.get(key)
            if cached is None or not await self._chunks_valid(cached, path):
                # New file, or any chunk changed → re-embed the whole file.
                to_embed_files.append(path)
                continue
            for c in cached:
                entries.append((c["doc_name"], c["content"], key))
                embeddings.append(c["embedding"])
            fh = cached[0].get("file_hash")         # P2 #19: carry forward (None on legacy)
            if fh:
                file_hash_by_key[key] = fh

        reused = len(entries)
        if to_embed_files:
            try:
                new_entries, new_embeddings, new_fh = await self._embed_files(to_embed_files)
            except Exception as exc:
                logger.warning(
                    "Incremental update failed (%d file(s) not re-embedded: %s): %s — "
                    "serving %d cached chunk(s) only",
                    len(to_embed_files), [_rel_key(self.kb_path, p) for p in to_embed_files], exc, reused,
                )
                new_entries, new_embeddings, new_fh = [], [], {}
            entries.extend(new_entries)
            embeddings.extend(new_embeddings)
            file_hash_by_key.update(new_fh)         # P2 #19
            logger.info(
                "Incremental load: %d cached + %d newly embedded chunk(s) (%d file(s) refreshed)",
                reused, len(new_entries), len(to_embed_files),
            )
        else:
            logger.info("Cache HIT: %d chunk(s) loaded from disk, 0 API calls", reused)

        aligned_hashes = [file_hash_by_key.get(e[2]) for e in entries] if embeddings else None  # P2 #19
        idx = KBVectorIndex.from_entries(entries, embeddings, aligned_hashes) if embeddings else KBVectorIndex()
        # P2 #20: incremental persist — only the refreshed files changed; rows
        # for files deleted since the cache was written are pruned by the upsert path.
        if idx is not None:
            changed = {_rel_key(self.kb_path, p) for p in to_embed_files}
            await self._save_to_disk_from(idx, changed=changed)
        return idx

    # ── Embedding helpers ───────────────────────────────────────────────

    async def _embed_files(self, paths: list[pathlib.Path]) -> tuple[list[tuple[str, str, str]], list[list[float]], dict[str, str]]:
        """Chunk + embed a set of files. Returns (entries, embeddings, file_hashes).

        *file_hashes* maps each file's KB-relative key to its SHA-256 (P2 #19),
        so the cache can skip re-chunking unchanged files on the next load.
        """
        from kb.chunker import Chunker

        flat: list[tuple[str, str, str]] = []  # (display_name, content, source_file)
        file_hashes: dict[str, str] = {}
        for path in paths:
            key = _rel_key(self.kb_path, path)
            # P2 #19: record the source file's hash (off-loop) while we're here,
            # so a later load can skip re-chunking if the file is unchanged.
            fh = await self._file_hash(path)
            if fh is not None:
                file_hashes[key] = fh
            try:
                chunks = await Chunker.split_file(path)
            except Exception as exc:
                logger.warning("Chunking failed for '%s': %s", path.name, exc)
                chunks = []
            if not chunks:
                # Whole-file fallback (small/unsupported files).
                try:
                    text = (await self._read_file_text(path)).strip()
                except OSError:
                    continue
                if text:
                    flat.append((path.name, text, key))
                continue
            for c in chunks:
                flat.append((f"{c.display_name} [{c.section_path}]", c.content, key))

        if not flat:
            return [], [], file_hashes

        embeddings = await self._embedder.encode([c for _, c, _ in flat])
        return flat, embeddings, file_hashes

    async def _embed_one_file(self, path: pathlib.Path) -> tuple[list[tuple[str, str, str]], list[list[float]], dict[str, str]]:
        return await self._embed_files([path])

    async def _file_hash(self, path: pathlib.Path) -> str | None:
        """P2 #19: SHA-256 of *path*'s bytes, off the event loop (P0 #4)."""
        try:
            data = await asyncio.to_thread(path.read_bytes)
        except OSError:
            return None
        return _file_hash_bytes(data)

    async def _chunks_valid(self, cached: list[dict], file: pathlib.Path) -> bool:
        """True when cached chunks for *file* match the on-disk chunking exactly.

        P2 #19 fast path: if the cached rows carry a *source file hash* and it
        matches the file on disk, the chunks are guaranteed valid — the chunker
        is a deterministic function of the file's bytes, so unchanged bytes mean
        unchanged chunks.  This skips the expensive re-chunk entirely (one file
        read for the hash instead of read + regex-split + ChunkInfo build).

        When no file hash is stored (legacy v3 cache) we fall back to the
        original re-chunk-and-compare path.
        """
        if not cached:
            return False
        if any(_content_hash(c["content"]) != c["content_hash"] for c in cached):
            return False
        # P2 #19: source-file hash decides validity when available.
        stored_fh = cached[0].get("file_hash")
        if stored_fh:
            disk_fh = await self._file_hash(file)
            if disk_fh is None:
                return False  # file vanished / unreadable → stale
            # Unchanged bytes → chunks still valid; changed bytes → stale.
            # (Re-chunking on mismatch is unnecessary: different bytes always
            # produce different chunks, so the hash is a sufficient test.)
            return disk_fh == stored_fh
        # Legacy (no stored file hash): re-chunk the file from disk and compare.
        try:
            chunks = await self._rechunk(file)
        except Exception:
            chunks = []

        expected: list[str] = []
        if chunks:
            expected = [c.content for c in chunks]
        else:
            # Fallback mirrors _embed_files: whole-file raw text.
            try:
                text = (await self._read_file_text(file)).strip()
            except OSError:
                return False
            if text:
                expected = [text]

        if len(expected) != len(cached):
            return False
        return all(ec == c["content"] for ec, c in zip(expected, cached))

    # ── In-memory merge helpers ─────────────────────────────────────────

    # ── P0 #4: blocking-IO helpers (offloaded to a worker thread) ──

    async def _iter_files(self) -> list[pathlib.Path]:
        """Directory walk off the event loop (P0 #4)."""
        return await asyncio.to_thread(_iter_kb_files, self.kb_path)

    async def _read_cache_rows_async(self) -> dict[str, list[dict]]:
        """SQLite cache read off the event loop (P0 #4)."""
        return await asyncio.to_thread(self._read_cache_rows)

    async def _read_file_text(self, path: pathlib.Path) -> str:
        """Whole-file read + decode off the event loop (P0 #4)."""
        return await asyncio.to_thread(
            _read_text_sync, path
        )

    async def _rechunk(self, path: pathlib.Path) -> list:
        """Re-chunk from disk off the event loop (P0 #4).

        ``Chunker.split_file`` is a pure CPU/IO pass (read file, regex
        split) — running it on the loop froze the whole bot for large docs.
        """
        from kb.chunker import Chunker
        return await asyncio.to_thread(Chunker.split_file_sync, path)

    # ── In-memory merge helpers ─────────────────────────────────────────

    @staticmethod
    def _entries_from_docs(docs: list[_DocEntry]) -> list[tuple[str, str, str]]:
        return [(d.display_name, d.content, d.source()) for d in docs]

    @staticmethod
    def _merge_replace(
        docs: list[_DocEntry],
        source_file: str,
        new_entries: list[tuple[str, str, str]],
        new_embeddings: list[list[float]],
        file_hash: str | None = None,  # P2 #19
    ) -> list[_DocEntry]:
        """Replace all chunks for *source_file* with the newly embedded ones."""
        kept = [d for d in docs if d.source().lower() != source_file.lower()]
        for (name, content, src), emb in zip(new_entries, new_embeddings):
            kept.append(_DocEntry(display_name=name, content=content, embedding=emb,
                                  source_file=src, file_hash=file_hash))
        return kept

    # ── SQLite persistence ──────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        os.makedirs(self.persist_dir, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(_SCHEMA_CREATE_DOC_INDEX)
        conn.execute(_SCHEMA_CREATE_METADATA)
        # P2 #19: migrate v3 caches that lack the file_hash column
        # (CREATE TABLE IF NOT EXISTS won't add a new column to an existing table).
        cols = {r[1] for r in conn.execute("PRAGMA table_info(document_index)")}
        if "file_hash" not in cols:
            conn.execute("ALTER TABLE document_index ADD COLUMN file_hash TEXT")
        return conn

    def _read_cache_rows(self) -> dict[str, list[dict]]:
        """Return cached chunks keyed by source file (ALL rows per file, in id order).

        Returns an empty dict for legacy/corrupt caches (no content-hash schema),
        which callers treat as "nothing cached — re-embed everything".
        Synchronous — call via ``_read_cache_rows_async`` (P0 #4) from async code.
        """
        if not self._db_path.exists():
            return {}
        try:
            conn = sqlite3.connect(str(self._db_path))
            # Tolerate pre-v3 schemas (no content_hash / embedding columns).
            cols = {r[1] for r in conn.execute("PRAGMA table_info(document_index)")}
            if "content_hash" not in cols or "embedding" not in cols:
                conn.close()
                return {}
            # Uniform 6-column select (aliases) so unpacking is stable across
            # v2 (no source_file) / v3 (no file_hash) / v4 caches.
            src_col = "source_file" if "source_file" in cols else "doc_name"
            fh_col = "file_hash" if "file_hash" in cols else "NULL"  # P2 #19
            select = (f"SELECT {src_col} AS s, doc_name AS d, content AS c, "
                      f"content_hash AS ch, {fh_col} AS fh, embedding AS e "
                      f"FROM document_index ORDER BY id")
            rows = conn.execute(select).fetchall()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Could not read index cache: %s", exc)
            return {}

        result: dict[str, list[dict]] = {}
        # P2 #22: legacy rows are keyed by *basename* (no "/").  Re-keying
        # those used to run a full ``_iter_kb_files`` rglob *per row*.  Instead
        # resolve the KB file list ONCE (lazily, only when a legacy row is seen)
        # and match every row against it in a single pass.  A fully modern
        # cache (all keys already contain "/") never walks the directory.
        basename_map: dict[str, list[str]] | None = None

        def _resolve_legacy(source: str) -> str:
            if "/" in source:
                return source
            nonlocal basename_map
            if basename_map is None:
                basename_map = {}
                for p in _iter_kb_files(self.kb_path):  # exactly one walk
                    basename_map.setdefault(p.name, []).append(
                        _rel_key(self.kb_path, p)
                    )
            matches = basename_map.get(source)
            if matches and len(matches) == 1:
                return matches[0]
            # Ambiguous (or absent) basename — leave as-is; the caller will
            # re-embed that file rather than guess wrong.
            return source

        for source, display, content, content_hash, file_hash, emb_blob in rows:
            try:
                emb = _unpack_embedding(emb_blob)
            except Exception:
                continue
            if not emb:
                continue
            # v2 caches: source == display (both are the display name).
            # Derive the real source filename from it.
            if source == display and " [" in source:
                m = re.match(r"^\S+\.(?:txt|md|csv|html|xml|rtf)\b", source)
                source = m.group(0) if m else source.split(" [")[0]
            # v3 caches that predate the per-session folder layout keyed rows by
            # *basename* — re-key to the file's KB-relative path (single pass,
            # one directory walk total; ambiguous rows fall through unchanged).
            source = _resolve_legacy(source)
            result.setdefault(source, []).append({
                "doc_name": display,
                "content": content,
                "content_hash": content_hash,
                "embedding": emb,
                "file_hash": file_hash,  # P2 #19 (None on legacy v3 caches)
            })
        return result

    def _embedding_dim(self, idx: KBVectorIndex | None) -> str:
        if idx is None:
            return ""
        for doc in idx._docs:
            if doc.embedding:
                return str(len(doc.embedding))
        return ""

    def _write_metadata(self, conn: sqlite3.Connection, idx: KBVectorIndex | None, now: float) -> None:
        rows = {
            "kb_path": str(self.kb_path),
            "schema_version": _SCHEMA_VERSION,
            "updated_at": str(now),
            "embedding_model": self.model_name,
            "embedding_dim": self._embedding_dim(idx),
        }
        conn.executemany(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            list(rows.items()),
        )

    def _cache_identity_ok(self) -> bool:
        """False when the on-disk cache was built for another model or schema.

        Old vectors must not be ranked against a new model's query embedding:
        the same width returns nonsense, and a different width matches nothing.
        """
        if not self._db_path.exists():
            return False
        try:
            conn = sqlite3.connect(str(self._db_path))
            meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
            conn.close()
        except sqlite3.Error:
            return False
        if meta.get("schema_version") != _SCHEMA_VERSION:
            return False
        if meta.get("embedding_model") != self.model_name:
            return False
        return True

    def _persist_index_to_db(self, idx: KBVectorIndex) -> None:
        """Synchronous SQLite persistence — atomic temp-file swap.

        Runs in a worker thread via ``asyncio.to_thread`` (P0 #4) so a large
        index never blocks the event loop.  The connection is created and
        used entirely within that one thread.
        """
        tmp_path = self._db_path.with_suffix(".tmp")
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            if tmp_path.exists():
                tmp_path.unlink()

            conn = sqlite3.connect(str(tmp_path))
            conn.execute(_SCHEMA_CREATE_DOC_INDEX)
            conn.execute(_SCHEMA_CREATE_METADATA)

            now = time.time()
            rows = []
            for doc in idx._docs:
                if doc.embedding is None:
                    continue
                emb_bytes = _pack_embedding(doc.embedding)
                rows.append(
                    (doc.source(), doc.display_name, doc.content,
                     _content_hash(doc.content), doc.file_hash,  # P2 #19
                     emb_bytes, now)
                )

            conn.executemany(
                "INSERT INTO document_index "
                "(source_file, doc_name, content, content_hash, file_hash, embedding, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._write_metadata(conn, idx, now)
            conn.commit()
            conn.close()

            os.replace(tmp_path, self._db_path)  # atomic on POSIX
            logger.debug("Saved %d chunk(s) to %s", len(rows), self._db_path)
        except Exception as exc:
            logger.error("Failed to save index to disk: %s", exc)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    def _persist_upsert(self, idx: KBVectorIndex, changed: set[str]) -> None:
        """P2 #20: incrementally persist — replace changed files, delete removed
        ones — WITHOUT rewriting the whole cache.

        Runs in a worker thread (P0 #4) directly on the LIVE db (no tmp-file
        swap), so only the affected rows are written and every other row keeps
        its SQLite page/offset.  ``changed`` holds the source_file keys whose
        rows were (re)embedded; rows for files that are in the cache but absent
        from *idx* are deleted (covers remove_document / removed-in-sync).

        The full rewrite (:meth:`_persist_index_to_db`) is kept as the periodic
        compaction path — it's used for fresh builds, which is the one case that
        legitimately touches every row.
        """
        conn = self._conn()  # ensures schema + file_hash migration on the live DB
        try:
            # A pre-v3 (v2) cache has no source_file column — the incremental
            # upsert can't target rows by key on it.  Fall back to the atomic
            # full rewrite, which rebuilds the table with the current schema.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(document_index)")}
            if "source_file" not in cols:
                conn.close()
                conn = None
                self._persist_index_to_db(idx)
                return
            now = time.time()
            # 1. Replace rows for the changed files (delete old, insert new).
            for key in changed:
                conn.execute("DELETE FROM document_index WHERE source_file = ?", (key,))
                file_rows = []
                for doc in idx._docs:
                    if doc.source().lower() != key.lower() or doc.embedding is None:
                        continue
                    file_rows.append(
                        (doc.source(), doc.display_name, doc.content,
                         _content_hash(doc.content), doc.file_hash,  # P2 #19
                         _pack_embedding(doc.embedding), now)
                    )
                if file_rows:
                    conn.executemany(
                        "INSERT INTO document_index "
                        "(source_file, doc_name, content, content_hash, file_hash, "
                        "embedding, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        file_rows,
                    )
            # 2. Delete rows for files no longer present in the index.
            kept = {doc.source().lower() for doc in idx._docs}
            for (src,) in conn.execute(
                "SELECT source_file FROM document_index"
            ).fetchall():
                if src.lower() not in kept:
                    conn.execute("DELETE FROM document_index WHERE source_file = ?", (src,))
            # 3. Refresh metadata (schema/model may be unset on a legacy cache).
            self._write_metadata(conn, idx, now)
            conn.commit()
            logger.debug("Incremental save: %d changed file(s) upserted, pruned removed → %s",
                         len(changed), self._db_path)
        except Exception as exc:
            logger.error("Failed to incrementally save index: %s", exc)
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

    async def _save_to_disk(self, *, changed: set[str] | None = None) -> None:
        if self._index is not None:
            await self._save_to_disk_from(self._index, changed=changed)

    async def _save_to_disk_from(
        self, idx: KBVectorIndex, *, changed: set[str] | None = None
    ) -> None:
        """Persist the index to SQLite, off the event loop (P0 #4).

        P2 #20: when *changed* is provided (a set of source_file keys, possibly
        empty), persist incrementally — upsert those files + delete removed ones
        instead of rewriting the whole cache.  When it is ``None`` (a full build
        or shutdown flush), do the atomic full rewrite (which also serves as the
        periodic compaction pass).
        """
        if changed is not None:
            await asyncio.to_thread(self._persist_upsert, idx, changed)
        else:
            await asyncio.to_thread(self._persist_index_to_db, idx)

    async def _save_empty_cache(self) -> None:
        """Persist an empty (but schema-valid) cache so we don't retry forever."""
        await self._save_to_disk_from(KBVectorIndex())
