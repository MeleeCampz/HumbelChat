"""Persistent vector index for KB document retrieval.

Caches the in-memory ``KBVectorIndex`` to disk (SQLite) so bot restarts
don't require re-indexing the entire knowledge base — saving seconds of
startup time and avoiding repeated API calls to the embedding backend.

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
    await store.rebuild()                              # full rebuild (force)

"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
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
    embedding   BLOB,                   -- pickle'd list[float]
    updated_at  REAL DEFAULT (strftime('%s','now'))
);
"""

_SCHEMA_CREATE_METADATA = """\
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_SCHEMA_VERSION = "3"  # content-hash based incremental cache


# ──────────────────────────── Helpers ────────────────────────────────────

def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _iter_kb_files(kb_path: pathlib.Path) -> list[pathlib.Path]:
    """All indexable files under *kb_path*, sorted for determinism."""
    files: list[pathlib.Path] = []
    for p in sorted(kb_path.rglob("*")):
        if not p.is_file() or "?" in p.name:
            continue
        ext = p.suffix.lower()
        if ext not in KB_FILE_EXTENSIONS:
            continue
        files.append(p)
    return files


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
        model_name: str = "nomic-embed-text:latest",
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
        self.model_name = model_name

        self._db_path = self.persist_dir / "vector_index.db"
        self._index: KBVectorIndex | None = None
        self._embedder = Embedder(model_name=model_name)

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def load(self, force_rebuild: bool = False) -> KBVectorIndex:
        """Load the vector index, using the disk cache when possible."""
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
        else:
            self._index = await self._load_incremental()

        return self._index

    async def rebuild(self) -> KBVectorIndex:
        """Force a full rebuild (drops the existing cache)."""
        self._index = None
        return await self.load(force_rebuild=True)

    async def shutdown(self) -> None:
        """Persist the index before shutdown (belt-and-suspenders)."""
        if self._index is not None and not self._index.is_empty():
            await self._save_to_disk()

    # ── Public updates ──────────────────────────────────────────────────

    async def update_single_document(self, file_path: str | pathlib.Path) -> bool:
        """Re-index a single document (add or replace). Returns True on success."""
        path = pathlib.Path(file_path)
        if not path.is_file():
            logger.warning("File '%s' does not exist; skipping update", file_path)
            return False

        # Ensure we have a working index to merge into.
        if self._index is None or self._index.is_empty():
            self._index = await self._load_incremental()

        try:
            entries, embeddings = await self._embed_one_file(path)
        except Exception as exc:
            logger.warning("Failed to embed '%s': %s", path.name, exc)
            return False

        old_docs = list(self._index._docs) if self._index is not None else []
        merged = self._merge_replace(old_docs, path.name, entries, embeddings)
        self._index = KBVectorIndex.from_entries(
            self._entries_from_docs(merged),
            [d.embedding for d in merged],
        )

        await self._save_to_disk()
        logger.info("Updated index with '%s' (%d chunk(s))", path.name, len(entries))
        return True

    async def remove_document(self, file_path: str | pathlib.Path) -> bool:
        """Remove a document from the index. Returns True if something was removed."""
        if self._index is None or self._index.is_empty():
            return False

        target = pathlib.Path(file_path).name.lower()
        old_count = self._index.count()
        self._index._docs = [  # type: ignore[union-attr]
            doc for doc in self._index._docs  # type: ignore[union-attr]
            if doc.source().lower() != target
        ]

        removed = old_count - self._index.count()
        if removed > 0:
            await self._save_to_disk()
            logger.info("Removed %d chunk(s) for '%s'", removed, pathlib.Path(file_path).name)
            return True

        logger.warning("No matching chunks found to remove for '%s'", file_path)
        return False

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
        files = _iter_kb_files(self.kb_path)
        if not files:
            logger.warning("No indexable files found in '%s'", self.kb_path)
            idx = KBVectorIndex()
            await self._save_empty_cache()
            return idx

        cached_rows = self._read_cache_rows() if self._db_path.exists() else {}
        to_embed_files: list[pathlib.Path] = []

        entries: list[tuple[str, str, str]] = []   # (display_name, content, source_file)
        embeddings: list[list[float]] = []

        for path in files:
            cached = cached_rows.get(path.name)
            if cached is not None and await self._chunks_valid(cached, path):
                for c in cached:
                    entries.append((c["doc_name"], c["content"], path.name))
                    embeddings.append(c["embedding"])
                continue

            # Not cached, changed, or stale — re-embed the whole file.
            to_embed_files.append(path)

        logger.info(
            "Index build: %d file(s) — %d chunk(s) from cache, %d file(s) need embedding",
            len(files), len(entries), len(to_embed_files),
        )

        if to_embed_files:
            try:
                new_entries, new_embeddings = await self._embed_files(to_embed_files)
            except Exception as exc:
                logger.warning(
                    "Fresh index build failed: %s — embedding backend may be down", exc
                )
                new_entries, new_embeddings = [], []
            entries.extend(new_entries)
            embeddings.extend(new_embeddings)

        if not embeddings:
            logger.error("Index build produced no chunks for '%s' — check embedding backend connectivity", self.kb_path)
            await self._save_empty_cache()
            return KBVectorIndex()

        idx = KBVectorIndex.from_entries(entries, embeddings)
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
        files = _iter_kb_files(self.kb_path)
        cached_rows = self._read_cache_rows()

        entries: list[tuple[str, str, str]] = []
        embeddings: list[list[float]] = []
        to_embed_files: list[pathlib.Path] = []

        for path in files:
            cached = cached_rows.get(path.name)
            if cached is None or not await self._chunks_valid(cached, path):
                # New file, or any chunk changed → re-embed the whole file.
                to_embed_files.append(path)
                continue
            for c in cached:
                entries.append((c["doc_name"], c["content"], path.name))
                embeddings.append(c["embedding"])

        reused = len(entries)
        if to_embed_files:
            try:
                new_entries, new_embeddings = await self._embed_files(to_embed_files)
            except Exception as exc:
                logger.warning(
                    "Incremental update failed (%d file(s) not re-embedded: %s): %s — "
                    "serving %d cached chunk(s) only",
                    len(to_embed_files), [p.name for p in to_embed_files], exc, reused,
                )
                new_entries, new_embeddings = [], []
            entries.extend(new_entries)
            embeddings.extend(new_embeddings)
            logger.info(
                "Incremental load: %d cached + %d newly embedded chunk(s) (%d file(s) refreshed)",
                reused, len(new_entries), len(to_embed_files),
            )
        else:
            logger.info("Cache HIT: %d chunk(s) loaded from disk, 0 API calls", reused)

        idx = KBVectorIndex.from_entries(entries, embeddings) if embeddings else KBVectorIndex()
        # Persist the cleaned-up cache (drops rows for deleted files).
        if idx is not None:
            await self._save_to_disk_from(idx)
        return idx

    # ── Embedding helpers ───────────────────────────────────────────────

    async def _embed_files(self, paths: list[pathlib.Path]) -> tuple[list[tuple[str, str, str]], list[list[float]]]:
        """Chunk + embed a set of files. Returns (entries, embeddings) aligned lists."""
        from kb.chunker import Chunker

        flat: list[tuple[str, str, str]] = []  # (display_name, content, source_file)
        for path in paths:
            try:
                chunks = await Chunker.split_file(path)
            except Exception as exc:
                logger.warning("Chunking failed for '%s': %s", path.name, exc)
                chunks = []
            if not chunks:
                # Whole-file fallback (small/unsupported files).
                try:
                    text = path.read_bytes().decode("utf-8", errors="replace").strip()
                except OSError:
                    continue
                if text:
                    flat.append((path.name, text, path.name))
                continue
            for c in chunks:
                flat.append((f"{c.display_name} [{c.section_path}]", c.content, path.name))

        if not flat:
            return [], []

        embeddings = await self._embedder.encode([c for _, c, _ in flat])
        return flat, embeddings

    async def _embed_one_file(self, path: pathlib.Path) -> tuple[list[tuple[str, str, str]], list[list[float]]]:
        return await self._embed_files([path])

    async def _chunks_valid(self, cached: list[dict], file: pathlib.Path) -> bool:
        """True when cached chunks for *file* match the on-disk chunking exactly.

        Compares (a) the per-chunk content hash stored in the cache and
        (b) the sequence of chunk contents against the current chunker output,
        so content changes OR re-chunking both invalidate the cache entry.
        """
        if not cached:
            return False
        if any(_content_hash(c["content"]) != c["content_hash"] for c in cached):
            return False
        # 2. Re-chunk the file from disk and compare.
        from kb.chunker import Chunker
        try:
            chunks = await Chunker.split_file(file)
        except Exception:
            chunks = []

        expected: list[str] = []
        if chunks:
            expected = [c.content for c in chunks]
        else:
            # Fallback mirrors _embed_files: whole-file raw text.
            try:
                text = file.read_bytes().decode("utf-8", errors="replace").strip()
            except OSError:
                return False
            if text:
                expected = [text]

        if len(expected) != len(cached):
            return False
        return all(ec == c["content"] for ec, c in zip(expected, cached))

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
    ) -> list[_DocEntry]:
        """Replace all chunks for *source_file* with the newly embedded ones."""
        kept = [d for d in docs if d.source().lower() != source_file.lower()]
        for (name, content, src), emb in zip(new_entries, new_embeddings):
            kept.append(_DocEntry(display_name=name, content=content, embedding=emb, source_file=src))
        return kept

    # ── SQLite persistence ──────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        os.makedirs(self.persist_dir, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(_SCHEMA_CREATE_DOC_INDEX)
        conn.execute(_SCHEMA_CREATE_METADATA)
        return conn

    def _read_cache_rows(self) -> dict[str, list[dict]]:
        """Return cached chunks keyed by source file (ALL rows per file, in id order).

        Returns an empty dict for legacy/corrupt caches (no content-hash schema),
        which callers treat as "nothing cached — re-embed everything".
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
            if "source_file" in cols:
                select = "SELECT source_file, doc_name, content, content_hash, embedding FROM document_index ORDER BY id"
            else:
                # v2 caches kept only the display name; derive the source from it.
                select = "SELECT doc_name, doc_name, content, content_hash, embedding FROM document_index ORDER BY id"
            rows = conn.execute(select).fetchall()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Could not read index cache: %s", exc)
            return {}

        import pickle
        import re as _re
        result: dict[str, list[dict]] = {}
        for source, display, content, content_hash, emb_blob in rows:
            try:
                emb = pickle.loads(emb_blob)
            except Exception:
                continue
            if not emb:
                continue
            # v2 caches: source == display (both are the display name).
            # Derive the real source filename from it.
            if source == display and " [" in source:
                m = _re.match(r"^\S+\.(?:txt|md|csv|html|xml|rtf)\b", source)
                source = m.group(0) if m else source.split(" [")[0]
            result.setdefault(source, []).append({
                "doc_name": display,
                "content": content,
                "content_hash": content_hash,
                "embedding": emb,
            })
        return result

    async def _save_to_disk(self) -> None:
        if self._index is not None:
            await self._save_to_disk_from(self._index)

    async def _save_to_disk_from(self, idx: KBVectorIndex) -> None:
        """Persist the index to SQLite (atomic temp-file swap)."""
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
                import pickle
                emb_bytes = pickle.dumps(doc.embedding)
                rows.append(
                    (doc.source(), doc.display_name, doc.content,
                     _content_hash(doc.content), emb_bytes, now)
                )

            conn.executemany(
                "INSERT INTO document_index "
                "(source_file, doc_name, content, content_hash, embedding, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('kb_path', ?)",
                (str(self.kb_path),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
                (_SCHEMA_VERSION,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('updated_at', ?)",
                (str(now),),
            )
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

    async def _save_empty_cache(self) -> None:
        """Persist an empty (but schema-valid) cache so we don't retry forever."""
        await self._save_to_disk_from(KBVectorIndex())
