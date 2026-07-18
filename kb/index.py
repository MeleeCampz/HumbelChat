"""Persistent vector index for KB document retrieval.

Caches the in-memory ``KBVectorIndex`` to disk (SQLite) so bot restarts
don't require re-indexing the entire knowledge base — saving seconds of
startup time and avoiding repeated API calls to the embedding backend.

Supports incremental updates: adding or removing a single file triggers an
index update without rebuilding from scratch.

Usage
-----
    from kb.index import KBIndexStore

    store = KBIndexStore("path/to/kb", persist_dir="kb/.index_cache")

    # Load (creates index on disk if exists, or builds & saves)
    await store.load()

    # Get the in-memory vector index
    idx = store.get_index()
    results = await idx.query("time system", top_n=5)

    # Update after adding/removing a file
    await store.update_single_document("new_doc.md")  # re-indexes only this file
    await store.remove_document("old_doc.txt")        # removes from index

"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import sqlite3
import time
from typing import Any, Optional

from kb.vector_db import KBVectorIndex, _DocEntry
from kb.embedder_openai import OpenAIEmbedder

logger = logging.getLogger("kb.index")


# ──────────────────────────── Schema ────────────────────────────────────

_SCHEMA_CREATE_DOC_INDEX = """\
CREATE TABLE IF NOT EXISTS document_index (
    file_path   TEXT PRIMARY KEY,
    doc_name    TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   BLOB NOT NULL,          -- pickle'd list[float]
    updated_at  REAL DEFAULT (strftime('%s','now'))
);
"""

_SCHEMA_CREATE_METADATA = """\
CREATE TABLE IF NOT EXISTS metadata (
    key       TEXT PRIMARY KEY,
    value     TEXT NOT NULL
);
"""


class KBIndexStore:
    """Persistent vector index with SQLite caching and incremental updates."""

    def __init__(
        self,
        kb_path: str | pathlib.Path,
        *,
        persist_dir: str = "kb/.index_cache",
        model_name: str = "nomic-embed-text:latest",
        max_lines_per_file: int = 50,
        max_bytes_per_file: int = 1024 * 1024,
    ) -> None:
        self.kb_path = pathlib.Path(kb_path)
        self.persist_dir = pathlib.Path(persist_dir)
        self.max_lines = max_lines_per_file
        self.max_bytes = max_bytes_per_file

        self._db_path = self.persist_dir / "vector_index.db"
        self._index: Optional[KBVectorIndex] = None
        self._embedder = OpenAIEmbedder(model_name=model_name)

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def load(self, force_rebuild: bool = False) -> KBVectorIndex:
        """Load or build the vector index.

        1. If a valid SQLite cache exists (and not forced rebuild), load embeddings from disk.
        2. Otherwise, build from scratch using OpenWebUI /embeddings endpoint.
        3. Save to disk for future runs.

        Parameters
        ----------
        force_rebuild : If True, skips the cache and always rebuilds from KB files.
        """
        if self._index is not None:
            return self._index

        # If we have a valid in-memory index, reuse it
        # (Note: _index is only set below after load completes)

        # Try loading from disk cache first — unless forced to rebuild
        if not force_rebuild and self._db_path.exists() and self._is_cache_valid():
            logger.info("Loading vector index from SQLite cache (%s)", self._db_path)
            self._index = await self._load_from_disk()
            if self._index is not None and not self._index.is_empty():
                return self._index
            # Cache was valid but empty — fall through to rebuild below
            logger.warning("SQLite cache exists but has 0 entries; rebuilding from KB files")

        # Build fresh (or forced rebuild)
        logger.info("Building vector index from scratch for '%s'", self.kb_path)
        self._index = await KBVectorIndex.from_kb_path(
            self.kb_path,
            max_lines_per_file=self.max_lines,
            max_bytes_per_file=self.max_bytes,
        )
        if self._index is None or self._index.is_empty():
            logger.error("Index build produced empty result for '%s' — check embedding backend connectivity", self.kb_path)
        else:
            await self._save_to_disk()
            logger.info("Index built and saved to cache (%d docs)", self._index.count())
        return self._index

    async def shutdown(self) -> None:
        """Save index before shutdown (belt-and-suspenders)."""
        if self._index is not None and not self._index.is_empty():
            await self._save_to_disk()

    # ── Public Updates ──────────────────────────────────────────────────

    async def update_single_document(self, file_path: str | pathlib.Path) -> bool:
        """Re-index a single document (add or replace). Returns True on success."""
        if self._index is None or self._index.is_empty():
            # Index doesn't exist yet — rebuild entire index
            logger.warning("No active index; rebuilding full index after adding '%s'", file_path)
            self._index = await KBVectorIndex.from_kb_path(
                self.kb_path, max_lines_per_file=self.max_lines, max_bytes_per_file=self.max_bytes
            )
            if self._index is not None:
                await self._save_to_disk()
            return self._index is not None and not self._index.is_empty()

        # Re-embed this specific file and replace its chunk in the index
        from kb.chunker import Chunker

        path = pathlib.Path(file_path)
        if not path.exists():
            logger.warning("File '%s' does not exist; skipping update", file_path)
            return False

        chunks = await Chunker.split_file(path, max_lines_per_file=self.max_lines)
        if not chunks:
            # No chunked content — fall back to whole-file blob
            content_text = path.read_bytes().decode("utf-8", errors="replace")
            lines = content_text.splitlines()[: self.max_lines]
            truncated = "\n".join(lines)
            if len(content_text.splitlines()) > self.max_lines:
                truncated += "\n... [truncated]"
            chunks = [
                type("_SingleChunk", (), {
                    "display_name": pathlib.Path(file_path).stem,
                    "source_file": path.name,
                    "section_path": "Full file",
                    "content": truncated,
                })()
            ]

        embeddings = await self._embedder.encode([c.content for c in chunks])

        # Update in-memory index by reconstructing _docs with new vectors
        old_docs = self._index._docs  # type: ignore[attr-defined]
        new_docs: list[_DocEntry] = []
        chunk_index = 0

        for doc in old_docs:
            if chunk_index < len(chunks):
                chunk = chunks[chunk_index]
                # Check if this is the file we're updating (by source_file or name match)
                if chunk.source_file == path.name or self._matches_file(doc, file_path):
                    new_docs.append(_DocEntry(
                        display_name=f"{chunk.display_name} [{chunk.section_path}]",
                        content=chunk.content,
                        embedding=embeddings[chunk_index],
                    ))
                    chunk_index += 1
                else:
                    new_docs.append(doc)
            else:
                new_docs.append(doc)

        self._index._docs = new_docs  # type: ignore[attr-defined]
        await self._save_to_disk()
        logger.info("Updated index with '%s' (%d chunks)", file_path, len(chunks))
        return True

    async def remove_document(self, file_path: str | pathlib.Path) -> bool:
        """Remove a document from the index. Returns True if something was removed."""
        if self._index is None or self._index.is_empty():
            return False

        path = pathlib.Path(file_path)
        old_count = self._index.count()
        new_docs = [
            doc for doc in self._index._docs  # type: ignore[attr-defined]
            if not self._matches_file(doc, file_path)
        ]

        removed = len(old_docs) - len(new_docs)
        if removed > 0:
            self._index._docs = new_docs  # type: ignore[attr-defined]
            await self._save_to_disk()
            logger.info("Removed %d chunk(s) for '%s'", removed, file_path)
            return True

        logger.warning("No matching chunks found to remove for '%s'", file_path)
        return False

    # ── Querying ────────────────────────────────────────────────────────

    def get_index(self) -> Optional[KBVectorIndex]:
        """Return the in-memory vector index (or None if not loaded)."""
        return self._index

    # ── Internal Helpers ────────────────────────────────────────────────

    def _matches_file(self, doc: _DocEntry, file_path: str | pathlib.Path) -> bool:
        """Check if a document entry belongs to the given file."""
        path_name = pathlib.Path(file_path).name.lower()
        display = doc.display_name.lower()
        return path_name in display

    def _is_cache_valid(self) -> bool:
        """Check if SQLite cache is newer than any KB file."""
        kb_files = list(self.kb_path.rglob("*"))
        if not kb_files:
            return False

        # Check mtime of all KB files vs. the db file
        db_mtime = self._db_path.stat().st_mtime
        for f in kb_files:
            if f.is_file() and f.suffix.lower() in {".txt", ".md"}:
                if f.stat().st_mtime > db_mtime:
                    logger.debug("KB file '%s' newer than cache — invalidating", f.name)
                    return False
        return True

    async def _load_from_disk(self) -> Optional[KBVectorIndex]:
        """Load index entries from SQLite and reconstruct a KBVectorIndex."""
        if not self._db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(self._db_path))
            # Ensure tables exist before querying (IF NOT EXISTS is safe to call repeatedly)
            conn.execute(_SCHEMA_CREATE_DOC_INDEX)
            conn.commit()
            conn.execute(_SCHEMA_CREATE_METADATA)
            conn.commit()
            cursor = conn.execute("SELECT doc_name, content, embedding FROM document_index ORDER BY updated_at DESC")
        except sqlite3.Error as exc:
            logger.error("Failed to load SQLite index: %s", exc)
            return None

        docs: list[_DocEntry] = []
        for name, content, emb_blob in cursor:
            # Embeddings are stored as JSON text in SQLite
            if isinstance(emb_blob, str):
                emb_data = json.loads(emb_blob)
            else:
                import pickle  # type: ignore[import-untyped]
                try:
                    emb_data = pickle.loads(emb_blob)  # type: ignore[arg-type]
                except Exception:
                    continue

            docs.append(_DocEntry(display_name=name, content=content, embedding=emb_data))

        conn.close()

        if not docs:
            return None

        # Create a minimal index with pre-loaded docs (skip re-embedding)
        idx = KBVectorIndex.__new__(KBVectorIndex)  # bypass __init__
        idx._docs = docs
        idx._embedder = self._embedder
        return idx

    async def _save_to_disk(self) -> None:
        """Save the in-memory index to SQLite."""
        if not self._index or self._index.is_empty():
            return

        try:
            os.makedirs(self.persist_dir, exist_ok=True)

            conn = sqlite3.connect(str(self._db_path))
            # Execute each CREATE TABLE separately (sqlite3 prohibits multi-statement execute)
            conn.execute(_SCHEMA_CREATE_DOC_INDEX)
            conn.commit()
            conn.execute(_SCHEMA_CREATE_METADATA)
            conn.commit()

            # Clear existing cache (simple approach — replace entire table)
            conn.execute("DELETE FROM document_index")

            now = time.time()
            rows = []
            for doc in self._index._docs:  # type: ignore[attr-defined]
                import pickle  # type: ignore[import-untyped]
                emb_bytes = pickle.dumps(doc.embedding) if doc.embedding else b""
                rows.append((doc.display_name, doc.content, emb_bytes, now))

            conn.executemany(
                "INSERT INTO document_index (doc_name, content, embedding, updated_at) VALUES (?, ?, ?, ?)",
                rows,
            )

            # Metadata
            conn.execute("DELETE FROM metadata WHERE key='kb_path'")
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('kb_path', ?)",
                (str(self.kb_path),),
            )
            conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('updated_at', ?)", (str(now),))

            conn.commit()
            conn.close()
            logger.debug("Saved %d entries to SQLite cache", len(rows))

        except Exception as exc:
            logger.error("Failed to save index to disk: %s", exc)
