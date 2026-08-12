"""In-memory vector index for KB document retrieval using OpenAI-compatible backend embeddings.

Provides cosine-similarity based document ranking as an alternative to the
keyword-matching engine in reader.py.  Documents are **chunked** semantically
before embedding so that queries for specific topics (e.g., "time system")
hit only relevant sections — not drowned out by unrelated content.

Uses ``kb.embedder.Embedder`` powered by the configured INFER_URL
(OpenAI-compatible) backend with model ``nomic-embed-text:latest``.
"""
from __future__ import annotations

import math
import pathlib
import re
from dataclasses import dataclass, field


# ──────────────────────────── Chunking provider ──────────────────────

from kb.chunker import Chunker, _normalize_display_name
from kb.reader import _extract_ext


# ──────────────────────────── Embedding provider ──────────────────────

from kb.embedder import Embedder


# ──────────────────────────── Data structures ─────────────────────────

@dataclass
class _DocEntry:
    """Internal representation of an indexed document chunk."""
    display_name: str
    content: str
    embedding: list[float] | None = field(default=None, repr=False)


# ──────────────────────────── Helpers ─────────────────────────────────

# ──────────────────────────── Vector Index ───────────────────────────

class KBVectorIndex:
    """Lightweight in-memory vector index for KB documents.

    Build once at startup with ``KBVectorIndex.from_kb_path()``, then query
    with ``index.query("some text", top_n=5)``.

    Uses the configured inference backend (see ``kb.embedder``).
    Returns an empty index when the embedding backend is unreachable — caller
    should fall back to keyword search via ``kb.retrievers.is_vector_available()``.
    """

    def __init__(self) -> None:
        self._docs: list[_DocEntry] = []
        self._embedder = Embedder()

    # ── Construction ────────────────────────────────────────────────

    @classmethod
    async def from_kb_path(
        cls,
        kb_path: str | pathlib.Path,
        max_bytes_per_file: int = 1024 * 1024,
    ) -> KBVectorIndex:
        """Scan a KB directory and build the vector index.

        Documents are **semantically chunked** (by Markdown headers or paragraphs)
        before embedding so that each chunk targets a specific topic area.

        Returns an (possibly empty) ``KBVectorIndex``.  When embedding fails
        the caller should fall back to keyword retrieval.
        """
        root = pathlib.Path(kb_path)
        if not root.exists():
            return cls()  # empty index

        index = cls()

        entries: list[tuple[str, str]] = []  # (display_name_with_section, content)
        for p in sorted(root.rglob("*")):
            if not p.is_file() or "?" in p.name:
                continue
            ext = _extract_ext(p.name)
            if ext not in {".txt", ".md", ".csv", ".html", ".xml", ".rtf"}:
                continue

            # Let Chunker handle file reading and sizing internally
            chunks = await Chunker.split_file(p)
            for chunk in chunks:
                display_name = f"{chunk.display_name} [{chunk.section_path}]"
                entries.append((display_name, chunk.content))

        # If chunking produced nothing (e.g., tiny files), fall back to whole-file
        if not entries:
            for p in sorted(root.rglob("*")):
                if not p.is_file() or "?" in p.name:
                    continue
                ext = _extract_ext(p.name)
                if ext not in {".txt", ".md", ".csv", ".html", ".xml", ".rtf"}:
                    continue
                content_text = p.read_bytes().decode("utf-8", errors="replace")
                if not content_text or len(content_text) > max_bytes_per_file:
                    continue

                # No line cap — let the full content reach the Chunker for intelligent splitting
                display_name = _normalize_display_name(p, p.stem)
                entries.append((display_name, content_text.strip()))

        # Build the index (embeds all chunks via OpenWebUI backend)
        if entries:
            names, contents = zip(*entries)
            try:
                embeddings = await index._embedder.encode(list(contents))
                index._docs = [
                    _DocEntry(display_name=n, content=c, embedding=e)  # type: ignore[arg-type]
                    for n, c, e in zip(names, contents, embeddings)
                ]
            except Exception as exc:
                index._log_error(str(exc))

        return index

    def is_empty(self) -> bool:
        return len(self._docs) == 0

    def count(self) -> int:
        return len(self._docs)

    # ── Querying ────────────────────────────────────────────────────

    async def query(
        self,
        text: str,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Return the *top_n* most similar documents for *text*.

        Returns ``[(display_name, similarity_score), ...]`` sorted descending.
        Scores are cosine similarities in [0, 1].
        """
        if self.is_empty() or not text.strip():
            return []

        try:
            q_emb = await self._embedder.encode([text])
            q_emb = q_emb[0]
        except Exception:
            return []

        scored: list[tuple[str, float]] = []
        for doc in self._docs:
            emb = doc.embedding
            if emb is None:
                continue
            sim = _cosine_similarity(q_emb, emb)
            if sim > 0:
                scored.append((doc.display_name, sim))

        scored.sort(key=lambda t: -t[1])
        return scored[:top_n]

    @staticmethod
    def _log_error(msg: str) -> None:
        try:
            import logging
            logging.getLogger("kb.vector_db").warning("Vector index build failed: %s", msg)
        except Exception:
            pass


# ──────────────────────────── Utility ─────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
