"""In-memory vector index for KB document retrieval using fastembed embeddings.

Provides cosine-similarity based document ranking as an alternative to the
keyword-matching engine in reader.py.  Documents are indexed once and can be
queried efficiently without external services.
"""
from __future__ import annotations

import math
import os
import pathlib
import re
from dataclasses import dataclass, field


# ──────────────────────────── Lazy-import of fastembed ────────────────

try:
    from fastembed import TextEmbedding  # type: ignore[import-untyped]
except ImportError:
    TextEmbedding = None  # type: ignore[misc]


# ──────────────────────────── Data structures ─────────────────────────

@dataclass
class _DocEntry:
    """Internal representation of an indexed document chunk."""
    display_name: str
    content: str
    embedding: list[float] | None = field(default=None, repr=False)


# ──────────────────────────── Helpers ─────────────────────────────────

def _extract_ext(name: str) -> str:
    """Extract file extension (lowercased), stripping any query-string suffix."""
    base = name.split("?")[0]
    i = base.rfind(".")
    return base[i:].lower() if i > 0 else ""


def _normalize_display_name(p: pathlib.Path, base_name: str) -> str:
    """Build human-readable display name from path and filename."""
    stem = p.stem
    clean_stem = re.sub(r'^\d+', '', stem)
    return f"{clean_stem}{p.suffix}" if clean_stem else base_name


# ──────────────────────────── Vector Index ───────────────────────────

class KBVectorIndex:
    """Lightweight in-memory vector index for KB documents.

    Build once at startup with `KBVectorIndex.from_kb_path()`, then query
    with `index.query("some text", top_n=5)`.

    Requires `fastembed` and `onnxruntime` installed.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._docs: list[_DocEntry] = []
        self._model_name = model_name
        self._embedding_model: TextEmbedding | None = None

    # ── Construction ────────────────────────────────────────────────

    @classmethod
    def from_kb_path(
        cls,
        kb_path: str | pathlib.Path,
        max_lines_per_file: int = 50,
        max_bytes_per_file: int = 1024 * 1024,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> KBVectorIndex | None:
        """Scan a KB directory and build the vector index.

        Returns ``None`` if fastembed is not available.
        """
        if TextEmbedding is None:
            return None  # type: ignore[return-value]

        root = pathlib.Path(kb_path)
        if not root.exists():
            return None

        index = cls(model_name)

        entries: list[tuple[str, str]] = []
        for p in sorted(root.rglob("*")):
            if not p.is_file() or "?" in p.name:
                continue
            ext = _extract_ext(p.name)
            if ext not in {".txt", ".md"}:
                continue
            content_text = p.read_bytes().decode("utf-8", errors="replace")
            if not content_text or len(content_text) > max_bytes_per_file:
                continue

            lines = content_text.splitlines()[:max_lines_per_file]
            truncated = "\n".join(lines)
            if len(content_text.splitlines()) > max_lines_per_file:
                truncated += "\n... [truncated]"

            base_name = os.path.basename(p.name)
            display_name = _normalize_display_name(p, base_name)
            entries.append((display_name, truncated))

        # Build the index (embeds all documents)
        if entries:
            names, contents = zip(*entries)
            try:
                embeddings = list(index._get_model().embed(contents))  # type: ignore[attr-defined]
                index._docs = [
                    _DocEntry(display_name=n, content=c, embedding=e.tolist())
                    for n, c, e in zip(names, contents, embeddings)
                ]
            except Exception as exc:
                os.environ.setdefault("KB_VECTOR_INDEX_LOG", "1")  # signal caller to fall back
                index._log_error(str(exc))

        return index

    def is_empty(self) -> bool:
        return len(self._docs) == 0

    def count(self) -> int:
        return len(self._docs)

    # ── Querying ────────────────────────────────────────────────────

    def query(
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

        q_emb = self._get_query_embedding(text)
        if q_emb is None:
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

    # ── Internal helpers ────────────────────────────────────────────

    _EMBED_DIM = 384  # bge-small-en-v1.5 dimension

    def _get_model(self) -> TextEmbedding:
        if self._embedding_model is None:
            self._embedding_model = TextEmbedding(
                model_name=self._model_name,
                cuda=False,
            )
        return self._embedding_model

    def _get_query_embedding(self, text: str) -> list[float] | None:
        """Encode query text to a vector. Returns None on failure."""
        try:
            emb = list(self._get_model().embed([text]))
            if emb and len(emb[0]) == self._EMBED_DIM:
                return emb[0].tolist()
        except Exception:
            pass
        return None

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
