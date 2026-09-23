"""In-memory vector index for KB document retrieval using OpenAI-compatible backend embeddings.

Provides cosine-similarity based document ranking as an alternative to the
keyword-matching engine in reader.py.  Documents are **chunked** semantically
before embedding so that queries for specific topics (e.g., "time system")
hit only relevant sections — not drowned out by unrelated content.

Uses ``kb.embedder.Embedder`` powered by the configured INFER_URL
(OpenAI-compatible) backend with model from ``EMBEDDING_MODEL`` env var (default: ``nomic-embed-text:latest``).
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field


# ──────────────────────────── Chunking provider ──────────────────────

from kb.chunker import Chunker
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
    source_file: str | None = field(default=None, repr=False)
    # P2 #19: SHA-256 of the SOURCE file on disk. Lets the cache skip
    # re-chunking a file whose content is unchanged (chunker is deterministic).
    file_hash: str | None = field(default=None, repr=False)

    def source(self) -> str:
        """Best-effort original filename for this chunk (for cache bookkeeping)."""
        if self.source_file:
            return self.source_file
        # Legacy entries: display name is "name.md [Section]" — the stem is the file.
        return self.display_name.split(" [")[0]


# ──────────────────────────── Helpers ─────────────────────────────────

# ──────────────────────────── Vector Index ───────────────────────────

class KBVectorIndex:
    """Lightweight in-memory vector index for KB documents.

    Build once at startup with ``KBVectorIndex.from_kb_path()``, then query
    with ``index.query("some text", top_n=5)``.

    Uses the configured inference backend (see ``kb.embedder``).
    Returns an empty index when the embedding backend is unreachable — caller
    should fall back to keyword search (see ``kb.retrievers._keyword_fallback``).
    """

    def __init__(self) -> None:
        self._docs: list[_DocEntry] = []
        # P2 #17: lazily-built numpy (matrix, norms) cache, invalidated whenever
        # _docs is *replaced* (tracked by object identity + length). None until
        # first query.
        self._mat_cache: tuple | None = None
        from config.settings import EMBEDDING_MODEL
        self._embedder = Embedder(model_name=EMBEDDING_MODEL)

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

        entries: list[tuple[str, str, str]] = []  # (display_name_with_section, content, source_file)
        for p in sorted(root.rglob("*")):
            if not p.is_file() or "?" in p.name:
                continue
            ext = _extract_ext(p.name)
            if ext not in {".txt", ".md", ".csv", ".html", ".xml", ".rtf"}:
                continue

            # Key by the KB-relative path (unique across subfolders) so per-session
            # files with identical basenames don't collide.  Mirrors the keying
            # used by KBIndexStore in kb.index (computed inline here rather than
            # imported, to avoid a circular import between the two modules).
            try:
                src_key = p.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                src_key = p.name

            # Let Chunker handle file reading and sizing internally
            chunks = await Chunker.split_file(p)
            for chunk in chunks:
                display_name = f"{chunk.display_name} [{chunk.section_path}]"
                entries.append((display_name, chunk.content, src_key))


        # Build the index (embeds all chunks via OpenWebUI backend)
        if entries:
            names, contents, sources = zip(*entries)
            try:
                embeddings = await index._embedder.encode(list(contents))
                index._docs = [
                    _DocEntry(display_name=n, content=c, embedding=e, source_file=s)  # type: ignore[arg-type]
                    for n, c, e, s in zip(names, contents, embeddings, sources)
                ]
            except Exception as exc:
                index._log_error(str(exc))

        return index

    @classmethod
    def from_entries(
        cls,
        entries: list[tuple[str, str, str]],  # (display_name, content, source_file)
        embeddings: list[list[float]],
        file_hashes: list[str | None] | None = None,  # P2 #19
    ) -> KBVectorIndex:
        """Build an index from pre-embedded entries (no API calls)."""
        index = cls.__new__(cls)
        from config.settings import EMBEDDING_MODEL
        index._embedder = Embedder(model_name=EMBEDDING_MODEL)
        index._mat_cache = None  # P2 #17
        docs: list[_DocEntry] = []
        for (n, c, s), e in zip(entries, embeddings):
            fh = file_hashes[len(docs)] if file_hashes else None
            docs.append(_DocEntry(display_name=n, content=c, embedding=e,
                                  source_file=s, file_hash=fh))
        index._docs = docs
        return index

    def is_empty(self) -> bool:
        return len(self._docs) == 0

    def count(self) -> int:
        return len(self._docs)

    # ── P2 #17: numpy-accelerated ranking (pure-Python fallback) ──

    def _ensure_matrix(self) -> tuple:
        """Build (matrix, norms, n) from ``_docs``, caching the result.

        *matrix* is a float32 ``(n, D)`` array holding every document's
        embedding (rows with no embedding are all-zero). *norms* are its
        per-row L2 norms.  Rebuilt only when ``_docs`` is *replaced* (a new
        list object or a different length), so repeated queries reuse the
        matrix and each query is a single C-level matmul rather than an
        O(N·D) Python loop.
        """
        cache = self._mat_cache
        if (cache is not None and cache[2] is self._docs
                and cache[3] == len(self._docs)):
            return cache[0], cache[1], cache[3]
        import numpy as np

        docs = self._docs
        n = len(docs)
        d = next((len(doc.embedding) for doc in docs if doc.embedding), 0)
        if n and d:
            mat = np.zeros((n, d), dtype="float32")
            for i, doc in enumerate(docs):
                emb = doc.embedding
                if emb:
                    mat[i] = emb
        else:
            mat = np.zeros((n, 0), dtype="float32")
        norms = np.linalg.norm(mat, axis=1)
        self._mat_cache = (mat, norms, docs, n)
        return mat, norms, n

    def _rank_py(self, q_emb: list[float], top_n: int) -> list[tuple[float, int]]:
        """Pure-Python cosine ranking (fallback when numpy is unavailable).

        Returns ``[(similarity, doc_index), ...]`` sorted descending, where
        *doc_index* is an index into ``self._docs``.
        """
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._docs):
            emb = doc.embedding
            if emb is None:
                continue
            sim = _cosine_similarity(q_emb, emb)
            if sim > 0:
                scored.append((sim, i))
        scored.sort(key=lambda t: -t[0])
        return scored[:top_n]

    def _rank(self, q_emb: list[float], top_n: int) -> list[tuple[float, int]]:
        """Top-*top_n* docs for *q_emb* as ``[(similarity, doc_index), ...]``.

        Uses a numpy matmul (P2 #17); any failure (numpy missing, ragged
        embeddings, …) falls back to :meth:`_rank_py`.
        """
        if self.is_empty() or not q_emb:
            return []
        try:
            import numpy as np
        except Exception:
            return self._rank_py(q_emb, top_n)
        try:
            matrix, norms, n = self._ensure_matrix()
            if matrix.size == 0:
                return []
            q = np.asarray(q_emb, dtype="float32")
            qn = float(np.linalg.norm(q))
            if qn == 0.0:
                return []
            with np.errstate(invalid="ignore", divide="ignore"):
                sims = (matrix @ q) / (norms * qn)
            sims = np.where(np.isfinite(sims) & (sims > 0), sims, 0.0)
            if top_n >= n:
                order = np.argsort(-sims)
            else:
                part = np.argpartition(-sims, top_n - 1)[:top_n]
                order = part[np.argsort(-sims[part])]
            out: list[tuple[float, int]] = []
            for i in order:
                s = float(sims[i])
                if s <= 0.0:
                    break
                out.append((s, int(i)))
                if len(out) >= top_n:
                    break
            return out
        except Exception:
            return self._rank_py(q_emb, top_n)

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

        # P2 #17: numpy matmul ranking (falls back to pure Python internally).
        ranked = self._rank(q_emb, top_n)
        return [(self._docs[i].display_name, sim) for sim, i in ranked]

    async def query_with_embeddings(
        self,
        text: str,
        top_n: int = 5,
    ) -> tuple[list[tuple[str, str, float]], list[float]]:
        """Query and also return the query embedding.

        Returns ``(results, q_embedding)`` where ``results`` is
        ``[(display_name, content, similarity), ...]`` sorted descending.
        Callers that need to score additional chunks (e.g. disk-backed
        fallbacks) can reuse *q_embedding* instead of paying for a second
        embedding API call.
        """
        if self.is_empty() or not text.strip():
            return [], []

        try:
            q_emb = (await self._embedder.encode([text]))[0]
        except Exception:
            return [], []

        # P2 #17: numpy matmul ranking (falls back to pure Python internally).
        ranked = self._rank(q_emb, top_n)
        scored = [
            (self._docs[i].display_name, self._docs[i].content, sim)
            for sim, i in ranked
        ]
        return scored, q_emb

    async def rank_texts(
        self,
        texts: list[str],
        top_n: int = 5,
    ) -> list[list[tuple[str, str, float]]]:
        """Embed multiple queries in one batched call and rank the index against each.

        Returns one ranked ``[(display_name, content, similarity), ...]`` list
        per input text (same order as *texts*).  Used by the low-confidence
        query-rewrite path to merge several query rankings via RRF without
        paying for a separate embedding call per expansion.  Raises whatever
        the embedder raises when the backend is unreachable.
        """
        if self.is_empty():
            return [[] for _ in texts]

        embeddings = await self._embedder.encode(texts)  # one batched call (dedups inside)

        results: list[list[tuple[str, str, float]]] = []
        for emb in embeddings:
            if emb:
                # P2 #17: numpy matmul ranking (falls back to pure Python internally).
                ranked = self._rank(emb, top_n)
                scored = [
                    (self._docs[i].display_name, self._docs[i].content, sim)
                    for sim, i in ranked
                ]
            else:
                scored = []
            results.append(scored)
        return results

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
