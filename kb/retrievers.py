"""Unified KB document retriever — keyword or vector similarity strategies.

Provides a single entry point ``retrieve_kb_documents()`` that bot_core.py
calls for RAG context injection.  The active strategy is controlled by the
environment variable **RAG_RETRIEVAL_METHOD** (default: ``vector``).

Available strategies
--------------------
keyword — heuristic scoring of filenames, headers, and body overlap
          (existing engine in kb.reader).

vector  — cosine-similarity embedding search via the configured inference backend
          using model ``nomic-embed-text:latest``.  Documents are chunked
          semantically before embedding and indexed on first use with
          SQLite persistence for fast bot restarts.

Usage
-----
    from kb.retrievers import retrieve_kb_documents, KB_STRATEGIES

    results = retrieve_kb_documents(
        query="Tell me about the unique time system in humblewood.",
        kb_path="/path/to/knowledge",
        strategy="vector",       # or "keyword" (default)
        top_n=5,
    )

Both strategies return ``list[tuple[str, str]]`` of ``(display_name, content)``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kb.index import KBIndexStore

logger = logging.getLogger("kb.retrievers")

# ───────────────────────────── Constants ──────────────────────────────

KB_STRATEGIES = frozenset({"keyword", "vector"})
DEFAULT_METHOD = os.getenv("RAG_RETRIEVAL_METHOD", "vector").lower()

# Hard timeout on RAG retrieval (in seconds) — prevents blocking the entire request
_RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT_SECONDS", "30"))

# Singleton index store — lazily initialized
_index_store: Optional["KBIndexStore"] = None
_kb_path_for_store: str | pathlib.Path | None = None
# Serializes lazy init so two simultaneous first RAG requests can't each build
# a KBIndexStore and race on the same SQLite temp-file swap.
_index_init_lock: Optional[asyncio.Lock] = None


def _get_index_init_lock() -> asyncio.Lock:
    global _index_init_lock
    if _index_init_lock is None:
        _index_init_lock = asyncio.Lock()
    return _index_init_lock


async def _ensure_index_store(kb_path: str | pathlib.Path) -> Optional["KBIndexStore"]:
    """Create or return the cached index store, loading/building the index."""
    global _index_store, _kb_path_for_store

    if _index_store is not None:
        return _index_store  # already built

    async with _get_index_init_lock():
        # Re-check inside the lock — another coroutine may have finished init.
        if _index_store is not None:
            return _index_store

        from kb.index import KBIndexStore

        store = KBIndexStore(kb_path)
        await store.load()
        _index_store = store
        _kb_path_for_store = kb_path
        logger.info("Vector index store ready (%d chunks)", store.get_index().count() if store.get_index() else 0)
    return _index_store


# ───────────────────────────── Helpers ──────────────────────────────

async def _keyword_fallback(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
    window_lines: int,
) -> list[tuple[str, str]]:
    """Fallback to keyword/TF-IDF retrieval when vector index is unavailable."""
    from kb.reader import read_kb_files, get_relevant_chunks

    scored = read_kb_files(kb_path, query=query, top_n=top_n * 3, max_lines_per_file=300)
    doc_names = [name for name, _ in scored[:top_n]] if scored else []
    chunks = get_relevant_chunks(kb_path, doc_names, query=query, window_lines=window_lines)
    logger.info(
        "Keyword fallback: %d files ranked → %d relevant chunk(s) with ~%.0f chars",
        len(scored), len(chunks),
        sum(len(c) for _, c in chunks) if chunks else 0,
    )
    return chunks


# ───────────────────── Adaptive-k Retrieval ─────────────────────

def _adaptive_k_threshold(scores: list[float]) -> int:
    """Determine optimal k using the largest gap in sorted similarity scores.

    Implements the Adaptive-k method: finds the position of the steepest drop
    in similarity scores, which corresponds to the boundary between relevant
    and irrelevant documents. Returns the count of chunks to retrieve (k).

    A small buffer (5) is added after the threshold to avoid missing marginal docs.
    """
    if len(scores) <= 1:
        return len(scores)

    # Sort descending (should already be, but ensure it)
    sorted_scores = sorted(scores, reverse=True)

    # Find gaps between consecutive scores
    max_gap = -1.0
    gap_idx = 0

    for i in range(len(sorted_scores) - 1):
        gap = sorted_scores[i] - sorted_scores[i + 1]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    # The optimal k is the index at the largest gap, with a small buffer
    k = gap_idx + 1  # index to count conversion (1-based)
    k += 5  # buffer to capture nearby candidates
    return min(k, len(sorted_scores))


# ───────────── FlashRank Marginal Utility Reranking ─────────────

def _flashrank_reorder(
    results: list[tuple[float, str, str]],
    top_n: int = 5,
) -> list[tuple[str, str]]:
    """Select chunks using marginal utility (FlashRank-style).

    Greedily selects documents that maximize information gain per token while
    avoiding redundancy. Similar/contributing documents are deprioritized in
    favor of novel, complementary evidence.

    Parameters
    ----------
    results : list of (score, name, content) sorted by relevance descending
    top_n : maximum number of chunks to return

    Returns
    -------
    Reordered list of (name, content) tuples representing the optimal selection.
    """
    if not results:
        return []

    # Track what we've already seen for redundancy checking
    seen_stems: set[str] = set()  # document stems to avoid near-duplicate files

    selected: list[tuple[str, str]] = []

    # First pass: compute per-document utility scores and deduplicate file stems
    doc_entries: list[tuple[float, str, str]] = []  # (utility, name, content)
    for score, name, content in results:
        stem = name.split(" [")[0] if " [" in name else name
        if stem not in seen_stems:
            seen_stems.add(stem)
            # Utility = relevance score × information density (chars per token ratio)
            char_len = len(content)
            token_est = max(1, char_len // 4)  # rough token estimate
            info_density = min(char_len / token_est, 2.0) if token_est > 0 else 1.0
            marginal_util = score * info_density
            doc_entries.append((marginal_util, name, content))

    # Sort by marginal utility descending
    doc_entries.sort(key=lambda x: -x[0])

    # Second pass: greedy selection
    for util, name, content in doc_entries[:top_n]:
        selected.append((name, content))

    return selected


# ───────────────────────────── Strategies ─────────────────────────────

def _retrieve_keyword(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
) -> list[tuple[str, str]]:
    """Keyword-based retrieval using the existing heuristic engine.

    Phase 1: Score all files against *query* terms using a lightweight scan
             (first 300 lines is sufficient for relevance ranking).
    Phase 2: Extract only relevant line-windows from the top-N documents
             via get_relevant_chunks(), avoiding full-file dump in context.
    """
    from kb.reader import read_kb_files, get_relevant_chunks
    from config.settings import RAG_WINDOW_LINES

    # Phase 1 — quick scoring pass (300 lines is plenty for keyword overlap)
    scored = read_kb_files(kb_path, query=query, top_n=top_n * 3, max_lines_per_file=300)
    if not scored:
        return []

    # Phase 2 — extract only matched windows from top documents
    doc_names = [name for name, _ in scored[:top_n]]
    chunks = get_relevant_chunks(kb_path, doc_names, query=query, window_lines=RAG_WINDOW_LINES)

    logger.info(
        "Keyword retrieval: %d files ranked → %d relevant chunk(s) with ~%.0f chars",
        len(scored), len(chunks),
        sum(len(c) for _, c in chunks) if chunks else 0,
    )
    return chunks


async def _retrieve_vector(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
    window_lines: int = 80,
) -> list[tuple[str, str]]:
    """Vector-based retrieval using the persist-backed index.

    Pure cosine-similarity ranking over the in-memory index (which the
    disk-backed store keeps fully hydrated at load time).  Falls back to
    keyword-only if the vector index is unavailable or the query returns
    no hits.  Only **one** embedding API call is made per query.
    """
    from kb.reader import get_relevant_chunks
    from config.settings import RAG_WINDOW_LINES

    store = await _ensure_index_store(kb_path)
    idx = store.get_index() if store is not None else None
    if idx is None or idx.is_empty():
        logger.warning("Vector index unavailable or empty for '%s'; falling back to keyword", kb_path)
        return await _keyword_fallback(query, kb_path, top_n, window_lines)

    # One embedding call: rank the in-memory chunks AND keep the query vector.
    ranked, _q_emb = await idx.query_with_embeddings(query, top_n=min(top_n * 4, 32))
    if not ranked:
        logger.warning("Vector query returned no hits for '%s'; falling back to keyword", kb_path)
        return await _keyword_fallback(query, kb_path, top_n, window_lines)

    # Deduplicate by source file, preserving rank order, cap at top_n files.
    doc_stems: list[str] = []
    seen: set[str] = set()
    for name, _content, _score in ranked:
        stem = name.split(" [")[0] if " [" in name else name
        if stem not in seen:
            seen.add(stem)
            doc_stems.append(stem)
    doc_stems = doc_stems[:top_n]

    ranked_list = get_relevant_chunks(kb_path, doc_stems, query=query, window_lines=RAG_WINDOW_LINES)
    if not ranked_list:
        # Disk extraction failed (e.g. files moved) — serve chunk content directly.
        ranked_list = [(name, content) for name, content, _ in ranked[:top_n]]

    logger.info(
        "Vector retrieval: %d ranked chunk(s) → %d file(s) → %d relevant chunk(s) with ~%.0f chars",
        len(ranked), len(doc_stems), len(ranked_list),
        sum(len(c) for _, c in ranked_list) if ranked_list else 0,
    )
    return ranked_list


# ───────────────────────────── Public API ──────────────────────────────

async def retrieve_kb_documents(
    query: str,
    kb_path: str | pathlib.Path,
    *,
    strategy: str = DEFAULT_METHOD,
    top_n: int = 5,
    window_lines: int = 80,
) -> list[tuple[str, str]]:
    """Retrieve relevant KB documents for *query* using the selected strategy.

    Parameters
    ----------
    query : The user's question / prompt used for retrieval.
    kb_path : Path to the knowledge-base root directory.
    strategy : ``"keyword"`` or ``"vector"`` (default: ``"vector"``).
    top_n : Soft cap on documents retrieved.
    window_lines : Lines above/below each match anchor (default 80).

    Returns
    -------
    list of ``(display_name, content)`` tuples in relevance order.

    Notes
    -----
    This function is **asynchronous** — callers must await it:
        ``results = await retrieve_kb_documents(query=query, kb_path=kb_path, strategy="vector")``
    """
    method = strategy.lower() if strategy else DEFAULT_METHOD

    if method == "keyword":
        return _retrieve_keyword(query, kb_path, top_n)

    if method == "vector":
        return await _retrieve_vector(query, kb_path, top_n, window_lines=window_lines)

    # Unknown strategy — fall back to keyword with a warning
    logger.warning(
        "Unknown retrieval strategy '%s'; falling back to keyword", method
    )
    return _retrieve_keyword(query, kb_path, top_n)


async def update_kb_document(file_path: str | pathlib.Path) -> bool:
    """Re-index or add a single KB document. Use after ``!add_kb_file``."""
    global _index_store, _kb_path_for_store
    if _index_store is not None and _index_store.get_index() is not None:
        return await _index_store.update_single_document(file_path)

    # No store yet — initialize one (loads the disk cache) and update through it.
    logger.warning("No index loaded; initializing store for single-document update")
    from kb.index import KBIndexStore
    from config.settings import KB_PATH
    store = KBIndexStore(KB_PATH)
    await store.load()
    if store.get_index() is None or store.get_index().is_empty():
        return False
    _index_store = store
    _kb_path_for_store = KB_PATH
    return await _index_store.update_single_document(file_path)


async def remove_kb_document(file_path: str | pathlib.Path) -> bool:
    """Remove a KB document from the vector index."""
    if _index_store is None or _index_store.get_index() is None:
        return False
    return await _index_store.remove_document(file_path)


async def shutdown_vector_store() -> None:
    """Persist index before bot shutdown."""
    if _index_store is not None:
        await _index_store.shutdown()
        logger.info("Vector index store shut down and persisted")


def get_available_strategies() -> list[str]:
    """Return the list of available retrieval strategies."""
    from config.settings import INFER_URL, INFER_API_KEY

    has_vector = bool(INFER_URL and INFER_API_KEY)
    strategies = ["keyword"]
    if has_vector:
        strategies.append("vector")
    return strategies


def is_vector_available() -> bool:
    """Quick check whether the vector retrieval backend is configured."""
    from config.settings import INFER_URL, INFER_API_KEY

    return bool(INFER_URL and INFER_API_KEY)
