"""Unified KB document retriever — keyword or vector similarity strategies.

Provides a single entry point ``retrieve_kb_documents()`` that bot_core.py
calls for RAG context injection.  The active strategy is controlled by the
environment variable **RAG_RETRIEVAL_METHOD** (default: ``keyword``).

Available strategies
--------------------
keyword — heuristic scoring of filenames, headers, and body overlap
          (existing engine in kb.reader).

vector  — cosine-similarity embedding search via fastembed / BAAI/bge-small-en-v1.5.
          Documents are indexed once at module import time; query-time
          computation is a single vector encoding + dot products.

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

import os
import pathlib
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: nocover
    from kb.vector_db import KBVectorIndex


# ───────────────────────────── Constants ──────────────────────────────

KB_STRATEGIES = frozenset({"keyword", "vector"})
DEFAULT_METHOD = os.getenv("RAG_RETRIEVAL_METHOD", "keyword").lower()

# Singleton vector index (lazily initialized once the first time it's needed)
_vector_index: "KBVectorIndex | None" = None
_index_lock = threading.Lock()
_kb_path_for_index: str | pathlib.Path | None = None


def _ensure_vector_index(
    kb_path: str | pathlib.Path,
) -> "KBVectorIndex | None":
    """Build the vector index once and cache it (thread-safe)."""
    global _vector_index, _kb_path_for_index

    if _vector_index is not None:
        return _vector_index  # already built

    with _index_lock:
        # Double-check after acquiring lock
        if _vector_index is not None:
            return _vector_index

        from kb.vector_db import KBVectorIndex  # type: ignore[import]

        new_index = KBVectorIndex.from_kb_path(kb_path)
        if new_index is not None and not new_index.is_empty():
            _vector_index = new_index
            _kb_path_for_index = kb_path

    return _vector_index


# ───────────────────────────── Strategies ─────────────────────────────

def _retrieve_keyword(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
) -> list[tuple[str, str]]:
    """Keyword-based retrieval using the existing heuristic engine."""
    from kb.reader import read_kb_files  # type: ignore[import]

    return read_kb_files(kb_path, query=query, top_n=top_n)


def _retrieve_vector(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
) -> list[tuple[str, str]]:
    """Vector-similarity retrieval using the in-memory fastembed index.

    Falls back to keyword if the vector index is unavailable.
    """
    try:
        from kb.reader import read_kb_files  # type: ignore[import]
    except ImportError:
        read_kb_files = None  # pragma: nocover

    idx = _ensure_vector_index(kb_path)
    if idx is None or idx.is_empty():
        # Vector index failed — fall back to keyword with a log hint
        import logging
        logging.getLogger("kb.retrievers").warning(
            "Vector index unavailable for '%s'; falling back to keyword retrieval",
            kb_path,
        )
        if read_kb_files:
            return read_kb_files(kb_path, query=query, top_n=top_n)
        return []

    ranked_names = idx.query(query, top_n=top_n)  # [(name, score), ...]
    if not ranked_names:
        return []

    # Pull the actual content for the ranked names (keyword engine reads files)
    all_docs = read_kb_files(kb_path, top_n=999) if read_kb_files else []
    name_set = {name for name, _ in ranked_names}
    ranked_list: list[tuple[str, str]] = []
    for display_name, content in all_docs:
        if display_name in name_set:
            ranked_list.append((display_name, content))
        if len(ranked_list) >= top_n:
            break

    return ranked_list


# ───────────────────────────── Public API ──────────────────────────────

def retrieve_kb_documents(
    query: str,
    kb_path: str | pathlib.Path,
    *,
    strategy: str = DEFAULT_METHOD,
    top_n: int = 5,
) -> list[tuple[str, str]]:
    """Retrieve relevant KB documents for *query* using the selected strategy.

    Parameters
    ----------
    query : The user's question / prompt used for retrieval.
    kb_path : Path to the knowledge-base root directory.
    strategy : ``"keyword"`` (default) or ``"vector"``.
    top_n : Number of documents to return.

    Returns
    -------
    list of ``(display_name, content)`` tuples in relevance order.
    """
    method = strategy.lower() if strategy else DEFAULT_METHOD

    if method == "keyword":
        return _retrieve_keyword(query, kb_path, top_n)

    if method == "vector":
        return _retrieve_vector(query, kb_path, top_n)

    # Unknown strategy — fall back to keyword with a warning
    import logging
    logging.getLogger("kb.retrievers").warning(
        "Unknown retrieval strategy '%s'; falling back to keyword", method
    )
    return _retrieve_keyword(query, kb_path, top_n)


def get_available_strategies() -> list[str]:
    """Return the list of available retrieval strategies."""
    # Check if vector is available without actually building the index
    has_fastembed = False
    try:
        from fastembed import TextEmbedding  # type: ignore[import]
        has_fastembed = TextEmbedding is not None  # type: ignore[attr-defined]
    except ImportError:
        pass
    strategies = ["keyword"]
    if has_fastembed:
        strategies.append("vector")
    return strategies
