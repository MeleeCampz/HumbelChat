"""Unified KB document retriever — keyword or vector similarity strategies.

Provides a single entry point ``retrieve_kb_documents()`` that bot_core.py
calls for RAG context injection.  The active strategy is controlled by the
environment variable **RAG_RETRIEVAL_METHOD** (default: ``keyword``).

Available strategies
--------------------
keyword — heuristic scoring of filenames, headers, and body overlap
          (existing engine in kb.reader).

vector  — cosine-similarity embedding search via the OpenWebUI backend
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
from typing import Optional

logger = logging.getLogger("kb.retrievers")

# ───────────────────────────── Constants ──────────────────────────────

KB_STRATEGIES = frozenset({"keyword", "vector"})
DEFAULT_METHOD = os.getenv("RAG_RETRIEVAL_METHOD", "keyword").lower()

# Singleton index store — lazily initialized
_index_store: Optional["kb.index.KBIndexStore"] = None
_kb_path_for_store: str | pathlib.Path | None = None


async def _ensure_index_store(kb_path: str | pathlib.Path) -> Optional["kb.index.KBIndexStore"]:
    """Create or return the cached index store, loading/building the index."""
    global _index_store, _kb_path_for_store

    if _index_store is not None:
        return _index_store  # already built

    if _kb_path_for_store == kb_path and os.path.exists(str(kb_path)):
        return _index_store  # same path, reuse

    from kb.index import KBIndexStore

    store = KBIndexStore(kb_path)
    await store.load()
    _index_store = store
    _kb_path_for_store = kb_path
    logger.info("Vector index store ready (%d chunks)", store.get_index().count() if store.get_index() else 0)
    return _index_store


# ───────────────────────────── Strategies ─────────────────────────────

def _retrieve_keyword(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
) -> list[tuple[str, str]]:
    """Keyword-based retrieval using the existing heuristic engine."""
    from kb.reader import read_kb_files  # type: ignore[import]

    return read_kb_files(kb_path, query=query, top_n=top_n)


async def _retrieve_vector(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
) -> list[tuple[str, str]]:
    """Vector-similarity retrieval using the persist-backed index.

    Falls back to keyword if the vector index is unavailable.
    """
    try:
        from kb.reader import read_kb_files  # type: ignore[import]
    except ImportError:
        read_kb_files = None  # pragma: nocover

    store = await _ensure_index_store(kb_path)
    if store is None:
        logger.warning("Vector index store unavailable for '%s'; falling back to keyword", kb_path)
        return read_kb_files(kb_path, query=query, top_n=top_n) if read_kb_files else []

    idx = store.get_index()
    if idx is None or idx.is_empty():
        logger.warning("Vector index is empty for '%s'; falling back to keyword", kb_path)
        return read_kb_files(kb_path, query=query, top_n=top_n) if read_kb_files else []

    ranked_names = await idx.query(query, top_n=top_n)
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

    Notes
    -----
    When *strategy* is ``"vector"``, this function builds the vector index
    on first use (chunking documents, encoding via OpenWebUI).  On subsequent
    calls the cached SQLite index is loaded directly for fast access.
    If embedding fails at any point, retrieval falls back to keyword search.
    """
    method = strategy.lower() if strategy else DEFAULT_METHOD

    if method == "keyword":
        return _retrieve_keyword(query, kb_path, top_n)

    if method == "vector":
        import nest_asyncio  # type: ignore[import]
        nest_asyncio.apply()
        return asyncio.run(_retrieve_vector(query, kb_path, top_n))  # type: ignore[arg-type]

    # Unknown strategy — fall back to keyword with a warning
    logger.warning(
        "Unknown retrieval strategy '%s'; falling back to keyword", method
    )
    return _retrieve_keyword(query, kb_path, top_n)


async def update_kb_document(file_path: str | pathlib.Path) -> bool:
    """Re-index or add a single KB document. Use after ``!add_kb_file``."""
    global _index_store
    if _index_store is None or _index_store.get_index() is None:
        logger.warning("No index to update; full rebuild needed")
        return False
    return await _index_store.update_single_document(file_path)


async def remove_kb_document(file_path: str | pathlib.Path) -> bool:
    """Remove a KB document from the vector index."""
    global _index_store
    if _index_store is None or _index_store.get_index() is None:
        return False
    return await _index_store.remove_document(file_path)


async def shutdown_vector_store() -> None:
    """Persist index before bot shutdown."""
    global _index_store
    if _index_store is not None:
        await _index_store.shutdown()
        logger.info("Vector index store shut down and persisted")


def get_available_strategies() -> list[str]:
    """Return the list of available retrieval strategies."""
    from config.settings import settings  # type: ignore[attr-defined]

    has_vector = bool(settings.INFER_URL and settings.INFER_API_KEY)
    strategies = ["keyword"]
    if has_vector:
        strategies.append("vector")
    return strategies


def is_vector_available() -> bool:
    """Quick check whether the vector retrieval backend is configured."""
    from config.settings import settings  # type: ignore[attr-defined]

    return bool(settings.INFER_URL and settings.INFER_API_KEY)
