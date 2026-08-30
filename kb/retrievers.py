"""Unified KB document retriever — keyword or vector similarity strategies.

Provides a single entry point ``retrieve_kb_documents()`` that bot_core.py
calls for RAG context injection.  The active strategy is controlled by the
environment variable **RAG_RETRIEVAL_METHOD** (default: ``vector``).

Available strategies
--------------------
keyword — heuristic scoring of filenames, headers, and body overlap
          (existing engine in kb.reader).

vector  — cosine-similarity embedding search via the configured inference backend
          using model from ``EMBEDDING_MODEL`` env var (default: ``nomic-embed-text:latest``).  Documents are chunked
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
import time
import logging
import os
import pathlib
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kb.index import KBIndexStore
    from kb.vector_db import KBVectorIndex

logger = logging.getLogger("kb.retrievers")

# ───────────────────────────── Constants ──────────────────────────────

KB_STRATEGIES = frozenset({"keyword", "vector"})
DEFAULT_METHOD = os.getenv("RAG_RETRIEVAL_METHOD", "vector").lower()

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


# ────────────────── Ranked-chunk selection (vector path) ──────────────────

# Keep several semantically-matched chunks per file so a query can include
# both an overview section and the detail sections it references, while still
# capping how much any single file may dominate the prompt.
MAX_CHUNKS_PER_FILE = 5
# Per-file char cap so one large file cannot consume the whole RAG budget.
MAX_CHARS_PER_FILE = 24_000


def select_ranked_chunks(
    ranked: list[tuple[str, str, float]],
    top_n: int = 5,
    max_chunks_per_file: int = MAX_CHUNKS_PER_FILE,
    max_chars_per_file: int = MAX_CHARS_PER_FILE,
) -> list[tuple[str, str]]:
    """Collapse ranked index chunks into per-file document entries.

    ``ranked`` is ``(display_name, content, similarity)`` sorted by relevance
    descending, where *display_name* is ``"<file> [<section path>]"``.  Each
    file contributes at most *max_chunks_per_file* of its highest-ranked
    chunks (subject to a per-file character cap); the selected chunks are
    joined in rank order into one entry per file so downstream per-file
    budgets (``RAG_MAX_DOCS``) still apply.

    Works for any chunk shape — header-split sections as well as unstructured
    "Full Document" chunks (e.g. player session logs) — because it consumes
    the index content directly instead of re-deriving windows from disk.
    """
    if not ranked:
        return []

    per_file: dict[str, list[str]] = {}
    seen_names: set[str] = set()
    total_chunks = 0
    for name, content, _score in ranked:
        if total_chunks >= top_n * max_chunks_per_file:
            break
        stem = name.split(" [")[0] if " [" in name else name
        # Don't start new files once top_n distinct files are covered.
        if stem not in per_file and len(per_file) >= top_n:
            continue
        bucket = per_file.setdefault(stem, [])
        if len(bucket) >= max_chunks_per_file:
            continue
        if sum(len(c) for c in bucket) + len(content) > max_chars_per_file and bucket:
            continue  # file budget exhausted — try the next ranked chunk
        if name in seen_names:
            continue
        seen_names.add(name)
        bucket.append(content)
        total_chunks += 1

    return [(stem, "\n\n".join(chunks)) for stem, chunks in per_file.items()]


# ─────────── Low-confidence query rewriting (option A) ───────────
#
# Vector search alone can miss the right chunk when the player's phrasing
# differs from the KB's vocabulary.  Instead of paying for an LLM rewrite on
# *every* query, we only trigger it when the top cosine similarity score is
# below ``RAG_REWRITE_MIN_SCORE`` — i.e. the index itself says "I'm guessing".
# Confident queries keep their single-embedding-call latency.
#
# When triggered: the rewriter produces up to N alternative phrasings, all
# expansions are embedded in ONE batched call, and every query's ranking is
# merged with reciprocal rank fusion (RRF).  RRF needs no score calibration —
# it only uses each list's ordering, which makes the merge robust across
# differently-scored queries.

_RRF_K = 60  # standard RRF constant (Cormack et al. 2009)


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, str, float]]],
) -> list[tuple[str, str, float]]:
    """Merge several ranked chunk lists via reciprocal rank fusion.

    Each ranking is ``[(display_name, content, similarity), ...]`` sorted by
    relevance descending.  A chunk's fused score is the sum of
    ``1 / (RRF_K + rank)`` over every list it appears in; ties are broken by
    best (lowest) rank seen.

    Returns ``[(display_name, content, fused_score), ...]`` sorted by fused
    score descending.  Empty/missing lists are ignored; an empty input yields
    an empty result.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    contents: dict[str, str] = {}

    for ranking in rankings:
        for rank, (name, content, _sim) in enumerate(ranking, start=1):
            scores[name] = scores.get(name, 0.0) + 1.0 / (_RRF_K + rank)
            if name not in best_rank or rank < best_rank[name]:
                best_rank[name] = rank
            contents.setdefault(name, content)

    merged = sorted(scores, key=lambda n: (-scores[n], best_rank[n]))
    return [(name, contents[name], scores[name]) for name in merged]


async def _expand_low_confidence_query(
    idx: "KBVectorIndex",
    query: str,
    top_n: int,
) -> tuple[list[list[tuple[str, str, float]]], float]:
    """Generate expansion rankings for a low-confidence query.

    Runs the LLM rewriter (bounded by ``RAG_REWRITE_BUDGET_SECONDS``), embeds
    all expansions in one batched call via ``idx.rank_texts()``, and returns
    ``(expansion_rankings, elapsed_seconds)`` — one ranked list per expansion.
    Returns ``([], 0.0)`` when rewriting is disabled or produces nothing.
    Never raises: callers fall back to the original ranking alone.
    """
    from config.settings import (
        RAG_QUERY_MAX_EXPANSIONS,
        RAG_QUERY_REWRITER,
        RAG_REWRITE_BUDGET_SECONDS,
    )

    if not RAG_QUERY_REWRITER:
        return [], 0.0

    t0 = time.monotonic()
    try:
        from kb.query_rewriter import create_query_rewriter

        rewriter = create_query_rewriter(max_expansions=RAG_QUERY_MAX_EXPANSIONS)
        expanded = await asyncio.wait_for(
            rewriter.expand(query), timeout=RAG_REWRITE_BUDGET_SECONDS
        )
        # expand() returns [original, *expansions]; the original's ranking is
        # already known — drop it so its chunks are not double-weighted.
        extra = [e for e in expanded[1:] if e and e.strip()]
    except Exception as exc:
        logger.warning("Low-confidence rewrite failed (%s); using original ranking only", exc)
        return [], time.monotonic() - t0

    if not extra:
        return [], 0.0

    try:
        rankings = await idx.rank_texts(extra, top_n=min(top_n * 4, 32))
    except Exception as exc:
        logger.warning("Expansion embedding failed (%s); using original ranking only", exc)
        return [], time.monotonic() - t0

    return [r for r in rankings if r], time.monotonic() - t0


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
    no hits.  Confident queries cost exactly one embedding call; when the
    top score is below ``RAG_REWRITE_MIN_SCORE`` a bounded LLM rewrite adds
    expansion rankings merged via RRF (see ``_expand_low_confidence_query``).
    """
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

    # Log the score distribution so RAG_REWRITE_MIN_SCORE can be tuned from
    # real traffic (option A: rewrite only when the index is "guessing").
    top_scores = [s for _, _, s in ranked[:8]]
    logger.info(
        "Vector scores for %r: top=%.3f median=%.3f min=%.3f (%d chunks)",
        query, top_scores[0], top_scores[len(top_scores) // 2], top_scores[-1], len(ranked),
    )

    from config.settings import RAG_REWRITE_MIN_SCORE

    if ranked[0][2] < RAG_REWRITE_MIN_SCORE:
        expansion_rankings, elapsed = await _expand_low_confidence_query(idx, query, top_n)
        if expansion_rankings:
            merged = reciprocal_rank_fusion([ranked] + expansion_rankings)
            logger.info(
                "Low-confidence query (top=%.3f < %.2f): rewrite added %d expansion list(s) in %.1fs; RRF-merged to %d chunks",
                ranked[0][2], RAG_REWRITE_MIN_SCORE, len(expansion_rankings), elapsed, len(merged),
            )
            ranked = merged

    docs = select_ranked_chunks(ranked, top_n=top_n)

    logger.info(
        "Vector retrieval: %d ranked chunk(s) → %d file(s) with ~%.0f chars",
        len(ranked), len(docs),
        sum(len(c) for _, c in docs) if docs else 0,
    )
    return docs


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
