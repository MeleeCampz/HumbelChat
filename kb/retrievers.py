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
import re
import time
import logging
import os
import pathlib
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from kb.index import KBIndexStore
    from kb.lexical import BM25
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
        idx = store.get_index()
        logger.info("Vector index store ready (%d chunks)", idx.count() if idx else 0)
    return _index_store


# ─────────────────────── Singleton store management ───────────────────────

def replace_index_store(
    store: "KBIndexStore | None",
    kb_path: str | pathlib.Path | None = None,
) -> None:
    """Atomically swap in a freshly built index store (P0 #1).

    ``/reindex_kb`` rebuilds the index in a *separate* ``KBIndexStore``; without
    this swap the module singleton that the live RAG path uses keeps serving
    the **old** in-memory index until the bot restarts.  After the swap, the
    next ``retrieve_kb_documents`` / ``update_kb_document`` call immediately
    uses the rebuilt index.

    The previous store is shut down in the background (best effort) so its
    resources are released without blocking the swap.  Pass ``kb_path`` to
    remember which path the store belongs to (used for log context).
    """
    global _index_store, _kb_path_for_store
    old_store = _index_store
    _index_store = store
    if kb_path is not None:
        _kb_path_for_store = kb_path
    new_idx = store.get_index() if store is not None else None
    old_idx = old_store.get_index() if old_store is not None else None
    logger.info(
        "Vector index store swapped in (%d chunks)%s",
        new_idx.count() if new_idx else 0,
        f" (was {old_idx.count()} chunks)" if old_idx else "",
    )
    if old_store is not None and old_store is not store:
        # Best-effort async cleanup of the replaced store.  It is no longer
        # referenced by any retrieval path, so shut it down out-of-band.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_safe_shutdown_store(old_store))
        except RuntimeError:
            # No running loop (e.g. called from a sync test) — nothing to do;
            # the store's state is already persisted to disk.
            pass


async def _safe_shutdown_store(store: "KBIndexStore") -> None:
    try:
        await store.shutdown()
    except Exception as exc:
        logger.warning("Shutdown of replaced index store failed: %s", exc)


async def reset_index_store() -> None:
    """Clear the singleton so the next retrieval call rebuilds it.

    Useful after the on-disk cache was invalidated out-of-band.  Unlike
    :func:`replace_index_store` this does NOT take a replacement store — the
    next ``_ensure_index_store`` call loads a fresh one from disk.
    """
    global _index_store, _kb_path_for_store
    old_store = _index_store
    _index_store = None
    _kb_path_for_store = None
    if old_store is not None:
        logger.info("Vector index store reset; will reload from disk on next use")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_safe_shutdown_store(old_store))
        except RuntimeError:
            pass


# ───────────────────────────── Helpers ──────────────────────────────

async def _keyword_fallback(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
    window_lines: int,
) -> list[tuple[str, str]]:
    """Fallback to keyword/TF-IDF retrieval when vector index is unavailable."""
    # P0 #4: the file scan + window extraction is blocking IO — run it in a
    # worker thread so a large KB can't freeze the event loop mid-fallback.
    from kb.reader import read_kb_files_async, get_relevant_chunks_async

    scored = await read_kb_files_async(kb_path, query=query, top_n=top_n * 3, max_lines_per_file=300)
    doc_names = [name for name, _ in scored[:top_n]] if scored else []
    chunks = await get_relevant_chunks_async(kb_path, doc_names, query=query, window_lines=window_lines)
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
    attach_min_score: float = 0.0,
    dense_scores: dict[str, float] | None = None,
    gate_scores: dict[str, float] | None = None,
    selected_names: list[str] | None = None,
    embeddings_by_name: dict[str, list[float]] | None = None,
) -> list[tuple[str, str]]:
    """Collapse ranked index chunks into per-file document entries.

    ``ranked`` is ``(display_name, content, similarity)`` sorted by relevance
    descending, where *display_name* is ``"<file> [<section path>]"``.  Each
    file contributes at most *max_chunks_per_file* of its highest-ranked
    chunks (subject to a per-file character cap); the selected chunks are
    joined in rank order into one entry per file so downstream per-file
    budgets (``RAG_MAX_DOCS``) still apply.

    ``attach_min_score`` + a per-chunk score map implement an optional
    relevance gate. The score map is *gate_scores* when given (e.g. Qwen3
    rerank scores, #43), else *dense_scores* (original DENSE cosine). When
    ``attach_min_score > 0`` and the map contains a chunk's name, any ranked
    chunk whose score is below the floor is skipped before it claims a slot.
    The check runs BEFORE a file's bucket is opened, so a weak chunk can
    neither consume a distinct-file slot nor leave an empty one behind.
    Chunks with no entry in the map (pure lexical/BM25 hits) are always kept
    — exact-term matches are valuable even when the embedding was lukewarm.

    Works for any chunk shape — header-split sections as well as unstructured
    "Full Document" chunks (e.g. player session logs) — because it consumes
    the index content directly instead of re-deriving windows from disk.

    ``embeddings_by_name`` enables near-duplicate dedup (Tier 1 item 4): a
    candidate whose embedding is within cosine ``RAG_DEDUP_COSINE_TAU`` of an
    already-SELECTED chunk is skipped and reported in the representative's
    file entry as "(near-duplicates omitted: …)". Chunks without an embedding
    (lexical-only hits) are never deduped.
    """
    if not ranked:
        return []

    from config.settings import RAG_DEDUP_COSINE_TAU, RAG_DEDUP_ENABLED
    dedup = RAG_DEDUP_ENABLED and bool(embeddings_by_name)
    import numpy as np  # lazy: only paid when dedup is active

    sel_mat: np.ndarray | None = None     # (k, dim) matrix of selected embeddings
    sel_norms: np.ndarray | None = None   # (k,) L2 norms, parallel to sel_mat
    sel_names: list[str] = []             # parallel to sel_mat rows
    skipped_by_rep: dict[str, list[str]] = {}

    per_file: dict[str, list[str]] = {}
    seen_names: set[str] = set()
    total_chunks = 0
    for name, content, _score in ranked:
        if total_chunks >= top_n * max_chunks_per_file:
            break
        stem = name.split(" [")[0] if " [" in name else name
        # Per-chunk relevance floor (opt-in): skip weak matches BEFORE opening
        # a file bucket, so they can't waste a distinct-file slot. Chunks
        # absent from the score map (lexical-only hits) always pass.
        if attach_min_score > 0:
            score_map = gate_scores if gate_scores is not None else dense_scores
            if score_map is not None:
                s = score_map.get(name)
                if s is not None and s < attach_min_score:
                    continue
        bucket = per_file.get(stem)
        if bucket is None:
            # New file — only open it while distinct-file budget remains.
            if len(per_file) >= top_n:
                continue
            bucket = per_file[stem] = []
        if len(bucket) >= max_chunks_per_file:
            continue
        if sum(len(c) for c in bucket) + len(content) > max_chars_per_file and bucket:
            continue  # file budget exhausted — try the next ranked chunk
        if name in seen_names:
            continue
        # Near-duplicate dedup: skip if this chunk is ~the same text as one
        # already selected (sibling entries crowding out distinct content).
        emb = (
            embeddings_by_name.get(name)
            if dedup and embeddings_by_name is not None else None
        )
        v: np.ndarray | None = None
        n = 0.0
        if emb is not None:
            v = np.asarray(emb, dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 1e-9 and sel_mat is not None and sel_norms is not None:
                sims = (sel_mat @ v) / (sel_norms * n)
                best_i = int(np.argmax(sims))
                if float(sims[best_i]) > RAG_DEDUP_COSINE_TAU:
                    skipped_by_rep.setdefault(sel_names[best_i], []).append(name)
                    continue
        seen_names.add(name)
        bucket.append(content)
        if selected_names is not None:
            selected_names.append(name)
        total_chunks += 1
        if v is not None and n > 1e-9:
            sel_mat = v if sel_mat is None else np.vstack([sel_mat, v[None, :]])
            sel_norms = (np.array([n]) if sel_norms is None
                         else np.append(sel_norms, n))
            sel_names.append(name)

    # Drop any file that ended up with no chunks (defensive: shouldn't happen
    # given the floor-first ordering, but never attach an empty entry).
    docs = [(stem, "\n\n".join(chunks)) for stem, chunks in per_file.items() if chunks]

    # Dedup notes: tell the model which near-duplicate entries were trimmed so
    # it doesn't present a partial list as complete. Max 3 names per note.
    if skipped_by_rep:
        def _short(nm: str) -> str:
            return nm.rsplit(" [", 1)[-1] if " [" in nm else nm.split("/")[-1]

        notes_by_stem: dict[str, list[str]] = {}
        for rep, skipped in skipped_by_rep.items():
            stem = rep.split(" [", 1)[0] if " [" in rep else rep
            notes_by_stem.setdefault(stem, []).extend(_short(s) for s in skipped)
        out: list[tuple[str, str]] = []
        for stem, text in docs:
            names = notes_by_stem.get(stem)
            if names:
                shown = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
                word = "entry" if len(names) == 1 else "entries"
                text += f"\n\n(Near-duplicate {word} omitted from attachment: {shown})"
            out.append((stem, text))
        n_skipped = sum(len(v) for v in skipped_by_rep.values())
        logger.info(
            "Dedup (cosine > %.2f): %d near-duplicate chunk(s) omitted from attachment",
            RAG_DEDUP_COSINE_TAU, n_skipped,
        )
        return out
    return docs


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

# ── Non-English (German) query detection ───────────────────────────────
# The KB is written in English; German game queries only score ~0.45-0.58
# against it even on direct hits, so the RAG_REWRITE_MIN_SCORE threshold
# (calibrated for confident matches) never fires and the LLM rewriter — the
# one component that can bridge the language gap by translating key terms
# into English D&D 5e vocabulary — stays dormant. Detect German queries
# explicitly and always route them through the expansion path.
_UMLAUT_RE = re.compile(r"[äöüßÄÖÜ]")
_GERMAN_STOPWORDS = frozenset({
    "der", "die", "das", "und", "ist", "sind", "war", "waren", "von", "dem",
    "den", "mit", "für", "wie", "was", "wer", "wenn", "dass", "nicht", "oder",
    "gib", "gibt", "zeig", "zeige", "erkläre", "erklärt", "kann", "können",
    "muss", "müssen", "willst", "will", "kostet", "kosten", "welche",
    "welcher", "welches", "vielen", "viel", "mehr", "beim", "über", "unter",
    "nach", "auch", "sich", "einen", "eine", "einem", "einer", "mein",
    "meine", "dein", "ihre", "seine", "unsere", "stufe", "würfel",
})


def _looks_german(query: str) -> bool:
    """Cheap heuristic: umlauts or ≥2 German stopwords ⇒ treat as German.

    Note (2026-09-27): with the default ``RAG_REWRITE_ALL_QUERIES=1`` every
    query is rewritten, so this heuristic only matters behind the
    ``RAG_REWRITE_ALL_QUERIES=0`` kill-switch — there it re-enables the old
    German/low-confidence-only rewrite trigger.
    """
    if _UMLAUT_RE.search(query):
        return True
    tokens = re.findall(r"[a-zäöüß]+", query.lower())
    hits = {t for t in tokens if t in _GERMAN_STOPWORDS}
    return len(hits) >= 2


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


async def _llm_expansions(query: str, rewrite_model: str = "") -> list[str]:
    """LLM query expansion ONLY (no embedding/ranking). Never raises.

    Returns the expansion strings (original excluded — its ranking is already
    known and must not be double-weighted), or ``[]`` when rewriting is disabled
    or fails.  Bounded by ``RAG_REWRITE_BUDGET_SECONDS``.  Split out of
    :func:`_expand_low_confidence_query` so the (independent) LLM call can run
    CONCURRENTLY with the dense query ranking — see ``_retrieve_vector``.
    """
    from config.settings import RAG_QUERY_REWRITER, RAG_REWRITE_BUDGET_SECONDS

    if not RAG_QUERY_REWRITER:
        return []
    try:
        from kb.query_rewriter import create_query_rewriter

        # Use the SAME model as the main completion call — on a single-model
        # local backend a different slug would either fail or evict the loaded
        # weights.  Empty string = rewriter falls back to DEFAULT_MODEL.
        rewriter = create_query_rewriter(model_slug=rewrite_model)

        # Load-state-aware budget (2026-10-16): when the backend reports the
        # chat model as NOT resident, this call will pay the weight-load cost
        # (~30-50 s on :8888) — which the main completion would pay anyway.
        # Extend the budget so the load happens DURING this concurrent call and
        # the expansions survive; when warm (or state unknown), keep the tight
        # budget so a genuinely stalled call fails fast.
        budget = RAG_REWRITE_BUDGET_SECONDS
        try:
            loaded = await rewriter.model_is_loaded()
        except Exception:  # noqa: BLE001 — the probe is best-effort; never break the rewrite path
            loaded = None
        if loaded is False:
            from config.settings import RAG_REWRITE_LOAD_BUDGET_SECONDS
            budget = max(budget, RAG_REWRITE_LOAD_BUDGET_SECONDS)
            logger.info(
                "Chat model not loaded — extending rewrite budget to %ds so the "
                "load happens during this call", budget)
        expanded = await asyncio.wait_for(rewriter.expand(query), timeout=budget)
        # expand() returns [original, *expansions] — drop the original.
        return [e for e in expanded[1:] if e and e.strip()]
    except Exception as exc:  # noqa: BLE001 — rewrite is a pure enhancement
        logger.warning("Query rewrite failed (%s); using original ranking only", exc)
        return []


async def _expand_low_confidence_query(
    idx: "KBVectorIndex",
    query: str,
    top_n: int,
    rewrite_model: str = "",
) -> tuple[list[list[tuple[str, str, float]]], float, list[str]]:
    """Generate expansion rankings for a low-confidence query.

    Runs the LLM rewriter (bounded by ``RAG_REWRITE_BUDGET_SECONDS``), embeds
    all expansions in one batched call via ``idx.rank_texts()``, and returns
    ``(expansion_rankings, elapsed_seconds, expansion_texts)`` — one ranked list
    per expansion plus the expansion strings themselves (the hybrid lexical leg
    scores them too, so BM25 benefits from the translated English terms).
    Returns ``([], 0.0, [])`` when rewriting is disabled or produces nothing.
    Never raises: callers fall back to the original ranking alone.
    """
    t0 = time.monotonic()
    extra = await _llm_expansions(query, rewrite_model=rewrite_model)
    if not extra:
        return [], 0.0, []

    try:
        from config.settings import RAG_DENSE_TOP_K
        rankings = await idx.rank_texts(extra, top_n=RAG_DENSE_TOP_K)
    except Exception as exc:  # noqa: BLE001 — the lexical leg is a pure enhancement
        logger.warning("Expansion embedding failed (%s); using original ranking only", exc)
        return [], time.monotonic() - t0, []

    return [r for r in rankings if r], time.monotonic() - t0, extra


# ───────────────────────────── Strategies ─────────────────────────────

async def _retrieve_keyword(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
) -> list[tuple[str, str]]:
    """Keyword-based retrieval using the existing heuristic engine.

    Phase 1: Score all files against *query* terms using a lightweight scan
             (first 300 lines is sufficient for relevance ranking).
    Phase 2: Extract only relevant line-windows from the top-N documents
             via get_relevant_chunks(), avoiding full-file dump in context.

    P0 #4: both phases are blocking filesystem IO, so the whole pass runs
    off the event loop (see ``read_kb_files_async`` / ``get_relevant_chunks_async``).
    """
    from kb.reader import read_kb_files_async, get_relevant_chunks_async
    from config.settings import RAG_WINDOW_LINES

    # Phase 1 — quick scoring pass (300 lines is plenty for keyword overlap)
    scored = await read_kb_files_async(kb_path, query=query, top_n=top_n * 3, max_lines_per_file=300)
    if not scored:
        return []

    # Phase 2 — extract only matched windows from top documents
    doc_names = [name for name, _ in scored[:top_n]]
    chunks = await get_relevant_chunks_async(kb_path, doc_names, query=query, window_lines=RAG_WINDOW_LINES)

    logger.info(
        "Keyword retrieval: %d files ranked → %d relevant chunk(s) with ~%.0f chars",
        len(scored), len(chunks),
        sum(len(c) for _, c in chunks) if chunks else 0,
    )
    return chunks


# ── Hybrid lexical (BM25) leg ─────────────────────────────────────────────
# A BM25 index over every chunk's content, cached by identity of ``idx._docs`` so it is
# built once and reused across queries (rebuilt automatically when the doc list object is
# replaced). Provides an exact-term signal to fuse with dense vectors via RRF.
_bm25_cache: tuple["BM25", Any, int] | None = None  # (BM25, docs_ref, count)


def _get_bm25(idx: "KBVectorIndex") -> "BM25":
    """Return a cached BM25 index for *idx*, building it on first use."""
    global _bm25_cache
    docs = idx._docs
    if _bm25_cache is not None and _bm25_cache[1] is docs and _bm25_cache[2] == len(docs):
        return _bm25_cache[0]
    from kb.lexical import BM25
    bm25 = BM25([d.content for d in docs])
    _bm25_cache = (bm25, docs, len(docs))
    return bm25


def _lexical_ranking(idx: "KBVectorIndex", query: str, limit: int) -> list[tuple[str, str, float]]:
    """BM25 ranking of all chunks for *query* as ``[(name, content, score), ...]``.

    Synchronous (CPU) — call via ``asyncio.to_thread``.
    """
    bm25 = _get_bm25(idx)
    scores = bm25.scores(query)
    docs = idx._docs
    ranked = [
        (docs[i].retrieval_name(), docs[i].content, float(s))
        for i, s in enumerate(scores) if s > 0
    ]
    ranked.sort(key=lambda t: -t[2])
    return ranked[:limit]


async def _retrieve_vector(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
    window_lines: int = 80,
    rewrite_model: str = "",
) -> list[tuple[str, str]]:
    """Vector-based retrieval using the persist-backed index.

    Pure cosine-similarity ranking over the in-memory index (which the
    disk-backed store keeps fully hydrated at load time).  Falls back to
    keyword-only if the vector index is unavailable or the query returns
    no hits.  By default every query gets a bounded LLM rewrite/expansion pass
    (``RAG_REWRITE_ALL_QUERIES``, thinking disabled on the call) whose ranked
    lists are merged via RRF; with the flag off, only German or low-confidence
    (top score below ``RAG_REWRITE_MIN_SCORE``) queries do (see
    ``_expand_low_confidence_query``).
    """
    store = await _ensure_index_store(kb_path)
    idx = store.get_index() if store is not None else None
    if idx is None or idx.is_empty():
        logger.warning("Vector index unavailable or empty for '%s'; falling back to keyword", kb_path)
        return await _keyword_fallback(query, kb_path, top_n, window_lines)

    from config.settings import (
        RAG_DENSE_TOP_K,
        RAG_QUERY_REWRITER,
        RAG_REWRITE_ALL_QUERIES,
        RAG_REWRITE_MIN_SCORE,
        RAG_MIN_ATTACH_SCORE,
    )

    # ── Parallel rewrite (latency) ────────────────────────────────────────
    # With RAG_REWRITE_ALL_QUERIES the LLM expansion is unconditional and
    # depends on NOTHING from the dense ranking — so start it as a task BEFORE
    # the query embedding and let the ~0.5–2 s rewrite call overlap with the
    # embed+rank instead of serializing behind it.
    _rewrite_task: asyncio.Task | None = None
    if RAG_QUERY_REWRITER and RAG_REWRITE_ALL_QUERIES:
        _rewrite_task = asyncio.create_task(_llm_expansions(query, rewrite_model=rewrite_model))

    # One embedding call: rank the in-memory chunks AND keep the query vector.
    # Pool size is tunable (RAG_DENSE_TOP_K) — a larger candidate set improves
    # recall for hard lookups at negligible cost while the reranker is off.
    ranked, _q_emb = await idx.query_with_embeddings(query, top_n=RAG_DENSE_TOP_K)
    if not ranked:
        if _rewrite_task is not None:
            _rewrite_task.cancel()  # keyword fallback needs no expansions
        logger.warning("Vector query returned no hits for '%s'; falling back to keyword", kb_path)
        return await _keyword_fallback(query, kb_path, top_n, window_lines)

    # Snapshot the original DENSE cosine score per chunk NOW. After hybrid RRF
    # fusion / rerank the scores become rank-based (~0.03), so the attach-time
    # relevance floor (RAG_ATTACH_FLOOR) must gate on these raw cosine values.
    dense_scores: dict[str, float] = {name: float(s) for name, _c, s in ranked}

    # Log the score distribution so RAG_REWRITE_MIN_SCORE can be tuned from
    # real traffic (option A: rewrite only when the index is "guessing").
    top_scores = [s for _, _, s in ranked[:8]]
    logger.info(
        "Vector scores for %r: top=%.3f median=%.3f min=%.3f (%d chunks)",
        query, top_scores[0], top_scores[len(top_scores) // 2], top_scores[-1], len(ranked),
    )

    # ── Min-attachment relevance floor (opt-in) ───────────────────────
    # When RAG_MIN_ATTACH_SCORE > 0, drop dense chunks scoring below the floor
    # BEFORE hybrid fusion so low-relevance dense hits don't crowd out good BM25
    # matches. Default 0 = off (keep all). Only the dense leg is filtered here,
    # so lexical-only matches still survive via the hybrid step below.
    if RAG_MIN_ATTACH_SCORE > 0:
        kept = [t for t in ranked if t[2] >= RAG_MIN_ATTACH_SCORE]
        dropped = len(ranked) - len(kept)
        if dropped:
            logger.info(
                "Min-attach floor %.2f: dropped %d/%d dense chunk(s) below threshold",
                RAG_MIN_ATTACH_SCORE, dropped, len(ranked),
            )
        ranked = kept

    expansion_texts: list[str] = []
    low_confidence = bool(ranked) and ranked[0][2] < RAG_REWRITE_MIN_SCORE
    german_query = _looks_german(query)
    # Default: expand EVERY query (RAG_REWRITE_ALL_QUERIES=1) — with thinking off
    # the rewrite call costs ~0.5-1s and fixes both German queries and mangled
    # English alike. Fallback mode: only German or low-confidence queries.
    if RAG_REWRITE_ALL_QUERIES or low_confidence or german_query:
        if _rewrite_task is not None:
            # All-queries path: the LLM call already ran in parallel with the
            # dense ranking above — just await it, then embed+rank the results.
            t0 = time.monotonic()
            expansion_texts = await _rewrite_task
            expansion_rankings: list[list[tuple[str, str, float]]] = []
            if expansion_texts:
                try:
                    all_ranked = await idx.rank_texts(
                        expansion_texts, top_n=RAG_DENSE_TOP_K
                    )
                    expansion_rankings = [r for r in all_ranked if r]
                except Exception as exc:  # noqa: BLE001 — pure enhancement
                    logger.warning("Expansion embedding failed (%s); using original ranking only", exc)
            elapsed = time.monotonic() - t0
        else:
            expansion_rankings, elapsed, expansion_texts = await _expand_low_confidence_query(
                idx, query, top_n, rewrite_model=rewrite_model
            )
        if expansion_rankings:
            merged = reciprocal_rank_fusion([ranked] + expansion_rankings)
            reason = (
                "all-queries"
                if RAG_REWRITE_ALL_QUERIES
                else ("german" if german_query else f"low-confidence < {RAG_REWRITE_MIN_SCORE:.2f}")
            )
            logger.info(
                "Query rewrite triggered (%s, top=%.3f): added %d expansion list(s) in %.1fs; RRF-merged to %d chunks",
                reason, ranked[0][2] if ranked else 0.0, len(expansion_rankings), elapsed, len(merged),
            )
            ranked = merged

    # ── Hybrid lexical (BM25) leg, fused with dense via RRF ───────────────
    # Dense vectors capture semantics; BM25 captures exact terms (names, stats,
    # spell names). Merging both with RRF improves recall when phrasing differs.
    # Toggleable via RAG_HYBRID_ENABLED (default on) so dense-only behaviour is
    # still available and the rewrite path can be tested in isolation.
    #
    # IMPORTANT: score EVERY query variant (original + expansions), not just the
    # raw query. BM25 on the raw German query alone matches "stat block" monster
    # text and German session notes verbatim, which drowned out the expansion-
    # fused dense ranking (see docs/internal/RAG_ANALYSIS_2026-09-27.md). The translated
    # English expansions let BM25 exact-match KB terms like "Chain Mail".
    hybrid_ranked = ranked
    from config.settings import RAG_HYBRID_ENABLED
    if RAG_HYBRID_ENABLED:
        from config.settings import RAG_LEXICAL_TOP_K
        candidate_limit = RAG_LEXICAL_TOP_K
        try:
            lex_lists: list[list[tuple[str, str, float]]] = []
            for variant in [query, *expansion_texts]:
                lr = await asyncio.to_thread(_lexical_ranking, idx, variant, candidate_limit)
                if lr:
                    lex_lists.append(lr)
            if lex_lists:
                lex_merged = (
                    reciprocal_rank_fusion(lex_lists) if len(lex_lists) > 1 else lex_lists[0]
                )
                hybrid_ranked = reciprocal_rank_fusion([ranked, lex_merged])
                logger.info(
                    "Hybrid retrieval: dense(%d) + lexical(%d list(s), %d chunk(s)) → RRF-merged %d chunk(s)",
                    len(ranked), len(lex_lists), len(lex_merged), len(hybrid_ranked),
                )
        except Exception as exc:  # noqa: BLE001 — the lexical leg is a pure enhancement
            logger.warning("Hybrid lexical leg failed (%s); using dense ranking only", exc)

    # ── Cross-encoder rerank of the top candidates (quality boost) ─────────
    from kb.reranker import rerank, is_available as _rerank_available
    final_ranked = hybrid_ranked
    rerank_gate: dict[str, float] | None = None  # #43: set when rerank runs
    if _rerank_available():
        from config.settings import RERANK_TOP_K, RERANK_USE_EXPANSION
        cands = hybrid_ranked[:RERANK_TOP_K]  # (name, content, score)
        contents = [c for _, c, _ in cands]
        name_map: dict[str, str] = {}
        for name, content, _ in cands:
            name_map.setdefault(content, name)
        # Score against the original query AND — when expansions exist and
        # RERANK_USE_EXPANSION is on — the first expansion, keeping the MAX per
        # chunk. Cross-encoders score cross-lingual pairs (German question vs
        # English KB text) near-noise, which DEMOTES the right chunk; the
        # rewriter's English expansion rescues it (TQ case: rank 25 → 1).
        # A failed pass returns all-0.0 scores, so max-fusion degrades safely:
        # one good pass dominates, both failing keeps the original order.
        rerank_queries = [query]
        if RERANK_USE_EXPANSION and expansion_texts:
            rerank_queries.append(expansion_texts[0])
        merged_scores: dict[str, float] = {}
        for rq in rerank_queries:
            scored = await rerank(rq, contents)  # [(content, score), ...]; original order on fallback
            for c, s in scored:
                if s > merged_scores.get(c, 0.0):
                    merged_scores[c] = s
        final_ranked = sorted(
            ((name_map.get(c, ""), c, merged_scores.get(c, 0.0)) for c in contents),
            key=lambda t: t[2],
            reverse=True,  # stable: all-0 (both passes failed) keeps candidate order
        )
        logger.info(
            "Reranked top %d candidate(s) against %d quer%s", len(final_ranked), len(rerank_queries), "y" if len(rerank_queries) == 1 else "ies",
        )
        # #43: per-chunk rerank scores for the attach floor (used only when
        # RAG_ATTACH_FLOOR_SOURCE=rerank). If BOTH rerank passes failed every
        # score is 0.0 — treat that as "no gate" so the floor can't drop all.
        rerank_gate = {
            name_map[c]: s for c, s in merged_scores.items()
            if name_map.get(c) and s > 0.0
        } or None

    from config.settings import (
        RAG_MAX_CHUNKS_PER_FILE,
        RAG_ATTACH_FLOOR,
        RAG_ATTACH_FLOOR_SOURCE,
        RAG_DEDUP_ENABLED,
    )
    # #43: gate the attach floor on Qwen rerank scores when opted in; the
    # dense-source default keeps the historical behavior untouched.
    gate = rerank_gate if RAG_ATTACH_FLOOR_SOURCE == "rerank" else None
    # Tier 1 item 4: near-duplicate dedup over the selection (see
    # select_ranked_chunks). The map is built per call (~ms for 5K chunks).
    emb_map = idx.embedding_map() if RAG_DEDUP_ENABLED else None
    selected: list[str] = []
    docs = select_ranked_chunks(
        final_ranked, top_n=top_n,
        max_chunks_per_file=RAG_MAX_CHUNKS_PER_FILE,
        attach_min_score=RAG_ATTACH_FLOOR, dense_scores=dense_scores,
        gate_scores=gate, selected_names=selected,
        embeddings_by_name=emb_map,
    )
    if not docs and final_ranked:
        # Safety net: an over-aggressive floor must never yield an empty context.
        docs = select_ranked_chunks(
            final_ranked, top_n=top_n, max_chunks_per_file=RAG_MAX_CHUNKS_PER_FILE,
        )
        if docs:
            logger.info(
                "Attach floor %.2f dropped every chunk; fell back to unfiltered selection",
                RAG_ATTACH_FLOOR,
            )

    logger.info(
        "Vector retrieval: %d ranked chunk(s) → %d file(s) with ~%.0f chars",
        len(final_ranked), len(docs),
        sum(len(c) for _, c in docs) if docs else 0,
    )
    # Chunk-level visibility: which sections actually got attached (the
    # per-file cap can silently drop a right chunk that ranked just below
    # its file's bucket — the TQ case, 2026-10-15).
    for name in selected:
        logger.info("  attached chunk: %s", name)
    return docs


# ───────────────────────────── Public API ──────────────────────────────

async def retrieve_kb_documents(
    query: str,
    kb_path: str | pathlib.Path,
    *,
    strategy: str = DEFAULT_METHOD,
    top_n: int = 5,
    window_lines: int = 80,
    rewrite_model: str = "",
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
        return await _retrieve_keyword(query, kb_path, top_n)

    if method == "vector":
        return await _retrieve_vector(
            query, kb_path, top_n,
            window_lines=window_lines, rewrite_model=rewrite_model,
        )

    # Unknown strategy — fall back to keyword with a warning
    logger.warning(
        "Unknown retrieval strategy '%s'; falling back to keyword", method
    )
    return await _retrieve_keyword(query, kb_path, top_n)


async def update_kb_document(file_path: str | pathlib.Path) -> bool:
    """Re-index or add a single KB document. Use after ``!add_kb_file``.

    P0 #3: concurrent first-time calls used to each build a store and race on
    the same SQLite temp-file swap — the init is now serialized behind
    ``_index_init_lock``.
    """
    global _index_store, _kb_path_for_store
    if _index_store is not None and _index_store.get_index() is not None:
        return await _index_store.update_single_document(file_path)

    async with _get_index_init_lock():
        # Re-check inside the lock — a concurrent caller may have finished init.
        if _index_store is not None and _index_store.get_index() is not None:
            return await _index_store.update_single_document(file_path)

        # No store yet — initialize one (loads the disk cache) and update through it.
        logger.warning("No index loaded; initializing store for single-document update")
        from kb.index import KBIndexStore
        from config.settings import KB_PATH
        store = KBIndexStore(KB_PATH)
        await store.load()
        _index_store = store
        _kb_path_for_store = KB_PATH
        # A cold cache is empty until this file is indexed. Returning here
        # used to skip the first upload after startup.
        return await store.update_single_document(file_path)


async def sync_kb_store() -> tuple[Optional["KBVectorIndex"], dict[str, Any]]:
    """Sync the (singleton) index with files changed on disk outside the bot.

    Re-embeds only new/renamed/changed files and drops rows for files that
    were deleted externally.  Goes through the module-level singleton store so
    the live RAG path sees the updated index immediately.  Returns
    ``(index, report)`` — see ``KBIndexStore.sync_changes`` for the report
    shape.
    """
    global _index_store, _kb_path_for_store

    async with _get_index_init_lock():
        if _index_store is None:
            from config.settings import KB_PATH
            from kb.index import KBIndexStore
            _index_store = KBIndexStore(KB_PATH)
            await _index_store.load()
            _kb_path_for_store = KB_PATH

    if _index_store is None:
        return None, {"ok": False, "failed": [], "error": "index store unavailable"}
    return await _index_store.sync_changes()


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
