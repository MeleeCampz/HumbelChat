"""CPU cross-encoder reranker for RAG (quality boost).

Reranks the top-K chunks retrieved by vector search against the user's query using a
cross-encoder (default ``BAAI/bge-reranker-v2-m3``) that scores each ``(query, chunk)``
pair directly. This is a *separate* model from the embedding model — it does not produce
index vectors, so it never conflicts with the embedding space and runs in-process on CPU.

Design goals
------------
* **Never breaks retrieval.** Any failure (missing dependency, model load error, timeout,
  malformed output) falls back to the caller's original ordering, so a reranker problem can
  never make an /ai request fail or return no context.
* **Loaded once, kept resident.** The cross-encoder is a process-wide lazy singleton warmed
  at startup (see :func:`preload_reranker`) so it isn't reloaded per request.

Usage
-----
    from kb.reranker import rerank
    scored = await rerank(query, [chunk1, chunk2, ...], top_n=5)
    # -> [(chunk, score), ...] sorted by rerank score descending (original order if disabled/fail)
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import threading
from typing import Any

logger = logging.getLogger("kb.reranker")

_model: Any = None
_model_lock: threading.Lock | None = None


def _get_model() -> Any:
    """Return the shared cross-encoder, loading it on first use (blocking)."""
    global _model, _model_lock
    if _model is not None:
        return _model
    if _model_lock is None:
        _model_lock = threading.Lock()
    with _model_lock:
        if _model is None:
            from sentence_transformers import CrossEncoder
            from config.settings import RERANK_MODEL
            logger.info("Loading in-process reranker model '%s' (CPU)", RERANK_MODEL)
            # max_length bounds per-pair compute; 512 covers most KB chunks.
            _model = CrossEncoder(RERANK_MODEL, max_length=512)
    return _model


def preload_reranker() -> None:
    """Warm the reranker at startup (no-op unless enabled + available). Never raises."""
    if not is_available():
        return
    try:
        _get_model()
        logger.info("Preloaded reranker model")
    except Exception as exc:  # noqa: BLE001 — startup must not crash on this
        logger.warning("Failed to preload reranker: %s", exc)


def is_available() -> bool:
    """True when reranking is enabled AND sentence-transformers is importable."""
    from config.settings import RERANK_ENABLED
    if not RERANK_ENABLED:
        return False
    return importlib.util.find_spec("sentence_transformers") is not None


def reset_model() -> None:
    """Test helper: drop the cross-encoder singleton between event loops."""
    global _model
    _model = None


def _score_sync(query: str, chunks: list[str]) -> list[float]:
    """Score ``(query, chunk)`` pairs. Synchronous — call via ``asyncio.to_thread``.

    Cross-encoders emit a relevance logit per pair; higher is more relevant. We return
    raw scores (no sigmoid needed) since only their *ordering* is used.
    """
    model = _get_model()
    pairs = [(query, c) for c in chunks]
    scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
    return [float(s) for s in scores]


async def rerank(
    query: str,
    chunks: list[str],
    *,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Rerank *chunks* against *query*.

    Returns ``[(chunk, score), ...]`` sorted by rerank score descending (optionally
    truncated to *top_n*). When reranking is disabled or unavailable — or on any error /
    timeout — returns the chunks in their **original order** with a 0.0 score, so callers
    can treat the result as an always-usable ranked list.
    """
    if not chunks:
        return []
    if not is_available():
        logger.debug("Rerank disabled/unavailable — returning original order")
        return [(c, 0.0) for c in chunks]

    from config.settings import RERANK_TIMEOUT_SECONDS
    try:
        scores = await asyncio.wait_for(
            asyncio.to_thread(_score_sync, query, chunks),
            timeout=RERANK_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Rerank timed out after %ss — using original order", RERANK_TIMEOUT_SECONDS)
        return [(c, 0.0) for c in chunks]
    except Exception as exc:  # noqa: BLE001 — any failure degrades to original order
        logger.warning("Rerank failed (%s); using original order", exc)
        return [(c, 0.0) for c in chunks]

    if len(scores) != len(chunks):  # defensive: malformed output → don't trust it
        logger.warning(
            "Rerank returned %d scores for %d chunks — using original order",
            len(scores), len(chunks),
        )
        return [(c, 0.0) for c in chunks]

    scored = sorted(zip(chunks, scores), key=lambda t: t[1], reverse=True)
    if top_n is not None and top_n > 0:
        scored = scored[:top_n]
    logger.debug("Reranked %d chunk(s); top score=%.3f", len(scored), scored[0][1] if scored else 0.0)
    return scored
