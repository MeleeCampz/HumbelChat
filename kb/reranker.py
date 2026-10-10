"""Cross-encoder reranker for RAG (quality boost).

Reranks the top-K chunks retrieved by vector search against the user's query using a
cross-encoder (default ``BAAI/bge-reranker-v2-m3``) that scores each ``(query, chunk)``
pair directly. This is a *separate* model from the embedding model — it does not produce
index vectors, so it never conflicts with the embedding space.

Two backends (``RERANK_MODE``):

* ``"local"`` (default) — in-process CPU cross-encoder via sentence-transformers.
  Loaded once, kept resident (see :func:`preload_reranker`).
* ``"http"`` — remote ``/rerank`` endpoint (Hugging Face TEI, see
  ``docs/internal/Todo_Reranker.md``). Keeps the model on a GPU and out of the bot's
  process/CPU. Payload/response follow the TEI API:
  ``POST {RERANK_API_BASE}/rerank`` with ``{"query": ..., "texts": [...]}`` →
  ``[{"index": i, "score": s}, ...]``.

Design goals
------------
* **Never breaks retrieval.** Any failure (missing dependency, model load error,
  unreachable endpoint, timeout, malformed output) falls back to the caller's original
  ordering, so a reranker problem can never make an /ai request fail or return no context.
* **Loaded once, kept resident** (local mode) / one shared HTTP client (http mode).

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

import httpx

logger = logging.getLogger("kb.reranker")

_model: Any = None
_model_lock: threading.Lock | None = None

# Shared HTTP client for RERANK_MODE=http (mirrors kb.embedder's pattern: one
# client, keep-alive across requests; closed on bot shutdown via close_client()).
_http_client: httpx.AsyncClient | None = None
_http_client_lock: asyncio.Lock | None = None


def _mode() -> str:
    from config.settings import RERANK_MODE
    return RERANK_MODE or "local"


# ─────────────────────────── local (in-process) backend ───────────────────────────

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


def _score_sync(query: str, chunks: list[str]) -> list[float]:
    """Score ``(query, chunk)`` pairs with the local model. Synchronous — call via
    ``asyncio.to_thread``. Cross-encoders emit a relevance logit per pair; higher is
    more relevant. We return raw scores (no sigmoid needed) since only their *ordering*
    is used."""
    model = _get_model()
    pairs = [(query, c) for c in chunks]
    scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
    return [float(s) for s in scores]


# ─────────────────────────────── http (TEI) backend ───────────────────────────────

async def _get_http_client() -> httpx.AsyncClient:
    """Return the shared rerank ``httpx.AsyncClient``, creating it on first use."""
    global _http_client, _http_client_lock
    if _http_client is not None:
        return _http_client
    if _http_client_lock is None:
        _http_client_lock = asyncio.Lock()
    async with _http_client_lock:
        if _http_client is None:
            from config.settings import RERANK_TIMEOUT_SECONDS
            _http_client = httpx.AsyncClient(timeout=RERANK_TIMEOUT_SECONDS)
    return _http_client


async def close_client() -> None:
    """Close the shared HTTP client (wired into bot shutdown). Safe to call anytime."""
    global _http_client
    if _http_client is not None:
        try:
            await _http_client.aclose()
        finally:
            _http_client = None


async def _rerank_http(base_url: str, query: str, chunks: list[str]) -> list[float]:
    """Score chunks against the query via a TEI-compatible ``/rerank`` endpoint.

    Returns one score per chunk, aligned to the input order. Raises on any HTTP /
    schema problem — the caller (``rerank``) degrades to the original order.
    """
    client = await _get_http_client()
    resp = await client.post(f"{base_url}/rerank", json={"query": query, "texts": chunks})
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"unexpected /rerank response type: {type(data).__name__}")
    scores = [0.0] * len(chunks)
    for item in data:
        idx = int(item["index"])
        if not 0 <= idx < len(chunks):
            raise ValueError(f"/rerank returned out-of-range index {idx}")
        scores[idx] = float(item["score"])
    return scores


# ──────────────────────────────── public interface ────────────────────────────────

def preload_reranker() -> None:
    """Warm the reranker at startup (no-op unless enabled + available). Never raises."""
    if not is_available():
        return
    if _mode() == "http":
        from config.settings import RERANK_API_BASE
        try:
            r = httpx.get(f"{RERANK_API_BASE}/health", timeout=5)
            r.raise_for_status()
            logger.info("Reranker endpoint healthy at %s (HTTP mode)", RERANK_API_BASE)
        except Exception as exc:  # noqa: BLE001 — startup must not crash on this
            logger.warning(
                "Reranker endpoint %s unreachable at startup (%s); will retry per request",
                RERANK_API_BASE, exc,
            )
        return
    try:
        _get_model()
        logger.info("Preloaded reranker model")
    except Exception as exc:  # noqa: BLE001 — startup must not crash on this
        logger.warning("Failed to preload reranker: %s", exc)


def is_available() -> bool:
    """True when reranking is enabled AND its backend is usable."""
    from config.settings import RERANK_ENABLED, RERANK_API_BASE
    if not RERANK_ENABLED:
        return False
    if _mode() == "http":
        return bool(RERANK_API_BASE)
    return importlib.util.find_spec("sentence_transformers") is not None


def reset_model() -> None:
    """Test helper: drop the cross-encoder singleton between event loops."""
    global _model
    _model = None


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

    from config.settings import RERANK_TIMEOUT_SECONDS, RERANK_API_BASE
    try:
        if _mode() == "http":
            scores = await asyncio.wait_for(
                _rerank_http(RERANK_API_BASE, query, chunks),
                timeout=RERANK_TIMEOUT_SECONDS,
            )
        else:
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
