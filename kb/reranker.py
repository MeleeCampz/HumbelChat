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
import time
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


def _failure_detail(exc: Exception) -> str:
    """Render an exception for logging. HTTP errors include a truncated response
    body — a bare status line ("400 Bad Request") says nothing about *why* the
    server rejected the request (e.g. vLLM serving the model with the wrong
    runner/task). The body is whitespace-collapsed and capped at 300 chars so it
    stays on one log line."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = " ".join(exc.response.text.split())[:300]
        except Exception:  # noqa: BLE001 — logging must never raise
            body = "<unreadable body>"
        return f"{exc} | server response: {body}"
    return str(exc)


async def close_client() -> None:
    """Close the shared HTTP client (wired into bot shutdown). Safe to call anytime."""
    global _http_client
    if _http_client is not None:
        try:
            await _http_client.aclose()
        finally:
            _http_client = None


# Newer TEI images validate /rerank against --max-batch-size (default 32) and
# answer 422 above it; our RERANK_TOP_K is 100. Chunk to the smallest common
# server cap so any TEI version accepts our requests (docker-compose.yml also
# raises the cap for speed — this keeps the bot safe across image updates).
_MAX_TEXTS_PER_RERANK_REQUEST = 32


async def _rerank_http(base_url: str, query: str, chunks: list[str]) -> list[float]:
    """Score chunks against the query via a remote ``/rerank`` endpoint.

    Two wire formats (``RERANK_API_FORMAT``):

    * ``tei`` — Hugging Face TEI: ``{"query", "texts"}`` → bare list of
      ``{"index", "score"}``.
    * ``jina`` — vLLM's Jina-compatible rerank (Qwen3-Reranker sidecar):
      ``{"query", "documents", "instruction"?}`` →
      ``{"results": [{"index", "relevance_score"}, ...]}``.

    Sends at most ``_MAX_TEXTS_PER_RERANK_REQUEST`` texts per request and merges
    the partial results. Returns one score per chunk, aligned to the input order.
    Raises on any HTTP / schema problem — the caller (``rerank``) degrades to the
    original order.
    """
    from config.settings import RERANK_API_FORMAT, RERANK_INSTRUCT, RERANK_MODEL

    jina = RERANK_API_FORMAT == "jina"
    doc_field = "documents" if jina else "texts"
    client = await _get_http_client()
    scores = [0.0] * len(chunks)
    for start in range(0, len(chunks), _MAX_TEXTS_PER_RERANK_REQUEST):
        batch = chunks[start : start + _MAX_TEXTS_PER_RERANK_REQUEST]
        payload: dict[str, Any] = {"query": query, doc_field: batch}
        if jina:
            # llama.cpp's OpenAI-style /v1/rerank expects a model name; vLLM
            # accepts (and matches) it too — safe for both backends.
            payload["model"] = RERANK_MODEL
            if RERANK_INSTRUCT:
                payload["instruction"] = RERANK_INSTRUCT
        resp = await client.post(f"{base_url}/rerank", json=payload)
        resp.raise_for_status()
        data = resp.json()
        items: list[Any]
        if jina:
            # vLLM wraps results in an object; TEI returns a bare list.
            if not isinstance(data, dict) or "results" not in data:
                raise ValueError("jina /rerank response missing 'results'")
            items = data["results"]
        else:
            if not isinstance(data, list):
                raise ValueError(f"unexpected /rerank response type: {type(data).__name__}")
            items = data
        for item in items:
            idx = int(item["index"])
            if not 0 <= idx < len(batch):
                raise ValueError(f"/rerank returned out-of-range index {idx}")
            scores[start + idx] = float(item["relevance_score" if jina else "score"])
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
            if _is_connect_failure(exc):
                # Circuit opens immediately: the first real query won't waste
                # an attempt ladder on a sidecar that is provably down.
                _breaker_trip(RERANK_API_BASE.rstrip("/"), exc)
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


# ── Circuit breaker (RAG code review 2026-11, R2a) ────────────────────
# Mirror of kb.embedder's breaker: a stopped reranker sidecar (4070 image
# window) fails at the socket level instantly, but retrieval asks for TWO
# rerank passes per query plus one per agent tool call — dozens of fresh
# attempts per turn against a service that cannot answer until the GPU
# switches back. A connect-class failure opens the breaker for a cooldown:
# rerank() degrades to original order instantly (no attempt) and retrieval
# skips the rerank stage outright. Cooldown knob: RAG_SIDECAR_CIRCUIT_COOLDOWN_S.
_circuit_open_until: dict[str, float] = {}   # base_url -> monotonic open-until


def _circuit_cooldown_s() -> float:
    from config.settings import RAG_SIDECAR_CIRCUIT_COOLDOWN_S
    return RAG_SIDECAR_CIRCUIT_COOLDOWN_S


def _breaker_open(base_url: str) -> bool:
    until = _circuit_open_until.get(base_url)
    return until is not None and time.monotonic() < until


def _breaker_trip(base_url: str, exc: BaseException) -> None:
    _circuit_open_until[base_url] = time.monotonic() + _circuit_cooldown_s()
    logger.warning(
        "Rerank endpoint %s marked unreachable for %.0fs (circuit breaker; "
        "connect error: %s) — reranking skipped until it answers again",
        base_url, _circuit_cooldown_s(), exc,
    )


def _breaker_close(base_url: str) -> None:
    _circuit_open_until.pop(base_url, None)


def reranker_circuit_open() -> bool:
    """True while the HTTP rerank endpoint is in breaker cooldown."""
    from config.settings import RERANK_API_BASE
    return bool(RERANK_API_BASE) and _breaker_open(RERANK_API_BASE.rstrip("/"))


def reset_reranker_circuit_breaker() -> None:
    """Clear all breaker state (tests / manual re-probe after repair)."""
    _circuit_open_until.clear()


def reset_circuit_breaker() -> None:
    """Clear all breaker state (tests / manual re-probe after repair)."""
    _circuit_open_until.clear()


def _is_connect_failure(exc: BaseException) -> bool:
    """Socket-level failure (DNS gone / refused) — endpoint is DOWN, unlike a
    5xx or a slow-answer timeout on a reachable server."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    cause = exc.__cause__ or exc.__context__
    return isinstance(cause, (ConnectionError, OSError))


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

    from config.settings import RERANK_TIMEOUT_SECONDS, RERANK_API_BASE, RERANK_MAX_CHARS
    if _mode() == "http" and reranker_circuit_open():
        # Circuit open — degrade instantly, don't hammer the downed sidecar.
        logger.debug("Rerank circuit open — returning original order")
        return [(c, 0.0) for c in chunks]
    # Score truncated copies when configured: entry chunks lead with title + type +
    # core mechanics, so the head carries the answer signal. Originals are zipped
    # back with the scores below — callers must receive full chunk text untouched.
    score_input = [c[:RERANK_MAX_CHARS] for c in chunks] if RERANK_MAX_CHARS > 0 else chunks
    t0 = time.monotonic()
    try:
        if _mode() == "http":
            scores = await asyncio.wait_for(
                _rerank_http(RERANK_API_BASE, query, score_input),
                timeout=RERANK_TIMEOUT_SECONDS,
            )
        else:
            scores = await asyncio.wait_for(
                asyncio.to_thread(_score_sync, query, score_input),
                timeout=RERANK_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        logger.warning("Rerank timed out after %ss — using original order", RERANK_TIMEOUT_SECONDS)
        return [(c, 0.0) for c in chunks]
    except Exception as exc:  # noqa: BLE001 — any failure degrades to original order
        if _mode() == "http" and _is_connect_failure(exc):
            _breaker_trip(RERANK_API_BASE.rstrip("/"), exc)
        logger.warning("Rerank failed (%s); using original order", _failure_detail(exc))
        return [(c, 0.0) for c in chunks]

    if _mode() == "http":
        _breaker_close(RERANK_API_BASE.rstrip("/"))

    if len(scores) != len(chunks):  # defensive: malformed output → don't trust it
        logger.warning(
            "Rerank returned %d scores for %d chunks — using original order",
            len(scores), len(chunks),
        )
        return [(c, 0.0) for c in chunks]

    scored = sorted(zip(chunks, scores), key=lambda t: t[1], reverse=True)
    final_n = len(scored)
    if top_n is not None and top_n > 0:
        scored = scored[:top_n]
    logger.debug(
        "Reranked %d candidate(s) -> %d kept in %.0f ms; top score=%.3f",
        len(chunks), final_n, (time.monotonic() - t0) * 1000,
        scored[0][1] if scored else 0.0,
    )
    return scored
