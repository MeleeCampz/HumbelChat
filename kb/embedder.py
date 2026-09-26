"""Async embedding engine for an OpenAI-compatible /embeddings backend.

Uses ``INFER_URL`` + ``INFER_API_KEY`` from settings plus the model name
configured for embeddings (defaults to ``EMBEDDING_MODEL`` env var, "nomic-embed-text:latest").

This module provides both single-doc and batch encoding with automatic
retries and fallback logging if the backend is unreachable.

Usage
-----
    from kb.embedder import Embedder

    embedder = Embedder()
    vectors = await embedder.encode(["query text", "doc content"])
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import threading
from typing import Any

import httpx

logger = logging.getLogger("kb.embedder")

# ──────────────── Constants ───────────────────────────────────────────────

from config.settings import EMBEDDING_MODEL as _DEFAULT_MODEL
from config.settings import EMBED_TIMEOUT

# ── In-process (CPU) embedding backend ─────────────────────────────────────
# When EMBED_BACKEND == "local" we embed with a sentence-transformers model that
# lives *inside* the bot process instead of round-tripping to the chat backend.
# This removes the per-request model swap entirely: the llama.cpp backend only
# ever loads the chat model. The model is loaded once (lazy) and kept resident,
# so per-query cost is just a small CPU encode of the short text.
_local_model: Any = None
_local_model_lock: threading.Lock | None = None


def _get_local_model() -> Any:
    """Return the shared in-process embedding model, loading it on first use.

    Guarded by a plain ``threading`` lock (not asyncio) because ``encode`` runs
    this inside ``asyncio.to_thread`` — the load is blocking and thread-safe.
    """
    global _local_model, _local_model_lock
    if _local_model is not None:
        return _local_model
    if _local_model_lock is None:
        _local_model_lock = threading.Lock()
    with _local_model_lock:
        if _local_model is None:
            from sentence_transformers import SentenceTransformer
            from config.settings import LOCAL_EMBED_MODEL
            logger.info("Loading in-process embedding model '%s' (CPU)", LOCAL_EMBED_MODEL)
            _local_model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _local_model


def preload_local_embedder() -> None:
    """Warm the in-process embedding model at startup (no-op unless local).

    Called from bot startup so the first /ai request isn't cold. Never raises —
    a failure just means the first real encode will retry the load.
    """
    from config.settings import EMBED_BACKEND
    if EMBED_BACKEND != "local":
        return
    try:
        _get_local_model()
        logger.info("Preloaded local embedding model")
    except Exception as exc:  # noqa: BLE001 — startup must not crash on this
        logger.warning("Failed to preload local embedder: %s", exc)


def reset_local_model() -> None:
    """Test helper: drop the in-process model singleton between event loops."""
    global _local_model
    _local_model = None


def _local_available() -> bool:
    """True when sentence-transformers is importable (local CPU fallback possible)."""
    return importlib.util.find_spec("sentence_transformers") is not None
_BATCH_SIZE = 8  # documents per batch (conservative for shared inference backends)
_RETRY_ATTEMPTS = 3          # per endpoint — transient 5xx / connection errors are common
_RETRY_BACKOFF_SECONDS = 1.5 # base delay; multiplied by the attempt number


# ── P2 #21: shared HTTP client ─────────────────────────────────────────────
# The old code built a fresh ``httpx.AsyncClient`` for *every batch* (and
# actually for every retry attempt), so no TCP connection was ever reused across
# batches — each batch paid a full handshake to the local inference backend.
# One process-wide client is created lazily on first use and reused for every
# subsequent batch (keep-alive).  ``close_client()`` is called on bot shutdown
# (see main.on_shutdown) so the connection pool is torn down cleanly.
_shared_client: httpx.AsyncClient | None = None
_client_lock: asyncio.Lock | None = None


async def _get_client() -> httpx.AsyncClient:
    """Return the shared embeddings ``httpx.AsyncClient``, creating it on first use.

    Double-checked under an ``asyncio.Lock`` so concurrent first-time calls
    create exactly one client.  The per-request timeout comes from
    ``config.settings.EMBED_TIMEOUT`` (was hardcoded ``timeout=30`` per batch).
    """
    global _shared_client, _client_lock
    if _shared_client is not None:
        return _shared_client
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    async with _client_lock:
        if _shared_client is None:
            _shared_client = httpx.AsyncClient(timeout=EMBED_TIMEOUT)
    return _shared_client


async def close_client() -> None:
    """Close + reset the shared client (idempotent). Called on bot shutdown
    and by tests to isolate the process-wide client between event loops.

    Only the *client* is reset — the lock is kept (recreating asyncio locks
    is the one thing the codebase deliberately avoids; a single lock instance
    is safe to reuse across loops)."""
    global _shared_client
    if _shared_client is not None:
        try:
            await _shared_client.aclose()
        except Exception:
            pass
    _shared_client = None


class Embedder:
    """Async embedding provider wrapping an OpenAI-compatible /embeddings endpoint.

    Parameters
    ----------
    model_name : str
        Embedding model name (e.g. ``nomic-embed-text:latest``). Defaults to the
        value above when not provided.
    batch_size : int
        Maximum number of documents per API call.  Must be > 0.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        batch_size: int = _BATCH_SIZE,
        backend: str = "auto",
        fallback_local: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        # Decide the active backend once. ``backend`` overrides the global
        # EMBED_BACKEND when it is not "auto": "local" forces in-process CPU
        # encoding; "remote" forces the HTTP /embeddings path (used for GPU index
        # builds). "local" requires sentence-transformers; if it's missing we warn
        # and fall back to remote so retrieval still works.
        self._use_local = False
        self._fallback_local = fallback_local
        from config.settings import EMBED_BACKEND, LOCAL_EMBED_MODEL
        chosen = EMBED_BACKEND if backend == "auto" else backend
        if chosen == "local":
            if importlib.util.find_spec("sentence_transformers") is not None:
                self._use_local = True
                # Reflect the local model in metadata so the index cache rebuilds.
                self.model_name = LOCAL_EMBED_MODEL
            else:
                logger.warning(
                    "EMBED_BACKEND=local but sentence-transformers is not installed; "
                    "falling back to remote /embeddings (install the rag-cpu extra)."
                )

    # ── Public API ─────────────────────────────────────────────────────

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode *texts* into embedding vectors.  Returns ``[vec1, vec2, …]``.

        Handles batching automatically and raises :class:`EmbeddingError` on
        persistent backend failure (caller should fall back to keyword search).
        When constructed with ``fallback_local=True`` a backend failure instead
        transparently re-encodes in-process on CPU, so batch index builds never
        hard-fail on a downed backend.
        """
        if not texts:
            return []

        # Local (in-process CPU) backend: encode off the event loop. Deterministic
        # and fast, so no HTTP batching/endpoint probing is needed.
        if self._use_local:
            return await asyncio.to_thread(self._local_encode, list(texts))

        try:
            return await self._remote_encode(list(texts))
        except EmbeddingError as exc:
            if self._fallback_local and importlib.util.find_spec("sentence_transformers") is not None:
                logger.warning(
                    "Backend /embeddings failed (%s); falling back to local CPU for %d text(s).",
                    exc, len(texts),
                )
                return await asyncio.to_thread(self._local_encode, list(texts))
            raise

    async def _remote_encode(self, texts: list[str]) -> list[list[float]]:
        """Encode *texts* via the HTTP /embeddings endpoint (batched + deduped)."""
        # Deduplicate while preserving order for result alignment
        seen: dict[str, int] = {}
        unique_texts: list[str] = []
        for t in texts:
            if t not in seen:
                seen[t] = len(unique_texts)
                unique_texts.append(t)

        all_embeddings: dict[int, list[float]] = {}
        for i in range(0, len(unique_texts), self.batch_size):
            batch = unique_texts[i : i + self.batch_size]
            try:
                vectors = await self._call_api(batch)
            except EmbeddingError as exc:
                # Per-batch fallback: a single bad batch (e.g. very long chunks that
                # time out) degrades only itself to local CPU instead of forcing the
                # whole build off the backend. Local vectors are geometrically
                # identical to the backend's (same model, L2-normalized), so mixing is safe.
                if not (self._fallback_local and _local_available()):
                    raise
                logger.warning(
                    "Backend failed for a %d-text batch (%s); embedding that batch locally.",
                    len(batch), exc,
                )
                vectors = await asyncio.to_thread(self._local_encode, list(batch))
            for idx, vec in enumerate(vectors):
                all_embeddings[seen[batch[idx]]] = vec

        # Reconstruct result in original order, mapping each text back to its seen index
        result: list[list[float]] = []
        for t in texts:
            result.append(all_embeddings[seen[t]])
        return result

    # ── HTTP helpers ───────────────────────────────────────────────────

    def _local_encode(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* with the in-process model. Synchronous — call via to_thread.

        Vectors are L2-normalized so they match the cosine ranking used by
        ``KBVectorIndex`` (identical geometry to the remote path's expectations).
        """
        import numpy as np
        model = _get_local_model()
        vecs = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vecs, dtype="float32")
        return [row.tolist() for row in arr]

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Send a batch to the configured /embeddings endpoint."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }

        # ── Resolve runtime settings (lazy to avoid circular import) ────
        from config.settings import INFER_URL

        base_url = INFER_URL.rstrip("/")

        # Determine what suffix to append so the /embeddings endpoint is correct.
        # Users may set INFER_URL already ending in /api/v1 or /v1, or none of those.
        if base_url.endswith(("/api/v1", "/v1")):
            remaining_suffixes = ["/embeddings"]
        else:
            # For generic URLs (e.g. http://host:port), try both conventions
            remaining_suffixes = ["/api/v1/embeddings", "/v1/embeddings"]

        from config.settings import INFER_API_KEY
        api_key = INFER_API_KEY or ""
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # OpenAI-compatible embeddings endpoints are typically /embeddings or /v1/embeddings
        endpoints_to_try = remaining_suffixes

        # P2 #21: reuse ONE shared client across every batch/endpoint/attempt
        # (connection pooling / keep-alive) instead of a fresh client per call.
        client = await _get_client()

        last_exc: Exception | None = None
        for suffix in endpoints_to_try:
            url = base_url + suffix
            for attempt in range(1, _RETRY_ATTEMPTS + 1):
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()

                    # OpenAI-compatible response format: {"data": [...], "model": ...}
                    if isinstance(data, dict) and "data" in data:
                        embeddings = [d["embedding"] for d in data["data"]]  # type: ignore[index]
                    else:
                        raise ValueError(f"Unexpected response shape: {data}")

                    logger.debug(
                        "Embedded %d texts via %s (model=%s)",
                        len(texts),
                        url,
                        self.model_name,
                    )
                    return embeddings

                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    status = exc.response.status_code if exc.response is not None else 0
                    # 4xx (except 429) means this endpoint is wrong or rejects the
                    # request — retrying won't help, move on to the next one.
                    if 400 <= status < 500 and status != 429:
                        logger.warning(
                            "Embeddings endpoint %s returned %d — trying next", url, status
                        )
                        break
                    logger.warning(
                        "Embeddings endpoint %s attempt %d/%d failed (HTTP %d)",
                        url, attempt, _RETRY_ATTEMPTS, status,
                    )
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "Embeddings endpoint %s attempt %d/%d error: %s",
                        url, attempt, _RETRY_ATTEMPTS, exc,
                    )

                if attempt < _RETRY_ATTEMPTS:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS * attempt)

        # All endpoints failed
        raise EmbeddingError(
            f"All embedding endpoints failed. Last error: {last_exc}",  # type: ignore[arg-type]
        )


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend is unreachable or returns invalid data."""
