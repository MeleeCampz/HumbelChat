"""Async embedding engine powered by the OpenWebUI-compatible backend.

Uses ``INFER_URL`` + ``INFER_API_KEY`` from settings plus the model name
configured for embeddings (defaults to ``nomic-embed-text:latest``).

This module provides both single-doc and batch encoding with automatic
retries and fallback logging if the backend is unreachable.

Usage
-----
    from kb.embedder_openai import OpenAIEmbedder

    embedder = OpenAIEmbedder()
    vectors = await embedder.encode(["query text", "doc content"])
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from config.settings import Settings  # noqa: F401

logger = logging.getLogger("kb.embedder_openai")

# ──────────────── Constants ───────────────────────────────────────────────

_DEFAULT_MODEL = "nomic-embed-text:latest"
_BATCH_SIZE = 2048  # tokens per batch (safe for most embedding models)


class OpenAIEmbedder:
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
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, batch_size)

    # ── Public API ─────────────────────────────────────────────────────

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode *texts* into embedding vectors.  Returns ``[vec1, vec2, …]``.

        Handles batching automatically and raises :class:`EmbeddingError` on
        persistent backend failure (caller should fall back to keyword search).
        """
        if not texts:
            return []

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
                logger.error("Embedding API call failed: %s", exc)
                raise

            for idx, vec in enumerate(vectors):
                all_embeddings[seen[batch[idx]]] = vec

        return [all_embeddings[j] for j in range(len(texts))]

    # ── HTTP helpers ───────────────────────────────────────────────────

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Send a batch to the OpenWebUI /embeddings endpoint."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }

        # ── Resolve runtime settings (lazy to avoid circular import) ────
        from config.settings import settings as _s  # type: ignore[attr-defined]

        base_url = _s.INFER_URL.rstrip("/")

        # Determine what suffix to append so the /embeddings endpoint is correct.
        # Users may set INFER_URL already ending in /api/v1 or /v1, or none of those.
        if base_url.endswith("/api/v1"):
            api_v1_base = base_url
            remaining_suffixes = ["/embeddings"]
        elif base_url.endswith("/v1"):
            api_v1_base = base_url
            remaining_suffixes = ["/embeddings"]
        else:
            api_v1_base = base_url + "/api/v1"
            # For generic URLs (e.g. http://host:port), try both conventions
            remaining_suffixes = ["/embeddings", "/v1/embeddings"]

        api_key = _s.INFER_API_KEY or ""
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # OpenWebUI embeddings endpoint is typically /embeddings or /v1/embeddings
        endpoints_to_try = remaining_suffixes

        last_exc: Exception | None = None
        for suffix in endpoints_to_try:
            url = base_url + suffix
            try:
                async with httpx.AsyncClient(timeout=30) as client:
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
                logger.warning("Embeddings endpoint %s returned %d — trying next", url, resp.status_code if "resp" in dir() else 0)
                continue
            except Exception as exc:
                last_exc = exc
                logger.warning("Embeddings endpoint %s error: %s", url, exc)
                continue

        # All endpoints failed
        raise EmbeddingError(
            f"All embedding endpoints failed. Last error: {last_exc}",  # type: ignore[arg-type]
        )


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend is unreachable or returns invalid data."""
