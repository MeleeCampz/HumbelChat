"""P2 #21: the embedder must reuse ONE shared httpx client across batches.

The old code built a fresh ``httpx.AsyncClient`` for every batch (indeed for
every retry attempt), so no TCP connection was ever kept alive across batches.
These tests assert that N batches produce exactly ONE client instance, that the
per-request timeout comes from ``config.settings.EMBED_TIMEOUT`` (no longer a
hardcoded ``timeout=30``), and that ``close_client()`` (wired into
``main.on_shutdown``) tears the client down cleanly.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import patch

import kb.embedder as E
from config.settings import EMBED_TIMEOUT


# ──────────────────────────── Fake client ──────────────────────────────────

class _Resp:
    def __init__(self, vecs):
        self._vecs = vecs

    def raise_for_status(self):
        return None

    def json(self):
        # OpenAI-compatible: {"data": [{"embedding": [...]}, ...]}
        return {"data": [{"embedding": v} for v in self._vecs], "model": "fake"}


class _FakeClient:
    """Records construction; ``post`` returns one vector per input item."""

    def __init__(self, *a, **k):
        self.timeout = k.get("timeout")
        self.aclose_called = 0

    async def post(self, url, json=None, headers=None):
        n = len(json.get("input", [])) if json else 0
        return _Resp([[0.1] * 8 for _ in range(n)])

    async def aclose(self):
        self.aclose_called += 1


@pytest_asyncio.fixture(autouse=True)
async def _reset_shared_client():
    """Isolate the process-wide client between tests (and across loops)."""
    await E.close_client()
    yield
    await E.close_client()


def _factory(recorded):
    def factory(*a, **k):
        recorded.append(k)
        return _FakeClient(*a, **k)
    return factory


# ──────────────────────────── Tests ────────────────────────────────────────

class TestSharedClient:

    @pytest.mark.asyncio
    async def test_multiple_batches_one_client(self):
        """5 texts at batch_size=2 → 3 batches, but exactly 1 client built."""
        recorded: list[dict] = []
        emb = E.Embedder(model_name="m", batch_size=2)
        texts = [f"text {i}" for i in range(5)]  # 5 → 3 batches (2+2+1)

        with patch.object(E.httpx, "AsyncClient", _factory(recorded)):
            vecs = await emb.encode(texts)

        assert len(recorded) == 1, "must build exactly ONE httpx client for all batches"
        assert len(vecs) == 5  # results aligned back to original order

    @pytest.mark.asyncio
    async def test_same_client_reused_across_separate_encode_calls(self):
        """Even across *separate* encode() calls (process lifetime), 1 client."""
        recorded: list[dict] = []
        emb = E.Embedder(model_name="m", batch_size=4)

        with patch.object(E.httpx, "AsyncClient", _factory(recorded)):
            await emb.encode(["a", "b"])
            await emb.encode(["c", "d"])  # second batch round

        assert len(recorded) == 1

    @pytest.mark.asyncio
    async def test_timeout_comes_from_settings(self):
        """The client is constructed with config.settings.EMBED_TIMEOUT."""
        recorded: list[dict] = []
        with patch.object(E.httpx, "AsyncClient", _factory(recorded)):
            client = await E._get_client()

        assert recorded[0].get("timeout") == EMBED_TIMEOUT
        assert client.timeout == EMBED_TIMEOUT

    @pytest.mark.asyncio
    async def test_get_client_is_singleton_within_callers(self):
        """Two callers see the very same client object."""
        recorded: list[dict] = []
        with patch.object(E.httpx, "AsyncClient", _factory(recorded)):
            a = await E._get_client()
            b = await E._get_client()

        assert a is b
        assert len(recorded) == 1


class TestCloseClient:

    @pytest.mark.asyncio
    async def test_close_client_acloses_and_resets(self):
        recorded: list[dict] = []
        with patch.object(E.httpx, "AsyncClient", _factory(recorded)):
            client = await E._get_client()
        assert E._shared_client is client

        await E.close_client()
        assert client.aclose_called == 1, "aclose() must be awaited on the client"
        assert E._shared_client is None, "module global must be reset to None"

    @pytest.mark.asyncio
    async def test_close_client_is_idempotent(self):
        """Calling it with no live client must not raise."""
        await E.close_client()
        await E.close_client()  # no-op, must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
