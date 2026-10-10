"""RERANK_MODE=http: the reranker must score chunks via a TEI-compatible /rerank
endpoint and keep the never-breaks-retrieval contract (any failure → original
order with 0.0 scores). Mirrors the fake-client pattern of test_p2_embedder_client.py.
"""
from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
from unittest.mock import patch

import kb.reranker as R
import config.settings as S


# ──────────────────────────── Fake TEI client ──────────────────────────────────

class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeTEIClient:
    """Records construction; ``post`` returns a TEI-style score list."""

    def __init__(self, *a, **k):
        self.timeout = k.get("timeout")
        self.payload = getattr(_FakeTEIClient, "payload", None)
        self.raise_exc = getattr(_FakeTEIClient, "raise_exc", None)
        self.sleep_s = getattr(_FakeTEIClient, "sleep_s", 0)
        self.aclose_called = 0

    async def post(self, url, json=None, headers=None):
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        if self.raise_exc is not None:
            raise self.raise_exc
        assert url.endswith("/rerank")
        _FakeTEIClient.requests.append({"url": url, "json": json})
        return _Resp(self.payload)

    async def aclose(self):
        self.aclose_called += 1


@pytest_asyncio.fixture(autouse=True)
async def _isolate(monkeypatch):
    """HTTP mode + base URL on, shared client reset between tests."""
    monkeypatch.setattr(S, "RERANK_ENABLED", True)
    monkeypatch.setattr(S, "RERANK_MODE", "http")
    monkeypatch.setattr(S, "RERANK_API_BASE", "http://reranker.test:80")
    monkeypatch.setattr(S, "RERANK_TIMEOUT_SECONDS", 1)
    _FakeTEIClient.payload = None
    _FakeTEIClient.raise_exc = None
    _FakeTEIClient.sleep_s = 0
    _FakeTEIClient.requests = []
    await R.close_client()
    yield
    await R.close_client()


def _factory(recorded):
    def factory(*a, **k):
        recorded.append(k)
        return _FakeTEIClient(*a, **k)
    return factory


# ──────────────────────────── Tests ────────────────────────────────────────

class TestHttpRerank:

    @pytest.mark.asyncio
    async def test_sorts_by_score_and_truncates(self):
        """TEI returns scores in arbitrary order; rerank must sort desc + top_n."""
        chunks = ["alpha", "beta", "gamma"]
        # gamma is best, alpha worst; indices reference the input order.
        _FakeTEIClient.payload = [
            {"index": 2, "score": 0.9},
            {"index": 1, "score": 0.5},
            {"index": 0, "score": 0.1},
        ]
        recorded: list[dict] = []
        with patch.object(R.httpx, "AsyncClient", _factory(recorded)):
            out = await R.rerank("q", chunks, top_n=2)

        assert [c for c, _ in out] == ["gamma", "beta"]
        assert [round(s, 1) for _, s in out] == [0.9, 0.5]
        # Payload must carry the query and the texts in original order.
        req = _FakeTEIClient.requests[0]
        assert req["url"] == "http://reranker.test:80/rerank"
        assert req["json"] == {"query": "q", "texts": chunks}

    @pytest.mark.asyncio
    async def test_one_shared_client_across_calls(self):
        _FakeTEIClient.payload = [{"index": 0, "score": 1.0}]
        recorded: list[dict] = []
        with patch.object(R.httpx, "AsyncClient", _factory(recorded)):
            await R.rerank("q1", ["a"])
            await R.rerank("q2", ["b"])
        assert len(recorded) == 1, "must reuse one httpx client across rerank calls"

    @pytest.mark.asyncio
    async def test_http_error_falls_back_to_original_order(self):
        _FakeTEIClient.raise_exc = RuntimeError("connection refused")
        recorded: list[dict] = []
        with patch.object(R.httpx, "AsyncClient", _factory(recorded)):
            out = await R.rerank("q", ["a", "b"])
        assert [c for c, _ in out] == ["a", "b"]
        assert all(s == 0.0 for _, s in out)

    @pytest.mark.asyncio
    async def test_out_of_range_index_falls_back(self):
        """Malformed TEI response (index beyond input) must not corrupt results."""
        _FakeTEIClient.payload = [{"index": 7, "score": 0.9}]
        recorded: list[dict] = []
        with patch.object(R.httpx, "AsyncClient", _factory(recorded)):
            out = await R.rerank("q", ["a", "b"])
        assert [c for c, _ in out] == ["a", "b"]
        assert all(s == 0.0 for _, s in out)

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_original_order(self):
        """A hung endpoint must hit RERANK_TIMEOUT_SECONDS and degrade gracefully."""
        _FakeTEIClient.sleep_s = 10
        recorded: list[dict] = []
        with patch.object(R.httpx, "AsyncClient", _factory(recorded)):
            out = await R.rerank("q", ["a", "b"])
        assert [c for c, _ in out] == ["a", "b"]
        assert all(s == 0.0 for _, s in out)

    @pytest.mark.asyncio
    async def test_empty_chunks_no_request(self):
        recorded: list[dict] = []
        with patch.object(R.httpx, "AsyncClient", _factory(recorded)):
            out = await R.rerank("q", [])
        assert out == []
        assert recorded == []


class TestHttpAvailability:

    def test_http_mode_without_base_url_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(S, "RERANK_API_BASE", "")
        assert R.is_available() is False

    @pytest.mark.asyncio
    async def test_unavailable_returns_original_order_without_client(self, monkeypatch):
        monkeypatch.setattr(S, "RERANK_API_BASE", "")
        recorded: list[dict] = []
        with patch.object(R.httpx, "AsyncClient", _factory(recorded)):
            out = await R.rerank("q", ["a", "b"])
        assert [c for c, _ in out] == ["a", "b"]
        assert recorded == [], "no HTTP client may be built when unavailable"

    def test_disabled_is_unavailable_in_any_mode(self, monkeypatch):
        monkeypatch.setattr(S, "RERANK_ENABLED", False)
        assert R.is_available() is False

    def test_local_mode_still_requires_sentence_transformers(self, monkeypatch):
        monkeypatch.setattr(S, "RERANK_MODE", "local")
        import importlib.util
        with patch.object(importlib.util, "find_spec", return_value=None):
            assert R.is_available() is False
