"""P5 regression tests: streaming /ai with explicit time budgets.

The 2026-09-24 failure: a 41 K-char prompt on a 27 B thinking model was
killed by the request timeout, but the SDK's *default* max_retries=3 turned
one "attempt" into 3×145 s of retries inside the client. The agreed fix:

  * exactly ONE retry when a request fails (bot-owned, transient only);
  * streamed responses with three configurable time budgets —
      AI_STREAM_INITIAL_TIMEOUT_S (default 90 s, larger because prompt
      processing + the hidden thinking phase emit no tokens),
      AI_STREAM_CHUNK_TIMEOUT_S   (default 60 s per-chunk gap),
      AI_STREAM_TOTAL_TIMEOUT_S   (default 300 s hard cap);
  * streaming is the default delivery path (AI_STREAM=1);
  * all of the above overridable in the config (.env).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

import config.settings as settings
from bot_core import ai_client


# ─────────────────────────── helpers ───────────────────────────


def _http_response(status_code: int) -> httpx.Response:
    req = httpx.Request("POST", "http://backend.invalid/v1/chat/completions")
    return httpx.Response(status_code, request=req)


def _openai_status_error(status_code: int, message: str = "error") -> openai.APIStatusError:
    cls = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        429: openai.RateLimitError,
    }.get(status_code, openai.InternalServerError)
    return cls(message, response=_http_response(status_code), body=None)


def _success_resp() -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="PONG"))]
    return resp


def _stub_ask_ai_env(monkeypatch, create_side_effect):
    """Point ask_ai / ask_ai_stream at a stub client + empty KB."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    client.models.list = AsyncMock(return_value=MagicMock(data=[]))

    monkeypatch.setattr(ai_client, "_make_client", lambda: client)
    monkeypatch.setattr(ai_client, "_validate_model",
                        AsyncMock(return_value="test-model"))
    from kb import retrievers

    monkeypatch.setattr(retrievers, "retrieve_kb_documents",
                        AsyncMock(return_value=[]))
    return client


class _FakeStream:
    """Async iterator mimicking the openai SDK stream."""

    def __init__(self, items, delay: float = 0.0):
        self._items = list(items)
        self._delay = delay
        self._i = 0
        self.closed = False
        self._final_finish_reason = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._i]
        self._i += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return item

    def close(self):
        self.closed = True


def _stream_chunk(delta: str | None, finish_reason: str | None = None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=delta), finish_reason=finish_reason)]
    return chunk


# ─────────────────── retry policy: exactly ONE retry ───────────────────


def test_shared_client_disables_sdk_retries(monkeypatch):
    """The shared AsyncOpenAI client must be created with max_retries=0 so
    the bot's own single retry is the only retry layer (no 3× SDK retries)."""
    monkeypatch.setattr(ai_client, "_shared_client", None)
    with patch("bot_core.ai_client.AsyncOpenAI") as m:
        m.return_value = MagicMock()
        client = ai_client._make_client()
        m.assert_called_once()
        kwargs = m.call_args.kwargs
        assert kwargs.get("max_retries") == 0
        assert client is m.return_value


class TestAskAiCompletionRetry:
    @pytest.mark.asyncio
    async def test_transient_failure_retries_exactly_once_and_succeeds(self, monkeypatch):
        first = _openai_status_error(500, message="upstream exploded")
        client = _stub_ask_ai_env(monkeypatch, [first, _success_resp()])
        with patch.object(ai_client, "_COMPLETION_RETRY_BACKOFF_S", 0.01):
            reply, _ = await ai_client.ask_ai(
                user_message="ping", model_slug="test-model",
                guild_id=1, channel_id=2, username="Alice", user_id=None,
            )
        assert reply == "PONG"
        assert client.chat.completions.create.await_count == 2  # 1 + ONE retry

    @pytest.mark.asyncio
    async def test_persistent_failure_stops_after_single_retry(self, monkeypatch):
        exc = _openai_status_error(500, message="upstream exploded")
        client = _stub_ask_ai_env(monkeypatch, exc)
        with patch.object(ai_client, "_COMPLETION_RETRY_BACKOFF_S", 0.01):
            with pytest.raises(Exception):
                await ai_client.ask_ai(
                    user_message="ping", model_slug="test-model",
                    guild_id=3, channel_id=4, username="Alice", user_id=None,
                )
        # 1 initial attempt + exactly 1 retry — no SDK-level retries on top.
        assert client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_non_transient_failure_is_not_retried(self, monkeypatch):
        exc = _openai_status_error(400, message="bad request")
        client = _stub_ask_ai_env(monkeypatch, exc)
        with pytest.raises(Exception):
            await ai_client.ask_ai(
                user_message="ping", model_slug="test-model",
                guild_id=5, channel_id=6, username="Alice", user_id=None,
            )
        assert client.chat.completions.create.await_count == 1


# ──────────────────── stream time budgets ────────────────────


def _patch_budgets(monkeypatch, initial: float, chunk: float, total: float):
    monkeypatch.setattr(settings, "AI_STREAM_INITIAL_TIMEOUT_S", initial)
    monkeypatch.setattr(settings, "AI_STREAM_CHUNK_TIMEOUT_S", chunk)
    monkeypatch.setattr(settings, "AI_STREAM_TOTAL_TIMEOUT_S", total)
    # ai_client imports the names at module import time — patch them there too.
    monkeypatch.setattr(ai_client, "AI_STREAM_INITIAL_TIMEOUT_S", initial)
    monkeypatch.setattr(ai_client, "AI_STREAM_CHUNK_TIMEOUT_S", chunk)
    monkeypatch.setattr(ai_client, "AI_STREAM_TOTAL_TIMEOUT_S", total)


class TestStreamBudgets:
    @pytest.mark.asyncio
    async def test_success_within_budgets_persists_final_reply(self, monkeypatch):
        fake = _FakeStream([_stream_chunk("Hel"), _stream_chunk("lo", "stop")])
        client = _stub_ask_ai_env(monkeypatch, lambda **kw: fake)
        _patch_budgets(monkeypatch, 90, 60, 300)

        collected = []
        async for text in ai_client.ask_ai_stream(
            user_message="ping", model_slug="test-model",
            guild_id=7, channel_id=8, username="Alice", user_id=None,
        ):
            collected.append(text)

        assert collected == ["Hel", "Hello", "Hello"]  # progressive + final
        assert fake.closed is True                      # connection released
        # stream=True was requested with the initial budget as HTTP timeout
        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["timeout"] == 90

    @pytest.mark.asyncio
    async def test_initial_wait_larger_than_chunk_gap(self, monkeypatch):
        """Budgets flow through from settings: 90 s initial > 60 s chunk,
        300 s total hard cap — the values the user asked for as defaults."""
        assert settings.AI_STREAM_INITIAL_TIMEOUT_S == 90
        assert settings.AI_STREAM_CHUNK_TIMEOUT_S == 60
        assert settings.AI_STREAM_TOTAL_TIMEOUT_S == 300
        assert settings.AI_STREAM_INITIAL_TIMEOUT_S > settings.AI_STREAM_CHUNK_TIMEOUT_S

    @pytest.mark.asyncio
    async def test_no_first_chunk_raises_initial_timeout(self, monkeypatch):
        class _NeverStream:
            def __init__(self):
                self.closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Event().wait()  # never completes

            def close(self):
                self.closed = True

        fake = _NeverStream()
        _stub_ask_ai_env(monkeypatch, lambda **kw: fake)
        _patch_budgets(monkeypatch, 0.05, 60, 300)

        with pytest.raises(ValueError) as ei:
            async for _ in ai_client.ask_ai_stream(
                user_message="ping", model_slug="test-model",
                guild_id=9, channel_id=10, username="Alice", user_id=None,
            ):
                pass
        # user sees the friendly timeout message; the classified cause
        # carries the initial-phase detail (no chunk yet → initial, not per-chunk)
        assert "too long to respond" in str(ei.value)
        cause = ei.value.__cause__
        assert "initial timeout" in str(cause)
        assert "per-chunk" not in str(cause)
        assert fake.closed is True

    @pytest.mark.asyncio
    async def test_stalled_between_chunks_raises_chunk_timeout(self, monkeypatch):
        class _StalledStream:
            def __init__(self):
                self.closed = False
                self._done_first = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._done_first:
                    self._done_first = True
                    return _stream_chunk("start")
                await asyncio.Event().wait()  # stall forever after first chunk

            def close(self):
                self.closed = True

        fake = _StalledStream()
        _stub_ask_ai_env(monkeypatch, lambda **kw: fake)
        _patch_budgets(monkeypatch, 90, 0.05, 300)

        with pytest.raises(ValueError) as ei:
            async for _ in ai_client.ask_ai_stream(
                user_message="ping", model_slug="test-model",
                guild_id=11, channel_id=12, username="Alice", user_id=None,
            ):
                pass
        assert "per-chunk" in str(ei.value.__cause__)
        assert fake.closed is True

    @pytest.mark.asyncio
    async def test_total_cap_stops_long_running_stream(self, monkeypatch):
        # A slow trickle that never hits the per-chunk gap (0.01 s << 60 s)
        # must still be cut off by the 300 s hard cap (shrunk to 0.1 s here).
        class _DribbleStream:
            def __init__(self):
                self.closed = False
                self._n = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self._n += 1
                await asyncio.sleep(0.01)
                return _stream_chunk(f"t{self._n}")

            def close(self):
                self.closed = True

        fake = _DribbleStream()
        _stub_ask_ai_env(monkeypatch, lambda **kw: fake)
        _patch_budgets(monkeypatch, 90, 60, 0.1)

        with pytest.raises(ValueError) as ei:
            async for _ in ai_client.ask_ai_stream(
                user_message="ping", model_slug="test-model",
                guild_id=13, channel_id=14, username="Alice", user_id=None,
            ):
                pass
        assert "total time budget" in str(ei.value.__cause__)
        assert fake.closed is True


# ─────────────────── config overrides ───────────────────


class TestConfigOverrides:
    def test_defaults_are_user_requested_values(self):
        # "hard cap of 5 minutes total ... 1 minute timeout for chunks,
        # with a bit of a larger initial wait"
        assert settings.AI_STREAM_TOTAL_TIMEOUT_S == 300        # 5 minutes
        assert settings.AI_STREAM_CHUNK_TIMEOUT_S == 60         # 1 minute
        assert settings.AI_STREAM_INITIAL_TIMEOUT_S == 90       # larger initial wait
        assert settings.AI_STREAM is True                       # streaming on by default

    def test_env_override_parsed(self, monkeypatch):
        import importlib

        monkeypatch.setenv("AI_STREAM_INITIAL_TIMEOUT_S", "120.5")
        monkeypatch.setenv("AI_STREAM_CHUNK_TIMEOUT_S", "30")
        monkeypatch.setenv("AI_STREAM_TOTAL_TIMEOUT_S", "600")
        monkeypatch.setenv("AI_STREAM", "0")
        import config.settings as s
        importlib.reload(s)
        try:
            assert s.AI_STREAM_INITIAL_TIMEOUT_S == 120.5
            assert s.AI_STREAM_CHUNK_TIMEOUT_S == 30
            assert s.AI_STREAM_TOTAL_TIMEOUT_S == 600
            assert s.AI_STREAM is False
        finally:
            monkeypatch.delenv("AI_STREAM_INITIAL_TIMEOUT_S", raising=False)
            monkeypatch.delenv("AI_STREAM_CHUNK_TIMEOUT_S", raising=False)
            monkeypatch.delenv("AI_STREAM_TOTAL_TIMEOUT_S", raising=False)
            monkeypatch.delenv("AI_STREAM", raising=False)
            importlib.reload(s)
