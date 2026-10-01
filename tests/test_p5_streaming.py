"""P5 regression tests: streaming /ai with a single idle timeout.

The 2026-09-24 failure: a 41 K-char prompt on a 27 B thinking model was
killed by the request timeout, but the SDK's *default* max_retries=3 turned
one "attempt" into 3×145 s of retries inside the client. The agreed fix:

  * exactly ONE retry when a request fails (bot-owned, transient only);
  * a single timeout knob AI_TIMEOUT_S (default 120 s) — maximum SILENCE
    from the backend. On a stream httpx applies it per read, so it acts as
    an idle watchdog: any arriving chunk resets it and an active generation
    may run indefinitely; there is deliberately NO total wall-clock cap;
  * streaming is the default delivery path (AI_STREAM=1);
  * overridable in the config (.env).
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


# ──────────────────── stream idle timeout (#9) ────────────────────


def _patch_timeout(monkeypatch, timeout_s: float):
    monkeypatch.setattr(settings, "AI_TIMEOUT_S", timeout_s)
    # ai_client imports the name at module import time — patch it there too.
    monkeypatch.setattr(ai_client, "AI_TIMEOUT_S", timeout_s)


class TestStreamIdleTimeout:
    @pytest.mark.asyncio
    async def test_success_persists_final_reply(self, monkeypatch):
        fake = _FakeStream([_stream_chunk("Hel"), _stream_chunk("lo", "stop")])
        client = _stub_ask_ai_env(monkeypatch, lambda **kw: fake)
        # No per-character budget → max_tokens must be omitted (model-max).
        monkeypatch.setattr(ai_client, "_resolve_request_params",
                            lambda char_obj: (None, 0.7))
        _patch_timeout(monkeypatch, 120.0)

        collected = []
        async for text in ai_client.ask_ai_stream(
            user_message="ping", model_slug="test-model",
            guild_id=7, channel_id=8, username="Alice", user_id=None,
        ):
            collected.append(text)

        assert collected == ["Hel", "Hello", "Hello"]  # progressive + final
        assert fake.closed is True                      # connection released
        # stream=True with the single AI_TIMEOUT_S as the HTTP (per-read /
        # idle) timeout, and NO max_tokens (model-max output).
        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["timeout"] == 120.0
        assert "max_tokens" not in kwargs

    @pytest.mark.asyncio
    async def test_stalled_stream_surfaces_friendly_timeout(self, monkeypatch):
        """A stalled stream surfaces as httpx's read timeout (the idle
        watchdog fires inside the SDK transport); ask_ai_stream must classify
        it into the friendly timeout error and release the connection."""
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
                # Simulate httpx's per-read idle timeout firing mid-stream.
                raise httpx.ReadTimeout(
                    "timed out", request=httpx.Request("GET", "http://backend.invalid/stream")
                )

            def close(self):
                self.closed = True

        fake = _StalledStream()
        _stub_ask_ai_env(monkeypatch, lambda **kw: fake)
        _patch_timeout(monkeypatch, 0.05)

        with pytest.raises(ValueError) as ei:
            async for _ in ai_client.ask_ai_stream(
                user_message="ping", model_slug="test-model",
                guild_id=9, channel_id=10, username="Alice", user_id=None,
            ):
                pass
        assert "too long to respond" in str(ei.value)
        assert fake.closed is True

    @pytest.mark.asyncio
    async def test_active_slow_stream_runs_to_completion(self, monkeypatch):
        """No total wall-clock cap: a slow-but-alive stream (every chunk well
        within the idle timeout) completes fully, however long it takes."""
        # 20 chunks × 0.01 s = 0.2 s of active generation — previously this
        # would have been measured against a total budget; now only silence
        # matters.
        fake = _FakeStream([_stream_chunk(f"t{i}") for i in range(20)], delay=0.01)
        _stub_ask_ai_env(monkeypatch, lambda **kw: fake)
        monkeypatch.setattr(ai_client, "_resolve_request_params",
                            lambda char_obj: (None, 0.7))
        _patch_timeout(monkeypatch, 60.0)

        collected = []
        async for text in ai_client.ask_ai_stream(
            user_message="ping", model_slug="test-model",
            guild_id=13, channel_id=14, username="Alice", user_id=None,
        ):
            collected.append(text)

        # 20 progressive yields + 1 final full-text yield.
        assert len(collected) == 21
        assert collected[-1] == "".join(f"t{i}" for i in range(20))
        assert fake.closed is True


# ─────────────────── config overrides ───────────────────


class TestConfigOverrides:
    def test_defaults_are_user_requested_values(self):
        # Single knob: maximum silence from the backend (default 120 s).
        assert settings.AI_TIMEOUT_S == 120.0
        assert settings.AI_STREAM is True                       # streaming on by default

    def test_env_override_parsed(self, monkeypatch):
        import importlib

        monkeypatch.setenv("AI_TIMEOUT_S", "300")
        monkeypatch.setenv("AI_STREAM", "0")
        import config.settings as s
        importlib.reload(s)
        try:
            assert s.AI_TIMEOUT_S == 300.0
            assert s.AI_STREAM is False
        finally:
            monkeypatch.delenv("AI_TIMEOUT_S", raising=False)
            monkeypatch.delenv("AI_STREAM", raising=False)
            importlib.reload(s)
