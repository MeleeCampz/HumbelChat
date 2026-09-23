"""Regression tests for P1 #6 + #7: AI 429 Retry-After + transient retry.

P1 #6 — a 429 from the backend must carry its ``Retry-After`` header
       (parsed + clamped to [5, 120] s), with 30 s only as the no-header
       fallback.
P1 #7 — the *completion* call in ``ask_ai`` retries once (exponential
       backoff) on transient 5xx / timeout / connection failures and never
       retries 4xx auth/validation errors.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import formatdate
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from bot_core import ai_client
from bot_core import errors as bot_errors
from bot_core.ai_client import RateLimitError, _is_transient_api_failure
from bot_core.errors import _parse_retry_after


def _http_response(status_code: int, retry_after: str | None = None) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    req = httpx.Request("POST", "http://backend.invalid/v1/chat/completions")
    return httpx.Response(status_code, headers=headers, request=req)


def _openai_status_error(status_code: int, retry_after: str | None = None,
                         message: str = "error") -> openai.APIStatusError:
    """Build a *real* openai SDK exception carrying an httpx response."""
    cls = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        429: openai.RateLimitError,
    }.get(status_code, openai.InternalServerError)
    return cls(message, response=_http_response(status_code, retry_after), body=None)


# ─────────────────────── P1 #6: Retry-After parsing ───────────────────────

class TestParseRetryAfter:
    def test_delta_seconds(self):
        assert _parse_retry_after("5") == 5
        assert _parse_retry_after("42") == 42
        assert _parse_retry_after(" 7 ") == 7

    def test_clamped_to_min(self):
        assert _parse_retry_after("0") == 5
        assert _parse_retry_after("-3") == 5
        assert _parse_retry_after("1") == 5

    def test_clamped_to_max(self):
        assert _parse_retry_after("9999") == 120
        assert _parse_retry_after("121") == 120

    def test_malformed_falls_back(self):
        assert _parse_retry_after("abc", fallback=30) == 30
        assert _parse_retry_after("", fallback=30) == 30
        assert _parse_retry_after(None, fallback=30) == 30
        assert _parse_retry_after("12:34:56", fallback=30) == 30

    def test_http_date_form(self):
        # HTTP-date has 1 s resolution, so allow ±1 s of truncation.
        future = datetime.now(timezone.utc) + timedelta(seconds=60)
        http_date = formatdate(timeval=future.timestamp(), usegmt=True)
        assert 59 <= _parse_retry_after(http_date, fallback=30) <= 60

    def test_http_date_in_past_clamps_to_min(self):
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        http_date = formatdate(timeval=past.timestamp(), usegmt=True)
        assert _parse_retry_after(http_date, fallback=30) == 5

    def test_custom_fallback(self):
        assert _parse_retry_after(None, fallback=45) == 45


class TestExtractRetryAfter:
    def test_reads_header_from_sdk_response(self):
        exc = _openai_status_error(429, retry_after="12")
        assert bot_errors._extract_retry_after(exc) == "12"

    def test_absent_header_returns_none(self):
        exc = _openai_status_error(429)
        assert bot_errors._extract_retry_after(exc) is None

    def test_non_response_exception_returns_none(self):
        assert bot_errors._extract_retry_after(RuntimeError("x")) is None


class TestClassify429UsesHeader:
    def test_429_with_retry_after_5(self):
        exc = _openai_status_error(429, retry_after="5")
        classified = bot_errors.classify_ai_error(exc, model="gemma")
        assert isinstance(classified, RateLimitError)
        assert classified.retry_after == 5

    def test_429_malformed_header_uses_fallback(self):
        exc = _openai_status_error(429, retry_after="not-a-number")
        classified = bot_errors.classify_ai_error(exc, model="gemma")
        assert isinstance(classified, RateLimitError)
        assert classified.retry_after == 30  # AI_RETRY_AFTER_FALLBACK_S default

    def test_429_without_header_uses_fallback(self):
        exc = _openai_status_error(429)
        classified = bot_errors.classify_ai_error(exc, model="gemma")
        assert isinstance(classified, RateLimitError)
        assert classified.retry_after == 30

    def test_429_huge_header_clamped(self):
        exc = _openai_status_error(429, retry_after="3600")
        classified = bot_errors.classify_ai_error(exc, model="gemma")
        assert classified.retry_after == 120

    def test_429_honours_configured_fallback(self, monkeypatch):
        import config.settings as settings

        monkeypatch.setattr(settings, "AI_RETRY_AFTER_FALLBACK_S", 45)
        exc = _openai_status_error(429)
        classified = bot_errors.classify_ai_error(exc, model="gemma")
        assert classified.retry_after == 45


class TestClassify5xx:
    def test_500_is_backend_down(self):
        from bot_core.errors import BackendDownError

        exc = _openai_status_error(500, message="upstream exploded")
        classified = bot_errors.classify_ai_error(exc, model="gemma")
        assert isinstance(classified, BackendDownError)
        assert classified.category == "backend_down"
        assert "had a problem" in classified.user_message

    def test_502_is_backend_down(self):
        from bot_core.errors import BackendDownError

        exc = _openai_status_error(502)
        classified = bot_errors.classify_ai_error(exc, model="gemma")
        assert isinstance(classified, BackendDownError)


# ─────────────────── P1 #7: transient-failure detection ───────────────────

class TestIsTransientApiFailure:
    def test_5xx_is_transient(self):
        assert _is_transient_api_failure(_openai_status_error(500))
        assert _is_transient_api_failure(_openai_status_error(502))

    def test_429_is_transient(self):
        assert _is_transient_api_failure(_openai_status_error(429, retry_after="5"))

    def test_4xx_are_not_transient(self):
        assert not _is_transient_api_failure(_openai_status_error(400))
        assert not _is_transient_api_failure(_openai_status_error(401))
        assert not _is_transient_api_failure(_openai_status_error(404))

    def test_real_openai_exception_types(self):
        req = httpx.Request("POST", "http://backend.invalid/v1")
        assert _is_transient_api_failure(openai.APITimeoutError(request=req))
        assert _is_transient_api_failure(openai.APIConnectionError(request=req))
        assert _is_transient_api_failure(httpx.ConnectError("connection refused"))
        assert not _is_transient_api_failure(RuntimeError("unexpected shape"))


# ──────────────────── P1 #7: ask_ai completion retries ────────────────────

def _success_resp() -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="PONG"))]
    return resp


def _stub_ask_ai_env(monkeypatch, create_side_effect):
    """Point ask_ai at a stub client + empty KB. Returns (client, resp)."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="PONG"))]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    client.models.list = AsyncMock(return_value=MagicMock(data=[]))

    monkeypatch.setattr(ai_client, "_make_client", lambda: client)
    monkeypatch.setattr(ai_client, "_validate_model",
                        AsyncMock(return_value="test-model"))
    from kb import retrievers

    monkeypatch.setattr(retrievers, "retrieve_kb_documents",
                        AsyncMock(return_value=[]))
    return client, resp


class TestAskAiCompletionRetry:
    @pytest.mark.asyncio
    async def test_500_once_then_success_retries_and_succeeds(self, monkeypatch):
        """The TODO's canonical case: backend 500s once, second attempt works."""
        first = _openai_status_error(500, message="upstream exploded")
        client, resp = _stub_ask_ai_env(monkeypatch, [first, _success_resp()])
        with patch.object(ai_client, "_COMPLETION_RETRY_BACKOFF_S", 0.01):
            reply, _ = await ai_client.ask_ai(
                user_message="ping", model_slug="test-model",
                guild_id=1, channel_id=2, username="Alice", user_id=None,
            )

        assert reply == "PONG"
        assert client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_persistent_500_fails_after_single_retry(self, monkeypatch):
        exc = _openai_status_error(500, message="upstream exploded")
        client, _ = _stub_ask_ai_env(monkeypatch, exc)
        with patch.object(ai_client, "_COMPLETION_RETRY_BACKOFF_S", 0.01):
            with pytest.raises(Exception) as ei:
                await ai_client.ask_ai(
                    user_message="ping", model_slug="test-model",
                    guild_id=3, channel_id=4, username="Alice", user_id=None,
                )

        assert client.chat.completions.create.await_count == 2  # one retry only
        # Classified as backend-down → friendly ValueError (not a raw traceback).
        assert "had a problem" in str(ei.value)

    @pytest.mark.asyncio
    async def test_400_is_never_retried(self, monkeypatch):
        exc = _openai_status_error(400, message="invalid request body")
        client, _ = _stub_ask_ai_env(monkeypatch, exc)
        with patch.object(ai_client, "_COMPLETION_RETRY_BACKOFF_S", 0.01):
            with pytest.raises(Exception):
                await ai_client.ask_ai(
                    user_message="ping", model_slug="test-model",
                    guild_id=5, channel_id=6, username="Alice", user_id=None,
                )

        assert client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_timeout_retried_once(self, monkeypatch):
        req = httpx.Request("POST", "http://backend.invalid/v1")
        exc = openai.APITimeoutError(request=req)
        client, resp = _stub_ask_ai_env(monkeypatch, [exc, _success_resp()])
        with patch.object(ai_client, "_COMPLETION_RETRY_BACKOFF_S", 0.01):
            reply, _ = await ai_client.ask_ai(
                user_message="ping", model_slug="test-model",
                guild_id=7, channel_id=8, username="Alice", user_id=None,
            )

        assert reply == "PONG"
        assert client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_backoff_delays_are_exponential_and_bounded(self, monkeypatch):
        exc = _openai_status_error(503, message="service unavailable")
        client, _ = _stub_ask_ai_env(monkeypatch, exc)
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        with patch.object(ai_client, "asyncio") as mock_asyncio:
            mock_asyncio.sleep = fake_sleep
            with pytest.raises(Exception):
                await ai_client.ask_ai(
                    user_message="ping", model_slug="test-model",
                    guild_id=9, channel_id=10, username="Alice", user_id=None,
                )
        # one retry ⇒ exactly one sleep of the default base delay
        assert sleeps == [1.5]

    @pytest.mark.asyncio
    async def test_429_with_header_surfaces_header_retry_after(self, monkeypatch):
        """P1 #6 end-to-end: the raised 429 carries the header value (7 s)."""
        exc = _openai_status_error(429, retry_after="7")
        client, _ = _stub_ask_ai_env(monkeypatch, exc)
        with patch.object(ai_client, "_COMPLETION_RETRY_BACKOFF_S", 0.01):
            with pytest.raises(RateLimitError) as ei:
                await ai_client.ask_ai(
                    user_message="ping", model_slug="test-model",
                    guild_id=11, channel_id=12, username="Alice", user_id=None,
                )

        assert ei.value.retry_after == 7
        assert client.chat.completions.create.await_count == 2  # one retry, then surface
