"""Tests for structured error classification (§3.7)."""
from __future__ import annotations

import httpx
import pytest

from bot_core.errors import AIError, BackendDownError, ModelNotFoundError, TimeoutError, classify_ai_error


class TestClassifyAIError:
    def test_timeout_message_is_classified(self):
        exc = RuntimeError("request timed out after 120s")
        classified = classify_ai_error(exc, model="gemma", backend_url="http://backend.invalid/v1")
        assert isinstance(classified, TimeoutError)
        assert classified.category == "timeout"
        assert "took too long" in classified.user_message

    def test_404_model_not_found(self):
        class FakeAPIError(Exception):
            response = type("Response", (), {"status_code": 404})

        exc = FakeAPIError("model not found")
        classified = classify_ai_error(exc, model="stale-model", backend_url="http://backend.invalid/v1")
        assert isinstance(classified, ModelNotFoundError)
        assert classified.model == "stale-model"
        assert "stale-model" in classified.user_message

    def test_400_model_not_found_message(self):
        class FakeAPIError(Exception):
            response = type("Response", (), {"status_code": 400})

        exc = FakeAPIError("invalid model name")
        classified = classify_ai_error(exc, model="bad-model")
        assert isinstance(classified, ModelNotFoundError)
        assert classified.category == "model_not_found"

    def test_connection_error_is_backend_down(self):
        exc = httpx.ConnectError("connection refused")
        classified = classify_ai_error(exc, model="gemma")
        assert isinstance(classified, BackendDownError)
        assert classified.category == "backend_down"

    def test_rate_limit_exception_is_returned(self):
        """A bot RateLimitError must be returned (not raised) — the classifier
        is a pure function; callers handle RateLimitError uniformly."""
        from bot_core.ai_client import RateLimitError

        exc = RateLimitError("user-1", retry_after=12)
        classified = classify_ai_error(exc)
        assert classified is exc

    def test_generic_error_is_preserved(self):
        exc = ValueError("unexpected provider behaviour")
        classified = classify_ai_error(exc, model="gemma")
        assert isinstance(classified, AIError)
        assert classified.category == "unknown"
        assert "unexpected provider behaviour" in str(classified)
        assert classified.cause is exc
