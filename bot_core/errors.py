"""Structured AI error taxonomy (code review §3.7).

Distinguishes the error categories that matter for user-facing messages and
retry logic:

- ``TimeoutError``  — backend did not answer in time
- ``ModelNotFoundError`` — model slug does not exist on the backend
- ``BackendDownError`` — backend unreachable / connection error
- ``RateLimitError`` — already defined in ``bot_core.ai_client`` (429)

``classify_ai_error()`` inspects an arbitrary exception (from the ``openai``
SDK, ``httpx``, or any other source) and returns a concrete subclass of
:class:`AIError` with a user-friendly ``user_message``.
"""
from __future__ import annotations

import logging

log = logging.getLogger("bot.errors")


class AIError(Exception):
    """Base class for classified AI errors. Carries a user-friendly message."""

    #: Short category name, useful for logs/metrics.
    category: str = "unknown"

    def __init__(self, message: str, *, user_message: str | None = None, cause: BaseException | None = None):
        super().__init__(message)
        self.user_message = user_message or message
        self.cause = cause


class TimeoutError(AIError):
    category = "timeout"

    def __init__(self, message: str = "The AI backend took too long to respond.", **kw):
        kw.setdefault("user_message", "⏱️ The AI backend took too long to respond. Please try again.")
        super().__init__(message, **kw)


class ModelNotFoundError(AIError):
    category = "model_not_found"

    def __init__(self, model: str, backend_url: str = "", **kw):
        self.model = model
        self.backend_url = backend_url
        msg = f"Model '{model}' not found on the AI backend."
        kw.setdefault("user_message", f"❌ Model `{model}` is not available on the backend. Ask the bot admin to check the configuration.")
        super().__init__(msg, **kw)


class BackendDownError(AIError):
    category = "backend_down"

    def __init__(self, message: str = "The AI backend is unreachable.", **kw):
        kw.setdefault("user_message", "🔌 The AI backend is unreachable right now. Please try again in a minute.")
        super().__init__(message, **kw)


def classify_ai_error(exc: BaseException, *, model: str = "", backend_url: str = "") -> AIError:
    """Inspect *exc* and return a concrete :class:`AIError` subclass.

    Heuristics (checked in order):
      1. Already an ``AIError`` (or the bot's own ``RateLimitError``) → return as-is
         (wrapped if necessary).
      2. openai ``APITimeoutError`` / ``openai.APIConnectionError`` / message
         contains "timed out" → :class:`TimeoutError`
      3. HTTP 400 / 404 / "model not found" → :class:`ModelNotFoundError`
      4. ``httpx.ConnectError`` / message contains "connection" → :class:`BackendDownError`
      5. HTTP 429 → ``RateLimitError`` (re-raised from bot_core)
      6. Anything else → generic :class:`AIError`
    """
    # 1. Already classified (or the bot's own rate-limit exception)
    if isinstance(exc, AIError):
        return exc

    try:
        from bot_core.ai_client import RateLimitError

        if isinstance(exc, RateLimitError):
            raise exc
    except ImportError:
        pass

    error_msg = str(exc).lower()
    status_code = (
        getattr(getattr(exc, "response", None), "status_code", None)
        or getattr(exc, "status_code", None)
    )

    # 2. Timeouts
    _exc_type_name = type(exc).__name__
    if (
        _exc_type_name in ("APITimeoutError", "ReadTimeout", "ConnectTimeout", "Timeout")
        or "timed out" in error_msg
        or "timeout" in error_msg
    ):
        return TimeoutError(str(exc), cause=exc)

    # 3. Model-not-found
    if status_code in (400, 404) or "model not found" in error_msg or "does not exist" in error_msg:
        return ModelNotFoundError(model or "?", backend_url=backend_url, cause=exc)

    # 4. Backend down / connection
    if (
        _exc_type_name in ("APIConnectionError", "ConnectError", "ConnectionError", "RemoteProtocolError")
        or "connection" in error_msg
        or "unreachable" in error_msg
        or "refused" in error_msg
    ):
        return BackendDownError(str(exc), cause=exc)

    # 5. Rate limit (429) — surface as RateLimitError for handler consistency
    if status_code == 429:
        from bot_core.ai_client import RateLimitError
        raise RateLimitError("ai", retry_after=30)

    # 6. Fallback
    return AIError(str(exc), cause=exc)


__all__ = [
    "AIError",
    "TimeoutError",
    "ModelNotFoundError",
    "BackendDownError",
    "classify_ai_error",
]
