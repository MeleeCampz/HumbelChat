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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

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


class AIBackendError(AIError):
    """Backend was reached but returned a malformed/empty response.

    Distinct from :class:`BackendDownError` (network unreachable): here the
    request completed, but ``choices`` was empty or the response object was
    missing — which used to crash ``resp.choices[0]`` with an unhelpful
    ``IndexError`` (P1 #8).
    """

    category = "backend_error"

    def __init__(self, message: str = "The AI backend returned an empty response.", **kw):
        kw.setdefault("user_message", "🤖 The AI backend returned an empty response. Please try again.")
        super().__init__(message, **kw)


class AIResponseTruncatedError(AIError):
    """The model's response was cut off by ``max_tokens`` before any answer.

    Thinking models (e.g. Qwen3) spend part of ``max_tokens`` on internal
    reasoning (returned in ``reasoning_content``) *before* the visible
    ``content``. When the budget is exhausted mid-reasoning the API still
    completes "successfully" — but with ``content == ""`` and
    ``finish_reason == "length"``. Treating that as an empty *answer*
    (a ``(empty response)`` placeholder) hid the real failure and polluted
    channel history with the placeholder. This error lets callers retry with
    a bigger budget and, if that still fails, surface a real message.
    """

    category = "truncated_response"

    def __init__(
        self,
        message: str = "The model used its full token budget on reasoning and produced no answer.",
        max_tokens: int | None = None,
        **kw,
    ):
        kw.setdefault(
            "user_message",
            "⚠️ The model ran out of response budget before answering "
            "(the request was too demanding for the configured max tokens). "
            "Please ask a narrower question or try again.",
        )
        super().__init__(message, **kw)
        self.max_tokens = max_tokens


def _is_truncated_empty_response(resp, content: str) -> bool:
    """True when *resp* is a max-tokens truncation with an empty answer.

    ``finish_reason == "length"`` means the backend cut the generation off at
    the token budget. If no visible content was produced, the whole budget
    went into reasoning (or a partial/empty answer) — that is a truncation
    failure, not a normal empty reply.
    """
    if content and content.strip():
        return False
    choices = getattr(resp, "choices", None)
    if not choices:
        return False
    finish = getattr(choices[0], "finish_reason", None)
    return finish == "length"


def extract_reply_text(resp, *, default: str = "(empty response)") -> str:
    """Safely return the first choice's message content from a completion.

    P1 #8: some backends return an empty ``choices`` array (or a ``None``
    response); ``resp.choices[0].message.content`` would raise an unhelpful
    ``IndexError``. This returns *default* when the message content is empty,
    and raises a friendly :class:`AIBackendError` when there is no choice at
    all so callers can surface a real error (and, for multi-model callers, try
    the next model).

    Budget-exhausted truncations are distinguished: when the response was cut
    off at ``max_tokens`` (``finish_reason == "length"``) and no visible
    content was produced, :class:`AIResponseTruncatedError` is raised instead
    of silently returning *default*. ``ask_ai`` catches that and retries once
    at ``MAX_TOKENS_HARD_CAP``; other callers surface a real error message
    instead of posting an empty placeholder.
    """
    if resp is None:
        raise AIBackendError("The AI backend returned no response.")
    choices = getattr(resp, "choices", None)
    if not choices:
        raise AIBackendError("The AI backend returned no choices.")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not content or not content.strip():
        if _is_truncated_empty_response(resp, content or ""):
            max_tokens = getattr(choices[0], "max_tokens", None)
            raise AIResponseTruncatedError(max_tokens=max_tokens)
        return default
    return content


def classify_ai_error(exc: BaseException, *, model: str = "", backend_url: str = "") -> AIError:
    """Inspect *exc* and return a concrete :class:`AIError` subclass.

    Heuristics (checked in order):
      1. Already an ``AIError`` (or the bot's own ``RateLimitError``) → return as-is
         (wrapped if necessary).
      2. openai ``APITimeoutError`` / ``openai.APIConnectionError`` / message
         contains "timed out" → :class:`TimeoutError`
      3. HTTP 400 / 404 / "model not found" → :class:`ModelNotFoundError`
      4. ``httpx.ConnectError`` / message contains "connection" → :class:`BackendDownError`
      5. HTTP 429 → ``RateLimitError`` carrying the backend's ``Retry-After``
         (parsed + clamped; see :func:`_parse_retry_after`)
      6. Anything else → generic :class:`AIError`
    """
    # 1. Already classified (or the bot's own rate-limit exception).
    # The classifier is a *pure* function: it never raises — callers decide
    # how to handle a returned RateLimitError.
    if isinstance(exc, AIError):
        return exc

    try:
        from bot_core.ai_client import RateLimitError

        if isinstance(exc, RateLimitError):
            return exc
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

    # 4b. HTTP 5xx — the backend answered, but its side failed. Surfaces a
    # friendly "try again" instead of a raw SDK exception (P1 #7).
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return BackendDownError(
            f"AI backend returned HTTP {status_code}.",
            user_message="🔌 The AI backend had a problem processing your request. Please try again in a moment.",
            cause=exc,
        )

    # 5. Rate limit (429) — return a RateLimitError so callers can handle it
    # uniformly. The classifier is a *pure* function (never raises). P1 #6:
    # honour the backend's Retry-After header instead of a hardcoded 30 s.
    if status_code == 429:
        from bot_core.ai_client import RateLimitError
        from config.settings import AI_RETRY_AFTER_FALLBACK_S
        raw = _extract_retry_after(exc)
        retry_after = _parse_retry_after(raw, fallback=AI_RETRY_AFTER_FALLBACK_S)
        if raw:
            log.info("429 retry-after: header=%r -> %ds", raw, retry_after)
        return RateLimitError("ai", retry_after=retry_after)

    # 6. Fallback
    return AIError(str(exc), cause=exc)


# P1 #6: sane bounds for a backend-supplied 429 Retry-After value.
_RETRY_AFTER_MIN_S = 5
_RETRY_AFTER_MAX_S = 120


def _extract_retry_after(exc: BaseException) -> str | None:
    """Pull the raw ``Retry-After`` header value out of a raised exception.

    Looks at ``exc.response.headers`` (openai SDK: ``APIStatusError.response``
    is an ``httpx.Response``) and falls back to an explicit ``retry_after``
    attribute. Returns ``None`` when the header is absent or not a string.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            val = headers.get("Retry-After")
        except Exception:
            val = None
        if isinstance(val, (str, bytes)):
            if isinstance(val, bytes):
                val = val.decode("latin-1")
            val = val.strip()
            if val:
                return val
    val = getattr(exc, "retry_after", None)
    if isinstance(val, str):
        return val.strip() or None
    return None


def _parse_retry_after(raw: str | None, *, fallback: int = 30) -> int:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) → seconds.

    P1 #6: the backend's value is honoured, clamped to
    ``[_RETRY_AFTER_MIN_S, _RETRY_AFTER_MAX_S]`` so a hostile/broken header
    can neither stall the bot (huge value) nor spam it (zero/negative).
    Malformed or missing values fall back to *fallback* (default 30 s).
    """
    if raw is None or not str(raw).strip():
        return int(fallback)
    val = str(raw).strip()
    try:
        seconds = float(val)  # delta-seconds form (also covers "5.0")
    except ValueError:
        seconds = None
    if seconds is None:
        # HTTP-date form, e.g. "Wed, 21 Oct 2026 07:28:00 GMT"
        try:
            when = parsedate_to_datetime(val)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            seconds = (when - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            seconds = None
    if seconds is None or seconds != seconds:  # NaN guard
        return int(fallback)
    return max(_RETRY_AFTER_MIN_S, min(_RETRY_AFTER_MAX_S, int(seconds)))


__all__ = [
    "AIError",
    "TimeoutError",
    "ModelNotFoundError",
    "BackendDownError",
    "AIBackendError",
    "AIResponseTruncatedError",
    "extract_reply_text",
    "classify_ai_error",
]
