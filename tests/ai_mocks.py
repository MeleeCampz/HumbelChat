"""Shared fakes for AI-client tests.

``bot_core.ai_client.complete_text`` consumes its completion as a STREAM
(openai ``AsyncStream``), so mocks must return stream-shaped fakes — a bare
``MagicMock`` would hang the suite forever: ``await MagicMock().__anext__()``
yields fresh AsyncMocks and never raises ``StopAsyncIteration``.

For non-streaming call sites (``ask_ai``, /ocr, /translate) plain response
mocks (``resp.choices[0].message.content``) remain correct.
"""
from __future__ import annotations

from unittest.mock import MagicMock


class FakeStream:
    """Minimal stand-in for openai's AsyncStream (``__anext__`` + ``close``)."""

    def __init__(self, chunks):
        self._it = iter(chunks)
        self.closed = False

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


def chunk(content: str | None, finish_reason: str | None = None) -> MagicMock:
    """One streamed delta: ``choices[0].delta.content`` + optional finish."""
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.delta.content = content
    return MagicMock(choices=[choice])
