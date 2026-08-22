"""Streaming AI response delivery for Discord.

Accumulates streamed text chunks and periodically edits a single Discord
message so the user sees the reply "grow" in place — without hammering
Discord's API (edits are rate-limited to ~5/s per channel).

Design:
  - First visible update fires once ≥ MIN_FLUSH_CHARS of text has arrived
    (so we don't post an empty stub for a one-word reply).
  - Subsequent updates fire when the buffer exceeds MIN_FLUSH_CHARS OR
    MAX_FLUSH_INTERVAL seconds have elapsed since the last edit.
  - Final flush guarantees the complete text is posted.
  - If the final text exceeds Discord's 2000-char message limit, the
    remaining overflow is sent as follow-up messages (chunked).

The caller defers the interaction first, so ``followup.send`` is available
and the streamed message stays attached to the interaction.
"""
from __future__ import annotations

import logging
import os
import time

log = logging.getLogger("bot.stream")

MIN_FLUSH_CHARS: int = int(os.getenv("STREAM_MIN_FLUSH_CHARS", "100"))
MAX_FLUSH_INTERVAL: float = float(os.getenv("STREAM_MAX_FLUSH_INTERVAL", "1.5"))
DISCORD_MSG_LIMIT: int = 2000  # hard Discord limit


class StreamToDiscord:
    """Accumulate an async iterator of text chunks and mirror it to a
    Discord interaction as progressive edits of a single follow-up message."""

    def __init__(self, interaction, *, ephemeral: bool = False) -> None:
        self._interaction = interaction
        self._ephemeral = ephemeral
        self._buffer: list[str] = []
        self._first_msg = None  # discord.Message returned by followup.send
        self._last_edit_ts: float = 0.0
        self._sent_chars = 0
        self._closed = False

    # ── Public API ────────────────────────────────────────────────

    async def feed(self, chunk: str) -> None:
        """Ingest a chunk and flush if due."""
        if not chunk:
            return
        self._buffer.append(chunk)
        total = sum(len(c) for c in self._buffer)
        now = time.monotonic()

        should_flush = (
            total >= MIN_FLUSH_CHARS
            and (
                self._first_msg is None  # first send
                or total - self._sent_chars >= MIN_FLUSH_CHARS
                or (now - self._last_edit_ts) >= MAX_FLUSH_INTERVAL
            )
        )
        if should_flush:
            await self._flush(final=False)

    async def close(self) -> str:
        """Final flush + overflow handling. Returns the full reply text."""
        if self._closed:
            return "".join(self._buffer)
        self._closed = True
        await self._flush(final=True)
        return "".join(self._buffer)

    # ── Internal ──────────────────────────────────────────────────

    async def _send_first(self, text: str) -> None:
        msg = await self._interaction.followup.send(text, ephemeral=self._ephemeral)
        self._first_msg = msg
        self._sent_chars = len(text)
        self._last_edit_ts = time.monotonic()

    async def _edit_first(self, text: str) -> None:
        if self._first_msg is None:
            await self._send_first(text)
            return
        try:
            await self._first_msg.edit(content=text)
            self._sent_chars = len(text)
            self._last_edit_ts = time.monotonic()
        except Exception as e:
            log.warning("Stream edit failed (will retry next flush): %s", e)
            self._sent_chars = 0  # force retry

    async def _flush(self, final: bool) -> None:
        full_text = "".join(self._buffer)
        if not full_text:
            return

        if len(full_text) <= DISCORD_MSG_LIMIT:
            await self._edit_first(full_text)
            return

        # Overflow: split into a first part that fits + remainder.
        head, rest = self._split_for_first(full_text, DISCORD_MSG_LIMIT)
        await self._edit_first(head)
        self._sent_chars = len(head)

        if final:
            await self._send_overflow(rest)

    @staticmethod
    def _split_for_first(text: str, limit: int) -> tuple[str, str]:
        """Split text into (head, rest) where head fits in `limit` chars.

        Prefers to break at a double-newline (paragraph boundary) or
        single-newline (line boundary) before falling back to a hard cut.
        """
        if len(text) <= limit:
            return text, ""

        window = text[:limit]
        para_break = window.rfind("\n\n")
        if para_break > limit // 2:
            return text[:para_break], text[para_break + 2:]

        line_break = window.rfind("\n")
        if line_break > limit // 2:
            return text[:line_break], text[line_break + 1:]

        return window, text[limit:]

    async def _send_overflow(self, rest: str) -> None:
        """Send remaining text as chunked follow-up messages."""
        if not rest:
            return
        from utils.response_splitter import _split_long_message
        chunks = _split_long_message(rest, "")
        for idx, chunk in enumerate(chunks, start=1):
            try:
                await self._interaction.followup.send(chunk, ephemeral=self._ephemeral)
            except Exception as e:
                log.error("Failed to send overflow chunk %d: %s", idx, e)


async def stream_ai_response(interaction, chunk_iter, *, ephemeral: bool = False) -> str:
    """Convenience wrapper: consume an async iterator of chunks and deliver
    them to a Discord interaction via progressive edits.

    Returns the full accumulated text (useful for logging / history).
    """
    stream = StreamToDiscord(interaction, ephemeral=ephemeral)
    try:
        async for chunk in chunk_iter:
            await stream.feed(chunk)
    finally:
        return await stream.close()
