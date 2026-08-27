"""Streaming AI response delivery for Discord (multi-message, freeze-on-split).

Accumulates streamed text chunks and mirrors them to the channel as one or
more Discord messages. Each message "grows" in place via progressive edits
(edits are rate-limited to ~5/s per channel, so updates are throttled to
~every MIN_FLUSH_CHARS / MAX_FLUSH_INTERVAL seconds).

When a reply outgrows Discord's 2000-char message limit it is split into
sections. The split points are deterministic (they depend only on the text
already received), which makes the freeze rule possible:

  - Message N streams in via edits, exactly like the first message.
  - The moment section N+1 begins, message N receives its final edit and is
    FROZEN — it is never edited again.
  - Section N+1 starts as a new message and streams in the same way.

So the user sees every part of the reply stream in live, top to bottom, and
no message is ever re-edited after the next one has appeared.

The caller defers the interaction first, so ``followup.send`` is available
and the streamed messages stay attached to the interaction.
"""
from __future__ import annotations

import logging
import os
import time

log = logging.getLogger("bot.stream")

MIN_FLUSH_CHARS: int = int(os.getenv("STREAM_MIN_FLUSH_CHARS", "100"))
MAX_FLUSH_INTERVAL: float = float(os.getenv("STREAM_MAX_FLUSH_INTERVAL", "1.5"))
DISCORD_MSG_LIMIT: int = 2000  # hard Discord limit (Create/Edit Message content)


def _split_for_first(text: str, limit: int) -> tuple[str, str]:
    """Split text into (head, rest) where head fits in `limit` chars.

    Prefers to break at a double-newline (paragraph boundary) or
    single-newline (line boundary) before falling back to a hard cut.
    The decision depends only on ``text[:limit]``, so once enough text has
    been received the split point is stable forever — this is what allows
    already-sent messages to be frozen without re-editing them.
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


def _greedy_split(text: str, limit: int = DISCORD_MSG_LIMIT) -> list[str]:
    """Split *text* into sections of at most ``limit`` chars.

    Every section except the last is "complete" (its boundary was chosen
    from a fully-received window), so it can be sent and frozen; the last
    section is still open and will keep growing via edits.
    """
    parts: list[str] = []
    while len(text) > limit:
        head, rest = _split_for_first(text, limit)
        if not head:  # defensive — never loop forever
            head, rest = text[:limit], text[limit:]
        parts.append(head)
        text = rest
    if text:
        parts.append(text)
    return parts


class StreamToDiscord:
    """Accumulate an async iterator of text chunks and mirror it to a Discord
    interaction as one or more progressively-edited messages.

    Invariant at any point in time:
      - ``_frozen_msgs[i]`` holds section i's final content (never edited again)
      - ``_active_msg``  displays the open (last) section, updated via edits
    """

    def __init__(self, interaction, *, ephemeral: bool = False) -> None:
        self._interaction = interaction
        self._ephemeral = ephemeral
        self._buffer: list[str] = []
        self._frozen_msgs: list = []   # completed sections — never touched again
        self._active_msg = None        # live message for the open section
        self._active_shown: str = ""   # content currently displayed on _active_msg
        self._last_edit_ts: float = 0.0
        self._closed = False

    # ── Public API ────────────────────────────────────────────────

    async def feed(self, chunk: str) -> None:
        """Ingest a chunk; freeze completed sections and grow the live one."""
        if not chunk:
            return
        self._buffer.append(chunk)
        full = "".join(self._buffer)
        parts = _greedy_split(full)
        open_part = parts[-1] if parts else ""

        # 1. Structural transitions (never throttled): freeze every section
        #    that is now complete, in order. The first edit a frozen message
        #    receives after the split is its LAST — it is never edited again.
        while len(self._frozen_msgs) < len(parts) - 1:
            if not await self._freeze_part(parts[len(self._frozen_msgs)]):
                return  # API hiccup — retry on the next chunk

        # 2. If a new section just opened, start its message right away so it
        #    can stream in live. Continuation sections start immediately even
        #    if small; the very first message still waits for MIN_FLUSH_CHARS
        #    (avoids posting a stub for one-word replies).
        if open_part and self._active_msg is None:
            if self._frozen_msgs or len(open_part) >= MIN_FLUSH_CHARS:
                await self._start_active(open_part)

        # 3. Throttled live growth of the active message (same cadence as
        #    before the multi-message redesign).
        await self._maybe_update_active(open_part, force=False)

    async def close(self) -> str:
        """Final flush + overflow handling. Returns the full reply text."""
        if self._closed:
            return "".join(self._buffer)
        self._closed = True
        full = "".join(self._buffer)
        parts = _greedy_split(full) if full else []
        open_part = parts[-1] if parts else ""

        # Freeze any still-open complete sections (no throttling).
        while len(self._frozen_msgs) < len(parts) - 1:
            if not await self._freeze_part(parts[len(self._frozen_msgs)]):
                break

        # Guarantee the open section has a message and holds its final text.
        if open_part and self._active_msg is None:
            await self._start_active(open_part)
        await self._maybe_update_active(open_part, force=True)

        log.info(
            "DELIVER stream complete: parts=%d total_len=%d chars",
            len(parts), len(full),
        )
        return full

    # ── Internal ──────────────────────────────────────────────────

    async def _freeze_part(self, final_content: str) -> bool:
        """Give the current live message its final content and freeze it.

        If no live message exists yet (e.g. the whole section arrived in a
        single chunk), the section is sent once — already complete — and
        frozen immediately. Returns False on API failure so the caller can
        retry on the next chunk without corrupting state.
        """
        if self._active_msg is not None:
            if self._active_shown != final_content:
                try:
                    await self._active_msg.edit(content=final_content)
                except Exception as e:
                    log.warning("Stream freeze-edit failed (will retry): %s", e)
                    return False
            msg = self._active_msg
        else:
            try:
                msg = await self._interaction.followup.send(
                    final_content, ephemeral=self._ephemeral
                )
            except Exception as e:
                log.warning("Stream freeze-send failed (will retry): %s", e)
                return False

        self._frozen_msgs.append(msg)
        self._active_msg = None
        self._active_shown = ""
        self._last_edit_ts = time.monotonic()
        log.info(
            "DELIVER stream part %d FROZEN (no further edits): msg_id=%s len=%d first=%r",
            len(self._frozen_msgs), getattr(msg, "id", None),
            len(final_content), final_content[:60],
        )
        return True

    async def _start_active(self, content: str) -> bool:
        """Start the live message for a newly opened section."""
        try:
            self._active_msg = await self._interaction.followup.send(
                content, ephemeral=self._ephemeral
            )
            self._active_shown = content
            self._last_edit_ts = time.monotonic()
            log.info(
                "DELIVER stream part %d STARTED (streaming via edits): msg_id=%s len=%d first=%r",
                len(self._frozen_msgs) + 1, getattr(self._active_msg, "id", None),
                len(content), content[:60],
            )
            return True
        except Exception as e:
            log.error("Failed to start stream part %d: %s", len(self._frozen_msgs) + 1, e)
            self._active_msg = None
            self._active_shown = ""
            return False

    async def _maybe_update_active(self, open_part: str, *, force: bool = False) -> None:
        """Edit the live message with the current section content if due."""
        if not open_part or self._active_msg is None:
            return
        if open_part == self._active_shown:
            return
        now = time.monotonic()
        grew = len(open_part) - len(self._active_shown)
        due = force or grew >= MIN_FLUSH_CHARS or (now - self._last_edit_ts) >= MAX_FLUSH_INTERVAL
        if not due:
            return
        try:
            await self._active_msg.edit(content=open_part)
            self._active_shown = open_part
            self._last_edit_ts = time.monotonic()
        except Exception as e:
            log.warning("Stream edit failed (will retry next flush): %s", e)


async def stream_ai_response(interaction, chunk_iter, *, ephemeral: bool = False) -> str:
    """Convenience wrapper: consume an async iterator of chunks and deliver
    them to a Discord interaction via progressive edits.

    Returns the full accumulated text (useful for logging / history).

    NOTE: ``close()`` must run in a ``finally`` so partial text is flushed on
    error — but it must NOT be *returned* from inside the ``finally`` block:
    a return there would swallow any exception raised by the chunk iterator
    (e.g. rate-limit / timeout / input-too-long), leaving the user with no
    feedback at all.
    """
    stream = StreamToDiscord(interaction, ephemeral=ephemeral)
    try:
        async for chunk in chunk_iter:
            await stream.feed(chunk)
    finally:
        # Final flush (best effort) — exceptions from the iterator must still
        # propagate to the caller after this runs.
        try:
            await stream.close()
        except Exception as e:  # pragma: no cover - defensive
            log.warning("Stream final flush failed: %s", e)
    return "".join(stream._buffer)
