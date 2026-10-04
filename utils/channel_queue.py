"""Per-channel FIFO serialization for outbound AI replies.

Why this exists
---------------
Discord orders messages by *send time*, not by request time. When two /ai
requests run concurrently in the same channel, an older request's later
messages (e.g. the overflow chunks of a >2000-char reply) can land AFTER a
newer request's first message — so the user sees responses interleaved:

    [req 1 part 1]   [req 2 reply]   [req 1 part 2]     ← wrong order

Every AI reply path (slash and prefix) must therefore hold this channel's
slot for the *entire* duration of its request
+ delivery. Requests are served strictly FIFO: a new request never cuts in
front of an in-flight one.

Design notes
------------
- One ``asyncio.Queue(maxsize=1)`` per channel id, pre-loaded with a single
  token. Holding the slot == holding the token.
- Hand-off is strictly FIFO: ``asyncio.Queue`` wakes waiting ``get()``
  coroutines in arrival order (internal getter deque), and the token is
  handed to exactly one waiter at a time. Cancellation while waiting is
  handled inside ``asyncio.Queue.get`` — a cancelled waiter simply drops
  out of line without corrupting the queue.
- One tiny Queue object per channel ever used; kept for process lifetime so
  a late ``put`` can never target a torn-down queue.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

log = logging.getLogger("bot.channel_queue")

# channel_id -> single-token FIFO queue (the token is the "channel slot")
# Per-channel FIFO token queues, created on first use and NEVER evicted
# (audit L10, deliberate): each entry is one tiny Queue for a channel the bot
# has replied in at least once — bounded by active channels, negligible
# memory. Any eviction scheme would have to race new _queue_for() callers or
# hand over a queue with waiters on it; that risk is worse than the leak.
_queues: dict[int, asyncio.Queue[None]] = {}


def _queue_for(channel_key: int) -> asyncio.Queue[None]:
    q = _queues.get(channel_key)
    if q is None:
        q = asyncio.Queue(maxsize=1)
        q.put_nowait(None)  # the token — channel starts free
        _queues[channel_key] = q
    return q


@asynccontextmanager
async def channel_slot(channel_id: int | None, *, name: str = "") -> AsyncIterator[None]:
    """Hold this channel's reply slot for the duration of the block.

    Grants are strictly FIFO in arrival order. ``channel_id=None`` falls
    back to a single shared global slot (defensive; DMs should still have a
    real channel id).

    Usage::

        async with channel_slot(channel_id, name="ai-command"):
            ... do the AI request AND deliver every message of the reply ...
    """
    key = channel_id if channel_id is not None else 0
    q = _queue_for(key)
    # Time the WAIT around q.get() itself — computing waited_s BEFORE the
    # await always logged ~0s, and deriving held_s from (started + waited_s)
    # measured queue+hold as "hold".
    wait_started = time.monotonic()
    await q.get()  # wait our turn (FIFO); we now own the channel slot
    waited_s = time.monotonic() - wait_started
    log.info(
        "channel_slot: channel=%s name=%r ACQUIRED after %.1fs wait (%d waiting behind)",
        key, name, waited_s, q.qsize(),
    )
    hold_started = time.monotonic()
    try:
        yield
    finally:
        held_s = time.monotonic() - hold_started
        log.info(
            "channel_slot: channel=%s name=%r RELEASED after %.1fs hold",
            key, name, held_s,
        )
        q.put_nowait(None)  # hand the slot to the next waiter (or leave free)


def reset_channel_queue() -> None:
    """Clear all bookkeeping (test helper)."""
    _queues.clear()
