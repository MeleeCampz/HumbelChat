"""Tests for per-channel FIFO reply serialization (utils/channel_queue.py).

Regression context: Discord orders messages by send time, not request time.
Two concurrent /ai requests in one channel could interleave their replies
(e.g. the overflow part of a long first reply landed after the second
request's reply). channel_slot() must serialize delivery per channel, FIFO.
"""
from __future__ import annotations

import asyncio

import pytest

from utils.channel_queue import channel_slot, reset_channel_queue


@pytest.fixture(autouse=True)
def _fresh_queue():
    reset_channel_queue()
    yield
    reset_channel_queue()


@pytest.mark.asyncio
async def test_fifo_ordering():
    """Three concurrent requests are served strictly in arrival order."""
    order: list[int] = []
    started = asyncio.Event()

    async def worker(idx: int, hold: float):
        async with channel_slot(100, name=f"w{idx}"):
            order.append(idx)
            if idx == 0:
                started.set()
            await asyncio.sleep(hold)

    t0 = asyncio.create_task(worker(0, 0.05))
    await started.wait()          # worker 0 now holds the slot
    t1 = asyncio.create_task(worker(1, 0.02))
    await asyncio.sleep(0.01)     # let w1 register before w2 (arrival order)
    t2 = asyncio.create_task(worker(2, 0.01))

    await asyncio.gather(t0, t1, t2)
    assert order == [0, 1, 2]


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_break_queue():
    """A head waiter cancelled mid-wait drops out; the next waiter still gets served."""
    events: list[str] = []

    async def waiter(name: str):
        try:
            async with channel_slot(200):
                events.append(name)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            events.append(f"{name}-cancelled")
            raise

    # Holder keeps the slot while w1 and w2 both queue up; w1 (the head
    # waiter) is cancelled before release. On release, the token must skip
    # past the cancelled w1 and land on w2.
    async with channel_slot(200):
        w1 = asyncio.create_task(waiter("w1"))
        await asyncio.sleep(0.02)   # w1 is now waiting (head of queue)
        w2 = asyncio.create_task(waiter("w2"))
        await asyncio.sleep(0.02)   # w2 is now waiting behind w1
        w1.cancel()
        await asyncio.gather(w1, return_exceptions=True)
    # outer block releases the slot here → should wake w2
    await asyncio.wait_for(w2, timeout=2.0)

    assert events == ["w1-cancelled", "w2"]


@pytest.mark.asyncio
async def test_different_channels_are_independent():
    """Holding channel A's slot does not block channel B."""
    async with channel_slot(300):
        # while 300 is busy, 301 must be immediately available
        got = []

        async def other_channel():
            async with channel_slot(301):
                got.append("b")

        t = asyncio.create_task(other_channel())
        await asyncio.wait_for(t, timeout=0.5)
        assert got == ["b"]


@pytest.mark.asyncio
async def test_slot_is_released_after_block():
    """Sequential acquisitions on the same channel all succeed."""
    for i in range(3):
        async with channel_slot(400):
            await asyncio.sleep(0.001)
