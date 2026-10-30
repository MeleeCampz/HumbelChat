"""utils.typing_loop — stop conditions (code review §2.6).

The docstring always promised "stops on Discord Forbidden / channel-gone
errors"; the old code swallowed every exception and kept retrying every 10 s
until the duration lapsed. These tests pin the promised behaviour.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import discord

from utils.typing_loop import typing_loop_task


class _BoomChannel:
    """Raises ``exc`` on every typing() call."""

    def __init__(self, exc: BaseException):
        self.calls = 0
        self.exc = exc

    async def typing(self):
        self.calls += 1
        raise self.exc


class _FlakyChannel:
    """1st call: transient error. 2nd: success. 3rd: raises CancelledError
    purely to end the (open-ended) loop deterministically without real sleeps."""

    def __init__(self) -> None:
        self.calls = 0

    async def typing(self):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("hub hiccup")
        if self.calls == 3:
            raise asyncio.CancelledError


class TestStopsOnDiscordErrors:
    async def test_forbidden_stops_after_one_attempt(self, monkeypatch):
        async def _no_sleep(_s):
            return

        monkeypatch.setattr("utils.typing_loop.asyncio.sleep", _no_sleep)
        # duration None would loop forever if Forbidden were swallowed.
        ch = _BoomChannel(discord.Forbidden(MagicMock(), {}))
        await asyncio.wait_for(typing_loop_task(ch, None), timeout=2)
        assert ch.calls == 1

    async def test_not_found_stops_after_one_attempt(self, monkeypatch):
        async def _no_sleep(_s):
            return

        monkeypatch.setattr("utils.typing_loop.asyncio.sleep", _no_sleep)
        ch = _BoomChannel(discord.NotFound(MagicMock(), {}))
        await asyncio.wait_for(typing_loop_task(ch, None), timeout=2)
        assert ch.calls == 1

    async def test_transient_errors_are_retried(self, monkeypatch):
        async def _no_sleep(_s):
            return

        monkeypatch.setattr("utils.typing_loop.asyncio.sleep", _no_sleep)
        ch = _FlakyChannel()
        await asyncio.wait_for(typing_loop_task(ch, None), timeout=2)
        # error → (retry) → success → success → intentional stop at call 3
        assert ch.calls == 3
