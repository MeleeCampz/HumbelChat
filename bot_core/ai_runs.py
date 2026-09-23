"""Registry of in-flight /ai runs, keyed by channel (P3 #25).

Lets ``/ai stop`` cancel the generation currently in progress for a channel
without touching discord.py's command-task lifecycle. The /ai handler runs its
AI call + delivery in a *child* task and registers it here; ``/ai stop`` looks
the channel's task up and cancels just that child task, so the parent command
task stays healthy and can acknowledge the stop cleanly.

A channel has at most one in-flight run because the /ai path already holds
that channel's reply slot (utils.channel_queue) for the whole turn, so the
per-channel key is unambiguous.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger("bot.ai_runs")

# channel_key (int; 0 for DMs / no channel) -> in-flight /ai run task
_ACTIVE_RUNS: dict[int, asyncio.Task[Any]] = {}


def register_run(channel_key: int, task: asyncio.Task[Any]) -> None:
    """Record the in-flight run for *channel_key* (overwrites any prior one)."""
    _ACTIVE_RUNS[channel_key] = task


def get_run(channel_key: int) -> asyncio.Task[Any] | None:
    """Return the in-flight run for *channel_key*, or None."""
    return _ACTIVE_RUNS.get(channel_key)


def cancel_run(channel_key: int) -> bool:
    """Cancel + forget the in-flight run for *channel_key*.

    Returns True if a live run was cancelled, False if there was none.
    """
    task = _ACTIVE_RUNS.pop(channel_key, None)
    if task is not None and not task.done():
        log.info("ai_runs: cancelling in-flight run for channel %s", channel_key)
        task.cancel()
        return True
    return False


def clear_run(channel_key: int, task: asyncio.Task[Any] | None = None) -> None:
    """Remove *task* from the registry, but only if it is still the current one.

    Passing *task* guards against a late ``finally`` clearing a *newer* run that
    was registered after the old one finished (a rapid stop→restart).
    """
    cur = _ACTIVE_RUNS.get(channel_key)
    if task is not None:
        if cur is task:
            _ACTIVE_RUNS.pop(channel_key, None)
    else:
        _ACTIVE_RUNS.pop(channel_key, None)


def _reset() -> None:
    """Test helper: forget every in-flight run."""
    _ACTIVE_RUNS.clear()


__all__ = [
    "register_run",
    "get_run",
    "cancel_run",
    "clear_run",
    "_reset",
]
