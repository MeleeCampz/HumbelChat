"""Utilities for retaining background asyncio tasks.

``asyncio.create_task`` returns a weak reference from the event loop's
perspective; if the only reference to the task disappears, the task can be
garbage-collected before it finishes. This helper keeps a strong reference in
a module-level set until the task completes or is cancelled.
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("bot.utils.background_tasks")

_ACTIVE_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _handle_done_task(task: asyncio.Task) -> None:
    """Discard the task and retrieve any exception so it is not silently lost."""
    _ACTIVE_BACKGROUND_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning("Background task %s failed: %s", task.get_name(), exc, exc_info=exc)


def spawn_tracked_task(coro, *, name: str | None = None) -> asyncio.Task:
    """Create a task and retain a strong reference until it finishes."""
    task = asyncio.create_task(coro, name=name)
    _ACTIVE_BACKGROUND_TASKS.add(task)
    task.add_done_callback(_handle_done_task)
    return task


__all__ = ["spawn_tracked_task", "_ACTIVE_BACKGROUND_TASKS"]
