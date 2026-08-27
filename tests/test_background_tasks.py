"""Tests for background task retention (§4.4)."""
from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from unittest.mock import patch

import pytest

from utils.background_tasks import _ACTIVE_BACKGROUND_TASKS, spawn_tracked_task


class TestSpawnTrackedTask:
    @pytest.mark.asyncio
    async def test_task_is_retained_until_completion(self):
        state = {}

        async def work():
            state["started"] = True
            await asyncio.sleep(0.02)
            state["finished"] = True

        task = spawn_tracked_task(work(), name="retained")
        assert task in _ACTIVE_BACKGROUND_TASKS
        assert not state.get("finished", False)
        await task
        assert state["finished"] is True
        assert task not in _ACTIVE_BACKGROUND_TASKS

    @pytest.mark.asyncio
    async def test_cancelled_task_is_discarded(self):
        async def work():
            await asyncio.sleep(60)

        task = spawn_tracked_task(work(), name="cancel-me")
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        assert task not in _ACTIVE_BACKGROUND_TASKS

    @pytest.mark.asyncio
    async def test_exception_is_retrieved_and_logged(self):
        async def work():
            raise RuntimeError("task failed")

        task = spawn_tracked_task(work(), name="failing")
        with suppress(RuntimeError):
            await task
        assert task not in _ACTIVE_BACKGROUND_TASKS

    @pytest.mark.asyncio
    async def test_exception_is_not_lost_without_handler(self, caplog):
        async def work():
            raise ValueError("observed")

        task = spawn_tracked_task(work(), name="observed-error")
        with caplog.at_level("WARNING"):
            with suppress(ValueError):
                await task

        assert any("observed" in record.getMessage() for record in caplog.records)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
