"""Tests for the lightweight AI backend health probe (§3.9)."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from bot_core import health


class TestCheckBackendHealth:
    @pytest.mark.asyncio
    async def test_empty_url_is_unhealthy(self, monkeypatch):
        monkeypatch.setattr(health, "INFER_URL", "")
        ok, detail = await health.check_backend_health(timeout_sec=1)
        assert not ok
        assert "INFER_URL" in detail

    @pytest.mark.asyncio
    async def test_http_response_is_reachable(self, monkeypatch):
        monkeypatch.setattr(health, "INFER_URL", "http://backend.invalid/v1")

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, headers=None):
                assert url == "http://backend.invalid/v1/models"
                return httpx.Response(200)

        with patch.object(health.httpx, "AsyncClient", FakeClient):
            ok, detail = await health.check_backend_health(timeout_sec=1)

        assert ok
        assert "200" in detail

    @pytest.mark.asyncio
    async def test_timeout_is_unhealthy(self, monkeypatch):
        monkeypatch.setattr(health, "INFER_URL", "http://backend.invalid/v1")

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, headers=None):
                raise httpx.ReadTimeout("timed out")

        with patch.object(health.httpx, "AsyncClient", FakeClient):
            ok, detail = await health.check_backend_health(timeout_sec=1)

        assert not ok
        assert "ReadTimeout" in detail


class TestStartHealthProbe:
    @pytest.mark.asyncio
    async def test_initial_probe_is_spawned(self, monkeypatch):
        calls: list[str | None] = []

        async def fake_check(timeout_sec=None):
            calls.append(timeout_sec)
            return True, "OK"

        monkeypatch.setattr(health, "AI_HEALTH_CHECK_INTERVAL", 0)
        with patch.object(health, "check_backend_health", side_effect=fake_check):
            health.start_backend_health_probe(None)
            # Give the spawned task a chance to run.
            for _ in range(20):
                if calls:
                    break
                await asyncio.sleep(0.01)

        assert calls == [None]

    @pytest.mark.asyncio
    async def test_periodic_monitor_is_not_started_when_interval_is_zero(self, monkeypatch):
        monkeypatch.setattr(health, "AI_HEALTH_CHECK_INTERVAL", 0)

        async def fake_check(timeout_sec=None):
            return True, "OK"

        with patch.object(health, "check_backend_health", side_effect=fake_check), \
             patch.object(health, "_health_monitor") as mock_monitor:
            health.start_backend_health_probe(None)
            await asyncio.sleep(0)

        mock_monitor.assert_not_called()

    def test_repeated_starts_do_not_create_multiple_monitors(self, monkeypatch):
        class BotStub:
            pass

        bot = BotStub()
        monitor_tasks = []

        class TaskStub:
            def done(self):
                return False

        monkeypatch.setattr(health, "AI_HEALTH_CHECK_INTERVAL", 60)

        def fake_spawn(coro, *, name=None):
            coro.close()
            task = TaskStub()
            if name == "backend-health-monitor":
                monitor_tasks.append(task)
            return task

        with patch.object(health, "spawn_tracked_task", side_effect=fake_spawn):
            health.start_backend_health_probe(bot)
            health.start_backend_health_probe(bot)

        assert len(monitor_tasks) == 1
        assert getattr(bot, "_health_probe_task", None) is monitor_tasks[0]
