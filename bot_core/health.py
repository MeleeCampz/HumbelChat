"""Lightweight AI-backend health probing (code review §3.9).

The bot targets any OpenAI-compatible backend. The probe performs a simple
``GET /models`` request and treats any HTTP response as evidence that the
backend is reachable; only network/timeout errors mark it as down. This lets
users see a clear startup warning instead of waiting 120+ seconds for their
first request to fail.

Set ``AI_HEALTH_CHECK_INTERVAL`` to a positive number of seconds to start a
periodic liveness monitor as well. By default (``0``), only one initial
probe is executed.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from config.settings import (
    INFER_URL,
    INFER_API_KEY,
    AI_HEALTH_CHECK_INTERVAL,
    AI_HEALTH_CHECK_TIMEOUT,
)
from utils.background_tasks import spawn_tracked_task

log = logging.getLogger("bot.health")


async def check_backend_health(timeout_sec: int | None = None) -> tuple[bool, str]:
    """Check whether the AI backend is reachable.

    Returns ``(ok, detail)``. Network timeouts and connection failures are
    considered unhealthy; any HTTP response — including 401/404 — is
    considered reachable.
    """
    if not INFER_URL:
        return False, "INFER_URL is not configured"

    timeout_sec = timeout_sec if timeout_sec is not None else AI_HEALTH_CHECK_TIMEOUT
    url = INFER_URL.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {INFER_API_KEY}"} if INFER_API_KEY else {}

    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.get(url, headers=headers)
        return True, f"HTTP {response.status_code}"
    except httpx.TimeoutException as exc:
        return False, f"timeout after {timeout_sec}s: {exc.__class__.__name__}"
    except httpx.HTTPError as exc:
        return False, f"connection error: {exc}"
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return False, f"unexpected probe error: {exc}"


async def _health_monitor(interval_sec: int) -> None:
    """Continuously probe the backend at the configured interval."""
    while True:
        ok, detail = await check_backend_health()
        if ok:
            log.debug("AI backend health check: OK (%s)", detail)
        else:
            log.warning("AI backend health check: DOWN (%s)", detail)
        await asyncio.sleep(interval_sec)


def start_backend_health_probe(bot: Any | None = None) -> None:
    """Spawn the initial health check and optionally start a periodic monitor.

    ``bot`` is used only to retain the periodic task on a well-known
    attribute (``bot._health_probe_task``) so repeated ``on_ready`` events do
    not start multiple monitors.
    """
    async def _initial_probe() -> None:
        ok, detail = await check_backend_health()
        if ok:
            log.info("AI backend health check at startup: OK (%s)", detail)
        else:
            log.warning(
                "AI backend health check at startup: DOWN (%s). "
                "Requests may fail until the backend at %s is reachable.",
                detail,
                INFER_URL,
            )

    spawn_tracked_task(_initial_probe(), name="backend-health-initial")

    interval = AI_HEALTH_CHECK_INTERVAL
    if interval <= 0:
        return

    if bot is not None:
        existing = getattr(bot, "_health_probe_task", None)
        if existing is not None and not existing.done():
            return
        task = spawn_tracked_task(
            _health_monitor(interval),
            name="backend-health-monitor",
        )
        setattr(bot, "_health_probe_task", task)
