"""Persistent reminder store with re-arm on startup.

Reminders survive bot restarts via a JSON file (data/reminders.json).
Call ``rearm_pending_reminders()`` from ``on_ready`` to re-schedule
any that were pending before a crash/restart.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import threading
import time
import uuid

log = logging.getLogger("bot.reminders")

# ── Disk layout ──────────────────────────────────────────────────────────
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _REPO_ROOT / "data" / "reminders.json"

_store_path: pathlib.Path | None = None
_lock = threading.Lock()

# rid -> dict  (in-memory cache of the file)
_reminders: dict[str, dict] = {}

# rid -> live asyncio.Task  (not persisted)
_tasks: dict[str, asyncio.Task] = {}


# ── Path resolution ──────────────────────────────────────────────────────

def _resolve_path() -> pathlib.Path | None:
    global _store_path
    if _store_path is not None:
        return _store_path if _store_path else None
    env = os.environ.get("REMINDERS_PERSIST_FILE")
    if env is not None:
        _store_path = pathlib.Path(env) if env else None
        return _store_path if _store_path else None
    return _DEFAULT_PATH


# ── Persistence ──────────────────────────────────────────────────────────

def _save() -> None:
    path = _resolve_path()
    if path is None:
        return
    with _lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(_reminders, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, path)
        except Exception as e:
            log.warning("Failed to persist reminders: %s", e)


def _load() -> None:
    path = _resolve_path()
    if path is None or not path.exists():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _reminders.clear()
        _reminders.update(raw)
        log.info("Loaded %d persisted reminder(s)", len(_reminders))
    except Exception as e:
        log.warning("Could not load reminders file: %s", e)


# ── Firing logic ─────────────────────────────────────────────────────────

async def _fire(rid: str, channel_id: int, message: str, delay: int) -> None:
    """Sleep then send the reminder message."""
    try:
        from main import bot as _bot  # lazy: avoid circular import
    except Exception:
        log.warning("Cannot import bot for reminder %s", rid)
        return
    try:
        await asyncio.sleep(delay)
        chan = _bot.get_channel(channel_id)
        if chan:
            await chan.send(f"⏰ **Reminder:** {message}")
        if rid in _reminders:
            _reminders[rid]["fired"] = True
            _save()
        log.info("Reminder %s fired", rid)
    except asyncio.CancelledError:
        log.info("Reminder %s cancelled", rid)
    except Exception as e:
        log.error("Failed to send reminder %s: %s", rid, e)


# ── Public API ───────────────────────────────────────────────────────────

def schedule_reminder(channel_id: int, message: str, delay_sec: int) -> str:
    """Create a persisted reminder and start its background task.

    Returns the reminder id.  Call from an async context.
    """
    rid = uuid.uuid4().hex[:12]
    _reminders[rid] = {
        "channel_id": channel_id,
        "message": message,
        "delay_sec": delay_sec,
        "fires_at": time.time() + delay_sec,
        "created_at": time.time(),
        "fired": False,
    }
    _save()
    _start_task(rid, delay_sec, channel_id, message)
    return rid


def cancel_reminder(rid: str) -> bool:
    """Cancel and remove a reminder.  Returns True if it existed."""
    existed = rid in _reminders
    _reminders.pop(rid, None)
    task = _tasks.pop(rid, None)
    if task is not None:
        task.cancel()
    if existed:
        _save()
    return existed


def list_reminders() -> list[dict]:
    """Return all non-fired reminders."""
    return [r for r in _reminders.values() if not r.get("fired")]


def _start_task(rid: str, delay: int, channel_id: int, message: str) -> None:
    """Spawn the asyncio task for a reminder (must be in a running loop)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no event loop — task won't start (tests, CLI, etc.)
    task = loop.create_task(_fire(rid, channel_id, message, delay))
    _tasks[rid] = task
    task.add_done_callback(lambda _t: _tasks.pop(rid, None))


def rearm_pending_reminders() -> int:
    """Load persisted reminders and re-arm all non-fired ones.

    Call from ``on_ready`` (must be in a running event loop).
    Returns the number of reminders re-armed.

    ``on_ready`` fires again on full gateway reconnects, so this may run more
    than once: any existing live task for a reminder is cancelled before it is
    replaced, otherwise both tasks would fire and the user would be pinged twice.
    """
    _load()
    now = time.time()
    count = 0
    for rid, info in list(_reminders.items()):
        if info.get("fired"):
            continue
        existing = _tasks.get(rid)
        if existing is not None and not existing.done():
            existing.cancel()
        delay = max(int(info["fires_at"] - now), 0)
        _start_task(rid, delay, info["channel_id"], info["message"])
        count += 1
    log.info("Re-armed %d pending reminder(s)", count)
    return count
