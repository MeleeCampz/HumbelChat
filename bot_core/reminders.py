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


def _resolve_bot():
    """The running bot instance, or None if it is not usable right now.

    Kept separate from the diagnostic probe in :func:`_fire` so tests can
    monkeypatch exactly one thing.  Returns ``None`` for both "no bot"
    and "bot present but not ready" — only the error reason distinguishes.
    """
    try:
        from main import bot as _bot
    except Exception:
        return None
    return _bot if getattr(_bot, "is_ready", lambda: False)() else None

# How many failed delivery attempts are tolerated before a reminder is
# dropped.  Protects permanently-undeliverable reminders (e.g. a channel
# whose permissions were revoked after scheduling) from being retried on
# every restart/reconnect forever.
MAX_DELIVERY_ATTEMPTS = 3

# Transient failures: the bot instance itself is momentarily unavailable
# (gateway hiccup / resume in progress).  These get an IN-SESSION backoff
# retry so a reminder does not sit until the next restart just because the
# websocket blipped at the wrong second.  Hard channel errors (forbidden,
# not-found) do NOT retry — they will keep failing identically.
_RETRY_DELAYS: tuple[int, ...] = (30, 120)
# A reminder re-armed this far (or more) past its deadline is stale — the
# user isn't sitting by the channel for it anymore.  Drop instead of dump.
STALE_GRACE_SEC: int = 5 * 60
_retry_tasks: dict[str, asyncio.Task] = {}


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


def _resolve_bot():
    """The running logged-in bot, or None (see :func:`bot_core.channel_delivery.get_bot`).

    Delegates to the shared resolver so every delivery path uses ONE lookup
    with real diagnostics.  Kept as a separate function so tests can
    monkeypatch exactly one thing.
    """
    from bot_core.channel_delivery import get_bot
    return get_bot()


# ── Firing logic ─────────────────────────────────────────────────────────

async def _fire(rid: str, channel_id: int, message: str, delay: int) -> None:
    """Sleep then send the reminder message.

    Delivery uses :func:`bot_core.channel_delivery.send_to_channel` so a
    missing local cache entry (e.g. a private channel the bot cannot see)
    does not cause a silent skip — it falls back to a REST fetch and, on
    genuine failure, logs loudly and keeps the reminder un-fired (so a
    re-arm can retry it) instead of marking a delivery that never happened.
    """
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        log.info("Reminder %s cancelled", rid)
        return

    info = _reminders.get(rid)
    if info is not None and "fires_at" in info and \
            time.time() - info["fires_at"] > STALE_GRACE_SEC:
        # Re-armed far after the deadline (bot was down, or repeated delivery
        # failures).  The point of a timed reminder has passed — deliver it
        # anyway: users reported "it didn't work" when reminders silently
        # expired, and a late ping is strictly better than none.  Log loudly.
        log.warning(
            "Reminder %s is %.0f s overdue (fires_at=%d) — delivering late; "
            "channel: %s", rid, time.time() - info["fires_at"], int(info["fires_at"]),
            channel_id)

    from bot_core.channel_delivery import (
        ChannelNotDeliverableError, send_to_channel,
    )
    bot = _resolve_bot()
    if bot is None:
        # get_bot() has already logged WHY nothing usable was found
        # (module state, login state, import error + traceback).  The miss is
        # transient (startup / teardown) — backoff-retry in-session.
        _record_delivery_failure(rid, reason="no logged-in bot instance available to deliver")
        return
    try:
        await send_to_channel(bot, channel_id, f"⏰ **Reminder:** {message}")
    except ChannelNotDeliverableError as e:
        _record_delivery_failure(rid, reason=f"{e.reason}")
        return
    except Exception as e:
        log.exception("Unexpected error delivering reminder %s", rid)
        _record_delivery_failure(rid, reason=f"{type(e).__name__}: {e}", exc=e)
        return

    if rid in _reminders:
        _reminders[rid]["fired"] = True
        _save()
    log.info("Reminder %s fired and delivered (channel %s)", rid, channel_id)


def _is_transient(reason: str) -> bool:
    """True for failures caused by the process/connection being momentarily
    unavailable (worth an in-session retry), as opposed to channel-level
    errors (forbidden / not-found) that will keep failing identically.

    Connection-layer failure names (aiohttp/dis/httpx transport errors seen
    while the gateway is down or mid-reconnect) are matched by substring so
    library-version differences don't silently disable the retry."""
    r = reason.lower()
    return (
        r.startswith("no bot")           # no bot object / no bot instance
        or "no ready bot instance" in r  # legacy pre-fix wording
        or "no logged-in bot" in r       # current _fire wording
        or any(k in r for k in (
            "disconnected", "connection closed", "cannot connect",
            "not connected", "closed", "handshake", "reconnect",
        ))
    )


def _record_delivery_failure(rid: str, reason: str, exc: BaseException | None = None) -> None:
    """Count a failed delivery for ``rid``.

    Hard failures (forbidden / not-found / lookup errors) leave the reminder
    queued so a restart/reconnect can still try again, and it is dropped
    after MAX_DELIVERY_ATTEMPTS.  Transient bot-availability failures also get
    an IN-SESSION backoff retry (30s, then 120s), because waiting for a
    restart over a gateway blip would make reminders feel broken.
    """
    if rid not in _reminders:
        return  # cancelled meanwhile
    info = _reminders[rid]
    attempts = int(info.get("attempts", 0)) + 1
    info["attempts"] = attempts
    if attempts >= MAX_DELIVERY_ATTEMPTS:
        log.error(
            "Reminder %s DROPPED after %d failed delivery attempts — channel: %s, "
            "last error: %s. The user should re-issue the reminder in a channel "
            "the bot can post to.", rid, attempts,
            info.get("channel_id"), reason)
        if exc is not None:
            log.exception("Delivery traceback for dropped reminder %s", rid)
        del _reminders[rid]
    else:
        retry_note = (f" — will auto-retry in {_RETRY_DELAYS[min(attempts - 1, len(_RETRY_DELAYS) - 1)]}s"
                      if _is_transient(reason) else "")
        log.error(
            "Reminder %s NOT DELIVERED (attempt %d/%d): %s — keeping it queued; "
            "it will be retried on the next restart/reconnect.%s Fix the channel "
            "permissions (View Channel / Send Messages) if this keeps happening.",
            rid, attempts, MAX_DELIVERY_ATTEMPTS, reason, retry_note)
        if exc is not None:
            log.exception("Delivery traceback for reminder %s", rid)
    _save()
    if _is_transient(reason):
        _schedule_retry(rid, min(attempts - 1, len(_RETRY_DELAYS) - 1))


def _schedule_retry(rid: str, delay_index: int) -> None:
    """Schedule an in-session re-fire for a transiently-failed reminder."""
    info = _reminders.get(rid)
    if info is None:
        return
    delay = _RETRY_DELAYS[min(delay_index, len(_RETRY_DELAYS) - 1)]
    existing = _retry_tasks.pop(rid, None)
    if existing is not None and not existing.done():
        existing.cancel()

    async def _run() -> None:
        try:
            await asyncio.sleep(delay)
            if rid in _reminders and not _reminders[rid].get("fired"):
                await _fire(rid, info["channel_id"], info["message"], 0)
        except asyncio.CancelledError:
            log.info("Reminder %s retry cancelled", rid)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no event loop — the restart/reconnect re-arm covers it
    task = loop.create_task(_run())
    _retry_tasks[rid] = task
    task.add_done_callback(lambda _t, r=rid: _retry_tasks.pop(r, None))


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
    retry = _retry_tasks.pop(rid, None)
    if retry is not None:
        retry.cancel()
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
