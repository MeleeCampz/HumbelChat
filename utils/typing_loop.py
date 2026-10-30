"""Typing-indicator task for long-pending AI responses."""
from __future__ import annotations
import asyncio
from typing import Any

import discord


async def typing_loop_task(channel: Any, duration_sec: float | None = 30) -> None:
    """Send typing indicators every ~10 s.

    Used while a deferred interaction is waiting on the AI backend.

    Args:
        channel: The Discord TextChannel to send typing indicators on.
        duration_sec: How long to keep sending typing. Defaults to 30 s.
            Pass ``None`` (or a non-positive value) to run *until the task is
            cancelled* — used by the streaming path where generation time is
            unbounded and the caller cancels the loop when the reply completes
            (P3 #24).

    The loop stops on Discord ``Forbidden``/channel-gone errors, on loop
    closure (shutting down), or when the duration elapses.
    """
    loop = asyncio.get_running_loop()
    if duration_sec is None or duration_sec <= 0:
        end_at = None
    else:
        end_at = loop.time() + duration_sec
    while end_at is None or loop.time() < end_at:
        try:
            await channel.typing()
        except (TypeError, asyncio.CancelledError):
            # Event loop already closed (shutting down) — stop immediately
            break
        except discord.Forbidden:
            # Typing permission revoked (or channel deleted under us) —
            # retrying every 10 s until the duration lapses was the old
            # behaviour; the docstring always promised a stop (code review §2.6).
            break
        except discord.NotFound:
            # Channel gone — nothing to signal on.
            break
        except Exception:
            pass  # transient transport hiccup — retry at the next interval
        await asyncio.sleep(10)
