"""Typing-indicator task for long-pending AI responses."""
from __future__ import annotations
import asyncio


async def typing_loop_task(channel, duration_sec: int = 30) -> None:
    """Send typing indicators every 10s for up to *duration_sec* seconds.
    
    Used when defer is called and we're waiting on the AI backend.
    Automatically stops when discord.Forbidden or after timeout.
    
    Args:
        channel: The Discord TextChannel to send typing indicators on.
        duration_sec: How long to keep sending typing (default 30s).
    """
    loop = asyncio.get_running_loop()
    end_at = loop.time() + duration_sec
    while loop.time() < end_at:
        try:
            await channel.typing()
        except (TypeError, asyncio.CancelledError):
            # Event loop already closed (shutting down) — stop immediately
            break
        except Exception:
            pass  # Permission error, channel deleted, etc.
        await asyncio.sleep(10)

