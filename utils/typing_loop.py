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
    end_at = asyncio.get_event_loop().time() + duration_sec
    while asyncio.get_event_loop().time() < end_at:
        try:
            await channel.typing()
        except TypeError:
            # Event loop already closed (shutting down) — stop immediately
            break
        except Exception:
            pass  # Permission error, channel deleted, etc.
        await asyncio.sleep(10)
