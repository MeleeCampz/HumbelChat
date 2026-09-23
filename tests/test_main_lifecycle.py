"""P1 #11 + #12: main.py application-command error handler + graceful shutdown.

P1 #11 — an unhandled exception in a slash command body used to surface as
        Discord's generic "Application command failed" popup with no log trail.
        ``bot.tree.on_error`` (``on_app_command_error``) now logs the traceback
        and sends a short, friendly, *ephemeral* message.
P1 #12 — no graceful-close hook existed; ``kb.retrievers.shutdown_vector_store``
        was defined but never called. ``on_shutdown`` now flushes the vector
        store, stops typing loops, and cancels tracked background tasks.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from discord import app_commands


class _FakeCommand:
    """Minimal stand-in for an app_commands.Command (only needs .name)."""
    def __init__(self, name="explode"):
        self.name = name


def _ix(done: bool):
    ix = MagicMock()
    ix.response.is_done = MagicMock(return_value=done)
    ix.response.send_message = AsyncMock()
    ix.followup.send = AsyncMock()
    return ix


class TestOnAppCommandError:

    @pytest.mark.asyncio
    async def test_unhandled_exception_is_friendly_and_ephemeral(self):
        """The TODO's case: a body exception (CommandInvokeError) -> friendly,
        names the original cause, ephemeral, and leaks no raw traceback."""
        import main
        err = app_commands.CommandInvokeError(_FakeCommand("explode"), ValueError("boom"))

        ix = _ix(done=True)  # already responded -> use followup
        await main.on_app_command_error(ix, err)

        ix.followup.send.assert_awaited_once()
        text = ix.followup.send.call_args.args[0]
        assert ix.followup.send.call_args.kwargs.get("ephemeral") is True
        assert "ValueError" in text and "boom" in text
        assert "Traceback" not in text

    @pytest.mark.asyncio
    async def test_check_failure_is_permission_message(self):
        import main
        ix = _ix(done=True)
        await main.on_app_command_error(ix, app_commands.CheckFailure())
        text = ix.followup.send.call_args.args[0]
        assert "permission" in text.lower()

    @pytest.mark.asyncio
    async def test_signature_mismatch_is_generic(self):
        import main
        ix = _ix(done=True)
        await main.on_app_command_error(ix, app_commands.CommandSignatureMismatch(_FakeCommand("foo")))
        text = ix.followup.send.call_args.args[0]
        assert "internal error" in text

    @pytest.mark.asyncio
    async def test_primary_response_used_when_not_done(self):
        """If the interaction hasn't been answered yet, use the primary
        (ephemeral) response instead of a follow-up."""
        import main
        ix = _ix(done=False)
        await main.on_app_command_error(ix, app_commands.CheckFailure())
        ix.response.send_message.assert_awaited_once()
        ix.followup.send.assert_not_awaited()
        assert ix.response.send_message.call_args.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_falls_back_to_followup_if_primary_send_fails(self):
        import main
        ix = _ix(done=False)
        ix.response.send_message = AsyncMock(side_effect=RuntimeError("4011: unknown interaction"))
        await main.on_app_command_error(ix, app_commands.CheckFailure())
        ix.followup.send.assert_awaited_once()


class TestOnShutdown:

    @pytest.mark.asyncio
    async def test_on_shutdown_flushes_vector_store(self):
        """The TODO's case: on_shutdown awaits the vector store shutdown
        (which was previously dead code, never called)."""
        import main
        with patch("kb.retrievers.shutdown_vector_store", new=AsyncMock()) as shut:
            await main.on_shutdown()
        shut.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_shutdown_cancels_tracked_tasks(self, monkeypatch):
        import main
        import utils.background_tasks as bt

        bg_task = MagicMock()
        bg_task.done = MagicMock(return_value=False)
        bg_task.cancel = MagicMock()
        monkeypatch.setattr(bt, "_ACTIVE_BACKGROUND_TASKS", {bg_task})

        typing_task = MagicMock()
        typing_task.done = MagicMock(return_value=False)
        typing_task.cancel = MagicMock()
        monkeypatch.setattr(main.bot, "typing_tasks", [typing_task])

        with patch("kb.retrievers.shutdown_vector_store", new=AsyncMock()):
            await main.on_shutdown()

        bg_task.cancel.assert_called_once()
        typing_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_shutdown_skips_done_tasks(self, monkeypatch):
        import main
        import utils.background_tasks as bt

        done_task = MagicMock()
        done_task.done = MagicMock(return_value=True)
        done_task.cancel = MagicMock()
        monkeypatch.setattr(bt, "_ACTIVE_BACKGROUND_TASKS", {done_task})
        monkeypatch.setattr(main.bot, "typing_tasks", [])

        with patch("kb.retrievers.shutdown_vector_store", new=AsyncMock()):
            await main.on_shutdown()

        done_task.cancel.assert_not_called()
