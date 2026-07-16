"""Tests for clear history slash command."""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock

import pytest


class TestClearHistoryCommand:

    @pytest.mark.asyncio
    async def test_clear_history_clears_messages(self):
        """Test that clear history clears chat history."""
        from commands.clear_history_command import handle_clear_history_command
        from bot_core import _chat_history, get_current_message_count

        # Set up some history first
        guild_id = 123456
        channel_id = 789012
        _chat_history.setdefault(guild_id, {})
        _chat_history[guild_id][channel_id] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        # Verify history exists
        assert get_current_message_count(guild_id, channel_id) == 2

        sent = []

        async def on_send(content="", ephemeral=False):
            sent.append(str(content))

        response_mock = MagicMock()
        response_mock.send_message = AsyncMock(side_effect=on_send)

        ix = MagicMock()
        ix.followup.send = AsyncMock(side_effect=on_send)
        ix.response = response_mock
        ix.guild_id = guild_id
        ix.channel_id = channel_id

        await handle_clear_history_command(ix)

        # History should be cleared
        assert get_current_message_count(guild_id, channel_id) == 0
        assert len(sent) > 0

    @pytest.mark.asyncio
    async def test_clear_history_confirmation_message(self):
        """Test that clear history sends a confirmation message."""
        from commands.clear_history_command import handle_clear_history_command
        from bot_core import _chat_history

        guild_id = 123456
        channel_id = 789012
        _chat_history.setdefault(guild_id, {})
        _chat_history[guild_id][channel_id] = [
            {"role": "user", "content": "Hello"},
        ]

        sent = []
        async def on_send(content="", ephemeral=False):
            sent.append(str(content))

        response_mock = MagicMock()
        response_mock.send_message = AsyncMock(side_effect=on_send)

        ix = MagicMock()
        ix.followup.send = AsyncMock(side_effect=on_send)
        ix.response = response_mock
        ix.guild_id = guild_id
        ix.channel_id = channel_id

        await handle_clear_history_command(ix)

        assert len(sent) > 0
        assert "cleared" in sent[0].lower()


class TestClearHistoryStructural:

    def test_handle_clear_history_command_exists(self):
        from commands.clear_history_command import handle_clear_history_command
        assert callable(handle_clear_history_command)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
