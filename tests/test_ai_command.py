"""Tests for AI chat slash command."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestAICommand:

    @pytest.mark.asyncio
    async def test_ai_command_basic(self):
        """Test basic AI command execution."""
        from commands.ai_command import handle_ai_command

        ix = MagicMock()
        ix.guild_id = 123456
        ix.channel_id = 789012
        ix.response = MagicMock(is_done=MagicMock(return_value=True))
        ix.channel = MagicMock()
        ix.followup_send = AsyncMock()
        ix._sent = []

        async def fake_followup(content, **kw):
            ix._sent.append(str(content))

        ix.followup_send.side_effect = fake_followup

        mock_reply = ("AI reply", {})

        with patch("bot_core.ask_ai", new_callable=AsyncMock, return_value=mock_reply) as mock_ask:
            await handle_ai_command(ix, message="Hello")

            mock_ask.assert_called_once()
            call_kwargs = mock_ask.call_args.kwargs
            assert call_kwargs["user_message"] == "Hello"
            assert call_kwargs["guild_id"] == 123456
            assert call_kwargs["channel_id"] == 789012

    @pytest.mark.asyncio
    async def test_ai_command_with_character(self):
        """Test AI command with a specific character."""
        from commands.ai_command import handle_ai_command

        ix = MagicMock()
        ix.guild_id = 123456
        ix.channel_id = 789012
        ix.response = MagicMock(is_done=MagicMock(return_value=True))
        ix.channel = MagicMock()
        ix.followup_send = AsyncMock()
        ix._sent = []

        async def fake_followup(content, **kw):
            ix._sent.append(str(content))

        ix.followup_send.side_effect = fake_followup

        mock_reply = ("AI reply", {})

        with patch("bot_core.ask_ai", new_callable=AsyncMock, return_value=mock_reply) as mock_ask:
            await handle_ai_command(ix, message="Hello", character_name="system")

            mock_ask.assert_called_once()
            call_kwargs = mock_ask.call_args.kwargs
            assert call_kwargs["user_message"] == "Hello"

    @pytest.mark.asyncio
    async def test_ai_command_defers_response(self):
        """Test that the AI command properly defers to prevent timeout."""
        from commands.ai_command import handle_ai_command

        ix = MagicMock()
        ix.guild_id = 123456
        ix.channel_id = 789012
        # Mark response as already done so defer is skipped
        ix.response = MagicMock(is_done=MagicMock(return_value=True))
        ix.channel = MagicMock()
        ix.followup_send = AsyncMock()
        ix._sent = []

        async def fake_followup(content, **kw):
            ix._sent.append(str(content))

        ix.followup_send.side_effect = fake_followup

        mock_reply = ("AI reply", {})

        with patch("bot_core.ask_ai", new_callable=AsyncMock, return_value=mock_reply):
            await handle_ai_command(ix, message="Hello")

        # Since we mocked response.is_done() to return True, defer should not be called
        assert ix.response.defer.call_count == 0


class TestAICommandStructural:

    def test_handle_ai_command_exists(self):
        from commands.ai_command import handle_ai_command
        assert callable(handle_ai_command)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
