"""Tests for AI chat slash command."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


class TestAICommand:

    @pytest.mark.asyncio
    async def test_ai_command_basic(self):
        """Test basic AI command execution."""
        from commands.ai_command import handle_ai_command
        from config.characters import load_characters

        # Ensure characters are loaded (mirrors main.py startup)
        load_characters(Path("characters.json.example"))

        ix = MagicMock()
        ix.guild_id = 123456
        ix.channel_id = 789012
        ix.response = MagicMock(is_done=MagicMock(return_value=True))
        ix.channel = MagicMock()
        ix.followup.send = AsyncMock()
        ix._sent = []

        async def fake_followup(content, **kw):
            ix._sent.append(str(content))

        ix.followup.send.side_effect = fake_followup

        mock_reply = ("AI reply", {})

        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock, return_value=mock_reply) as mock_ask:
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
        from config.characters import load_characters

        load_characters(Path("characters.json.example"))

        ix = MagicMock()
        ix.guild_id = 123456
        ix.channel_id = 789012
        ix.response = MagicMock(is_done=MagicMock(return_value=True))
        ix.channel = MagicMock()
        ix.followup.send = AsyncMock()
        ix._sent = []

        async def fake_followup(content, **kw):
            ix._sent.append(str(content))

        ix.followup.send.side_effect = fake_followup

        mock_reply = ("AI reply", {})

        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock, return_value=mock_reply) as mock_ask:
            await handle_ai_command(ix, message="Hello", character_name="system")

            mock_ask.assert_called_once()
            call_kwargs = mock_ask.call_args.kwargs
            assert call_kwargs["user_message"] == "Hello"

    @pytest.mark.asyncio
    async def test_ai_command_defers_response(self):
        """Test that the AI command properly defers to prevent timeout."""
        from commands.ai_command import handle_ai_command
        from config.characters import load_characters

        load_characters(Path("characters.json.example"))

        ix = MagicMock()
        ix.guild_id = 123456
        ix.channel_id = 789012
        ix.response = MagicMock(is_done=MagicMock(return_value=True))
        ix.channel = MagicMock()
        ix.followup.send = AsyncMock()
        ix._sent = []

        async def fake_followup(content, **kw):
            ix._sent.append(str(content))

        ix.followup.send.side_effect = fake_followup

        mock_reply = ("AI reply", {})

        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock, return_value=mock_reply):
            await handle_ai_command(ix, message="Hello")

        # Since we mocked response.is_done() to return True, defer should not be called
        assert ix.response.defer.call_count == 0


def _make_ix():
    """Build a mock Interaction whose followup.send / response.send_message record to _sent."""
    sent = []

    async def on_send(content="", **kw):
        sent.append(str(content))

    ix = MagicMock()
    ix.guild_id = 123456
    ix.channel_id = 789012
    ix.response = MagicMock(is_done=MagicMock(return_value=True))
    ix.channel = MagicMock()
    ix.followup.send.side_effect = on_send
    ix.response.send_message.side_effect = on_send
    ix._sent = sent
    return ix


class TestP1_9StaleActiveCharacter:
    """P1 #9: a *persisted* active character key that has since been deleted
    from characters.json must not break /ai with ``Character `None` not found``.
    """

    def _load_chars(self):
        from config.characters import load_characters
        load_characters(Path("characters.json.example"))  # default persona: "System"
        return "System"

    @pytest.mark.asyncio
    async def test_deleted_persisted_key_falls_back_to_default(self, monkeypatch):
        """No character named + stale persisted key -> default persona + a heads-up.

        This is the TODO's verify case. The old code rendered
        ``Character `None` not found`` (it interpolated ``character_name``) and
        hard-returned, so /ai was dead until the user re-picked a persona.
        """
        from commands import ai_command
        default_key = self._load_chars()
        ix = _make_ix()

        # The channel has a persisted active key pointing at a deleted char.
        monkeypatch.setattr(ai_command, "get_active_char_key",
                            lambda g, c: "DeletedCharacter")
        monkeypatch.setattr(ai_command._settings, "EMBED_FORMAT", False)
        mock_reply = ("Hello!", {})

        from commands.ai_command import handle_ai_command
        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock,
                   return_value=mock_reply) as mock_ask:
            await handle_ai_command(ix, message="hi")  # NOTE: no character_name

        # /ai must NOT have died: a real reply was produced with the default model.
        mock_ask.assert_awaited_once()
        assert mock_ask.call_args.kwargs["user_message"] == "hi"

        joined = " ".join(ix._sent)
        # No confusing "Character `None` not found".
        assert "`None`" not in joined
        assert "not found" not in joined
        # A fallback notice naming the stale key and the default persona.
        assert "DeletedCharacter" in joined
        assert "no longer exists" in joined
        assert default_key in joined
        # ...and the actual AI reply still went out.
        assert "Hello!" in joined

    @pytest.mark.asyncio
    async def test_explicit_missing_character_still_named(self, monkeypatch):
        """A character the user *typed* that doesn't exist -> named, not 'None'."""
        from commands import ai_command
        self._load_chars()
        ix = _make_ix()
        monkeypatch.setattr(ai_command, "get_active_char_key",
                            lambda g, c: "System")

        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock) as mock_ask:
            await ai_command.handle_ai_command(ix, message="hi", character_name="Gone")

        mock_ask.assert_not_awaited()
        joined = " ".join(ix._sent)
        assert "Gone" in joined           # the typed name is shown...
        assert "not found" in joined
        assert "`None`" not in joined     # ...and never the bare None

    @pytest.mark.asyncio
    async def test_normal_implicit_ai_no_fallback_notice(self, monkeypatch):
        """Regression guard: a *valid* active key (or none) produces no warning.

        The fallback notice must only appear when the persisted key is actually
        stale — never on a normal /ai.
        """
        from commands import ai_command
        self._load_chars()
        ix = _make_ix()
        monkeypatch.setattr(ai_command, "get_active_char_key",
                            lambda g, c: "System")  # a valid, existing key
        monkeypatch.setattr(ai_command._settings, "EMBED_FORMAT", False)

        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock,
                   return_value=("hi back", {})):
            await ai_command.handle_ai_command(ix, message="hi")

        joined = " ".join(ix._sent)
        assert "no longer exists" not in joined   # no spurious fallback warning
        assert "hi back" in joined


class TestAICommandStructural:

    def test_handle_ai_command_exists(self):
        from commands.ai_command import handle_ai_command
        assert callable(handle_ai_command)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
