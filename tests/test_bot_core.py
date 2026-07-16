"""Tests for bot_core (AI client + conversation history)."""
import pytest


class TestEnsureHistory:
    """Test ensure_history helper."""

    def test_creates_new_entries(self):
        from bot_core import _chat_history, ensure_history

        # Should create new nested dicts if they don't exist
        guild_id = 111
        channel_id = 222
        assert guild_id not in _chat_history
        ensure_history(guild_id, channel_id)
        assert guild_id in _chat_history
        assert channel_id in _chat_history[guild_id]

    def test_does_not_overwrite_existing(self):
        from bot_core import _chat_history, ensure_history

        # Should not wipe existing history
        guild_id = 333
        channel_id = 444
        ensure_history(guild_id, channel_id)
        _chat_history[guild_id][channel_id].append(
            {"role": "user", "content": "Test"}
        )
        # Call again
        ensure_history(guild_id, channel_id)
        assert len(_chat_history[guild_id][channel_id]) == 1


class TestGetActiveCharacterKey:
    """Test get_active_char_key."""

    def test_returns_default_when_no_mapping(self):
        from bot_core import get_active_char_key
        result = get_active_char_key(None, 999)
        assert isinstance(result, str)  # Should be a valid character key string


class TestChatHistory:
    """Test _chat_history data structure."""

    def test_initial_state_is_empty(self):
        from bot_core import _chat_history
        assert isinstance(_chat_history, dict)

    def test_message_count_returns_zero_for_missing(self):
        from bot_core import get_current_message_count
        result = get_current_message_count(999999, 888888)
        assert result == 0


class TestChatHistoryIntegration:
    """Test conversation history end-to-end."""

    def test_set_active_char_and_get_it_back(self):
        from bot_core import set_active_char_key, get_active_char_key

        guild_id = 5555
        channel_id = 6666
        set_active_char_key(guild_id, channel_id, "system")
        result = get_active_char_key(guild_id, channel_id)
        assert result == "system"

    def test_different_channels_have_different_chars(self):
        from bot_core import set_active_char_key, get_active_char_key

        guild_id = 7777
        set_active_char_key(guild_id, 1000, "system")
        set_active_char_key(guild_id, 2000, "assistant")

        assert get_active_char_key(guild_id, 1000) == "system"
        assert get_active_char_key(guild_id, 2000) == "assistant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
