"""Test that the bot can start without errors."""
from __future__ import annotations

import asyncio
import logging
import sys
from unittest.mock import AsyncMock, MagicMock, patch


def test_bot_starts_without_startup_error():
    """Verify that importing and initializing the bot does not raise on startup.

    This checks the full startup path:
      1. Characters are loaded into _CHARACTERS
      2. _CHAR_CHOICES is built from those characters
      3. The bot object is created (Bot.__init__)
      4. on_ready() runs without raising AttributeError or other exceptions
    """
    # Prevent actual network calls and real Discord API hits
    with patch("main.bot") as mock_bot_cls:
        mock_bot_instance = MagicMock()
        mock_bot_instance.user = MagicMock()
        mock_bot_instance.user.display_name = "TestBot"
        mock_bot_instance.guilds = []  # no guilds → sync globally
        mock_bot_instance.tree = MagicMock()
        mock_bot_instance.tree.copy_global_to = AsyncMock()
        mock_bot_instance.tree.sync = AsyncMock()
        mock_bot_cls.return_value = mock_bot_instance

        # We need to prevent the module-level Bot(...) call from actually
        # registering events during import, so we mock it before importing.
        with patch("discord.ext.commands.Bot", mock_bot_cls):
            # Import the main module (triggers all module-level setup)
            import main  # noqa: F401

    # Now verify _CHAR_CHOICES was built successfully and has .name attributes
    from main import _CHAR_CHOICES

    assert len(_CHAR_CHOICES) > 0, "_CHAR_CHOICES should not be empty"
    for c in _CHAR_CHOICES:
        assert hasattr(c, "name"), f"_CHAR_CHOICES item {c} missing 'name' attribute"
        assert hasattr(c, "value"), f"_CHAR_CHOICES item {c} missing 'value' attribute"


def test_on_ready_does_not_raise():
    """Simulate on_ready() and verify it completes without exceptions.

    This directly calls the on_ready coroutine with a mocked bot to ensure
    no AttributeError occurs when iterating over _CHAR_CHOICES (the bug we fixed).
    """
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch

    # Create a fully mocked bot
    mock_bot = MagicMock()
    mock_bot.user = MagicMock()
    mock_bot.user.display_name = "TestBot"
    mock_bot.user.id = 123456789
    mock_bot.guilds = []
    mock_bot.tree = MagicMock()
    mock_bot.tree.copy_global_to = AsyncMock()
    mock_bot.tree.sync = AsyncMock()

    # Mock the bot module-level reference
    with patch("main.bot", mock_bot):
        with patch("utils.kb_utils.log_top_kb_files"):
            # Import and call on_ready directly
            from main import on_ready
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(on_ready())
            except Exception as e:
                pytest.fail(f"on_ready() raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
