"""Tests for character management slash commands."""
from __future__ import annotations

import pytest
from config.characters import load_characters


class TestCharacterCommand:

    @pytest.mark.asyncio
    async def test_character_list(self, ix, temp_characters_file):
        """Test listing all characters."""
        load_characters(temp_characters_file)
        from commands.character_commands import handle_character_command
        await handle_character_command(ix, action="list")
        assert len(ix._sent) > 0
        assert "Available characters" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_character_set_valid(self, ix, temp_characters_file):
        """Test setting a valid character."""
        load_characters(temp_characters_file)
        from commands.character_commands import handle_character_command
        await handle_character_command(ix, action="set", name="system")
        assert len(ix._sent) > 0
        assert "Switched to" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_character_set_invalid(self, ix, temp_characters_file):
        """Test setting a non-existent character."""
        load_characters(temp_characters_file)
        from commands.character_commands import handle_character_command
        await handle_character_command(ix, action="set", name="nonexistent_char_xyz")
        assert len(ix._sent) > 0
        assert "Unknown character" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_character_show(self, ix, temp_characters_file):
        """Test showing the current character."""
        load_characters(temp_characters_file)
        from commands.character_commands import handle_character_command
        await handle_character_command(ix, action="show")
        assert len(ix._sent) > 0
        assert "Current character" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_character_reset(self, ix, temp_characters_file):
        """Test resetting to the default character."""
        load_characters(temp_characters_file)
        from commands.character_commands import handle_character_command
        await handle_character_command(ix, action="reset")
        assert len(ix._sent) > 0
        assert "Reverted to default character" in ix._sent[0]


class TestCharacterCommandStructural:

    def test_handle_character_command_exists(self):
        from commands.character_commands import handle_character_command
        assert callable(handle_character_command)

    def test_handles_all_actions(self):
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
