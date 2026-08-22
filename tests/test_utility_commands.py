"""Tests for utility commands: /remind, /ocr, /summarize, /translate."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRemindCommand:

    @pytest.mark.asyncio
    async def test_remind_valid_seconds(self, ix):
        """Test reminder with seconds unit (delay >= 10)."""
        from commands.utility_commands import handle_remind_command
        ix.channel = MagicMock()
        ix.channel.id = 123456

        await handle_remind_command(ix, time_value=10, time_unit="seconds", message="Test reminder")

        assert len(ix._sent) > 0
        assert "Reminder set for" in ix._sent[0]
        assert "10 second" in ix._sent[0]
        assert "Test reminder" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_remind_invalid_unit(self, ix):
        """Test that invalid time unit returns error."""
        from commands.utility_commands import handle_remind_command
        await handle_remind_command(ix, time_value=5, time_unit="days", message="Test")
        assert "Unknown unit" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_remind_too_short(self, ix):
        """Test that reminders under 10 seconds are rejected."""
        from commands.utility_commands import handle_remind_command
        await handle_remind_command(ix, time_value=5, time_unit="seconds", message="Test")
        assert "at least 10 seconds" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_remind_minutes(self, ix):
        """Test reminder with minutes unit."""
        from commands.utility_commands import handle_remind_command
        ix.channel = MagicMock()
        ix.channel.id = 123456
        await handle_remind_command(ix, time_value=2, time_unit="minutes", message="Meeting")
        assert "Reminder set for" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_remind_hours(self, ix):
        """Test reminder with hours unit."""
        from commands.utility_commands import handle_remind_command
        ix.channel = MagicMock()
        ix.channel.id = 123456
        await handle_remind_command(ix, time_value=1, time_unit="hour", message="Dinner")
        assert "Reminder set for" in ix._sent[0]


class TestOCRCommand:

    @pytest.mark.asyncio
    async def test_ocr_with_image(self, ix):
        """Test OCR with valid image attachment."""
        from commands.utility_commands import handle_ocr_command

        image = MagicMock()
        image.url = "https://example.com/image.png"
        image.filename = "test_image.png"
        image.read = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n")

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Extracted text here"))]

        with patch("commands.utility_commands._make_client") as MockClient:
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(return_value=mock_resp)
            MockClient.return_value = inst
            await handle_ocr_command(ix, image=image)

        assert len(ix._sent) > 0
        assert "Extracted text" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_ocr_no_image(self, ix):
        """Test OCR with no image attachment."""
        from commands.utility_commands import handle_ocr_command
        await handle_ocr_command(ix, image=None)
        assert "Please attach an image" in ix._sent[0]


class TestSummarizeCommand:

    @pytest.mark.asyncio
    async def test_summarize_with_history(self, ix):
        """Test summarizing recent chat history (no file_url)."""
        from commands.utility_commands import handle_summarize_command
        from bot_core.history import _chat_history

        gid = 4441237890
        cid = 5552348901
        ix.guild_id = gid
        ix.channel_id = cid
        _chat_history.setdefault(gid, {})[cid] = [
            {"role": "user", "content": "Hello AI"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Summary text"))]

        with patch("commands.utility_commands._make_client") as MockClient:
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(return_value=mock_resp)
            MockClient.return_value = inst
            await handle_summarize_command(ix)

        assert len(ix._sent) > 0
        assert "Summary" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_summarize_empty_history(self, ix):
        """Test summarizing with no history."""
        from commands.utility_commands import handle_summarize_command
        await handle_summarize_command(ix)
        assert "Nothing to summarize" in ix._sent[0]


class TestTranslateCommand:

    @pytest.mark.asyncio
    async def test_translate_with_explicit_text(self, ix):
        """Test translation with explicit source text."""
        from commands.utility_commands import handle_translate_command

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Traducción"))]

        with patch("commands.utility_commands._make_client") as MockClient:
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(return_value=mock_resp)
            MockClient.return_value = inst
            await handle_translate_command(ix, target_language="Spanish: Hello world", source_language="English")

        assert len(ix._sent) > 0
        assert "Translated" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_translate_no_text_available(self, ix):
        """Test translation with no text (no history, no explicit text)."""
        from commands.utility_commands import handle_translate_command
        await handle_translate_command(ix, target_language="Spanish")
        assert "No text to translate" in ix._sent[0]


class TestUtilityActiveCharacter:
    """Code review §1.8: utility commands must honour the active character's model."""

    @pytest.mark.asyncio
    async def test_ocr_uses_active_character_model(self, ix, monkeypatch):
        from commands import utility_commands as uc

        monkeypatch.setattr(uc, "_validated_utility_model",
                            AsyncMock(return_value="char-specific-model"))

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Text"))]

        with patch("commands.utility_commands._make_client") as MockClient:
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(return_value=mock_resp)
            MockClient.return_value = inst
            await uc.handle_ocr_command(ix, image=_fake_image("scan.png"))

        assert inst.chat.completions.create.awaited
        _, kwargs = inst.chat.completions.create.call_args
        assert kwargs["model"] == "char-specific-model"

    @pytest.mark.asyncio
    async def test_translate_uses_active_character_model(self, ix, monkeypatch):
        from commands import utility_commands as uc

        monkeypatch.setattr(uc, "_validated_utility_model",
                            AsyncMock(return_value="char-specific-model"))

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Hola"))]

        with patch("commands.utility_commands._make_client") as MockClient:
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(return_value=mock_resp)
            MockClient.return_value = inst
            await uc.handle_translate_command(ix, target_language="Spanish: Hello")

        _, kwargs = inst.chat.completions.create.call_args
        assert kwargs["model"] == "char-specific-model"

    def test_resolve_prefers_active_character(self, monkeypatch):
        from commands import utility_commands as uc
        from config import characters as cfg

        char = cfg.Character("hero", "Hero", "gemma4:hero", temperature=0.4, max_tokens=512)
        monkeypatch.setattr(cfg, "_CHARACTERS", [char])
        monkeypatch.setattr(cfg, "_DEFAULT_KEY", "default")

        from bot_core import history as hist
        monkeypatch.setitem(hist._active_characters, (111, 222), "hero")

        model, temp, max_tok = uc._resolve_utility_model(111, 222)
        assert model == "gemma4:hero"
        assert temp == 0.4
        assert max_tok == 512

    def test_resolve_falls_back_to_default_model(self, monkeypatch):
        from commands import utility_commands as uc
        from config import characters as cfg

        char = cfg.Character("plain", "Plain", "", temperature=None, max_tokens=None)
        monkeypatch.setattr(cfg, "_CHARACTERS", [char])
        monkeypatch.setattr(cfg, "_DEFAULT_KEY", "plain")

        from config.settings import DEFAULT_MODEL
        model, temp, max_tok = uc._resolve_utility_model(111, 222)
        assert model == DEFAULT_MODEL


class TestOcrGuards:
    """Code review §1.9: reject non-image uploads before any API call."""

    @pytest.mark.asyncio
    async def test_ocr_rejects_text_file(self, ix):
        from commands.utility_commands import handle_ocr_command
        await handle_ocr_command(ix, image=_fake_image("notes.txt"))
        assert "looks like a text file" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_ocr_rejects_markdown_file(self, ix):
        from commands.utility_commands import handle_ocr_command
        await handle_ocr_command(ix, image=_fake_image("README.md"))
        assert "looks like a text file" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_ocr_rejects_unknown_extension(self, ix):
        from commands.utility_commands import handle_ocr_command
        await handle_ocr_command(ix, image=_fake_image("data.zip"))
        assert "doesn't look like an image" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_ocr_rejects_oversized_download(self, ix):
        from commands import utility_commands as uc
        big = b"x" * (uc.OCR_MAX_DOWNLOAD_BYTES + 1)
        await uc.handle_ocr_command(ix, image=_fake_image("huge.png", data=big))
        assert "too large" in ix._sent[0]

    @pytest.mark.asyncio
    async def test_ocr_rejects_empty_download(self, ix):
        from commands import utility_commands as uc
        await uc.handle_ocr_command(ix, image=_fake_image("empty.png", data=b""))
        assert "Could not download" in ix._sent[0]


def _fake_image(filename: str, data: bytes = b"\x89PNG\r\n\x1a\nfake"):
    image = MagicMock()
    image.url = f"https://cdn.example.com/{filename}"
    image.filename = filename
    image.read = AsyncMock(return_value=data)
    return image


class TestUtilityModuleImports:

    def test_handle_remind_command_exists(self):
        """Verify handle_remind_command is defined."""
        from commands.utility_commands import handle_remind_command
        assert callable(handle_remind_command)

    def test_handle_ocr_command_exists(self):
        """Verify handle_ocr_command is defined."""
        from commands.utility_commands import handle_ocr_command
        assert callable(handle_ocr_command)

    def test_handle_summarize_command_exists(self):
        """Verify handle_summarize_command is defined."""
        from commands.utility_commands import handle_summarize_command
        assert callable(handle_summarize_command)

    def test_handle_translate_command_exists(self):
        """Verify handle_translate_command is defined."""
        from commands.utility_commands import handle_translate_command
        assert callable(handle_translate_command)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
