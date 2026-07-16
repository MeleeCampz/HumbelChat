"""Tests for knowledge base commands: /upload_kb, /list_kb_docs, /reindex_kb."""
import pathlib
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from kb.scorch import ChunkIndex
from tests._shared import Interaction, Followup


class TestUploadKBCommand:
    """Test the /upload_kb slash command logic."""

    @pytest.mark.asyncio
    async def test_upload_kb_with_attachment(self, temp_kb_dir):
        """Test uploading a file via attachment."""
        from kb.storage import validate_upload
        from kb.scorch import ChunkIndex

        # Create a mock attachment
        attachment = MagicMock()
        attachment.filename = "test_document.txt"
        attachment.read = AsyncMock(return_value=b"This is uploaded content.")

        ix = Interaction()
        await ix.followup.send("placeholder", ephemeral=True)  # initialize _sent
        sent = ix._sent

        with patch("kb.storage._compute_sha256", return_value="abc123def456"):
            with patch.object(ChunkIndex, "from_text", return_value=[]):
                with patch("config.settings.settings") as mock_settings:
                    mock_settings.KB_PATH = temp_kb_dir
                    from commands.kb_commands import handle_upload_kb

                    await handle_upload_kb(ix, attachment=attachment, kb_name=None, url=None)

        assert any("test_document" in s or "uploaded_doc" in s for s in sent)

    @pytest.mark.asyncio
    async def test_upload_kb_with_url(self, temp_kb_dir):
        """Test uploading a file via URL."""
        ix = Interaction()
        await ix.followup.send("placeholder", ephemeral=True)  # initialize _sent
        sent = ix._sent

        # Mock httpx client
        mock_resp = MagicMock()
        mock_resp.content = b"Remote file content."
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch.object(ChunkIndex, "from_text", return_value=[]):
                with patch("config.settings.settings") as mock_settings:
                    mock_settings.KB_PATH = temp_kb_dir
                    from commands.kb_commands import handle_upload_kb

                    await handle_upload_kb(ix, attachment=None, url="https://example.com/file.txt", kb_name=None)

        assert any("file.txt" in s or "upload_kb" in s for s in sent)


    @pytest.mark.asyncio
    async def test_upload_kb_early_return(self, temp_kb_dir):
        """Test upload_kb early return path uses response.send_message."""
        ix = Interaction()
        await ix.response.send_message("placeholder", ephemeral=True)
        sent = ix._sent

        with patch("config.settings.settings") as mock_settings:
            mock_settings.KB_PATH = temp_kb_dir
            from commands.kb_commands import handle_upload_kb
            await handle_upload_kb(ix, attachment=None, url=None)

        assert any("provide either a URL" in s for s in sent)


class TestListKBDocsCommand:
    """Test the /list_kb_docs slash command."""

    @pytest.mark.asyncio
    async def test_list_kb_docs_with_files(self, temp_kb_dir):
        """Test listing KB files when files exist."""
        from commands.kb_commands import handle_list_kb_docs

        # Use the real response.send_message path (no defer)
        ix = Interaction()
        await ix.response.send_message("placeholder", ephemeral=True)  # init _sent via response.send
        sent = ix._sent

        with patch("config.settings.settings") as mock_settings:
            mock_settings.KB_PATH = temp_kb_dir
            await handle_list_kb_docs(ix)

        assert any(s for s in sent if "test_doc" in s or "nested" in s.lower())


class TestReindexKBCommand:
    """Test the /reindex_kb slash command."""

    @pytest.mark.asyncio
    async def test_reindex_kb_success(self, temp_kb_dir):
        """Test successful reindexing."""
        ix = Interaction()
        await ix.response.defer(ephemeral=True)  # call defer first
        await ix.followup.send("placeholder", ephemeral=True)  # initialize _sent
        sent = ix._sent

        with patch("commands.kb_commands.reindex_all_kb_files", return_value=2):
            with patch("config.settings.settings") as mock_settings:
                mock_settings.KB_PATH = temp_kb_dir
                from commands.kb_commands import handle_reindex_kb
                await handle_reindex_kb(ix)

        assert any("reindexed" in s.lower() for s in sent)


class TestKBCommandsStructural:
    """Test that KB commands module is properly structured."""

    def test_handle_upload_kb_exists(self):
        from commands.kb_commands import handle_upload_kb
        assert callable(handle_upload_kb)

    def test_handle_list_kb_docs_exists(self):
        from commands.kb_commands import handle_list_kb_docs
        assert callable(handle_list_kb_docs)

    def test_handle_reindex_kb_exists(self):
        from commands.kb_commands import handle_reindex_kb
        assert callable(handle_reindex_kb)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
