"""Tests for knowledge base commands: /upload_kb, /list_kb_docs, /reindex_kb."""
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from tests._shared import Interaction


class TestUploadKBCommand:
    """Test the /upload_kb slash command logic."""

    @pytest.mark.asyncio
    async def test_upload_kb_with_attachment(self, temp_kb_dir):
        """Test uploading a file via attachment."""
        # Create a mock attachment
        attachment = MagicMock()
        attachment.filename = "test_document.txt"
        attachment.size = len(b"This is uploaded content.")  # real attachments always have int size
        attachment.read = AsyncMock(return_value=b"This is uploaded content.")

        ix = Interaction()
        await ix.followup.send("placeholder", ephemeral=True)  # initialize _sent
        sent = ix._sent

        # NOTE: patch kb.storage.KB_PATH (the name validate_upload actually
        # reads) — config.settings.KB_PATH is a separate binding.
        with patch("kb.storage._compute_sha256", return_value="abc123def456"):
            with patch("kb.storage.KB_PATH", temp_kb_dir):
                from commands.kb_commands import handle_upload_kb

                await handle_upload_kb(ix, attachment=attachment, url=None)

        assert any("test_document" in s or "uploaded_doc" in s for s in sent)

    @pytest.mark.asyncio
    async def test_upload_kb_with_url(self, temp_kb_dir):
        """Test uploading a file via URL."""
        ix = Interaction()
        await ix.followup.send("placeholder", ephemeral=True)  # initialize _sent
        sent = ix._sent

        # P1 #16: the fetch goes through utils.url_fetch.fetch_url (streaming +
        # scheme guard). Stub it to return the bytes so this test stays focused
        # on the upload flow.
        with patch("utils.url_fetch.fetch_url", new=AsyncMock(return_value=b"Remote file content.")):
            with patch("kb.storage.KB_PATH", temp_kb_dir):
                from commands.kb_commands import handle_upload_kb

                await handle_upload_kb(ix, attachment=None, url="https://example.com/file.txt")

        assert any("file.txt" in s or "upload_kb" in s for s in sent)

    @pytest.mark.asyncio
    async def test_upload_kb_rejects_zero_byte(self, temp_kb_dir):
        """P3 #36: a 0-byte attachment must be rejected, not stored."""
        attachment = MagicMock()
        attachment.filename = "empty.txt"
        attachment.size = 0
        attachment.read = AsyncMock(return_value=b"")  # 0 bytes

        ix = Interaction()
        sent = ix._sent
        with patch("kb.storage.KB_PATH", temp_kb_dir):
            from commands.kb_commands import handle_upload_kb
            await handle_upload_kb(ix, attachment=attachment, url=None)

        joined = " ".join(sent)
        assert "empty" in joined.lower()
        # Nothing was written to the KB.
        from kb.storage import list_kb_files
        names = [d["name"] for d in list_kb_files(temp_kb_dir, recursive=True)]
        assert not any("empty" in n for n in names)

    @pytest.mark.asyncio
    async def test_upload_kb_rejects_oversized_attachment_before_reading(
        self, temp_kb_dir,
    ):
        """Audit L6: oversized attachments are rejected from Discord's size
        metadata WITHOUT reading the bytes into memory."""
        from commands.kb_commands import UPLOAD_MAX_DOWNLOAD_BYTES
        attachment = MagicMock()
        attachment.filename = "huge.txt"
        attachment.size = UPLOAD_MAX_DOWNLOAD_BYTES + 1
        attachment.read = AsyncMock(side_effect=AssertionError("must not be read"))

        ix = Interaction()
        sent = ix._sent
        with patch("kb.storage.KB_PATH", temp_kb_dir):
            from commands.kb_commands import handle_upload_kb
            await handle_upload_kb(ix, attachment=attachment, url=None)

        assert any("too large" in s for s in sent)
        attachment.read.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upload_kb_early_return(self, temp_kb_dir):
        """Test upload_kb early return path uses response.send_message."""
        ix = Interaction()
        await ix.response.send_message("placeholder", ephemeral=True)
        sent = ix._sent

        with patch("config.settings.KB_PATH", temp_kb_dir):
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

        with patch("config.settings.KB_PATH", temp_kb_dir):
            await handle_list_kb_docs(ix)

        assert any(s for s in sent if "test_doc" in s or "nested" in s.lower())

    @pytest.mark.asyncio
    async def test_list_kb_docs_rejects_path_outside_kb(self, temp_kb_dir, tmp_path):
        """A subfolder of ../ must not list or hash files outside KB_PATH."""
        from commands.kb_commands import handle_list_kb_docs
        from kb.storage import list_kb_files

        secret = tmp_path / "secret.txt"
        secret.write_text("do not list me")

        ix = Interaction()
        await ix.response.send_message("placeholder")
        sent = ix._sent

        with patch("config.settings.KB_PATH", temp_kb_dir):
            await handle_list_kb_docs(ix, subfolder_path="..")

        assert any("outside the knowledge base" in s for s in sent)
        assert not any("secret" in s for s in sent)

        with pytest.raises(ValueError):
            list_kb_files(temp_kb_dir, subfolder="..")


class TestReindexKBCommand:
    """Test the /reindex_kb slash command."""

    @pytest.mark.asyncio
    async def test_reindex_kb_success(self, temp_kb_dir):
        """Test successful reindexing."""
        ix = Interaction()
        await ix.response.defer(ephemeral=True)  # call defer first
        await ix.followup.send("placeholder", ephemeral=True)  # initialize _sent
        sent = ix._sent

        mock_idx = MagicMock()
        mock_idx.count.return_value = 5
        mock_idx.is_empty.return_value = False

        mock_store = AsyncMock()
        mock_store.load = AsyncMock(return_value=mock_idx)
        mock_store.shutdown = AsyncMock()
        # get_index is sync in the real class — keep it sync here too so the
        # retrievers' `store.get_index()` calls don't create un-awaited coroutines.
        mock_store.get_index = MagicMock(return_value=mock_idx)

        with patch("kb.index.KBIndexStore", return_value=mock_store):
            with patch("config.settings.KB_PATH", temp_kb_dir):
                from commands.kb_commands import handle_reindex_kb
                await handle_reindex_kb(ix)

        assert any("rebuilt" in s.lower() for s in sent)


class TestMessageUploads:
    """#60: plain messages with file attachment(s) are stored like regular uploads."""

    @staticmethod
    def _message(attachments):
        message = MagicMock()
        message.author = MagicMock(id=9, __str__=lambda s: "Ada")
        message.channel.send = AsyncMock()
        message.attachments = attachments
        return message

    @pytest.mark.asyncio
    async def test_message_attachment_stored_and_indexed(self, temp_kb_dir, monkeypatch):
        attachment = MagicMock()
        attachment.filename = "long_notes.txt"
        attachment.size = len(b"Session notes content.")
        attachment.read = AsyncMock(return_value=b"Session notes content.")

        # Indexing is out of scope here — stub the retriever path.
        monkeypatch.setattr(
            "kb.retrievers.update_kb_document", AsyncMock(return_value=True), raising=False)

        message = self._message([attachment])
        with patch("kb.storage.KB_PATH", temp_kb_dir):
            from commands.kb_commands import handle_message_uploads
            await handle_message_uploads(message)

        sent = " ".join(str(c) for c in message.channel.send.call_args_list)
        assert "long_notes" in sent
        assert "Indexed for search." in sent
        from kb.storage import list_kb_files
        names = [d["name"] for d in list_kb_files(temp_kb_dir, recursive=True)]
        assert any("long_notes" in n for n in names)

    @pytest.mark.asyncio
    async def test_message_oversized_text_attachment_rejected_before_read(self, temp_kb_dir):
        attachment = MagicMock()
        attachment.filename = "huge.txt"
        attachment.size = 99 * 1024 * 1024  # over the 20 MB cap
        attachment.read = AsyncMock()  # must NOT be called

        message = self._message([attachment])
        with patch("kb.storage.KB_PATH", temp_kb_dir):
            from commands.kb_commands import handle_message_uploads
            await handle_message_uploads(message)

        attachment.read.assert_not_called()
        sent = " ".join(str(c) for c in message.channel.send.call_args_list)
        assert "too large" in sent.lower()
        from kb.storage import list_kb_files
        names = [d["name"] for d in list_kb_files(temp_kb_dir, recursive=True)]
        assert not any("huge" in n for n in names)

    @pytest.mark.asyncio
    async def test_message_image_attachment_skipped_silently(self, temp_kb_dir):
        """#64: dropping an image into the channel (no upload intent) must not
        answer 'Upload rejected' — non-KB file types are skipped, no reply."""
        attachment = MagicMock()
        attachment.filename = "screenshot.png"
        attachment.size = 123456
        attachment.content_type = "image/png"
        attachment.read = AsyncMock()  # must NOT be called

        message = self._message([attachment])
        with patch("kb.storage.KB_PATH", temp_kb_dir):
            from commands.kb_commands import handle_message_uploads
            await handle_message_uploads(message)

        attachment.read.assert_not_called()
        message.channel.send.assert_not_called()  # no reply at all
        from kb.storage import list_kb_files
        names = [d["name"] for d in list_kb_files(temp_kb_dir, recursive=True)]
        assert not any("screenshot" in n for n in names)

    @pytest.mark.asyncio
    async def test_message_mixed_text_and_image(self, temp_kb_dir, monkeypatch):
        """#64: only the KB-supported attachment is stored; the image is skipped."""
        text_att = MagicMock()
        text_att.filename = "notes.md"
        text_att.size = len(b"# Notes")
        text_att.read = AsyncMock(return_value=b"# Notes")
        img_att = MagicMock()
        img_att.filename = "pic.jpg"
        img_att.size = 10
        img_att.content_type = "image/jpeg"
        img_att.read = AsyncMock()

        monkeypatch.setattr(
            "kb.retrievers.update_kb_document", AsyncMock(return_value=True), raising=False)

        message = self._message([img_att, text_att])
        with patch("kb.storage.KB_PATH", temp_kb_dir):
            from commands.kb_commands import handle_message_uploads
            await handle_message_uploads(message)

        img_att.read.assert_not_called()
        sent = " ".join(str(c) for c in message.channel.send.call_args_list)
        assert "notes.md" in sent          # text stored + summarized
        assert "rejected" not in sent.lower()  # no error about the image

    @pytest.mark.asyncio
    async def test_on_message_routes_attachments_to_upload_handler(self, monkeypatch):
        """main.on_message: non-prefix message with attachments → upload path."""
        import main

        calls = []
        monkeypatch.setattr(
            "commands.kb_commands.handle_message_uploads",
            AsyncMock(side_effect=lambda m: calls.append(m)),
        )

        attachment = MagicMock(filename="notes.txt", size=10)
        message = MagicMock()
        message.author = MagicMock(id=9, __str__=lambda s: "Ada")
        message.guild = None
        message.channel.id = 7
        message.content = "here are my notes"
        message.attachments = [attachment]

        await main.on_message(message)
        assert calls and calls[0] is message

    @pytest.mark.asyncio
    async def test_on_message_plain_text_without_attachments_is_ignored(self, monkeypatch):
        import main

        monkeypatch.setattr(
            "commands.kb_commands.handle_message_uploads", AsyncMock())

        message = MagicMock()
        message.author = MagicMock(id=9)
        message.guild = None
        message.channel.id = 7
        message.content = "just a remark"
        message.attachments = []

        await main.on_message(message)  # must return quietly, no upload path


class TestKBCommandsStructural:
    """Test that KB commands module is properly structured."""

    def test_handle_upload_kb_exists(self):
        from commands.kb_commands import handle_upload_kb
        assert callable(handle_upload_kb)

    def test_store_upload_bytes_exists(self):
        from commands.kb_commands import store_upload_bytes
        assert callable(store_upload_bytes)

    def test_handle_message_uploads_exists(self):
        from commands.kb_commands import handle_message_uploads
        assert callable(handle_message_uploads)

    def test_handle_list_kb_docs_exists(self):
        from commands.kb_commands import handle_list_kb_docs
        assert callable(handle_list_kb_docs)

    def test_handle_reindex_kb_exists(self):
        from commands.kb_commands import handle_reindex_kb
        assert callable(handle_reindex_kb)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
