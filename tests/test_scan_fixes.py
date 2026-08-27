"""Regression tests for the second code-review scan (bugfix batch).

Each test pins down a specific bug that was found, reproduced, and fixed:

1. stream_ai_response must NOT swallow exceptions from the chunk iterator
   (a ``return`` inside ``finally`` used to discard them silently).
2. Conversation history must store the *clean* user message — not the
   RAG-inflated / username-decorated ``user_content``.
3. validate_upload must reject path-traversal subfolders and unsupported
   file types.
4. /upload_kb URL path: defers, caps download size, reports failures.
5. /list_kb_docs must not crash on 0-byte files.
6. /summarize <url> must actually fetch (httpx was never imported).
7. /ocr and /translate must surface backend errors instead of dying silently.
8. rearm_pending_reminders must cancel the previous task (no double-fire).
"""
from __future__ import annotations

import asyncio
import pathlib

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────────── 1. Streaming errors ─────────────────────────

class TestStreamErrorPropagation:

    @pytest.mark.asyncio
    async def test_exception_from_chunk_iterator_propagates(self):
        """A ValueError raised mid-stream must reach the caller (bug #1)."""
        from utils.stream_response import stream_ai_response

        async def failing_chunks():
            yield "partial text here, definitely long enough to flush"
            raise ValueError("input too long")

        ix = MagicMock()
        ix.followup.send = AsyncMock(return_value=MagicMock())

        with pytest.raises(ValueError, match="input too long"):
            await stream_ai_response(ix, failing_chunks())

        # Partial text was still flushed before the error surfaced.
        assert ix.followup.send.called

    @pytest.mark.asyncio
    async def test_successful_stream_returns_full_text(self):
        from utils.stream_response import stream_ai_response

        async def chunks():
            yield "hello "
            yield "world"

        ix = MagicMock()
        ix.followup.send = AsyncMock(return_value=MagicMock())

        result = await stream_ai_response(ix, chunks())
        assert result == "hello world"


# ─────────────────────── 2. History not polluted by RAG ──────────────────────

class TestHistoryCleanliness:

    @pytest.mark.asyncio
    async def test_ask_ai_stores_clean_user_message(self):
        """History must contain the raw prompt, never the RAG blob (bug #2)."""
        from bot_core import ai_client
        from bot_core.history import get_history

        g, c = 111, 222
        marker = "UNIQUE_KB_MARKER_DO_NOT_LEAK"

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Hi there!"))]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_resp)
        client.models.list = AsyncMock(return_value=MagicMock(data=[]))

        with patch.object(ai_client, "_make_client", return_value=client), \
             patch("kb.retrievers.retrieve_kb_documents",
                   new=AsyncMock(return_value=[("doc.md", marker)])):
            reply, _extra = await ai_client.ask_ai(
                user_message="hello world",
                model_slug="test-model",
                guild_id=g,
                channel_id=c,
                username="Alice",
                user_id=None,
            )

        assert reply == "Hi there!"
        history = get_history(g, c)
        user_msgs = [m for m in history if m["role"] == "user"]
        assert user_msgs, "no user message stored in history"
        # Clean prompt only — no RAG blob, no "**Alice:**" decoration.
        assert user_msgs[-1]["content"] == "hello world"
        for m in history:
            assert marker not in m["content"], "RAG context leaked into history"
            assert "Alice" not in m["content"], "username decoration leaked into history"

    @pytest.mark.asyncio
    async def test_ask_ai_stream_stores_clean_user_message(self):
        from bot_core import ai_client
        from bot_core.history import get_history

        g, c = 333, 444
        marker = "UNIQUE_KB_MARKER_STREAM"

        stream_chunk = MagicMock()
        stream_chunk.choices = [MagicMock(delta=MagicMock(content="streamed reply"))]
        stream_chunk2 = MagicMock()
        stream_chunk2.choices = []

        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=async_iter([stream_chunk, stream_chunk2])
        )
        client.models.list = AsyncMock(return_value=MagicMock(data=[]))

        with patch.object(ai_client, "_make_client", return_value=client), \
             patch("kb.retrievers.retrieve_kb_documents",
                   new=AsyncMock(return_value=[("doc.md", marker)])):
            collected = []
            async for chunk in ai_client.ask_ai_stream(
                user_message="streaming hello",
                model_slug="test-model",
                guild_id=g,
                channel_id=c,
                username="Bob",
                user_id=None,
            ):
                collected.append(chunk)

        assert "".join(collected) == "streamed reply"
        history = get_history(g, c)
        user_msgs = [m for m in history if m["role"] == "user"]
        assert user_msgs[-1]["content"] == "streaming hello"
        for m in history:
            assert marker not in m["content"]


def async_iter(items):
    """Minimal async iterator helper for mocked streams."""
    async def _gen():
        for it in items:
            yield it
    return _gen()


# ─────────────────────── 3. Upload validation hardening ──────────────────────

class TestUploadValidation:

    def test_traversal_subfolder_rejected(self, tmp_path):
        """subfolder='../../evil' must not escape the KB root (bug #4)."""
        from kb.storage import validate_upload

        kb = tmp_path / "kb"
        kb.mkdir()
        with pytest.raises(ValueError, match="Invalid subfolder"):
            validate_upload(b"data", filename="x.txt", kb_path=kb, subfolder="../../evil")
        assert not (tmp_path / "evil").exists()

    def test_traversal_subfolder_inside_kb_allowed(self, tmp_path):
        from kb.storage import validate_upload

        kb = tmp_path / "kb"
        kb.mkdir()
        dest, _ = validate_upload(b"data", filename="x.txt", kb_path=kb, subfolder="a/b")
        assert dest.parent.is_relative_to(kb.resolve())

    @pytest.mark.parametrize("fname", ["malware.exe", "archive.zip", "script.sh"])
    def test_unsupported_extension_rejected(self, tmp_path, fname):
        from kb.storage import validate_upload

        kb = tmp_path / "kb"
        kb.mkdir()
        with pytest.raises(ValueError, match="not supported"):
            validate_upload(b"data", filename=fname, kb_path=kb)
        assert not (kb / fname).exists(), "rejected file must not touch disk"

    def test_supported_extension_accepted(self, tmp_path):
        from kb.storage import validate_upload

        kb = tmp_path / "kb"
        kb.mkdir()
        dest, summary = validate_upload(b"data", filename="notes.md", kb_path=kb)
        assert dest.exists()
        assert summary["name"] == "notes.md"


# ─────────────────────── 4. /upload_kb URL hardening ─────────────────────────

class TestUploadKBUrl:

    @pytest.mark.asyncio
    async def test_url_upload_defers_first(self, ix, temp_kb_dir):
        """The interaction must be deferred before any download starts (bug #3)."""
        from commands.kb_commands import handle_upload_kb

        mock_resp = MagicMock()
        mock_resp.content = b"tiny remote file"
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("commands.kb_commands.httpx.AsyncClient", return_value=mock_client), \
             patch("kb.storage.KB_PATH", temp_kb_dir):
            await handle_upload_kb(ix, attachment=None, url="https://example.com/doc.txt")

        assert ix.response.defer.called, "must defer before downloading"
        assert any("stored" in s for s in ix._sent)

    @pytest.mark.asyncio
    async def test_url_upload_oversized_rejected(self, ix, temp_kb_dir):
        """Downloads above the cap are rejected before validate_upload (bug #3)."""
        from commands.kb_commands import handle_upload_kb

        mock_resp = MagicMock()
        mock_resp.content = b"x" * 5000
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("commands.kb_commands.httpx.AsyncClient", return_value=mock_client), \
             patch("commands.kb_commands.UPLOAD_MAX_DOWNLOAD_BYTES", 1000), \
             patch("kb.storage.KB_PATH", temp_kb_dir):
            await handle_upload_kb(ix, attachment=None, url="https://example.com/big.bin")

        assert any("too large" in s for s in ix._sent)

    @pytest.mark.asyncio
    async def test_url_download_failure_reports_error(self, ix, temp_kb_dir):
        import httpx as _httpx

        from commands.kb_commands import handle_upload_kb

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.get = AsyncMock(side_effect=_httpx.ConnectError("refused"))

        with patch("commands.kb_commands.httpx.AsyncClient", return_value=mock_client), \
             patch("kb.storage.KB_PATH", temp_kb_dir):
            await handle_upload_kb(ix, attachment=None, url="https://dead.example/f.txt")

        assert any("Failed to download" in s for s in ix._sent)


# ─────────────────────── 5. /list_kb_docs 0-byte files ───────────────────────

class TestListKBDocsZeroByte:

    @pytest.mark.asyncio
    async def test_zero_byte_file_does_not_crash(self, ix, tmp_path):
        """0-byte KB files used to raise ValueError in the size format (bug #6)."""
        from commands.kb_commands import handle_list_kb_docs

        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "empty.txt").write_bytes(b"")
        (kb / "normal.txt").write_text("some content here")

        with patch("config.settings.KB_PATH", kb):
            await handle_list_kb_docs(ix)  # must not raise

        listing = "\n".join(ix._sent)
        assert "empty.txt" in listing
        assert "0.0 KB" in listing


# ─────────────────────── 6. /summarize URL fetch ─────────────────────────────

class TestSummarizeUrl:

    @pytest.mark.asyncio
    async def test_summarize_url_fetches_and_succeeds(self, ix):
        """httpx must be importable in the module (bug #5) and the happy path works."""
        from commands.utility_commands import handle_summarize_command

        mock_resp = MagicMock()
        mock_resp.text = "A long document to summarize."
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.get = AsyncMock(return_value=mock_resp)

        ai_resp = MagicMock()
        ai_resp.choices = [MagicMock(message=MagicMock(content="SUMMARY_TEXT"))]
        ai_client = MagicMock()
        ai_client.chat.completions.create = AsyncMock(return_value=ai_resp)

        with patch("commands.utility_commands.httpx.AsyncClient", return_value=mock_client), \
             patch("commands.utility_commands._make_client", return_value=ai_client), \
             patch("commands.utility_commands._resolve_utility_model",
                   return_value=("test-model", None, None)):
            await handle_summarize_command(ix, file_url="https://example.com/doc.txt")

        assert any("SUMMARY_TEXT" in s for s in ix._sent)


# ─────────────── 7. OCR / translate error surfacing ──────────────────────────

class TestUtilityErrorSurfacing:

    @pytest.mark.asyncio
    async def test_ocr_backend_error_shows_message(self, ix):
        from commands.utility_commands import handle_ocr_command

        image = MagicMock()
        image.filename = "pic.png"
        image.read = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n")

        ai_client = MagicMock()
        ai_client.chat.completions.create = AsyncMock(side_effect=Exception("backend down"))

        with patch("commands.utility_commands._make_client", return_value=ai_client), \
             patch("commands.utility_commands._validated_utility_model",
                   new=AsyncMock(return_value="test-model")):
            await handle_ocr_command(ix, image=image)  # must not raise

        assert any("OCR failed" in s for s in ix._sent)

    @pytest.mark.asyncio
    async def test_translate_backend_error_shows_message(self, ix):
        from commands.utility_commands import handle_translate_command

        ai_client = MagicMock()
        ai_client.chat.completions.create = AsyncMock(side_effect=Exception("timeout"))

        with patch("commands.utility_commands._make_client", return_value=ai_client), \
             patch("commands.utility_commands._validated_utility_model",
                   new=AsyncMock(return_value="test-model")):
            await handle_translate_command(ix, target_language="Spanish: hola")  # must not raise

        assert any("Translation failed" in s for s in ix._sent)


# ─────────────── 8. Reminder re-arm cancels old task ─────────────────────────

class TestReminderRearm:

    @pytest.mark.asyncio
    async def test_rearm_cancels_previous_task(self):
        """A second on_ready (full reconnect) must not double-fire (bug #8)."""
        from bot_core import reminders as R

        rid = R.schedule_reminder(999, "wake up", delay_sec=60)
        old_task = R._tasks.get(rid)
        assert old_task is not None and not old_task.done()

        n = R.rearm_pending_reminders()
        assert n >= 1

        new_task = R._tasks.get(rid)
        assert new_task is not None
        assert new_task is not old_task, "old task was replaced but not cancelled"
        # Give the loop a moment to process the cancellation.
        await asyncio.sleep(0)
        assert old_task.cancelled(), "previous reminder task must be cancelled"

        # Cleanup: don't leave a pending 60 s sleep around.
        new_task.cancel()
        R._tasks.pop(rid, None)
        R._reminders.pop(rid, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
