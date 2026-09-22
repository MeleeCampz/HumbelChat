"""Regression tests for P1 #8: ``IndexError`` on empty ``choices``.

Some backends complete a request but return an empty ``choices`` array.
``resp.choices[0].message.content`` used to crash with an unhelpful
``IndexError``.  We now route *every* completion read through
``bot_core.errors.extract_reply_text``:

* empty *content*  → a caller-supplied placeholder (no crash, no error)
* empty *choices*  → a friendly ``AIBackendError`` that callers convert to a
  user-friendly message (or, for multi-model callers, a retry of the next
  model)

The TODO's canonical verify case: a stub response whose ``choices == []`` must
produce a friendly error, never an ``IndexError``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot_core import ai_client
from bot_core.errors import AIBackendError, extract_reply_text


def _resp(content: str | None, *, choices_empty: bool = False) -> MagicMock:
    if choices_empty:
        resp = MagicMock()
        resp.choices = []
        return resp
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


# ─────────────────────── extract_reply_text helper ───────────────────────

class TestExtractReplyText:
    def test_normal_content_returned(self):
        assert extract_reply_text(_resp("hello")) == "hello"

    def test_empty_content_returns_default(self):
        # content is None/"" -> placeholder, NOT an error.
        assert extract_reply_text(_resp(None), default="(empty)") == "(empty)"
        assert extract_reply_text(_resp(""), default="(empty)") == "(empty)"

    def test_empty_choices_raises_ai_backend_error(self):
        # The P1 #8 core case: choices == [] -> AIBackendError, no IndexError.
        with pytest.raises(AIBackendError):
            extract_reply_text(_resp(None, choices_empty=True))
        # ...and it is NOT an IndexError.
        try:
            extract_reply_text(_resp(None, choices_empty=True))
        except AIBackendError as e:
            assert not isinstance(e, IndexError)
        except IndexError:  # pragma: no cover - the bug we fixed
            pytest.fail("choices == [] raised IndexError instead of AIBackendError")

    def test_none_response_raises(self):
        with pytest.raises(AIBackendError):
            extract_reply_text(None)

    def test_category_is_backend_error(self):
        from bot_core.errors import AIError
        err = AIBackendError("x")
        assert AIBackendError.category == "backend_error"
        assert isinstance(err, AIError)
        assert isinstance(err, Exception)
        # Carries a user-facing message distinct from the internal one.
        assert err.user_message


# ─────────────────────── ask_ai end-to-end path ───────────────────────

def _stub_ask_ai_env(monkeypatch, create_side_effect):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    client.models.list = AsyncMock(return_value=MagicMock(data=[]))
    monkeypatch.setattr(ai_client, "_make_client", lambda: client)
    monkeypatch.setattr(ai_client, "_validate_model",
                        AsyncMock(return_value="test-model"))
    from kb import retrievers
    monkeypatch.setattr(retrievers, "retrieve_kb_documents",
                        AsyncMock(return_value=[]))
    return client


class TestAskAiEmptyChoices:
    @pytest.mark.asyncio
    async def test_empty_choices_is_friendly_not_indexerror(self, monkeypatch):
        """The TODO's verify case: choices == [] -> user-friendly error."""
        _stub_ask_ai_env(monkeypatch, [_resp(None, choices_empty=True)])
        with pytest.raises(ValueError) as exc:  # friendly, not IndexError
            await ai_client.ask_ai(
                user_message="ping", model_slug="test-model",
                guild_id=1, channel_id=2, username="Alice", user_id=None,
            )
        assert not isinstance(exc.value, IndexError)
        assert "empty response" in str(exc.value).lower()
        assert "AIBackendError" not in type(exc.value).__name__


# ─────────────────────── /ocr and /translate command paths ───────────────────────

class TestUtilityCommandsEmptyChoices:
    @pytest.mark.asyncio
    async def test_crash_free_ocr_on_empty_choices(self, ix):
        """An empty-choices OCR response must yield a friendly failure, not an IndexError."""
        from commands.utility_commands import handle_ocr_command

        image = MagicMock()
        image.url = "https://example.com/image.png"
        image.filename = "test_image.png"
        image.read = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n")

        mock_resp = MagicMock()
        mock_resp.choices = []  # P1 #8 trigger

        with patch("commands.utility_commands._make_client") as MockClient, \
             patch("commands.utility_commands._validated_utility_model",
                   new=AsyncMock(return_value="test-model")):
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(return_value=mock_resp)
            MockClient.return_value = inst
            await handle_ocr_command(ix, image=image)

        assert len(ix._sent) > 0
        joined = " ".join(ix._sent)
        assert "OCR failed" in joined
        assert "AIBackendError" in joined  # surfaced as class name in the message
        # Crucially: no traceback leaked to the user.
        assert "Traceback" not in joined

    @pytest.mark.asyncio
    async def test_translate_empty_choices_is_friendly(self, ix):
        """Empty-choices translate -> friendly 'Translation failed', not an IndexError."""
        from commands.utility_commands import handle_translate_command

        mock_resp = MagicMock()
        mock_resp.choices = []  # P1 #8 trigger

        with patch("commands.utility_commands._make_client") as MockClient, \
             patch("commands.utility_commands._validated_utility_model",
                   new=AsyncMock(return_value="test-model")):
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(return_value=mock_resp)
            MockClient.return_value = inst
            await handle_translate_command(
                ix, target_language="Spanish: Hello world", source_language="English",
            )

        assert len(ix._sent) > 0
        joined = " ".join(ix._sent)
        assert "Translation failed" in joined
        assert "Traceback" not in joined
