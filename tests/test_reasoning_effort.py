"""#20 regression tests: per-task reasoning-effort settings.

Covers:
  * ``_reasoning_extra_body`` mapping (None / off / low / medium / high / xhigh)
  * the env-var parser allowlist (case-insensitive, invalid → None + warning)
  * ``ask_ai`` / ``ask_ai_stream`` sending (or omitting) extra_body for chat
  * the 4xx retry-without-param fallback (and that 5xx gets NO plain retry)
  * ``complete_text`` reasoning control (disable_thinking + reasoning_effort)
  * call sites: /summarize, /translate, /ocr default to thinking-off and
    honour SUMMARY_REASONING_EFFORT; recap honours the same var; merged log
    omits the param when MERGE_LOG_REASONING_EFFORT is unset.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from bot_core import ai_client
from tests.ai_mocks import FakeStream, chunk


# ─────────────────────────── helpers ───────────────────────────


def _http_response(status_code: int) -> httpx.Response:
    req = httpx.Request("POST", "http://backend.invalid/v1/chat/completions")
    return httpx.Response(status_code, request=req)


def _openai_status_error(status_code: int, message: str = "error") -> openai.APIStatusError:
    cls = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        429: openai.RateLimitError,
    }.get(status_code, openai.InternalServerError)
    return cls(message, response=_http_response(status_code), body=None)


def _success_resp() -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="PONG"))]
    return resp


def _stub_ask_ai_env(monkeypatch, create_side_effect):
    """Point ask_ai / ask_ai_stream at a stub client + empty KB (pattern from
    tests/test_p5_streaming.py)."""
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


class _FakeStream:
    """Async iterator mimicking the openai SDK stream."""

    def __init__(self, items):
        self._items = list(items)
        self._i = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._i]
        self._i += 1
        return item

    def close(self):
        self.closed = True


def _stream_chunk(delta: str | None, finish_reason: str | None = None) -> MagicMock:
    c = MagicMock()
    c.choices = [MagicMock(delta=MagicMock(content=delta), finish_reason=finish_reason)]
    return c


# ──────────────── _reasoning_extra_body mapping ────────────────


class TestReasoningExtraBody:

    def test_none_omits_param(self):
        assert ai_client._reasoning_extra_body(None) is None

    def test_off_maps_to_enable_thinking_false(self):
        assert ai_client._reasoning_extra_body("off") == {"enable_thinking": False}

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
    def test_graded_efforts_map_to_reasoning_effort(self, effort):
        assert ai_client._reasoning_extra_body(effort) == {"reasoning_effort": effort}


# ──────────────── env parser allowlist ────────────────


class TestEnvParser:

    def test_valid_value(self):
        import config.settings as settings
        assert settings._reasoning_effort_env("X", "low") == "low"

    def test_case_and_whitespace_insensitive(self):
        import config.settings as settings
        assert settings._reasoning_effort_env("X", "  XHIGH ") == "xhigh"
        assert settings._reasoning_effort_env("X", "Off") == "off"

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_unset_or_empty_is_none(self, raw):
        import config.settings as settings
        assert settings._reasoning_effort_env("X", raw) is None

    def test_invalid_value_falls_back_to_none_with_warning(self, caplog):
        import config.settings as settings
        with caplog.at_level("WARNING"):
            assert settings._reasoning_effort_env("FAKE_VAR", "ultra") is None
        assert any("FAKE_VAR" in r.message for r in caplog.records)


# ──────────────── ask_ai (non-streaming chat) ────────────────


class TestAskAiReasoning:

    @pytest.mark.asyncio
    async def test_sends_extra_body_when_set(self, monkeypatch):
        client = _stub_ask_ai_env(monkeypatch, [_success_resp()])
        monkeypatch.setattr(ai_client, "AI_REASONING_EFFORT", "low")
        reply, _ = await ai_client.ask_ai(
            user_message="ping", model_slug="test-model",
            guild_id=1, channel_id=2, username="Alice", user_id=None,
        )
        assert reply == "PONG"
        kw = client.chat.completions.create.await_args.kwargs
        assert kw.get("extra_body") == {"reasoning_effort": "low"}

    @pytest.mark.asyncio
    async def test_omits_extra_body_when_unset(self, monkeypatch):
        client = _stub_ask_ai_env(monkeypatch, [_success_resp()])
        monkeypatch.setattr(ai_client, "AI_REASONING_EFFORT", None)
        await ai_client.ask_ai(
            user_message="ping", model_slug="test-model",
            guild_id=1, channel_id=2, username="Alice", user_id=None,
        )
        kw = client.chat.completions.create.await_args.kwargs
        assert "extra_body" not in kw

    @pytest.mark.asyncio
    async def test_4xx_rejection_retries_once_without_param(self, monkeypatch):
        err = _openai_status_error(400, message="unknown parameter")
        client = _stub_ask_ai_env(monkeypatch, [err, _success_resp()])
        monkeypatch.setattr(ai_client, "AI_REASONING_EFFORT", "xhigh")
        reply, _ = await ai_client.ask_ai(
            user_message="ping", model_slug="test-model",
            guild_id=1, channel_id=2, username="Alice", user_id=None,
        )
        assert reply == "PONG"
        calls = client.chat.completions.create.await_args_list
        assert len(calls) == 2
        assert calls[0].kwargs.get("extra_body") == {"reasoning_effort": "xhigh"}
        assert "extra_body" not in calls[1].kwargs

    @pytest.mark.asyncio
    async def test_5xx_gets_no_plain_fallback(self, monkeypatch):
        # 500 is transient → the existing bounded retry applies (2 attempts),
        # but BOTH keep the extra_body — no param-stripping plain retry.
        exc = _openai_status_error(500, message="upstream exploded")
        client = _stub_ask_ai_env(monkeypatch, [exc, exc])
        monkeypatch.setattr(ai_client, "AI_REASONING_EFFORT", "low")
        with patch.object(ai_client, "_COMPLETION_RETRY_BACKOFF_S", 0.01):
            with pytest.raises(Exception):
                await ai_client.ask_ai(
                    user_message="ping", model_slug="test-model",
                    guild_id=1, channel_id=2, username="Alice", user_id=None,
                )
        calls = client.chat.completions.create.await_args_list
        assert len(calls) == 2
        for c in calls:
            assert c.kwargs.get("extra_body") == {"reasoning_effort": "low"}


# ──────────────── ask_ai_stream (streaming chat) ────────────────


class TestAskAiStreamReasoning:

    @pytest.mark.asyncio
    async def test_sends_extra_body_when_set(self, monkeypatch):
        client = _stub_ask_ai_env(
            monkeypatch, [_FakeStream([_stream_chunk("PONG", "stop")])]
        )
        monkeypatch.setattr(ai_client, "AI_REASONING_EFFORT", "medium")
        async for text in ai_client.ask_ai_stream(
            user_message="ping", model_slug="test-model",
            guild_id=1, channel_id=2, username="Alice", user_id=None,
        ):
            pass
        assert text == "PONG"
        kw = client.chat.completions.create.await_args.kwargs
        assert kw.get("extra_body") == {"reasoning_effort": "medium"}

    @pytest.mark.asyncio
    async def test_omits_extra_body_when_unset(self, monkeypatch):
        client = _stub_ask_ai_env(
            monkeypatch, [_FakeStream([_stream_chunk("PONG", "stop")])]
        )
        monkeypatch.setattr(ai_client, "AI_REASONING_EFFORT", None)
        async for text in ai_client.ask_ai_stream(
            user_message="ping", model_slug="test-model",
            guild_id=1, channel_id=2, username="Alice", user_id=None,
        ):
            pass
        assert "extra_body" not in client.chat.completions.create.await_args.kwargs

    @pytest.mark.asyncio
    async def test_4xx_rejection_retries_once_without_param(self, monkeypatch):
        err = _openai_status_error(400, message="unknown parameter")
        client = _stub_ask_ai_env(
            monkeypatch, [err, _FakeStream([_stream_chunk("PONG", "stop")])]
        )
        monkeypatch.setattr(ai_client, "AI_REASONING_EFFORT", "off")
        async for text in ai_client.ask_ai_stream(
            user_message="ping", model_slug="test-model",
            guild_id=1, channel_id=2, username="Alice", user_id=None,
        ):
            pass
        assert text == "PONG"
        calls = client.chat.completions.create.await_args_list
        assert len(calls) == 2
        assert calls[0].kwargs.get("extra_body") == {"enable_thinking": False}
        assert "extra_body" not in calls[1].kwargs


# ──────────────── _create_with_reasoning_fallback (direct create) ────────────────


class TestCreateWithReasoningFallback:

    @pytest.mark.asyncio
    async def test_4xx_retries_without_extra_body(self):
        client = MagicMock()
        err = _openai_status_error(400, message="unknown parameter")
        client.chat.completions.create = AsyncMock(side_effect=[err, _success_resp()])
        resp = await ai_client._create_with_reasoning_fallback(
            client, "high", model="m", messages=[], temperature=0.3,
        )
        assert resp.choices[0].message.content == "PONG"
        calls = client.chat.completions.create.await_args_list
        assert len(calls) == 2
        assert calls[0].kwargs.get("extra_body") == {"reasoning_effort": "high"}
        assert "extra_body" not in calls[1].kwargs

    @pytest.mark.asyncio
    async def test_5xx_propagates_without_plain_retry(self):
        client = MagicMock()
        exc = _openai_status_error(500, message="upstream exploded")
        client.chat.completions.create = AsyncMock(side_effect=exc)
        with pytest.raises(openai.APIStatusError):
            await ai_client._create_with_reasoning_fallback(
                client, "low", model="m", messages=[], temperature=0.3,
            )
        assert client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_none_effort_sends_no_extra_body(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_success_resp())
        await ai_client._create_with_reasoning_fallback(
            client, None, model="m", messages=[], temperature=0.3,
        )
        assert "extra_body" not in client.chat.completions.create.await_args.kwargs


# ──────────────── complete_text (summary-call helper) ────────────────


class TestCompleteTextReasoning:

    @pytest.mark.asyncio
    async def test_disable_thinking_still_sends_enable_thinking_false(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=FakeStream([chunk("hi", "stop")])
        )
        text = await ai_client.complete_text(
            client, model="m", messages=[{"role": "user", "content": "x"}],
            temperature=0.3, disable_thinking=True,
        )
        assert text == "hi"
        kw = client.chat.completions.create.await_args.kwargs
        assert kw.get("extra_body") == {"enable_thinking": False}

    @pytest.mark.asyncio
    async def test_reasoning_effort_param_maps_through(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=FakeStream([chunk("hi", "stop")])
        )
        await ai_client.complete_text(
            client, model="m", messages=[{"role": "user", "content": "x"}],
            temperature=0.3, reasoning_effort="low",
        )
        kw = client.chat.completions.create.await_args.kwargs
        assert kw.get("extra_body") == {"reasoning_effort": "low"}

    @pytest.mark.asyncio
    async def test_no_control_sends_no_extra_body(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=FakeStream([chunk("hi", "stop")])
        )
        await ai_client.complete_text(
            client, model="m", messages=[{"role": "user", "content": "x"}],
            temperature=0.3,
        )
        assert "extra_body" not in client.chat.completions.create.await_args.kwargs

    @pytest.mark.asyncio
    async def test_4xx_rejection_retries_without_param(self):
        client = MagicMock()
        err = _openai_status_error(400, message="unknown parameter")
        client.chat.completions.create = AsyncMock(
            side_effect=[err, FakeStream([chunk("ok", "stop")])]
        )
        text = await ai_client.complete_text(
            client, model="m", messages=[{"role": "user", "content": "x"}],
            temperature=0.3, reasoning_effort="high",
        )
        assert text == "ok"
        calls = client.chat.completions.create.await_args_list
        assert len(calls) == 2
        assert calls[0].kwargs.get("extra_body") == {"reasoning_effort": "high"}
        assert "extra_body" not in calls[1].kwargs


# ──────────────── command call sites ────────────────


class TestSummaryCallSites:

    @pytest.mark.asyncio
    async def test_summarize_defaults_thinking_off(self, ix):
        from commands import utility_commands as uc
        from bot_core.history import _chat_history

        gid, cid = 4442000001, 5552000001
        ix.guild_id, ix.channel_id = gid, cid
        _chat_history.setdefault(gid, {})[cid] = [
            {"role": "user", "content": "Hello AI"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        with patch.object(uc, "_make_client") as MockClient:
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(
                return_value=FakeStream([chunk("Summary text", "stop")])
            )
            MockClient.return_value = inst
            await uc.handle_summarize_command(ix)

        kw = inst.chat.completions.create.await_args.kwargs
        assert kw.get("extra_body") == {"enable_thinking": False}

    @pytest.mark.asyncio
    async def test_summarize_honours_env_override(self, ix, monkeypatch):
        from commands import utility_commands as uc
        from bot_core.history import _chat_history

        gid, cid = 4442000002, 5552000002
        ix.guild_id, ix.channel_id = gid, cid
        _chat_history.setdefault(gid, {})[cid] = [
            {"role": "user", "content": "Hello AI"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        monkeypatch.setattr(uc, "SUMMARY_REASONING_EFFORT", "low")
        with patch.object(uc, "_make_client") as MockClient:
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(
                return_value=FakeStream([chunk("Summary text", "stop")])
            )
            MockClient.return_value = inst
            await uc.handle_summarize_command(ix)

        kw = inst.chat.completions.create.await_args.kwargs
        assert kw.get("extra_body") == {"reasoning_effort": "low"}

    @pytest.mark.asyncio
    async def test_translate_defaults_thinking_off(self, ix):
        from commands import utility_commands as uc

        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="Hola mundo"))]
        with patch.object(uc, "_make_client") as MockClient:
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(return_value=resp)
            MockClient.return_value = inst
            await uc.handle_translate_command(
                ix, target_language="Spanish: Hello world", source_language="English"
            )

        kw = inst.chat.completions.create.await_args.kwargs
        assert kw.get("extra_body") == {"enable_thinking": False}

    @pytest.mark.asyncio
    async def test_translate_honours_env_override(self, ix, monkeypatch):
        from commands import utility_commands as uc

        monkeypatch.setattr(uc, "SUMMARY_REASONING_EFFORT", "xhigh")
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="Hola mundo"))]
        with patch.object(uc, "_make_client") as MockClient:
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(return_value=resp)
            MockClient.return_value = inst
            await uc.handle_translate_command(
                ix, target_language="Spanish: Hello world", source_language="English"
            )

        kw = inst.chat.completions.create.await_args.kwargs
        assert kw.get("extra_body") == {"reasoning_effort": "xhigh"}

    @pytest.mark.asyncio
    async def test_ocr_defaults_thinking_off(self, ix):
        from commands import utility_commands as uc

        image = MagicMock()
        image.url = "https://example.com/image.png"
        image.filename = "test_image.png"
        image.read = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n")
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="Extracted text here"))]
        with patch.object(uc, "_make_client") as MockClient:
            inst = MagicMock()
            inst.chat.completions.create = AsyncMock(return_value=resp)
            MockClient.return_value = inst
            await uc.handle_ocr_command(ix, image=image)

        kw = inst.chat.completions.create.await_args.kwargs
        assert kw.get("extra_body") == {"enable_thinking": False}


class TestSessionCallSites:

    @pytest.mark.asyncio
    async def test_recap_defaults_thinking_off(self, monkeypatch):
        from commands import session_commands as sc

        monkeypatch.setattr(sc, "_make_client", lambda: MagicMock())
        monkeypatch.setattr(sc, "_validate_model", AsyncMock(return_value="m"))
        with patch.object(sc, "complete_text", AsyncMock(return_value="recap text")) as ct:
            out = await sc._generate_recap(
                {"name": "T", "notes": [(1.0, "did a thing")]}, None, None
            )
        assert out == "recap text"
        assert ct.await_args.kwargs.get("reasoning_effort") == "off"
        assert "disable_thinking" not in ct.await_args.kwargs

    @pytest.mark.asyncio
    async def test_recap_honours_env_override(self, monkeypatch):
        from commands import session_commands as sc

        monkeypatch.setattr(sc, "_make_client", lambda: MagicMock())
        monkeypatch.setattr(sc, "_validate_model", AsyncMock(return_value="m"))
        monkeypatch.setattr(sc, "SUMMARY_REASONING_EFFORT", "medium")
        with patch.object(sc, "complete_text", AsyncMock(return_value="recap text")) as ct:
            await sc._generate_recap(
                {"name": "T", "notes": [(1.0, "did a thing")]}, None, None
            )
        assert ct.await_args.kwargs.get("reasoning_effort") == "medium"

    @pytest.mark.asyncio
    async def test_merged_log_omits_param_when_unset(self, monkeypatch, tmp_path):
        from commands import session_commands as sc

        # A session dir with one document so the merge actually runs.
        (tmp_path / ".attachments").mkdir()
        (tmp_path / ".attachments" / "log.md").write_text("session text", encoding="utf-8")

        monkeypatch.setattr(sc, "_make_client", lambda: MagicMock())
        monkeypatch.setattr(sc, "_validate_model", AsyncMock(return_value="m"))
        monkeypatch.setattr(sc, "MERGE_LOG_REASONING_EFFORT", None)
        with patch.object(sc, "complete_text", AsyncMock(return_value="merged")) as ct:
            out = await sc._generate_merged_log(
                {"name": "T", "dir": str(tmp_path)}, None, None
            )
        assert out == "merged"
        assert ct.await_args.kwargs.get("reasoning_effort") is None

    @pytest.mark.asyncio
    async def test_merged_log_honours_env_override(self, monkeypatch, tmp_path):
        from commands import session_commands as sc

        (tmp_path / ".attachments").mkdir()
        (tmp_path / ".attachments" / "log.md").write_text("session text", encoding="utf-8")

        monkeypatch.setattr(sc, "_make_client", lambda: MagicMock())
        monkeypatch.setattr(sc, "_validate_model", AsyncMock(return_value="m"))
        monkeypatch.setattr(sc, "MERGE_LOG_REASONING_EFFORT", "high")
        with patch.object(sc, "complete_text", AsyncMock(return_value="merged")) as ct:
            await sc._generate_merged_log(
                {"name": "T", "dir": str(tmp_path)}, None, None
            )
        assert ct.await_args.kwargs.get("reasoning_effort") == "high"
