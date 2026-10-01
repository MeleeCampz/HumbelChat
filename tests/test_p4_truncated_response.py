"""Regression tests for P4: thinking-model truncation (empty answer at max_tokens).

Qwen3 spends part of ``max_tokens`` on internal reasoning *before* the
visible answer. When the budget is exhausted mid-thinking the backend still
completes "successfully" — but with ``content == ""`` and
``finish_reason == "length"``. The old code turned that into the placeholder
``(empty response)`` and even persisted the placeholder into channel history.

Covered here:
  * ``extract_reply_text`` raises :class:`AIResponseTruncatedError` for
    empty-content + ``finish_reason="length"`` (and keeps the legacy
    placeholder behaviour for *ordinary* empty replies).
  * ``ask_ai`` omits ``max_tokens`` by default (model-max output) while still
    honouring an explicit per-character budget, and surfaces a real friendly
    error — never a placeholder, never a persisted empty turn — when the
    model's own output limit truncates the answer (#9: no more 4× retry).
  * ``complete_text`` (shared summary-call helper): omits max_tokens by
    default, passes ``enable_thinking: False`` with a plain-call fallback,
    and warns on partial truncation.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot_core import ai_client
from bot_core.errors import AIResponseTruncatedError, extract_reply_text


# ─────────────────────────── helpers ───────────────────────────


def _resp(content: str | None, *, finish_reason: str | None = None,
          choices_empty: bool = False) -> MagicMock:
    if choices_empty:
        resp = MagicMock()
        resp.choices = []
        return resp
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = MagicMock(content=content)
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _stub_ask_ai_env(monkeypatch, create_side_effect, *, initial_budget=None):
    """Point ask_ai at a stub client + empty KB.

    ``initial_budget`` is the per-request max_tokens: None (default) means
    "not specified" → ask_ai must omit the parameter entirely.
    """
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    client.models.list = AsyncMock(return_value=MagicMock(data=[]))
    monkeypatch.setattr(ai_client, "_make_client", lambda: client)
    monkeypatch.setattr(ai_client, "_validate_model",
                        AsyncMock(return_value="test-model"))
    monkeypatch.setattr(ai_client, "_resolve_request_params",
                        lambda char_obj: (initial_budget, 0.7))
    from kb import retrievers
    monkeypatch.setattr(retrievers, "retrieve_kb_documents",
                        AsyncMock(return_value=[]))
    return client


def _ask(**kw):
    kw.setdefault("user_message", "ping")
    kw.setdefault("model_slug", "test-model")
    kw.setdefault("username", "Alice")
    kw.setdefault("user_id", None)
    return ai_client.ask_ai(**kw)


# ─────────────── extract_reply_text: truncation detection ───────────────


class TestExtractReplyTextTruncation:
    def test_empty_content_with_length_finish_raises_truncation(self):
        with pytest.raises(AIResponseTruncatedError):
            extract_reply_text(_resp(None, finish_reason="length"))
        with pytest.raises(AIResponseTruncatedError):
            extract_reply_text(_resp("", finish_reason="length"))
        with pytest.raises(AIResponseTruncatedError):
            extract_reply_text(_resp("   ", finish_reason="length"))

    def test_partial_content_with_length_finish_is_kept(self):
        # A truncated-but-non-empty answer is a valid (partial) answer —
        # returning it is strictly better than raising.
        assert extract_reply_text(_resp("partial", finish_reason="length")) == "partial"

    def test_empty_content_without_length_finish_uses_legacy_placeholder(self):
        # finish_reason "stop" (or absent): the model finished normally and
        # simply had nothing to say → legacy placeholder behaviour.
        assert extract_reply_text(_resp(None, finish_reason="stop"),
                                  default="(empty)") == "(empty)"
        assert extract_reply_text(_resp(""), default="(empty)") == "(empty)"

    def test_error_carries_friendly_user_message(self):
        err = AIResponseTruncatedError(max_tokens=2000)
        assert "ran out of response budget" in err.user_message
        assert err.category == "truncated_response"
        assert err.max_tokens == 2000


# ─────────────────────── ask_ai: model-max output (#9) ───────────────────────


class TestAskAiModelMaxOutput:
    @pytest.mark.asyncio
    async def test_max_tokens_omitted_when_not_specified(self, monkeypatch):
        """No per-character budget → the parameter is OMITTED so the backend
        (LM Studio) uses the model's maximum output length."""
        client = _stub_ask_ai_env(
            monkeypatch, [_resp("PONG", finish_reason="stop")],
            initial_budget=None,
        )
        reply, _ = await _ask(guild_id=1, channel_id=2)

        assert reply == "PONG"
        assert client.chat.completions.create.await_count == 1
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in kwargs

    @pytest.mark.asyncio
    async def test_per_character_budget_still_honoured(self, monkeypatch):
        client = _stub_ask_ai_env(
            monkeypatch, [_resp("PONG", finish_reason="stop")],
            initial_budget=2048,
        )
        await _ask(guild_id=3, channel_id=4)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_timeout_is_single_ai_timeout_s(self, monkeypatch):
        """The request timeout is the single AI_TIMEOUT_S knob (no more
        prompt/budget-scaled timeouts)."""
        monkeypatch.setattr(ai_client, "AI_TIMEOUT_S", 300.0)
        client = _stub_ask_ai_env(
            monkeypatch, [_resp("PONG", finish_reason="stop")],
            initial_budget=None,
        )
        await _ask(guild_id=5, channel_id=6)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["timeout"] == pytest.approx(300.0)

    @pytest.mark.asyncio
    async def test_truncation_surfaces_real_error_no_retry(self, monkeypatch):
        """With model-max output, finish_reason 'length' means the model's true
        limit was hit — there is nothing bigger to retry at. One call, real
        friendly error, never a placeholder."""
        client = _stub_ask_ai_env(
            monkeypatch,
            [_resp(None, finish_reason="length")],
            initial_budget=None,
        )
        with pytest.raises(ValueError) as ei:
            await _ask(guild_id=7, channel_id=8)

        assert client.chat.completions.create.await_count == 1
        assert "ran out of response budget" in str(ei.value)
        assert "(empty response)" not in str(ei.value)

    @pytest.mark.asyncio
    async def test_failed_turn_persists_no_placeholder_in_history(self, monkeypatch):
        """The P4 data-pollution case: after a hard failure the channel history
        must not contain an assistant turn of '(empty response)'."""
        from bot_core.history import get_history

        _client = _stub_ask_ai_env(
            monkeypatch,
            [_resp(None, finish_reason="length")],
            initial_budget=None,
        )
        with pytest.raises(ValueError):
            await _ask(guild_id=11, channel_id=12)

        history = get_history(11, 12)
        assistant_turns = [m["content"] for m in history
                           if m.get("role") == "assistant"]
        assert "(empty response)" not in assistant_turns


# ─────────────────────── complete_text: summary-call helper (#9) ───────────────────────


class TestCompleteText:
    @pytest.mark.asyncio
    async def test_omits_max_tokens_and_passes_thinking_flag(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_resp("SUMMARY", finish_reason="stop"))

        text = await ai_client.complete_text(
            client, model="m", messages=[{"role": "user", "content": "x"}],
            temperature=0.3, disable_thinking=True,
        )

        assert text == "SUMMARY"
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in kwargs
        assert kwargs["extra_body"] == {"enable_thinking": False}
        assert kwargs["timeout"] == pytest.approx(ai_client.AI_TIMEOUT_S)

    @pytest.mark.asyncio
    async def test_explicit_max_tokens_passed(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_resp("SUMMARY", finish_reason="stop"))

        await ai_client.complete_text(
            client, model="m", messages=[], temperature=0.3, max_tokens=4096,
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 4096
        assert "extra_body" not in kwargs

    @pytest.mark.asyncio
    async def test_thinking_param_rejected_falls_back_to_plain(self, monkeypatch):
        """Backend rejects the extra param → one plain retry without it."""
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            RuntimeError("unknown parameter: enable_thinking"),
            _resp("SUMMARY", finish_reason="stop"),
        ])

        text = await ai_client.complete_text(
            client, model="m", messages=[], temperature=0.3,
            disable_thinking=True,
        )

        assert text == "SUMMARY"
        assert client.chat.completions.create.await_count == 2
        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        second_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert first_kwargs["extra_body"] == {"enable_thinking": False}
        assert "extra_body" not in second_kwargs

    @pytest.mark.asyncio
    async def test_empty_answer_at_model_limit_raises(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_resp(None, finish_reason="length"))

        with pytest.raises(ValueError) as ei:
            await ai_client.complete_text(
                client, model="m", messages=[], temperature=0.3,
            )
        assert "ran out of response budget" in str(ei.value)

    @pytest.mark.asyncio
    async def test_partial_answer_at_model_limit_is_returned(self, monkeypatch):
        """A truncated-but-non-empty answer is a valid partial answer."""
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_resp("partial summary", finish_reason="length"))

        text = await ai_client.complete_text(
            client, model="m", messages=[], temperature=0.3,
        )
        assert text == "partial summary"
