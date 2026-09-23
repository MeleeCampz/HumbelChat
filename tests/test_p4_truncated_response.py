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
  * ``ask_ai`` retries ONCE with a bigger budget when truncated, and surfaces
    a real friendly error (never a placeholder, never a persisted empty
    turn) when the bigger budget is still truncated or already the cap.
  * ``_scaled_timeout`` accounts for large output budgets.
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


def _stub_ask_ai_env(monkeypatch, create_side_effect, *, initial_budget: int,
                     hard_cap: int = 4096):
    """Point ask_ai at a stub client + empty KB with a KNOWN token budget."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    client.models.list = AsyncMock(return_value=MagicMock(data=[]))
    monkeypatch.setattr(ai_client, "_make_client", lambda: client)
    monkeypatch.setattr(ai_client, "_validate_model",
                        AsyncMock(return_value="test-model"))
    monkeypatch.setattr(ai_client, "_resolve_request_params",
                        lambda char_obj: (initial_budget, 0.7))
    monkeypatch.setattr(ai_client, "MAX_TOKENS_HARD_CAP", hard_cap)
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


# ─────────────────────── ask_ai: budget retry ───────────────────────


class TestAskAiBudgetRetry:
    @pytest.mark.asyncio
    async def test_truncation_then_retry_with_bigger_budget(self, monkeypatch):
        """The P4 core case: first call truncated → one retry at the hard cap."""
        client = _stub_ask_ai_env(
            monkeypatch,
            [_resp(None, finish_reason="length"), _resp("REAL ANSWER")],
            initial_budget=1024, hard_cap=4096,
        )
        reply, _ = await _ask(guild_id=1, channel_id=2)

        assert reply == "REAL ANSWER"
        assert client.chat.completions.create.await_count == 2
        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        second_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert first_kwargs["max_tokens"] == 1024
        # Retry budget: min(hard_cap, 4× initial) = min(4096, 4096) = 4096.
        assert second_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_persistent_truncation_surfaces_real_error_not_placeholder(
            self, monkeypatch):
        """Two truncated calls → friendly error; no placeholder is returned."""
        client = _stub_ask_ai_env(
            monkeypatch,
            [_resp(None, finish_reason="length"),
             _resp(None, finish_reason="length")],
            initial_budget=1024, hard_cap=4096,
        )
        with pytest.raises(ValueError) as ei:
            await _ask(guild_id=3, channel_id=4)

        assert client.chat.completions.create.await_count == 2
        # A REAL error message, not the old "(empty response)" placeholder.
        assert "ran out of response budget" in str(ei.value)
        assert "(empty response)" not in str(ei.value)

    @pytest.mark.asyncio
    async def test_truncation_at_hard_cap_is_not_retried(self, monkeypatch):
        """Budget already at the cap → nothing bigger to try → one call only."""
        client = _stub_ask_ai_env(
            monkeypatch,
            [_resp(None, finish_reason="length")],
            initial_budget=4096, hard_cap=4096,
        )
        with pytest.raises(ValueError) as ei:
            await _ask(guild_id=5, channel_id=6)

        assert client.chat.completions.create.await_count == 1
        assert "ran out of response budget" in str(ei.value)

    @pytest.mark.asyncio
    async def test_failed_retry_call_is_classified_too(self, monkeypatch):
        """If the retry CALL itself fails (e.g. timeout), the user still gets
        a friendly classified error, not a raw traceback."""
        import httpx
        import openai
        req = httpx.Request("POST", "http://backend.invalid/v1")
        timeout_exc = openai.APITimeoutError(request=req)
        client = _stub_ask_ai_env(
            monkeypatch,
            # 1st call: truncated. Retry call: transient timeout on BOTH of
            # its internal attempts (the retry helper retries transient
            # failures once), then it propagates.
            [_resp(None, finish_reason="length"), timeout_exc, timeout_exc],
            initial_budget=1024, hard_cap=4096,
        )
        with patch.object(ai_client, "_COMPLETION_RETRY_BACKOFF_S", 0.01):
            with pytest.raises(ValueError) as ei:
                await _ask(guild_id=7, channel_id=8)

        assert "too long to respond" in str(ei.value)
        # 1 initial call + 2 internal attempts of the (failed) retry call.
        assert client.chat.completions.create.await_count == 3

    @pytest.mark.asyncio
    async def test_successful_path_unchanged_single_call(self, monkeypatch):
        client = _stub_ask_ai_env(
            monkeypatch, [_resp("PONG", finish_reason="stop")],
            initial_budget=1024, hard_cap=4096,
        )
        reply, _ = await _ask(guild_id=9, channel_id=10)
        assert reply == "PONG"
        assert client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_failed_turn_persists_no_placeholder_in_history(self, monkeypatch):
        """The P4 data-pollution case: after a hard failure the channel history
        must not contain an assistant turn of '(empty response)'."""
        from bot_core.history import get_history

        _client = _stub_ask_ai_env(
            monkeypatch,
            [_resp(None, finish_reason="length"),
             _resp(None, finish_reason="length")],
            initial_budget=1024, hard_cap=4096,
        )
        with pytest.raises(ValueError):
            await _ask(guild_id=11, channel_id=12)

        history = get_history(11, 12)
        assistant_turns = [m["content"] for m in history
                           if m.get("role") == "assistant"]
        assert "(empty response)" not in assistant_turns

    @pytest.mark.asyncio
    async def test_successful_retry_persists_real_answer(self, monkeypatch):
        """After a successful budget-retry the REAL answer (not a placeholder)
        lands in the channel history."""
        from bot_core.history import get_history

        _client = _stub_ask_ai_env(
            monkeypatch,
            [_resp(None, finish_reason="length"), _resp("REAL ANSWER")],
            initial_budget=1024, hard_cap=4096,
        )
        await _ask(guild_id=13, channel_id=14)

        history = get_history(13, 14)
        assistant_turns = [m["content"] for m in history
                           if m.get("role") == "assistant"]
        assert assistant_turns == ["REAL ANSWER"]


# ─────────────────────── scaled timeout for big budgets ───────────────────────


class TestScaledTimeoutWithBudget:
    def test_scales_with_prompt_only(self, monkeypatch):
        # Backwards-compatible: no budget → prompt-only scaling (unchanged).
        monkeypatch.setattr(ai_client, "REQUEST_TIMEOUT", 120)
        assert ai_client._scaled_timeout(2000) == pytest.approx(121.0)

    def test_grows_with_max_tokens(self, monkeypatch):
        monkeypatch.setattr(ai_client, "REQUEST_TIMEOUT", 120)
        # 120 base + 0.5 (2000 chars) + 4.0 (8000 tokens) = 125
        assert ai_client._scaled_timeout(2000, 8000) == pytest.approx(125.0)

    def test_capped_at_eight_times_base(self, monkeypatch):
        monkeypatch.setattr(ai_client, "REQUEST_TIMEOUT", 120)
        huge = ai_client._scaled_timeout(1_000_000, 1_000_000)
        assert huge == 120 * 8  # cap, not 120 + 500 + 50
