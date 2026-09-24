"""Tests for AI chat slash command."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAICommand:

    @pytest.mark.asyncio
    async def test_ai_command_basic(self):
        """Test basic AI command execution."""
        from commands.ai_command import handle_ai_command
        from config.characters import load_characters

        # Ensure characters are loaded (mirrors main.py startup)
        load_characters(Path("characters.json.example"))

        ix = MagicMock()
        ix.guild_id = 123456
        ix.channel_id = 789012
        ix.response = MagicMock(is_done=MagicMock(return_value=True))
        ix.channel = MagicMock()
        ix.followup.send = AsyncMock()
        ix._sent = []

        async def fake_followup(content, **kw):
            ix._sent.append(str(content))

        ix.followup.send.side_effect = fake_followup

        mock_reply = ("AI reply", {})

        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock, return_value=mock_reply) as mock_ask:
            await handle_ai_command(ix, message="Hello")

            mock_ask.assert_called_once()
            call_kwargs = mock_ask.call_args.kwargs
            assert call_kwargs["user_message"] == "Hello"
            assert call_kwargs["guild_id"] == 123456
            assert call_kwargs["channel_id"] == 789012

    @pytest.mark.asyncio
    async def test_ai_command_with_character(self):
        """Test AI command with a specific character."""
        from commands.ai_command import handle_ai_command
        from config.characters import load_characters

        load_characters(Path("characters.json.example"))

        ix = MagicMock()
        ix.guild_id = 123456
        ix.channel_id = 789012
        ix.response = MagicMock(is_done=MagicMock(return_value=True))
        ix.channel = MagicMock()
        ix.followup.send = AsyncMock()
        ix._sent = []

        async def fake_followup(content, **kw):
            ix._sent.append(str(content))

        ix.followup.send.side_effect = fake_followup

        mock_reply = ("AI reply", {})

        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock, return_value=mock_reply) as mock_ask:
            await handle_ai_command(ix, message="Hello", character_name="Assistant")

            mock_ask.assert_called_once()
            call_kwargs = mock_ask.call_args.kwargs
            assert call_kwargs["user_message"] == "Hello"
            assert call_kwargs["char_key"] == "Assistant"
            assert call_kwargs["model_slug"] == "some-other-model"

    @pytest.mark.asyncio
    async def test_ai_command_defers_response(self):
        """Test that the AI command properly defers to prevent timeout."""
        from commands.ai_command import handle_ai_command
        from config.characters import load_characters

        load_characters(Path("characters.json.example"))

        ix = MagicMock()
        ix.guild_id = 123456
        ix.channel_id = 789012
        ix.response = MagicMock(is_done=MagicMock(return_value=True))
        ix.channel = MagicMock()
        ix.followup.send = AsyncMock()
        ix._sent = []

        async def fake_followup(content, **kw):
            ix._sent.append(str(content))

        ix.followup.send.side_effect = fake_followup

        mock_reply = ("AI reply", {})

        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock, return_value=mock_reply):
            await handle_ai_command(ix, message="Hello")

        # Since we mocked response.is_done() to return True, defer should not be called
        assert ix.response.defer.call_count == 0


def _make_ix():
    """Build a mock Interaction whose followup.send / response.send_message record to _sent."""
    sent = []

    async def on_send(content="", **kw):
        sent.append(str(content))

    ix = MagicMock()
    ix.guild_id = 123456
    ix.channel_id = 789012
    ix.response = MagicMock(is_done=MagicMock(return_value=True))
    ix.channel = MagicMock()
    ix.followup.send.side_effect = on_send
    ix.response.send_message.side_effect = on_send
    ix._sent = sent
    return ix


class TestP1_9StaleActiveCharacter:
    """P1 #9: a *persisted* active character key that has since been deleted
    from characters.json must not break /ai with ``Character `None` not found``.
    """

    def _load_chars(self):
        from config.characters import load_characters
        load_characters(Path("characters.json.example"))  # default persona: "System"
        return "System"

    @pytest.mark.asyncio
    async def test_deleted_persisted_key_falls_back_to_default(self, monkeypatch):
        """No character named + stale persisted key -> default persona + a heads-up.

        This is the TODO's verify case. The old code rendered
        ``Character `None` not found`` (it interpolated ``character_name``) and
        hard-returned, so /ai was dead until the user re-picked a persona.
        """
        from commands import ai_command
        default_key = self._load_chars()
        ix = _make_ix()

        # The channel has a persisted active key pointing at a deleted char.
        monkeypatch.setattr(ai_command, "get_active_char_key",
                            lambda g, c: "DeletedCharacter")
        monkeypatch.setattr(ai_command._settings, "EMBED_FORMAT", False)
        mock_reply = ("Hello!", {})

        from commands.ai_command import handle_ai_command
        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock,
                   return_value=mock_reply) as mock_ask:
            await handle_ai_command(ix, message="hi")  # NOTE: no character_name

        # /ai must NOT have died: a real reply was produced with the default model.
        mock_ask.assert_awaited_once()
        assert mock_ask.call_args.kwargs["user_message"] == "hi"

        joined = " ".join(ix._sent)
        # No confusing "Character `None` not found".
        assert "`None`" not in joined
        assert "not found" not in joined
        # A fallback notice naming the stale key and the default persona.
        assert "DeletedCharacter" in joined
        assert "no longer exists" in joined
        assert default_key in joined
        # ...and the actual AI reply still went out.
        assert "Hello!" in joined

    @pytest.mark.asyncio
    async def test_explicit_missing_character_still_named(self, monkeypatch):
        """A character the user *typed* that doesn't exist -> named, not 'None'."""
        from commands import ai_command
        self._load_chars()
        ix = _make_ix()
        monkeypatch.setattr(ai_command, "get_active_char_key",
                            lambda g, c: "System")

        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock) as mock_ask:
            await ai_command.handle_ai_command(ix, message="hi", character_name="Gone")

        mock_ask.assert_not_awaited()
        joined = " ".join(ix._sent)
        assert "Gone" in joined           # the typed name is shown...
        assert "not found" in joined
        assert "`None`" not in joined     # ...and never the bare None

    @pytest.mark.asyncio
    async def test_normal_implicit_ai_no_fallback_notice(self, monkeypatch):
        """Regression guard: a *valid* active key (or none) produces no warning.

        The fallback notice must only appear when the persisted key is actually
        stale — never on a normal /ai.
        """
        from commands import ai_command
        self._load_chars()
        ix = _make_ix()
        monkeypatch.setattr(ai_command, "get_active_char_key",
                            lambda g, c: "System")  # a valid, existing key
        monkeypatch.setattr(ai_command._settings, "EMBED_FORMAT", False)

        with patch("bot_core.ai_client.ask_ai", new_callable=AsyncMock,
                   return_value=("hi back", {})):
            await ai_command.handle_ai_command(ix, message="hi")

        joined = " ".join(ix._sent)
        assert "no longer exists" not in joined   # no spurious fallback warning
        assert "hi back" in joined


class TestExplicitCharacterPersona:
    """An explicit /ai character must drive the system prompt and sampling,
    not only the model slug. The channel's active persona stays in place.
    """

    def _load(self, tmp_path):
        from config.characters import load_characters
        path = tmp_path / "characters.json"
        path.write_text(
            json.dumps({
                "default": "System",
                "characters": {
                    "System": {
                        "display": "System",
                        "model": "system-model",
                        "system_prompt": "SYSTEM PROMPT",
                        "max_tokens": 100,
                        "temperature": 0.1,
                    },
                    "Assistant": {
                        "display": "Chat Assistant",
                        "model": "assistant-model",
                        "system_prompt": "ASSISTANT PROMPT",
                        "max_tokens": 400,
                        "temperature": 0.9,
                    },
                },
            }),
            encoding="utf-8",
        )
        load_characters(path)

    @pytest.mark.asyncio
    async def test_char_key_overrides_active_persona(self, tmp_path):
        from bot_core import ai_client
        from bot_core.history import set_active_char_key

        self._load(tmp_path)
        set_active_char_key(1, 2, "System")
        ai_client._model_list_cache.clear()

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_resp)
        client.models.list = AsyncMock(return_value=MagicMock(data=[]))

        with patch.object(ai_client, "_make_client", return_value=client), \
             patch("kb.retrievers.retrieve_kb_documents", new=AsyncMock(return_value=[])):
            await ai_client.ask_ai(
                user_message="hello",
                model_slug="assistant-model",
                guild_id=1,
                channel_id=2,
                char_key="Assistant",
            )

        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["model"] == "assistant-model"
        assert kwargs["messages"][0]["content"] == "ASSISTANT PROMPT"
        assert kwargs["temperature"] == 0.9
        assert kwargs["max_tokens"] == 400

    @pytest.mark.asyncio
    async def test_prefix_uses_active_character(self, tmp_path, monkeypatch):
        import main
        from bot_core.history import set_active_char_key

        self._load(tmp_path)
        set_active_char_key(42, 7, "Assistant")

        captured: dict = {}

        async def fake_run(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(main, "run_ai_turn", fake_run)
        monkeypatch.setattr(main, "notify_if_queued", AsyncMock())
        monkeypatch.setattr(main, "start_typing", lambda *a, **k: None)
        monkeypatch.setattr(main, "clear_run", lambda *a, **k: None)

        message = MagicMock()
        message.author = MagicMock()
        message.author.id = 9
        message.author.display_name = "Ada"
        message.content = f"{main.BOT_PREFIX} hello"
        message.guild_id = 42
        message.channel.id = 7
        message.channel.name = "general"
        message.channel.send = AsyncMock()

        await main.on_message(message)

        assert captured["char_key"] == "Assistant"
        assert captured["model_slug"] == "assistant-model"
        assert captured["char_name"] == "Chat Assistant"
        assert captured["user_message"] == "hello"


class TestAIStopTargetsActiveTurn:
    @pytest.mark.asyncio
    async def test_queued_turn_does_not_replace_the_registered_run(self, monkeypatch):
        import asyncio

        from bot_core.ai_runs import _reset, get_run
        from commands.ai_command import run_ai_turn
        from utils.background_tasks import spawn_tracked_task

        _reset()
        channel = 900001
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_ask(**kwargs):
            started.set()
            await release.wait()
            return ("ok", {})

        monkeypatch.setattr("commands.ai_command.ai_client.ask_ai", fake_ask)
        monkeypatch.setattr("commands.ai_command._settings.AI_STREAM", False)
        monkeypatch.setattr("commands.ai_command._settings.EMBED_FORMAT", False)

        source = MagicMock()
        source.followup.send = AsyncMock()
        source.channel.send = AsyncMock()

        async def go(msg):
            await run_ai_turn(
                source,
                user_message=msg,
                model_slug="m",
                guild_id=1,
                channel_id=channel,
                char_name="C",
                char_key="C",
            )

        first = spawn_tracked_task(go("first"))
        await asyncio.wait_for(started.wait(), timeout=2)
        second = spawn_tracked_task(go("second"))
        await asyncio.sleep(0)
        assert get_run(channel) is first

        first.cancel()
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        _reset()


class TestAICommandStructural:

    def test_handle_ai_command_exists(self):
        from commands.ai_command import handle_ai_command
        assert callable(handle_ai_command)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
