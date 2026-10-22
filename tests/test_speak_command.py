"""#34: /speak — TTS read-aloud via an OpenAI-compatible /v1/audio/speech endpoint.

Covers the command handler (commands/speak_command.py) and the client module
(bot_core/tts.py). No network: the OpenAI client is mocked throughout.
"""
from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── /speak command handler ─────────────────────────────────────────────

class TestSpeakCommand:

    def _ix_with_files(self, ix):
        """Re-wire followup.send to also capture the ``file=`` kwarg."""
        sent = []
        files = []

        async def on_send(content="", *, file=None, **kw):
            sent.append(str(content))
            if file is not None:
                files.append(file)

        ix.followup.send.side_effect = on_send
        return sent, files

    @pytest.mark.asyncio
    async def test_not_configured(self, ix, monkeypatch):
        from commands import speak_command
        monkeypatch.setattr(speak_command, "tts_configured", lambda: False)
        monkeypatch.setattr(speak_command, "synthesize", AsyncMock())

        await speak_command.handle_speak_command(ix, text="hello")

        assert len(ix._sent) == 1
        assert "not configured" in ix._sent[0]
        assert "TTS_MODEL" in ix._sent[0]
        speak_command.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_sends_mp3_file(self, ix, monkeypatch):
        from commands import speak_command
        monkeypatch.setattr(speak_command, "tts_configured", lambda: True)
        monkeypatch.setattr(
            speak_command, "synthesize", AsyncMock(return_value=b"fake-mp3-bytes")
        )
        sent, files = self._ix_with_files(ix)

        await speak_command.handle_speak_command(ix, text="Hello, world!")

        assert len(sent) == 1
        assert sent[0].startswith("🔊")
        assert "Hello, world!" in sent[0]
        assert len(files) == 1
        assert re.fullmatch(r"speak-\d+\.mp3", files[0].filename)
        assert files[0].fp.read() == b"fake-mp3-bytes"

    @pytest.mark.asyncio
    async def test_long_caption_is_truncated(self, ix, monkeypatch):
        from commands import speak_command
        monkeypatch.setattr(speak_command, "tts_configured", lambda: True)
        monkeypatch.setattr(
            speak_command, "synthesize", AsyncMock(return_value=b"audio")
        )
        sent, _ = self._ix_with_files(ix)

        await speak_command.handle_speak_command(ix, text="x" * 200)

        assert len(sent) == 1
        assert sent[0].endswith("…")
        # caption preview is capped (80 chars + ellipsis), not the full text
        assert "x" * 90 not in sent[0]

    @pytest.mark.asyncio
    async def test_empty_text_rejected(self, ix, monkeypatch):
        from commands import speak_command
        monkeypatch.setattr(speak_command, "tts_configured", lambda: True)
        monkeypatch.setattr(speak_command, "synthesize", AsyncMock())

        await speak_command.handle_speak_command(ix, text="   ")

        assert len(ix._sent) == 1
        assert "Nothing to read" in ix._sent[0]
        speak_command.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_too_long_rejected_before_api_call(self, ix, monkeypatch):
        from commands import speak_command
        monkeypatch.setattr(speak_command, "tts_configured", lambda: True)
        monkeypatch.setattr(speak_command, "TTS_MAX_CHARS", 10)
        monkeypatch.setattr(speak_command, "synthesize", AsyncMock())

        await speak_command.handle_speak_command(ix, text="x" * 50)

        assert len(ix._sent) == 1
        assert "Too long" in ix._sent[0]
        assert "10" in ix._sent[0]
        speak_command.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_backend_failure_is_friendly(self, ix, monkeypatch):
        from bot_core.tts import TTSError
        from commands import speak_command
        monkeypatch.setattr(speak_command, "tts_configured", lambda: True)
        monkeypatch.setattr(
            speak_command, "TTS_MODEL", "kokoro-82m-v1.0-gguf"
        )
        monkeypatch.setattr(
            speak_command,
            "synthesize",
            AsyncMock(side_effect=TTSError("ConnectionError: backend down")),
        )

        await speak_command.handle_speak_command(ix, text="hello")

        assert len(ix._sent) == 1
        assert "TTS failed" in ix._sent[0]
        assert "kokoro-82m-v1.0-gguf" in ix._sent[0]


# ── bot_core.tts client module ────────────────────────────────────────

class TestTtsClient:

    def _fake_client(self, audio_bytes=b"mp3"):
        """A stand-in for the AsyncOpenAI client with an audio.speech route."""
        resp = MagicMock()
        resp.read.return_value = audio_bytes
        client = MagicMock()
        client.audio.speech.create = AsyncMock(return_value=resp)
        return client

    @pytest.mark.asyncio
    async def test_unconfigured_raises_value_error(self, monkeypatch):
        import bot_core.tts as tts
        monkeypatch.setattr(tts, "TTS_MODEL", "")

        with pytest.raises(ValueError, match="disabled"):
            await tts.synthesize("hello")

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio_bytes(self, monkeypatch):
        import bot_core.tts as tts
        monkeypatch.setattr(tts, "TTS_MODEL", "kokoro-82m-v1.0-gguf")
        client = self._fake_client(b"audio-bytes")
        monkeypatch.setattr(tts, "_tts_client", client)

        audio = await tts.synthesize("hello world")

        assert audio == b"audio-bytes"
        _, kwargs = client.audio.speech.create.call_args
        assert kwargs["model"] == "kokoro-82m-v1.0-gguf"
        assert kwargs["voice"] == tts.TTS_VOICE
        assert kwargs["input"] == "hello world"
        assert kwargs["response_format"] == "mp3"

    @pytest.mark.asyncio
    async def test_x_vector_only_mode_sent_when_enabled(self, monkeypatch):
        import bot_core.tts as tts
        monkeypatch.setattr(tts, "TTS_MODEL", "kokoro")
        monkeypatch.setattr(tts, "TTS_X_VECTOR_ONLY_MODE", True)
        client = self._fake_client(b"audio-bytes")
        monkeypatch.setattr(tts, "_tts_client", client)

        await tts.synthesize("hello")

        _, kwargs = client.audio.speech.create.call_args
        assert kwargs["extra_body"] == {
            "audio_options": {"x_vector_only_mode": True}
        }

    @pytest.mark.asyncio
    async def test_x_vector_only_mode_omitted_when_disabled(self, monkeypatch):
        import bot_core.tts as tts
        monkeypatch.setattr(tts, "TTS_MODEL", "kokoro")
        monkeypatch.setattr(tts, "TTS_X_VECTOR_ONLY_MODE", False)
        client = self._fake_client(b"audio-bytes")
        monkeypatch.setattr(tts, "_tts_client", client)

        await tts.synthesize("hello")

        _, kwargs = client.audio.speech.create.call_args
        assert kwargs.get("extra_body") is None

    @pytest.mark.asyncio
    async def test_empty_response_raises(self, monkeypatch):
        import bot_core.tts as tts
        monkeypatch.setattr(tts, "TTS_MODEL", "kokoro")
        client = self._fake_client(b"")
        monkeypatch.setattr(tts, "_tts_client", client)

        with pytest.raises(tts.TTSError, match="no audio"):
            await tts.synthesize("hello")

    @pytest.mark.asyncio
    async def test_backend_error_wrapped_as_ttserro(self, monkeypatch):
        import bot_core.tts as tts
        from openai import APIConnectionError
        monkeypatch.setattr(tts, "TTS_MODEL", "kokoro")
        client = MagicMock()
        client.audio.speech.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )
        monkeypatch.setattr(tts, "_tts_client", client)

        with pytest.raises(tts.TTSError, match="APIConnectionError"):
            await tts.synthesize("hello")

    @pytest.mark.asyncio
    async def test_not_loaded_triggers_reload_and_single_retry(self, monkeypatch):
        """Studio idle-unloaded the model: load it next to chat, retry once."""
        import bot_core.tts as tts
        from openai import APIStatusError
        monkeypatch.setattr(tts, "TTS_MODEL", "kokoro")

        resp = MagicMock()
        resp.read.return_value = b"audio"
        client = MagicMock()
        client.audio.speech.create = AsyncMock(
            side_effect=[APIStatusError("No model loaded.", response=MagicMock(), body=None), resp]
        )
        monkeypatch.setattr(tts, "_tts_client", client)
        ensure = AsyncMock(return_value=True)
        monkeypatch.setattr(tts, "_ensure_model_loaded", ensure)

        assert await tts.synthesize("hello") == b"audio"
        ensure.assert_awaited_once()
        assert client.audio.speech.create.await_count == 2

    @pytest.mark.asyncio
    async def test_not_loaded_without_studio_layout_propagates(self, monkeypatch):
        """A plain OpenAI-compatible TTS server has no load endpoint — surface the error."""
        import bot_core.tts as tts
        from openai import APIStatusError
        monkeypatch.setattr(tts, "TTS_MODEL", "kokoro")
        monkeypatch.setattr(tts, "TTS_URL", "http://kokoro-server:8000")  # no /v1

        client = MagicMock()
        client.audio.speech.create = AsyncMock(
            side_effect=APIStatusError("No model loaded.", response=MagicMock(), body=None)
        )
        monkeypatch.setattr(tts, "_tts_client", client)
        ensure = AsyncMock(return_value=False)
        monkeypatch.setattr(tts, "_ensure_model_loaded", ensure)

        with pytest.raises(tts.TTSError, match="No model loaded"):
            await tts.synthesize("hello")
        ensure.assert_awaited_once()
        assert client.audio.speech.create.await_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_other_errors_do_not_trigger_reload(self, monkeypatch):
        import bot_core.tts as tts
        from openai import APIConnectionError
        monkeypatch.setattr(tts, "TTS_MODEL", "kokoro")

        client = MagicMock()
        client.audio.speech.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )
        monkeypatch.setattr(tts, "_tts_client", client)
        ensure = AsyncMock(return_value=True)
        monkeypatch.setattr(tts, "_ensure_model_loaded", ensure)

        with pytest.raises(tts.TTSError):
            await tts.synthesize("hello")
        ensure.assert_not_awaited()

    def test_unsloth_api_base_detection(self, monkeypatch):
        import bot_core.tts as tts
        monkeypatch.setattr(tts, "TTS_URL", "http://host.docker.internal:8888/v1")
        assert tts._unsloth_api_base() == "http://host.docker.internal:8888"
        monkeypatch.setattr(tts, "TTS_URL", "http://kokoro-server:8000")
        assert tts._unsloth_api_base() is None

    @staticmethod
    def _fake_httpx(monkeypatch, model: str) -> dict[str, object]:
        """Patch tts.httpx.AsyncClient; returns the recorded load payload."""
        import bot_core.tts as tts

        sent: dict[str, object] = {}

        class _Resp:
            def json(self):
                return {"data": [{"id": model, "loaded": True}]}

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                sent["payload"] = json

            async def get(self, url, headers=None):
                return _Resp()

        monkeypatch.setattr(tts.httpx, "AsyncClient", _Client)
        return sent

    @pytest.mark.asyncio
    async def test_load_pins_configured_gpus(self, monkeypatch):
        import bot_core.tts as tts
        monkeypatch.setattr(tts, "TTS_MODEL", "qwen3-tts")
        monkeypatch.setattr(tts, "TTS_URL", "http://studio:8888/v1")
        monkeypatch.setattr(tts, "TTS_GPU_IDS", [1])
        sent = self._fake_httpx(monkeypatch, "qwen3-tts")

        assert await tts._ensure_model_loaded() is True
        # pinned to its own GPU — default placement would evict the chat model
        assert sent["payload"] == {"model_path": "qwen3-tts", "gpu_ids": [1]}

    @pytest.mark.asyncio
    async def test_load_omits_placement_when_unconfigured(self, monkeypatch):
        import bot_core.tts as tts
        monkeypatch.setattr(tts, "TTS_MODEL", "qwen3-tts")
        monkeypatch.setattr(tts, "TTS_URL", "http://studio:8888/v1")
        monkeypatch.setattr(tts, "TTS_GPU_IDS", [])
        sent = self._fake_httpx(monkeypatch, "qwen3-tts")

        assert await tts._ensure_model_loaded() is True
        assert sent["payload"] == {"model_path": "qwen3-tts"}

    @pytest.mark.asyncio
    async def test_timeout_wrapped_as_ttserro(self, monkeypatch):
        import bot_core.tts as tts
        monkeypatch.setattr(tts, "TTS_MODEL", "kokoro")
        monkeypatch.setattr(tts, "TTS_TIMEOUT_S", 0.01)

        async def _never(**_kw):
            await asyncio.sleep(10)

        client = MagicMock()
        client.audio.speech.create = _never
        monkeypatch.setattr(tts, "_tts_client", client)

        with pytest.raises(tts.TTSError, match="timed out"):
            await tts.synthesize("hello")


# ── settings fallbacks (reload-based, pattern of test_settings_numeric_env.py) ──

class TestTtsSettingsFallbacks:
    def test_tts_url_defaults_to_infer_url(self, monkeypatch):
        import importlib
        import config.settings as S
        monkeypatch.setenv("INFER_URL", "http://tts-host:8888/v1")
        # The host .env (loaded by main.py's load_dotenv in other tests) may set
        # these — pin them so the fallback under test is unambiguous.
        for var in ("TTS_MODEL", "TTS_VOICE", "TTS_URL", "TTS_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        importlib.reload(S)
        assert S.TTS_URL == "http://tts-host:8888/v1"
        assert S.TTS_MODEL == ""
        assert S.TTS_VOICE == "default"

    def test_tts_api_key_defaults_to_infer_api_key(self, monkeypatch):
        import importlib
        import config.settings as S
        monkeypatch.setenv("INFER_API_KEY", "sk-local")
        monkeypatch.delenv("TTS_URL", raising=False)
        monkeypatch.delenv("TTS_API_KEY", raising=False)
        importlib.reload(S)
        assert S.TTS_API_KEY == "sk-local"

    def test_tts_numeric_envs_respected(self, monkeypatch):
        import importlib
        import config.settings as S
        monkeypatch.setenv("TTS_TIMEOUT_S", "45")
        monkeypatch.setenv("TTS_MAX_CHARS", "1234")
        importlib.reload(S)
        assert S.TTS_TIMEOUT_S == 45.0
        assert S.TTS_MAX_CHARS == 1234

    def test_tts_numeric_garbage_falls_back_to_defaults(self, monkeypatch):
        import importlib
        import config.settings as S
        monkeypatch.setenv("TTS_TIMEOUT_S", "abc")
        monkeypatch.setenv("TTS_MAX_CHARS", "")
        importlib.reload(S)  # must NOT raise
        assert S.TTS_TIMEOUT_S == 120.0
        assert S.TTS_MAX_CHARS == 5000
