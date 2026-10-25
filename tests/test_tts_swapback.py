"""Chat-model swap-back after /speak (bot_core.tts).

A GPU TTS load evicts the resident chat model; the bot must bring it back in
the background so the next question doesn't pay the 30–90 s cold load.
"""
from __future__ import annotations

import asyncio

import pytest

import bot_core.tts as tts


@pytest.fixture(autouse=True)
def _reset_reload_task():
    tts._chat_reload_task = None
    yield
    task = tts._chat_reload_task
    tts._chat_reload_task = None
    if task is not None and not task.done():
        task.cancel()


class TestEnsureSlugLoaded:
    async def test_non_studio_url_returns_false(self, monkeypatch):
        monkeypatch.setattr(tts, "TTS_URL", "http://other-backend/api")
        assert await tts._ensure_slug_loaded("some/model", 1) is False

    async def test_empty_model_path_returns_false(self, monkeypatch):
        monkeypatch.setattr(tts, "TTS_URL", "http://studio:8888/v1")
        assert await tts._ensure_slug_loaded("", 1) is False


class TestScheduleChatSwapBack:
    async def test_noop_without_chat_model(self, monkeypatch):
        monkeypatch.setattr(tts, "DEFAULT_MODEL", None)
        monkeypatch.setattr(tts, "TTS_URL", "http://studio:8888/v1")
        tts._schedule_chat_swap_back()
        assert tts._chat_reload_task is None

    async def test_noop_without_studio_layout(self, monkeypatch):
        monkeypatch.setattr(tts, "DEFAULT_MODEL", "unsloth/model")
        monkeypatch.setattr(tts, "TTS_URL", "http://other-backend/api")
        tts._schedule_chat_swap_back()
        assert tts._chat_reload_task is None

    async def test_schedules_and_dedupes(self, monkeypatch):
        monkeypatch.setattr(tts, "DEFAULT_MODEL", "unsloth/model")
        monkeypatch.setattr(tts, "TTS_URL", "http://studio:8888/v1")
        # The reload task itself is stubbed out — we test scheduling, not HTTP.
        started: list[bool] = []

        async def fake_reload() -> None:
            started.append(True)
            tts._chat_reload_task = None

        monkeypatch.setattr(tts, "_swap_chat_model_back", fake_reload)

        tts._schedule_chat_swap_back()
        first = tts._chat_reload_task
        assert first is not None
        tts._schedule_chat_swap_back()  # in flight → must NOT double-schedule
        assert tts._chat_reload_task is first

        await asyncio.sleep(0)  # let the task run
        assert started == [True]
        assert tts._chat_reload_task is None

    async def test_reschedules_after_previous_finished(self, monkeypatch):
        monkeypatch.setattr(tts, "DEFAULT_MODEL", "unsloth/model")
        monkeypatch.setattr(tts, "TTS_URL", "http://studio:8888/v1")

        async def fake_reload() -> None:
            tts._chat_reload_task = None

        monkeypatch.setattr(tts, "_swap_chat_model_back", fake_reload)

        tts._schedule_chat_swap_back()
        first = tts._chat_reload_task
        await asyncio.sleep(0)
        assert first is not None and first.done()

        tts._schedule_chat_swap_back()  # previous done → may schedule again
        assert tts._chat_reload_task is not None
        assert tts._chat_reload_task is not first


class TestOllamaBackend:
    """#57: TTS_BACKEND=ollama (phonon via the Ollama container)."""

    async def test_ensure_model_loaded_is_noop(self, monkeypatch):
        # Ollama loads on demand — no Studio /api/inference/load round-trip.
        monkeypatch.setattr(tts, "TTS_BACKEND", "ollama")
        monkeypatch.setattr(tts, "TTS_URL", "http://embedder-qwen:11434/v1")
        assert await tts._ensure_model_loaded() is True

    async def test_no_swap_back_scheduled(self, monkeypatch):
        # Separate process — an Ollama TTS load never evicts Studio's chat model.
        monkeypatch.setattr(tts, "TTS_BACKEND", "ollama")
        monkeypatch.setattr(tts, "DEFAULT_MODEL", "unsloth/model")
        monkeypatch.setattr(tts, "TTS_URL", "http://embedder-qwen:11434/v1")

        def boom() -> None:
            raise AssertionError("swap-back must not run for the ollama backend")

        monkeypatch.setattr(tts, "_swap_chat_model_back", boom)
        tts._schedule_chat_swap_back()
        assert tts._chat_reload_task is None

    async def test_speech_uses_default_voice_and_no_studio_extras(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        calls: dict = {}

        def capture(**kwargs):
            calls.update(kwargs)
            resp = MagicMock()
            resp.read = lambda: b"fake-mp3"
            return resp

        client = MagicMock()
        client.audio.speech.create = AsyncMock(side_effect=capture)

        monkeypatch.setattr(tts, "TTS_BACKEND", "ollama")
        monkeypatch.setattr(tts, "TTS_MODEL", "phonon:de")
        monkeypatch.setattr(tts, "TTS_VOICE", "f1a84294-studio-clone-id")

        audio = await tts._speech_once(client, "Hallo Welt")
        assert audio == b"fake-mp3"
        assert calls["model"] == "phonon:de"
        # phonon rejects unknown voice names — always the fixed "default".
        assert calls["voice"] == "default"
        assert calls.get("extra_body") is None
        # Ollama serves wav/pcm only (mp3 → 400).
        assert calls["response_format"] == "wav"

    async def test_speech_studio_backend_keeps_clone_and_xvector(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        calls: dict = {}

        def capture(**kwargs):
            calls.update(kwargs)
            resp = MagicMock()
            resp.read = lambda: b"fake-mp3"
            return resp

        client = MagicMock()
        client.audio.speech.create = AsyncMock(side_effect=capture)

        monkeypatch.setattr(tts, "TTS_BACKEND", "studio")
        monkeypatch.setattr(tts, "TTS_MODEL", "audio-cpp/Qwen3-TTS")
        monkeypatch.setattr(tts, "TTS_VOICE", "f1a84294-studio-clone-id")
        monkeypatch.setattr(tts, "TTS_X_VECTOR_ONLY_MODE", True)

        await tts._speech_once(client, "Hallo Welt")
        assert calls["voice"] == "f1a84294-studio-clone-id"
        assert calls["extra_body"] == {"audio_options": {"x_vector_only_mode": True}}
        assert calls["response_format"] == "mp3"
