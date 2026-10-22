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
