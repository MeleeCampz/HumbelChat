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

    # ── #73 class fix (code review §1.1): slug split + POST status check ──

    @staticmethod
    def _fake_http(monkeypatch, post_status=200, models=None, *, get=None):
        """Stub tts.httpx.AsyncClient; returns the captured POST payload."""
        captured: dict = {}
        models = models or []

        class _Resp:
            def __init__(self, status: int = 200, text: str = "") -> None:
                self.status_code = status
                self.text = text

            def json(self):
                return {"data": models}

        class _Client:
            def __init__(self, *a, **kw) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["payload"] = json
                return _Resp(post_status, "AutoConfig error" if post_status >= 400 else "")

            async def get(self, url, headers=None):
                if get is not None:
                    return get()
                return _Resp()

        monkeypatch.setattr(tts.httpx, "AsyncClient", _Client)
        monkeypatch.setattr(tts, "TTS_GPU_IDS", [])
        monkeypatch.setattr(tts, "TTS_URL", "http://studio:8888/v1")
        monkeypatch.setattr(tts.asyncio, "sleep", lambda s: _done())
        return captured

    async def test_payload_splits_repo_and_variant(self, monkeypatch):
        # Studio's load endpoint needs repo + gguf_variant — NOT the colon
        # slug (sending it verbatim is the #73 failure).
        captured = self._fake_http(monkeypatch)
        result = await tts._ensure_slug_loaded(
            "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M", 0.1
        )
        assert result is False  # never resident in this stub
        assert captured["url"] == "http://studio:8888/api/inference/load"
        assert captured["payload"] == {
            "model_path": "unsloth/Qwen3.8-27B-GGUF",
            "gguf_variant": "Qwen3.8-27B-NVFP4_50-Q4K_M",
        }

    async def test_plain_slug_gets_no_variant_key(self, monkeypatch):
        captured = self._fake_http(monkeypatch)
        await tts._ensure_slug_loaded("unsloth/Qwen-Image-2.1-Turbo-GGUF", 0.1)
        assert captured["payload"] == {"model_path": "unsloth/Qwen-Image-2.1-Turbo-GGUF"}

    async def test_failed_post_returns_false_without_polling(self, monkeypatch):
        # A rejected load (e.g. 500) must surface immediately instead of
        # silently polling a model that can never become resident.
        def no_get():
            raise AssertionError("must not poll after a failed load POST")

        self._fake_http(monkeypatch, post_status=500, get=no_get)
        assert await tts._ensure_slug_loaded("repo:variant", 1) is False

    async def test_residency_matches_bare_id_plus_quant(self, monkeypatch):
        # Studio reports loaded models under the BARE repo id with the quant
        # in a separate field — matching on the full slug never succeeded.
        models = [{
            "id": "unsloth/Qwen3.8-27B-GGUF",
            "quant": "Qwen3.8-27B-NVFP4_50-Q4K_M",
            "loaded": True,
        }]
        self._fake_http(monkeypatch, models=models)
        assert await tts._ensure_slug_loaded(
            "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M", 1
        ) is True

    async def test_residency_wrong_quant_is_not_a_match(self, monkeypatch):
        models = [{
            "id": "unsloth/Qwen3.8-27B-GGUF",
            "quant": "Some-Other-Quant",
            "loaded": True,
        }]
        self._fake_http(monkeypatch, models=models)
        assert await tts._ensure_slug_loaded(
            "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M", 0.1
        ) is False


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
        monkeypatch.setattr(tts, "TTS_URL", "http://tts-ollama:11434/v1")
        assert await tts._ensure_model_loaded() is True

    async def test_no_swap_back_scheduled(self, monkeypatch):
        # Separate process — an Ollama TTS load never evicts Studio's chat model.
        monkeypatch.setattr(tts, "TTS_BACKEND", "ollama")
        monkeypatch.setattr(tts, "DEFAULT_MODEL", "unsloth/model")
        monkeypatch.setattr(tts, "TTS_URL", "http://tts-ollama:11434/v1")

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


async def _done() -> None:
    return None


class TestTransientRetry:
    """#57 follow-up: Ollama's MLX runner panics intermittently on the first
    request after a reload (500) — synthesize() must retry exactly once."""

    async def test_server_error_retries_once_then_succeeds(self, monkeypatch):
        from unittest.mock import AsyncMock

        monkeypatch.setattr(tts, "TTS_BACKEND", "ollama")
        monkeypatch.setattr(tts, "TTS_MODEL", "phonon:de")
        calls = []

        async def flaky(client, text):
            calls.append(1)
            if len(calls) == 1:
                raise tts.TTSError("InternalServerError: mlx runner failed", retryable=True)
            return b"wav-bytes"

        monkeypatch.setattr(tts, "_speech_once", flaky)
        sleeps = []
        monkeypatch.setattr(tts.asyncio, "sleep", lambda s: sleeps.append(s) or _done())

        audio = await tts.synthesize("Hallo")
        assert audio == b"wav-bytes"
        assert calls == [1, 1]  # exactly one retry

    async def test_permanent_error_raises_immediately(self, monkeypatch):
        monkeypatch.setattr(tts, "TTS_BACKEND", "ollama")
        monkeypatch.setattr(tts, "TTS_MODEL", "phonon:de")
        calls = []

        async def broken(client, text):
            calls.append(1)
            raise tts.TTSError("BadRequestError: voice not found")  # retryable=False

        monkeypatch.setattr(tts, "_speech_once", broken)

        with pytest.raises(tts.TTSError, match="voice not found"):
            await tts.synthesize("Hallo")
        assert calls == [1]  # no retry for client errors

    async def test_5xx_from_openai_client_is_retryable(self, monkeypatch):
        from openai import APIStatusError
        from unittest.mock import AsyncMock, MagicMock

        resp = MagicMock(status_code=500)
        boom = APIStatusError(
            "InternalServerError: mlx runner failed", response=resp, body={"error": "x"}
        )
        client = MagicMock()
        client.audio.speech.create = AsyncMock(side_effect=boom)

        monkeypatch.setattr(tts, "TTS_BACKEND", "ollama")
        monkeypatch.setattr(tts, "TTS_MODEL", "phonon:de")

        with pytest.raises(tts.TTSError) as excinfo:
            await tts._speech_once(client, "Hallo")
        assert excinfo.value.retryable is True

    async def test_4xx_from_openai_client_is_not_retryable(self, monkeypatch):
        from openai import APIStatusError
        from unittest.mock import AsyncMock, MagicMock

        resp = MagicMock(status_code=400)
        boom = APIStatusError(
            'BadRequestError: response_format "mp3" is not supported',
            response=resp, body={"error": "x"},
        )
        client = MagicMock()
        client.audio.speech.create = AsyncMock(side_effect=boom)

        monkeypatch.setattr(tts, "TTS_BACKEND", "ollama")
        monkeypatch.setattr(tts, "TTS_MODEL", "phonon:de")

        with pytest.raises(tts.TTSError) as excinfo:
            await tts._speech_once(client, "Hallo")
        assert excinfo.value.retryable is False
