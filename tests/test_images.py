"""Image generation (#42) — bot_core.images + commands/generate_image_command.

GPU-split architecture (2026-11): chat runs on Strata (INFER_URL, 5070) and
image generation on Studio (IMAGE_INFER_URL, 4070). The 4070 alternates "rag"
mode (reranker+embedder) ⇄ "image" mode via the gpu-switch helper.

Hermetic: the OpenAI clients, the Studio management API and the gpu-switch
helper are mocked; no network, no live services. The critical invariant under
test: after EVERY generation (incl. failed ones) the bot asks the helper to
restore rag mode, so the reranker/embedder are never left down.
"""
from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

import bot_core.images as images
from commands.generate_image_command import handle_generate_image_command


_PNG = b"\x89PNG\r\n\x1a\nfake-png-bytes"


def _b64_png() -> str:
    return base64.b64encode(_PNG).decode()


class _FakeClient:
    """Stand-in for httpx.AsyncClient that records (METHOD path) calls."""

    def __init__(self, log: list, responses=None, **_kwargs):
        self.log = log
        self.responses = responses or []
        self._n = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def request(self, method, url, **kwargs):
        self.log.append(f"{method} {url}")
        if self.responses:
            resp = self.responses[min(self._n, len(self.responses) - 1)]
            self._n += 1
            return resp
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": True, "mode": "rag", "detail": "fake"}
        resp.text = ""
        return resp


def _resp_json(payload: dict, status: int = 200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.text = "" if status < 400 else "boom"
    return r


@pytest.fixture(autouse=True)
def _reset_gen_state():
    images._gen_lock = type(images._gen_lock)()
    images._mode_switched_at = 0.0
    images._hold_started_at = 0.0
    images._background_tasks.clear()
    yield


class TestImageGenConfigured:
    def test_true_when_model_set(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/Qwen-Image-2.1-Turbo-GGUF")
        assert images.image_gen_configured() is True

    def test_false_when_empty(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_MODEL", "")
        assert images.image_gen_configured() is False


class TestResolveSize:
    def test_none_gives_default(self):
        assert images.resolve_size(None) is None
        assert images.resolve_size("  ") is None

    def test_parses_and_clamps_to_max(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_MAX_SIZE", 2048)
        assert images.resolve_size("1024") == 1024
        assert images.resolve_size("99999") == 2048

    def test_floors_at_256(self):
        assert images.resolve_size("1") == 256

    def test_invalid_raises(self):
        with pytest.raises(images.ImageGenError, match="invalid size"):
            images.resolve_size("big")


class TestClientSplit:
    """Two backends, two clients — rewriting talks to CHAT, painting to IMAGE."""

    def test_image_client_uses_image_backend(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_INFER_URL", "http://image-host:8888/v1")
        monkeypatch.setattr(images, "IMAGE_INFER_API_KEY", "img-key")
        c = images._image_client()
        assert str(c.base_url).rstrip("/") == "http://image-host:8888/v1"
        assert c.api_key == "img-key"

    def test_chat_client_uses_chat_backend(self, monkeypatch):
        monkeypatch.setattr(images, "INFER_URL", "http://chat-host:8080/v1")
        monkeypatch.setattr(images, "INFER_API_KEY", "chat-key")
        c = images._chat_client()
        assert str(c.base_url).rstrip("/") == "http://chat-host:8080/v1"
        assert c.api_key == "chat-key"

    def test_image_client_uses_image_backend_settings(self, monkeypatch):
        # The fallback to INFER_* happens in config/settings.py (import time);
        # the module must read the IMAGE_* settings, never the chat ones.
        monkeypatch.setattr(images, "IMAGE_INFER_URL", "http://chat-host:8888/v1")
        monkeypatch.setattr(images, "IMAGE_INFER_API_KEY", "chat-key")
        c = images._image_client()
        assert str(c.base_url).rstrip("/") == "http://chat-host:8888/v1"


class TestRewritePrompt:
    """The rewrite runs the SAME bounded tool-calling loop as normal chat
    (bot_core.agent_loop) on the CHAT backend — so campaign specifics can be
    looked up in the KB (which needs rag mode live — hence rewrite BEFORE
    the mode switch)."""

    @staticmethod
    def _resp(content: str):
        return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])

    async def test_plain_answer_returned_stripped(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=self._resp('  "A rich painterly bird-folk prompt in Humblewood" '))
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        out = await images.rewrite_prompt("a bird")
        assert out == "A rich painterly bird-folk prompt in Humblewood"

    async def test_all_empty_raises(self, monkeypatch):
        # Every turn empty (reasoning truncation) → the loop's forced final is
        # empty too → ImageGenError lets the caller fall back to the raw prompt.
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._resp("   "))
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        with pytest.raises(images.ImageGenError, match="empty rewrite"):
            await images.rewrite_prompt("a bird")
        assert client.chat.completions.create.call_count >= 2

    async def test_empty_then_recovery(self, monkeypatch):
        # First turn empty (nudged), second a real answer → recovered.
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            self._resp(""),
            self._resp("A rich painterly bird-folk prompt in Humblewood"),
        ])
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        assert await images.rewrite_prompt("a bird") == \
            "A rich painterly bird-folk prompt in Humblewood"

    async def test_total_backend_failure_raises(self, monkeypatch):
        # The agent loop never raises — it returns an apology sentence on total
        # failure; the sentinel guard must turn that into ImageGenError so the
        # caller falls back to the raw prompt (not paint the apology).
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        with pytest.raises(images.ImageGenError, match="rewrite"):
            await images.rewrite_prompt("a bird")

    async def test_broken_tool_call_fragment_raises_not_painted(self, monkeypatch):
        """Regression (2026-10-28 live incident): the model answered a rewrite
        with a mangled mixed JSON/XML tool call; it must become ImageGenError
        (→ raw-prompt fallback), never a painted prompt."""
        fragment = (
            "< tool_call>\n{\"search_kb\"}\n< /parameter>\n"
            "< parameter=query>\nRomi character species appearance class gear\n"
            "< /parameter>\n< /function>\n< /tool_call>"
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=self._resp(fragment))  # every turn, incl. the fallback
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        with pytest.raises(images.ImageGenError, match="tool-call fragment"):
            await images.rewrite_prompt("Romi eating a nice steaming mushroom soop")
        # prepare_prompt degrades to the RAW user prompt instead.
        assert await images.prepare_prompt(
            "Romi eating a nice steaming mushroom soop") == \
            "Romi eating a nice steaming mushroom soop"

    async def test_campaign_lookup_goes_through_kb_tools(self, monkeypatch):
        """The user-facing capability: the rewrite may query the KB (same tools
        as normal chat) before answering — e.g. grounding 'Marvin #13'."""
        import bot_core.agent_loop as agent_loop

        executed: list[str] = []

        async def fake_execute(name, args):
            executed.append(name)
            return f"{name} result for {args}"

        monkeypatch.setattr(agent_loop, "execute_tool", fake_execute)

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            self._resp('{"tool": "search_kb", "args": {"query": "Marvin #13"}}'),
            self._resp("A rich painterly prompt with Marvin the Steel Defender golem"),
        ])
        monkeypatch.setattr(images, "_chat_client", lambda: client)

        out = await images.rewrite_prompt("Marvin #13 guarding a camp")
        assert "Steel Defender" in out
        assert executed == ["search_kb"]


class TestModeSwitching:
    """The gpu-switch helper contract (GPU_SWITCH_URL → POST/GET /mode)."""

    async def test_switch_to_image_mode_posts_helper(self, monkeypatch):
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        calls: list[str] = []
        monkeypatch.setattr(images.httpx, "AsyncClient",
                            lambda **kw: _FakeClient(calls, [_resp_json({"ok": True, "mode": "image", "detail": "d"})]))
        await images._switch_gpu_mode("image")
        assert calls == ["POST http://gpu-switch:8080/mode"]

    async def test_switch_to_rag_hits_the_helper(self, monkeypatch):
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        calls: list[str] = []
        monkeypatch.setattr(images.httpx, "AsyncClient",
                            lambda **kw: _FakeClient(calls))
        await images._switch_gpu_mode("rag")
        assert calls == ["POST http://gpu-switch:8080/mode"]

    async def test_http_error_becomes_imagegenerror(self, monkeypatch):
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images.httpx, "AsyncClient",
                            lambda **kw: _FakeClient([], [_resp_json({}, status=500)]))
        with pytest.raises(images.ImageGenError, match="gpu-switch"):
            await images._switch_gpu_mode("image")

    async def test_no_helper_url_image_mode_raises_manual_hint(self, monkeypatch):
        """Manual fallback contract: without GPU_SWITCH_URL the bot refuses to
        paint rather than generating in rag mode; the hint names the script."""
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "")
        with pytest.raises(images.ImageGenError, match="gpu_switch.sh image"):
            await images._switch_gpu_mode("image")

    async def test_rag_restore_is_best_effort_and_never_raises(self, monkeypatch):
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "_switch_gpu_mode",
                            AsyncMock(side_effect=images.ImageGenError("helper down")))
        await images._restore_rag_after_painting()  # must NOT raise


class TestEnsureImageMode:
    async def test_rag_mode_switches_then_loads(self, monkeypatch):
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        calls: list[str] = []
        monkeypatch.setattr(images.httpx, "AsyncClient",
                            lambda **kw: _FakeClient(calls, [
                                _resp_json({"mode": "rag"}),          # GET /mode
                                _resp_json({"ok": True, "mode": "image"}),  # POST /mode
                            ]))
        monkeypatch.setattr(images, "_image_status",
                            AsyncMock(return_value={"loaded": False}))
        loaded = AsyncMock()
        monkeypatch.setattr(images, "_ensure_image_loaded", loaded)
        await images.ensure_image_mode()
        assert calls == ["GET http://gpu-switch:8080/mode",
                         "POST http://gpu-switch:8080/mode"]
        loaded.assert_awaited_once()

    async def test_image_mode_with_model_resident_skips_switch_and_load(self, monkeypatch):
        """Back-to-back generations inside the hold window: no switch, no load."""
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        calls: list[str] = []
        monkeypatch.setattr(images.httpx, "AsyncClient",
                            lambda **kw: _FakeClient(calls, [_resp_json({"mode": "image"})]))
        monkeypatch.setattr(images, "_image_status",
                            AsyncMock(return_value={"loaded": True, "resident_mib": 12456}))
        loaded = AsyncMock()
        monkeypatch.setattr(images, "_ensure_image_loaded", loaded)
        await images.ensure_image_mode()
        assert calls == ["GET http://gpu-switch:8080/mode"]  # GET only — no POST
        loaded.assert_not_awaited()

    async def test_no_helper_url_delegates_to_manual_load(self, monkeypatch):
        """GPU_SWITCH_URL empty: ensure_image_mode skips the switch step and
        relies on the load path (the manual gpu_switch.sh image window)."""
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "")
        monkeypatch.setattr(images, "_image_status",
                            AsyncMock(return_value={"loaded": False}))
        loaded = AsyncMock()
        monkeypatch.setattr(images, "_ensure_image_loaded", loaded)
        await images.ensure_image_mode()  # no switch attempted, no raise
        loaded.assert_awaited_once()


class TestGenerateImage:
    async def test_success_returns_png_and_restores_rag(self, monkeypatch):
        """Happy path: image mode ensured, image generated, rag mode restored."""
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/Qwen-Image-2.1-Turbo-GGUF")
        monkeypatch.setattr(images, "IMAGE_DEFAULT_SIZE", 512)
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "ensure_image_mode", AsyncMock())
        monkeypatch.setattr(images, "_restore_rag_after_painting", AsyncMock())

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_image_client", lambda: client)

        out = await images.generate_image("a bird")
        assert out == _PNG
        images.ensure_image_mode.assert_awaited_once()
        images._restore_rag_after_painting.assert_awaited_once()
        # generation used the default size + configured model, on the image client
        _, kwargs = client.images.generate.call_args
        assert kwargs["model"] == "unsloth/Qwen-Image-2.1-Turbo-GGUF"
        assert kwargs["size"] == "512x512"

    async def test_on_phase_reports_loading_then_painting(self, monkeypatch):
        """Stall visibility: callers get phase updates around the switch+load."""
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/Qwen-Image-2.1-Turbo-GGUF")
        monkeypatch.setattr(images, "IMAGE_DEFAULT_SIZE", 512)
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "_image_status",
                            AsyncMock(return_value={"loaded": False}))

        order: list[str] = []

        async def fake_switch(mode: str) -> None:
            order.append(f"switch:{mode}")

        async def fake_load() -> None:
            order.append("load")

        monkeypatch.setattr(images, "_switch_gpu_mode", fake_switch)
        monkeypatch.setattr(images, "_ensure_image_loaded", fake_load)

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_image_client", lambda: client)

        async def on_phase(name: str) -> None:
            order.append(name)

        out = await images.generate_image("a bird", on_phase=on_phase)
        assert out == _PNG
        assert order == ["switch:image", "loading", "load", "painting", "switch:rag"]

    async def test_hold_zero_switches_back_immediately(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "IMAGE_MODE_HOLD_S", 0)
        monkeypatch.setattr(images, "ensure_image_mode", AsyncMock())
        sw = AsyncMock()
        monkeypatch.setattr(images, "_switch_gpu_mode", sw)

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_image_client", lambda: client)

        await images.generate_image("a")
        sw.assert_awaited_once_with("rag")

    async def test_hold_defers_the_switch_back_and_fires(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "IMAGE_MODE_HOLD_S", 0.2)
        monkeypatch.setattr(images, "ensure_image_mode", AsyncMock())
        sw = AsyncMock()
        monkeypatch.setattr(images, "_switch_gpu_mode", sw)

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_image_client", lambda: client)

        await images.generate_image("a")
        sw.assert_not_awaited()  # held — the deferred task is pending
        assert images._hold_started_at > 0
        for _ in range(50):
            await asyncio.sleep(0.05)
            if sw.await_count:
                break
        sw.assert_awaited_once_with("rag")

    async def test_hold_reentered_defers_restore_again(self, monkeypatch):
        """If a new generation re-entered image mode during the hold window,
        the pending restore must NOT kill it — it reschedules itself."""
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "IMAGE_MODE_HOLD_S", 0.2)
        sw = AsyncMock()
        monkeypatch.setattr(images, "_switch_gpu_mode", sw)
        now = __import__("time").monotonic()
        images._hold_started_at = now - 0.1
        images._mode_switched_at = now  # painting happened AFTER the hold was armed
        await images.ensure_image_still_needed_or_restore()
        sw.assert_not_awaited()
        for t in list(images._background_tasks):
            t.cancel()

    async def test_hold_window_skips_second_switch(self, monkeypatch):
        """IMAGE_MODE_HOLD_S: after a generation that switched INTO image mode
        the switch-back is skipped inside the hold window, and the next
        back-to-back generation finds image mode live — no second switch."""
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "IMAGE_DEFAULT_SIZE", 512)
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "IMAGE_MODE_HOLD_S", 300)
        monkeypatch.setattr(images, "_image_status",
                            AsyncMock(return_value={"loaded": True}))

        order: list[str] = []

        async def fake_switch(mode: str) -> None:
            order.append(f"switch:{mode}")

        async def fake_load() -> None:
            order.append("load")

        monkeypatch.setattr(images, "_switch_gpu_mode", fake_switch)
        monkeypatch.setattr(images, "_ensure_image_loaded", fake_load)

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_image_client", lambda: client)

        # Gen 1: helper reports rag mode → switch in + load; hold skips back.
        monkeypatch.setattr(images.httpx, "AsyncClient",
                            lambda **kw: _FakeClient([], [_resp_json({"mode": "rag"})]))
        await images.generate_image("a")
        assert order == ["switch:image", "load"]
        assert images._mode_switched_at > 0

        # Gen 2: helper now reports image mode + resident model → no switch,
        # no load, no switch-back (still inside the hold window).
        monkeypatch.setattr(images.httpx, "AsyncClient",
                            lambda **kw: _FakeClient([], [_resp_json({"mode": "image"})]))
        await images.generate_image("b")
        assert order == ["switch:image", "load"]  # unchanged
        assert client.images.generate.call_count == 2

    async def test_on_phase_callback_error_never_fails_generation(self, monkeypatch):
        """A broken progress callback (e.g. Discord edit rejected) must not kill the gen."""
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "IMAGE_DEFAULT_SIZE", 512)
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "")
        monkeypatch.setattr(images, "ensure_image_mode", AsyncMock())
        monkeypatch.setattr(images, "_restore_rag_after_painting", AsyncMock())

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_image_client", lambda: client)

        async def broken(name: str) -> None:
            raise RuntimeError("discord edit failed")

        out = await images.generate_image("a bird", on_phase=broken)
        assert out == _PNG
