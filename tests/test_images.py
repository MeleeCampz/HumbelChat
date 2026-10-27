"""Image generation (#42) — bot_core.images + commands/generate_image_command.

Hermetic: the OpenAI client and the Studio management API are mocked; no
network, no live services. The critical invariant under test: the image model
is ALWAYS unloaded after a generation (even failed ones) so the chat model can
come back.
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

import bot_core.images as images
from commands.generate_image_command import handle_generate_image_command


_PNG = b"\x89PNG\r\n\x1a\nfake-png-bytes"


def _b64_png() -> str:
    return base64.b64encode(_PNG).decode()


@pytest.fixture(autouse=True)
def _reset_gen_lock():
    images._gen_lock = type(images._gen_lock)()
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


class TestRewritePrompt:
    async def test_returns_stripped_content(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content='  "a bird" '))]))
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)
        out = await images.rewrite_prompt("a bird")
        assert out == "a bird"

    async def test_empty_content_raises_after_retry(self, monkeypatch):
        # Reasoning model can truncate before any content (observed live) —
        # one retry, then the error lets the caller fall back to the raw prompt.
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="   "))]))
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)
        with pytest.raises(images.ImageGenError, match="empty rewrite"):
            await images.rewrite_prompt("a bird")
        assert client.chat.completions.create.call_count == 2

    async def test_empty_then_retry_recovers(self, monkeypatch):
        empty = MagicMock(choices=[MagicMock(message=MagicMock(content=""))])
        good = MagicMock(choices=[MagicMock(message=MagicMock(content="a rich bird"))])
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[empty, good])
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)
        assert await images.rewrite_prompt("a bird") == "a rich bird"
        assert client.chat.completions.create.call_count == 2

    async def test_api_failure_wrapped(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)
        with pytest.raises(images.ImageGenError, match="prompt rewrite failed"):
            await images.rewrite_prompt("a bird")


class TestGenerateImage:
    async def test_success_returns_png_and_unloads(self, monkeypatch):
        """Happy path: load ensured, image generated, ALWAYS unloaded after."""
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/Qwen-Image-2.1-Turbo-GGUF")
        monkeypatch.setattr(images, "IMAGE_DEFAULT_SIZE", 512)
        monkeypatch.setattr(images, "_warm_chat_reload", AsyncMock())

        loads: list[bool] = []

        async def fake_ensure() -> None:
            loads.append(True)

        monkeypatch.setattr(images, "ensure_image_loaded", fake_ensure)

        unloaded: list[bool] = []

        async def fake_unload() -> None:
            unloaded.append(True)

        monkeypatch.setattr(images, "_unload_image", fake_unload)

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)

        out = await images.generate_image("a bird")
        assert out == _PNG
        assert loads == [True]
        assert unloaded == [True]  # swap-back happened
        # generation used the default size + configured model
        _, kwargs = client.images.generate.call_args
        assert kwargs["model"] == "unsloth/Qwen-Image-2.1-Turbo-GGUF"
        assert kwargs["size"] == "512x512"

    async def test_negative_prompt_passed_via_extra_body(self, monkeypatch):
        # Studio accepts negative_prompt; this SDK version has no native kwarg.
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "IMAGE_NEGATIVE_PROMPT", "humans, text, blurry")
        monkeypatch.setattr(images, "_warm_chat_reload", AsyncMock())
        monkeypatch.setattr(images, "ensure_image_loaded", AsyncMock())
        monkeypatch.setattr(images, "_unload_image", AsyncMock())

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)

        await images.generate_image("a bird")
        _, kwargs = client.images.generate.call_args
        assert kwargs["extra_body"] == {"negative_prompt": "humans, text, blurry"}

    async def test_empty_negative_prompt_omitted(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "IMAGE_NEGATIVE_PROMPT", "")
        monkeypatch.setattr(images, "_warm_chat_reload", AsyncMock())
        monkeypatch.setattr(images, "ensure_image_loaded", AsyncMock())
        monkeypatch.setattr(images, "_unload_image", AsyncMock())

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)

        await images.generate_image("a bird")
        _, kwargs = client.images.generate.call_args
        assert kwargs["extra_body"] is None

    async def test_failed_generation_still_unloads(self, monkeypatch):
        """The critical invariant: even a failed generation swaps the chat back."""
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "_warm_chat_reload", AsyncMock())
        monkeypatch.setattr(images, "ensure_image_loaded", AsyncMock())

        unloaded: list[bool] = []

        async def fake_unload() -> None:
            unloaded.append(True)

        monkeypatch.setattr(images, "_unload_image", fake_unload)

        client = MagicMock()
        client.images.generate = AsyncMock(side_effect=RuntimeError("gpu oom"))
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)

        with pytest.raises(RuntimeError, match="gpu oom"):
            await images.generate_image("a bird")
        assert unloaded == [True]  # swap-back despite the failure

    async def test_no_image_data_raises_but_unloads(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "_warm_chat_reload", AsyncMock())
        monkeypatch.setattr(images, "ensure_image_loaded", AsyncMock())

        unloaded: list[bool] = []
        monkeypatch.setattr(images, "_unload_image", AsyncMock(side_effect=lambda: unloaded.append(True)))

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=None)]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)

        with pytest.raises(images.ImageGenError, match="no image data"):
            await images.generate_image("a bird")
        assert unloaded == [True]


class TestCommandHandler:
    """commands/generate_image_command — mocked core, fake interaction."""

    def _interaction(self):
        inter = MagicMock()
        inter.response.defer = AsyncMock()
        followups: list[dict] = []

        async def capture(content=None, **kwargs):
            followups.append({"content": content, **kwargs})

        inter.followup.send = capture
        return inter, followups

    async def test_not_configured(self, monkeypatch):
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: False)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="a bird")
        assert len(followups) == 1
        assert "not configured" in followups[0]["content"]

    async def test_empty_prompt(self, monkeypatch):
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="   ")
        assert "Nothing to paint" in followups[0]["content"]

    async def test_raw_skips_rewrite(self, monkeypatch):
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        monkeypatch.setattr(cmd, "rewrite_prompt", AsyncMock())
        generated: list[str] = []

        async def fake_gen(prompt, size=None):
            generated.append(prompt)
            return _PNG

        monkeypatch.setattr(cmd, "generate_image", fake_gen)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="verbatim bird", raw=True)
        cmd.rewrite_prompt.assert_not_called()
        assert generated == ["verbatim bird"]
        assert "file" in followups[0]

    async def test_rewrite_used_when_not_raw_and_falls_back_on_failure(self, monkeypatch):
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        generated: list[str] = []

        async def fake_gen(prompt, size=None):
            generated.append(prompt)
            return _PNG

        monkeypatch.setattr(cmd, "generate_image", fake_gen)
        inter, followups = self._interaction()

        # rewrite succeeds → rewritten prompt is what gets generated
        monkeypatch.setattr(cmd, "rewrite_prompt", AsyncMock(return_value="rich bird"))
        await handle_generate_image_command(inter, prompt="bird")
        assert generated == ["rich bird"]

        # rewrite fails → falls back to the raw prompt (no error reply)
        generated.clear()
        followups.clear()
        monkeypatch.setattr(cmd, "rewrite_prompt", AsyncMock(side_effect=images.ImageGenError("down")))
        await handle_generate_image_command(inter, prompt="bird")
        assert generated == ["bird"]
        assert "file" in followups[0]

    async def test_generation_failure_replies_error(self, monkeypatch):
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        monkeypatch.setattr(cmd, "rewrite_prompt", AsyncMock(return_value="rich"))

        async def boom(prompt, size=None):
            raise images.ImageGenError("studio 500")

        monkeypatch.setattr(cmd, "generate_image", boom)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="bird")
        assert "failed" in followups[0]["content"].lower()

    async def test_bad_size_replies_error(self, monkeypatch):
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="bird", size="gigantic")
        assert "invalid size" in followups[0]["content"]
