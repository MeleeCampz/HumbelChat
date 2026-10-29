"""Image generation (#42) — bot_core.images + commands/generate_image_command.

Hermetic: the OpenAI client and the Studio management API are mocked; no
network, no live services. The critical invariant under test: the image model
is ALWAYS unloaded after a generation (even failed ones) so the chat model can
come back.
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
    """The rewrite runs the SAME bounded tool-calling loop as normal chat
    (bot_core.agent_loop) — so campaign specifics can be looked up in the KB."""

    @staticmethod
    def _resp(content: str):
        return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])

    async def test_plain_answer_returned_stripped(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=self._resp('  "A rich painterly bird-folk prompt in Humblewood" '))
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)
        out = await images.rewrite_prompt("a bird")
        assert out == "A rich painterly bird-folk prompt in Humblewood"

    async def test_all_empty_raises(self, monkeypatch):
        # Every turn empty (reasoning truncation) → the loop's forced final is
        # empty too → ImageGenError lets the caller fall back to the raw prompt.
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._resp("   "))
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)
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
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)
        assert await images.rewrite_prompt("a bird") == \
            "A rich painterly bird-folk prompt in Humblewood"

    async def test_total_backend_failure_raises(self, monkeypatch):
        # The agent loop never raises — it returns an apology sentence on total
        # failure; the sentinel guard must turn that into ImageGenError so the
        # caller falls back to the raw prompt (not paint the apology).
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)
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
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)
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
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)

        out = await images.rewrite_prompt("Marvin #13 guarding a camp")
        assert "Steel Defender" in out
        assert executed == ["search_kb"]


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

    async def test_on_phase_reports_loading_then_painting(self, monkeypatch):
        """Stall visibility: callers get phase updates around the GPU-idle swap."""
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/Qwen-Image-2.1-Turbo-GGUF")
        monkeypatch.setattr(images, "IMAGE_DEFAULT_SIZE", 512)
        monkeypatch.setattr(images, "_warm_chat_reload", AsyncMock())

        order: list[str] = []

        async def fake_ensure() -> None:
            order.append("ensure")

        monkeypatch.setattr(images, "ensure_image_loaded", fake_ensure)
        monkeypatch.setattr(images, "_unload_image", AsyncMock())

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)

        async def on_phase(name: str) -> None:
            order.append(name)

        out = await images.generate_image("a bird", on_phase=on_phase)
        assert out == _PNG
        assert order.index("loading") < order.index("ensure") < order.index("painting")

    async def test_on_phase_callback_error_never_fails_generation(self, monkeypatch):
        """A broken progress callback (e.g. Discord edit rejected) must not kill the gen."""
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/Qwen-Image-2.1-Turbo-GGUF")
        monkeypatch.setattr(images, "IMAGE_DEFAULT_SIZE", 512)
        monkeypatch.setattr(images, "_warm_chat_reload", AsyncMock())
        monkeypatch.setattr(images, "ensure_image_loaded", AsyncMock())
        monkeypatch.setattr(images, "_unload_image", AsyncMock())

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)

        async def broken(name: str) -> None:
            raise RuntimeError("discord edit failed")

        out = await images.generate_image("a bird", on_phase=broken)
        assert out == _PNG

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

    async def test_backend_5xx_retried_once(self, monkeypatch):
        # Observed live: Studio 'Image generation failed' (HTTP 500) mid-run,
        # recovered on the next attempt. 4xx must NOT retry.
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "_warm_chat_reload", AsyncMock())
        monkeypatch.setattr(images, "ensure_image_loaded", AsyncMock())
        monkeypatch.setattr(images, "_unload_image", AsyncMock())

        class _Fake5xx(Exception):
            status_code = 500

        monkeypatch.setattr(images, "APIStatusError", _Fake5xx)

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(side_effect=[_Fake5xx("boom"), resp])
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)

        out = await images.generate_image("a bird")
        assert out == _PNG
        assert client.images.generate.call_count == 2

    async def test_backend_4xx_not_retried(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "_warm_chat_reload", AsyncMock())
        monkeypatch.setattr(images, "ensure_image_loaded", AsyncMock())
        monkeypatch.setattr(images, "_unload_image", AsyncMock())

        class _Fake4xx(Exception):
            status_code = 400

        monkeypatch.setattr(images, "APIStatusError", _Fake4xx)

        client = MagicMock()
        client.images.generate = AsyncMock(side_effect=_Fake4xx("bad model"))
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)

        with pytest.raises(_Fake4xx):
            await images.generate_image("a bird")
        assert client.images.generate.call_count == 1

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


class TestGenerateImagesBatch:
    """generate_images (#69): every prompt painted in ONE swap window —
    load once, paint all (one failure never aborts the rest), unload + warm-
    reload the chat model exactly once."""

    def _patch(self, monkeypatch) -> tuple[MagicMock, list[str]]:
        """Fake the whole swap pipeline; returns (client, order log)."""
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "_warm_chat_reload", AsyncMock())
        order: list[str] = []

        async def fake_ensure() -> None:
            order.append("load")

        monkeypatch.setattr(images, "ensure_image_loaded", fake_ensure)

        async def fake_unload() -> None:
            order.append("unload")

        monkeypatch.setattr(images, "_unload_image", fake_unload)

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_rewrite_client", lambda: client)
        return client, order

    async def test_one_load_n_paints_one_unload(self, monkeypatch):
        client, order = self._patch(monkeypatch)
        out = await images.generate_images(["a", "b", "c"])
        assert out == [_PNG, _PNG, _PNG]
        # ONE swap window: load once before all paints, unload once after.
        assert order.count("load") == 1
        assert order.count("unload") == 1
        assert order.index("load") < order.index("unload")
        assert client.images.generate.call_count == 3

    async def test_phases_loading_once_then_per_image(self, monkeypatch):
        self._patch(monkeypatch)
        phases: list[str] = []

        async def on_phase(name: str) -> None:
            phases.append(name)

        await images.generate_images(["a", "b", "c"], on_phase=on_phase)
        assert phases == ["loading", "painting 1/3", "painting 2/3", "painting 3/3"]

    async def test_one_failure_does_not_abort_the_rest(self, monkeypatch):
        client, _order = self._patch(monkeypatch)

        class _Fake4xx(Exception):
            status_code = 400

        monkeypatch.setattr(images, "APIStatusError", _Fake4xx)
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(side_effect=[_Fake4xx("bad prompt"), resp, resp])

        out = await images.generate_images(["a", "b", "c"])
        # Moment 1 failed (4xx → no retry), moments 2+3 still painted.
        assert isinstance(out[0], _Fake4xx)
        assert out[1] == _PNG
        assert out[2] == _PNG

    async def test_failed_batch_still_unloads_and_warm_reloads(self, monkeypatch):
        client, order = self._patch(monkeypatch)
        warm = AsyncMock()
        monkeypatch.setattr(images, "_warm_chat_reload", warm)

        class _Fake4xx(Exception):
            status_code = 400

        monkeypatch.setattr(images, "APIStatusError", _Fake4xx)
        client.images.generate = AsyncMock(side_effect=_Fake4xx("down"))

        out = await images.generate_images(["a", "b"])
        assert all(isinstance(r, _Fake4xx) for r in out)
        assert order == ["load", "unload"]  # swap-back still happened
        # The warm reload runs as a BACKGROUND task — let it start.
        for _ in range(5):
            await asyncio.sleep(0)
            if warm.await_count:
                break
        warm.assert_awaited_once()

    async def test_broken_phase_callback_never_fails_batch(self, monkeypatch):
        self._patch(monkeypatch)

        def broken(name: str) -> None:
            raise RuntimeError("discord edit failed")

        out = await images.generate_images(["a", "b"], on_phase=broken)
        assert out == [_PNG, _PNG]

    async def test_empty_batch_is_a_noop_swap(self, monkeypatch):
        client, order = self._patch(monkeypatch)
        out = await images.generate_images([])
        assert out == []
        # Still one full swap window (consistent contract; caller gates anyway).
        assert order == ["load", "unload"]
        client.images.generate.assert_not_awaited()


class TestPreparePrompt:
    """prepare_prompt: shared final-prompt step (live path + dry run, #65b)."""

    async def test_raw_passthrough(self):
        assert await images.prepare_prompt("  verbatim bird ", raw=True) == "verbatim bird"

    async def test_empty_returns_empty_without_model_call(self, monkeypatch):
        called = False

        async def fake_rewrite(text):
            nonlocal called
            called = True
            return "x"

        monkeypatch.setattr(images, "rewrite_prompt", fake_rewrite)
        assert await images.prepare_prompt("   ") == ""
        assert not called

    async def test_rewrite_used_when_not_raw(self, monkeypatch):
        monkeypatch.setattr(
            images, "rewrite_prompt", AsyncMock(return_value="rich bird"))
        assert await images.prepare_prompt("bird") == "rich bird"

    async def test_rewrite_failure_falls_back_to_raw(self, monkeypatch):
        monkeypatch.setattr(
            images, "rewrite_prompt",
            AsyncMock(side_effect=images.ImageGenError("backend down")))
        assert await images.prepare_prompt("bird") == "bird"


class TestRewriteSystemPolicy:
    """Base-prompt policy strings (regression guards for user decisions)."""

    def test_gogglies_only_column_bidae(self):
        # 2026-10-27 correction: NOBODY has googly eyes except Column Bidae (he).
        import bot_core.images as images_mod
        s = images_mod._load_rewrite_system()
        assert "Googly eyes are NOT a species trait" in s
        assert "belong ONLY to Column Bidae" in s
        assert "give EVERY one of them those" not in s

    def test_fixed_template_with_length_budgets(self):
        # 2026-10-28 optimization: fixed [STYLE][SCENE][CHARACTERS][LIGHTING]
        # structure + verbatim visual anchors + hard character budget, so
        # prompts stay detailed but never bloat.
        import bot_core.images as images_mod
        s = images_mod._load_rewrite_system()
        for part in ("[STYLE]", "[SCENE]", "[CHARACTERS]", "[LIGHTING]"):
            assert part in s
        assert "copy the ENTIRE anchor block VERBATIM" in s
        assert "under 1300 characters" in s
        assert "up to 2000 characters" in s
        # species words must come from the sheet, never be invented (live: Seta
        # kept coming out a mangled 'cochran')
        assert "Never invent or mangle a species word" in s
        # group scenes: one sheet read, and no silent character-dropping
        # (live: 'the whole party' came back with 3 of 5 members)
        assert "read the 'Player Characters' sheet ONCE" in s
        assert "never silently drop any" in s
        # negations stay OUT of the positive prompt (negative prompt handles them)
        assert "Never write negations" in s

    def test_character_style_lines_never_copied_into_prompt(self):
        import bot_core.images as images_mod
        s = images_mod._load_rewrite_system()
        assert "do NOT copy them into the final prompt" in s

    def test_no_painterly_style_prescribed(self):
        # 'painterly' may only appear as a prohibition, never as the style itself.
        import bot_core.images as images_mod
        s = images_mod._load_rewrite_system().lower()
        assert "semi-realistic painterly" not in s


class TestRewritePolicyLoading:
    """The policy is layered: repo file + vault character-rules note, both
    re-read per generation; missing files degrade to built-in defaults."""

    def test_vault_character_rules_are_injected(self, monkeypatch, tmp_path):
        import bot_core.images as images_mod
        note = tmp_path / "note.md"
        note.write_text("ZZZ_CUSTOM_RULE_MARKER applies.", encoding="utf-8")
        monkeypatch.setattr(images_mod, "_CHAR_RULES_FILE", note)
        assert "ZZZ_CUSTOM_RULE_MARKER" in images_mod._load_rewrite_system()

    def test_missing_vault_note_falls_back_to_default(self, monkeypatch, tmp_path):
        import bot_core.images as images_mod
        monkeypatch.setattr(
            images_mod, "_CHAR_RULES_FILE", tmp_path / "absent.md")
        s = images_mod._load_rewrite_system()
        assert "Googly eyes are NOT a species trait" in s  # built-in default

    def test_missing_repo_file_falls_back_to_builtin_policy(
            self, monkeypatch, tmp_path):
        import bot_core.images as images_mod
        monkeypatch.setattr(images_mod, "_POLICY_FILE", tmp_path / "absent.md")
        s = images_mod._load_rewrite_system()
        assert "[STYLE]" in s  # built-in fallback, not a crash

    def test_human_header_and_placeholder_not_injected(self):
        import bot_core.images as images_mod
        s = images_mod._load_rewrite_system()
        assert "Source of truth for" not in s  # human-facing header stripped
        assert "{{CHARACTER_RULES}}" not in s  # placeholder fully replaced


class TestWarmChatReloadSerialization:
    """The warm chat-load must never run CONCURRENTLY with an image load —
    two concurrent Studio model loads stall each other for minutes (observed
    live: every back-to-back generation idled ~300 s on a stuck load)."""

    async def test_waits_for_generation_lock(self, monkeypatch):
        import bot_core.images as images_mod

        monkeypatch.setattr(images_mod, "DEFAULT_MODEL", "unsloth/model")
        ran: list[bool] = []

        async def fake_studio(client, method, path, **kwargs):
            if path == "/api/inference/loaded-models":
                return {"data": []}
            ran.append(True)
            return {}

        monkeypatch.setattr(images_mod, "_studio", fake_studio)

        # Hold the generation lock (a next generation is in its swap window).
        images_mod._gen_lock = type(images_mod._gen_lock)()
        held = images_mod._gen_lock
        async with held:
            task = asyncio.create_task(images_mod._warm_chat_reload())
            await asyncio.sleep(0.2)
            assert ran == []  # still waiting behind the generation — no race
        await asyncio.wait_for(task, timeout=5)
        assert ran == [True]  # ran only after the lock was released


class TestWarmChatReloadPayload:
    """Studio's /api/inference/load rejects the bot's "repo:quant" model slug
    (HF repo-id validation chokes on the colon — observed live: deferred 500
    after every image generation, chat left evicted = backlog #73). The warm
    reload must split it into model_path + gguf_variant."""

    def _patch(self, monkeypatch, default_model, loaded):
        import bot_core.images as images_mod

        monkeypatch.setattr(images_mod, "DEFAULT_MODEL", default_model)
        posts: list[dict] = []

        async def fake_studio(client, method, path, **kwargs):
            if path == "/api/inference/loaded-models":
                return {"data": loaded}
            assert method == "POST"
            assert path == "/api/inference/load"
            posts.append(kwargs.get("json", {}))
            return {}

        monkeypatch.setattr(images_mod, "_studio", fake_studio)
        return posts

    async def test_splits_repo_and_variant(self, monkeypatch):
        posts = self._patch(
            monkeypatch,
            "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M",
            [],
        )
        await images._warm_chat_reload()
        assert posts == [{"model_path": "unsloth/Qwen3.8-27B-GGUF",
                          "gguf_variant": "Qwen3.8-27B-NVFP4_50-Q4K_M"}]

    async def test_plain_repo_has_no_variant_key(self, monkeypatch):
        posts = self._patch(monkeypatch, "unsloth/Qwen3.8-27B-GGUF", [])
        await images._warm_chat_reload()
        assert posts == [{"model_path": "unsloth/Qwen3.8-27B-GGUF"}]

    async def test_skips_when_repo_and_quant_already_resident(self, monkeypatch):
        posts = self._patch(
            monkeypatch,
            "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M",
            [{"id": "unsloth/Qwen3.8-27B-GGUF",
              "quant": "Qwen3.8-27B-NVFP4_50-Q4K_M", "loaded": True}],
        )
        await images._warm_chat_reload()
        assert posts == []  # no redundant full reload of a resident model

    async def test_reloads_when_quant_differs(self, monkeypatch):
        posts = self._patch(
            monkeypatch,
            "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M",
            [{"id": "unsloth/Qwen3.8-27B-GGUF", "quant": "UD-Q4_K_XL", "loaded": True}],
        )
        await images._warm_chat_reload()
        assert len(posts) == 1  # wrong quant resident → load the right one

    async def test_no_default_model_is_a_noop(self, monkeypatch):
        import bot_core.images as images_mod

        monkeypatch.setattr(images_mod, "DEFAULT_MODEL", None)
        called = []

        async def fake_studio(client, method, path, **kwargs):
            called.append(path)
            return {}

        monkeypatch.setattr(images_mod, "_studio", fake_studio)
        await images._warm_chat_reload()
        assert called == []


class TestCommandHandler:
    """commands/generate_image_command — mocked core, fake interaction."""

    def _interaction(self):
        inter = MagicMock()
        inter.response.defer = AsyncMock()  # discord.py 2.7: no content kwarg
        followups: list[dict] = []

        async def capture(content=None, **kwargs):
            msg = {"content": content, **kwargs}
            followups.append(msg)

            edits: list[str] = []

            async def edit(**ekwargs):
                if "content" in ekwargs:
                    edits.append(str(ekwargs["content"]))
                msg.update(ekwargs)  # latest state wins (single message)

            async def delete():
                msg["deleted"] = True

            msg["edit"] = edit
            msg["edits"] = edits
            msg["deleted"] = False
            return MagicMock(edit=edit, delete=delete)

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
        prepared = AsyncMock(side_effect=lambda text, raw=False: text)
        monkeypatch.setattr(cmd, "prepare_prompt", prepared)
        generated: list[str] = []

        async def fake_gen(prompt, size=None, on_phase=None):
            generated.append(prompt)
            if on_phase is not None:
                await on_phase("painting")
            return _PNG

        monkeypatch.setattr(cmd, "generate_image", fake_gen)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="verbatim bird", raw=True)
        prepared.assert_awaited_once_with("verbatim bird", raw=True)
        assert generated == ["verbatim bird"]
        # ONE message: the final answer (text + image) landed in place on the
        # progress message — no separate reply.
        assert len(followups) == 1
        assert followups[0]["content"].startswith("🎨")
        assert "attachments" in followups[0]

    async def test_prepare_prompt_used_when_not_raw(self, monkeypatch):
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        generated: list[str] = []

        async def fake_gen(prompt, size=None, on_phase=None):
            generated.append(prompt)
            if on_phase is not None:
                await on_phase("painting")  # simulate the real phase callback
            return _PNG

        monkeypatch.setattr(cmd, "generate_image", fake_gen)
        inter, followups = self._interaction()

        # prepare_prompt (rewrite inside) → its output is what gets generated
        monkeypatch.setattr(cmd, "prepare_prompt", AsyncMock(return_value="rich bird"))
        await handle_generate_image_command(inter, prompt="bird")
        assert generated == ["rich bird"]
        # ONE message: live updates in place, final answer (text + image) on
        # the same message — no separate reply, no leftover "✅".
        assert len(followups) == 1
        progress = followups[0]
        assert progress["content"].startswith("🎨")
        assert "attachments" in progress
        assert progress["deleted"] is False

    async def test_dry_run_returns_prompt_without_generating(self, monkeypatch):
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        monkeypatch.setattr(cmd, "prepare_prompt", AsyncMock(return_value="rich bird"))

        def explode(prompt, size=None):  # must never be called in dry run
            raise AssertionError("generate_image called during dry run")

        monkeypatch.setattr(cmd, "generate_image", explode)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="bird", dry_run=True)
        # progress message (deleted) + the dry-run reply (no image, no generation)
        assert len(followups) == 2
        reply = followups[1]
        assert "Dry run" in reply["content"]
        assert "rich bird" in reply["content"]
        assert "file" not in reply
        assert followups[0]["deleted"] is True

    async def test_painting_update_shows_effective_size_not_none(self, monkeypatch):
        """Regression: with no --size given, `side` is None and the update used
        to read 'Painting…(None px)' — it must show the effective default."""
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        monkeypatch.setattr(cmd, "IMAGE_DEFAULT_SIZE", 1024)
        monkeypatch.setattr(cmd, "prepare_prompt", AsyncMock(return_value="rich bird"))

        async def fake_gen(prompt, size=None, on_phase=None):
            if on_phase is not None:
                await on_phase("painting")
            return _PNG

        monkeypatch.setattr(cmd, "generate_image", fake_gen)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="bird")  # no size
        edits = followups[0]["edits"]
        assert any("1024 px" in e for e in edits)
        assert not any("None px" in e for e in edits)

    async def test_in_place_edit_failure_falls_back_to_fresh_message(self, monkeypatch):
        """If the final in-place edit can't carry the attachment (API edge),
        degrade to a fresh follow-up — the image is never lost."""
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        monkeypatch.setattr(cmd, "prepare_prompt", AsyncMock(return_value="rich bird"))

        async def fake_gen(prompt, size=None, on_phase=None):
            return _PNG

        monkeypatch.setattr(cmd, "generate_image", fake_gen)

        inter = MagicMock()
        inter.response.defer = AsyncMock()
        followups: list[dict] = []

        async def capture(content=None, **kwargs):
            msg = {"content": content, **kwargs}
            followups.append(msg)

            async def edit(**ekwargs):
                if "attachments" in ekwargs:
                    raise RuntimeError("edit-with-attachment rejected")
                msg.update(ekwargs)

            msg["edit"] = edit
            msg["deleted"] = False
            return MagicMock(edit=edit, delete=AsyncMock())

        inter.followup.send = capture

        await handle_generate_image_command(inter, prompt="bird")
        # progress message (kept) + fresh image message as the fallback
        assert len(followups) == 2
        assert followups[0]["deleted"] is False
        assert "file" in followups[1]

    async def test_generation_failure_replies_error(self, monkeypatch):
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        monkeypatch.setattr(cmd, "prepare_prompt", AsyncMock(return_value="rich"))

        async def boom(prompt, size=None, on_phase=None):
            raise images.ImageGenError("studio 500")

        monkeypatch.setattr(cmd, "generate_image", boom)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="bird")
        # [0] progress message (deleted), [1] the error reply
        assert "failed" in followups[1]["content"].lower()
        assert followups[0]["deleted"] is True

    async def test_bad_size_replies_error(self, monkeypatch):
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="bird", size="gigantic")
        assert "invalid size" in followups[0]["content"]
