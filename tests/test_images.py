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
import json
import logging
import time

import httpx
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

import bot_core.images as images
from bot_core import prompt_profile
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
    images._last_bot_switch_at = 0.0
    images._hold_started_at = 0.0
    images._bot_switched = False
    images._background_tasks.clear()
    yield


class TestImageGenConfigured:
    def test_true_when_model_set(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/Qwen-Image-2.1-Turbo-GGUF")
        assert images.image_gen_configured() is True

    def test_negative_prompt_suppresses_duplicate_figures(self):
        # The prompt side bans the unnamed crowd; the negative-prompt side
        # steers the diffusion model away from the visible symptom of it.
        from config.settings import _DEFAULT_IMAGE_NEGATIVE_PROMPT as d
        for needle in ("duplicate of the same character",
                       "identical clone of a character",
                       "doppelganger of a main figure"):
            assert needle in d, needle

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
            return_value=self._resp('  "Semi-realistic fantasy illustration — detailed '
                                    'fur and feather texture, natural light, storybook '
                                    'charm. A blue-tit bird-folk warlock perched on a '
                                    'mossy log." '))
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        out = await images.rewrite_prompt("a bird")
        assert out == ('Semi-realistic fantasy illustration — detailed fur and feather '
                       'texture, natural light, storybook charm. A blue-tit bird-folk '
                       'warlock perched on a mossy log.')

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
        # First turn empty (nudged), second a real answer → recovered. This is
        # the FALLBACK prose path exercised directly (#84: the plan path runs
        # first in rewrite_prompt and has its own tests in TestComposedRewrite).
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            self._resp(""),
            self._resp("Semi-realistic fantasy illustration — detailed fur and feather "
                       "texture, natural light, storybook charm. A blue-tit warlock in "
                       "a moonlit clearing."),
        ])
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        complete = images._make_completer(client)
        assert await images._rewrite_prose("a bird", complete) == \
            ("Semi-realistic fantasy illustration — detailed fur and feather texture, "
             "natural light, storybook charm. A blue-tit warlock in a moonlit clearing.")

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
        as normal chat) before answering — e.g. grounding 'Marvin #14'."""
        import bot_core.agent_loop as agent_loop

        executed: list[str] = []

        async def fake_execute(name, args):
            executed.append(name)
            return f"{name} result for {args}"

        monkeypatch.setattr(agent_loop, "execute_tool", fake_execute)

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            self._resp('{"tool": "search_kb", "args": {"query": "Marvin #14"}}'),
            self._resp('<image_plan>\n{"scene": "Marvin guards a campfire at night", '
                       '"figures": [{"name": "Marvin", "pose": "guarding the fire"}], '
                       '"lighting": "firelight", "palette": "brass, ember", '
                       '"mood": "watchful"}\n</image_plan>'),
        ])
        monkeypatch.setattr(images, "_chat_client", lambda: client)

        out = await images.rewrite_prompt("Marvin #14 guarding a camp")
        # The lookup happened, and the composed prompt opens with the style
        # sentence and states the figure count (both are now code guarantees).
        assert executed == ["search_kb"]
        assert out.startswith(images._base_style())
        assert "Exactly 1 named figure" in out
        assert "Marvin" in out


    async def test_a_look_taken_from_a_search_hit_is_grounded(self, monkeypatch):
        """A description read out of a SEARCH result counts as retrieved.

        The verification must not demand an explicit read: a hit carries its
        file, and a `look` citing that file is grounded.
        """
        import bot_core.agent_loop as agent_loop
        executed = []

        async def fake_execute(name, args):
            executed.append(name)
            return ('8 hit(s): [{"file": '
                    '"humblewood/HumbleWood/SessionLogs/SESSION_05/notes.md", '
                    '"section": "Lexikon", "snippet": "Boss Frey: a scarred '
                    'bear-folk bandit chief with a silver tooth"}]')

        monkeypatch.setattr(agent_loop, "execute_tool", fake_execute)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            self._resp('{"tool": "search_kb", "args": {"query": "Boss Frey '
                       'bandit appearance", "scope": "lore"}}'),
            self._resp('<image_plan>\n{"scene": "Boss Frey blocks the path as '
                       'Seta Onix draws her bow", "figures": [{"name": "Seta '
                       'Onix", "pose": "drawing her bow"}, {"name": "Boss Frey", '
                       '"pose": "brandishing an axe", "look": "a scarred '
                       'bear-folk bandit chief with a silver tooth", "source": '
                       '"SESSION_05/notes.md"}], "lighting": "dappled forest '
                       'light"}\n</image_plan>'),
        ])
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        out = await images.rewrite_prompt("die ganze Gruppe stellt sich Boss Frey")
        assert executed == ["search_kb"]
        assert "bear-folk bandit chief with a silver tooth" in out
        assert "SESSION_05" not in out          # provenance is not painted

    async def test_a_named_figure_with_no_look_gets_one_lookup_pass(self, monkeypatch):
        """Code-enforced lookup, not a rule the model may ignore.

        Pass 1 names Boss Frey with no appearance and no tool call. The
        composer reports him as ungrounded, one corrective pass runs (which
        does the search), and pass 2's grounded look is what gets painted.
        """
        import bot_core.agent_loop as agent_loop
        executed = []

        async def fake_execute(name, args):
            executed.append(name)
            return ('8 hit(s): [{"file": '
                    '"humblewood/HumbleWood/SessionLogs/SESSION_05/notes.md", '
                    '"section": "Lexikon", "snippet": "Boss Frey: a scarred '
                    'bear-folk bandit chief with a silver tooth"}]')

        monkeypatch.setattr(agent_loop, "execute_tool", fake_execute)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            self._resp('<image_plan>\n{"scene": "Boss Frey blocks the path as '
                       'Seta Onix draws her bow", "figures": [{"name": "Seta '
                       'Onix", "pose": "drawing her bow"}, {"name": "Boss Frey", '
                       '"pose": "brandishing an axe"}], "lighting": "dappled '
                       'forest light"}\n</image_plan>'),
            self._resp('{"tool": "search_kb", "args": {"query": "Boss Frey '
                       'bandit appearance", "scope": "lore"}}'),
            self._resp('<image_plan>\n{"scene": "Boss Frey blocks the path as '
                       'Seta Onix draws her bow", "figures": [{"name": "Seta '
                       'Onix", "pose": "drawing her bow"}, {"name": "Boss Frey", '
                       '"pose": "brandishing an axe", "look": "a scarred '
                       'bear-folk bandit chief with a silver tooth", "source": '
                       '"SESSION_05/notes.md"}], "lighting": "dappled forest '
                       'light"}\n</image_plan>'),
        ])
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        out = await images.rewrite_prompt("die ganze Gruppe stellt sich Boss Frey")
        assert executed == ["search_kb"]
        assert client.chat.completions.create.call_count == 3
        assert "bear-folk bandit chief with a silver tooth" in out

    async def test_a_collective_scene_sentence_comes_back_as_a_finding(self,
                                                                       monkeypatch):
        """Analysis §17.3, end to end: the composer does not rewrite the plan's
        sentence. It reports the shape, one correction turn puts that report in
        front of the model that wrote it, and the SECOND plan's sentence is what
        gets painted.

        The failure this replaces was scripted: a regex generated a name list to
        replace "die ganze Gruppe", and that generated sentence had its own bugs
        (§13) — the cast was named twice in one sentence, which is exactly the
        shape that paints a character twice.
        """
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            self._resp('<image_plan>\n{"scene": "die ganze Gruppe stellt sich '
                       'Boss Frey in den Weg", "figures": [{"name": "Seta Onix", '
                       '"pose": "drawing her bow"}, {"name": "Boss Frey", "pose": '
                       '"brandishing an axe"}], "lighting": "dappled forest '
                       'light"}\n</image_plan>'),
            self._resp('<image_plan>\n{"scene": "Seta Onix and Marvin stand '
                       'against Boss Frey on the Old Bridge", "figures": '
                       '[{"name": "Seta Onix", "pose": "drawing her bow"}, '
                       '{"name": "Marvin", "pose": "rolling forward"}, '
                       '{"name": "Boss Frey", "pose": "brandishing an axe"}], '
                       '"lighting": "dappled forest light"}\n</image_plan>'),
        ])
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        out = await images.rewrite_prompt("die ganze Gruppe stellt sich Boss Frey")
        users = [m.content if hasattr(m, "content") else m["content"]
                 for call in client.chat.completions.create.call_args_list
                 for m in call.kwargs["messages"] if m["role"] == "user"]
        assert client.chat.completions.create.call_count == 2, "exactly one turn back"
        assert any("collective_subject" in u for u in users), users[-1]
        assert "Do not describe the picture in prose" in users[-1]
        assert "die ganze Gruppe" not in out, out
        assert "Seta Onix and Marvin stand against Boss Frey" in out, out
        assert "Exactly 3 named figures" in out, out

    async def test_a_correction_that_makes_the_plan_worse_is_not_painted(self,
                                                                         monkeypatch):
        """Live 2026-11-19, round 15: the first plan said "Boss Frey, a bandit chief,
        stands alone in a moonlit clearing" — one open finding, the description sitting
        in the scene instead of in the figure's `look`. It was asked to move it.

        The plan that came back fixed that and said, in the sentence the renderer
        paints: "Boss Frey stands alone in a moonlit clearing (species unknown to the
        vault) holding no weapon" — a meta aside the model would paint as text, a
        negation (which paints the thing it names), and a figure with no description
        left at all. Two problems answered with three.

        The composer has both plans composed. It paints the one with fewer open
        findings, and says so in the log. (That comparison belongs to the CORRECTING
        configuration — `IMAGE_CODE_CORRECTIONS` on — so this test turns it on; the
        shipped default is off, analysis §25.)
        """
        client = MagicMock()
        monkeypatch.setattr(images, "IMAGE_CODE_CORRECTIONS", True)
        client.chat.completions.create = AsyncMock(side_effect=[
            self._resp('<image_plan>\n{"scene": "Boss Frey, a bandit chief, stands '
                       'alone in a moonlit clearing", "figures": [{"name": "Boss Frey", '
                       '"pose": "gripping a heavy axe"}], "lighting": "cold moonlight"}'
                       '\n</image_plan>'),
            self._resp('<image_plan>\n{"scene": "Boss Frey stands alone in a moonlit '
                       'clearing (species unknown to the vault) holding no weapon", '
                       '"figures": [{"name": "Boss Frey", "pose": "standing"}], '
                       '"lighting": "cold moonlight"}\n</image_plan>'),
        ])
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        out = await images.rewrite_prompt("Boss Frey allein im Mondlicht")
        assert client.chat.completions.create.call_count == 2
        assert "a bandit chief" in out, out          # the FIRST plan is what got painted
        assert "unknown to the vault" not in out, out
        assert "no weapon" not in out, out
        assert "Exactly 1 named figure" in out, out

    async def test_a_lookup_that_finds_nothing_still_paints_the_name(self, monkeypatch):
        """One corrective pass, then the policy's fallback: name + pose."""
        import bot_core.agent_loop as agent_loop
        executed = []

        async def fake_execute(name, args):
            executed.append(name)
            return "0 hit(s): []"

        monkeypatch.setattr(agent_loop, "execute_tool", fake_execute)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            self._resp('<image_plan>\n{"scene": "Boss Frey blocks the path", '
                       '"figures": [{"name": "Boss Frey", "pose": "brandishing '
                       'an axe"}], "lighting": "forest light"}\n</image_plan>'),
            self._resp('{"tool": "search_kb", "args": {"query": "Boss Frey '
                       'appearance", "scope": "lore"}}'),
            self._resp('<image_plan>\n{"scene": "Boss Frey blocks the path", '
                       '"figures": [{"name": "Boss Frey", "pose": "brandishing '
                       'an axe"}], "lighting": "forest light"}\n</image_plan>'),
        ])
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        out = await images.rewrite_prompt("Boss Frey blocks the path")
        assert "Boss Frey: brandishing an axe" in out
        assert executed == ["search_kb"]          # looked, once
        assert "invent" not in out.lower()


class TestRewriteReasoningEffort:
    """IMAGE_REASONING_EFFORT / IMAGE_REWRITE_MAX_TOKENS reach the rewrite turns.

    Measured on the live backend (12 rewrites: 2 requests × 3 efforts × 2 rounds
    run in opposite order) — docs/internal/
    IMAGE_PROMPT_PIPELINE_ANALYSIS_2026-11.md §12: `low` median 42.7 s,
    `medium` 52.2 s, `high` 152.8 s. `medium` wrote measurably more PAINTABLE
    detail (the shape of a three-lens frame, a coat's collar construction) with
    identical override correctness; `high` bought nothing at 3.6× the time. The
    routing turn stays unreasoned — it only picks tools.
    """

    @staticmethod
    def _resp(content: str):
        return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])

    async def test_a_thinking_turn_pays_the_configured_effort(self, monkeypatch):
        from config import settings
        assert settings.IMAGE_REASONING_EFFORT == "medium"      # the measured default
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._resp("ok"))
        complete = images._make_completer(client)
        await complete([{"role": "user", "content": "plan"}], "m", thinking=True)
        assert client.chat.completions.create.call_args.kwargs["extra_body"] == \
            {"reasoning_effort": "medium"}

    async def test_every_turn_pays_the_configured_effort_thinking_flag_included(self):
        """A plan answer arrives out of a DECISION turn (measured: one rewrite,
        one turn, thinking=False) — gating the effort on `thinking` made the
        setting inert, so both turn kinds now get it."""
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._resp("ok"))
        complete = images._make_completer(client)
        await complete([{"role": "user", "content": "pick tools"}], "m", thinking=False)
        await complete([{"role": "user", "content": "plan"}], "m", thinking=True)
        bodies = [c.kwargs["extra_body"]
                  for c in client.chat.completions.create.call_args_list]
        assert bodies == [{"reasoning_effort": "medium"}, {"reasoning_effort": "medium"}]

    async def test_an_explicit_off_setting_still_thinks_on_no_turn(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_REASONING_EFFORT", "off")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._resp("ok"))
        complete = images._make_completer(client)
        await complete([{"role": "user", "content": "pick tools"}], "m", thinking=False)
        await complete([{"role": "user", "content": "plan"}], "m", thinking=True)
        off = {"reasoning_effort": "none", "enable_thinking": False}
        assert [c.kwargs["extra_body"]
                for c in client.chat.completions.create.call_args_list] == [off, off]

    async def test_the_effort_is_overridable_without_touching_the_loop(self, monkeypatch):
        monkeypatch.setattr(images, "IMAGE_REASONING_EFFORT", "low")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._resp("ok"))
        complete = images._make_completer(client)
        await complete([{"role": "user", "content": "plan"}], "m", thinking=True)
        assert client.chat.completions.create.call_args.kwargs["extra_body"] == \
            {"reasoning_effort": "low"}

    async def test_the_output_cap_is_the_one_settings_derives(self, monkeypatch):
        # A thinking model spends part of the cap on thinking and can then return
        # an empty or truncated plan (observed live at 2048) — so the cap the
        # rewrite sends has to be the effort-aware one.
        from config import settings
        monkeypatch.setattr(images, "IMAGE_REWRITE_MAX_TOKENS", settings.IMAGE_REWRITE_MAX_TOKENS)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._resp("ok"))
        complete = images._make_completer(client)
        await complete([{"role": "user", "content": "plan"}], "m", thinking=True)
        sent = client.chat.completions.create.call_args.kwargs
        assert sent["max_tokens"] == settings.IMAGE_REWRITE_MAX_TOKENS == 6144

    def test_the_cap_grows_with_the_effort_and_never_shrinks_below_the_old_one(self):
        from config.settings import _rewrite_max_tokens as cap
        assert cap(None) == cap("off") == cap("low") == 4096
        assert cap("medium") == cap("high") == cap("xhigh") == 6144


class TestPromptShapeGuard:
    """A rewrite must be a PROMPT, not narration (found live 2026-11-18).

    A group request returned the model's plan — "I need to look up the Player
    Characters sheet…" — with no tool calls, and nothing in the old guards
    would have stopped it from being PAINTED.
    """

    @staticmethod
    def _resp(content: str):
        return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])

    def test_the_policy_file_has_no_style_sentence_of_its_own(self):
        """The fallback policy used to paste the style sentence a second time, which
        meant a profile edit and the policy file could drift apart — the model was
        told to copy VERBATIM a sentence the composer no longer used. The file now
        carries the `{{OPENING_STYLE}}` placeholder and no wording of its own, so the
        Campaign Prompt Profile is the only source."""
        policy = images._POLICY_FILE.read_text(encoding="utf-8")
        assert "{{OPENING_STYLE}}" in policy
        assert "Semi-realistic fantasy illustration" not in policy
        style = images._base_style()
        if style:
            assert style not in policy

    def test_real_prompts_pass(self):
        base = images._base_style()
        assert images.looks_like_image_prompt(base)
        assert images.looks_like_image_prompt(base + ". Colum stands on a log.")
        # live model habit: stray space before the period
        assert images.looks_like_image_prompt(base + " . Seta stands in a clearing.")
        # paraphrased opening still passes (drift, not narration)
        assert images.looks_like_image_prompt(
            "Semi-realistic fantasy illustration with soft fur texture, a raccoon "
            "artificer at a workbench.")

    def test_narration_and_empty_rejected(self):
        assert not images.looks_like_image_prompt(
            "I need to look up the Player Characters sheet to get all party members "
            "and their Visual Anchors.")
        assert not images.looks_like_image_prompt(
            "Sure! Here is what I would draw: a hedgehog monk.")
        assert not images.looks_like_image_prompt("")
        assert not images.looks_like_image_prompt("   ")

    async def test_narration_gets_one_corrective_retry(self, monkeypatch):
        # Fallback prose path, driven directly (see TestComposedRewrite for the
        # plan path that normally runs first).
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[
            self._resp("I need to look up the Player Characters sheet first."),
            self._resp(images._base_style() + ". Colum and Seta face a bandit; Colum "
                       "lunges, Seta nocks an arrow."),
        ])
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        complete = images._make_completer(client)
        out = await images._rewrite_prose("die ganze Gruppe", complete)
        assert out.startswith(images._base_style())
        # the retry restates the request and says no lookup is needed
        sent = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        assert sent[-1]["content"].startswith("Your previous answer was not an image prompt")
        assert "die ganze Gruppe" in sent[-1]["content"]

    async def test_narration_twice_raises_not_painted(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=self._resp("Let me search the knowledge base for that."))
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        with pytest.raises(images.ImageGenError, match="no prompt"):
            await images.rewrite_prompt("die ganze Gruppe")
        # degrades to the RAW request rather than painting a plan
        assert await images.prepare_prompt("die ganze Gruppe") == "die ganze Gruppe"


def _parse_party_sheet_safe():
    """Read the sheet the test points at (patched or the campaign profile's) and
    parse it."""
    return images._parse_party_sheet(images._party_sheet_file().read_text(encoding="utf-8"))


class TestPartyDossier:
    """Party roster + Visual Anchors injected into the rewrite system prompt
    from the vault sheet, so group requests never depend on a retrieval win."""

    _SHEET = """# Colum Bidae
Birdfolk (blue tit), Warlock. Pronouns: he/him.

![[Colum.webp]]

## Appearance
A small plump blue-tit warlock bird.
Rendering: high-detail digital fantasy illustration.

## Visual Anchor (group images — reuse VERBATIM for named appearances)

> A small plump blue-tit warlock bird: golden breast, blue wings.

# Romi Stachelpfote
Hedge (hedgehog), Monk. Pronouns: he/him.

## Appearance
A small round hedgehog.

## Visual Anchor (group images — reuse VERBATIM for named appearances)

> A small anthropomorphic hedgehog monk: taupe quills, golden rose.

# Some NPC Without An Anchor
Squirrel rogue.

## Appearance
A squirrel.
"""

    def _point_at(self, monkeypatch, tmp_path, text=None):
        sheet = tmp_path / "Player Characters.md"
        sheet.write_text(text if text is not None else self._SHEET, encoding="utf-8")
        monkeypatch.setattr(images, "_party_sheet_file", lambda: sheet)
        return sheet

    def test_parses_names_species_line_and_anchor(self, monkeypatch, tmp_path):
        self._point_at(monkeypatch, tmp_path)
        entries = images._parse_party_sheet(images._party_sheet_file().read_text(encoding="utf-8"))
        assert [e[0] for e in entries] == ["Colum Bidae", "Romi Stachelpfote"]
        assert entries[0][1].startswith("Birdfolk (blue tit)")
        assert entries[1][2] == ("A small anthropomorphic hedgehog monk: taupe "
                                 "quills, golden rose.")

    def test_dossier_reaches_the_system_prompt_verbatim(self, monkeypatch, tmp_path):
        self._point_at(monkeypatch, tmp_path)
        s = images._load_rewrite_system()
        assert "{{PARTY_DOSSIER}}" not in s
        assert "Visual Anchor: A small plump blue-tit warlock bird: golden breast, " \
               "blue wings." in s
        assert "copy their anchor" in s
        # the Appearance/Rendering blocks are NOT injected — anchors only
        assert "high-detail digital fantasy illustration" not in s
        # a character without an anchor is not advertised as party
        assert "Some NPC Without An Anchor" not in s

    def test_dossier_orders_batched_lookup_of_named_non_party_characters(self, tmp_path,
                                                                         monkeypatch):
        # The party half of a group prompt needs no retrieval; the NPC half MUST
        # be retrieved, in ONE batch — and the dossier is where that instruction
        # reliably reaches the model (it sits in the system prompt, not in a
        # policy section the model may skim past).
        sheet = tmp_path / "Player Characters.md"
        sheet.write_text(
            "# Colum Bidae\n## Visual Anchor\n> A small plump blue-tit warlock "
            "bird, blue wings.\n\n", encoding="utf-8")
        monkeypatch.setattr(images, "_party_sheet_file", lambda: sheet)
        head = images._load_party_dossier()
        for needle in ("MUST be looked up in the",
                       "knowledge base BEFORE you write it",
                       "Lexikon",
                       "NEVER invent an appearance for a named NPC",
                       "ONE array of tool calls in a SINGLE decision turn"):
            assert needle in head, needle

    def test_missing_sheet_degrades_to_a_lookup_instruction(self, monkeypatch, tmp_path):
        monkeypatch.setattr(images, "_party_sheet_file", lambda: tmp_path / "absent.md")
        s = images._load_rewrite_system()
        assert "No party dossier available" in s
        assert "{{PARTY_DOSSIER}}" not in s

    def test_fenced_blocks_are_not_read_as_character_headings(self, monkeypatch, tmp_path):
        # A stat block inside the sheet must not create a fake party member.
        self._point_at(monkeypatch, tmp_path, text=self._SHEET + "\n```\n# Fake Head\n"
                       "## Visual Anchor\n> fake anchor.\n```\n")
        names = [e[0] for e in _parse_party_sheet_safe()]
        assert names == ["Colum Bidae", "Romi Stachelpfote"]

    def test_sheet_without_anchors_degrades_too(self, monkeypatch, tmp_path):
        self._point_at(monkeypatch, tmp_path,
                       text="# Nobody\nSquirrel rogue.\n\n## Appearance\nA squirrel.\n")
        assert images._load_party_dossier() == ""
        assert "No party dossier available" in images._load_rewrite_system()


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

        # The real ensure_image_mode switches the card (that is what earns the
        # hold) — the fake must do the same.
        async def fake_ensure(on_phase=None):
            images._bot_switched = True

        monkeypatch.setattr(images, "ensure_image_mode", fake_ensure)
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
            if mode == "image":
                images._bot_switched = True

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


class TestGenerateImagesBatch:
    """generate_images (#69): every prompt painted in ONE painting window —
    image mode ensured once, all paints, rag mode restored exactly once."""

    def _patch(self, monkeypatch) -> tuple[MagicMock, list[str]]:
        """Fake the whole mode pipeline; returns (image client, order log)."""
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        order: list[str] = []

        async def fake_ensure(on_phase=None) -> None:
            order.append("ensure")
            # The real ensure_image_mode emits "loading" through the guarded
            # _emit_phase — mirror that (a broken callback must not escape).
            if on_phase is not None:
                try:
                    await on_phase("loading")
                except Exception:
                    pass

        async def fake_restore() -> None:
            order.append("restore")

        monkeypatch.setattr(images, "ensure_image_mode", fake_ensure)
        monkeypatch.setattr(images, "_restore_rag_after_painting", fake_restore)

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_image_client", lambda: client)
        return client, order

    async def test_one_ensure_n_paints_one_restore(self, monkeypatch):
        client, order = self._patch(monkeypatch)
        out = await images.generate_images(["a", "b", "c"])
        assert out == [_PNG, _PNG, _PNG]
        # ONE window: ensure once before all paints, restore once after.
        assert order.count("ensure") == 1
        assert order.count("restore") == 1
        assert order.index("ensure") < order.index("restore")
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
        client.images.generate = AsyncMock(
            side_effect=[_Fake4xx("bad prompt"), resp, resp])

        out = await images.generate_images(["a", "b", "c"])
        # Moment 1 failed (4xx -> no retry), moments 2+3 still painted.
        assert isinstance(out[0], _Fake4xx)
        assert out[1] == _PNG
        assert out[2] == _PNG

    async def test_failed_batch_still_restores_rag(self, monkeypatch):
        client, order = self._patch(monkeypatch)

        class _Fake4xx(Exception):
            status_code = 400

        monkeypatch.setattr(images, "APIStatusError", _Fake4xx)
        client.images.generate = AsyncMock(side_effect=_Fake4xx("down"))

        out = await images.generate_images(["a", "b"])
        assert all(isinstance(r, _Fake4xx) for r in out)
        assert order == ["ensure", "restore"]  # rag mode restored no matter what

    async def test_broken_phase_callback_never_fails_batch(self, monkeypatch):
        self._patch(monkeypatch)

        def broken(name: str) -> None:
            raise RuntimeError("discord edit failed")

        out = await images.generate_images(["a", "b"], on_phase=broken)
        assert out == [_PNG, _PNG]

    async def test_empty_batch_is_a_noop(self, monkeypatch):
        client, order = self._patch(monkeypatch)
        out = await images.generate_images([])
        assert out == []
        # No prompts -> no GPU mode cycle paid for nothing.
        assert order == []
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
        # 2026-10-27 correction: NOBODY has googly eyes except Colum Bidae (he).
        # 2026-11-17 vault rename: "Column Bidae" → "Colum Bidae".
        s = images._load_rewrite_system()
        assert "Googly eyes are NOT a species trait" in s
        assert "belong ONLY to Colum Bidae" in s
        assert "give EVERY one of them those" not in s

    def test_fixed_template_with_length_budgets(self):
        # 2026-10-28 optimization: fixed [STYLE][SCENE][CHARACTERS][LIGHTING]
        # structure + verbatim visual anchors + hard character budget.
        s = images._load_rewrite_system()
        for part in ("[STYLE]", "[SCENE]", "[CHARACTERS]", "[LIGHTING]"):
            assert part in s
        assert "copy the ENTIRE anchor block VERBATIM" in s
        assert "under 1300 characters" in s
        assert "up to 2000 characters" in s
        # species words must come from the sheet, never be invented (live: Seta
        # kept coming out a mangled 'cochran')
        assert "Never invent or mangle a species word" in s
        # group scenes: the party anchors are IN the system prompt now, so
        # there is no sheet lookup left to fail, and no silent character drop
        # (live: 'the whole party' came back with 3 of 5 members; 2026-11-18:
        # it came back with invented members instead)
        assert "EVERY entry of the Party dossier" in s
        assert "never re-read the 'Player Characters' sheet" in s
        assert "never silently drop any" in s
        # negations stay OUT of the positive prompt (negative prompt handles them)
        assert "Never write negations" in s

    def test_named_npc_rule_batching_and_amounts(self):
        # Everything the tricky group prompts exposed, now stated as policy: a
        # NAMED character outside the dossier is looked up before being written
        # (Frey and Alisa came out as invented fox-folk / hare-folk until this
        # rule existed); the lookups go out as ONE batched array because the
        # agent loop has only a few decision turns and one search per turn runs
        # out before the model can write; and an amount in the request is a fact
        # about the scene, not lettering to paint (a render once drew a pouch
        # labeled "70 Gold").
        s = images._load_rewrite_system()
        for needle in ("every NAMED character the Party dossier does not list is "
                       "looked up before you write it",
                       "ONE array of tool calls in a SINGLE decision turn",
                       "never assign a species or a colouring to a named NPC from "
                       "imagination",
                       "NOT lettering to render"):
            assert needle in s, needle

    def test_no_collective_subject_and_a_figure_count(self):
        # Both group images that came back wrong had the cast named TWICE: a
        # collective subject in the scene sentence ("Party members battle…",
        # "the whole upright party squaring off…") plus six individually
        # described characters. The model then painted an unnamed crowd on top
        # of the enumerated cast — a second Colum in the background in one
        # image, no individual Colum at all in the other. Policy answer: a
        # collective subject is banned and the scene sentence must state how
        # many bodies the image contains.
        s = images._load_rewrite_system()
        for needle in ("NEVER give the cast a collective subject",
                       "an extra UNNAMED crowd on top of the characters",
                       "Exactly N named figures in this image",
                       "list each character EXACTLY ONCE",
                       "never add an unnamed look-alike"):
            assert needle in s, needle

    def test_character_style_lines_never_copied_into_prompt(self):
        s = images._load_rewrite_system()
        assert "do NOT copy them into the final prompt" in s

    def test_no_painterly_style_prescribed(self):
        # 'painterly' may only appear as a prohibition, never as the style itself.
        assert "semi-realistic painterly" not in images._load_rewrite_system().lower()


class TestRewritePolicyLoading:
    """The policy is layered: repo file + vault character-rules note, both
    re-read per generation; missing files degrade to built-in defaults."""

    def test_vault_character_rules_are_injected(self, monkeypatch, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("ZZZ_CUSTOM_RULE_MARKER applies.", encoding="utf-8")
        monkeypatch.setattr(images, "_CHAR_RULES_FILE", note)
        assert "ZZZ_CUSTOM_RULE_MARKER" in images._load_rewrite_system()

    def test_missing_vault_note_falls_back_to_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(images, "_CHAR_RULES_FILE", tmp_path / "absent.md")
        s = images._load_rewrite_system()
        # The default is a neutral marker, not another campaign's rules: the built-in
        # used to inject "Googly eyes … belong ONLY to Colum Bidae", which is
        # Humblewood canon living in Python (retired 2026-11-22, analysis §20).
        assert "character-rules note is missing" in s
        assert "belong ONLY to Colum Bidae" not in s

    def test_the_opening_style_sentence_comes_from_the_profile(self, monkeypatch):
        """Neither rewrite path carries a style sentence in the repo."""
        def fake_load():
            return images.prompt_profile.Profile(
                style="ZZZ_STYLE_SENTENCE", world_rule="ZZZ_WORLD_RULE",
                negative_prompt="", party_sheet="", base_chars=1100,
                chars_per_figure=420, max_chars=3600, foreground_figures=4,
                source="<test>")
        monkeypatch.setattr(images.prompt_profile, "load", fake_load)
        assert images._opening_style() == "ZZZ_STYLE_SENTENCE. ZZZ_WORLD_RULE"
        s = images._load_rewrite_system()
        assert "ZZZ_STYLE_SENTENCE. ZZZ_WORLD_RULE" in s
        assert "Semi-realistic fantasy illustration" not in s

    def test_a_profile_with_no_world_rule_gets_no_such_sentence(self, monkeypatch):
        def fake_load():
            return images.prompt_profile.Profile(
                style="ZZZ_STYLE_ONLY", world_rule="", negative_prompt="",
                party_sheet="", base_chars=1100, chars_per_figure=420,
                max_chars=3600, foreground_figures=4, source="<test>")
        monkeypatch.setattr(images.prompt_profile, "load", fake_load)
        assert images._opening_style() == "ZZZ_STYLE_ONLY"

    def test_missing_repo_file_falls_back_to_builtin_policy(self, monkeypatch, tmp_path):
        monkeypatch.setattr(images, "_POLICY_FILE", tmp_path / "absent.md")
        s = images._load_rewrite_system()
        assert "[STYLE]" in s  # built-in fallback, not a crash

    def test_human_header_and_placeholder_not_injected(self):
        s = images._load_rewrite_system()
        assert "Source of truth for" not in s  # human-facing header stripped
        assert "{{CHARACTER_RULES}}" not in s  # placeholder fully replaced


class TestBackendHardening:
    """Review follow-ups: a retry re-ensures model residency; connection-level
    failures surface as the user-safe ImageGenError instead of a raw httpx error."""

    async def test_transient_failure_re_ensures_residency_before_retry(self, monkeypatch):
        from openai import APIStatusError as RealAPIStatusError

        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        # Mode entry is not what this test is about — patch it so the only
        # status/load calls come from the retry re-ensure itself.
        monkeypatch.setattr(images, "ensure_image_mode", AsyncMock())
        monkeypatch.setattr(images, "_restore_rag_after_painting", AsyncMock())
        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        request = httpx.Request("POST", "http://x/v1/images/generate")
        http_resp = httpx.Response(500, request=request)
        # openai 1.x: __init__(message, request=…, body=…, response=…)
        # openai 3.x: __init__(message, response=…, body=…) — build only the
        # kwargs this SDK version actually accepts.
        import inspect
        params = inspect.signature(RealAPIStatusError.__init__).parameters
        kwargs = {k: v for k, v in
                  (("request", request), ("body", None), ("response", http_resp))
                  if k in params}
        err = RealAPIStatusError("boom", **kwargs)
        assert getattr(err, "status_code", 500) == 500
        client.images.generate = AsyncMock(side_effect=[err, resp])
        monkeypatch.setattr(images, "_image_client", lambda: client)

        status = AsyncMock(return_value={"loaded": False})
        load = AsyncMock()
        monkeypatch.setattr(images, "_image_status", status)
        monkeypatch.setattr(images, "_ensure_image_loaded", load)
        # _reensure_before_retry opens a real httpx.AsyncClient — stand in with
        # an async-context-manager double.
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=MagicMock())
        cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(images.httpx, "AsyncClient", lambda **kw: cm)
        monkeypatch.setattr(images.asyncio, "sleep", AsyncMock())

        out = await images.generate_image("a bird")
        assert out == _PNG
        status.assert_awaited_once()   # one residency check before the retry
        load.assert_awaited_once()     # model had been dropped -> reload

    async def test_unreachable_backend_raises_user_safe_error(self, monkeypatch):
        client = MagicMock()
        client.request = AsyncMock(
            side_effect=images.httpx.ConnectError("connection refused"))
        with pytest.raises(images.ImageGenError, match="unreachable"):
            await images._image_status(client)


class TestRewritePromptShape:
    """What the rewrite is actually TOLD (analysis §15).

    Measured in the prompt as sent: the shared chat-path tool addendum ended
    "answer in plain text (no JSON)" directly under a contract demanding a JSON
    block; the vault note's 610-character header written for the human who edits
    it was injected whole; the contract's own example named a live campaign NPC
    with an invented species; and the request arrived as a bare sentence with no
    memory of the frame the same channel had just painted.
    """

    @staticmethod
    def _resp(content: str):
        return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])

    @staticmethod
    def _plan(scene: str, figures: str) -> str:
        return ('<image_plan>\n{"scene": "%s", "figures": [%s], '
                '"lighting": "forest light", "palette": "moss green", '
                '"mood": "tense"}\n</image_plan>' % (scene, figures))

    @staticmethod
    def _systems(client) -> list[str]:
        """The system message of every completion the loop made."""
        return [c.kwargs["messages"][0]["content"]
                for c in client.chat.completions.create.call_args_list]

    @staticmethod
    def _users(client) -> list[list[str]]:
        return [[m["content"] for m in c.kwargs["messages"] if m["role"] == "user"]
                for c in client.chat.completions.create.call_args_list]

    @staticmethod
    def _fresh_memory(monkeypatch):
        from collections import OrderedDict
        monkeypatch.setattr(images, "_PLAN_MEMORY", OrderedDict())

    async def test_the_answer_shape_is_said_by_code_not_left_to_the_chat_block(
            self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._resp(
            self._plan("Seta Onix draws her bow on the forest path",
                       '{"name": "Seta Onix", "pose": "drawing her bow"}')))
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        await images.rewrite_prompt("seta aiming on the path")
        system = self._systems(client)[0]
        assert "must be the <image_plan> block" in system
        # The contradiction that made narration-instead-of-plan answers plausible.
        assert "plain text (no JSON)" not in system

    async def test_the_image_rewrite_is_not_told_not_to_re_verify_nothing(
            self, monkeypatch):
        """No KB context is attached here — that paragraph only ever has the
        roster PREVIEW in view, and it discourages the lookups a missing NPC
        appearance needs."""
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._resp(
            self._plan("Seta Onix draws her bow on the forest path",
                       '{"name": "Seta Onix", "pose": "drawing her bow"}')))
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        await images.rewrite_prompt("seta aiming on the path")
        assert "already attached to this conversation" not in self._systems(client)[0]

    def _client_answering(self, monkeypatch, pick):
        """A fake backend that answers EVERY turn, chosen from the turn in view.

        Robust to how many turns a path actually takes: a lookup pass, a retry or
        a forced final completion all still get an answer instead of running the
        mock dry.
        """
        client = MagicMock()

        def _side(*a, **kw):
            users = [m["content"] for m in kw["messages"] if m["role"] == "user"]
            return self._resp(pick(users[-1] if users else ""))

        client.chat.completions.create = AsyncMock(side_effect=_side)
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        return client

    async def test_the_lookup_pass_is_told_the_same_thing(self, monkeypatch):
        """The enforced lookup pass is a SECOND loop; it must carry the same
        answer shape, not the chat-path one."""
        first = self._plan(
            "Boss Frey blocks the path as Seta Onix draws her bow",
            '{"name": "Seta Onix", "pose": "drawing her bow"}, '
            '{"name": "Boss Frey", "pose": "blocking the path"}')
        after = self._plan(
            "Boss Frey blocks the path as Seta Onix draws her bow",
            '{"name": "Seta Onix", "pose": "drawing her bow"}, '
            '{"name": "Boss Frey", "pose": "blocking the path"}')
        client = self._client_answering(
            monkeypatch,
            lambda turn: after if "not in the party sheet" in turn else first)
        await images.rewrite_prompt("boss frey blocks the path")
        systems = self._systems(client)
        assert len(systems) >= 2          # the lookup pass ran as its own loop
        assert all("must be the <image_plan> block" in s for s in systems)
        assert all("plain text (no JSON)" not in s for s in systems)

    def test_the_chat_path_keeps_its_own_answer_shape(self):
        from kb import tools
        default = tools.tool_prompt_block()
        assert "answer in plain text (no JSON)" in default
        assert "already attached to this conversation" in default
        tailored = tools.tool_prompt_block(
            answer_instruction="Answer with the <image_plan> block only.",
            mention_attached_context=False)
        assert "Answer with the <image_plan> block only." in tailored
        assert "already attached to this conversation" not in tailored
        for name in ("search_kb", "read_section", "list_sections",
                     "find_in_kb", "list_sessions"):
            assert name in tailored          # the tools themselves are shared

    def test_the_vault_note_owners_instructions_are_not_sent_to_the_model(self):
        for loaded in (images._load_plan_system(), images._load_rewrite_system()):
            assert "Edit freely" not in loaded
            assert "This note is loaded into" not in loaded
        if not images._CHAR_RULES_FILE.exists():
            pytest.skip("vault policy note not present (data/ is gitignored)")
        # The rules themselves still arrive — only the human-facing header goes.
        assert "Googly eyes" in images._load_plan_system()

    def test_the_plan_example_names_nobody_from_the_campaign(self):
        raw = images._PLAN_FILE.read_text(encoding="utf-8")
        example = raw.split("```")[1]
        assert "<image_plan>" in example
        assert "Pell Wicker" in example and "Orin Vek" in example
        # Measured: the example used to hand the model "Boss Frey … heavy-set
        # badger-folk bandit in a torn grey coat" — a ready-made answer to the
        # live request we were testing, for an NPC the vault has no appearance for.
        for name in ("Boss Frey", "badger", "Colum", "Seta", "Marvin",
                     "Trixy", "Romi", "Bubo"):
            assert name.lower() not in example.lower()

    async def test_the_request_travels_as_a_labelled_block(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._resp(
            self._plan("Seta Onix draws her bow on the forest path",
                       '{"name": "Seta Onix", "pose": "drawing her bow"}')))
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        self._fresh_memory(monkeypatch)
        await images.rewrite_prompt("seta aiming on the path", memory_key=(1, 2))
        users = self._users(client)[0]
        assert any("<request>\nseta aiming on the path\n</request>" in u
                   for u in users)

    async def test_a_follow_up_in_the_same_channel_sees_the_previous_plan(
            self, monkeypatch):
        self._fresh_memory(monkeypatch)
        plans = {
            "seta aiming": self._plan(
                "Seta Onix draws her bow on the forest path",
                '{"name": "Seta Onix", "pose": "drawing her bow"}'),
            "torches": self._plan(
                "Seta Onix draws her bow under torchlight",
                '{"name": "Seta Onix", "pose": "drawing her bow"}'),
            "forge": self._plan(
                "Trixy Smoldersome works at the forge",
                '{"name": "Trixy Smoldersome", "pose": "turning a wrench"}'),
        }

        def _pick(turn: str) -> str:
            for needle, plan in plans.items():
                if needle in turn:
                    return plan
            return plans["forge"]

        client = self._client_answering(monkeypatch, _pick)
        await images.rewrite_prompt("seta aiming on the path", memory_key=(1, 2))
        assert "<previous_image_plan>" not in self._users(client)[0][-1]
        await images.rewrite_prompt("again but with torches", memory_key=(1, 2))
        second = self._users(client)[1][-1]
        assert "<previous_image_plan>" in second
        assert "Seta Onix draws her bow on the forest path" in second
        assert "drawing her bow" in second
        # A different channel is a different conversation.
        await images.rewrite_prompt("a forge at dusk", memory_key=(1, 3))
        assert "<previous_image_plan>" not in self._users(client)[2][-1]

    async def test_the_previous_plan_is_small_and_marked_as_optional(
            self, monkeypatch):
        self._fresh_memory(monkeypatch)
        long_scene = " ".join(f"word{i}" for i in range(300))
        images._remember_plan((1, 2), images.image_prompt.ImagePlan(
            scene=long_scene,
            figures=[images.image_prompt.Figure(name="Seta Onix", pose="walking")]))
        block = images._previous_plan_block((1, 2))
        assert len(block) <= images._PREVIOUS_PLAN_MAX_CHARS + 400
        assert "ignore this block entirely" in block   # unrelated → no contamination
        assert "word299" not in block

    def test_the_plan_memory_is_bounded(self, monkeypatch):
        self._fresh_memory(monkeypatch)
        plan = images.image_prompt.ImagePlan(scene="s",
                                             figures=[images.image_prompt.Figure(
                                                 name="Seta Onix", pose="walking")])
        for i in range(images._PLAN_MEMORY_MAX + 8):
            images._remember_plan((1, i), plan)
        assert len(images._PLAN_MEMORY) == images._PLAN_MEMORY_MAX
        assert (1, 0) not in images._PLAN_MEMORY

    def test_the_plan_memory_forgets_a_stale_frame_instead_of_hoarding_it(self):
        assert images._previous_plan_block(None) == ""


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
        prepared = AsyncMock(side_effect=lambda text, raw=False, **kw: text)
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
        prepared.assert_awaited_once_with(
            "verbatim bird", raw=True,
            memory_key=(inter.guild_id, inter.channel_id),
            on_phase=ANY)
        assert generated == ["verbatim bird"]
        # ONE message: the final answer (text + image) landed in place on the
        # progress message — no separate reply.
        assert len(followups) == 1
        assert followups[0]["content"].startswith("🎨")
        # The PREP callback is wired too: a rewrite that must wait for a painting
        # window says so on the progress message instead of going silent.
        prep_phase = prepared.await_args.kwargs["on_phase"]
        await prep_phase("awaiting_lookup")
        assert any("vault lookup" in e for e in followups[0]["edits"]), \
            followups[0]["edits"]
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

        # prepare_prompt (rewrite inside) -> its output is what gets generated
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

    async def test_backend_connectivity_failure_replies_error_not_crash(self, monkeypatch):
        """Review follow-up: a raw OpenAI connectivity error must degrade to the
        same user-safe reply (the mode restore in generate_image still runs)."""
        from openai import APIConnectionError

        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        monkeypatch.setattr(cmd, "prepare_prompt", AsyncMock(return_value="rich"))

        async def boom(prompt, size=None, on_phase=None):
            raise APIConnectionError(request=MagicMock())

        monkeypatch.setattr(cmd, "generate_image", boom)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="bird")
        assert "failed" in followups[1]["content"].lower()
        assert followups[0]["deleted"] is True

    async def test_bad_size_replies_error(self, monkeypatch):
        import commands.generate_image_command as cmd
        monkeypatch.setattr(cmd, "image_gen_configured", lambda: True)
        inter, followups = self._interaction()
        await handle_generate_image_command(inter, prompt="bird", size="gigantic")
        assert "invalid size" in followups[0]["content"]


class TestHoldAndRefusalSemantics:
    """Review follow-ups: the hold is earned only by windows where the BOT
    switched the card, and a helper refusal is an error, not a silent success."""

    async def test_hold_not_earned_when_card_was_already_in_image_mode(self, monkeypatch):
        """Manual gpu_switch.sh window: the bot painted in a mode it did not
        switch — it must restore rag IMMEDIATELY (no hold, no stamp), so a
        manually-opened window never leaves the card image-side by hold logic."""
        monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/x")
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "IMAGE_MODE_HOLD_S", 300)
        monkeypatch.setattr(images, "_image_status",
                            AsyncMock(return_value={"loaded": True}))
        monkeypatch.setattr(images.httpx, "AsyncClient",
                            lambda **kw: _FakeClient([], [_resp_json({"mode": "image"})]))
        sw = AsyncMock()
        monkeypatch.setattr(images, "_switch_gpu_mode", sw)

        client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=_b64_png())]
        client.images.generate = AsyncMock(return_value=resp)
        monkeypatch.setattr(images, "_image_client", lambda: client)

        await images.generate_image("a bird")
        sw.assert_awaited_once_with("rag")      # immediate restore, not deferred
        assert images._hold_started_at == 0.0   # hold never armed
        assert images._last_bot_switch_at == 0.0  # no bot switch to earn a hold

    async def test_helper_refusal_surfaces_as_error(self, monkeypatch):
        """Helper 200-OK with detail "refused: ..." (Studio down) must NOT be
        treated as a successful switch — painting would hit a card without the
        diffusion slot."""
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images.httpx, "AsyncClient",
                            lambda **kw: _FakeClient([], [_resp_json(
                                {"ok": True, "mode": "rag",
                                 "detail": "refused: image backend unreachable"})]))
        with pytest.raises(images.ImageGenError, match="refused"):
            await images._switch_gpu_mode("image")


class TestHoldEventuallyExpires:
    """Regression: the re-entry deferral must ANCHOR to now, else every later
    painting window looks 're-entered' and the restore defers forever."""

    async def test_second_deferred_check_restores_when_no_new_painting(self, monkeypatch):
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "IMAGE_MODE_HOLD_S", 0.05)
        sw = AsyncMock()
        monkeypatch.setattr(images, "_switch_gpu_mode", sw)
        now = time.monotonic()
        images._hold_started_at = now - 0.1
        images._mode_switched_at = now          # painted during the hold -> defer
        await images.ensure_image_still_needed_or_restore()
        sw.assert_not_awaited()
        for t in list(images._background_tasks):
            t.cancel()
        images._background_tasks.clear()
        # New generation during the deferred window: paint, no new switch.
        images._mode_switched_at = time.monotonic()
        await images.ensure_image_still_needed_or_restore()   # defers again (anchored)
        sw.assert_not_awaited()
        for t in list(images._background_tasks):
            t.cancel()
        images._background_tasks.clear()
        # No further painting -> the third check MUST restore rag.
        await asyncio.sleep(0.06)
        await images.ensure_image_still_needed_or_restore()
        sw.assert_awaited_once_with("rag")
        assert images._hold_started_at == 0.0   # hold fully discharged


class TestStartupHeal:
    """main.py on_ready → heal_gpu_mode_on_start(): a card left image-side by
    a crash must not stay rag-degraded for the whole session."""

    async def test_rag_switch_requested_when_switcher_enabled(self, monkeypatch):
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "GPU_HEAL_ON_START", True)
        sw = AsyncMock()
        monkeypatch.setattr(images, "_switch_gpu_mode", sw)
        await images.heal_gpu_mode_on_start()
        sw.assert_awaited_once_with("rag")

    async def test_noop_without_switcher_or_when_disabled(self, monkeypatch):
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "")
        monkeypatch.setattr(images, "GPU_HEAL_ON_START", True)
        sw = AsyncMock()
        monkeypatch.setattr(images, "_switch_gpu_mode", sw)
        await images.heal_gpu_mode_on_start()
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "GPU_HEAL_ON_START", False)
        await images.heal_gpu_mode_on_start()
        sw.assert_not_awaited()

    async def test_failure_never_raises_out_of_startup(self, monkeypatch):
        monkeypatch.setattr(images, "GPU_SWITCH_URL", "http://gpu-switch:8080")
        monkeypatch.setattr(images, "GPU_HEAL_ON_START", True)
        monkeypatch.setattr(images, "_switch_gpu_mode",
                            AsyncMock(side_effect=images.ImageGenError("helper down")))
        await images.heal_gpu_mode_on_start()  # must not raise


def test_a_search_query_is_not_vault_evidence():
    """The model's own words cannot confirm a species (round 8, analysis §17).

    The gathered context stores each tool call as `[search_kb {"query": …}]` plus
    its hits. Feeding that whole string to the composer as "what the campaign says"
    let a live run retag Boss Frey as lizard-folk from a query string — the model
    asking "is he a lizard?" came back as evidence that he is.
    """
    gathered = ('[search_kb {"query": "Frey species beast-folk badger rat lizard", '
                '"scope": "lore"}]\n2 hit(s):\n'
                '[{"file": "HumbleWood/Session/SESSION_01.md", '
                '"preview": "eine Fuchsdame kommt heraus"}]')
    ev = images._vault_evidence(gathered)
    assert "lizard" not in ev
    assert "Fuchsdame" in ev          # the hits themselves stay
    assert "search_kb" not in ev


def test_the_party_sheet_is_never_evidence_about_everybody_else():
    """Round 12, request 4: the plan wrote the vault's own 'two rat-folk bandits'
    and was told that disagrees with the campaign — because the retrieved material
    contains the PARTY SHEET ('Birdfolk (blue tit)') and a bounty line naming a
    bandit called Luma. Our own species lines are not evidence about other beings,
    and the party sheet is text the bot injects for identity, not something it read
    about the bandits.

    This is provenance arithmetic (which file a hit came from), which is what the
    composer layer is for; judging which sentence 'is about' whom is not.
    """
    gathered = (
        '[search_kb {"query": "bandits tavern"}]\n'
        '8 hit(s): [{"file": "humblewood/HumbleWood/Characters/Player Characters.md", '
        '"snippet": "Colum Bidae — Birdfolk (blue tit), Warlock"}, {"file": '
        '"humblewood/HumbleWood/SessionLogs/SESSION_06/notes.md", "snippet": '
        '"two rat-folk bandits barred the road"}]\n\n'
        '[read_section {"file": "humblewood/HumbleWood/Characters/Player '
        'Characters.md", "section": "Colum Bidae"}]\nColum Bidae\n'
        'Birdfolk (blue tit), Warlock. Pronouns: he/him.\n\n'
        '[find_in_kb {"term": "Alisa Pennygleam"}]\n'
        'Alisa Pennygleam was rescued near Winnowing Reach.'
    )
    ev = images._vault_evidence(gathered)
    assert "two rat-folk bandits barred the road" in ev
    assert "Alisa Pennygleam was rescued" in ev
    assert "Birdfolk" not in ev, ev
    assert "Colum Bidae" not in ev, ev
    assert "find_in_kb" not in ev            # the agent's own words are not evidence


def test_an_agent_query_line_never_becomes_campaign_evidence_again(monkeypatch):
    """read_section and find_in_kb were missing from the echo pattern, so their
    call lines — the model's own wording — stayed in the evidence, where a guess the
    model asked about would read back to it as a statement the campaign makes."""
    gathered = ('[find_in_kb {"term": "Frey beast-folk badger lizard"}]\n'
                'Frey appears in SESSION_05.\n'
                '[read_section {"file": "a/b/notes.md", "section": "Lexikon"}]\n'
                'Boss Frey führt zwei rat-folk Banditen.')
    ev = images._vault_evidence(gathered)
    assert "badger" not in ev and "lizard" not in ev, ev
    assert "rat-folk" in ev


class TestDeferredLookup:
    """Live 2026-11-19: the second of three /image calls rewrote at 15:00:54,
    INSIDE the first one's painting window (opened 15:00:35 — "Retrieval GPU
    mode: rag → image (dense leg skipped while image)"), so its KB search would
    have run with the reranker and the embedder stopped.

    The rewrite's lookup pass now queues behind the generation window instead.
    The rag restore that ends the window brings the sidecars back, so the wait
    costs nothing the pipeline was not paying anyway — and the lookup gets real
    retrieval. A finding the model fixes from its own words never waits.
    """

    @staticmethod
    def _resp(content: str):
        return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])

    @staticmethod
    def _plan(scene: str, figures: list[dict]) -> str:
        return ('<image_plan>\n' + json.dumps(
            {"scene": scene, "figures": figures,
             "lighting": "dappled forest light"}) + '\n</image_plan>')

    @staticmethod
    async def _paint_window(events: list[str], hold_s: float) -> None:
        """Stand in for generate_image(): holds the lock, then restores rag."""
        async with images._gen_lock:
            events.append("painting")
            await asyncio.sleep(hold_s)
            events.append("window-closed")

    def _client(self, events: list[str], plans: list[str]):
        """A chat backend that records WHICH turn happens when."""
        n = {"i": 0}

        async def create(*a, **kw):
            n["i"] += 1
            events.append(f"model-turn-{n['i']}")
            return self._resp(plans[n["i"] - 1])

        client = MagicMock()
        client.chat.completions.create = create
        return client

    # Plan 1 carries a fixable finding the fix for is READING the vault: the
    # figure has no anchor and no look, and the plan's only description of him
    # was left in the scene sentence (identity_left_in_scene).
    _BOSS = ('Boss Frey, a heavy-set bandit chief in a patched coat, blocks the '
             'group on the Old Bridge',
             [{"name": "Boss Frey", "pose": "brandishing an axe"}])
    # Plan 2: the aside is gone out of the paintable sentence, the crowd noun too.
    _BOSS_FIXED = ('Boss Frey blocks the party on the torch-lit Old Bridge',
                   [{"name": "Boss Frey", "pose": "brandishing an axe"}])

    async def test_a_lookup_needed_mid_paint_waits_for_the_window(self, monkeypatch,
                                                                  caplog):
        """The lookup pass must not answer with lexical-only guesses."""
        import kb.retrievers as retrievers
        monkeypatch.setattr(retrievers, "_gpu_mode", "image")
        events: list[str] = []
        window = asyncio.create_task(self._paint_window(events, 0.08))
        await asyncio.sleep(0.02)                      # the window takes the lock
        assert images._gen_lock.locked()

        async def fake_switch(mode):                   # no switcher in tests
            monkeypatch.setattr(retrievers, "_gpu_mode", "rag")
            events.append(f"switch-{mode}")
        monkeypatch.setattr(images, "_switch_gpu_mode", fake_switch)
        monkeypatch.setattr(images, "_chat_client", lambda: self._client(
            events, [self._plan(*self._BOSS), self._plan(*self._BOSS_FIXED)]))
        phases: list[str] = []
        with caplog.at_level(logging.INFO, logger="bot.bot_core.images"):
            out = await images.prepare_prompt(
                "Boss Frey blockiert die Gruppe auf dem Weg", on_phase=phases.append)
        await window

        assert events.index("window-closed") < events.index("model-turn-2"), events
        assert "switch-rag" in events and (
            events.index("switch-rag") < events.index("model-turn-2")), events
        assert "the card is painting, so the lookup waits" in caplog.text
        assert "awaiting_lookup" in phases, "the pause was silent"
        assert "torch-lit" in out, out
        assert not images._gen_lock.locked()

    async def test_a_finding_the_model_fixes_itself_never_waits(self, monkeypatch,
                                                                caplog):
        """A collective subject needs no retrieval: making it queue behind a
        painting window would only make the user wait for nothing."""
        import kb.retrievers as retrievers
        monkeypatch.setattr(retrievers, "_gpu_mode", "image")
        switched: list[str] = []

        async def fake_switch(mode):
            switched.append(mode)
        monkeypatch.setattr(images, "_switch_gpu_mode", fake_switch)
        events: list[str] = ["painting"]

        async def hold_forever():
            async with images._gen_lock:
                await asyncio.sleep(5)
        window = asyncio.create_task(hold_forever())
        await asyncio.sleep(0.02)
        monkeypatch.setattr(images, "_chat_client", lambda: self._client(
            events, [self._plan("die ganze Gruppe chillt am Lagerfeuer",
                                [{"name": "Seta Onix", "pose": "nudging a boot"},
                                 {"name": "Marvin", "pose": "rolling forward"}]),
                     self._plan("Seta Onix and Marvin chill at the campfire",
                                [{"name": "Seta Onix", "pose": "nudging a boot"},
                                 {"name": "Marvin", "pose": "rolling forward"}])]))
        with caplog.at_level(logging.INFO, logger="bot.bot_core.images"):
            out = await asyncio.wait_for(
                images.rewrite_prompt("die ganze Gruppe chillt am Lagerfeuer"),
                timeout=10)
        assert images._gen_lock.locked(), "the rewrite took the card it did not need"
        window.cancel()
        await asyncio.gather(window, return_exceptions=True)

        assert "collective_subject" in caplog.text
        assert events[-1] == "model-turn-2", events
        assert switched == [], switched
        assert "the card is painting, so the lookup waits" not in caplog.text
        assert "Exactly 2 named figures" in out, out

    async def test_a_window_that_outlasts_the_wait_runs_the_lookup_degraded(
            self, monkeypatch, caplog):
        """Waiting forever is worse than a lexical-only lookup: the cap releases
        the request, and the late lock acquisition is handed back."""
        import kb.retrievers as retrievers
        monkeypatch.setattr(retrievers, "_gpu_mode", "image")
        monkeypatch.setattr(images, "IMAGE_REWRITE_LOOKUP_WAIT_S", 0.05)
        events: list[str] = []
        window = asyncio.create_task(self._paint_window(events, 0.5))
        await asyncio.sleep(0.02)
        async def fake_switch(mode):
            events.append(f"switch-{mode}")
        monkeypatch.setattr(images, "_switch_gpu_mode", fake_switch)
        monkeypatch.setattr(images, "_chat_client", lambda: self._client(
            events, [self._plan(*self._BOSS), self._plan(*self._BOSS_FIXED)]))
        with caplog.at_level(logging.WARNING, logger="bot.bot_core.images"):
            out = await images.rewrite_prompt("Boss Frey blockiert die Gruppe")
        assert "IMAGE_REWRITE_LOOKUP_WAIT_S" in caplog.text
        assert "model-turn-2" in events, events          # ran without waiting
        assert "torch-lit" in out, out
        await window
        assert events.index("model-turn-2") < events.index("window-closed"), events
        assert "switch-rag" not in events, events        # the card was left alone
        await asyncio.sleep(0.05)
        assert not images._gen_lock.locked(), "the late acquisition stayed held"

    async def test_an_image_mode_the_bot_did_not_open_is_not_touched(self, monkeypatch,
                                                                     caplog):
        """A manual `gpu_switch.sh image` window is someone else's card state:
        the lookup runs lexical-only rather than unloading their model."""
        import kb.retrievers as retrievers
        monkeypatch.setattr(retrievers, "_gpu_mode", "image")

        async def forbidden_switch(mode):
            raise AssertionError(f"the rewrite switched the card to {mode}")
        monkeypatch.setattr(images, "_switch_gpu_mode", forbidden_switch)
        assert not images._gen_lock.locked()
        monkeypatch.setattr(images, "_chat_client", lambda: self._client(
            [], [self._plan(*self._BOSS), self._plan(*self._BOSS_FIXED)]))
        with caplog.at_level(logging.WARNING, logger="bot.bot_core.images"):
            out = await images.rewrite_prompt("Boss Frey blockiert die Gruppe")
        assert "OUTSIDE a generation window" in caplog.text
        assert "torch-lit" in out, out

    async def test_two_back_to_back_rewrites_queue_behind_the_same_window(
            self, monkeypatch, caplog):
        """The live shape: three /image calls in a row. The second rewrites while
        the first paints, its lookup waits, rag is restored, and the lock comes
        back exactly once — no leak, no double-hold."""
        import kb.retrievers as retrievers
        monkeypatch.setattr(retrievers, "_gpu_mode", "image")
        events: list[str] = []
        window = asyncio.create_task(self._paint_window(events, 0.12))
        await asyncio.sleep(0.02)

        async def fake_switch(mode):
            events.append(f"switch-{mode}")
            monkeypatch.setattr(retrievers, "_gpu_mode", "rag")
        monkeypatch.setattr(images, "_switch_gpu_mode", fake_switch)
        monkeypatch.setattr(images, "_chat_client", lambda: self._client(
            events, [self._plan(*self._BOSS), self._plan(*self._BOSS_FIXED),
                     self._plan(*self._BOSS), self._plan(*self._BOSS_FIXED)]))
        with caplog.at_level(logging.INFO, logger="bot.bot_core.images"):
            outs = await asyncio.gather(
                images.rewrite_prompt("Boss Frey blockiert die Gruppe auf der Brücke"),
                images.rewrite_prompt("Boss Frey blockiert die zweite Gruppe"))
        await window
        turns = sorted(e for e in events
                       if e in ("model-turn-2", "model-turn-4"))
        assert len(turns) == 2, events
        assert all(events.index(t) > events.index("window-closed")
                   for t in turns), events
        assert all("torch-lit" in o for o in outs), outs
        assert not images._gen_lock.locked(), "a lookup window leaked the lock"

    async def test_a_lookup_that_needs_nothing_never_opens_the_window(self, monkeypatch,
                                                                      caplog):
        """rag mode live → the correction pass runs exactly as before, no wait."""
        import kb.retrievers as retrievers
        monkeypatch.setattr(retrievers, "_gpu_mode", "rag")
        events: list[str] = []
        monkeypatch.setattr(images, "_chat_client", lambda: self._client(
            events, [self._plan(*self._BOSS), self._plan(*self._BOSS_FIXED)]))
        with caplog.at_level(logging.INFO, logger="bot.bot_core.images"):
            out = await images.rewrite_prompt("Boss Frey blockiert die Gruppe")
        assert "the card is painting, so the lookup waits" not in caplog.text
        assert not images._gen_lock.locked()
        assert "torch-lit" in out, out


class TestRewriteLoopCaps:
    """§21: an image rewrite is not a chat RAG turn, so it does not run on the
    chat loop's caps.

    Live probe 3 (2026-11-20): the correction loop ended in
    `agent fallback (decision turn timed out)` with a 45 s budget while the only
    slow thing was the model composing an answer — the measured tool round trips
    were 4.9-6.5 s in rag mode, ~1 s in image mode. `RAG_AGENT_BUDGET_SECONDS`
    was tuned for decision turns that only PICK tools; the rewrite's last turn
    WRITES a plan at IMAGE_REASONING_EFFORT (~52 s median at medium, §12). The
    image paths now state their own budget, and all three of them say it.
    """

    @staticmethod
    def _plan(scene: str, figures: list[dict]) -> str:
        return ('<image_plan>\n' + json.dumps(
            {"scene": scene, "figures": figures,
             "lighting": "dappled forest light"}) + '\n</image_plan>')

    def _spy(self, monkeypatch, replies: list[str]):
        """Stand in for the agent loop, recording the caps each caller asked for."""
        import bot_core.agent_loop as agent_loop
        seen: list[dict] = []

        async def fake_loop(**kwargs):
            seen.append(kwargs)
            meta = agent_loop.AgentTurnMeta()
            meta.tool_calls_detail = []
            # A caller may loop more times than there are scripted answers (the
            # plan path retries once, a fixable finding opens the correction
            # pass) — the LAST scripted answer repeats, because this test is
            # about the caps each call states, not about how many calls happen.
            return replies[min(len(seen) - 1, len(replies) - 1)], meta

        monkeypatch.setattr(images, "run_agent_loop", fake_loop)
        # The client is built before the loop runs; nothing here may reach for
        # real credentials.
        monkeypatch.setattr(images, "_chat_client", lambda: MagicMock())
        return seen

    async def test_the_plan_path_states_its_own_wall_clock(self, monkeypatch):
        seen = self._spy(monkeypatch, [self._plan(
            "Romi stirs a pot by the fire", [{"name": "Romi", "pose": "stirring"}])])
        out = await images.rewrite_prompt("Romi stirring a pot")
        assert "Romi" in out
        assert seen, "the plan path ran no agent loop"
        # (This plan earns a composer finding, so the correction pass runs too —
        # the point of the test is that EVERY image-rewrite loop states the cap.)
        assert all(k.get("budget_seconds") == images.IMAGE_REWRITE_AGENT_BUDGET_SECONDS
                   for k in seen), [k.get("budget_seconds") for k in seen]

    async def test_the_lookup_correction_pass_states_it_too(self, monkeypatch):
        """The loop that waits for the card also gets the fresh budget — the wait
        is NOT paid out of it (it has its own cap, IMAGE_REWRITE_LOOKUP_WAIT_S)."""
        seen = self._spy(monkeypatch, [
            self._plan("Boss Frey, a scarred bandit chief, blocks the Old Bridge",
                       [{"name": "Boss Frey", "pose": "brandishing an axe"}]),
            self._plan("Boss Frey blocks the party on the torch-lit Old Bridge",
                       [{"name": "Boss Frey", "pose": "brandishing an axe"}]),
        ])
        out = await images.rewrite_prompt("Boss Frey blocks the Old Bridge")
        assert "Boss Frey" in out
        assert len(seen) == 2, "the correction pass did not run"
        assert all(k["budget_seconds"] == images.IMAGE_REWRITE_AGENT_BUDGET_SECONDS
                   for k in seen)

    async def test_the_prose_fallback_states_it_too(self, monkeypatch):
        seen = self._spy(monkeypatch, [
            "not a plan at all",                       # plan path yields nothing
            # The prose fallback's output must look like a finished prompt, which
            # means it opens with the style sentence the fallback was loaded with.
            images._opening_style() + " A hedgehog cleric kneels by a moss-covered"
            " shrine, warm lantern light, no text in the frame.",
        ])

        async def no_plan(*a, **kw):
            raise images.ImageGenError("no plan")

        monkeypatch.setattr(images, "_rewrite_composed", no_plan)
        out = await images.rewrite_prompt("a cleric kneeling by a shrine")
        assert "shrine" in out
        assert seen, "the prose fallback ran no agent loop"
        assert seen[-1]["budget_seconds"] == (
            images.IMAGE_REWRITE_AGENT_BUDGET_SECONDS)

    def test_the_image_budget_is_not_the_chat_budget(self):
        """One number cannot serve both: the chat loop's cap is a decision-turn
        budget, the rewrite's is a decision budget PLUS a reasoning-model answer
        turn. Equal values would mean the knob is decorative."""
        from config.settings import IMAGE_REWRITE_AGENT_BUDGET_SECONDS, RAG_AGENT_BUDGET_SECONDS
        assert IMAGE_REWRITE_AGENT_BUDGET_SECONDS > RAG_AGENT_BUDGET_SECONDS

    def test_the_lookup_wait_cap_is_a_persons_patience_not_a_card_timeout(self):
        """900 s was the first draft and it was wrong: the wait is paid by someone
        watching one Discord message, and the measured wait for a real
        back-to-back pair was 3 s. It must stay under a generation window's worth
        of patience (IMAGE_GEN_TIMEOUT_S is the painting deadline)."""
        from config.settings import IMAGE_GEN_TIMEOUT_S, IMAGE_REWRITE_LOOKUP_WAIT_S
        assert IMAGE_REWRITE_LOOKUP_WAIT_S <= 300 < IMAGE_GEN_TIMEOUT_S



class TestCorrectionKeepsTheFatterPlan:
    """A correction that fixes nothing may not make a figure thinner (§22.1).

    Live 2026-11-24, "Die Gruppe wird von Banditen überfallen, die die Old Bridge
    kontrollieren": the first composition painted Boss Frey in 215 characters; the
    correction pass answered with a plan whose description of him the composer could
    not verify, his block collapsed to 98 characters against a 447-character median,
    and the FIXABLE FINDING COUNT WAS IDENTICAL (2 → 2) — so the comparison that
    existed called the two plans equal and painted the worse one. The image came back
    with only the player characters in it, which is the thin-block shape §22.1 measures
    as the figure the renderer leaves out.
    """

    _SHEET = """# Colum Bidae
Birdfolk (blue tit), Warlock.

## Visual Anchor (group images — reuse VERBATIM for named appearances)

> species: Birdfolk (blue tit)
> body: small and plump, short clipped tail
> outfit: moss-green cloak, small round spectacles
> signature: golden rose carried in both claws

# Romi Stachelpfote
Hedge (hedgehog), Monk.

## Visual Anchor (group images — reuse VERBATIM for named appearances)

> species: Hedge (hedgehog)
> body: taupe quills, low wide stance
> outfit: ochre monk's robe
> quirks: thick padded fists held ready
"""

    _OPENED = "humblewood/SessionLogs/SESSION_05/notes.md"

    def _anchors(self, monkeypatch, tmp_path):
        sheet = tmp_path / "Player Characters.md"
        sheet.write_text(self._SHEET, encoding="utf-8")
        monkeypatch.setattr(images, "_party_sheet_file", lambda: sheet)
        return images._load_anchors()

    def _meta(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            tool_calls=[{"tool": "kb_read_section"}], iterations=1,
            answer_truncated=False, degraded=False,
            tool_calls_detail=[{"tool": "kb_read_section",
                                "args": {"path": self._OPENED}}],
            gathered_context="Boss Frey: a bandit chief of the Old Bridge toll crew.")

    def _compose(self, anchors, look):
        from bot_core import image_prompt
        plan = image_prompt.extract_plan("<image_plan>\n" + json.dumps({
            "scene": "Boss Frey blocks the torch-lit Old Bridge opposite Colum "
                     "Bidae and Romi Stachelpfote",
            "figures": [
                {"name": "Boss Frey", "pose": "blocking the bridge with an axe",
                 "look": look, "source": "SESSION_05/notes.md"},
                {"name": "Colum Bidae", "pose": "perched on the railing"},
                {"name": "Romi Stachelpfote", "pose": "crouched in front"},
            ],
            "lighting": "torch light", "palette": "amber", "mood": "tense"})
            + "\n</image_plan>")
        profile = prompt_profile.load()
        composed = image_prompt.compose_prompt(
            plan, anchors, "cinematic fantasy illustration",
            foreground=profile.foreground_figures,
            max_clauses_per_figure=profile.clauses_per_figure,
            budget_kwargs=images._budget_kwargs(profile),
            world_rule=profile.world_rule, read_paths=[self._OPENED],
            retrieved="", request_text="boss frey blocks the old bridge")
        return plan, composed

    def _answer(self, look):
        data = {
            "scene": "Boss Frey blocks the torch-lit Old Bridge opposite Colum "
                     "Bidae and Romi Stachelpfote",
            "figures": [
                {"name": "Boss Frey", "pose": "blocking the bridge with an axe",
                 "look": look, "source": "SESSION_05/notes.md"},
                {"name": "Colum Bidae", "pose": "perched on the railing"},
                {"name": "Romi Stachelpfote", "pose": "crouched in front"}]}
        return "<image_plan>\n" + json.dumps(data) + "\n</image_plan>"

    async def _correct(self, monkeypatch, tmp_path, composed, plan, revised_text):
        async def fake_loop(**kw):
            return revised_text, self._meta()
        monkeypatch.setattr(images, "run_agent_loop", fake_loop)
        return await images._correction_pass(
            complete=AsyncMock(), system="sys", msg="user",
            out="<image_plan>original</image_plan>", ask=[], worth=[],
            meta=self._meta(), composed=composed, plan=plan,
            anchors=self._anchors(monkeypatch, tmp_path),
            profile=prompt_profile.load(), text="boss frey blocks the bridge")

    async def test_a_correction_that_only_thins_a_figure_is_not_the_one_painted(
            self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(images, "IMAGE_CODE_CORRECTIONS", True)   # the arbitration
        anchors = self._anchors(monkeypatch, tmp_path)
        plan, composed = self._compose(anchors, "a bandit chief in a coat")
        codes = [f.code for f in composed.findings]
        assert codes == ["figure_thin_line"], codes      # the setup, not the claim
        # The correction answers with the same characters and the same one open
        # finding — it just says LESS about the one figure that was already thin.
        caplog.set_level(logging.WARNING)
        out, _ = await self._correct(monkeypatch, tmp_path, composed, plan,
                                     self._answer("a bandit"))
        assert "bandit chief in a coat" in out.prompt, "the first plan's figure survived"
        assert any("thinned the smallest figure block" in m for m in caplog.messages), \
            caplog.messages

    async def test_a_correction_that_actually_thickens_the_thin_figure_wins(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(images, "IMAGE_CODE_CORRECTIONS", True)   # the arbitration
        anchors = self._anchors(monkeypatch, tmp_path)
        plan, composed = self._compose(anchors, "a bandit chief in a coat")
        out, _ = await self._correct(
            monkeypatch, tmp_path, composed, plan,
            self._answer("a heavy-set bandit chief in a patched coat with an iron "
                         "hook hand, a battered buckler and a crooked silver crown"))
        assert "crooked silver crown" in out.prompt

    async def test_enforce_off_paints_the_refinement_answer_with_no_arbitration(
            self, monkeypatch, tmp_path, caplog):
        """With the corrections off there is no plan-vs-plan comparison (analysis §24–§25).

        Same setup as the tie test above: the refinement answer says LESS about the thin
        figure. In the correcting configuration the first plan wins; with the corrections off
        — the shipped configuration since 2026-11-26 — the answer is painted exactly as
        answered, which is what the switch exists to be able to measure.
        """
        anchors = self._anchors(monkeypatch, tmp_path)
        plan, composed = self._compose(anchors, "a bandit chief in a coat")
        monkeypatch.setattr(images, "IMAGE_CODE_CORRECTIONS", False)
        caplog.set_level(logging.INFO)
        out, _ = await self._correct(monkeypatch, tmp_path, composed, plan,
                                     self._answer("a bandit"))
        assert "bandit chief in a coat" not in out.prompt
        assert any("off — painting the refinement turn's plan" in m
                   for m in caplog.messages), caplog.messages

    def test_the_enforce_switch_defaults_to_the_shipped_behaviour(self):
        from config.settings import IMAGE_CODE_CORRECTIONS
        assert IMAGE_CODE_CORRECTIONS is False, \
            "corrections off is the shipped configuration after the 20-image A/B (§25)"

