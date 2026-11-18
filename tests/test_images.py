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
import time

import httpx
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
        assert out.startswith(images._BASE_STYLE)
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

    async def test_the_routing_turn_still_runs_with_thinking_off(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._resp("ok"))
        complete = images._make_completer(client)
        await complete([{"role": "user", "content": "pick tools"}], "m", thinking=False)
        assert client.chat.completions.create.call_args.kwargs["extra_body"] == \
            {"reasoning_effort": "none", "enable_thinking": False}

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

    def test_base_style_constant_matches_the_policy_file(self):
        """The guard and the policy must not drift apart (the policy is the
        source of truth the model is told to follow VERBATIM)."""
        policy = images._POLICY_FILE.read_text(encoding="utf-8")
        assert images._BASE_STYLE in policy

    def test_real_prompts_pass(self):
        base = images._BASE_STYLE
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
            self._resp(images._BASE_STYLE + ". Colum and Seta face a bandit; Colum "
                       "lunges, Seta nocks an arrow."),
        ])
        monkeypatch.setattr(images, "_chat_client", lambda: client)
        complete = images._make_completer(client)
        out = await images._rewrite_prose("die ganze Gruppe", complete)
        assert out.startswith(images._BASE_STYLE)
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
    """Read the monkeypatched sheet and parse it (test helper)."""
    return images._parse_party_sheet(images._PARTY_SHEET_FILE.read_text(encoding="utf-8"))


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
        monkeypatch.setattr(images, "_PARTY_SHEET_FILE", sheet)
        return sheet

    def test_parses_names_species_line_and_anchor(self, monkeypatch, tmp_path):
        self._point_at(monkeypatch, tmp_path)
        entries = images._parse_party_sheet(images._PARTY_SHEET_FILE.read_text(encoding="utf-8"))
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
        monkeypatch.setattr(images, "_PARTY_SHEET_FILE", sheet)
        head = images._load_party_dossier()
        for needle in ("MUST be looked up in the",
                       "knowledge base BEFORE you write it",
                       "Lexikon",
                       "NEVER assign a species or a",
                       "ONE array of tool calls in a SINGLE decision turn"):
            assert needle in head, needle

    def test_missing_sheet_degrades_to_a_lookup_instruction(self, monkeypatch, tmp_path):
        monkeypatch.setattr(images, "_PARTY_SHEET_FILE", tmp_path / "absent.md")
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
        assert "Googly eyes are NOT a species trait" in s  # built-in default

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
