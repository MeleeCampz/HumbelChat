"""Tests for the /help command (backlog #17).

Covers: dynamic overview built from the live tree, category grouping,
unknown-command fallback ("Other"), None descriptions, the per-command detail
view (options with type/required/default), unknown detail targets, and that
the reply is public (not ephemeral).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from discord.utils import MISSING


def _cmd(name: str, description: str | None = "desc", params=None) -> SimpleNamespace:
    """A stand-in for discord.app_commands.Command (only the attrs we read)."""
    return SimpleNamespace(name=name, description=description, parameters=params or [])


def _param(name: str, *, type=int, description: str | None = None,
           default=MISSING, required=None, min_value=None, max_value=None) -> SimpleNamespace:
    if required is None:
        required = default is MISSING
    return SimpleNamespace(
        name=name, type=type, description=description,
        default=default, required=required,
        min_value=min_value, max_value=max_value,
    )


def _make_ix(commands: list[SimpleNamespace]) -> tuple[MagicMock, list]:
    """Build a fake interaction; returns (ix, sent) where sent collects calls."""
    sent: list[tuple[dict, object]] = []

    def capture(content=None, **kwargs):
        embed = kwargs.get("embed")
        sent.append(({"ephemeral": kwargs.get("ephemeral", False)}, content if embed is None else embed))
        return MagicMock()

    response_mock = MagicMock()
    response_mock.send_message = AsyncMock(side_effect=capture)

    ix = MagicMock()
    ix.response = response_mock
    ix.client.tree.get_commands = lambda: list(commands)
    return ix, sent


SAMPLE_COMMANDS = [
    _cmd("ai", "Send a prompt to the AI and get a reply."),
    _cmd("clear_history", "Clear conversation history for this channel."),
    _cmd("roll_items", "Roll random items from an item table.",
         params=[_param("count", default=3, description="How many items to roll",
                        min_value=1, max_value=25)]),
    _cmd("upload_kb", None),  # no description
    _cmd("brand_new_command", "A command added after the category map was written."),
]


class TestHelpOverview:

    @pytest.mark.asyncio
    async def test_all_commands_listed_with_descriptions(self):
        from commands.help_command import handle_help_command

        ix, sent = _make_ix(SAMPLE_COMMANDS)
        await handle_help_command(ix)

        assert len(sent) == 1
        embed = sent[0][1]
        text = "\n".join(f.name + "\n" + f.value for f in embed.fields)
        for c in SAMPLE_COMMANDS:
            assert f"`/{c.name}`" in text, f"{c.name} missing from overview"
        assert "Send a prompt to the AI" in text

    @pytest.mark.asyncio
    async def test_category_grouping_and_order(self):
        from commands.help_command import handle_help_command

        ix, sent = _make_ix(SAMPLE_COMMANDS)
        await handle_help_command(ix)

        embed = sent[0][1]
        field_names = [f.name for f in embed.fields]
        assert "AI & Chat" in field_names
        assert "D&D Items" in field_names
        assert "Knowledge Base" in field_names
        # fixed display order: AI & Chat before D&D Items before Knowledge Base
        assert field_names.index("AI & Chat") < field_names.index("D&D Items") < field_names.index("Knowledge Base")
        # empty categories are omitted
        assert "Sessions" not in field_names
        assert "Voice Recording" not in field_names

    @pytest.mark.asyncio
    async def test_unknown_command_falls_into_other(self):
        from commands.help_command import handle_help_command

        ix, sent = _make_ix(SAMPLE_COMMANDS)
        await handle_help_command(ix)

        embed = sent[0][1]
        other = next(f for f in embed.fields if f.name == "Other")
        assert "`/brand_new_command`" in other.value
        # it must NOT leak into a known category
        ai_field = next(f for f in embed.fields if f.name == "AI & Chat")
        assert "brand_new_command" not in ai_field.value

    @pytest.mark.asyncio
    async def test_none_description_does_not_crash(self):
        from commands.help_command import handle_help_command

        ix, sent = _make_ix(SAMPLE_COMMANDS)
        await handle_help_command(ix)  # upload_kb has description=None

        embed = sent[0][1]
        text = "\n".join(f.value for f in embed.fields)
        assert "`/upload_kb`" in text  # listed, name only

    @pytest.mark.asyncio
    async def test_empty_tree_still_replies(self):
        from commands.help_command import handle_help_command

        ix, sent = _make_ix([])
        await handle_help_command(ix)

        assert len(sent) == 1
        assert isinstance(sent[0][1], object) and sent[0][1].title == "Available Commands"

    @pytest.mark.asyncio
    async def test_reply_is_public(self):
        from commands.help_command import handle_help_command

        ix, sent = _make_ix(SAMPLE_COMMANDS)
        await handle_help_command(ix)

        assert sent[0][0]["ephemeral"] is False


class TestHelpDetail:

    @pytest.mark.asyncio
    async def test_detail_shows_description_and_options(self):
        from commands.help_command import handle_help_command

        ix, sent = _make_ix(SAMPLE_COMMANDS)
        await handle_help_command(ix, "roll_items")

        embed = sent[0][1]
        assert embed.title == "/roll_items"
        assert "Roll random items" in (embed.description or "")
        options = next(f for f in embed.fields if f.name == "Options").value
        assert "**count**" in options
        assert "(int (1–25))" in options          # type + range bounds
        assert "How many items to roll" in options
        assert "default: 3" in options

    @pytest.mark.asyncio
    async def test_detail_required_option(self):
        from commands.help_command import handle_help_command

        cmds = [_cmd("upload_kb", "Upload a file.",
                     params=[_param("file", type=str, description="The file to upload")])]
        ix, sent = _make_ix(cmds)
        await handle_help_command(ix, "upload_kb")

        options = next(f for f in sent[0][1].fields if f.name == "Options").value
        assert "*required*" in options
        assert "default:" not in options

    @pytest.mark.asyncio
    async def test_detail_no_parameters(self):
        from commands.help_command import handle_help_command

        ix, sent = _make_ix(SAMPLE_COMMANDS)
        await handle_help_command(ix, "clear_history")

        embed = sent[0][1]
        assert embed.title == "/clear_history"
        options = next(f for f in embed.fields if f.name == "Options").value
        assert options == "No options."

    @pytest.mark.asyncio
    async def test_detail_unknown_command_is_friendly(self):
        from commands.help_command import handle_help_command

        ix, sent = _make_ix(SAMPLE_COMMANDS)
        await handle_help_command(ix, "nope")

        assert len(sent) == 1
        kwargs, content = sent[0]
        assert isinstance(content, str)
        assert "`/nope`" in content
        assert "not a known command" in content
        assert "/help" in content

    @pytest.mark.asyncio
    async def test_detail_reply_is_public(self):
        from commands.help_command import handle_help_command

        ix, sent = _make_ix(SAMPLE_COMMANDS)
        await handle_help_command(ix, "ai")

        assert sent[0][0]["ephemeral"] is False
