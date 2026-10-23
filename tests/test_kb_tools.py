"""Tests for kb.tools — the agent-mode tool layer (#39, research doc §3.2).

Covers the JSON protocol parser (fenced / bare / malformed), schema
validation, scope routing, and the three tools against a fake index store.
"""
import json

import pytest

from kb import tools
from kb.tools import ToolCallParseError, parse_tool_call


# ── Parsing ────────────────────────────────────────────────────────────────

class TestParseToolCall:
    def test_fenced_json(self):
        text = ('Looking it up.\n```json\n'
                '{"tool": "search_kb", "args": {"query": "plate armor AC"}}\n'
                '```')
        name, args = parse_tool_call(text)
        assert name == "search_kb"
        assert args == {"query": "plate armor AC"}

    def test_bare_json_in_prose(self):
        text = ('I will search now: {"tool": "read_section", '
                '"args": {"file": "equipment.md", "section": "Armor"}, '
                '"notes": "extra args are ignored"} done')
        name, args = parse_tool_call(text)
        assert name == "read_section"
        assert args == {"file": "equipment.md", "section": "Armor"}

    def test_unknown_tool_raises_readable(self):
        with pytest.raises(ToolCallParseError, match="unknown or missing tool"):
            parse_tool_call('{"tool": "fly_away", "args": {}}')

    def test_missing_required_arg(self):
        with pytest.raises(ToolCallParseError, match="missing required arg 'query'"):
            parse_tool_call('{"tool": "search_kb", "args": {"scope": "rules"}}')

    def test_wrong_type(self):
        with pytest.raises(ToolCallParseError, match="must be a string"):
            parse_tool_call('{"tool": "search_kb", "args": {"query": 42}}')

    def test_no_json_at_all(self):
        with pytest.raises(ToolCallParseError, match="no JSON object found"):
            parse_tool_call("I think the answer is forty-two.")

    def test_list_sections_needs_file_or_dir_at_execution(self):
        name, args = parse_tool_call('{"tool": "list_sections", "args": {}}')
        assert name == "list_sections" and args == {}


# ── Scope routing + prompt block ───────────────────────────────────────────

class TestScopesAndPrompt:
    def test_scope_filter_rules(self):
        keep = tools._scope_filter("rules")
        assert keep("humblewood/DnDRules/DnD5_5/equipment.md [Armor]")
        assert not keep("humblewood/HumbleWood/quests.md [Quests]")

    def test_scope_filter_lore(self):
        keep = tools._scope_filter("lore")
        assert keep("humblewood/HumbleWood/quests.md [Quests]")
        assert not keep("humblewood/DnDRules/DnD5_5/equipment.md [Armor]")

    def test_scope_all_no_filter(self):
        assert tools._scope_filter("all") is None
        assert tools._scope_filter(None) is None

    def test_bad_scope_raises(self):
        with pytest.raises(ToolCallParseError, match="scope must be one of"):
            tools._scope_filter("everywhere")

    def test_prompt_block_names_all_tools_and_limits(self):
        block = tools.tool_prompt_block()
        for t in ("search_kb", "read_section", "list_sections"):
            assert t in block
        assert '"tool"' in block


# ── Execution against a fake index store ───────────────────────────────────

class _FakeIdx:
    def __init__(self, chunks, nonempty=False):
        self._chunks = chunks  # [(display_name, content)]
        self._nonempty = nonempty

    def is_empty(self):
        return not (self._chunks or self._nonempty)

    async def query_with_embeddings(self, text, top_n=5):
        # Deterministic fake "dense" ranking: name contains the query head.
        key = text.split()[0].lower() if text else ""
        ranked = sorted(
            ((n, c, 0.9 if key in n.lower() else 0.3) for n, c in self._chunks),
            key=lambda t: t[2], reverse=True)
        return ranked[:top_n], [0.1] * 8

    def iter_chunks(self):
        return iter(self._chunks)


class _FakeStore:
    def __init__(self, idx):
        self._idx = idx

    def get_index(self):
        return self._idx


@pytest.fixture
def fake_store(monkeypatch):
    chunks = [
        ("humblewood/DnDRules/DnD5_5/equipment.md [Equipment -> Armor]",
         "Plate Armor: AC 18, 15 lb., 75 gp. Full entry text here."),
        ("humblewood/DnDRules/DnD5_5/equipment.md [Equipment -> Weapons]",
         "Longsword: 1d8 slashing, 1d10 two-handed. 15 sp."),
        ("humblewood/HumbleWood/HumbleWood/Humblewood_Backgrounds.md [Full Document]",
         "Backgrounds of Humblewood: Soldier, Sage, Entertainer."),
    ]
    store = _FakeStore(_FakeIdx(chunks))

    async def _fake_ensure(kb_path):
        return store

    import kb.retrievers as R
    monkeypatch.setattr(R, "_ensure_index_store", _fake_ensure)
    # No reranker in unit tests — dense order only.
    import kb.reranker as RK
    monkeypatch.setattr(RK, "is_available", lambda: False)
    return store


@pytest.mark.asyncio
async def test_search_kb_returns_snippets(fake_store, monkeypatch):
    monkeypatch.setattr("config.settings.RAG_DENSE_TOP_K", 10)
    out = await tools.execute_tool(
        "search_kb", {"query": "plate armor AC", "scope": "rules"})
    assert "hit(s)" in out
    hits = json.loads(out.split("\n", 1)[1])
    assert all(h["file"].endswith("equipment.md") for h in hits)
    assert "Plate Armor: AC 18" in hits[0]["snippet"]
    # Lore entry must be scope-filtered out.
    assert not any("Backgrounds" in h["file"] for h in hits)


@pytest.mark.asyncio
async def test_search_kb_unrelated_query_still_returns_candidates(fake_store, monkeypatch):
    """No hard score threshold: the model judges relevance from snippets.
    An unrelated query still surfaces the nearest candidates (dense order)."""
    monkeypatch.setattr("config.settings.RAG_DENSE_TOP_K", 10)
    out = await tools.execute_tool(
        "search_kb", {"query": "quantum flux capacitor", "scope": "rules"})
    assert "hit(s)" in out


@pytest.mark.asyncio
async def test_search_kb_empty_index_reports_no_hits(fake_store, monkeypatch):
    """An empty result set must report 'No hits', not crash."""
    async def _empty_ensure(kb_path):
        # nonempty=True: the index itself is fine, it just has no matching
        # chunks — the 'No hits' path (as opposed to 'unavailable').
        return _FakeStore(_FakeIdx([], nonempty=True))

    import kb.retrievers as R
    monkeypatch.setattr(R, "_ensure_index_store", _empty_ensure)
    out = await tools.execute_tool(
        "search_kb", {"query": "anything", "scope": "rules"})
    assert out.startswith("No hits")


@pytest.mark.asyncio
async def test_read_section_full_text(fake_store):
    out = await tools.execute_tool(
        "read_section",
        {"file": "equipment.md", "section": "Armor"})
    assert "Full entry text here." in out   # the FULL chunk, not a snippet
    assert "Longsword" not in out           # other section excluded


@pytest.mark.asyncio
async def test_read_section_missing(fake_store):
    out = await tools.execute_tool("read_section", {"file": "nope.md"})
    assert out.startswith("No chunk found")


@pytest.mark.asyncio
async def test_list_sections_file(fake_store):
    out = await tools.execute_tool(
        "list_sections", {"file": "equipment.md"})
    assert "Equipment -> Armor" in out
    assert "Equipment -> Weapons" in out
    assert "Backgrounds" not in out


@pytest.mark.asyncio
async def test_list_sections_dir(fake_store):
    out = await tools.execute_tool(
        "list_sections", {"dir": "HumbleWood"})
    assert "Humblewood_Backgrounds.md" in out
    assert "equipment.md" not in out


@pytest.mark.asyncio
async def test_list_sections_needs_arg(fake_store):
    with pytest.raises(ToolCallParseError, match="give 'file' or 'dir'"):
        await tools.execute_tool("list_sections", {})
