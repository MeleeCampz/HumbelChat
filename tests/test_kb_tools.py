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
        with pytest.raises(ToolCallParseError, match="no valid tool call"):
            parse_tool_call("I think the answer is forty-two.")


class TestParseToolCalls:
    """The multi-call (batch) protocol — arrays run in parallel."""

    def test_single_object_returns_one_call(self):
        calls = tools.parse_tool_calls(
            '{"tool": "list_sessions", "args": {}}')
        assert calls == [("list_sessions", {})]

    def test_array_of_two_validates_both(self):
        text = ('[{"tool": "find_in_kb", "args": {"term": "Alderheart"}},'
                '{"tool": "list_sessions", "args": {}}]')
        calls = tools.parse_tool_calls(text)
        assert calls == [("find_in_kb", {"term": "Alderheart"}),
                        ("list_sessions", {})]

    def test_fenced_array(self):
        text = ('```json\n'
                '[{"tool": "read_section", "args": {"file": "a.md"}},'
                '{"tool": "read_section", "args": {"file": "b.md"}}]'
                '\n```')
        assert len(tools.parse_tool_calls(text)) == 2

    def test_array_second_call_invalid_raises_positioned(self):
        with pytest.raises(ToolCallParseError, match="call 2"):
            tools.parse_tool_calls(
                '[{"tool": "list_sessions", "args": {}},'
                '{"tool": "nope", "args": {}}]')

    def test_array_element_missing_required_arg(self):
        with pytest.raises(
            ToolCallParseError, match="missing required arg 'query'"):
            tools.parse_tool_calls(
                '[{"tool": "search_kb", "args": {"scope": "rules"}}]')

    def test_unrelated_json_in_prose_falls_through_to_real_call(self):
        text = ('The format is {"foo": 1} — my actual call: '
                '{"tool": "list_sessions", "args": {}}')
        assert tools.parse_tool_calls(text) == [("list_sessions", {})]

    def test_no_json_returns_empty_list(self):
        assert tools.parse_tool_calls("plain answer only") == []

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
async def test_read_section_path_style_arg_matches_bare_name(fake_store, monkeypatch):
    """Display names are bare filenames; the model may pass a path it inferred
    from content/search context — basename fallback must still match."""
    async def _bare_ensure(kb_path):
        return _FakeStore(_FakeIdx([
            ("Humblewood_Backgrounds.md [Full Document]", "Bare-name body text here."),
        ]))

    import kb.retrievers as R
    monkeypatch.setattr(R, "_ensure_index_store", _bare_ensure)
    for arg in ("humblewood/HumbleWood/HumbleWood/Humblewood_Backgrounds.md",
                "HUMBLEWOOD_BACKGROUNDS.MD"):
        out = await tools.execute_tool("read_section", {"file": arg})
        assert "Bare-name body text here." in out, arg


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


# ── list_sessions + session-folder reads (date-named folders) ─────────────

def _make_session_kb(tmp_path):
    logs = tmp_path / "humblewood" / "HumbleWood" / "SessionLogs"
    for folder, first_line in [
        ("2026-09-09_01_DerAnfang", "# Session 1\nThe party leaves the farm at dawn."),
        ("2026-10-07_02", "They arrive at the witch's hut late evening."),
        ("2026-10-20_02_Back_to_Alderheart", "Bandits attack the city gates."),
    ]:
        d = logs / folder
        d.mkdir(parents=True)
        (d / "notes.md").write_text(first_line + "\nmore notes\n", encoding="utf-8")
    return tmp_path


@pytest.mark.asyncio
async def test_list_sessions_chronological_roster(fake_store, tmp_path, monkeypatch):
    kb_root = _make_session_kb(tmp_path)
    monkeypatch.setattr("config.settings.KB_PATH", str(kb_root))
    out = await tools.execute_tool("list_sessions", {})
    assert "3 campaign session(s)" in out
    # Ordinals follow CHRONOLOGICAL order, not folder names.
    i1 = out.index("Session 1")
    i2 = out.index("Session 2")
    i3 = out.index("Session 3")
    assert i1 < i2 < i3
    assert "2026-09-09" in out[i1:i2] and "DerAnfang" in out[i1:i2]
    # Title-less folder still listed (date + ordinal), teaser included.
    assert "2026-10-07" in out[i2:i3]
    assert "witch's hut" in out  # teaser line surfaced
    assert 'read_section(file=' in out  # follow-up hint


@pytest.mark.asyncio
async def test_list_sessions_none_found(fake_store, tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.KB_PATH", str(tmp_path))
    out = await tools.execute_tool("list_sessions", {})
    assert "No session logs" in out


def test_parse_list_sessions_no_args():
    name, args = parse_tool_call('{"tool": "list_sessions", "args": {}}')
    assert name == "list_sessions" and args == {}


@pytest.mark.asyncio
async def test_read_section_session_folder_fallback(tmp_path, monkeypatch):
    """Bare 'notes.md' is ambiguous across sessions; a FOLDER name is not —
    it reads that session's notes.md straight from disk."""
    kb_root = _make_session_kb(tmp_path)
    monkeypatch.setattr("config.settings.KB_PATH", str(kb_root))
    # Index chunks all share the bare name (as the real KB does).
    chunks = [
        ("notes.md [Full Document]", "indexed chunk one"),
        ("notes.md [Full Document -> Part 2]", "indexed chunk two"),
    ]

    async def _ensure(kb_path):
        return _FakeStore(_FakeIdx(chunks))

    import kb.retrievers as R
    monkeypatch.setattr(R, "_ensure_index_store", _ensure)
    out = await tools.execute_tool(
        "read_section", {"file": "2026-10-07_02"})
    assert "witch's hut" in out and "indexed chunk" not in out


@pytest.mark.asyncio
async def test_read_section_unknown_folder_falls_back_to_chunks(fake_store, tmp_path, monkeypatch):
    """No matching session folder → the old chunk-matching behavior."""
    monkeypatch.setattr("config.settings.KB_PATH", str(tmp_path))
    out = await tools.execute_tool("read_section", {"file": "no_such_folder"})
    assert "No chunk found" in out


# ── search_kb lexical leg ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_kb_lexical_leg_surfaces_exact_term(fake_store, monkeypatch):
    """A chunk dense ranks last but BM25 ranks first must surface on top
    after RRF fusion (the proper-noun case)."""
    import kb.retrievers as R

    def _fake_lexical(idx, query, limit):
        assert query == "fireball"
        return [("humblewood/DnDRules/spells.md [Spells -> Fireball]",
                 "Fireball deals 8d6 fire damage to each creature in a 20 ft radius.",
                 5.0)]

    monkeypatch.setattr(R, "_lexical_ranking", _fake_lexical)
    monkeypatch.setattr("config.settings.RAG_DENSE_TOP_K", 10)

    # The Fireball chunk must rank LAST in the dense leg (its name does not
    # contain the query head) so only the lexical leg can surface it.
    async def _dense_first(text, top_n=5):
        return ([("humblewood/DnDRules/DnD5_5/equipment.md [Equipment -> Armor]",
                  "Plate Armor: AC 18, 15 lb., 75 gp. Full entry text here.", 0.9),
                 ("humblewood/DnDRules/spells.md [Spells -> Fireball]",
                  "Fireball deals 8d6 fire damage to each creature in a 20 ft radius.",
                  0.1)], [0.1])

    # Re-point the store at an index whose dense ranking is fixed as above.
    async def _fixed_ensure(kb_path):
        class _FixedIdx(_FakeIdx):
            async def query_with_embeddings(self, text, top_n=5):
                return await _dense_first(text, top_n)
        return _FakeStore(_FixedIdx(fake_store._idx._chunks))

    monkeypatch.setattr(R, "_ensure_index_store", _fixed_ensure)
    out = await tools.execute_tool("search_kb", {"query": "fireball"})
    hits = json.loads(out.split("\n", 1)[1])
    assert "Fireball deals 8d6" in hits[0]["snippet"]


@pytest.mark.asyncio
async def test_search_kb_lexical_failure_degrades_to_dense(fake_store, monkeypatch):
    """A broken lexical leg must never break the search — dense-only result."""
    import kb.retrievers as R

    def _broken(idx, query, limit):
        raise RuntimeError("no BM25 in this fake")

    monkeypatch.setattr(R, "_lexical_ranking", _broken)
    monkeypatch.setattr("config.settings.RAG_DENSE_TOP_K", 10)
    out = await tools.execute_tool(
        "search_kb", {"query": "plate armor AC", "scope": "rules"})
    assert "hit(s)" in out


@pytest.mark.asyncio
async def test_search_kb_never_emits_truncated_json(fake_store, monkeypatch):
    """When hits exceed the tool-output cap, drop tail hits — never break
    the JSON array the model has to parse."""
    import kb.retrievers as R

    def _no_lex(idx, query, limit):
        return []

    monkeypatch.setattr(R, "_lexical_ranking", _no_lex)
    monkeypatch.setattr("config.settings.RAG_DENSE_TOP_K", 10)
    monkeypatch.setattr(tools, "_TOOL_RESULT_CAP", 400)
    out = await tools.execute_tool(
        "search_kb", {"query": "plate armor AC", "scope": "rules"})
    assert "hit(s):" in out and "[truncated]" not in out
    hits = json.loads(out[out.index("["):out.rindex("]") + 1])
    assert hits, "at least one complete hit must survive the cap"


# ── find_in_kb (exact presence, complete list) ────────────────────────────

@pytest.mark.asyncio
async def test_find_in_kb_complete_presence_list(fake_store, monkeypatch):
    """Every chunk containing the term is listed — case-insensitive, with
    per-location occurrence counts; no ranking, no cutoff by relevance."""
    import kb.retrievers as R

    async def _ensure(kb_path):
        return _FakeStore(_FakeIdx([
            ("humblewood/HumbleWood/SessionLogs/a.md [Full Document]",
             "They reached ALDERHEART at dawn. Alderheart's gates opened."),
            ("humblewood/HumbleWood/SessionLogs/b.md [Full Document]",
             "No mention here, just the forest."),
            ("humblewood/DnDRules/city.md [City -> Alderheart]",
             "Alderheart is a walled city of 4,000 souls."),
        ]))

    monkeypatch.setattr(R, "_ensure_index_store", _ensure)
    out = await tools.execute_tool("find_in_kb", {"term": "alderheart"})
    assert "2 location(s)" in out
    assert "×2" in out  # per-location occurrence count
    assert "city.md" in out and "a.md" in out
    assert "b.md" not in out


@pytest.mark.asyncio
async def test_find_in_kb_no_match(fake_store, monkeypatch):
    import kb.retrievers as R

    async def _ensure(kb_path):
        return _FakeStore(_FakeIdx(
            [("humblewood/x.md [Full Document]", "nothing relevant")]))

    monkeypatch.setattr(R, "_ensure_index_store", _ensure)
    out = await tools.execute_tool("find_in_kb", {"term": "Alderheart"})
    assert "No chunk contains" in out


@pytest.mark.asyncio
async def test_find_in_kb_caps_locations(fake_store, monkeypatch):
    import kb.retrievers as R

    async def _ensure(kb_path):
        return _FakeStore(_FakeIdx(
            [(f"humblewood/s{i}.md [Full Document]", "Alderheart once.")
             for i in range(5)]))

    monkeypatch.setattr(R, "_ensure_index_store", _ensure)
    monkeypatch.setattr(tools, "_FIND_MAX_LOCATIONS", 3)
    out = await tools.execute_tool("find_in_kb", {"term": "Alderheart"})
    assert "5 location(s)" in out          # the TRUE total is reported…
    assert "2 more location(s)" in out     # …beyond the shown cap


@pytest.mark.asyncio
async def test_find_in_kb_requires_term():
    with pytest.raises(ToolCallParseError, match="term"):
        await tools.execute_tool("find_in_kb", {})
