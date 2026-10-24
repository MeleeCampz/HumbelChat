"""Tests for bot_core.agent_loop — the bounded tool-calling loop (#39 §3.3).

A scripted fake completion stands in for the backend; the tools layer is
monkeypatched so nothing touches the network or the index.
"""
import asyncio

import pytest

from bot_core import agent_loop
from kb.tools import ToolCallParseError


class _ScriptedComplete:
    """Pops one reply per call; records (messages, thinking) per turn."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[tuple[list[dict], bool]] = []

    async def __call__(self, messages, model, *, thinking):
        self.calls.append((messages, thinking))
        if not self.script:
            raise AssertionError("scripted completions exhausted")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        await asyncio.sleep(0)  # yield to the loop
        return item


@pytest.fixture(autouse=True)
def _no_real_tools(monkeypatch):
    """execute_tool never touches the index in these tests."""
    async def _fake_execute(name, args):
        return f"RESULT[{name} {args}]"

    monkeypatch.setattr(agent_loop, "execute_tool", _fake_execute)


@pytest.fixture(autouse=True)
def _loop_limits(monkeypatch):
    import config.settings as S
    monkeypatch.setattr(S, "RAG_AGENT_MAX_ITERATIONS", 2)
    monkeypatch.setattr(S, "RAG_AGENT_MAX_TOOL_CALLS", 4)
    monkeypatch.setattr(S, "RAG_AGENT_BUDGET_SECONDS", 30)


def _start(script, **kwargs):
    comp = _ScriptedComplete(script)
    return agent_loop.run_agent_loop(complete=comp, model="test-model",
                                     system_prompt="PERSONA",
                                     user_message="What is plate armor AC?",
                                     **kwargs), comp


@pytest.mark.asyncio
async def test_search_then_answer():
    coro, comp = _start([
        '{"tool": "search_kb", "args": {"query": "plate armor AC"}}',
        "Plate armor is AC 18 (equipment.md [Equipment -> Armor]).",
    ])
    text, meta = await coro
    assert "AC 18" in text
    assert meta.tool_calls == ["search_kb"]
    assert not meta.fallback
    # Both turns are decision turns (the model answered on its own within
    # one) — both non-thinking. Forced/fallback final turns use thinking=True
    # (covered by the iteration-cap + fallback tests).
    assert comp.calls[0][1] is False
    assert comp.calls[1][1] is False
    # The tool result reached the model as a role:"tool" message.
    assert any(m.get("role") == "tool" for m in comp.calls[1][0])


@pytest.mark.asyncio
async def test_immediate_plain_answer_no_tools():
    coro, comp = _start([
        "No KB lookup needed — the answer is in the attached context: AC 18.",
    ])
    text, meta = await coro
    assert "AC 18" in text
    assert meta.tool_calls == []
    assert not meta.fallback
    assert len(comp.calls) == 1


@pytest.mark.asyncio
async def test_malformed_call_gets_one_nudge_then_answer():
    coro, comp = _start([
        "hmm",  # too short to be an answer → nudge
        '{"tool": "list_sections", "args": {"file": "equipment.md"}}',
        "Answer after listing: sections are Armor and Weapons.",
    ])
    text, meta = await coro
    assert "sections" in text.lower()
    assert meta.tool_calls == ["list_sections"]
    # The nudge message was fed back before the second decision turn.
    assert any("valid tool call" in str(m.get("content", ""))
               for m in comp.calls[1][0])


@pytest.mark.asyncio
async def test_tool_call_cap_forces_answer(monkeypatch):
    # One call per decision turn → with the default 2-iteration cap only two
    # calls can execute; raise the iteration cap to exercise the GLOBAL call
    # cap (C=4): the 5th call is refused and the model must answer.
    import config.settings as S
    monkeypatch.setattr(S, "RAG_AGENT_MAX_ITERATIONS", 6)
    coro, comp = _start([
        '{"tool": "search_kb", "args": {"query": "a"}}',
        '{"tool": "search_kb", "args": {"query": "b"}}',
        '{"tool": "search_kb", "args": {"query": "c"}}',
        '{"tool": "search_kb", "args": {"query": "d"}}',
        '{"tool": "search_kb", "args": {"query": "e"}}',  # cap hit → nudge
        "Final: a, b, c, d gathered — limit reached.",
    ])
    text, meta = await coro
    assert meta.tool_calls == ["search_kb"] * 4
    assert "Final:" in text


@pytest.mark.asyncio
async def test_iteration_cap_forces_answer():
    # 2 iterations max; both emit tool calls → the forced-answer turn fires.
    coro, comp = _start([
        '{"tool": "search_kb", "args": {"query": "a"}}',
        '{"tool": "read_section", "args": {"file": "x.md"}}',
        "Forced final answer with what I have.",
    ])
    text, meta = await coro
    assert meta.iterations == 2
    assert "Forced final" in text


@pytest.mark.asyncio
async def test_decision_timeout_falls_back(monkeypatch):
    import config.settings as S
    monkeypatch.setattr(S, "RAG_AGENT_BUDGET_SECONDS", 1)

    async def _complete(messages, model, *, thinking):
        if thinking:
            return "Fallback answer over the transcript."
        await asyncio.sleep(5)  # exceeds the 1 s budget → cancelled
        raise AssertionError("decision turn should have been cancelled")

    text, meta = await agent_loop.run_agent_loop(
        complete=_complete, model="m", system_prompt="P", user_message="Q")
    assert meta.fallback
    assert "timed out" in meta.fallback_reason
    assert "Fallback answer" in text


@pytest.mark.asyncio
async def test_backend_error_falls_back_over_transcript():
    coro, comp = _start([
        RuntimeError("backend 500"),
        "Fallback: using gathered info.",
    ])
    text, meta = await coro
    assert meta.fallback
    assert "decision turn error" in meta.fallback_reason
    assert "Fallback:" in text


@pytest.mark.asyncio
async def test_tool_error_is_data_not_fatal(monkeypatch):
    async def _boom(name, args):
        raise ToolCallParseError("give 'file' or 'dir'")

    monkeypatch.setattr(agent_loop, "execute_tool", _boom)
    # Turn 1: tool call → error result fed back as a tool message. Turn 2:
    # the model answers (≥40 chars so it counts as an answer, not a nudge).
    coro, comp = _start([
        '{"tool": "list_sections", "args": {}}',
        "Answer despite tool error: nothing found in the KB.",
    ])
    text, meta = await coro
    assert "Answer despite" in text
    # The error surfaced to the model as a tool message.
    assert any("Tool error" in str(m.get("content", ""))
               for m in comp.calls[1][0])


@pytest.mark.asyncio
async def test_forced_answer_that_is_a_tool_call_gets_nudged(monkeypatch):
    """Observed live: the final-answer turn can emit a tool call instead of
    prose (no budget left to run it). The loop must nudge for plain text."""
    import config.settings as S
    monkeypatch.setattr(S, "RAG_AGENT_MAX_ITERATIONS", 1)

    comp = _ScriptedComplete([
        '{"tool": "search_kb", "args": {"query": "a"}}',
        # Forced-answer turn emits a tool call instead of prose:
        'Let me read it. {"tool": "read_section", "args": {"file": "x.md"}}',
        # Nudge → plain text:
        "Final prose answer after the nudge.",
    ])
    text, meta = await agent_loop.run_agent_loop(
        complete=comp, model="m", system_prompt="P", user_message="Q")
    assert "Final prose answer" in text
    assert "read_section" not in text


@pytest.mark.asyncio
async def test_final_tool_call_surviving_nudge_is_stripped(monkeypatch):
    import config.settings as S
    monkeypatch.setattr(S, "RAG_AGENT_MAX_ITERATIONS", 1)

    comp = _ScriptedComplete([
        '{"tool": "search_kb", "args": {"query": "a"}}',
        'Prose lead-in. {"tool": "read_section", "args": {"file": "x.md"}}',
        # Nudge still yields a tool call → the JSON is stripped:
        'Still calling. {"tool": "search_kb", "args": {"query": "b"}}',
    ])
    text, meta = await agent_loop.run_agent_loop(
        complete=comp, model="m", system_prompt="P", user_message="Q")
    assert '"tool"' not in text
    assert "Still calling" in text


@pytest.mark.asyncio
async def test_initial_messages_get_protocol_addendum():
    # ≥40 chars so the turn counts as an answer, not a malformed call.
    comp = _ScriptedComplete(["Built-on-prompt answer: plate armor is AC 18."])
    text, meta = await agent_loop.run_agent_loop(
        complete=comp, model="m",
        initial_messages=[
            {"role": "system", "content": "PERSONA-ONLY"},
            {"role": "user", "content": "<history>…</history>"},
            {"role": "user", "content": "the question"},
        ],
    )
    assert "Built-on-prompt" in text
    sys_msg = comp.calls[0][0][0]
    assert sys_msg["content"].startswith("PERSONA-ONLY")
    assert '"tool"' in sys_msg["content"]       # protocol appended
    assert "Hard limits" in sys_msg["content"]


# ── Parallel batch execution (array protocol) ─────────────────────────────

@pytest.mark.asyncio
async def test_batch_of_two_runs_both_and_records_each():
    """An array of independent calls executes in ONE decision turn; every
    call gets its own role:"tool" result and is counted against the cap."""
    batch = ('[{"tool": "find_in_kb", "args": {"term": "Alderheart"}},'
             '{"tool": "list_sessions", "args": {}}]')
    coro, comp = _start([
        batch,
        "Alderheart appears in sessions 3, 4 and 7.",
    ])
    text, meta = await coro
    assert "sessions 3, 4 and 7" in text
    assert meta.tool_calls == ["find_in_kb", "list_sessions"]
    # Both results were fed back before the answering turn.
    tool_msgs = [m for m in comp.calls[1][0] if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert any("RESULT[find_in_kb" in m["content"] for m in tool_msgs)
    assert any("RESULT[list_sessions" in m["content"] for m in tool_msgs)


@pytest.mark.asyncio
async def test_batch_truncated_to_remaining_cap_reports_skipped(monkeypatch):
    """A batch bigger than the remaining call budget executes what fits and
    tells the model the rest was skipped — the cap still holds globally."""
    import config.settings as S

    executed: list[str] = []

    async def _fake_execute(name, args):
        executed.append(name)
        return f"RESULT[{name} {args}]"

    monkeypatch.setattr(agent_loop, "execute_tool", _fake_execute)
    monkeypatch.setattr(S, "RAG_AGENT_MAX_TOOL_CALLS", 1)
    batch = ('[{"tool": "read_section", "args": {"file": "a.md"}},'
             '{"tool": "read_section", "args": {"file": "b.md"}},'
             '{"tool": "read_section", "args": {"file": "c.md"}}]')
    coro, comp = _start([batch, "Only a.md was read."])
    text, meta = await coro
    assert executed == ["read_section"]          # only what fit the cap
    assert len(meta.tool_calls) == 1
    # The model was told the rest did not run.
    last_user = [m for m in comp.calls[1][0]
                 if m.get("role") == "user"][-1]["content"]
    assert "not executed" in last_user


@pytest.mark.asyncio
async def test_batch_runs_calls_concurrently(monkeypatch):
    """Independent calls in one batch overlap in time (gather, not sum)."""

    started: list[str] = []

    async def _fake_execute(name, args):
        started.append(args["file"])
        await asyncio.sleep(0.05)
        return f"RESULT[{name} {args}]"

    monkeypatch.setattr(agent_loop, "execute_tool", _fake_execute)
    t0 = asyncio.get_event_loop().time()
    batch = ('[{"tool": "read_section", "args": {"file": "a.md"}},'
             '{"tool": "read_section", "args": {"file": "b.md"}},'
             '{"tool": "read_section", "args": {"file": "c.md"}}]')
    coro, _comp = _start([batch, "done."])
    await coro
    elapsed = asyncio.get_event_loop().time() - t0
    # Three 50 ms calls in parallel ≈ one call's time (allow scheduler slack).
    assert elapsed < 0.12, f"batch took {elapsed:.3f}s — not parallel?"
    assert sorted(started) == ["a.md", "b.md", "c.md"]
