"""Bounded tool-calling loop for agent mode (#39, research doc §3.3).

Shape (per activation):

    system (persona + tool protocol) + history block + user question
      │  ≤ RAG_AGENT_MAX_ITERATIONS decision turns (non-thinking, fast)
      │  ≤ RAG_AGENT_MAX_TOOL_CALLS tool executions total
      │  ≤ RAG_AGENT_BUDGET_SECONDS wall clock for the whole loop
      ▼
    final answer turn (normal reasoning settings) — OR, on ANY cap / parse
    failure / timeout: single-turn fallback with the best context gathered
    so far (the tool transcript already contains every result).

Decision turns send the backend's "reasoning off" extra_body (fast + no FM6
reasoning leak); the final answer turn sends none (model default). Tool
results are appended as OpenAI-style ``role:"tool"`` messages.

The completion callable is injectable so the whole loop is unit-testable
with a scripted fake client (tests/test_agent_loop.py). Protocol::

    await complete(messages: list[dict], model: str, *, thinking: bool)
        -> str   (reply text)

``thinking=False`` on decision turns (fast, no reasoning leak); True on the
final answer / fallback turn (the caller's normal reasoning setting).
Every failure path degrades to the single-turn fallback — the loop never
raises for backend/tool errors.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Protocol

from kb.tools import (
    ToolCallParseError,
    execute_tool,
    parse_tool_call,
    tool_prompt_block,
)

log = logging.getLogger(__name__)

# A decision turn with NO tool call is treated as the final answer only when
# it looks like one (not a bare JSON fragment and at least this many chars).
_MIN_ANSWER_CHARS = 40

class CompletionFn(Protocol):
    """One backend completion: (messages, model, *, thinking) -> text."""

    def __call__(self, messages: list[dict[str, Any]], model: str,
                 *, thinking: bool) -> Awaitable[str]: ...


@dataclass
class AgentTurnMeta:
    iterations: int = 0
    tool_calls: list[str] = field(default_factory=list)
    fallback: bool = False
    fallback_reason: str = ""
    gathered_context: str = ""   # best-so-far context (search + reads)


def _agent_system_addendum() -> str:
    from config.settings import RAG_AGENT_MAX_ITERATIONS, RAG_AGENT_MAX_TOOL_CALLS

    return (
        tool_prompt_block()
        + f"\n(Hard limits: at most {RAG_AGENT_MAX_ITERATIONS} tool turns and "
          f"{RAG_AGENT_MAX_TOOL_CALLS} tool calls total — then you MUST "
          "answer with the information you have.)"
    )


async def _decision_completion(
    complete: CompletionFn,
    messages: list[dict[str, Any]],
    model: str,
) -> str:
    """One non-thinking decision turn (fast, no reasoning leak)."""
    return await complete(list(messages), model, thinking=False)


async def run_agent_loop(
    *,
    complete: CompletionFn,
    model: str,
    system_prompt: str = "",
    history_block: str = "",
    user_message: str = "",
    initial_context: str = "",
    initial_messages: list[dict[str, Any]] | None = None,
) -> tuple[str, AgentTurnMeta]:
    """Run the bounded loop; ALWAYS returns ``(answer_text, meta)``.

    Give EITHER the individual parts (system_prompt/history_block/
    user_message/initial_context) OR ``initial_messages`` — a fully built
    message list (e.g. the fast path's budget-trimmed prompt), which is used
    verbatim with only the tool-protocol addendum appended to its system
    message.
    """
    from config.settings import (
        RAG_AGENT_BUDGET_SECONDS,
        RAG_AGENT_MAX_ITERATIONS,
        RAG_AGENT_MAX_TOOL_CALLS,
    )

    meta = AgentTurnMeta()
    if initial_messages is not None:
        messages: list[dict[str, Any]] = [dict(m) for m in initial_messages]
        if messages and messages[0].get("role") == "system":
            base = str(messages[0].get("content") or "")
            messages[0]["content"] = (base + "\n\n" if base else "") \
                + _agent_system_addendum()
    else:
        messages = [
            {"role": "system",
             "content": system_prompt + "\n\n" + _agent_system_addendum()},
        ]
        if history_block:
            messages.append({"role": "user", "content": history_block})
        first_user = user_message
        if initial_context:
            first_user = (f"<kb-context>\n{initial_context}\n</kb-context>\n\n"
                          f"{user_message}")
        messages.append({"role": "user", "content": first_user})

    deadline = time.monotonic() + RAG_AGENT_BUDGET_SECONDS
    gathered: list[str] = [initial_context] if initial_context else []

    def _note_gathered(tool_line: str, result: str) -> None:
        gathered.append(f"{tool_line}\n{result}")
        meta.gathered_context = "\n\n".join(g for g in gathered if g)

    async def _finish_with_fallback(reason: str) -> tuple[str, AgentTurnMeta]:
        """Single-turn answer over the tool transcript (results included)."""
        nonlocal meta
        log.info("agent fallback (%s): single-turn answer over %d messages",
                 reason, len(messages))
        meta.fallback = True
        meta.fallback_reason = reason
        fb_messages = list(messages) + [{
            "role": "user",
            "content": (f"(The tool loop ended early: {reason}.) Give your "
                        "final answer now in plain text, using the "
                        "information above. Cite file/section per key claim; "
                        "if nothing relevant was found, say plainly that it "
                        "is not in the rules/campaign notes."),
        }]
        try:
            return (await complete(fb_messages, model, thinking=True)), meta
        except Exception as exc:  # noqa: BLE001 — even the fallback died
            raise RuntimeError(
                f"agent fallback completion failed: {exc}") from exc

    try:
        for iteration in range(1, RAG_AGENT_MAX_ITERATIONS + 1):
            meta.iterations = iteration
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return await _finish_with_fallback("wall-clock budget exceeded")

            try:
                turn_text = await asyncio.wait_for(
                    _decision_completion(complete, messages, model),
                    timeout=max(1.0, remaining),
                )
            except (asyncio.TimeoutError, TimeoutError):
                return await _finish_with_fallback("decision turn timed out")
            except Exception as exc:  # noqa: BLE001 — backend failure → fallback
                log.warning("agent decision turn failed (%s)", exc)
                return await _finish_with_fallback(f"decision turn error: {exc}")

            # ── Parse: tool call, or a plain-text answer ─────────────────
            try:
                name, args = parse_tool_call(turn_text)
            except ToolCallParseError:
                if len(turn_text.strip()) >= _MIN_ANSWER_CHARS:
                    return turn_text.strip(), meta  # plain-text answer
                # Too short to be an answer — malformed call; one nudge.
                messages.append({"role": "assistant", "content": turn_text})
                messages.append({
                    "role": "user",
                    "content": ("That was neither a valid tool call nor an "
                                "answer. Reply with exactly one JSON object "
                                '{"tool": "<name>", "args": {...}} or answer '
                                "in plain text."),
                })
                continue

            # ── Valid tool call → execute (global cap) ───────────────────
            messages.append({"role": "assistant", "content": turn_text})
            if len(meta.tool_calls) >= RAG_AGENT_MAX_TOOL_CALLS:
                messages.append({
                    "role": "user",
                    "content": ("Tool-call limit reached — answer now in "
                               "plain text with the information gathered so "
                               "far."),
                })
                continue

            call_id = f"call_{len(meta.tool_calls) + 1}"
            try:
                result = await execute_tool(name, args)
            except ToolCallParseError as exc:
                result = f"Tool error: {exc}"
            except Exception as exc:  # noqa: BLE001 — tool failure is data
                log.warning("agent tool %s failed: %s", name, exc)
                result = f"Tool error: {type(exc).__name__}: {exc}"

            meta.tool_calls.append(name)
            _note_gathered(
                f"[{name} {json.dumps(args, ensure_ascii=False)}]", result)
            messages.append({
                "role": "tool", "tool_call_id": call_id,
                "content": result[:4000],
            })

        # Iteration cap reached — force the final answer.
        messages.append({
            "role": "user",
            "content": ("Iteration limit reached — give your final answer "
                        "now in plain text with the information gathered."),
        })
        return (await complete(messages, model, thinking=True)), meta
    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        log.warning("agent loop failed (%s); falling back", exc)
        try:
            return await _finish_with_fallback(f"loop error: {exc}")
        except Exception:  # noqa: BLE001
            meta.fallback = True
            meta.fallback_reason = str(exc)
            return ("The knowledge-base lookup hit an unexpected error; "
                    "please try again."), meta
