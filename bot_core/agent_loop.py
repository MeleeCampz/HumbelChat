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
    parse_tool_calls,
    tool_prompt_block,
)

log = logging.getLogger(__name__)


def _is_tool_call(text: str) -> bool:
    """True when the text IS a valid tool call (not merely mentions one)."""
    try:
        parse_tool_call(text)
        return True
    except ToolCallParseError:
        return False


def _strip_tool_json(text: str) -> str:
    """Remove JSON objects carrying a 'tool' key (fenced or bare)."""
    decoder = json.JSONDecoder()
    out: list[str] = []
    i = 0
    while True:
        j = text.find("{", i)
        if j == -1:
            out.append(text[i:])
            break
        try:
            obj, end = decoder.raw_decode(text[j:])
        except json.JSONDecodeError:
            out.append(text[i:j + 1])
            i = j + 1
            continue
        if isinstance(obj, dict) and "tool" in obj:
            out.append(text[i:j])
            i = j + (end - j)
        else:
            out.append(text[i:j + 1])
            i = j + 1
    return " ".join("".join(out).split())

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
            text = await complete(fb_messages, model, thinking=True)
        except Exception as exc:  # noqa: BLE001 — even the fallback died
            raise RuntimeError(
                f"agent fallback completion failed: {exc}") from exc
        return await _plain_final(text, fb_messages, complete, model), meta

    async def _plain_final(
        text: str,
        messages: list[dict[str, Any]],
        complete: CompletionFn,
        model: str,
    ) -> str:
        """A forced/final answer must be PROSE. If the model emits a tool
        call instead (there is no budget left to execute it), nudge once for
        plain text; if that still yields a call, strip the JSON and return
        whatever prose remains."""
        text = text.strip()
        if not _is_tool_call(text):
            return text
        log.warning("final answer was a tool call; nudging once for plain text")
        nudged = list(messages) + [
            {"role": "assistant", "content": text},
            {"role": "user",
             "content": ("No more tool calls — the budget is spent. Answer in "
                         "PLAIN TEXT now, using the information above.")},
        ]
        try:
            retry = (await complete(nudged, model, thinking=True)).strip()
            if not _is_tool_call(retry):
                return retry
            text = retry
        except Exception:  # noqa: BLE001 — keep the stripped original
            pass
        prose = _strip_tool_json(text)
        return prose or ("I couldn't finish that lookup within the tool "
                         "budget; please try asking again.")

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

            # ── Parse: tool call(s), or a plain-text answer ─────────────
            try:
                calls = parse_tool_calls(turn_text)
            except ToolCallParseError:
                calls = None  # invalid JSON present — same handling as none
            if not calls:
                if len(turn_text.strip()) >= _MIN_ANSWER_CHARS:
                    return turn_text.strip(), meta  # plain-text answer
                # Too short to be an answer — malformed/missing call; nudge.
                messages.append({"role": "assistant", "content": turn_text})
                messages.append({
                    "role": "user",
                    "content": ("That was neither a valid tool call nor an "
                                "answer. Reply with exactly one JSON object "
                                '{"tool": "<name>", "args": {...}} (or an '
                                "array of them) or answer in plain text."),
                })
                continue

            # ── Valid tool call(s) → execute (global cap) ───────────────
            messages.append({"role": "assistant", "content": turn_text})
            remaining_calls = RAG_AGENT_MAX_TOOL_CALLS - len(meta.tool_calls)
            if remaining_calls <= 0:
                messages.append({
                    "role": "user",
                    "content": ("Tool-call limit reached — answer now in "
                                "plain text with the information gathered so "
                                "far."),
                })
                continue
            batch = calls[:remaining_calls]
            skipped = len(calls) - len(batch)

            # Independent, read-only tools — run the WHOLE BATCH in parallel
            # (wall clock ≈ slowest call, not the sum). Every failure is
            # data for the model, never fatal.
            async def _run_one(
                pair: tuple[str, dict[str, Any]],
            ) -> str:
                cname, cargs = pair
                try:
                    return await execute_tool(cname, cargs)
                except ToolCallParseError as exc:
                    return f"Tool error: {exc}"
                except Exception as exc:  # noqa: BLE001 — tool failure is data
                    log.warning("agent tool %s failed: %s", cname, exc)
                    return f"Tool error: {type(exc).__name__}: {exc}"

            results = await asyncio.gather(*(_run_one(c) for c in batch))
            for (cname, cargs), result in zip(batch, results):
                meta.tool_calls.append(cname)
                _note_gathered(
                    f"[{cname} {json.dumps(cargs, ensure_ascii=False)}]",
                    result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{len(meta.tool_calls)}",
                    "content": result[:4000],
                })
            if skipped > 0:
                messages.append({
                    "role": "user",
                    "content": (f"{skipped} more call(s) were not executed — "
                                "the tool-call limit was reached. Answer now "
                                "in plain text with what you have."),
                })

        # Iteration cap reached — force the final answer.
        messages.append({
            "role": "user",
            "content": ("Iteration limit reached — give your final answer "
                        "now in plain text with the information gathered."),
        })
        forced = await complete(messages, model, thinking=True)
        return await _plain_final(forced, messages, complete, model), meta
    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        log.warning("agent loop failed (%s); falling back", exc)
        try:
            return await _finish_with_fallback(f"loop error: {exc}")
        except Exception:  # noqa: BLE001
            meta.fallback = True
            meta.fallback_reason = str(exc)
            return ("The knowledge-base lookup hit an unexpected error; "
                    "please try again."), meta
