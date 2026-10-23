#!/usr/bin/env python3
"""Agent-mode live probe (#39 §3.5) — exercises the bounded tool-calling loop
against the REAL backend + index, with RAG_AGENT_MODE forced to "always".

Usage (from the repo root):
    .venv/bin/python scripts/rag_agent_probe.py ["extra question", ...]

Defaults to the two golden-set enumeration cases plus one control:
    1. "List all the backgrounds in Humblewood"   (enumeration shape)
    2. "Which special spells exist?"              (enumeration shape)
    3. "What is the AC of plate armor?"          (control — should answer fast)

Each question runs run_agent_loop() directly (no Discord, no history), with a
CompletionFn bridged to the shared openai client exactly like
bot_core.ai_client._run_agent_turn does. Prints per-question: iterations,
tool calls, fallback status, elapsed time, and the final answer.

NOTE: this hammers the shared Studio backend — run it when nothing else is
queuing (the agent loop makes up to ~3 completions per question).
"""
import asyncio
import sys
import time
from typing import Any

sys.path.insert(0, ".")

from bot_core.agent_loop import run_agent_loop  # noqa: E402
from config.settings import INFER_URL, MODEL_NAME  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

DEFAULT_QUESTIONS = [
    "List all the backgrounds in Humblewood",
    "Which special spells exist?",
    "What is the AC of plate armor? (control — single fact)",
]


async def main(questions: list[str]) -> None:
    client = AsyncOpenAI(base_url=INFER_URL, api_key="unused")

    async def complete(messages: list[dict[str, Any]], model: str, *,
                       thinking: bool) -> str:
        kwargs: dict[str, Any] = dict(
            model=model, messages=messages, stream=False, timeout=180,
        )
        if not thinking:
            # Decision turns: no reasoning (fast, no budget leak).
            kwargs["extra_body"] = {"reasoning_effort": "off"}
        resp = await client.chat.completions.create(**kwargs)
        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise RuntimeError(f"empty completion: {resp!r}")
        return (choices[0].message.content or "").strip()

    for q in questions:
        print(f"\n{'=' * 72}\nQ: {q}\n{'-' * 72}")
        t0 = time.monotonic()
        try:
            text, meta = await run_agent_loop(
                complete=complete, model=MODEL_NAME,
                system_prompt=("You are Humbel, a D&D 5e rules expert. "
                              "Answer precisely and cite file/section per "
                              "key claim."),
                user_message=q,
            )
        except Exception as exc:  # noqa: BLE001 — probe: report, don't hide
            print(f"FAILED after {time.monotonic() - t0:.1f}s: {exc}")
            continue
        dt = time.monotonic() - t0
        print(f"iterations={meta.iterations} tools={meta.tool_calls} "
              f"fallback={meta.fallback} ({meta.fallback_reason}) "
              f"{dt:.1f}s")
        print(text[:1200])

    await client.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or DEFAULT_QUESTIONS))
