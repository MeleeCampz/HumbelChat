#!/usr/bin/env python3
r"""Agent-mode live probe (#39 §3.5) — exercises the bounded tool-calling loop
against the REAL backend + index.

Usage (from the repo root, with the project venv):
    .venv/bin/python scripts/rag_agent_probe.py ["extra question", ...]
    Windows:  .\.venv\Scripts\python.exe scripts/rag_agent_probe.py

Defaults to the two golden-set enumeration cases plus one control:
    1. "List all the backgrounds in Humblewood"   (enumeration shape)
    2. "Which special spells exist?"              (enumeration shape)
    3. "What is the AC of plate armor?"          (control — should answer fast)

Each question runs run_agent_loop() directly (no Discord, no history, trigger
bypassed — this measures the LOOP itself, not RAG_AGENT_MODE), with a
CompletionFn bridged to the shared openai client exactly like
bot_core.ai_client._run_agent_turn does: same reasoning-off control for
decision turns (with the 4xx fallback) and the same AI_TIMEOUT_S idle
watchdog. Prints per-question: iterations, tool calls, fallback status,
elapsed time, and the final answer.

NOTE: this hammers the shared Studio backend — run it when nothing else is
queuing (the agent loop makes up to ~3 completions per question).
"""
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    import openai  # noqa: F401 — must come from the project venv
except ModuleNotFoundError:  # pragma: no cover — environment guard
    sys.exit(
        "openai not found — you are using the system Python.\n"
        "Run with the project venv instead, from the repo root:\n"
        "  Windows:  .\\.venv\\Scripts\\python.exe scripts/rag_agent_probe.py\n"
        "  Linux/macOS:  .venv/bin/python scripts/rag_agent_probe.py")

# The bot's main.py loads .env; standalone scripts must do it themselves or
# MODEL_NAME / INFER_URL silently fall back to defaults (see rag_floor_probe).
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from bot_core.ai_client import (  # noqa: E402 — same bridge as production
    _create_with_reasoning_fallback,
)
from bot_core.agent_loop import run_agent_loop  # noqa: E402
from config.settings import (  # noqa: E402
    AI_REASONING_EFFORT,
    AI_TIMEOUT_S,
    DEFAULT_MODEL,
    INFER_URL,
)
from openai import AsyncOpenAI  # noqa: E402

DEFAULT_QUESTIONS = [
    "List all the backgrounds in Humblewood",
    "Which special spells exist?",
    "What is the AC of plate armor? (control — single fact)",
]


async def main(questions: list[str]) -> None:
    if not DEFAULT_MODEL:
        sys.exit("MODEL_NAME is not set (.env) — the probe needs the chat model slug.")
    client = AsyncOpenAI(base_url=INFER_URL, api_key="unused")

    async def complete(messages: list[dict[str, Any]], model: str, *,
                       thinking: bool) -> str:
        # Same bridge as _run_agent_turn in bot_core/ai_client.py: decision
        # turns run with reasoning OFF (fast, no budget leak); the final /
        # fallback answer uses the configured effort. The helper carries the
        # 4xx retry-without-param fallback for older backends.
        resp = await _create_with_reasoning_fallback(
            client, "off" if not thinking else AI_REASONING_EFFORT,
            model=model, messages=list(messages), stream=False,
            timeout=AI_TIMEOUT_S,
        )
        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise RuntimeError(f"empty completion: {resp!r}")
        return (choices[0].message.content or "").strip()

    for q in questions:
        print(f"\n{'=' * 72}\nQ: {q}\n{'-' * 72}")
        t0 = time.monotonic()
        try:
            text, meta = await run_agent_loop(
                complete=complete, model=DEFAULT_MODEL,
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
