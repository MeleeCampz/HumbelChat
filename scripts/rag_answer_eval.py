#!/usr/bin/env python3
r"""Answer-side evaluation (#39 §5.2) — score what the bot actually ANSWERS,
not just what it retrieves.

Retrieval quality (hit rate, recall@k) is measured by rag_benchmark.py. This
script measures the two answer axes §5.2 calls for, LLM-judged with the local
chat model (thinking OFF, per §5.2 — offline eval pass, never the hot path):

  * faithfulness — fraction of the answer's factual claims supported by the
    KB context that was attached (0-1). Measures FM3 hallucination /
    reaching-beyond-context.
  * "not found" correctness — for negative queries (no KB answer), whether the
    answer honestly says so instead of inventing a confident guess (FM5). This
    is the axis that measures Tier 0 item 2's honesty rules.

Pipeline: each case runs the REAL production path — bot_core.ai_client.ask_ai
(same RAG retrieval, prompt assembly incl. RAG_HONESTY_RULES, retry logic) —
with a synthetic guild id and history persistence monkeypatched to a no-op so
the run never touches persisted bot state. The attached context is captured by
wrapping _build_ai_request.

Usage (from the repo root, with the project venv):
    .venv/bin/python scripts/rag_answer_eval.py ["extra query", ...]
    Windows:  .\.venv\Scripts\python.exe scripts/rag_answer_eval.py

NOTE: hammers the shared Studio backend (~2 completions per case) — run when
nothing else is queuing.
"""
import asyncio
import json
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
        "  Windows:  .\\.venv\\Scripts\\python.exe scripts/rag_answer_eval.py\n"
        "  Linux/macOS:  .venv/bin/python scripts/rag_answer_eval.py")

# The bot's main.py loads .env; standalone scripts must do it themselves.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sandbox_shim  # noqa: F401,E402 — endpoint shims BEFORE .env load (sandbox only)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import bot_core.ai_client as ai_client  # noqa: E402
from bot_core.ai_client import _create_with_reasoning_fallback  # noqa: E402
from config.settings import DEFAULT_MODEL, INFER_API_KEY, INFER_URL  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

# (query, kind) — kind in {fact, multi_hop, enumeration, negative}.
DEFAULT_CASES: list[tuple[str, str]] = [
    # Controls — single facts with a clear KB answer.
    ("What is the AC of plate armor?", "fact"),
    ("Was macht der Zauberspruch Feuerball?", "fact"),
    # Hard tail — the standing failure-loop cases from the golden set.
    ("Gibt es ein Item dass mir hilft nicht aufgedeckt zu werden, wenn ich aus "
     "Stealth heraus einen Fernkampfangriff mache?", "fact"),
    ("Welche besonderen Zaubersprüche gibt es in Humblewood?", "enumeration"),
    ("Welche Hintergründe gibt es in Humblewood?", "enumeration"),
    # Multi-hop (§5.1) — answer needs ≥2 files.
    ("Can a fighter cast Fireball?", "multi_hop"),
    ("Welcher Hintergrund verleiht die Sprache Auran und welche Spezies sprechen sie?",
     "multi_hop"),
    # Negatives — nothing in the KB; honesty axis (item 2's honesty rules).
    ("What is the capital of France?", "negative"),
    ("Wie funktioniert die Steuererklärung in Deutschland?", "negative"),
]

_FAKE_GUILD = 999_000_001  # namespaced so it can never collide with a real guild

JUDGE_SYSTEM = (
    "You are a strict evaluation judge for a D&D knowledge-base assistant. "
    "Respond with exactly one JSON object and nothing else.")


async def _judge(client: AsyncOpenAI, prompt: str) -> dict[str, Any]:
    """One thinking-OFF completion; parse the first JSON object in the reply.

    Goes through the same production bridge as the agent loop: 'off' is not a
    value every backend accepts (this one wants 'none'), and the bridge's 4xx
    retry-without-param fallback makes the judge call robust either way."""
    resp = await _create_with_reasoning_fallback(
        client, "off",
        model=DEFAULT_MODEL,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": prompt}],
        stream=False,
        timeout=120,
    )
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise ValueError(f"empty judge completion: {resp!r}")
    text = (choices[0].message.content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"judge returned no JSON object: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def _faithfulness_prompt(query: str, context: str, answer: str) -> str:
    return (
        f"QUERY:\n{query}\n\n"
        f"KB CONTEXT (the only allowed source of facts):\n{context[:12000]}\n\n"
        f"ANSWER TO JUDGE:\n{answer[:6000]}\n\n"
        "Rate how well the answer is supported by the KB context. Ignore style, "
        "tone and formatting — only factual claims count. Claims of game "
        "mechanics, names, numbers or lore that are NOT present in the context "
        "count against faithfulness (the assistant may use general D&D 5e "
        "knowledge ONLY to connect sentences, not to add facts). Respond with "
        'exactly: {"score": <0.0-1.0 fraction of factual claims supported by '
        'the context>, "unsupported_claims": ["short quote of each unsupported '
        'claim", ...], "verdict": "<one sentence>"}')


def _honesty_prompt(query: str, answer: str) -> str:
    return (
        f"QUERY (a question the D&D knowledge base CANNOT answer — it has no "
        f"relevant content):\n{query}\n\n"
        f"ANSWER TO JUDGE:\n{answer[:6000]}\n\n"
        "The correct behaviour is to say honestly that this is not in the "
        "knowledge base (or not a D&D question) WITHOUT inventing specific "
        "facts. A general, clearly-labelled non-D&D remark is acceptable; "
        'confident invented specifics are not. Respond with exactly: {"admits_'
        'not_found": <true|false>, "invented_specifics": <true|false>, '
        '"verdict": "<one sentence>"}')


async def main(extra_queries: list[str]) -> None:
    if not DEFAULT_MODEL:
        sys.exit("MODEL_NAME is not set (.env) — the eval needs the chat model slug.")

    cases = list(DEFAULT_CASES)
    for q in extra_queries:
        cases.append((q, "extra"))

    # ── Side-effect isolation: never persist eval turns into bot history ────
    captured_contexts: dict[str, str] = {}
    _orig_build = ai_client._build_ai_request

    async def _capturing_build(*args: Any, **kwargs: Any) -> Any:
        ctx = await _orig_build(*args, **kwargs)
        q = kwargs.get("user_message") or (args[0] if args else "")
        captured_contexts[str(q)] = getattr(ctx, "rag_context", "") or ""
        return ctx

    async def _no_persist(*args: Any, **kwargs: Any) -> None:
        return None

    ai_client._build_ai_request = _capturing_build  # type: ignore[assignment]
    ai_client._persist_turn = _no_persist  # type: ignore[assignment]

    client = AsyncOpenAI(base_url=INFER_URL, api_key=INFER_API_KEY or "unused")
    results: list[dict[str, Any]] = []
    try:
        for i, (query, kind) in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {kind}: {query[:70]}", flush=True)
            t0 = time.monotonic()
            answer = ""
            error = ""
            try:
                answer, _extra = await ai_client.ask_ai(
                    query, DEFAULT_MODEL, _FAKE_GUILD, None, username="eval")
            except Exception as exc:  # noqa: BLE001 — eval: record, don't hide
                error = f"{type(exc).__name__}: {exc}"
            dt = time.monotonic() - t0
            context = captured_contexts.get(query, "")

            row: dict[str, Any] = {
                "query": query, "kind": kind, "answer": answer[:2000],
                "error": error, "elapsed_s": round(dt, 1),
                "context_chars": len(context),
            }
            if not error:
                try:
                    if kind == "negative":
                        judged = await _judge(
                            client, _honesty_prompt(query, answer))
                        row.update(judged)
                    else:
                        judged = await _judge(
                            client, _faithfulness_prompt(query, context, answer))
                        row.update(judged)
                except Exception as exc:  # noqa: BLE001 — judge failure is data
                    row["judge_error"] = f"{type(exc).__name__}: {exc}"
            results.append(row)
            status = (row.get("score") if kind != "negative"
                      else row.get("admits_not_found"))
            print(f"      {dt:.0f}s context={len(context)}ch "
                  f"{'score=' + str(status) if status is not None else error or 'judge-error'}",
                  flush=True)
    finally:
        ai_client._build_ai_request = _orig_build  # type: ignore[assignment]
        await client.close()

    # ── Aggregate ────────────────────────────────────────────────────────────
    scored = [r for r in results if "score" in r and not r.get("error")]
    negatives = [r for r in results
                 if r["kind"] == "negative" and "admits_not_found" in r]
    print("\n" + "=" * 72)
    if scored:
        avg = sum(r["score"] for r in scored) / len(scored)
        print(f"FAITHFULNESS (n={len(scored)}): mean {avg:.2f} · "
              f"min {min(r['score'] for r in scored):.2f} · "
              f"max {max(r['score'] for r in scored):.2f}")
    if negatives:
        honest = sum(1 for r in negatives if r.get("admits_not_found"))
        clean = sum(1 for r in negatives
                    if r.get("admits_not_found") and not r.get("invented_specifics"))
        print(f"NEGATIVE HONESTY (n={len(negatives)}): admits-not-found {honest}/"
              f"{len(negatives)} · fully clean (no invented specifics) {clean}/"
              f"{len(negatives)}")
    for r in results:
        if r.get("error") or r.get("judge_error"):
            print(f"  ERROR  {r['query'][:60]}: {r.get('error') or r.get('judge_error')}")

    out = Path(__file__).with_name("rag_answer_eval_last.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nresults → {out}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
