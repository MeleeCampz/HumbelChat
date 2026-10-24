"""A/B evaluation: RAG agent mode ``always`` vs ``auto`` trigger counterfactual.

For every prompt we run TWO real end-to-end passes through ``ask_ai``:

  A. ``RAG_AGENT_MODE=off``    → the production fast path (single completion)
  B. ``RAG_AGENT_MODE=always`` → tools available, model decides per turn

From pass A's turn record we read the rerank top score and re-evaluate the
``auto`` trigger rule on the identical query — so for every prompt we know
what production (``auto``, the current .env setting) would have done:

  auto-triggered → its result IS pass B (same code path, same budget)
  auto-not       → its result IS pass A (fast path)

Captured per pass (via the per-turn records in bot_core/turn_logger): full
answer text, total wall time, agent iterations / tool calls / fallback, and
the retrieval rerank/dense top scores. Output: one JSONL file + a summary
table on stdout.

Isolation: history persistence is disabled (no cross-pass contamination, no
pollution of real campaign history), last-session context is disabled
(identical conditions both passes), and the turn-log dir is pointed at the
eval output dir. Production .env is NOT modified — mode switching is
in-process only.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # .env + relative KB paths resolve like in production

# Sandbox network shim: .env points EMBED_URL at the docker-internal service
# name (correct for the bot container). From the sandbox that hostname does
# not work; compose publishes the same sidecar on 127.0.0.1:11435, reachable
# as host.docker.internal:11435 — the endpoint previous healthy benchmarks
# used. Host runs are unaffected (shim only applies inside the sandbox).
# Sandbox detection: clone mode has /run/sandbox; direct mode mounts the host
# tree at a drive letter (/c/Repos/...).
def _in_sandbox() -> bool:
    return (Path("/run/sandbox").exists()
            or re.match(r"^/[a-z]{1,2}/", str(REPO)) is not None)

_SANDBOX_ENDPOINT_SHIMS = {
    # docker service name (bot container) → published port on the host
    "EMBED_URL": "http://host.docker.internal:11435/v1",
    "RERANK_API_BASE": "http://host.docker.internal:8000",
}
if _in_sandbox():
    os.environ.update(_SANDBOX_ENDPOINT_SHIMS)

from dotenv import load_dotenv

load_dotenv(REPO / ".env")  # BEFORE any bot module import (settings read env at import)
# load_dotenv does not override already-set vars — the shims above survive.
if _in_sandbox():
    os.environ.update(_SANDBOX_ENDPOINT_SHIMS)


# ─────────────────────────── prompt set ───────────────────────────
# (id, category, text). Categories: simple-rules, enumeration, compound,
# negative (impossible/absent), session-info, items, monsters. Mixed DE/EN
# like the live campaign. 36 prompts.
PROMPTS: list[tuple[str, str, str]] = [
    # ── simple rules (fast path should nail these; tools would be noise) ──
    ("r01", "simple-rules", "What is the formula for Armor Class with Dexterity?"),
    ("r02", "simple-rules", "Wie lange dauert eine kurze Rast und was darf man dabei?"),
    ("r03", "simple-rules", "How do advantage and disadvantage work exactly?"),
    ("r04", "simple-rules", "When do spell slots come back after a rest?"),
    ("r05", "simple-rules", "What is the base movement speed for humans?"),
    ("r06", "simple-rules", "Welche Waffeneigenschaften gibt es im Regelbuch?"),
    ("r07", "simple-rules", "How many ability checks can Dexterity cover?"),
    ("r08", "simple-rules", "What happens when you roll a natural 20 on an attack?"),
    # ── enumeration (auto trigger: shape) ──
    ("e01", "enumeration", "Which cantrips are in the player's spell list?"),
    ("e02", "enumeration", "List all magic weapons in the item collection."),
    ("e03", "enumeration", "Welche Feats gibt es, die Intelligenz verbessern?"),
    ("e04", "enumeration", "Which monsters have appeared in our campaign logs so far?"),
    ("e05", "enumeration", "List every healing spell that works on a single target."),
    ("e06", "enumeration", "Welche Backgrounds sind im Humblewood-Setting verfügbar?"),
    # ── compound (auto trigger: shape) ──
    ("c01", "compound", "What happened in session 5 and which NPCs were there?"),
    ("c02", "compound", "Which feats boost Intelligence and what do they each do?"),
    ("c03", "compound", "Welche Rassen haben Charisma-Spellcasting und welche Zauber nutzen sie?"),
    ("c04", "compound", "What did we find in the tomb and which traps were in it?"),
    # ── negative / impossible (honesty check — must say 'not in the KB') ──
    ("n01", "negative", "What is the weather like on Mars right now?"),
    ("n02", "negative", "Did the party find a dragon hoard in session 3?"),
    ("n03", "negative", "What are the rules for quantum entanglement in D&D?"),
    ("n04", "negative", "How do you craft a Tarrasque at home?"),
    ("n05", "negative", "Gibt es Regeln für Zeitreisen im Regelbuch?"),
    ("n06", "negative", "Who is the mayor of Strixhaven and what laws did they pass?"),
    # ── session info (campaign logs; contextual-retrieval territory) ──
    ("s01", "session-info", "What happened in session 7?"),
    ("s02", "session-info", "Was ist in der ersten Session passiert?"),
    ("s03", "session-info", "Which NPCs did we meet in the desert?"),
    ("s04", "session-info", "What did the bandits want from us?"),
    ("s05", "session-info", "Summarize our most recent session."),
    ("s06", "session-info", "Welche Entscheidungen hat die Partei in Session 4 getroffen?"),
    # ── items ──
    ("i01", "items", "What does the Headband of Intellect do exactly?"),
    ("i02", "items", "Welche magischen Ringe gibt es in der Sammlung?"),
    ("i03", "items", "Find an item that helps restore hit dice faster."),
    ("i04", "items", "What is written in the Tome of Clear Thought?"),
    # ── monsters ──
    ("m01", "monsters", "Describe a Wight: stats, traits, and weaknesses."),
    ("m02", "monsters", "Welche Monster leben in der Wüste?"),
]


async def _run_pass(prompt_text: str, mode: str) -> tuple[str, dict]:
    """One real ask_ai pass with the given agent mode; returns (text, record)."""
    import config.settings as settings
    from bot_core import ai_client

    settings.RAG_AGENT_MODE = mode  # read inside agent_trigger_reason per call
    t0 = time.monotonic()
    reply_text, extra = await ai_client.ask_ai(
        user_message=prompt_text,
        model_slug=os.environ.get("MODEL_NAME", "default"),
        guild_id=999_999_999,          # synthetic — never a real guild
        channel_id=999_999_990,
        username="mode-eval", user_id=None, char_key=None,
    )
    elapsed = time.monotonic() - t0
    record = _latest_turn_record()
    if record is not None:
        record["pass_wall_s"] = round(elapsed, 1)
    return reply_text, {**(extra or {}), "record": record}


def _latest_turn_record() -> dict | None:
    import glob

    files = glob.glob(str(TURN_DIR / "*.json"))
    if not files:
        return None
    newest = max(files, key=os.path.getmtime)
    try:
        return json.loads(Path(newest).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — eval must not die on a logging hiccup
        return None


async def main() -> None:
    from bot_core import ai_client, turn_logger
    import config.settings as settings

    # ── Isolation: no history writes, no cross-pass contamination ──────
    def _no_persist(*args: object, **kwargs: object) -> None:
        return None  # _persist_turn is a sync function in ai_client

    ai_client._persist_turn = _no_persist          # type: ignore[method-assign]
    ai_client.LAST_SESSION_CONTEXT_ENABLED = False  # type: ignore[attr-defined]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%Z")
    out_dir = REPO / "data" / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    global TURN_DIR
    TURN_DIR = out_dir / f"mode_compare_{stamp}"
    TURN_DIR.mkdir(parents=True, exist_ok=True)
    turn_logger.AI_TURN_LOG_ENABLED = True         # type: ignore[misc]
    turn_logger.AI_TURN_LOG_DIR = str(TURN_DIR)   # type: ignore[misc]
    turn_logger.AI_TURN_LOG_KEEP = 500             # type: ignore[misc]

    settings.RAG_AGENT_SCORE_GATE = 0.55  # document the gate used, explicitly

    from bot_core.ai_client import agent_trigger_reason

    results_path = out_dir / f"mode_compare_{stamp}.jsonl"
    rows: list[dict] = []
    t_start = time.monotonic()

    for idx, (pid, category, text) in enumerate(PROMPTS, 1):
        row: dict = {"id": pid, "category": category, "text": text}
        print(f"[{idx:02d}/{len(PROMPTS)}] {pid} ({category}) pass A (off)…", flush=True)
        try:
            text_a, out_a = await _run_pass(text, "off")
            row["fast"] = {
                "answer": text_a,
                "wall_s": out_a.get("record", {}).get("pass_wall_s"),
                "rerank_top": (out_a.get("record", {}).get("retrieval") or {}).get("rerank_top"),
                "dense_top": (out_a.get("record", {}).get("retrieval") or {}).get("dense_top"),
            }
        except Exception as exc:  # noqa: BLE001 — record and continue
            row["fast"] = {"error": repr(exc)}

        # auto counterfactual on the SAME query + observed rerank top
        settings.RAG_AGENT_MODE = "auto"
        row["auto_would_trigger"] = agent_trigger_reason(
            text, (row.get("fast") or {}).get("rerank_top"))

        print(f"[{idx:02d}/{len(PROMPTS)}] {pid} ({category}) pass B (always)…", flush=True)
        try:
            text_b, out_b = await _run_pass(text, "always")
            rec_b = out_b.get("record") or {}
            agent_b = rec_b.get("agent") or {}
            row["always"] = {
                "answer": text_b,
                "wall_s": rec_b.get("pass_wall_s"),
                "iterations": agent_b.get("iterations"),
                "tool_calls": agent_b.get("tool_calls"),
                "fallback": agent_b.get("fallback"),
                "fallback_reason": agent_b.get("fallback_reason"),
                "gathered_context": agent_b.get("gathered_context", ""),
            }
        except Exception as exc:  # noqa: BLE001 — record and continue
            row["always"] = {"error": repr(exc)}

        rows.append(row)
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        inv = bool((row.get("always", {}).get("tool_calls")))
        trig = row["auto_would_trigger"]
        print(f"    → tools={'YES ' + str(row['always']['tool_calls']) if inv else 'no'}  "
              f"auto-trigger={trig or '—'}  "
              f"fast={row.get('fast', {}).get('wall_s')}s  always={row.get('always', {}).get('wall_s')}s",
              flush=True)

    total = time.monotonic() - t_start
    summary = {
        "n_prompts": len(rows),
        "total_wall_s": round(total, 1),
        "always_invoked_tools": [r["id"] for r in rows if r.get("always", {}).get("tool_calls")],
        "always_fallbacks": [r["id"] for r in rows if r.get("always", {}).get("fallback")],
        "auto_would_trigger": {
            r["id"]: r["auto_would_trigger"] for r in rows if r["auto_would_trigger"]
        },
        "errors": {r["id"]: {k: v.get("error") for k, v in r.items()
                             if isinstance(v, dict) and v.get("error")}
                   for r in rows if any(isinstance(v, dict) and v.get("error")
                                         for v in r.values())},
    }
    results_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nresults: {results_path}")


TURN_DIR: Path = Path("logs/ai_turns")

if __name__ == "__main__":
    asyncio.run(main())
