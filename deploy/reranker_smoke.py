#!/usr/bin/env python3
"""Smoke test for the Qwen3-Reranker vLLM sidecar (reranker-qwen service).

Verifies, with zero dependencies:
  1. the server is up (/health),
  2. /rerank answers in the Jina-compatible shape the bot expects
     ({"results": [{"index", "relevance_score"}, ...]}),
  3. a relevant passage outranks an irrelevant one — for an English and a
     German query (the KB is mixed-language, so cross-lingual ordering
     matters).

Usage:
    python3 deploy/reranker_smoke.py [base_url]

    base_url defaults to http://127.0.0.1:8000 (run on the host).
    From the dev sandbox use:  http://host.docker.internal:8000
    (the service publishes 127.0.0.1:8000:8000 — see docker-compose.yml).

Exit code 0 = healthy, 1 = problem (with a human-readable reason).
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

# Mirrors RERANK_INSTRUCT in .env (kept short here; the exact string does not
# affect pass/fail of the ranking check).
INSTRUCT = (
    "Given a question about rules, items, monsters, or lore from a Dungeons & "
    "Dragons 5e campaign, rank passages by relevance."
)

# (label, query, relevant_doc, irrelevant_doc) — relevant must rank first.
CASES: list[tuple[str, str, str, str]] = [
    (
        "EN",
        "How do I try to hide from enemies without being seen?",
        "Stealth. To sneak past someone, hide, or slip away unnoticed, you can "
        "make a Dexterity (Stealth) check. The DM might call for a Stealth "
        "check to see if you can conceal your presence.",
        "The copper coin is the baseline unit of currency in most kingdoms. "
        "One gold piece is worth 100 copper pieces.",
    ),
    (
        "DE",
        "Welche Zustände gibt es in D&D und welche Auswirkungen haben sie?",
        "Verzweiflung (Exhaustion): Du erhältst einen Verzweiflungsgrad, der "
        "deinen Würfelabzug erhöht; bei sechs Grad stirbst du.",
        "Marvin aß ein Stück Kuchen und trank danach einen Becher Tee.",
    ),
]


def http_json(path: str, payload: dict | None = None, timeout: float = 120.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip() else {}


def main() -> int:
    print(f"Reranker smoke test → {BASE}")

    # 1. liveness
    try:
        http_json("/health", timeout=15)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"FAIL: /health unreachable ({exc})")
        return 1
    print("  /health: OK")

    failures = 0
    for label, query, relevant, irrelevant in CASES:
        try:
            data = http_json(
                "/rerank",
                {
                    "query": query,
                    "documents": [relevant, irrelevant],
                    "instruction": INSTRUCT,
                },
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            print(f"FAIL [{label}]: /rerank HTTP {exc.code}: {detail}")
            failures += 1
            continue
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"FAIL [{label}]: /rerank unreachable ({exc})")
            failures += 1
            continue

        # 2. shape the bot's jina parser requires
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or not results:
            print(f"FAIL [{label}]: response missing 'results' list: {str(data)[:200]}")
            failures += 1
            continue

        by_index = {int(r["index"]): float(r["relevance_score"]) for r in results}
        s_rel, s_irr = by_index.get(0), by_index.get(1)
        top = max(by_index, key=by_index.get)
        status = "OK " if top == 0 else "FAIL"
        verdict = "relevant ranked first" if top == 0 else "IRRELEVANT RANKED FIRST"
        print(f"  {status} [{label}] rel={s_rel:.3f} irr={s_irr:.3f} → {verdict}")
        if top != 0:
            failures += 1

    if failures:
        print(f"RESULT: {failures} problem(s) — see above")
        return 1
    print("RESULT: healthy — /rerank works end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
