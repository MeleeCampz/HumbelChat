"""Standing RAG evaluation harness.

Runs the 33-query golden set (same queries as ``scripts/rag_probe_after2.py``)
through the REAL production entry point (``kb.retrievers.retrieve_kb_documents``)
and prints one comparable score you can track across changes (reranker on/off,
attach floors, chunk sizes, embedding model switches):

    * hit rate        — expected file attached at all (the probe's metric)
    * recall@1/@2/@4  — expected file in the top-k attachments
    * latency         — median / p95 per query

The golden set is EMBEDDED below so the script is fully self-contained and can
be piped into the container without mounting anything. Keep it in sync with
``scripts/rag_probe_after2.py`` when you add queries (or just extend THIS list
— this harness is now the standing tool).

Usage (live container — read-only, does NOT touch the running bot's index):
    docker compose exec -T bot python - < scripts/rag_eval.py        # cmd/pwsh: use Get-Content ... |
    docker compose cp scripts/rag_eval.py bot:/tmp/rag_eval.py && docker compose exec bot python /tmp/rag_eval.py

Usage (local dev):
    .venv/bin/python scripts/rag_eval.py [--kb-path DIR] [--top-n 4] [--out FILE]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

# ─────────────────────────── Golden set ────────────────────────────
# (query, acceptable file stems — a hit = ANY of them attached).
# Mirrors scripts/rag_probe_after2.py QUERIES.
QUERIES: list[tuple[str, list[str]]] = [
    # ── The real failing query + armor variants (DE) ──
    ("Gib mir die Stat block vons verschiedneen Rüstungen", ["equipment.md"]),
    ("Was ist die Rüstungsklasse einer Kettenrüstung?", ["equipment.md"]),
    ("Wie viel kostet eine Plattenrüstung?", ["equipment.md"]),
    ("Zeig mir die Waffentabelle mit den Preisen", ["equipment.md"]),
    ("Was kostet ein Heiltrank?", ["equipment.md"]),
    ("Welche Werkzeuge gibt es und was kosten sie?", ["equipment.md"]),
    # ── Spells (DE) ──
    ("Was macht der Zauberspruch Feuerball?", ["spells.md"]),
    ("Gib mir alle Heilzauber bis Stufe 3", ["spells.md"]),
    ("Wie funktioniert Konzentration beim Zaubern?", ["playing-the-game.md", "rules-glossary.md", "spells.md"]),
    # ── Classes (DE) ──
    ("Was kann ein Krieger in Stufe 5?", ["classes.md"]),
    ("Zeig mir die Magier Tabelle", ["classes.md"]),
    ("Wie viele Zauber kennt ein Magier in Stufe 1?", ["classes.md"]),
    # ── Monsters (DE) ──
    ("Gib mir den Statblock vom Goblin", ["monsters-A-Z.md"]),
    ("Statblock von einem jungen Drachen", ["monsters-A-Z.md"]),
    ("Welche Monster leben in der Wüste?", ["monsters-A-Z.md", "animals.md", "monsters.md"]),
    # ── Rules (DE) ──
    ("Wie funktioniert Initiative im Kampf?", ["playing-the-game.md", "rules-glossary.md"]),
    ("Erkläre die Regeln für Heimlichkeit", ["playing-the-game.md", "rules-glossary.md"]),
    ("Was passiert bei einem kritischen Treffer?", ["playing-the-game.md", "rules-glossary.md"]),
    ("Wie funktionieren Sterben und Todeswürfe?", ["playing-the-game.md", "rules-glossary.md"]),
    # ── Character creation / feats (DE) ──
    ("Wie erstelle ich einen neuen Charakter?", ["character-creation.md"]),
    ("Welche Talente kann ein Krieger wählen?", ["feats.md", "classes.md"]),
    ("Wie würfle ich die Werte für Stärke und Geschicklichkeit?", ["character-creation.md", "rules-glossary.md"]),
    # ── Humblewood lore (DE) ──
    ("Wie ist der Kalender in Humblewood aufgebaut?", ["Humblewood_Calendar.md"]),
    ("Wer ist Hath und was ist sein Reich?", ["Hath.md"]),
    ("Was kann die Spezies Vulpin?", ["Vulpin.md", "species.md"]),
    ("Welche besonderen Zaubersprüche gibt es in Humblewood?",
     ["Ambush_Prey.md", "Elevated_Sight.md", "Feathered_Reach.md", "Globe_of_Twilight.md",
      "Gust_Barrier.md", "Invoke_the_Amaranthine.md", "Mend_Plants.md"]),
    ("Welche Hintergründe gibt es in Humblewood?",
     ["Bandit_Defector.md", "Grounded.md", "Wind-touched.md"]),
    # ── Session notes (DE) ──
    ("Was ist in der letzten Session bei Alderheart passiert?", ["notes.md"]),
    # ── English baselines ──
    ("armor stat block", ["equipment.md"]),
    ("fireball spell", ["spells.md"]),
    ("goblin statblock", ["monsters-A-Z.md"]),
    ("healing potion price", ["equipment.md"]),
    ("wizard class table", ["classes.md"]),
]


def _default_kb_path() -> Path:
    """KB_PATH env var (container) → repo data/knowledge (local dev)."""
    env = os.getenv("KB_PATH")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            return p
    # Container layout fallback, then local repo layout (works when the script
    # is piped via stdin — __file__ is '<stdin>' there, so only use it as a hint).
    hints: list[Path] = [Path("/app/data/knowledge")]
    try:
        hints.append(Path(__file__).resolve().parent.parent / "data" / "knowledge")
    except NameError:
        pass
    for cand in hints:
        if cand.exists():
            return cand
    raise SystemExit("No KB path found — pass --kb-path or set KB_PATH")


def stem_of(name: str) -> str:
    """Strip section suffix + directory so 'equipment.md [Armor]' → 'equipment.md'."""
    return name.split(" [")[0].rsplit("/", 1)[-1] if " [" in name else name.rsplit("/", 1)[-1]


async def run_eval(kb_path: Path, top_n: int) -> list[dict]:
    # Make the repo importable when run from anywhere (local or /app container).
    for cand in (Path("/app"), Path.cwd()):
        if (cand / "kb").exists():
            sys.path.insert(0, str(cand))
            break
    try:
        local_repo = Path(__file__).resolve().parent.parent
        if (local_repo / "kb").exists() and str(local_repo) not in sys.path:
            sys.path.insert(0, str(local_repo))
    except NameError:
        pass  # stdin execution — no __file__

    from kb.retrievers import retrieve_kb_documents

    results: list[dict] = []
    for q, expected in QUERIES:
        t0 = time.monotonic()
        try:
            docs = await retrieve_kb_documents(q, kb_path, strategy="vector", top_n=top_n)
        except Exception as exc:  # noqa: BLE001 — one bad query must not kill the eval
            results.append({"query": q, "expected": expected[0], "error": str(exc)})
            continue
        dt = time.monotonic() - t0
        doc_stems = [stem_of(n) for n, _c in docs]
        hit_pos = next((i for i, s in enumerate(doc_stems) if s in expected), None)
        results.append({
            "query": q, "expected": expected[0], "acceptable": expected,
            "hit": hit_pos is not None, "hit_position": hit_pos,
            "attached_files": doc_stems, "elapsed_ms": round(dt * 1000),
        })
    return results


def summarize(results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    n_hit = sum(r["hit"] for r in ok)
    print(f"\n{'=' * 78}")
    print(f"RAG eval: {n_hit}/{len(ok)} queries attached an expected file"
          + (f" ({sum('error' in r for r in results)} error(s))" if len(ok) != len(results) else ""))

    for k in (1, 2, 4):
        recall = sum(1 for r in ok if r["hit_position"] is not None and r["hit_position"] < k) / max(len(ok), 1)
        print(f"  recall@{k}: {recall:.0%}")

    ms = [r["elapsed_ms"] for r in ok]
    if ms:
        p95 = sorted(ms)[min(len(ms) - 1, int(0.95 * len(ms)))]
        print(f"  latency: median {statistics.median(ms)} ms, p95 {p95} ms")

    misses = [r for r in ok if not r["hit"]]
    if misses:
        print("  misses:")
        for r in misses:
            print(f"    - {r['query']!r} (expected {','.join(r['acceptable'])[:60]}; "
                  f"got {','.join(r['attached_files'][:4]) or 'nothing'})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Standing RAG evaluation harness (33-query golden set)")
    ap.add_argument("--kb-path", type=Path, default=None, help="KB root (default: KB_PATH env / repo data/knowledge)")
    ap.add_argument("--top-n", type=int, default=4, help="documents per query (default 4 = RAG_MAX_DOCS)")
    ap.add_argument("--out", type=Path, default=Path("data/rag_eval_results.json"),
                    help="where to write the JSON results (pass '' to skip)")
    args = ap.parse_args()

    kb_path = args.kb_path or _default_kb_path()
    print(f"RAG eval — KB: {kb_path}  top_n={args.top_n}")
    results = asyncio.run(run_eval(kb_path, args.top_n))

    if args.out:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"results → {args.out}")
        except OSError as exc:
            print(f"(could not write results: {exc})")

    summarize(results)


if __name__ == "__main__":
    main()
