"""RAG retrieval probe — run German + English game queries through the real pipeline.

Usage (inside the bot container):  python /app/scripts/rag_probe.py
Or pipe via stdin:                 docker compose exec -T bot python - < scripts/rag_probe.py

Read-only against the vector index. Disables the LLM query rewriter so results
are deterministic and no chat-model call is made. Writes a JSON report to
/app/data/rag_probe_results_after.json (bind-mounted → host data/).
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

sys.path.insert(0, "/app")
pass  # keep the LIVE rewriter enabled (after-fix validation)

from kb.index import KBIndexStore  # noqa: E402
from kb.retrievers import (  # noqa: E402
    _lexical_ranking,
    reciprocal_rank_fusion,
    select_ranked_chunks,
)

KB_PATH = "/app/data/knowledge"
OUT = "/app/data/rag_probe_results_after.json"

# (query, acceptable file stems — first entry is the *primary* expectation)
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


def stem_of(name: str) -> str:
    return name.split(" [")[0].rsplit("/", 1)[-1] if " [" in name else name.rsplit("/", 1)[-1]


async def main() -> None:
    store = KBIndexStore(KB_PATH)
    await store.load()
    idx = store.get_index()
    assert idx is not None and not idx.is_empty(), "index failed to load"
    print(f"Index loaded: {idx.count()} chunks")

    from config.settings import (
        RAG_DENSE_TOP_K, RAG_LEXICAL_TOP_K, RAG_MAX_DOCS,
        RAG_MAX_CHUNKS_PER_FILE, RAG_ATTACH_FLOOR, RAG_MIN_ATTACH_SCORE,
    )
    print(f"settings: dense_top_k={RAG_DENSE_TOP_K} lexical_top_k={RAG_LEXICAL_TOP_K} "
          f"max_docs={RAG_MAX_DOCS} chunks/file={RAG_MAX_CHUNKS_PER_FILE} "
          f"floor={RAG_ATTACH_FLOOR} min_attach={RAG_MIN_ATTACH_SCORE}")

    results = []
    for q, expected in QUERIES:
        t0 = time.monotonic()
        ranked, _ = await idx.query_with_embeddings(q, top_n=RAG_DENSE_TOP_K)
        if not ranked:
            results.append({"query": q, "expected": expected, "error": "no dense hits"})
            continue

        # min-attach floor (same as pipeline)
        kept = [t for t in ranked if t[2] >= RAG_MIN_ATTACH_SCORE] if RAG_MIN_ATTACH_SCORE > 0 else list(ranked)
        dense_scores = {n: float(s) for n, _c, s in kept}

        lex = await asyncio.to_thread(_lexical_ranking, idx, q, RAG_LEXICAL_TOP_K)
        hybrid = reciprocal_rank_fusion([kept, lex]) if lex else kept

        docs = select_ranked_chunks(
            hybrid, top_n=RAG_MAX_DOCS,
            max_chunks_per_file=RAG_MAX_CHUNKS_PER_FILE,
            attach_min_score=RAG_ATTACH_FLOOR, dense_scores=dense_scores,
        )
        dt = time.monotonic() - t0

        doc_stems = [stem_of(n) for n, _c in docs]
        hit_pos = next((i for i, s in enumerate(doc_stems) if s in expected), None)
        dense_top5 = [(n, round(s, 3)) for n, _c, s in ranked[:5]]
        lex_top3 = [(n, round(s, 2)) for n, _c, s in lex[:3]]

        results.append({
            "query": q,
            "expected": expected[0],
            "acceptable": expected,
            "hit": hit_pos is not None,
            "hit_position": hit_pos,
            "dense_top1": dense_top5[0][0] if dense_top5 else None,
            "dense_top1_score": dense_top5[0][1] if dense_top5 else None,
            "dense_top5": dense_top5,
            "lex_top3": lex_top3,
            "attached_files": doc_stems,
            "elapsed_ms": round(dt * 1000),
        })

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Console report ──
    print(f"\n{'PASS' if True else ''} {'#':>2}  {'q':<52} {'top1_score':>10}  attached")
    n_hit = 0
    for i, r in enumerate(results, 1):
        if "error" in r:
            print(f"ERR {i:>2}  {r['query'][:52]:<52} {r['error']}")
            continue
        mark = "OK " if r["hit"] else "MISS"
        n_hit += r["hit"]
        top1 = (r["dense_top1"] or "")[:40]
        print(f"{mark} {i:>2}  {r['query'][:52]:<52} {r['dense_top1_score']:>8.3f}  "
              f"top1={top1} → {','.join(r['attached_files'][:4])}")
    print(f"\n{ n_hit}/{len(results)} queries attached an expected file")
    print(f"report written to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
