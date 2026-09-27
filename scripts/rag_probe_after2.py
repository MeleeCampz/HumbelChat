"""After-fix validation: run the SAME 33 queries through retrieve_kb_documents()
(the exact production entry point, incl. the German→LLM-rewrite path).
Run inside container: python - < this file
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

sys.path.insert(0, "/app")

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

KB_PATH = "/app/data/knowledge"
OUT = "/app/data/rag_probe_results_after.json"


def stem_of(name: str) -> str:
    return name.split(" [")[0].rsplit("/", 1)[-1] if " [" in name else name.rsplit("/", 1)[-1]


async def main() -> None:
    from kb.retrievers import retrieve_kb_documents

    results = []
    for q, expected in QUERIES:
        t0 = time.monotonic()
        try:
            docs = await retrieve_kb_documents(q, KB_PATH, strategy="vector", top_n=4)
        except Exception as exc:  # noqa: BLE001
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

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_hit = sum(r.get("hit", False) for r in results)
    print("\n{0:>2}  {1:<52} {2:>6}  attached".format("#", "q", "ms"))
    for i, r in enumerate(results, 1):
        if "error" in r:
            print(f"ERR {i:>2}  {r['query'][:52]:<52} {r['error']}")
            continue
        mark = "OK  " if r["hit"] else "MISS"
        print(f"{mark}{i:>2}  {r['query'][:52]:<52} {r['elapsed_ms']:>6}  {','.join(r['attached_files'][:4])}")
    print("\n%d/%d queries attached an expected file (full production pipeline)" % (n_hit, len(results)))


if __name__ == "__main__":
    asyncio.run(main())
