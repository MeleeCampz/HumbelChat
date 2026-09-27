"""Test retrieval variants for cross-lingual (German query → English KB) lookups.

Variants:
  A) per-row chunk with section context ("## Armor\nColumns: ...\nChain Mail | 16 | ...")
  B) plain-text full Armor table chunk
  C) simulated rewriter: English expanded query vs the chunks above
Run inside container: python - < this file
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, "/app")
os.environ["RAG_QUERY_REWRITER"] = "0"

import config.settings as S  # noqa: E402
S.RAG_QUERY_REWRITER = False

from kb.embedder import _get_local_model  # noqa: E402


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


ARMOR_PT = """## Armor
Columns: Armor, Armor Class (AC), Strength, Stealth, Weight, Cost
[Light Armor (1 Minute to Don or Doff)]
Padded Armor | 11 + Dex modifier | — | Disadvantage | 4 lb. | 5 GP
Leather Armor | 11 + Dex modifier | — | — | 10 lb. | 10 GP
Studded Leather | 12 + Dex modifier | — | — | 13 lb. | 45 GP
[Medium Armor (5 Minutes to Don and 1 Minute to Doff)]
Hide Armor | 12 + Dex (max 2) | — | Disadvantage | 10 lb. | 10 GP
Chain Mail | 16 | — | Disadvantage | 55 lb. | 75 GP
Scale Mail | 14 + Dex (max 2) | Str 13 | Disadvantage | 45 lb. | 50 GP
Breastplate | 14 + Dex (max 2) | — | — | 20 lb. | 400 GP
Half Plate | 15 + Dex (max 2) | Str 15 | Disadvantage | 40 lb. | 750 GP
[Heavy Armor (10 Minutes to Don and 5 Minutes to Doff)]
Ring Mail | 14 | — | Disadvantage | 40 lb. | 30 GP
Splint Armor | 17 | Str 15 | Disadvantage | 60 lb. | 200 GP
Plate Armor | 18 | Str 15 | Disadvantage | 65 lb. | 1,500 GP"""

CHAIN_MAIL_ROW = """## Armor
Columns: Armor, Armor Class (AC), Strength, Stealth, Weight, Cost
Chain Mail | 16 | — | Disadvantage | 55 lb. | 75 GP"""

PLATE_ROW = """## Armor
Columns: Armor, Armor Class (AC), Strength, Stealth, Weight, Cost
Plate Armor | 18 | Str 15 | Disadvantage | 65 lb. | 1,500 GP"""

PREAMBLE = """# Equipment
This section describes the equipment available to characters, including coins, weapons, armor, tools, and adventuring gear. Prices are listed in gold pieces (GP), silver pieces (SP), and copper pieces (CP)."""

QUERIES = {
    "DE: Gib mir die Stat block vons verschiedneen Rüstungen":
        "Gib mir die Stat block vons verschiedneen Rüstungen",
    "DE: Was ist die Rüstungsklasse einer Kettenrüstung?":
        "Was ist die Rüstungsklasse einer Kettenrüstung?",
    "DE: Wie viel kostet eine Plattenrüstung?":
        "Wie viel kostet eine Plattenrüstung?",
    "EN: armor stat block": "armor stat block",
    "EN: What is the armor class of chain mail?": "What is the armor class of chain mail?",
    # Simulated rewriter output (what a good LLM expansion would produce):
    "REWRITE: armor table D&D 5e, armor class AC cost weight of armors":
        "armor table D&D 5e, armor class AC cost weight of armors",
    "REWRITE: chain mail armor class price medium armor":
        "chain mail armor class price medium armor",
}

CHUNKS = {
    "ArmorTable_PT": ARMOR_PT,
    "ChainMail_row": CHAIN_MAIL_ROW,
    "Plate_row": PLATE_ROW,
    "Preamble": PREAMBLE,
}


def main():
    model = _get_local_model()
    qtexts = list(QUERIES.values())
    ctexts = list(CHUNKS.values())
    vecs = model.encode(qtexts + ctexts, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
    qv = {q: v for q, v in zip(qtexts, vecs[: len(qtexts)])}
    cv = {k: v for k, v in zip(CHUNKS.keys(), vecs[len(qtexts):])}

    hdr = f"{'query':<48}" + "".join(f"{k:>16}" for k in CHUNKS)
    print(hdr)
    for label, q in QUERIES.items():
        row = "  ".join(f"{cos(qv[q], cv[k]):>15.3f}" for k in CHUNKS)
        print(f"{label[:48]:<48}{row}")


if __name__ == "__main__":
    main()
