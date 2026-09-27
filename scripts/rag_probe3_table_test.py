"""Prototype: does converting the Armor HTML table to plain text improve retrieval?

Embeds (a) original Armor chunk, (b) plain-text Armor chunk, (c) Preamble chunk
and scores them against German + English armor queries with full-precision bge-m3.
Run inside container: python - < this file
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, "/app")
os.environ["RAG_QUERY_REWRITER"] = "0"

import config.settings as S  # noqa: E402
S.RAG_QUERY_REWRITER = False

from kb.chunker import Chunker  # noqa: E402
from kb.embedder import _get_local_model  # noqa: E402


def table_to_text(html_table: str) -> str:
    """Convert an HTML <table> to plain text rows with a column legend."""
    headers = re.findall(r"<th[^>]*>(.*?)</th>", html_table, re.S)
    headers = [re.sub(r"<[^>]+>", "", h).strip() for h in headers]
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_table, re.S)
    lines = []
    if headers:
        lines.append("Columns: " + ", ".join(headers))
    for row in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        cells = [c for c in cells if c]
        if not cells:
            continue
        # skip category separator rows (single cell spanning all columns)
        colspan = re.search(r'colspan="\d"', row)
        if colspan and len(cells) == 1:
            lines.append(f"[{cells[0]}]")
            continue
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def convert_tables_in_section(section_text: str) -> str:
    def repl(m):
        return table_to_text(m.group(0))
    return re.sub(r"<table>.*?</table>", repl, section_text, flags=re.S)


def cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def main():
    chunks = Chunker.split_file_sync("/app/data/knowledge/DnD5_5/equipment.md")
    by_sec = {c.section_path: c.content for c in chunks}
    armor_orig = by_sec["Armor"]
    armor_pt = convert_tables_in_section(armor_orig)
    preamble = by_sec["Preamble"]

    print("── plain-text Armor chunk (first 500 chars) ──")
    print(armor_pt[:500])
    print(f"\norig len={len(armor_orig)} → plaintext len={len(armor_pt)}")

    queries = [
        "Gib mir die Stat block vons verschiedneen Rüstungen",
        "Was ist die Rüstungsklasse einer Kettenrüstung?",
        "Wie viel kostet eine Plattenrüstung?",
        "armor stat block",
        "What is the armor class of chain mail?",
    ]
    model = _get_local_model()
    texts = queries + [armor_orig, armor_pt, preamble]
    vecs = model.encode(texts, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
    qv = {q: v for q, v in zip(queries, vecs[:5])}
    a_o, a_p, pre = vecs[5], vecs[6], vecs[7]

    print(f"\n{'query':<52} {'ArmorHTML':>10} {'ArmorText':>10} {'Preamble':>10}")
    for q in queries:
        print(f"{q[:52]:<52} {cos(qv[q], a_o):>10.3f} {cos(qv[q], a_p):>10.3f} {cos(qv[q], pre):>10.3f}")


if __name__ == "__main__":
    main()
