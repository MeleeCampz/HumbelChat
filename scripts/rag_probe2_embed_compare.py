"""Compare full-precision local bge-m3 cosine scores against the GGUF-built index.

If local-vs-local scores are much higher than index scores for the same
(query, chunk) pairs, the quantized GGUF index vectors are degrading retrieval.
Read-only. Run inside the container: python - < this file
"""
from __future__ import annotations

import asyncio
import math
import os
import sys

sys.path.insert(0, "/app")
os.environ["RAG_QUERY_REWRITER"] = "0"

import config.settings as S  # noqa: E402
S.RAG_QUERY_REWRITER = False

from kb.chunker import Chunker  # noqa: E402
from kb.embedder import _get_local_model  # noqa: E402


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


async def main():
    model = _get_local_model()

    # Re-chunk equipment.md exactly like the index does, grab the interesting chunks.
    chunks = Chunker.split_file_sync("/app/data/knowledge/DnD5_5/equipment.md")
    by_section = {c.section_path: c.content for c in chunks}
    print("equipment.md sections:", list(by_section))

    targets = {}
    for sec in ("Preamble", "Armor", "Weapons"):
        if sec in by_section:
            targets[sec] = by_section[sec]

    queries = [
        "Gib mir die Stat block vons verschiedneen Rüstungen",
        "Was ist die Rüstungsklasse einer Kettenrüstung?",
        "Wie viel kostet eine Plattenrüstung?",
        "armor stat block",
        "What is the armor class of chain mail?",
    ]

    texts = queries + list(targets.values())
    vecs = model.encode(texts, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
    qv = {q: v for q, v in zip(queries, vecs[: len(queries)])}
    cv = {k: v for k, v in zip(targets.keys(), vecs[len(queries):])}

    print("\nFull-precision local bge-m3 (query vs chunk):")
    print(f"{'query':<52}", *[f"{k:>10}" for k in targets], sep="")
    for q in queries:
        row = [cos(qv[q], cv[k]) for k in targets]
        print(f"{q[:52]:<52}", *[f"{v:>10.3f}" for v in row], sep="")

    # Cross-check: index (GGUF-built) scores for the same pairs, from the live index.
    from kb.index import KBIndexStore  # noqa: E402
    store = KBIndexStore("/app/data/knowledge")
    await store.load()
    idx = store.get_index()

    print("\nIndex (GGUF-built) top-3 for the same queries:")
    for q in queries:
        ranked, _ = await idx.query_with_embeddings(q, top_n=3)
        print(f"  {q[:52]!r}:")
        for n, _c, s in ranked:
            print(f"      {s:.3f}  {n}")


if __name__ == "__main__":
    asyncio.run(main())
