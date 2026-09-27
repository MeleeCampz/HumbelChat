"""One-off debug: reproduce the 'Kina Brightspark' retrieval against the live index.

Reads a COPY of the vector-index cache (never the live one) and prints where
the chunks that actually contain 'Kina Brightspark' rank in the lexical (BM25)
leg, the dense leg, and the RRF fusion — for the exact query the user asked.
"""

import asyncio
import pathlib
import shutil
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv

load_dotenv()  # main.py does this; without it the embedding backend defaults change

QUERY = "Ihr name is Kina Brightspark oder nicht?"


async def main() -> None:
    kb_path = REPO / "data" / "knowledge"
    live_cache = kb_path / ".vector_index_cache"
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="kdb-"))
    shutil.copytree(live_cache, tmp / "cache")

    from kb.index import KBIndexStore
    from kb.retrievers import _lexical_ranking, reciprocal_rank_fusion, select_ranked_chunks

    store = KBIndexStore(kb_path, persist_dir=tmp / "cache")
    idx = await store.load()
    print(f"loaded {len(idx._docs)} chunks")

    # 1) Which indexed chunks contain the name at all?
    kina_chunks = [d for d in idx._docs if "Kina" in d.content]
    print(f"\nchunks containing 'Kina': {len(kina_chunks)}")
    for d in kina_chunks:
        print(f"  - {d.retrieval_name()}  ({len(d.content)} chars)")

    # 2) Lexical (BM25) ranking
    lex = await asyncio.to_thread(_lexical_ranking, idx, QUERY, 48)
    print(f"\nLEXICAL top-15 of {len(lex)}:")
    for i, (name, content, s) in enumerate(lex[:15], 1):
        mark = " <== KINA" if "Kina" in content else ""
        print(f"  {i:2d}. {s:8.3f}  {name}{mark}")
    for i, (name, content, s) in enumerate(lex, 1):
        if "Kina" in content:
            print(f"  ... Kina chunk at lexical rank {i} (score {s:.3f})")

    # 3) Dense ranking
    ranked, _ = await idx.query_with_embeddings(QUERY, top_n=48)
    print(f"\nDENSE top-10 of {len(ranked)}:")
    for i, (name, content, s) in enumerate(ranked[:10], 1):
        mark = " <== KINA" if "Kina" in content else ""
        print(f"  {i:2d}. {s:8.3f}  {name}{mark}")
    for i, (name, content, s) in enumerate(ranked, 1):
        if "Kina" in content:
            print(f"  ... Kina chunk at dense rank {i} (score {s:.3f})")

    # 4) RRF fusion — exactly like the bot: dense leg filtered by
    #    RAG_MIN_ATTACH_SCORE (0.35 in .env) BEFORE fusion.
    from config.settings import RAG_MIN_ATTACH_SCORE, RAG_ATTACH_FLOOR, RAG_MAX_DOCS, RAG_MAX_CHUNKS_PER_FILE
    print(f"\nsettings: MIN_ATTACH={RAG_MIN_ATTACH_SCORE} ATTACH_FLOOR={RAG_ATTACH_FLOOR} MAX_DOCS={RAG_MAX_DOCS}")
    dense_kept = [t for t in ranked if t[2] >= RAG_MIN_ATTACH_SCORE]
    fused = reciprocal_rank_fusion([dense_kept, lex])
    print(f"\nRRF (bot-exact: dense{len(dense_kept)}+lex{len(lex)}) top-12 of {len(fused)}:")
    for i, (name, content, s) in enumerate(fused[:12], 1):
        mark = " <== KINA" if "Kina" in content else ""
        print(f"  {i:2d}. {s:8.4f}  {name}{mark}")

    # 5) Final selection with the bot's RAG_ATTACH_FLOOR — and without it.
    dense_scores = {name: float(s) for name, _c, s in ranked}
    for floor in (RAG_ATTACH_FLOOR, 0.35, 0.0):
        docs = select_ranked_chunks(
            fused, top_n=RAG_MAX_DOCS, max_chunks_per_file=RAG_MAX_CHUNKS_PER_FILE,
            attach_min_score=floor, dense_scores=dense_scores,
        )
        names = [d[0].split("/")[-2] + "/" + d[0].split("/")[-1] for d in docs]
        kina = any("Kina" in c for _n, c in docs)
        print(f"\nselect_ranked_chunks(floor={floor}) -> {len(docs)} file(s), Kina attached: {kina}")
        for n in names:
            print(f"   - {n}")

    await store.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
