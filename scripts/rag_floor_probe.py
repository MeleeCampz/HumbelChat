"""Rerank-score floor calibration probe (backlog #43 — 'measure first').

Runs the 33-query golden set through the REAL production retrieval path with
the Qwen3-Reranker active, captures every candidate's rerank score, and:

  1. prints per-query candidate tables (score + relevant/irrelevant),
  2. summarizes relevant vs irrelevant score distributions,
  3. sweeps a RERANK-score attach floor and re-simulates selection offline
     ("take chunks while score >= floor", existing per-file/global caps kept)
     to show hit rate / recall@k for each candidate floor — WITHOUT touching
     any production code or config.

Read-only: no index writes, no config changes. Run from the repo with a
production-equivalent environment (remote embedder + reranker reachable):

    EMBED_URL=http://host.docker.internal:11435/v1 \
    RERANK_API_BASE=http://host.docker.internal:8000 \
    BOT_NO_LOG_FILES=1 python scripts/rag_floor_probe.py [--top-n 4] [--out FILE]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The bot's main.py loads .env; standalone scripts must do it themselves or
# INFER_URL/INFER_API_KEY (rewrite LLM) silently fall back to defaults.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from rag_eval import QUERIES, _default_kb_path, stem_of  # noqa: E402


class Capture:
    """Holds the (name, content, rerank_score) candidates for the current query."""

    def __init__(self) -> None:
        self.ranked: list[tuple[str, str, float]] = []
        self.query: str = ""


CAP = Capture()


async def run(kb_path: Path, top_n: int, out_path: Path | None = None) -> list[dict]:
    import kb.retrievers as R

    original_select = R.select_ranked_chunks

    def spy_select(ranked, *args, **kwargs):
        CAP.ranked = list(ranked)
        return original_select(ranked, *args, **kwargs)

    R.select_ranked_chunks = spy_select
    try:
        from kb.retrievers import retrieve_kb_documents

        results: list[dict] = []
        for i, (query, expected) in enumerate(QUERIES):
            CAP.query = query
            CAP.ranked = []
            t0 = time.monotonic()
            print(f"[{i + 1}/{len(QUERIES)}] {query[:60]!r}", flush=True)
            try:
                docs = await asyncio.wait_for(
                    retrieve_kb_documents(query, kb_path, strategy="vector", top_n=top_n),
                    timeout=900,  # local CPU embedding fallback makes queries slow
                )
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 — one bad query must not kill the probe
                msg = f"{type(exc).__name__}: {exc}"
                print(f"    ERROR {msg}", flush=True)
                results.append({"query": query, "error": msg})
                continue
            attached = [stem_of(n) for n, _c in docs]
            hit_pos = next((i for i, s in enumerate(attached) if s in expected), None)
            print(f"    ok: hit={hit_pos is not None} attached={attached[:3]}", flush=True)
            # Candidate table: (name, rerank_score, relevant?) — top 8 only.
            cands = [
                {"name": stem_of(n), "score": round(s, 4),
                 "relevant": stem_of(n) in expected}
                for n, _c, s in CAP.ranked[:8]
            ]
            results.append({
                "query": query, "expected": expected, "attached": attached,
                "hit": hit_pos is not None, "hit_position": hit_pos,
                "candidates": cands, "elapsed_ms": round((time.monotonic() - t0) * 1000),
            })
            if out_path and (i + 1) % 5 == 0:  # periodic save — a crash must not lose hours
                try:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"queries": results}, f, indent=2, ensure_ascii=False)
                except OSError:
                    pass
    finally:
        R.select_ranked_chunks = original_select
    return results


def sweep(results: list[dict], top_n: int) -> list[dict]:
    """Re-simulate selection offline: keep only chunks with rerank score >= floor."""
    from kb.retrievers import select_ranked_chunks
    from config.settings import RAG_MAX_CHUNKS_PER_FILE

    rows = []
    for floor_bps in range(0, 96, 5):
        floor = floor_bps / 100.0
        n_hit = 0
        recall = {1: 0, 2: 0, 4: 0}
        n_ok = 0
        for r in results:
            if "error" in r or not r.get("candidates"):
                # Fall back to full candidate list when capture was empty.
                continue
            # Rebuild (name, content, score) from the captured table? We only kept
            # top-8 stems — so instead re-run selection on the FULL ranked list is
            # impossible here; approximate with the stored candidates (top 8), which
            # is where any floor in [0, .95] bites.
            filtered = [(c["name"], "", c["score"]) for c in r["candidates"] if c["score"] >= floor]
            docs = select_ranked_chunks(filtered, top_n=top_n,
                                        max_chunks_per_file=RAG_MAX_CHUNKS_PER_FILE)
            stems = [stem_of(n) for n, _c in docs]
            pos = next((i for i, s in enumerate(stems) if s in r["expected"]), None)
            n_ok += 1
            if pos is not None:
                n_hit += 1
                for k in (1, 2, 4):
                    if pos < k:
                        recall[k] += 1
        rows.append({
            "floor": round(floor, 2),
            "queries_scored": n_ok,
            "hit_rate": round(n_hit / max(n_ok, 1), 3),
            **{f"recall@{k}": round(v / max(n_ok, 1), 3) for k, v in recall.items()},
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kb-path", type=Path, default=None)
    ap.add_argument("--top-n", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("data/rag_floor_probe_results.json"))
    args = ap.parse_args()

    kb_path = args.kb_path or _default_kb_path()
    print(f"Floor probe — KB: {kb_path}  top_n={args.top_n}  queries={len(QUERIES)}")
    results = asyncio.run(run(kb_path, args.top_n, out_path=args.out or None))
    if args.out:  # incremental save BEFORE the (bug-prone) analysis stage
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"queries": results}, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    ok = [r for r in results if "error" not in r]
    baseline_hit = sum(r["hit"] for r in ok) / max(len(ok), 1)
    print(f"\nBASELINE (current production selection): hit rate {baseline_hit:.0%} over {len(ok)} queries")

    # Distributions.
    rel, irr = [], []
    for r in ok:
        for c in r["candidates"]:
            (rel if c["relevant"] else irr).append(c["score"])
    def pct(xs):
        if not xs:
            return "n/a"
        xs = sorted(xs)
        q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
        return f"min={xs[0]:.3f} med={q(0.5):.3f} p90={q(0.9):.3f} max={xs[-1]:.3f} (n={len(xs)})"
    print("\nRerank score distribution:")
    print(f"  relevant:     {pct(rel)}")
    print(f"  irrelevant:   {pct(irr)}")

    rows = sweep(ok, args.top_n)
    print("\nFloor sweep (score-driven selection, offline simulation):")
    print(f"  {'floor':>6} | {'hit':>6} | recall@1  @2  @4")
    for row in rows:
        print(f"  {row['floor']:6.2f} | {row['hit_rate']:6.0%} | {row['recall@1']:.0%}  "
              f"{row['recall@2']:.0%}  {row['recall@4']:.0%}")

    if args.out:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"baseline_hit_rate": round(baseline_hit, 3),
                           "distribution": {"relevant": rel, "irrelevant": irr},
                           "sweep": rows, "queries": results}, f, indent=2, ensure_ascii=False)
            print(f"\nresults → {args.out}")
        except OSError as exc:
            print(f"(could not write results: {exc})")


if __name__ == "__main__":
    main()
