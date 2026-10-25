"""In-depth RAG benchmark — full pipeline, single run, detailed report.

Runs the standing golden set (``scripts/rag_eval.py`` QUERIES) PLUS an extra
torture/edge batch through the REAL production entry point
(``kb.retrievers.retrieve_kb_documents``) and produces a per-query detail
table + aggregates:

  * hit / recall@k against expected file stems
  * per-stage detail parsed from live log records (dense top score, rewrite
    fired + expansion count + merge time, hybrid pool size, rerank pass top
    scores, attached file/char counts)
  * latency (median / p95), rerank success rate, DE/EN split

Design notes
------------
* **One process, back-to-back queries.** The LLM backend (:8888) shares a GPU
  with everything else and swaps/unloads models after idle — a warmup phase
  (one embedding + one rerank + one tiny completion) loads every model up
  front so the timed section runs without model-switch stalls. Do not fire
  /ai requests in Discord while this runs.
* **Read-only.** Uses the same SQLite index cache as the bot (read-only);
  never writes to it.
* Endpoint reachability: inside the bot container the compose DNS names from
  ``.env`` work as-is. From the dev sandbox, override the two compose-internal
  URLs with their published loopback ports, e.g.::

      EMBED_URL=http://host.docker.internal:11435/v1 \
      RERANK_API_BASE=http://host.docker.internal:8000 \
      python scripts/rag_benchmark.py

Usage:
    docker compose cp scripts/rag_benchmark.py bot:/tmp/ && \
        docker compose exec bot python /tmp/rag_benchmark.py
    # or locally / from the sandbox with the overrides above
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ─────────────────────────── extra cases (beyond the golden set) ───────────────────────────
# (query, acceptable file stems, note). Empty expected list = negative case:
# nothing from the KB should match; reported informationally, excluded from hit rate.
EXTRA_CASES: list[tuple[str, list[str], str]] = [
    # EN → German session notes (cross-lingual, property/event specific)
    ("What happened when Tempa tried to fly?", ["notes.md"], "EN→DE session event"),
    ("Tell me about the Black Pudding incident.", ["notes.md"], "EN→DE session event"),
    # DE, deliberately ambiguous cluster (multiple tavern sessions exist)
    ("Erzähl mir von der Nacht in der Taverne – wer war dabei und was ist passiert?",
     ["notes.md"], "DE ambiguous tavern"),
    # Property-specific item lookup, EN-named item in DE rules books
    ("Welches Item macht mich unsichtbar, wenn ich einen Angriff verfehle?",
     ["magic-items.md", "Dungeon Master's Guide.md"], "property lookup (Cloak of Displacement)"),
    # EN lore + monster baselines
    ("Who is Hath and what is his realm?", ["Hath.md"], "EN lore"),
    ("Gib mir den Statblock vom Schwarzen Drachen", ["monsters-A-Z.md"], "DE monster"),
    # Negatives — KB has no answer; we watch what the pipeline attaches anyway
    ("What is the capital of France?", [], "negative (world knowledge)"),
    ("Wie funktioniert die Steuererklärung in Deutschland?", [], "negative (real-world DE)"),
]

# German stopword heuristic for the DE/EN split in aggregates.
_DE_WORDS = re.compile(
    r"\b(was|wie|wer|welche|rüstung|gib|zeig|mir|dem|den|der|die|das|ein|eine|einen|"
    r"ist|sind|kann|können|viel|viele|funktioniert|erkläre|kostet|kosten|macht|machen|"
    r"item|zwei|drei|stufe|stufen|tabellen|tabelle|zauber|monster|session|passiert)\b", re.I)


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — sets only vars NOT already in the environment, so
    explicit overrides (sandbox endpoint URLs) always win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        # Inline comments: real dotenv strips `  # ...` from UNQUOTED values.
        # Without this, `RERANK_TOP_K=150  # unchanged...` loads as the whole
        # string, _safe_int falls back to the default (15), and every sandbox
        # benchmark silently ran with a 15-chunk rerank window instead of 150.
        if val.startswith('"') or val.startswith("'"):
            val = val.strip('"').strip("'")
        else:
            val = re.split(r"\s+#", val, maxsplit=1)[0].strip()
        if key and key not in os.environ:
            os.environ[key] = val


class _Capture(logging.Handler):
    """Collects formatted records from the retrieval loggers for one query."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        self.lines.append(record.getMessage())


def _lang(query: str) -> str:
    return "DE" if _DE_WORDS.search(query) else "EN"


async def _preflight() -> dict:
    """Check all three endpoints before spending a run; abort clearly on failure."""
    import httpx

    from config.settings import (EMBED_URL, INFER_API_KEY, INFER_URL,
                                 RERANK_API_BASE, RERANK_MODEL)
    out = {}
    async with httpx.AsyncClient(timeout=30) as c:
        for name, check in (
            ("embedder", lambda: c.get(f"{EMBED_URL}/models")),
            # Functional probe, NOT /health: the vLLM build serving the Qwen3
            # reranker answers /health with 500 while /rerank works fine.
            ("reranker", lambda: c.post(
                f"{RERANK_API_BASE}/rerank",
                json={"model": RERANK_MODEL, "query": "ping",
                      "documents": ["a", "b"]},
            )),
            ("llm", lambda: c.get(f"{INFER_URL}/models",
                                   headers={"Authorization": f"Bearer {INFER_API_KEY}"})),
        ):
            try:
                r = await check()
                r.raise_for_status()
                out[name] = "OK"
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"PREFLIGHT FAIL — {name}: {exc}")
    return out


async def _warmup() -> dict:
    """Load every model up front (one-time cost), so the timed section runs
    back-to-back without GPU model swaps/unloads."""
    import httpx

    from config.settings import EMBED_URL, INFER_API_KEY, INFER_URL, RERANK_API_BASE
    model = os.environ.get("MODEL_NAME", "")
    t = {}
    async with httpx.AsyncClient(timeout=300) as c:
        t0 = time.monotonic()
        await c.post(f"{EMBED_URL}/embeddings",
                     json={"model": "qwen3-embedding:4b", "input": ["warmup"]})
        t["embed_ms"] = round((time.monotonic() - t0) * 1000)
        t0 = time.monotonic()
        await c.post(f"{RERANK_API_BASE}/rerank",
                     json={"query": "warmup", "documents": ["one", "two"]})
        t["rerank_ms"] = round((time.monotonic() - t0) * 1000)
        t0 = time.monotonic()
        await c.post(f"{INFER_URL}/chat/completions",
                     headers={"Authorization": f"Bearer {INFER_API_KEY}"},
                     json={"model": model, "messages": [{"role": "user", "content": "Hi"}],
                           "max_tokens": 1})
        t["llm_ms"] = round((time.monotonic() - t0) * 1000)
    return t


def _parse_stages(lines: list[str]) -> dict:
    """Extract per-stage numbers from the captured log records."""
    st: dict = {}
    m = re.search(r"Vector scores for .*?: top=([\d.]+) median=([\d.]+) min=([\d.]+) \((\d+) chunks\)",
                  " ".join(lines))
    if m:
        st.update(dense_top=float(m.group(1)), dense_median=float(m.group(2)),
                  dense_pool=int(m.group(4)))
    m = re.search(r"Query rewrite triggered .*?added (\d+) expansion list\(s\) in ([\d.]+)s;"
                  r" RRF-merged to (\d+) chunks", " ".join(lines))
    if m:
        st.update(rewrite=True, expansions=int(m.group(1)),
                  rewrite_s=float(m.group(2)), rewritten_pool=int(m.group(3)))
    else:
        st["rewrite"] = False
    m = re.search(r"Hybrid retrieval: dense\((\d+)\) \+ lexical\((\d+) list\(s\), (\d+) chunk\(s\)\)"
                  r" → RRF-merged (\d+) chunk\(s\)", " ".join(lines))
    if m:
        st["hybrid_pool"] = int(m.group(4))
    m = re.search(r"Reranked top (\d+) candidate\(s\) against (\d+) quer", " ".join(lines))
    if m:
        st.update(rerank_window=int(m.group(1)), rerank_queries=int(m.group(2)))
    # kb.reranker DEBUG: one line per scoring pass (original + each expansion)
    tops = [float(x) for x in re.findall(r"Reranked \d+ candidate\(s\) -> \d+ kept in [\d.]+ ms;"
                                         r" top score=([\d.]+)", " ".join(lines))]
    if tops:
        st["rerank_passes_ok"] = len(tops)
        st["rerank_top_scores"] = [round(s, 3) for s in tops]
    if any("Rerank failed" in ln or "Rerank timed out" in ln for ln in lines):
        st["rerank_errors"] = sum(1 for ln in lines
                                  if "Rerank failed" in ln or "Rerank timed out" in ln)
    m = re.search(r"Vector retrieval: (\d+) ranked chunk\(s\) → (\d+) file\(s\) with ~(\d+) chars",
                  " ".join(lines))
    if m:
        st.update(attach_chunks=int(m.group(1)), attach_files=int(m.group(2)),
                  attach_chars=int(m.group(3)))
    return st


async def _run_case(retrieve, kb_path: Path, top_n: int, query: str) -> dict:
    cap = _Capture()
    loggers = [logging.getLogger("kb.retrievers"), logging.getLogger("kb.reranker")]
    for lg in loggers:
        lg.addHandler(cap)
        prev_level = lg.level
        lg.setLevel(logging.DEBUG)
    try:
        t0 = time.monotonic()
        docs = await retrieve(query, kb_path, strategy="vector", top_n=top_n)
        dt_ms = round((time.monotonic() - t0) * 1000)
    except Exception as exc:  # noqa: BLE001 — one bad query must not kill the run
        dt_ms = round((time.monotonic() - t0) * 1000)
        return {"query": query, "error": f"{type(exc).__name__}: {exc}", "elapsed_ms": dt_ms}
    finally:
        for lg in loggers:
            lg.removeHandler(cap)
            lg.setLevel(prev_level)
    stems = [n.split(" [")[0].rsplit("/", 1)[-1] for n, _ in docs]
    return {"query": query, "elapsed_ms": dt_ms, "attached_files": stems,
            "stages": _parse_stages(cap.lines)}


def main() -> None:
    ap = argparse.ArgumentParser(description="In-depth RAG benchmark (single run)")
    ap.add_argument("--kb-path", type=Path, default=None)
    ap.add_argument("--top-n", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N cases (0 = all); use with an explicit --out "
                         "so the full results file is not clobbered")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "rag_benchmark_results.json")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "scripts"))
    import _sandbox_shim  # noqa: F401 — endpoint shims BEFORE .env load (sandbox only)
    _load_dotenv(REPO / ".env")  # sets only vars NOT already in the environment
    for cand in (Path("/app"), REPO):
        if (cand / "kb").exists():
            sys.path.insert(0, str(cand))
            break
    from kb.retrievers import retrieve_kb_documents  # noqa: E402  (after .env)

    kb_path = args.kb_path or REPO / "data" / "knowledge"
    cases: list[tuple[str, list[str], str]] = [(q, e, "golden") for q, e in _golden()] + EXTRA_CASES
    if args.limit > 0:
        cases = cases[:args.limit]
        if args.out == REPO / "data" / "rag_benchmark_results.json":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
            args.out = REPO / "data" / f"rag_benchmark_small_{args.limit}_{stamp}.json"

    print(f"RAG benchmark — {len(cases)} cases, KB={kb_path}, top_n={args.top_n}")
    print(f"started {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    pre = asyncio.run(_preflight())
    for k, v in pre.items():
        print(f"  {k}: {v}")
    warm = asyncio.run(_warmup())
    print(f"  warmup: embed {warm['embed_ms']} ms · rerank {warm['rerank_ms']} ms · "
          f"llm {warm['llm_ms']} ms (first call loads the model)")

    results = asyncio.run(_run_all(retrieve_kb_documents, kb_path, args.top_n, cases))

    _report(results, warm)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "top_n": args.top_n, "warmup": warm, "cases": results,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"results → {args.out}")


def _golden() -> list[tuple[str, list[str]]]:
    sys.path.insert(0, str(REPO / "scripts"))
    from rag_eval import QUERIES  # noqa: E402
    return QUERIES


async def _run_all(retrieve, kb_path: Path, top_n: int,
                   cases: list[tuple[str, list[str], str]]) -> list[dict]:
    out = []
    for i, (q, expected, tag) in enumerate(cases, 1):
        r = await _run_case(retrieve, kb_path, top_n, q)
        r["expected"] = expected
        r["tag"] = tag
        r["lang"] = _lang(q)
        if "error" not in r:
            pos = next((j for j, s in enumerate(r["attached_files"]) if s in expected), None) \
                if expected else None
            r["hit"] = pos is not None if expected else None  # None = negative case
            r["hit_position"] = pos
        out.append(r)
        mark = ("HIT@" + str(pos)) if r.get("hit") else ("MISS" if expected else "neg")
        top = (r.get("stages", {}).get("rerank_top_scores") or ["-"])[0]
        print(f"  [{i:2d}/{len(cases)}] {mark:<5} {r['elapsed_ms']:>6} ms  rr_top={top}  "
              f"{q[:70]}", flush=True)
    return out


def _report(results: list[dict], warm: dict) -> None:
    ok = [r for r in results if "error" not in r]
    scored = [r for r in ok if r.get("hit") is not None]
    negatives = [r for r in ok if r.get("hit") is None]
    n_hit = sum(1 for r in scored if r["hit"])

    print(f"\n{'=' * 100}")
    print(f"BENCHMARK RESULT: {n_hit}/{len(scored)} expected-file hits "
          f"({len(negatives)} negative cases, {sum('error' in r for r in results)} errors)")

    def rate(sub: list[dict]) -> str:
        h = sum(1 for r in sub if r["hit"])
        return f"{h}/{len(sub)}"

    de = [r for r in scored if r["lang"] == "DE"]
    en = [r for r in scored if r["lang"] == "EN"]
    print(f"  by language:      DE {rate(de)} · EN {rate(en)}")
    for k in (1, 2, 4):
        rec = sum(1 for r in scored if r["hit_position"] is not None and r["hit_position"] < k)
        print(f"  recall@{k}:          {rec}/{len(scored)} ({rec / max(len(scored), 1):.0%})")
    ms = [r["elapsed_ms"] for r in ok]
    p95 = sorted(ms)[min(len(ms) - 1, int(0.95 * len(ms)))]
    print(f"  latency:          median {statistics.median(ms)} ms · p95 {p95} ms")
    rr_ok = sum(r.get("stages", {}).get("rerank_passes_ok", 0) for r in ok)
    rr_err = sum(r.get("stages", {}).get("rerank_errors", 0) for r in ok)
    rw = sum(1 for r in ok if r.get("stages", {}).get("rewrite"))
    print(f"  rewrite fired:    {rw}/{len(ok)} · rerank passes OK: {rr_ok} · rerank errors: {rr_err}")

    tops = [max(s) for r in scored if r["hit"]
            for s in [r.get("stages", {}).get("rerank_top_scores") or []] if s]
    if tops:
        print(f"  hit top-rerank:   median {statistics.median(tops):.3f} "
              f"(min {min(tops):.3f}, max {max(tops):.3f}) — best pass's top score per hit")

    print(f"\n{'-' * 100}")
    print("PER-QUERY DETAIL")
    hdr = (f"{'#':>2} {'lang':<4} {'result':<6} {'ms':>6} {'d_top':>6} {'rew':>3} "
           f"{'pool':>5} {'rr_tops':<16} attached")
    print(hdr)
    for i, r in enumerate(ok, 1):
        st = r.get("stages", {})
        res = ("HIT@" + str(r["hit_position"])) if r.get("hit") else \
              ("MISS" if r.get("hit") is not None else "neg  ")
        tops = ",".join(str(s) for s in st.get("rerank_top_scores") or []) or "-"
        print(f"{i:>2} {r['lang']:<4} {res:<6} {r['elapsed_ms']:>6} "
              f"{st.get('dense_top', float('nan')):>6.3f} "
              f"{'Y' if st.get('rewrite') else 'n':>3} {st.get('hybrid_pool', '-'):>5} "
              f"{tops:<16} {', '.join(r['attached_files'][:4]) or '(none)'}")

    bad = [r for r in scored if not r["hit"]] + [r for r in results if "error" in r]
    if bad:
        print(f"\n{'-' * 100}")
        print("MISSES / ERRORS (expected vs attached)")
        for r in bad:
            if "error" in r:
                print(f"  ERROR {r['query'][:60]!r} → {r['error']}")
            else:
                print(f"  MISS  {r['query'][:60]!r}\n"
                      f"        expected: {', '.join(r['expected'])}\n"
                      f"        attached: {', '.join(r['attached_files']) or '(none)'}")
    if negatives:
        print("\nNEGATIVE CASES (informational — what got attached to unanswerable queries)")
        for r in negatives:
            print(f"  {r['query'][:60]!r} → {', '.join(r['attached_files']) or '(none)'} "
                  f"[{r.get('stages', {}).get('attach_chars', '?')} chars]")


if __name__ == "__main__":
    main()
