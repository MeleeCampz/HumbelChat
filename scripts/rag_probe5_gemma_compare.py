"""#24 eval — compare BAAI/bge-m3 vs EmbeddingGemma 2 (text-only 270M / full 740M).

Read-only retrieval-quality + CPU-latency probe for the RAG embedding model.
Chunks a fixed set of real KB files with the production chunker, ranks chunks
against curated German/English queries with ground-truth target files, and
measures in-process encode latency + RAM per model config.

Also measures (optionally) the EmbeddingGemma 2 Q8_0 GGUF served by the
Unsloth backend over OpenAI-compatible /embeddings, and cross-checks its
vectors against the local text-only ST vectors for identical texts — the
"query/index geometry interchangeable?" verification for a future switch.

Run (from anywhere):
    python scripts/rag_probe5_gemma_compare.py [--remote-url URL --remote-model NAME]

Defaults: no remote leg. For the production backend:
    --remote-url http://host.docker.internal:8888/v1 \
    --remote-model unsloth/embeddinggemma-2-GGUF

Environment notes:
  * Requires a recent sentence-transformers/transformers (EmbeddingGemma 2 needs
    transformers >= 5.18). The probe prints versions first and fails loudly if
    a model cannot load — it never silently substitutes.
  * CPU dtype is float32 for EmbeddingGemma 2 (fp16 silently degrades/NaNs).
  * Downloads land in HF_HOME (default ~/.cache/huggingface) on first run.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ──────────────── Eval corpus: files + queries with ground truth ────────────

KB = ROOT / "data" / "knowledge" / "humblewood" / "HumbleWood" / "HumbleWood"
SESSION_NOTES = (
    ROOT / "data" / "knowledge" / "humblewood" / "HumbleWood"
    / "SessionLogs" / "2026-09-30_01_We are in Alderheart" / "notes.md"
)

EVAL_FILES = [
    KB / "Species" / "Corvum.md",
    KB / "spells" / "Gust_Barrier.md",
    KB / "spells" / "Feathered_Reach.md",
    KB / "Alderheart.md",
    KB / "Gods" / "Hath.md",
    KB / "species.md",
    KB / "places.md",
    KB / "Humblewood_Calendar.md",
    SESSION_NOTES,
]

# query text -> list of acceptable target files (basename)
QUERIES: list[tuple[str, list[str]]] = [
    ("Welche Rasse kann gliden und wie funktioniert das genau?", ["Corvum.md"]),
    ("Wie schnell fällt ein Corvum pro Runde, wenn er glidet?", ["Corvum.md"]),
    ("Was passiert, wenn ein Nahkämpfer durch Gust Barrier trifft?", ["Gust_Barrier.md"]),
    ("What does the cantrip Gust Barrier do against ranged attackers?", ["Gust_Barrier.md"]),
    ("Welche Stufe ist Feathered Reach und was ermöglicht er beim Fliegen?", ["Feathered_Reach.md"]),
    ("How long does Feathered Reach last and how far can you jump with it?", ["Feathered_Reach.md"]),
    ("Wer regiert Alderheart und wer ist der Council Speaker?", ["Alderheart.md"]),
    ("Which districts make up the city of Alderheart?", ["Alderheart.md", "places.md"]),
    ("Wo haben sich die Flüchtlinge nach den Feuern in Alderheart niedergelassen?", ["Alderheart.md"]),
    ("Who is Ava and what did she found in Alderheart?", ["Alderheart.md"]),
    ("Welche Domäne hat der Gott Hath und was ist sein Dogma?", ["Hath.md"]),
    ("What is the holy symbol of Hath, the Amaranthine of secrets?", ["Hath.md"]),
    ("Welche fünf Völkerrassen gibt es in Humblewood?", ["species.md"]),
    ("Which race are the owl folk of Humblewood called?", ["species.md"]),
    ("Wie messen die Leute von Everden die Zeit und wie heißt das aktuelle Jahr?", ["Humblewood_Calendar.md"]),
    ("What musical time units does the Everden calendar use?", ["Humblewood_Calendar.md"]),
    ("Was ist das Roots-Viertel und wann wurde es offiziell Teil der Stadt?", ["places.md"]),
    ("What happened in the session where the party was in Alderheart?", ["notes.md"]),
]

BATCH_SIZE = 8          # production RAG_EMBED_BATCH_SIZE
LATENCY_RUNS = 5
WARMUP_RUNS = 2


# ──────────────── Helpers ───────────────────────────────────────────────────

def rss_mb() -> float:
    """Current process RSS in MB (Linux)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def median_ms(fn, runs: int = LATENCY_RUNS, warmup: int = WARMUP_RUNS) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def load_chunks() -> list:
    from kb.chunker import Chunker
    chunks = []
    for f in EVAL_FILES:
        got = Chunker.split_file_sync(f)
        if not got:
            print(f"  WARNING: no chunks from {f}")
        chunks.extend(got)
        print(f"  {f.name}: {len(got)} chunk(s)")
    return chunks


def env_precheck() -> None:
    import torch
    import transformers
    import sentence_transformers as st
    print("── Environment ─────────────────────────────────────────────")
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"transformers {transformers.__version__} | sentence-transformers {st.__version__}")
    if tuple(int(p) for p in transformers.__version__.split(".")[:2]) < (5, 18):
        print("WARNING: EmbeddingGemma 2 needs transformers >= 5.18 — "
              "the gemma configs will likely fail to load.")
    print(f"CPU threads available: {os.cpu_count()}")


# ──────────────── Per-config evaluation ────────────────────────────────────

def eval_st_model(name: str, model, chunks, *, prefixes: bool) -> dict:
    """Rank + latency for one loaded ST model. ``prefixes`` selects the
    EmbeddingGemma 2 task-instruction variants (ignored for bge-m3)."""
    qtexts = [q for q, _ in QUERIES]
    targets = {i: set(t) for i, (_, t) in enumerate(QUERIES)}

    # Encode docs + queries. Three input regimes for gemma-style models:
    if not prefixes:
        doc_vecs = model.encode([c.content for c in chunks], batch_size=BATCH_SIZE,
                                normalize_embeddings=True, show_progress_bar=False)
        q_vecs = model.encode(qtexts, batch_size=BATCH_SIZE,
                              normalize_embeddings=True, show_progress_bar=False)
    else:
        doc_vecs = model.encode([c.content for c in chunks], batch_size=BATCH_SIZE,
                                prompt_name="Document",
                                normalize_embeddings=True, show_progress_bar=False)
        q_vecs = model.encode(qtexts, batch_size=BATCH_SIZE,
                              prompt_name="SearchQuery",
                              normalize_embeddings=True, show_progress_bar=False)

    doc_list = [list(v) for v in doc_vecs]
    q_list = [list(v) for v in q_vecs]

    # Rank: for each query score all chunks, check top-1/top-3 against targets.
    hit1 = hit3 = 0
    rows = []
    for i, qt in enumerate(qtexts):
        scores = sorted(
            ((cos(q_list[i], dv), j) for j, dv in enumerate(doc_list)), reverse=True
        )
        top3 = [(s, chunks[j].source_file, chunks[j].section_path) for s, j in scores[:3]]
        ok1 = top3[0][1] in targets[i]
        ok3 = any(f in targets[i] for _, f, _ in top3)
        hit1 += ok1
        hit3 += ok3
        rows.append((qt, top3, ok1, ok3))

    n = len(qtexts)
    # Latency: single query + batch of 8 (production shapes), plain encode.
    lat1 = median_ms(lambda: model.encode([qtexts[0]], normalize_embeddings=True,
                                          show_progress_bar=False))
    lat8 = median_ms(lambda: model.encode(qtexts[:BATCH_SIZE], normalize_embeddings=True,
                                          show_progress_bar=False))

    return {
        "name": name,
        "hit1": hit1, "hit3": hit3, "n": n,
        "lat1_ms": lat1, "lat8_ms": lat8,
        "rows": rows,
    }


def release_model(model) -> None:
    del model
    gc.collect()


# ──────────────── Remote (Unsloth backend) leg ─────────────────────────────

def remote_embed(url: str, api_key: str, model_name: str, texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": model_name, "input": texts}).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/embeddings", data=body,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return [d["embedding"] for d in data["data"]]


def eval_remote(url: str, model_name: str, chunks, local_plain_vecs: list[list[float]] | None):
    qtexts = [q for q, _ in QUERIES]
    print(f"\n── Remote backend leg: {model_name} @ {url} ──────────────")

    # Cold load can take a while — first call is the warmup.
    t0 = time.perf_counter()
    remote_embed(url, os.environ.get("INFER_API_KEY", ""), model_name, ["warmup"])
    print(f"  warmup (incl. cold model load): {time.perf_counter() - t0:.1f}s")

    doc_vecs = remote_embed(url, os.environ.get("INFER_API_KEY", ""), model_name,
                            [c.content for c in chunks])
    q_vecs = remote_embed(url, os.environ.get("INFER_API_KEY", ""), model_name, qtexts)

    targets = {i: set(t) for i, (_, t) in enumerate(QUERIES)}
    hit1 = hit3 = 0
    rows = []
    for i, qt in enumerate(qtexts):
        scores = sorted(((cos(q_vecs[i], dv), j) for j, dv in enumerate(doc_vecs)), reverse=True)
        top3 = [(s, chunks[j].source_file, chunks[j].section_path) for s, j in scores[:3]]
        ok1 = top3[0][1] in targets[i]
        ok3 = any(f in targets[i] for _, f, _ in top3)
        hit1 += ok1
        hit3 += ok3
        rows.append((qt, top3, ok1, ok3))

    lat8 = median_ms(lambda: remote_embed(url, os.environ.get("INFER_API_KEY", ""),
                                          model_name, qtexts[:BATCH_SIZE]))

    # Cross-check: backend GGUF vectors vs local ST text-only (plain) vectors.
    cross = None
    if local_plain_vecs is not None and len(local_plain_vecs) == len(doc_vecs):
        sims = [cos(a, b) for a, b in zip(local_plain_vecs, doc_vecs)]
        cross = {"mean": sum(sims) / len(sims), "min": min(sims)}

    return {"name": f"remote:{model_name}", "hit1": hit1, "hit3": hit3, "n": len(qtexts),
            "lat1_ms": float("nan"), "lat8_ms": lat8, "rows": rows, "cross_check": cross}


# ──────────────── Reporting ────────────────────────────────────────────────

def print_rows(res: dict) -> None:
    for qt, top3, ok1, ok3 in res["rows"]:
        flag = "OK " if ok1 else ("ok3" if ok3 else "MISS")
        best = top3[0]
        print(f"  [{flag}] {qt[:58]!r}")
        targets = next((t for q, t in QUERIES if q == qt), set())
        for s, f, sec in top3:
            mark = "*" if f in targets else " "
            print(f"      {s:.3f} {mark}{f} :: {sec[:40]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-url", default=None, help="OpenAI-compatible base URL of the backend")
    ap.add_argument("--remote-model", default=None, help="Model slug served by the backend")
    args = ap.parse_args()

    env_precheck()

    print("\n── Chunking eval corpus (production chunker) ────────────────")
    chunks = load_chunks()
    print(f"  TOTAL: {len(chunks)} chunks from {len(EVAL_FILES)} files")

    results = []
    gemma_text_plain_vecs = None

    # 1) Baseline: bge-m3 (production config, plain encode)
    print("\n── Loading BAAI/bge-m3 ──────────────────────────────────────")
    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    rss0 = rss_mb()
    m3 = SentenceTransformer("BAAI/bge-m3")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s, RSS delta ≈ {rss_mb() - rss0:.0f} MB")
    results.append(eval_st_model("bge-m3 (baseline)", m3, chunks, prefixes=False))
    release_model(m3)

    # 2) EmbeddingGemma 2, text-only 270M — three prefix regimes
    print("\n── Loading google/embeddinggemma-2 (text-only 270M) ─────────")
    import torch
    t0 = time.perf_counter()
    rss0 = rss_mb()
    gem_t = SentenceTransformer(
        "google/embeddinggemma-2",
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s, RSS delta ≈ {rss_mb() - rss0:.0f} MB")

    results.append(eval_st_model("gemma-2 270M (no prefixes)", gem_t, chunks, prefixes=False))
    # keep plain doc vectors for the remote cross-check
    gemma_text_plain_vecs = [
        list(v) for v in gem_t.encode([c.content for c in chunks], batch_size=BATCH_SIZE,
                                      normalize_embeddings=True, show_progress_bar=False)
    ]
    results.append(eval_st_model("gemma-2 270M (task prefixes)", gem_t, chunks, prefixes=True))
    release_model(gem_t)

    # 3) EmbeddingGemma 2, full 740M — quality ceiling reference
    print("\n── Loading google/embeddinggemma-2 (full 740M) ──────────────")
    t0 = time.perf_counter()
    rss0 = rss_mb()
    gem_f = SentenceTransformer("google/embeddinggemma-2",
                                model_kwargs={"torch_dtype": torch.float32})
    print(f"  loaded in {time.perf_counter() - t0:.1f}s, RSS delta ≈ {rss_mb() - rss0:.0f} MB")
    results.append(eval_st_model("gemma-2 740M (task prefixes)", gem_f, chunks, prefixes=True))
    release_model(gem_f)

    # 4) Optional remote leg (Unsloth backend GGUF)
    if args.remote_url and args.remote_model:
        results.append(eval_remote(args.remote_url, args.remote_model, chunks,
                                   gemma_text_plain_vecs))

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────────────────")
    hdr = f"{'config':<34} {'top-1':>7} {'top-3':>7} {'1q ms':>8} {'8q ms':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<34} {r['hit1']:>3}/{r['n']:<3} {r['hit3']:>3}/{r['n']:<3} "
              f"{r['lat1_ms']:>8.0f} {r['lat8_ms']:>9.0f}")
    for r in results:
        if r.get("cross_check"):
            cc = r["cross_check"]
            print(f"\n  cross-check {r['name']} vs local ST text-only (plain): "
                  f"mean cos {cc['mean']:.4f}, min cos {cc['min']:.4f}")

    print("\n── Per-query detail ──────────────────────────────────────────")
    for r in results:
        print(f"\n[{r['name']}] top-1 = {r['hit1']}/{r['n']}")
        print_rows(r)

    # Paste-ready markdown block
    print("\n===MARKDOWN===")
    print("| config | top-1 | top-3 | 1-query ms (median) | 8-query batch ms (median) |")
    print("|---|---|---|---|---|")
    for r in results:
        lat1 = f"{r['lat1_ms']:.0f}" if r["lat1_ms"] == r["lat1_ms"] else "—"
        print(f"| {r['name']} | {r['hit1']}/{r['n']} | {r['hit3']}/{r['n']} | {lat1} | {r['lat8_ms']:.0f} |")
    for r in results:
        if r.get("cross_check"):
            cc = r["cross_check"]
            print(f"\nCross-check backend GGUF vs local ST text-only (plain texts): "
                  f"mean cosine {cc['mean']:.4f}, min {cc['min']:.4f}.")


if __name__ == "__main__":
    main()
