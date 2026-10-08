# Evaluation — EmbeddingGemma 2 as RAG embedding model (backlog #24)

**Date:** 2026-10-08 · **Probe:** `scripts/rag_probe5_gemma_compare.py` · **Verdict: keep bge-m3 for now.**

## What was evaluated

Google's **EmbeddingGemma 2** (`google/embeddinggemma-2`, Apache 2.0) vs the current
`BAAI/bge-m3`, on our own KB (Humblewood campaign content, German + English):

- 9 real KB files → 65 chunks via the **production chunker** (`kb.chunker.Chunker`)
- 18 curated queries (9 DE / 9 EN) with ground-truth target files
- All local configs in-process via sentence-transformers on CPU (float32), same path as
  production; plus the **Q8_0 GGUF served by the Unsloth backend** over `/embeddings`

Configs: bge-m3 (baseline, plain encode) · gemma-2 **text-only 270M** (`config_kwargs`
encoder omission), with and without the model's task-instruction prefixes · gemma-2 **full
740M** (quality ceiling) · remote `unsloth/embeddinggemma-2-GGUF` Q8_0.

## Results

| config | top-1 | top-3 | 1-query ms¹ | 8-query batch ms¹ |
|---|---|---|---|---|
| **bge-m3 (baseline)** | **18/18** | **18/18** | 214–287 | 420–500 |
| gemma-2 270M (no prefixes) | 15/18 | 17/18 | 117–124 | 305–451 |
| gemma-2 270M (task prefixes) | 16/18 | 16/18 | 135–277 | 388–432 |
| gemma-2 740M (task prefixes) | 16/18 | 16/18 | 125–270 | 295–505 |
| remote GGUF Q8_0 (Unsloth, GPU) | 15/18 | 17/18 | — | **46–50** |

¹ Median over 5 runs after warmup; ranges span two probe runs — CPU latency on this box is
noisy and **indicative only** (measured in a 24-core sandbox, not the bot host).

**Prefix ablation:** adding the `SearchQuery`/`Document` task prefixes moved top-1 up by one
hit (15→16) but top-3 down by one (17→16) — no clear win in this setup.

**Backend cross-check:** the Q8_0 GGUF vectors vs local ST text-only (plain texts): mean
cosine **0.9998**, min 0.9996 → quantized backend index vectors and local CPU query vectors
are effectively interchangeable. A GPU-index + local-query split with gemma-2 would be safe.

## Why bge-m3 still wins (narrowly)

The two queries every gemma config misses:

1. *"Welche Rasse kann gliden und wie funktioniert das genau?"* — genuinely ambiguous
   (Corvum's Glide trait **and** Feathered Reach both describe gliding); gemma ranks
   species.md / session notes above Corvum.md, bge-m3 picks Corvum.md.
2. *"Wo haben sich die Flüchtlinge nach den Feuern in Alderheart niedergelassen?"* — the
   session notes also discuss the fires; gemma ranks notes.md first, bge-m3 picks the
   lore file.

Neither miss is a "wrong topic" failure — decoys are topically related and scores are close
(Δ ≈ 0.01–0.02). The 2-query gap on an 18-query set is within eval noise; see caveats.

## Caveats (why this is "keep for now", not "gemma is worse")

- **Small eval:** 18 queries / 65 chunks. Production KB has thousands of chunks — much higher
  decoy density, where score-distribution differences could go either way. A production-scale
  re-measurement (e.g. via `/reindex_kb` + live retrieval sampling) is the only strong test.
- **Latency not measured on the bot host.** Gemma-2 270M was ≤ bge-m3 here, but the bot host
  CPU is what matters for per-query in-process encoding.
- Public benchmarks (MTEB multilingual v2: gemma-2 61.36) are only ~on par with bge-m3 — no
  paper advantage to cash in on.

## If we ever switch anyway — checklist (not executed)

1. **Dependency bump first:** EmbeddingGemma 2 needs `transformers ≥ 5.18` + `torchvision`
   (processor import); production pins `sentence-transformers<5`. Bump requirements/Dockerfile
   and re-run the full test suite.
2. **Query side:** `LOCAL_EMBED_MODEL=google/embeddinggemma-2` + text-only load
   (`config_kwargs={"vision_config": None, "audio_config": None}`, float32 — fp16 silently
   degrades) in `kb/embedder.py::_get_local_model`. Optionally add the `SearchQuery` prefix.
3. **Index side:** Q8_0 GGUF is already loaded in Unsloth (8K context); set
   `INDEX_EMBED_MODEL=unsloth/embeddinggemma-2-GGUF`. Index cache auto-rebuilds on model
   mismatch. GPU batch throughput measured: ~50 ms per 8 texts — a full reindex is cheap.
4. **Geometry check already done:** GGUF vs ST text-only cosine 0.9998 (above) — safe.
5. Docs: `docs/rag.md`, `.env.example`.

## Artifacts

- Probe: `scripts/rag_probe5_gemma_compare.py` (read-only; re-runnable, incl. `--remote-url` /
  `--remote-model` for the backend leg)
- Raw run logs kept locally at probe-run time (`/tmp/probe_run2.log` on the eval machine)
