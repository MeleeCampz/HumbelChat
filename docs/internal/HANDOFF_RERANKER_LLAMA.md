# Handoff — reranker on pinned llama-server (replaces vLLM container)

**STATUS: FULLY DONE 2026-10-25 (late evening).** Reranker AND embedder (§6)
both run as **compose services** on the official digest-pinned
ggml-org/llama.cpp server image (`sha256:2fe406…` = build 11870; numbered
tags stop at b4738, which is pre-qwen3 and cannot load these models) —
persistence solved for free (`restart: unless-stopped`). The bot reaches both
over the compose network (`reranker-llama:8080`, `embedder-llama:8080`);
loopback :8090/:8091 are host debug only (no auth — never LAN-bind).
Late-evening additions to the deviations below:

- **GPU pinning:** the Windows nvidia runtime exposes BOTH GPUs to a container
  regardless of the compose `device_ids` reservation (verified: inference ran
  on GPU 0 with zero GPU-1 activity), so both services set
  `CUDA_VISIBLE_DEVICES=1`. Corollary: host-side `nvidia-smi dmon` SM
  attribution is UNRELIABLE for guest processes under WDDM (showed GPU 0 busy
  while the model sat on GPU 1) — verify placement by VRAM delta instead.
- **Embedder (§6):** official `Qwen3-Embedding-4B-Q4_K_M.gguf` (copied to
  `C:\Users\Tobias\.unsloth\models\`), `--embeddings` (singular `--embed`
  removed upstream) `-c 8192` — 2048 overflowed the KV on code-heavy ~2k-token
  chunks (HTTP 500 killed a full reindex mid-flight; atomic swap preserved the
  old cache). Vectors ≈0.9987 cosine vs the old Ollama model; renamed
  `EMBEDDING_MODEL`/`INDEX_EMBED_MODEL` → `qwen3-embedding-4b` so the bot
  auto-reindexed once (5,901 chunks) — all vectors in one space.
  Ollama + vLLM containers stay in compose (stopped) as rollback.
- **Benchmark on the new stack:** 42 positive cases, single miss
  ('armor stat block' — ambiguous query, sibling case hits @3); negatives
  score 0.35–0.43 < the 0.50 floor; 150-pair rerank batch ~2.5 s in-container.
- **Known oddity (not caused by this cutover):** the live bot's
  `contextual-fill` worker never completes its `_ensure_index_store()`
  (no "Vector index store ready" log, no embedder traffic after the 300 s
  delay, in two separate boots; the identical code path works fine in a fresh
  process). py-spy can't attach (no ptrace in the container). See backlog #59.

Earlier deviations from this plan, all benign:

- §1: the "already in local HF cache" `Qwen/Qwen3-Reranker-4B-GGUF` was NOT on
disk, and **every** community 4B GGUF found (mradermacher incl. f16,
QuantFactory, Mungert, DevQuasar…) had dropped the classifier head — so
fallback (a) was taken: official `Qwen/Qwen3-Reranker-4B` safetensors →
`convert_hf_to_gguf.py` (this tree maps yes/no lm_head rows →
`cls.output.weight`, bakes `pooling_type=rank` + rerank chat template) →
`llama-quantize Q4_K_M`. Artifacts in `C:\Users\Tobias\.unsloth\models\
` (`…-f16.gguf` 8.0 GB kept for re-quants, `…-Q4_K_M.gguf` 2.38 GB served).
- §2: needed `--host 0.0.0.0` — on this box `host.docker.internal` resolves
to the LAN IP (192.168.178.96), and the default 127.0.0.1 bind was
unreachable from the bot container.
- §4: quality strong ≈ 0.9+ / weak 0.2–0.5 / irrelevant ≈ 1e-4 (probability
scale, same family as vLLM) — `RAG_ATTACH_FLOOR=0.50` re-checked against
known attach/detach pairs, still separates cleanly, **no .env change**.
Speed: 150-pair batch **4.15 s** (vLLM baseline 10.8 s); 12-case benchmark
11/12 (only miss = known Griffon part-2 long-tail), rerank passes 12/12 OK;
GPU-1 SM util 94% during batches (WDDM hides the VRAM number).
- §5: vLLM container already stopped by user; left in compose as rollback.
- §8: backlog #42 annotated + new #56 (embedder migration + server
persistence); docs/configuration.md reranker section updated.

(Both formerly-open items — server persistence and the §6 embedder
migration — are done: both servers are compose services with
`restart: unless-stopped`; see the status header above.)

## 0b. GPU accounting + dynamic switching (2026-10-25, after user pushback)

The earlier "~8 GB free on GPU 1" claim was **wrong** — it silently dropped
the chat model's share. Measured ground truth (llama.cpp `-lv 4` startup
"memory breakdown" logs are the ONLY reliable source here; nvidia-smi under
WDDM shows lazily-retained freed VRAM as "used" and is useless for this):

| Resident on GPU 1 (RTX 5070 Ti, 16,275 MiB) | VRAM |
|---|---|
| reranker-llama (Q4_K_M, ctx 2048 ×4 slots) | **2,742 MiB** (model 2375 + KV 288 + compute 79) + 325 host |
| embedder-llama (Q4_K_M, ctx 8192) | **3,612 MiB** (model 2375 + KV 1152 + compute 85) + 331 host |
| chat Qwen3.8-27B NVFP4_50 (Studio; 16.1 GB file, `--tensor-split 50,16`) | **~3.9 GB weights** on GPU 1 + KV/compute ≈ 4–6 GB total (estimate — verify by reloading in Studio and running a probe) |

→ With ALL THREE resident: ~10.5–12.5 GB used, **only ~3.5–5.5 GB left** for
an image model. Without the chat model: ~9.7–10 GB free (measured 8,086 MiB
after the embedder loaded). Cold start is cheap: full stop→healthy compose
cycle measures **~19 s** (reranker healthy ~8 s, embedder ~12 s).

Also found + fixed that day: a **stray bare-process reranker** from the first
cutover (pre-compose) was still running on `0.0.0.0:8090` — LAN-exposed with
no auth and holding a second copy of the model; killed. The compose service
is the only canonical path.

**Dynamic switching = `scripts/gpu_switch.sh`** (repo root):
- `./scripts/gpu_switch.sh image` — stops both RAG servers, frees ~6.4 GB on
  GPU 1 for an image model. The bot degrades gracefully while they're down:
  retrieval falls back to dense+BM25 (RRF) — the measured pre-reranker
  baseline; the rerank attach floor no-ops (no gate scores); nothing crashes.
- `./scripts/gpu_switch.sh rag` — starts both, waits for `/v1/models`
  health on loopback :8090/:8091 (~19 s total).
- `status` — service states + resident-VRAM table.

Note: the **embedder is only needed when INDEXING new chunks** (search reads
precomputed vectors and never touches it) — running it permanently costs
3.6 GB for near-zero benefit. Candidate optimization: keep it stopped by
default, start on demand for reindex/context-fill. Left as a user decision;
current default keeps both running.

---

Original plan — for a pi session WITH host/system access. Goal: serve Qwen3-Reranker-4B via a
standalone `llama-server` pinned to GPU 1 (dynamic memory, ~2.6 GB flat)
instead of the vLLM container (8 GB pool pre-allocation that crash-loops on
this contended WDDM card). Frees ~8 GB of GPU 1 permanently for the
image-generation work (#42/#50) and kills the Ollama-can't-pin-GPU problem
for the embedder migration in §6.

**Bot side is ALREADY DONE** (commits on `feat/tool-calling`, pushed):
jina `/rerank` payload now sends `"model"` (required by llama.cpp, accepted
by vLLM) + tests updated. Only `.env` + host work remain.

## 0. Current state (2026-10-25 evening)
- `humbelchat-reranker-qwen` container: **stopped** (user stopped it after the
  0.3/0.35 crash-loop saga). `.env` still points at `http://…:8000` → bot is
  running rerank-less (fails open to pre-rerank order — safe but degraded).
- compose reverted to the proven `--gpu-memory-utilization 0.5` (committed) —
  the rollback path in §7 still works.
- llama.cpp Windows build exists: `C:\Users\Tobias\.unsloth\llama.cpp\build\bin\Release\llama-server.exe`
- Candidate GGUF already in local HF cache: `Qwen/Qwen3-Reranker-4B-GGUF`
  (seen in the Studio catalog at :8888) →
  `C:\Users\Tobias\.cache\huggingface\hub\models--Qwen--Qwen3-Reranker-4B-GGUF\snapshots\<hash>\*.gguf`

## 1. VERIFY the GGUF (critical — do not skip)
Community 4B conversions sometimes drop the classifier head; such files load
and answer fine but score everything ~1e-28 (garbage). Check:

```powershell
python "C:\Users\Tobias\.unsloth\llama.cpp\gguf-py\gguf\scripts\gguf_dump.py" "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen3-Reranker-4B-GGUF\snapshots\*\*.gguf" | Select-String "cls.output|general.architecture"
```

- **Expect:** a `cls.output.weight` tensor + architecture `qwen3` (or similar).
  Note the exact file path + quant for step 2.
- **If `cls.output` is MISSING:** do not use it. Fallbacks, in order:
  (a) convert official weights yourself: `convert_hf_to_gguf.py` on
  `Qwen/Qwen3-Reranker-4B` then `llama-quantize` Q4_K_M (both tools are in
  the same llama.cpp dir); (b) use `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`
  (maintainers' build, known-good — but a quality step down, benchmark before/after);
  (c) keep the vLLM container at 0.5 and drop this plan.

## 2. Launch pinned to GPU 1
```powershell
$env:CUDA_VISIBLE_DEVICES = "1"
& "C:\Users\Tobias\.unsloth\llama.cpp\build\bin\Release\llama-server.exe" `
  -m "<gguf path from step 1>" `
  --alias reranker-4b --reranking --pooling rank `
  -c 2048 --no-context-shift --flash-attn on `
  --port 8090
```
Flag notes: `--reranking --pooling rank` enables the OpenAI-style `/v1/rerank`
(+ `/rerank`) endpoint returning `{results:[{index, relevance_score}]}` —
exactly our jina wire shape. `-c 2048` mirrors the old `--max-model-len 2048`
(scoring input unchanged). Leave this window open (or wrap in a Task
Scheduler task / docker service later — persistence is a follow-up, not a
gate).

Sanity from any machine: `curl http://host.docker.internal:8090/v1/models`
should list `reranker-4b`.

## 3. Point the bot at it (.env)
```
RERANK_API_BASE=http://host.docker.internal:8090/v1   # was ...:8000 (vLLM)
RERANK_MODEL=reranker-4b                               # must match --alias
# RERANK_API_FORMAT=jina stays; RERANK_INSTRUCT stays (llama.cpp ignores it)
```
Then `docker compose restart bot` (env change = recreate). Startup log should
show "Reranker endpoint healthy" — NOTE: the startup check hits `/health`;
if llama.cpp 404s there it only logs a warning and retries per request, so
verify with a real call instead of trusting the log line.

## 4. VERIFY quality + speed (before killing vLLM)
Run from the repo (sandbox shims apply automatically outside the sandbox):
```powershell
python -c "import sys; sys.path.insert(0,'scripts'); import kb.reranker as R, asyncio, time, config.settings as S; print(S.RERANK_API_BASE); ..."
```
…or simpler: two checks.
1. **Quality:** one `/rerank` call with 3 docs (one clearly relevant to the
   query) → relevant must score highest, in the same ballpark as vLLM's
   previous outputs (~0.3+ relevant vs <0.1 irrelevant; exact scale may
   differ slightly between servers — what matters is ORDERING + separation).
   ⚠ The attach floor is calibrated on vLLM scores (`RAG_ATTACH_FLOOR=0.50`
   rerank-source) — if llama.cpp's score scale differs materially, re-check
   the floor against a few known attach/detach cases and adjust .env.
2. **Speed:** timed 150-pair batch (the benchmark does this automatically):
   `python scripts/rag_benchmark.py --limit 12` → compare rerank stage times
   vs the 10.8 s /150-pairs vLLM baseline and the 41/42 hit rate.

## 5. Cut over
All green → `docker compose stop reranker-qwen` (leave it defined in compose
as the rollback). GPU 1 should drop ~8 GB. Watch a few live /ai turns
(`logs/ai_turns/`) for "Rerank failed" warnings — there should be none.

## 6. Follow-up: embedder migration (same pattern, separate session)
`Qwen/Qwen3-Embedding-4B-GGUF` (also in local cache) → second pinned
llama-server (`--embed`, `/v1/embeddings` is OpenAI-compatible — `EMBED_URL`
swap only), then retire the Ollama container. This finally solves
"embedder on a specific GPU, held resident" (Ollama cannot pin GPUs at all).
Do NOT do this in the same session as the reranker cutover — one variable.

## 7. Rollback (any step goes wrong)
```powershell
# .env: RERANK_API_BASE=http://host.docker.internal:8000
#       RERANK_MODEL=boboliu/Qwen3-Reranker-4B-W4A16-G128
docker compose up -d reranker-qwen     # 0.5 reservation, proven stable
docker compose restart bot
```

## 8. Record when done
Backlog: note under #42 (image gen) that the GPU-1 room prerequisite landed;
add/annotate a line for the embedder migration (#6 above); update
`docs/configuration.md` reranker section + this file's status header.
