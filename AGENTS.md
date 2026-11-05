## LLM backend topology (read before touching models — GPU split, 2026-11)

Two physical cards, two backends — **never mix them up**:

- **Chat (the bot AND this agent): Strata on the 5070 Ti (GPU 1)** — `INFER_URL`
  in `.env`. The 5070 is chat-only: image gen and the RAG sidecars must never
  load onto it. `tts-ollama` (phonon) also lives on this card.
- **Image generation: Unsloth Studio on the 4070 Ti SUPER (GPU 0)** —
  `IMAGE_INFER_URL` (default falls back to `INFER_URL`; production sets it to
  `http://host.docker.internal:8888/v1`, Studio pinned to GPU 0 only).
  Image model: `unsloth/Qwen-Image-2.1-Turbo-GGUF` Q4_K_M, measured resident
  ~12.5 GiB (`resident_mib=12456` in dev.log).
- **The 4070 runs ONE mode at a time**, owned by the `gpu-switch` compose
  service (`scripts/gpu_switch_helper.py`): `rag` mode (reranker-llama +
  embedder-llama, default steady state — both pinned `CUDA_VISIBLE_DEVICES=0`)
  ⇄ `image` mode (those stopped, diffusion model loaded). The bot switches via
  `GPU_SWITCH_URL` around every painting window (`IMAGE_MODE_HOLD_S` defers the
  switch-back) — see `bot_core/images.py`; manual CLI twin:
  `./scripts/gpu_switch.sh status|image|rag`.
- **Prompt rewriting ALWAYS happens before the switch** (the rewrite's KB agent
  tools include the reranker, which the switch takes offline).
- **Everything runs in Docker on the host**, not in this sandbox: the bot
  container, the RAG sidecars, the gpu-switch helper *and* the Unsloth Studio
  container. The sandbox can only reach them over the network
  (`host.docker.internal`) — it cannot `docker compose` anything itself. Ask
  the user to run host-side commands (`docker compose up -d --build`, GPU
  exposure, …).

## Backlog workflow

- **`.pi/backlog.md` is the single task list.** The user drives work with "Backlog reference —
  work on these item(s)" prompts that name one or more items already marked `[working]`.
- When an item is finished, mark it `[x]` and append a completion summary (what was done, where
  it lives, verification results) — the existing entries show the expected level of detail.
- Items may be **awaiting a user decision** (e.g. #39's prioritized plan in
  `docs/internal/RAG_ARCHITECTURE_RESEARCH.md` §4). Do not implement those until the user picks.

## Data pipelines (not covered by docs/ — read the PIPELINE.md files)

- **D&D book data** (knowledge base sources): pipeline + history in
  `data/book_collection/PIPELINE.md`. Regeneration goes through `collect_books.py`; verify entry
  counts and diffs against a pre-regen backup before regenerating.
- **Magic item collection**: pipeline in `data/item_collection/` (its own PIPELINE.md). Built
  from cookie-free public sources only (DDB SSR listings, dnd-wiki.org API, Basic Rules A-Z,
  per-book gap lists). `data/items/` holds ONLY the live bot tables.
- **Known gap (user decision 2026-10-19):** ~394 header-only magic-item stubs + 6 Arcana
  Unleashed monsters whose D&D Beyond URLs redirect to the paid marketplace never got content —
  they stay as name+rarity stubs unless a logged-in/headful DDB harvest is done.
