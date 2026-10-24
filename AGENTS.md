## LLM backend topology (read before touching models)

- **The agent's own chat and the HumbelChat bot share the same backend** — Unsloth Studio
  (`INFER_URL=http://host.docker.internal:8888/v1`, see `.env`), and both run on the same model
  (`MODEL_NAME` in `.env`, currently `unsloth/Qwen3.8-27B-GGUF:NVFP4_50`). There is a single
  resident model — anything that affects model residency or in-flight requests affects both.
- **Everything runs in Docker on the host**, not in this sandbox: the bot container *and* the
  Unsloth Studio container. The sandbox can only reach them over the network
  (`host.docker.internal`) — it cannot `docker compose` anything itself. Ask the user to run
  host-side commands (`docker compose up -d --build`, GPU exposure, …).

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
