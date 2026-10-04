# RAG / Knowledge Base

The bot can attach relevant local knowledge-base content to AI prompts. This is optional.

## How it works

1. Documents are stored in `KB_PATH`.
2. Files are chunked using the smart chunker.
3. A retrieval strategy (hybrid dense + BM25 by default) selects the most relevant chunks for a query.
4. Selected context is included in the AI request, up to `RAG_MAX_CHARS`.

## Embedding strategy — no per-request model swap

The single biggest problem this design solves: with one inference backend serving both
chat and embeddings, a naive setup would have to **swap the chat model for the embedding
model on every RAG query** (and back) — adding seconds of latency to every message. We
avoid that entirely by putting the two sides of embedding on different paths:

| Side | Where it runs | Config | When |
|---|---|---|---|
| **Query embeddings** (prompt processing) | In-process, **CPU** | `EMBED_BACKEND=local`, `LOCAL_EMBED_MODEL` (bge-m3) | Every `/ai` request — fast, no backend round-trip |
| **Index build / rebuild** | On the **GPU** backend | `INDEX_EMBED_BACKEND=backend`, `INDEX_EMBED_MODEL` (GGUF bge-m3) | One-time batch job (`/reindex_kb`) |

Key consequences:

- **Prompt processing never touches the inference backend.** Each query is embedded in the
  bot process on CPU, so the chat model can stay loaded at all times and query latency is
  independent of whatever is running on the GPU. There is no model swapping — ever.
- **The heavy work (embedding thousands of chunks) happens once, on the GPU**, where a full
  reindex finishes in a couple of minutes instead of ~20–60 min on pure CPU.
- Both sides use the **same model** (bge-m3), so their vectors are interchangeable — see
  "Why mixing CPU and GPU embeddings is safe" below.

The in-process query path requires the `rag-cpu` extra (sentence-transformers). If the
backend is unreachable during an index build, that build falls back to local CPU so a
reindex never hard-fails.

### Fast GPU reindex flow (Unsloth)

A full reindex is fastest when the embedding model runs on the GPU with the chat model
out of the way:

1. **Unload the chat model** in Unsloth Studio — this frees the VRAM the embedding model needs.
2. **Load bge-m3 on the GPU.** Set its batch/ubatch (physical batch) to at least 8192 so long
   KB chunks don't overflow the context. `INDEX_EMBED_MODEL` must be the *exact* name your
   backend serves for `/embeddings` — on an Unsloth host that is the GGUF server name
   (e.g. `gpustack/bge-m3-GGUF`), which routes to the GPU llama.cpp path. A bare HF id such as
   `BAAI/bge-m3` may instead resolve to a CPU sentence-transformers path, which is far slower
   for a full reindex (and can time out on long chunks).
3. **Run `/reindex_kb`.** With the model on the GPU this finishes in minutes rather than the
   ~20-60 min a pure-CPU build takes.
4. **Reload the chat model.** The bot's per-request query embeddings are unaffected (they run
   in-process on CPU via `EMBED_BACKEND=local`), so retrieval keeps working throughout.

### Why mixing CPU and GPU embeddings is safe

Both sides use the **same model** (bge-m3), so their vectors are interchangeable —
cosine similarity between a CPU query vector and a GPU-built index vector is valid. We
verified this empirically: rebuilding the full 1314-chunk index on the GPU backend and
re-running identical queries produced rankings that were **identical to 5-decimal
precision** (mean `|score|` delta ≈ 0, top-1 and top-3 matches for every query; only
near-tied chunks deep in the top-10 swapped). The practical consequence: build the index
once on the GPU for speed, serve queries in-process on CPU with zero backend round-trips,
and either side can be rebuilt later without invalidating the other.

## Retrieval methods

Controlled by `RAG_RETRIEVAL_METHOD`:

- `vector` (default): semantic search over the embedding index. A SQLite-backed index in `<KB_PATH>/.vector_index_cache/` caches embeddings so restarts do not require re-embedding.
- `keyword`: TF-IDF-style heuristic scoring based on filenames, headers, and body text overlap. Works without a vector backend.

If vector search is unavailable, keyword search can be used as a fallback.

### Hybrid dense + BM25 retrieval

By default (`RAG_HYBRID_ENABLED=1`) the vector path fuses **two** signals with reciprocal
rank fusion (RRF):

- **Dense** — cosine similarity between the query embedding and each chunk (captures
  meaning / paraphrase).
- **Lexical (BM25)** — an exact-term index over every chunk (captures names, spell titles,
  stat blocks, proper nouns that dense vectors can under-weight).

RRF merges the two orderings without needing score calibration, so a query still finds the
right chunk when the player's phrasing differs from the KB's vocabulary. Set
`RAG_HYBRID_ENABLED=0` to use dense-only retrieval.

### Min-attachment relevance floor (opt-in)

`RAG_MIN_ATTACH_SCORE` (default `0` = off) drops **dense** chunks whose cosine similarity
is below the threshold *before* hybrid fusion, so weak vector hits cannot crowd out strong
BM25 matches. Lexical-only matches still survive via the hybrid step. Tune it from the
`Vector scores for ...: top=... median=... min=...` log line.

### Per-file chunk budget + attach-time relevance floor

Two knobs control how much of each matched file actually reaches the prompt:

- **`RAG_MAX_CHUNKS_PER_FILE`** (default `3`) caps how many of a file's ranked
  chunks are attached. They're joined into one per-file entry, so lowering this
  trims the "whole file" bloat — e.g. a spell query no longer drags in four
  neighbouring sections when one is enough.
- **`RAG_ATTACH_FLOOR`** (default `0.50`, `0` = off) is a per-chunk relevance gate
  applied at *attach* time (after hybrid fusion / rerank). A ranked chunk whose
  original **dense cosine similarity** is below the floor is skipped, so only
  genuinely relevant sections are attached. It gates on the raw dense score — not
  the rank-based RRF score — and **lexical-only chunks** (exact-term BM25 hits with
  no dense score) are always kept. A safety net guarantees an over-aggressive floor
  can never produce an empty context: if it would drop everything, selection falls
  back to the unfiltered ranking.

## Low-confidence query rewriting (vector path)

Vector search alone can miss the right chunk when a player's phrasing differs
from the KB's vocabulary. To cover that case without slowing down every
query, the rewriter only runs when the index is *not* confident:

1. The query is embedded and ranked as usual (one embedding call).
2. If the top similarity score is **below `RAG_REWRITE_MIN_SCORE`**
   (default `0.35`), the LLM generates up to `RAG_QUERY_MAX_EXPANSIONS`
   alternative phrasings (wall-clock budget: `RAG_REWRITE_BUDGET_SECONDS`).
3. All expansions are embedded in **one batched call**, and every ranking is
   merged with reciprocal rank fusion (RRF) — order-based, so no score
   calibration is needed.
4. If the rewrite or the expansion embedding fails for any reason, retrieval
   silently falls back to the original ranking.

Confident queries (top score ≥ threshold) pay nothing extra. Each query logs
its score distribution (`Vector scores for ...: top=... median=... min=...`)
so you can tune `RAG_REWRITE_MIN_SCORE` from real traffic — raise it if the
rewriter triggers too often, lower it if answers still miss on unusual
phrasing. Set `RAG_QUERY_REWRITER=0` to disable the feature entirely.

The rewrite always uses **the same model as the main completion call** for
that request (a per-character model override is forwarded into the rewriter),
so a single-model local backend never has to load a second set of weights.

## Request serialization (single local backend)
The bot assumes one inference backend serving one model. AI requests are
therefore serialized **process-wide**: at most one request (RAG retrieval +
completion) is in flight at any time, served FIFO. Requests from different
channels queue up rather than running in parallel — a second `/ai` while the
first is still generating simply waits its turn.

## Smart chunking

The chunker uses a few strategies depending on file size and structure:

- **Full document** for small files, to preserve context
- **Header-based splitting** for larger documents, with minimum-size merging
- **Adaptive paragraph splitting** as a fallback for dense content

This helps queries hit relevant sections without being drowned out by unrelated content.

## Supported file types

The knowledge base accepts and meaningfully reads/indexes:

- `.txt`
- `.md`
- `.csv`
- `.html`
- `.xml`
- `.rtf`

These are the expected file types for KB use. Storage itself does not strictly enforce only these extensions — uploaded files use MIME-based inference and fall back to `.txt` when the extension or MIME type is unknown — but files whose extensions are not in the set above are generally not read or indexed by the same path.

## Obsidian vault sync (#19)

Vaults kept as (private) GitHub repos are mirrored into the KB and kept in step
by a background task (`bot_core/vault_sync.py`): each vault in `OBSIDIAN_VAULTS`
is cloned to `<KB_PATH>/<vault-name>/`, pulled on a fixed interval, and changed
files are re-indexed via the same incremental path as `/sync_kb`. The optional
designated **campaign vault** (`OBSIDIAN_CAMPAIGN_VAULT`) is two-way — session
notes live inside it (via `SESSIONS_NOTES_DIR`) and the bot commits + pushes
them, so bot-generated session logs appear in Obsidian on every machine. Dot-dirs
(`.git/`, `.obsidian/`) are skipped by the indexer as usual; non-text attachments
are never indexed. Full setup guide: `docs/obsidian-vault.md`.

## Session notes — one RAG document per session (#11)

Each session folder under `<SESSIONS_NOTES_DIR>/<date>_<idx>[_<name>]/`
(default `<KB_PATH>/session_notes/`; see Obsidian vault sync above) holds
exactly ONE RAG-indexed file — `notes.md` — which contains:

- the timestamped session notes (`## Notes`),
- the **AI-merged session log** (`## Session Log`) — when a session ends, the
  AI combines all player uploads/transcripts (the same chronological events,
  logged from different perspectives, with heavy overlap) into ONE complete,
  well-formatted, de-duplicated session log. That single text is the
  session's RAG content: near-identical uploads no longer exist as separate
  retrievable chunks, and `RAG_MAX_CHUNKS_PER_FILE` caps the whole session at
  a few chunks instead of a few × N files,
- the AI recap written when the session ended.

If the merge is impossible (AI backend down, stale auto-end without AI,
legacy sessions) the file falls back to a mechanical `## Documents` section
carrying each upload's full text under its own `### <filename>` subsection —
content stays reachable via RAG either way. The merge prompt is customizable
via `SESSION_MERGE_PROMPT` in `.env`.

The raw uploads/transcripts are kept on disk in **hidden dot-dirs**
(`.attachments/`, `.transcripts/`) as the verbatim record and as the source
for both the end-of-session recap and the log merge. The indexer skips
dot-dirs and dot-files, so they can never be re-indexed by `/sync_kb` or a
startup rebuild — which is what previously let near-identical session uploads
get attached multiple times by retrieval.

Legacy folders with visible `attachments/` / `transcripts/` sub-folders are
migrated automatically at startup (rename to dot-dirs + append the combined
section to `notes.md`); the vector index then self-heals on its next load
(stale per-file rows pruned, recombined notes re-embedded once).

Note: manual edits to the `## Session Log` / `## Documents` sections of a
`notes.md` are overwritten on the next session change (they are regenerated
from state + dot-dirs); edits to real notes in `## Notes` survive via
re-parsing.

## Last-session context (#7)

Continuity across sessions does not rely on RAG at all: when a session has
ended, every AI turn automatically gets a `[Previous session — <name> …]`
block (before the RAG block) containing **only the session's recap** —
a very brief orientation (what happened + next steps), deliberately nothing more. Detailed session content is
retrieved *on demand* by RAG from the indexed `notes.md` (one file per
session), so the proactive block stays small and the two mechanisms do not
duplicate each other. The block is a pure function of the ended session's
data — no per-turn timestamps or randomness — so it renders **byte-identical
on every turn** (stable context, unlike query-dependent RAG chunks). If the
recap alone exceeds the char budget it is truncated from the tail. A
session without a stored recap produces no block. An active session is
never attached this way (its content is already in the channel history).

| Variable | Purpose |
|---|---|
| `LAST_SESSION_CONTEXT_ENABLED` | Attach the last-session block at all (`1`/`0`, default on) |
| `LAST_SESSION_MAX_CHARS` | Char budget for the block (default `4000`; keep it small — recap only) |
| `SESSION_MERGE_PROMPT` | Override the AI session-log merge prompt at `/end_session` (empty = built-in default) |

## Commands

- `/upload_kb` — add files to the knowledge base
- `/list_kb_docs` — list indexed documents
- `/reindex_kb` — rebuild the vector index from scratch

## Useful settings

| Variable | Purpose |
|---|---|
| `KB_PATH` | Where KB files are stored |
| `CHUNK_SIZE` | Display-only "approx N chunks" estimate in `/list_kb_docs`; the chunker uses fixed character-based limits |
| `RAG_MAX_DOCS` | Max documents attached per query |
| `RAG_MAX_CHARS` | Max RAG context chars sent to the LLM |
| `RAG_WINDOW_LINES` | Window around each match anchor |
| `RAG_RETRIEVAL_METHOD` | `vector` or `keyword` |
| `RAG_QUERY_REWRITER` | Enable low-confidence query rewriting (`1`/`0`) |
| `RAG_REWRITE_MIN_SCORE` | Similarity threshold below which the rewriter triggers |
| `RAG_QUERY_MAX_EXPANSIONS` | Max alternative phrasings generated per rewrite |
| `RAG_REWRITE_BUDGET_SECONDS` | Wall-clock cap for the LLM rewrite call |
| `EMBED_BACKEND` | `local` (in-process CPU query embeddings, recommended) or `remote` (legacy backend `/embeddings`) |
| `LOCAL_EMBED_MODEL` | In-process query-embedding model (default `BAAI/bge-m3`) |
| `INDEX_EMBED_BACKEND` | Where index builds run: `backend` (GPU, default) or `local` (CPU) |
| `INDEX_EMBED_MODEL` | Exact name your backend serves for `/embeddings` during index builds (same model as `LOCAL_EMBED_MODEL`; on Unsloth use the GGUF server name, e.g. `gpustack/bge-m3-GGUF`, to hit the GPU path) |
| `RAG_HYBRID_ENABLED` | Fuse dense + BM25 via RRF (`1`/`0`, default on) |
| `RAG_MIN_ATTACH_SCORE` | Opt-in dense-similarity floor before fusion (`0` = off) |
| `RAG_MAX_CHUNKS_PER_FILE` | Max chunks attached per file (default `3`) |
| `RAG_ATTACH_FLOOR` | Per-chunk dense-score gate at attach time (`0` = off, default `0.50`) |
| `RAG_DENSE_TOP_K` | Candidate pool size for the dense leg before RRF fusion (default `48`; larger = better recall for hard lookups) |
| `RAG_LEXICAL_TOP_K` | Candidate pool size for the BM25 leg before RRF fusion (default `48`) |
| `RAG_EMBED_BATCH_SIZE` | Docs per `/embeddings` call during index builds (default `8`; the backend pads each batch to its longest sequence, so keep it modest for long/mixed content — raise only for short uniform chunks after measuring) |
| `OBSIDIAN_VAULTS` | Semicolon-separated `name=git_url` Obsidian vaults to mirror into `<KB_PATH>/<name>/` (empty = off); see `docs/obsidian-vault.md` | *(empty)* |
| `OBSIDIAN_CAMPAIGN_VAULT` | Name of the two-way vault holding session notes (bot commits + pushes there) | *(empty — all one-way)* |
| `OBSIDIAN_VAULT_PULL_INTERVAL` | Vault pull/commit period in seconds (`0` = startup pass only) | `300` |
| `SESSIONS_NOTES_DIR` | Where per-session folders live (point inside the campaign vault for two-way sync) | `<KB_PATH>/session_notes` |
