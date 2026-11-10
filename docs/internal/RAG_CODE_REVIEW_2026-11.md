# RAG code review — 2026-11-10

Follow-up to [`RAG_PIPELINE_REVIEW_2026-10.md`](./RAG_PIPELINE_REVIEW_2026-10.md) (#74).
That review is still correct about architecture; this one adds what it did not look at —
the **agent tools**, the **cache/invalidation machinery**, the **contextual sidecar's key
design**, and **production log evidence** from the 2026-11-03 → 2026-11-10 window — plus a
review of the #77 re-embed change itself. Prior items F1–F10 are *not* restated; their
status is in §6.

Method: static read of `kb/{retrievers,index,vector_db,chunker,embedder,reranker,tools,
lexical,query_rewriter,reader,contextual_gen}.py` + `bot_core/{ai_client,agent_loop}.py`,
against the **live** artifacts: `data/knowledge/.vector_index_cache/vector_index.db`
(5 944 chunks / 89 source files / 2560-dim `qwen3-embedding-4b` / schema v6),
`context_lines.json` (5 749 entries), and `logs/dev.log`. Every number below is measured,
not estimated; the scripts in §7 reproduce them.

**Findings index**

| ID | Sev | One-liner |
|----|-----|-----------|
| R1 | HIGH | `list_sections(dir=…)` cannot match *any* file in the live index — latent dead tool |
| R2 | HIGH | The GPU image switch kills retrieval *inside* an in-flight agent rewrite (log-proved) |
| R3 | MED | When the reranker is offline, `RAG_ATTACH_FLOOR=0.50` silently changes unit (probability → cosine) |
| R4 | MED | Cache-identity guard validates the **query** model, not the model that **wrote** the vectors |
| R5 | MED | Contextual sidecar is keyed by content hash alone → cross-file duplicate chunks share one context line |
| R6 | MED | Every index mutation invalidates the numpy matrix *and* BM25: measured **0.91 s** added to the next query |
| R7 | LOW | Context lines measurably reshuffle the lexical leg (~1 in 5 of the lexical top-96 changes) — never A/B'd |
| R8 | LOW | The CRAG agent gate is blind exactly when retrieval quality is worst |
| R9 | LOW | `MAX_CHARS_PER_FILE` (24 000, a module constant) is out of scale with `RAG_MAX_DOCS × RAG_MAX_CHARS` |
| R10 | LOW | Rewriter failure is logged as `Query rewrite failed ()` — 12 occurrences, reason unrecoverable |
| R11 | INFO | Corpus coverage audit: the chunker loses ≤1.2 % of bytes everywhere except 3 merged session files (by design) |
| R12 | INFO | Framing overhead measured: 19 % of indexed text is breadcrumb + context line; ~7 % of the prompt budget |
| R13 | INFO | Config-hygiene inventory (dead vars, doc/example drift) with exact counts |
| R14 | INFO | #77 self-review: the per-chunk fallback can leave a *mixed-space* index, and the offline script can be shot in the foot by the live bot |

---

## 1. R1 — HIGH: `list_sections(dir=…)` is unreachable in practice

`kb/tools.py:658` matches the `dir` argument against the **file part of the chunk display
name**:

```python
for name, _content in idx.iter_chunks():          # iter_chunks yields d.display_name
    file_part = name.split(" [")[0] if " [" in name else name
    if dir and dir not in file_part:              # ← substring test against a BARE basename
        continue
```

`KBVectorIndex.iter_chunks()` yields `display_name`, and in the live index **every**
display name is a bare basename (`Warlock.md [Warlock -> Class Features]`):

```
display-name file parts: 79 total | 0 carry a directory prefix | 79 are BARE basenames
```

So the `dir` branch can never match. Executed against the live index:

```
dir='humblewood/DnDRules'                  -> "Nothing found for 'humblewood/DnDRules'"
dir='DnDRules'                             -> "Nothing found for 'DnDRules'"
dir='humblewood'                           -> "Nothing found for 'humblewood'"
dir='humblewood/DnDRules/books/Warlock.md' -> "Nothing found for …Warlock.md'"   # even a real path
file=Warlock.md                            -> Warlock.md (28 sections)            # works
file=humblewood/DnDRules/books/Warlock.md  -> Warlock.md (28 sections)            # works via _file_matches
```

Root cause is a **two-name-space** split that the tools layer silently straddles:

| Name | Shape | Produced by | Consumed by |
|------|-------|-------------|-------------|
| `display_name` (DB `doc_name`) | `Warlock.md [A -> B]` | chunker | `iter_chunks()` → `read_section`, `list_sections` |
| `retrieval_name()` | `humblewood/DnDRules/books/Warlock.md [A -> B]` | `KBVectorIndex` | all retrieval, `search_kb` output, `turn_logger` |

`_file_matches()` bridges the gap for `file=` (basename + stem fallback, `kb/tools.py:388`);
nothing bridges it for `dir=`. `iter_chunks()`'s own docstring advertises the *path* shape
("Display names look like `humblewood/DnDRules/…/equipment.md`"), which is what the tools
author coded against — it is wrong for this corpus.

Why it matters beyond "one arg is dead": the model is *told* about `dir` in
`tool_prompt_block()` ("List … the files + sections under a directory"), and the only
paths it has ever seen came from `search_kb`. When it uses `dir` as designed it gets
`Nothing found`, which reads as "the KB has no such area" and pushes it to answer from
memory — the exact failure class as §2.5 of the October code review. The model *has*
reached for `list_sections` twice in the recorded traffic (`2026-10-23 17:53` agent mode,
`2026-11-03 18:37` inside an image-prompt rewrite), and `Nothing found` appears nowhere in
the current window — but tool **arguments** are never logged (`AgentTurnMeta.tool_calls`
stores names only, and the turn record inherits that), so there is no way to tell whether
either call used `file=` (which works) or `dir=` (which cannot). Treat "not yet observed"
as "not observable", not as "not happening".

Fix (either, ~15 lines): iterate `retrieval_name()` instead of `display_name` in
`_tool_list_sections`, or match `dir` against `source_file` (available on `_DocEntry`)
instead of the display-name prefix. Also fix the `iter_chunks()` docstring, and consider
returning the section list for `dir` **plus** the file count under that prefix, since a
prefix match is otherwise indistinguishable from a typo.

---

## 2. R2 — HIGH: the image-mode GPU switch overtakes an in-flight agent rewrite

`AGENTS.md` states "Prompt rewriting ALWAYS happens before the switch". That holds for a
*single* sequential image request. It does not hold for two requests in flight, and the
logs show it happening on 2026-11-09:

```
19:38:27  prompt rewrite done   (3 iterations, tools=search_kb,read_section,read_section)  ← request A
19:38:34  prompt rewrite started: "I steh auf einem Meteroid…"                              ← request B
19:38:43  kb.embedder: … attempt 1/3 error: [Errno -2] Name or service not known
19:38:50  gpu mode -> image (22.7 s): reranker+embedder stopped, image model resident      ← switch
19:38:55  kb.embedder: … attempt 3/3 error: [Errno -2] Name or service not known
19:39:03  kb.reranker: Rerank failed ([Errno -2] …); using original order
19:39:10  kb.embedder: … attempt 1/3 error: …          ← expansion ranking, second dense call
19:39:22  kb.embedder: … attempt 3/3 error: …
19:39:26  kb.reranker: Rerank failed (…); using original order
19:39:26  agent_loop: agent fallback (wall-clock budget exceeded): single-turn answer over 6 messages
19:40:02  prompt rewrite done: … fallback=True (wall-clock budget exceeded)
19:40:47  gpu mode -> rag (64.5 s)
```

Request B's own KB tools were switched off *underneath it*: ~52 s of rewrite, zero tool
results, and a fallback answer built with no gathered context. Two structural problems:

1. **The switch is not gated on in-flight rewrites** — `bot_core/images.py` flips the card
   when *its* rewrite finished (A), while B is mid-loop. The invariant is per-request, not
   global.
2. **The failure ladder is not budget-aware.** `kb/embedder.py:106` retries 3× with
   `1.5 s × attempt` backoff per endpoint; a query pays it **twice** (query embedding +
   expansion ranking), and `kb/reranker.py` then does 2 passes × ⌈150/32⌉ = 10 HTTP
   attempts. Measured here: ≈25 s of embed retries + ≈8 s of rerank failure inside a
   `RAG_AGENT_BUDGET_SECONDS=45` loop → the budget is gone before a single tool call runs.
   A dead sidecar fails *instantly* at the socket level (`Name or service not known`), yet
   the code spends 4.5 s of sleep re-failing it, per call.

Cheapest fixes, in increasing scope:

- **Circuit-break the sidecars**: cache "unreachable" for ~10 s per endpoint (the probe
  already exists at startup in `preload_reranker`); a DNS-level failure should cost one
  attempt, not three, and should mark the dense leg skippable so the lexical leg answers
  immediately (it already survives alone — `reciprocal_rank_fusion([[], lex])` works).
- **Tell retrieval which mode the card is in.** `bot_core/images.py` already knows
  (`gpu mode -> image`); a module-level flag consulted by `_retrieve_vector` turns "retry
  a dead endpoint" into "run lexical-only, say so in the log".
- **Serialize image windows** (one rewrite+generate at a time, or defer the switch until
  in-flight rewrite tasks drain) — restores the documented invariant.

Related: `RERANK_TIMEOUT_SECONDS=60` with a stopped container is harmless (connect
refused), but if the port were firewalled rather than closed, each of those 10 attempts
would burn the full 60 s. Worth remembering before changing the compose network shape.

---

## 3. R3 — MED: the attach floor changes unit when the reranker is unavailable

`kb/retrievers.py` (#43) gates attachment on **rerank** scores:

```python
gate = rerank_gate if RAG_ATTACH_FLOOR_SOURCE == "rerank" else None
…
score_map = gate_scores if gate_scores is not None else dense_scores   # select_ranked_chunks
```

`rerank_gate` is `None` when the reranker never scored — including the routine case where
gpu-switch stopped `reranker-llama` (R2). `.env` sets `RAG_ATTACH_FLOOR=0.50`, so the same
number is then compared against **cosine similarity** instead of a Qwen3 relevance
probability, with no log line announcing the unit change. For this corpus the two scales
are nowhere near equivalent: a *junk* query measures `top=0.576` dense cosine
(`logs/dev.log:16101`, query = "This is just a simple test message to check if the ai
backend is working") — comfortably above the 0.50 floor — while a real chunk can sit at
0.45 cosine and be dropped. The "floor dropped every chunk → unfiltered selection" safety
net fired 4× in the current log window (all four on junk/test queries, i.e. the gate doing
its job *while the reranker was up*).

Options: make the floor source-qualified in settings (`RAG_ATTACH_FLOOR_RERANK=0.50`,
`RAG_ATTACH_FLOOR_DENSE=0.35`) and pick by which map is live; or keep one constant but
refuse to gate when the gate source is missing (log it). Either way the current silent
substitution deserves a comment, because the calibrated value is documented as a *rerank*
threshold.

Also worth noting for `RERANK_MODE=local` (the default in `kb/reranker.py`'s own
docstring, not `.env`): a local CrossEncoder returns **logits**, and
`if s > merged_scores.get(c, 0.0)` never stores a negative score, so negative-logit chunks
are absent from the gate map — and per `select_ranked_chunks`'s contract ("chunks with no
entry in the map are always kept"), the *worst* chunks become ungatifiable while logits in
(0, 0.5) get dropped. Harmless in the current http/jina setup (scores ∈ [0,1]); a trap if
someone flips the mode back to `local`.

---

## 4. R4 — MED: the cache-identity guard checks the wrong model name

`KBIndexStore` records and validates `self.model_name`, which comes from
`effective_embedding_model()` (= `EMBEDDING_MODEL` when `EMBED_BACKEND=remote`,
`kb/index.py:271`):

```python
if meta.get("embedding_model") != self.model_name:   # _cache_identity_ok, kb/index.py:1331
```

But the embedder that **writes** the vectors is built with a different variable
(`kb/index.py:408`):

```python
self._embedder = …(model_name=INDEX_EMBED_MODEL, …)
```

and `INDEX_EMBED_MODEL` defaults to `LOCAL_EMBED_MODEL`, not `EMBEDDING_MODEL`
(`config/settings.py:547`, with a comment that still says "must be the SAME model as the
query embedder (LOCAL_EMBED_MODEL)" — stale since `EMBED_BACKEND=remote` became production).
The only guard that exists (`kb/index.py:382-405`) compares `INDEX_EMBED_MODEL` against
`LOCAL_EMBED_MODEL`, for the CPU-fallback decision — never against the query model.

Consequence: with `INDEX_EMBED_MODEL` ≠ `EMBEDDING_MODEL` the build writes model-A vectors,
stamps metadata with model-B's name, `_cache_identity_ok` passes forever, and every query
compares B-space queries against A-space documents. No error, no rebuild, degraded
similarity. Currently latent — `.env` pins both to `qwen3-embedding-4b` explicitly — but a
single edit to `EMBEDDING_MODEL` (the natural way to try a new embedding model, cf.
`EVAL_QWEN3_EMBEDDING.md`) walks straight into it.

Suggested: one startup assertion `INDEX_EMBED_MODEL == effective_embedding_model()` unless
`INDEX_EMBED_FALLBACK_LOCAL` is explicitly forced, plus writing the *write-side* model name
into metadata and validating that.

Related live mismatch worth flagging while it is live: Strata currently serves
`unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M` while `.env MODEL_NAME` says
`qwen3.8-flash-next-iq3_xxs-DualGPU`. That is not a RAG bug, but it is the input to two
RAG features — the query rewriter (whose `model_is_loaded()` probe turns a slug mismatch
into a 90 s "extend the budget for the weight load" path instead of a 10 s fail —
`kb/retrievers.py:460-471`) and the #77 context-line fill.

---

## 5. R5 — MED: one context line, three files (sidecar key design)

The sidecar key is the hash of the chunk's **content** only — `kb/contextual_gen.py:220`:

```python
def _content_hash_of(doc): return _content_hash(doc.content)
```

Nothing scopes it to the file or the chunk's position. Where two files contain byte-
identical chunk text, the second occurrence silently inherits the first file's generated
framing — and `passage_text()` prepends that framing to the *vector* and to the *attached
text*, so the model is told the wrong provenance. Live occurrence:

```
colliding hash 2734d2a71ed2, 3 files:
  humblewood/DnDRules/books/The Griffon's Saddlebag (part 1 of 3).md
  humblewood/DnDRules/books/The Griffon's Saddlebag (part 2 of 3).md
  humblewood/DnDRules/books/The Griffon's Saddlebag (part 3 of 3).md
stored line: "Source: The Griffon's Saddlebag (5e24); D&D magic item index header
            collected from dnd-wiki.org and D&D Beyond on 2026-10-22."
```

One collision today (`distinct content hashes = 5 942` for 5 944 chunks), so this is a
design note rather than a live outage — but the corpus is built from per-book gap lists and
SSR dumps, where shared headers/stat blocks are the norm, and the cost of a wrong framing
is highest for the "which book is this from?" questions the bot is asked. Cheap hardening:
key the sidecar as `f"{source_file}\0{content_hash}"` (with a one-time migration), or at
minimum refuse to reuse a stored line whose leading path token doesn't match the chunk's
`source_file`.

---

## 6. R6 — MED: mutation → full cache rebuild (measured 0.91 s of next-query latency)

Nothing mutates `_docs` in place: `sync_changes`, `update_single_document`, `_build_fresh`
and the new `reembed_contextualized` all publish a **new** `KBVectorIndex`
(`kb/index.py:536, 800, …`). Both caches are keyed on object identity, so a single added
file invalidates everything:

```python
# kb/retrievers.py:558 — module-global, keyed on (docs list identity, len)
if _bm25_cache is not None and _bm25_cache[1] is docs and _bm25_cache[2] == len(docs): …
# kb/vector_db.py _ensure_matrix — cached on (mat, norms, docs object)
```

Measured on the live 5 944-chunk / 2560-dim index:

```
first _ensure_matrix: 457 ms
first BM25 build:     456 ms
dense _rank warm:     3.3 ms
BM25 warm:            0.003 ms
```

≈0.91 s of rebuild lands on the **first query after every mutation** — `/kb add`, a
`sync_changes` hit, or the #77 background fill (which replaces the index after every
batch). Both builds run inside `asyncio.to_thread`, so the event loop is not blocked
(that part of the design is correct), but: the user pays it, and `_bm25_cache` is an
unlocked module global — two concurrent first-queries after a mutation can each build the
same 456 ms index in parallel. Prior finding F6 (cold first query) is the same machinery
at larger scale; the point added here is that the *steady-state* cost is not cold-start-only
— it recurs on every mutation, and #77 deliberately triggers it ~35 times on first run.

Worth doing: (a) rebuild both caches eagerly inside the mutation (off the query path — the
index is already fully known there), and (b) guard `_get_bm25` with a lock or build it
inside `KBVectorIndex`. Incremental BM25 is the bigger win but not worth the complexity at
5.9 K chunks.

---

## 7. R7 — LOW: contextual framing measurably moves the lexical leg

BM25 is built over `passage_text()`, so context lines are in the lexical corpus by design
(the dense and lexical legs "see the same text"). Nobody has measured the side effect. It
is not negligible:

```
term           df(with ctx)  df(content-only)  idf_with  idf_wo
humblewood          405             49           3.61    5.78
md                 1296              4           2.28    8.19
monster             471            127           3.45    4.82
magic              2621           2600           1.24    1.25

query                     positive(w/ctx)  positive(plain)  top-96 overlap
"magic items"                   2 665          2 634          79 / 96
"chain mail armor AC"           1 059          1 056          81 / 96
"session 7 ambush"                790            777          74 / 96
"humblewood"                      405             49          42 / 96
```

Many context lines open with the file path ("humblewood/DnDRules/DnD5_5/magic-items.md:
…"), so path tokens enter the term statistics: `md` goes from 4 documents to 1 296.
Roughly 1 in 5 chunks in the lexical top-96 changes versus content-only BM25. That is not
obviously good or bad — the same lines are what the dense leg is now embedded on (#77), and
the reranker sees them too — but the lexical leg's behaviour changed under a *vector*
feature, and `scripts/rag_probe*.py` already exist to A/B it. Worth one probe run before
the sidecar gets any bigger.

Secondary detail from the same code: `_lexical_ranking` materialises *every* positive-score
chunk before truncating — measured 1 124 tuples / 1.62 MB of transient passage text for
"ability check" (whole leg: 2.4 ms). Fine at this corpus size; it stops being fine if
`RAG_LEXICAL_TOP_K` and the corpus both grow.

---

## 8. R8 — LOW: the agent quality gate is blind when retrieval is weakest

`bot_core/ai_client.py:960`:

```python
if rerank_top is not None and rerank_top < RAG_AGENT_SCORE_GATE:   # 0.55
```

`rerank_top` is `None` whenever the reranker never scored — the R2 condition, i.e. the
worst-retrieval state the gate exists to catch. Masked in production only because
`RAG_AGENT_MODE=always`. If `auto` is ever reinstated (it is the mode the October review
recommends for cost, F3), a downed reranker silently disables the quality trigger. Same
shape for `meta["dense_top"] = dense_scores.get(final_ranked[0][0])`: `None` whenever the
top chunk is a lexical-only winner, which is the common case during image windows.
`turn_logger` therefore records `rerank_top: null` / `dense_top: null` for exactly the
turns worth analysing — worth a third meta key (`retrieval_legs: dense|lexical|both`) so
post-hoc analysis can tell "no reranker" from "no answer".

---

## 9. R9 — LOW: the per-file char cap is a constant out of scale with the prompt budget

`kb/retrievers.py:212`: `MAX_CHARS_PER_FILE = 24_000` is a **module constant** — not a
setting, unlike its neighbours `RAG_MAX_CHUNKS_PER_FILE` and `RAG_ATTACH_FLOOR`. Production
budget is `RAG_MAX_DOCS=4`, `RAG_MAX_CHARS=32_000`. So one file may claim 75 % of the
global budget, and `_build_rag_context` (`bot_core/ai_client.py:198`) *skips* (not truncates)
whatever no longer fits:

```python
if chars_used + len(doc_block) > max_chars:
    log.info("RAG: skipped doc '%s' (%d chars) — remaining budget %d chars", …)
    continue
```

10 `skipped doc` + 10 `budget cap reached` events in the current log window — the cap bites
in real traffic. Effect: whether `RAG_MAX_DOCS=4` is honoured depends on document *sizes*
in rank order, and rank-4 can survive while rank-2 is dropped because rank-2 is bigger.
Not wrong (skip-then-continue is the right shape), but the three numbers are tuned
independently — `MAX_CHARS_PER_FILE` should become a setting and be documented as a
fraction of `RAG_MAX_CHARS`, or the whole thing gets one budget split top-down.

---

## 10. R10 — LOW: failure logging loses the reason

```
12 × Query rewrite failed ()
```

`kb/retrievers.py:474`: `except Exception as exc: logger.warning("Query rewrite failed (%s)", exc)`.
`asyncio.TimeoutError` from the surrounding `wait_for` stringifies to nothing, so 12 real
failures are indistinguishable from backend errors, model-not-found, or empty responses.
`bot_core/agent_loop.py` already uses `type(exc).__name__` for the same problem elsewhere —
use it here (and include the budget in the message). Cheap, and it is the log line a future
tuning decision will be made from.

---

## 11. R11 — INFO: chunker coverage audit (nothing is being dropped)

Worth recording because "does the chunker lose data?" is the question this review was
expected to answer, and the answer is measured and mostly clean.

Per-file `disk → indexed` character loss, all 89 indexed files (worst 14):

```
file                                                     disk  indexed   Δ%   chunks
SessionLogs/2026-11-03_01_Nach dem Kampf/notes.md      11 739    9 724  17.2     12
SessionLogs/2026-10-20_02_We fucked up…/notes.md       11 502    9 692  15.7     13
SessionLogs/2026-09-09_01_DerAnfang/notes.md            8 495    7 342  13.6      1
CampainLogs/_private.md                                    18       17   5.6      1
CampainLogs/03_11_2026 Nach dem Kampf.md                7 563    7 469   1.2      1
… everything below 1.2 %
```

The three >13 % files are the **`## Notes` strip (#75)** working as designed
(`strip_merged_session_notes` — the canonical `## Session Log (combined…` supersedes raw
bullets). Text-level audit: 163 source lines are not reproduced verbatim anywhere in their
file's chunks; **153 of them are markdown header lines** whose text survives inside the
` -> ` breadcrumb (e.g. `## Vampires` → chunk starts `Monsters A–Z -> Vampires -> Vampire
Familiar`; "vampires" matches 20 chunks, "black dragons" 12, "fungi" 10) — so header
*markers* are dropped, header *content* is retrievable. Only 10 non-header lines are
genuinely absent: 6 stripped Notes bullets, 1 `(no notes)` placeholder, 3 short lines in
`CampainLogs/03_11_2026 Nach dem Kampf.md` (`LEVEL UP: 6`, `Marvin Upgraden`, one narrative
line — 94 chars total).

The stripped bullets do **not** lose facts, but they do lose *spellings*: the raw bullet
`500G Kopfgeld auf Cowurm names "Lucky"` is superseded by the Overview's version spelled
`Corvum/Corwum/Corwurm`, and the indexed corpus contains **0** chunks matching `cowurm`. A
user who asks "what was Cowurm's bounty?" gets nothing from retrieval. This is the
canonical-name-glossary problem (F9) showing up in a measurable form: the #75 strip removes
the variant spellings that were the only route back to the fact. A glossary/alias file is
now the concrete fix, not just an idea.

---

## 12. R12 — INFO: what framing costs

| Quantity | Measured |
|---|---|
| Indexed corpus | 5 944 chunks, 5 912 248 chars, p50 769, p95 2 452, max 7 616 |
| Breadcrumb text (`A -> B -> C` prefixes) | 5 856 chunks, 518 849 chars (8.8 %), longest 257 |
| Context lines | 5 749 entries, 621 710 chars, avg 108, p50 107, p95 141, max 185 |
| **Framing share of indexed text** | **≈19 %** |
| Prompt cost when 20 chunks attach | ≈2 200 chars ≈ 7 % of `RAG_MAX_CHARS=32 000` |

19 % of everything embedded and BM25-indexed is navigation, not source. That is a *deliberate*
cost (it is what makes a chunk self-describing) and it is currently the cheapest kind of
quality this system buys — but it should be stated somewhere when someone proposes raising
`RAG_LEXICAL_TOP_K`, adding parent chunks (Tier 2), or switching to a smaller embedding
model. The one asymmetry worth knowing: breadcrumbs reach the **index** but not the
**prompt** (attachment groups per file, so the model sees `--- file ---` headers, not the
parent chain), whereas context lines reach both.

---

## 13. R13 — INFO: config and cache hygiene, counted

- Live `.env` carries **17 variables nothing reads**: 15 `STT_*` + `RECORDINGS_DIR` +
  `SESSION_SUMMARY_PROMPT`. The voice/STT pipeline was deleted in `669b36a`
  ("remove the voice recording + STT pipeline completely (#38)"); the config outlived it.
  (Already noted as F7 in October — still not done.)
- **14 variables documented in `docs/configuration.md` are absent from `.env.example`**:
  `AI_REQUEST_TIMEOUT`, `EMBED_URL`, `EMBED_QUERY_INSTRUCT`, `IMAGE_GEN_TIMEOUT_S`,
  `IMAGE_LOAD_TIMEOUT_S`, `MAX_TOKENS`, `MAX_TOKENS_HARD_CAP`, `RERANK_API_BASE`,
  `RERANK_API_FORMAT`, `RERANK_INSTRUCT`, `RERANK_MAX_CHARS`, `RERANK_MODE`,
  `RERANK_TOP_K`, `RERANK_USE_EXPANSION` — i.e. the entire reranker family, which is the
  most safety-critical knob set in the system (R3).
- **9 variables read by `config/settings.py` appear in neither** `.env.example` nor
  `docs/configuration.md`: `DEFAULT_ITEM_TABLE`, `EMBED_API_KEY`,
  `INDEX_EMBED_FALLBACK_LOCAL`, `ITEMS_DIR`, `RAG_REWRITE_LOAD_BUDGET_SECONDS`,
  `SESSION_LOG_CHRONOLOGY_RULES`, `SESSION_MERGE_PROMPT`, `STAT_BLOCK_FORMAT_RULES`,
  `TTS_BACKEND`. `INDEX_EMBED_FALLBACK_LOCAL` is the one that matters (R4's guard rail).
- Sidecar housekeeping: **53 orphan entries** (hashes of chunks no longer indexed) and, at
  review time, **246 chunks with no context line** — the startup warning agrees exactly
  (`Contextual sidecar covers 5698/5944 indexed chunks`, `logs/dev.log:17663`). Both are now
  addressed by #77's fill + `scripts/contextual_reindex.py --prune-orphans`; run the prune
  once after the first self-healing fill so the sidecar and index stop drifting.

---

## 14. R14 — INFO: reviewing the #77 change itself

`feat/contextual-reembed` (`6d910f9`) — three things a future reader should know that the
commit message does not say:

1. **The per-chunk fallback can produce a mixed-space index.** `_encode_contextual_passages`
   retries a failed batch chunk-by-chunk so one oversized passage cannot fail its 11
   siblings. If that one chunk is genuinely un-embeddable, it *keeps its old content-only
   vector* while its file's other chunks get contextualized vectors — the exact inconsistency
   #77 exists to remove. It is logged, but there is no count, no threshold, and the chunk is
   not quarantined. If `reembed_contextualized` ever reports "N re-embedded < M requested",
   treat the difference as a retrieval-quality bug to investigate, not a warning to ignore.
2. **Replacing the docs list is the invalidation mechanism, deliberately.** `reembed_…`
   assigns a new `_docs` list so `_ensure_matrix` (identity-keyed) and `_bm25_cache`
   (identity+len keyed) both rebuild. That is R6's 0.91 s, once per fill batch (~35 on the
   first run). It is the correct trade, but it means the fill makes the *next* query slow.
3. **The offline script and the live bot must not both write.** `contextual_reindex.py` now
   warns about it, because the bot's `shutdown()` runs a full `_persist_index_to_db` and will
   happily write its stale in-memory vectors over a script's fresh rows. Until
   `CONTEXTUAL_BACKGROUND_FILL=1` is actually in the *running container* (`.env` is baked at
   container create — `docker compose up -d --build` is required, `--env-file` reloads are
   not a thing here), the script is the only safe way to fill.

---

## 15. What is solid (do not "fix")

Restating only what this pass independently re-verified, because these are the parts someone
refactoring is most likely to break:

- **No mutation ever touches `_docs` in place.** Every path publishes a new index under
  `_mutation_lock`, and `_load_inner()` is the documented lock-free core (`asyncio.Lock` is
  not re-entrant — `load()` from inside a mutation would deadlock; nothing does).
- **`_persist_upsert` is per-`source_file`** (P2 #20) and `content_hash` / `file_hash` stay
  byte-stable across re-embeds, so the P2 #19 fast path survives #77 unchanged.
- **Rebuilds never delete the cache before replacing it** (`_ignore_cache` + atomic temp-file
  swap) — a failed rebuild leaves a usable old cache. Verified in both load branches.
- **Every degradation path degrades to something useful**: rerank failure → original order;
  lexical failure → dense only; empty dense (embedder down) → lexical-only via RRF; floor
  kills everything → unfiltered re-selection; agent loop any error → single-turn answer over
  the tool transcript. The empty-dense case is what keeps the bot answering during image
  windows, and it is undocumented.
- `RERANK_USE_EXPANSION`'s max-fusion is correct under partial failure *because* a 0.0 score
  never beats the default (`s > merged_scores.get(c, 0.0)`), which is what the comment
  claims — checked line by line.

---

## 16. Status of the 2026-10 review's items

| Item | Status at 2026-11-10 |
|---|---|
| F1 dedup key | Fixed (#75) — re-verified, `embedding_map()` keys are `retrieval_name` |
| F2 195 chunks uncontextualized, fill off | Closed in code by #77 (fill + re-embed); **live container still needs `docker compose up -d --build`** for `CONTEXTUAL_BACKGROUND_FILL=1` to exist in the process env |
| F3 answer generated non-thinking | Still open (unchanged) |
| F4 rerank window 1 200 chars | Still open; note `RERANK_TOP_K` is now 150, so window truncation now applies to a wider candidate pool |
| F5 `reranker-qwen` profile | Closed — `docker-compose.yml:135` now carries `profiles: ["vllm-rerank"]` (added with the 2026-11 GPU split); rollback path documented in the service comment |
| F6 cold first query | Still open, and now recurring per-mutation (R6) |
| F7 docs/env drift | Partially done (this branch rewrote the `docs/rag.md` sidecar section — it described a `chunk_contexts` SQL table and a `.generated` column that never existed); `.env` dead-var cleanup and `.env.example` backfill still open (R13) |
| F8 prompt budget vs runtime ctx | Unchanged |
| F9 glossary / aliases | Still open, with new evidence (R11: `cowurm` → 0 indexed matches) |
| F10 micro-perf | Unchanged; `embedding_map()` per-call rebuild is still ~ms |

---

## 17. Suggested order

| # | Action | Effort | Risk |
|---|--------|--------|------|
| 1 | R1: `list_sections` matches `source_file`; fix `iter_chunks` docstring; log tool **args** (name-only telemetry makes this class of failure undiagnosable); regression test with a *path* arg | ~30 min + test | none |
| 2 | R10: `type(exc).__name__` in the rewrite-failure log | 2 min | none |
| 3 | R2a: embedder/reranker circuit breaker (~10 s unreachable cache) + skip dense leg instead of retrying a dead host | ~45 min | low — strictly less latency, same fallbacks |
| 4 | R2b: expose the gpu-switch mode to `kb.retrievers` (and log "image window: lexical-only retrieval") | ~30 min | low |
| 5 | R3: split the attach floor into `…_RERANK` / `…_DENSE` values, or don't gate when the gate source is gone | ~20 min | medium — changes what gets attached |
| 6 | R4: startup assertion `INDEX_EMBED_MODEL == effective_embedding_model()`; record write-side model in metadata | ~15 min | none (fails loudly instead of silently) |
| 7 | R6: rebuild matrix + BM25 inside the mutation, and lock `_get_bm25` | ~30 min | low |
| 8 | R5: sidecar key = `source_file` + content hash (one-time migration) | ~1 h | medium — touches the sidecar format; do after the #77 fill lands |
| 9 | R11/F9: alias glossary at the vault layer (`Cowurm`/`Corwurm`, `Column`/`Colum`) | data task | none for code |
| 10 | R13: `.env` dead-var removal, `.env.example` backfill, `--prune-orphans` once | ~30 min | none |
| 11 | R7: lexical-leg probe with/without context lines via `scripts/rag_probe*.py` | ~1 h | measurement only |

R1 and R2 are the two I would do before anything else: one is a documented tool that
provably cannot work, the other is already costing the campaign degraded image prompts and
agent turns, and both are visible in the logs I quoted.

---

## 18. Reproduction scripts

Three throwaway scripts were used for the measurements; kept here as recipes rather than
files in the repo (they read the live cache read-only):

1. **Corpus/coverage** — chunk count, size percentiles, breadcrumb and context-line totals,
   per-file disk→indexed loss, per-file missing-line audit (distinguishes header-marker
   loss from content loss).
2. **Tool repro** — loads `KBVectorIndex.from_entries` from `vector_index.db`, monkeypatches
   `kb.retrievers._ensure_index_store`, and calls `kb.tools._tool_list_sections` /
   `_tool_read_section` with basename vs path vs directory arguments (R1's evidence).
3. **Cache cost** — times `_ensure_matrix`, `_get_bm25`, `_rank`, `_lexical_ranking` on the
   live index, and BM25 `df`/`idf` with and without context lines (R6, R7).

Load pattern all three use (no GPU or compose access needed from the sandbox):

```python
conn = sqlite3.connect("data/knowledge/.vector_index_cache/vector_index.db")
rows = conn.execute("SELECT source_file, doc_name, content, embedding FROM document_index")
idx = KBVectorIndex.from_entries([(r[1], r[2], r[0]) for r in rows],
                                 [_unpack_embedding(r[3]) for r in rows])
```
