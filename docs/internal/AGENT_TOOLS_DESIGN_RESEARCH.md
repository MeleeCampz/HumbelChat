# Agent-Mode Tool System — Design Research & Industry Comparison

_Dated 2026-10-24. Companion to `RAG_ARCHITECTURE_RESEARCH.md` §3 (the design)
and `MODE_COMPARE_ALWAYS_VS_AUTO.md` (the measurement). This doc compares what
we shipped against how the industry builds the same thing, and records why our
deviations are deliberate._

## 1. What we have

- **Fast path first, agent on top.** The default turn is a single completion
  over attached RAG context (zero overhead). Agent mode (`RAG_AGENT_MODE`)
  layers a bounded tool loop on top when triggered (CRAG-style score gate) or
  forced (`always`).
- **Five in-process tools** over the existing vector index — no new services,
  no new VRAM: `search_kb` (hybrid dense+BM25, reranked), `read_section`
  (full chunk / session folder), `list_sections` (outline), `find_in_kb`
  (exact presence, complete list), `list_sessions` (chronological roster).
- **Bounded loop**: ≤N decision turns, ≤6 tool calls total, 45 s wall-clock
  budget; ANY failure degrades to a single-turn answer over the gathered
  transcript. Tool errors are data, never fatal.
- **Parallel batches** (2026-10-24): a decision turn may emit an ARRAY of
  independent calls; the loop runs the batch via `asyncio.gather` — wall
  clock ≈ slowest call, not the sum.

## 2. Comparison against industry patterns

### 2.1 Parallel tool calls — we match the recommended pattern

Every major provider (OpenAI, Anthropic, Gemini) now supports parallel
function calling natively, and production guides report **40–70% latency
cuts on multi-step tasks** when batches execute concurrently instead of
sequentially. The standard execution advice is exactly what we implemented:
"execute tools with `asyncio.gather` or a bounded semaphore — sequential
await defeats the whole point."

Two deliberate deviations, both forced by our stack:

- **JSON-array protocol instead of native function calling.** Our backend is a
  local 27B GGUF model via an OpenAI-compatible endpoint without reliable
  native tool-use; the fenced-JSON protocol + validator is the robust
  equivalent (and keeps every turn inspectable in logs).
- **Single calls still allowed.** The same guides warn that parallelism
  "occasionally reduces reasoning quality on tightly coupled tool chains" —
  dependent steps must serialize. Our protocol lets the model choose: one
  object when step B needs step A's output, an array when independent.

### 2.2 Agentic RAG — we sit at "tool use", gated like CRAG

The standard taxonomy of agentic retrieval has four patterns: reflection,
planning, tool use, and multi-agent collaboration. Production guides pair
these with the Self-RAG / Corrective-RAG (CRAG) research line: retrieve,
*grade the retrieval*, retry or answer. Our trigger is precisely the CRAG
gate — top rerank score below 0.55 means "retrieval itself is unsure"
(calibrated on our benchmark: true hits ≥ 0.65, negatives ≤ 0.36) — plus an
enumeration-shape heuristic (#48 pending).

Where frameworks (LangGraph `recursion_limit`, OpenAI Agents SDK
`max_turns`) and we converge: **two-layer limiting** — a soft iteration/turn
cap for shape and a hard budget as the safety net. Our twist is that the
WALL-CLOCK budget (45 s) is the primary governor, tuned to Discord user
patience, with the call cap (6) only preventing degenerate loops. When cheap
CPU tools (`find_in_kb`, `list_sessions` — milliseconds) joined the set, we
raised the cap 4→6 because it stopped being the binding constraint; the
budget is what still bounds latency.

Deliberately NOT adopted: planning steps, self-reflection loops, multi-agent
decomposition. Each costs full extra completions on a single shared 27B
model that also serves the user's chat — the latency/cost tradeoff (the same
tradeoff Anthropic's "building effective agents" guidance flags: "find the
simplest solution possible, and only increase complexity when needed") does
not pay off for Discord-turn times.

### 2.3 Lexical + semantic as COMPLEMENTARY tools — the coding-agent lesson

Production coding agents (Claude Code et al.) ship both a lexical grep and a
semantic/code search, because they answer different question shapes — the
same split we now have:

| Question shape | Tool | Why the other can't |
|---|---|---|
| "tell me about X" (approximate, ranked) | `search_kb` (dense+BM25, reranked) | — |
| "where/which/all X" (exact, complete) | `find_in_kb` | top-k ranking *structurally* omits matches past the cutoff with no signal — enumeration needs exhaustiveness |

This mirrors our own observed failure class (cross-log enumeration in the
mode-compare eval) rather than speculation — and follows the industry rule of
adding a tool only behind an observed failure.

### 2.4 Tool count & over-invocation — the open problem

Every tool costs ~150 tokens of schema on every agent turn, and more tools
widen the over-invocation surface (measured: 86% invocation rate under
`always`, incl. redundant verification searches — backlog #49). Industry
guidance is consistent: keep the set small, composable, and each tool tied
to a real question class. We hold at five tools; candidates are gated on
observed failures (`list_commands` with #49; command-execution tools are
#47's separate design project).

## 3. Sources

- AI Workflow Lab, "Parallel Tool Calls in Production: OpenAI, Anthropic, and
  Gemini Function Calling Compared (2026)" — latency figures, gather/semaphore
  execution advice, parallel-vs-serial quality caveat.
- Vectoring AI, "Agentic RAG: When Retrieval Needs Reasoning" — the four
  agentic patterns; Self-RAG / CRAG research line; LangGraph/LlamaIndex
  production implementations.
- Anthropic, "Building Effective Agents" — simplest-solution-first guidance;
  workflows-vs-agents distinction; single-call+retrieval suffices for many
  applications (our fast path).

## 4. Open items this comparison reinforces

1. **#48** — cross-log enumeration trigger rule (the `find_in_kb`-shaped
   questions the gate currently misses); re-run `scripts/rag_mode_compare.py`.
2. **#49** — "when NOT to call tools" prompt guidance; measure invocation-rate
   drop with unchanged answers.
3. **Batch-adoption telemetry** — now that arrays are supported, watch
   `logs/ai_turns/` for how often the model actually batches (it may prefer
   single calls; if so, the parallelism win only lands when we nudge it in
   #49's prompt work).
