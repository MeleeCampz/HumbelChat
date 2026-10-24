# Design: AI executes Discord commands as tools (#47)

**Status:** DESIGN ONLY — not implemented. Predecessor work (agent mode, tool
loop, parallel batches, turn logging) is merged and in production.
Last updated: 2026-10-24.

## Problem

The AI can *read* the world (KB search, sections, sessions, item tables) but
cannot *act* in Discord: it cannot roll items for a channel, queue a reminder,
add a session note, or trigger TTS. Users currently have to leave `/ai` and
type the slash command themselves. The agent machinery needed for this (bounded
tool loop, parallel batches, error-as-data, telemetry) already exists — what's
missing is an **action surface** with the right safety properties.

## Constraints inherited from the codebase

1. **Slash handlers are interaction-shaped.** Every command in `main.py` takes
   a `discord.Interaction` and responds through it. The agent has no
   interaction mid-turn — actions need a *synthetic execution path* (run the
   handler logic, deliver results by posting as the bot in the turn's channel).
2. **Permission gates live inside handlers.** Some commands are admin-gated;
   the gate checks the *interaction user*. An AI-initiated call must pass the
   **invoking member** (the human who asked), never the bot — or the gate is
   bypassed.
3. **State mutations are channel/session-scoped.** Rolls, reminders and session
   notes mutate per-channel state (`bot_core/item_state.py`, `reminders.py`,
   `sessions.py`). The agent turn already knows its channel + active character
   — that context must be threaded into every action tool.
4. **The shared 27B model is the bottleneck.** Action tools add prompt tokens
   and possible extra turns; the #49 lesson (steer hard, batch aggressively)
   applies even more to actions.

## Proposed tool surface (phased)

### Phase 1 — read-only Discord state (low risk, high value)

| Tool | Wraps | Notes |
|---|---|---|
| `channel_state()` | item_state + sessions | One call: active character, excluded items, pending reminders, current session. Complements the KB tools with LIVE state (the gap that made a dedicated item DB look tempting — it's live state, not knowledge). |

No mutations; pure getter over in-memory state. Ships almost for free.

### Phase 2 — scoped mutations (the actual #47 ask)

| Tool | Wraps | Guardrails |
|---|---|---|
| `roll_items(names)` | /roll_items logic | Channel-scoped; respects the channel's exclusion list; result posted in-channel as a normal roll message. Idempotent-ish: re-rolling is harmless (fresh rolls), but the prompt discourages repeat calls. |
| `add_session_note(text)` | /session_notes | Only when a session is active in the channel; appends, never overwrites. |
| `queue_reminder(text)` | /remind_next_session | Delivered at next session start (existing mechanism) — no arbitrary timers in v1. |

Each tool: (a) resolves the turn's channel + invoking member up front, (b) runs
the SAME handler logic a slash command would (extract the core from the
interaction-shaped wrapper so both paths share it), (c) enforces the same
permission gate with the invoking member, (d) returns a short confirmation the
model weaves into its answer.

### Phase 3 — expressive actions (only if 1–2 prove out)

`speak(text)` (TTS), `start/end_session`, item include/exclude management.
Each re-evaluated against observed usage from `logs/ai_turns/`.

## Cross-cutting design decisions

- **One dispatcher, not N handlers-in-disguise.** A single
  `execute_command(name, args, ctx)` tool with an allowlist beats one tool per
  command: smaller prompt (the #49 lesson — every tool costs steering
  attention), one permission/audit path, easy to extend. The allowlist starts
  at Phase-1's `channel_state` + Phase-2 set; everything else is rejected with
  an error-as-data message.
- **Synthetic delivery, real messages.** Results post as ordinary bot messages
  in the turn's channel (rolls look like rolls). No interaction object is
  faked end-to-end — handlers get a thin `AgentContext` (channel, member,
  sender callback) instead; the slash wrappers adapt it to a real interaction.
- **Permissions never bypassed.** The invoking member rides along in
  `AgentContext`; each wrapped command runs its existing gate against that
  member. AI cannot escalate itself.
- **Audit = turn logger.** Every action call is already captured in
  `logs/ai_turns/` (tool calls + results). Action tools log an extra structured
  line (command, args, member, outcome) for greppability.
- **Limits.** Actions count against the existing per-turn tool-call budget (6);
  a separate cap (e.g. 2 mutations/turn) prevents roll-spam loops.
- **Off by default** behind `RAG_ACTION_TOOLS=0` (mirrors agent-mode gating),
  so the prompt/tool surface grows only where opted in.

## Testing strategy

- Unit: dispatcher allowlist + rejection; AgentContext threading; permission
  gate sees the invoking member (fake member admin/non-admin); idempotency of
  read tools; mutation caps.
- Handler-extraction tests: slash wrapper and agent path produce identical
  state changes for the same args (shared-core property).
- Live probe: `scripts/rag_agent_probe.py` extended with an action prompt
  ("roll a d20 and note it") asserting one dispatcher call + one in-channel
  message, on a throwaway channel.

## Open questions (user decisions before implementation)

1. Phase 2 scope: all three mutations, or start with `roll_items` only?
2. Should the AI be allowed to act in channels **without** an active
   character (rolls yes / session notes no)?
3. TTS (`speak`) in v1 or strictly Phase 3?

## Effort estimate

Phase 1: ~half a day (getter + tool + tests). Phase 2: 1–2 days per command
(handler extraction is the bulk; dispatcher/audit/context are one-time).
Phase 3: re-estimate after telemetry.
