"""Automatic query rewriting for enhanced RAG retrieval.

Uses a lightweight LLM call to dynamically expand user queries based on the
KB domain context — no hardcoded synonym lists.  The rewriter is invoked by
``kb.retrievers`` **only for low-confidence vector queries** (top similarity
below ``RAG_REWRITE_MIN_SCORE``), so confident queries pay nothing extra.

Example flow:
    User: "What do they eat in Humblewood?"
    LLM Rewrites: "Humblewood food sources, diet, menu, ingredients"
    Retrieval merges the original + expansion rankings via RRF.

Configuration
-------------
RAG_QUERY_REWRITER (bool, default 1) — Toggle query rewriting on/off
RAG_REWRITE_MIN_SCORE (float, default 0.35) — Trigger threshold (retrievers)
RAG_QUERY_MAX_EXPANSIONS (int, default 3) — Maximum number of expanded terms
RAG_REWRITE_BUDGET_SECONDS (int, default 10) — Wall-clock budget for the call

Usage
-----
    from kb.query_rewriter import create_query_rewriter

    rewriter = create_query_rewriter()
    expanded_queries = await rewriter.expand("What's the time system in Humblewood?")
    # Returns: ["original query", "Humblewood time mechanics", ...]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger("kb.query_rewriter")


class QueryRewriter:
    """Dynamically expand queries using the configured LLM backend."""

    MAX_EXPANSIONS = 3
    # Per-call HTTP timeout. Must exceed BOTH RAG_REWRITE_BUDGET_SECONDS and
    # RAG_REWRITE_LOAD_BUDGET_SECONDS (90, the outer asyncio.wait_for budgets
    # and the true caps): a cold :8888 chat-model load can take upwards of a
    # minute before the first byte, so a 60 s read timeout killed the once-per-
    # session cold-load rewrite at the HTTP layer before the extended budget
    # could help (the "first call after unload always fails" symptom).
    # History: 10 s killed mid-swap rewrites (2026-10-15) → 60 s covered warm
    # swaps (~20 s to first byte) but not cold loads → 120 s (2026-10-20).
    REWRITE_TIMEOUT_SEC = 120
    REWRITE_PROMPT_TEMPLATE = """\
You are a search query expansion assistant. The knowledge base is written in ENGLISH (official D&D 5e SRD text plus campaign lore) about {kb_domain}. Your task is to generate additional search terms that capture related concepts, synonyms, and contextual meanings of the user's original query.

Rules:
- Generate exactly {max_expansions} alternative queries (one per line)
- Each expansion should be a short phrase (3-8 words) suitable for keyword/semantic search
- Write EVERY expansion in ENGLISH using standard D&D 5e terminology
- If the original query is in another language (e.g. German), translate its key terms into English D&D 5e vocabulary first (e.g. "Rüstung" → "armor", "Zauberspruch" → "spell", "Heiltrank" → "potion of healing", "Statblock" → "stat block")
- Focus on synonyms, related concepts, and contextual meanings — not rephrasing the entire question
- Each expansion must preserve the SPECIFIC mechanics asked about — the action type, the trigger condition, and the effect (e.g. "ranged attack", "on a miss", "does not reveal position") — not just generic topic words
- Give each distinct mechanic/condition in the query its own expansion line
- DO NOT repeat book titles, source books, or publication names from the original query (e.g. a book name in the question is a SOURCE hint, not the item's name). Describe the mechanics asked about instead
- Stay strictly within the domain: {kb_domain}
- DO NOT include the original query in your output
- Output ONLY the expansions, one per line, with no numbering or extra text

Original query: "{original_query}"

Expansions:
"""  # noqa: E501

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        kb_domain: str = "Humblewood fantasy worldbuilding",
        max_expansions: int = MAX_EXPANSIONS,
        enabled: bool = True,
        model_slug: str = "",  # empty = use configured default
    ) -> None:
        self._client = client
        self.kb_domain = kb_domain
        self.max_expansions = max(1, max_expansions)
        self.enabled = enabled
        self.model_slug = model_slug or ""

    # ── Public API ─────────────────────────────────────────────────────

    async def expand(self, original_query: str) -> list[str]:
        """Expand *original_query* into related search terms.

        Returns a list containing the original query + rewritten expansions.
        Always includes the original as the first element for fallback.
        """
        if not self.enabled or not original_query.strip():
            return [original_query]

        expansions = await self._rewrite(original_query)

        # Filter out duplicates and empty strings, preserve original at front
        seen: set[str] = {original_query.lower().strip()}
        results: list[str] = [original_query]
        for exp in expansions:
            if exp.strip() and exp.strip().lower() not in seen:
                seen.add(exp.strip().lower())
                results.append(exp.strip())

        # Limit total expansions (keep original + N terms)
        return results[: self.max_expansions + 1]

    # ── Rewrite Logic ──────────────────────────────────────────────────

    # ── Load-state probe ────────────────────────────────────────────────

    async def model_is_loaded(self) -> bool | None:
        """Is the rewrite target model currently resident on the backend?

        Backends that expose load state (LM Studio / Unsloth Studio: ``GET
        /v1/models`` entries carry a ``loaded`` flag) let callers distinguish
        "model cold — this call will pay the weight-load cost" from "model warm
        — a slow call is a real stall".  Returns ``None`` when the state is
        unknown (no client/slug, request failure, or a backend without the
        field); callers must fall back to default behaviour in that case.
        """
        if not self._client:
            return None
        slug = self.model_slug
        if not slug:
            from config.settings import DEFAULT_MODEL
            slug = DEFAULT_MODEL or ""
        if not slug:
            return None
        try:
            import httpx

            base = str(self._client.base_url).rstrip("/")
            async with httpx.AsyncClient(timeout=5) as hc:
                r = await hc.get(
                    f"{base}/models",
                    headers={"Authorization": f"Bearer {self._client.api_key}"},
                )
                r.raise_for_status()
            for entry in r.json().get("data", []):
                if entry.get("id") == slug and isinstance(entry.get("loaded"), bool):
                    return entry["loaded"]
        except Exception:  # noqa: BLE001 — probe is best-effort by design
            pass
        return None

    async def _rewrite(self, original_query: str) -> list[str]:
        """Generate expansions via LLM call to the configured backend."""
        if not self._client:
            logger.warning("No OpenAI client configured for query rewriting")
            return []

        if not self.model_slug:
            from config.settings import DEFAULT_MODEL
            self.model_slug = DEFAULT_MODEL or ""

        prompt = self.REWRITE_PROMPT_TEMPLATE.format(
            kb_domain=self.kb_domain,
            max_expansions=self.max_expansions,
            original_query=original_query,
        )

        # Qwen3-style hybrid-thinking models spend their ENTIRE 256-token budget on
        # reasoning and return no content ("full token budget on reasoning"). This is
        # a short utility call — disable thinking. Measured: 0.5s with it off vs
        # 5.7s of pure thinking (see docs/internal/RAG_ANALYSIS_2026-09-27.md).
        #
        # Send ONLY enable_thinking=False. On the Unsloth/llama.cpp backend (:8888)
        # top-level reasoning_effort="none" is actively HARMFUL: measured 38.6 s for
        # a ~50-token expansion vs 1.2 s with enable_thinking alone (it defeats a
        # fast generation path), and in the combo it overrode enable_thinking so the
        # model thought its way through the whole 256-token budget — the exact
        # failure of 2026-10-15 ("full output budget on reasoning", TQ case).
        # A backend that rejects the param (older LM Studio) gets one plain retry.
        try:
            response = await self._client.chat.completions.create(
                model=self.model_slug,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low for consistent, focused expansions
                max_tokens=256,
                timeout=self.REWRITE_TIMEOUT_SEC,
                extra_body={"enable_thinking": False},
            )
        except Exception:
            # Backend rejected the param — retry plain.
            try:
                response = await self._client.chat.completions.create(
                    model=self.model_slug,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=256,
                    timeout=self.REWRITE_TIMEOUT_SEC,
                )
            except Exception as exc:
                logger.warning("Query rewrite attempt failed (%s); returning no expansions", exc)
                return []

        # P1 #8: empty `choices` raises AIBackendError → caught below →
        # no expansions (instead of an IndexError from `choices[0]`).
        try:
            from bot_core.errors import extract_reply_text
            content = extract_reply_text(response, default="")
        except Exception as exc:
            logger.warning("Query rewrite attempt failed (%s); returning no expansions", exc)
            return []
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        return lines[: self.max_expansions]


# ── Module-level convenience ───────────────────────────────────────────

def create_query_rewriter(model_slug: str = "") -> QueryRewriter:
    """Factory for a configured query rewriter instance.

    Reuses the bot's shared ``AsyncOpenAI`` client (connection reuse, no
    per-call construction) and reads the live settings so env overrides apply.

    *model_slug* should be the SAME model as the main completion call — on a
    single-model local backend any other slug would either 404 or evict the
    loaded weights.  Empty = fall back to ``DEFAULT_MODEL``.
    """
    from config.settings import (
        DEFAULT_MODEL,
        RAG_QUERY_MAX_EXPANSIONS,
        RAG_QUERY_REWRITER,
    )

    from bot_core.ai_client import _make_client

    return QueryRewriter(
        client=_make_client(),
        kb_domain="Humblewood fantasy worldbuilding",
        max_expansions=RAG_QUERY_MAX_EXPANSIONS,
        enabled=RAG_QUERY_REWRITER,
        model_slug=model_slug or (DEFAULT_MODEL or ""),
    )
