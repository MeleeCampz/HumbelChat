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
    from openai import AsyncOpenAI  # type: ignore[import-untyped]

logger = logging.getLogger("kb.query_rewriter")


class QueryRewriter:
    """Dynamically expand queries using the configured LLM backend."""

    MAX_EXPANSIONS = 3
    REWRITE_TIMEOUT_SEC = 10
    REWRITE_PROMPT_TEMPLATE = """\
You are a search query expansion assistant. Your task is to generate additional search terms that capture related concepts, synonyms, and contextual meanings of the user's original query — all within the domain of {kb_domain}.

Rules:
- Generate exactly {max_expansions} alternative queries (one per line)
- Each expansion should be a short phrase (3-8 words) suitable for keyword/semantic search
- Focus on synonyms, related concepts, and contextual meanings — not rephrasing the entire question
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

        try:
            response = await self._client.chat.completions.create(
                model=self.model_slug,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low for consistent, focused expansions
                max_tokens=256,
                timeout=self.REWRITE_TIMEOUT_SEC,
            )

            content = response.choices[0].message.content or ""
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            return lines[: self.max_expansions]  # type: ignore[arg-type]

        except Exception as exc:
            logger.warning("Query rewrite attempt failed (%s); returning no expansions", exc)
            return []


# ── Module-level convenience ───────────────────────────────────────────

def create_query_rewriter() -> QueryRewriter:
    """Factory for a configured query rewriter instance.

    Reuses the bot's shared ``AsyncOpenAI`` client (connection reuse, no
    per-call construction) and reads the live settings so env overrides apply.
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
        model_slug=DEFAULT_MODEL or "",
    )
