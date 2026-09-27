"""Tests for Phase 3 prompt budget (decision Q3a) — _apply_prompt_budget.

Pure function: trims oldest history first, then lowest-ranked RAG docs, to
keep the estimated prompt (system + history + final user msg w/ RAG) under
PROMPT_BUDGET_CHARS. RAG context is always preserved over history.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from bot_core import ai_client as ac


class TestApplyPromptBudget:
    def test_no_trim_when_under_budget(self, monkeypatch):
        monkeypatch.setattr(ac, "PROMPT_BUDGET_CHARS", 10_000)
        hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        docs = [("a.md", "aaa")]
        hist2, docs2, ctx, names = ac._apply_prompt_budget(
            hist, docs, "=== Knowledge Base ===\naaa", ["a.md"],
            "You are Marvin.", "Master", "question",
        )
        assert hist2 == hist
        assert docs2 == docs
        assert names == ["a.md"]

    def test_trims_oldest_history_first(self, monkeypatch):
        monkeypatch.setattr(ac, "PROMPT_BUDGET_CHARS", 200)
        # 5 history messages; system + question + RAG are the "base".
        hist = [{"role": "user", "content": f"m{i}" * 20} for i in range(5)]
        docs = [("a.md", "aaa")]
        result = ac._apply_prompt_budget(
            hist, docs, "ctx", ["a.md"],
            "sys", "", "q",
        )
        hist2, docs2, _, _ = result
        # history was trimmed (some messages dropped), but RAG docs untouched.
        assert len(hist2) < len(hist)
        assert docs2 == docs
        # the SURVIVING messages are the most recent (kept from the tail).
        assert hist2[-1]["content"] == hist[-1]["content"]

    def test_rag_preserved_over_history(self, monkeypatch):
        """When the prompt is over budget and history is what makes it over,
        history is trimmed (oldest first) while RAG docs are kept in full —
        RAG context is always preserved over history."""
        monkeypatch.setattr(ac, "PROMPT_BUDGET_CHARS", 300)
        docs = [("a.md", "aaa")]  # small RAG — must be fully preserved
        ctx, names = ac._build_rag_context(docs)
        hist = [{"role": "user", "content": "m" * 50} for _ in range(5)]  # big history
        with patch.object(ac.log, "info"):
            hist2, docs2, _, names2 = ac._apply_prompt_budget(
                hist, docs, ctx, names, "sys", "", "q",
            )
        # history was trimmed (oldest-first), RAG untouched
        assert len(hist2) < len(hist)
        assert hist2[-1]["content"] == hist[-1]["content"]  # most recent kept
        assert docs2 == docs
        assert names2 == names

    def test_lowest_ranked_docs_dropped_first(self, monkeypatch):
        """Once history is exhausted, lowest-ranked RAG docs (the tail) are
        dropped first and the highest-ranked doc is kept."""
        monkeypatch.setattr(ac, "PROMPT_BUDGET_CHARS", 300)
        hist = []  # no history to trim
        docs = [(f"f{i}.md", "x" * 100) for i in range(3)]
        ctx, names = ac._build_rag_context(docs)
        with patch.object(ac.log, "info"):
            hist2, docs2, _, names2 = ac._apply_prompt_budget(
                hist, docs, ctx, names, "sys", "", "q",
            )
        # at least one doc dropped (lowest-ranked = tail)
        assert len(docs2) < len(docs)
        assert docs2[0][0] == "f0.md"  # highest-ranked kept
        assert names2 == [d[0] for d in docs2]

    def test_all_docs_dropped_if_base_alone_over(self, monkeypatch):
        """If even the base prompt (no history, no RAG) exceeds the budget,
        every RAG doc is dropped AND a warning notes the base is over."""
        # base = len("sys") + len("q") = 4, so budget 3 makes base over.
        monkeypatch.setattr(ac, "PROMPT_BUDGET_CHARS", 3)
        hist = [{"role": "user", "content": "x" * 50}]
        docs = [("a.md", "y" * 50)]
        with patch.object(ac.log, "info"), patch.object(ac.log, "warning") as mw:
            result = ac._apply_prompt_budget(hist, docs, "ctx", ["a.md"], "sys", "", "q")
        hist2, docs2, _, names2 = result
        assert docs2 == []
        assert names2 == []
        assert len(hist2) <= len(hist)
        assert mw.called  # base-alone-over warning

    def test_empty_inputs_noop(self, monkeypatch):
        monkeypatch.setattr(ac, "PROMPT_BUDGET_CHARS", 100)
        hist2, docs2, ctx, names = ac._apply_prompt_budget(
            [], [], "", [], "sys", "", "q",
        )
        assert hist2 == [] and docs2 == [] and names == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
