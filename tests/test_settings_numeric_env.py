"""P1 #10: malformed RAG_REWRITE_MIN_SCORE must not crash ``config.settings``
at import.

``RAG_REWRITE_MIN_SCORE`` was the one numeric env var read with a bare
``float(os.getenv(...))`` — a garbage value (e.g. ``RAG_REWRITE_MIN_SCORE=abc``)
raised ``ValueError`` the moment ``config.settings`` was imported, taking the
whole bot down. Every other numeric env var already routes through the
``_safe_*`` helpers; this test pins the same behaviour.
"""
from __future__ import annotations

import importlib

import pytest

import config.settings as S


@pytest.fixture(autouse=True)
def _restore_settings():
    """Re-read the module once after each test so the next test starts clean."""
    yield
    importlib.reload(S)


def _reload() -> None:
    importlib.reload(S)


class TestRAGRewriteMinScore:
    def test_garbage_value_falls_back_to_default(self, monkeypatch):
        """The TODO's verify case: ``"abc"`` → import succeeds, value == 0.35."""
        monkeypatch.setenv("RAG_REWRITE_MIN_SCORE", "abc")
        _reload()  # must NOT raise
        assert S.RAG_REWRITE_MIN_SCORE == 0.35

    def test_valid_value_is_respected(self, monkeypatch):
        monkeypatch.setenv("RAG_REWRITE_MIN_SCORE", "0.7")
        _reload()
        assert S.RAG_REWRITE_MIN_SCORE == 0.7

    def test_empty_string_falls_back_to_default(self, monkeypatch):
        """A bare ``RAG_REWRITE_MIN_SCORE=`` line must not be a crash either."""
        monkeypatch.setenv("RAG_REWRITE_MIN_SCORE", "")
        _reload()
        assert S.RAG_REWRITE_MIN_SCORE == 0.35

    def test_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv("RAG_REWRITE_MIN_SCORE", raising=False)
        _reload()
        assert S.RAG_REWRITE_MIN_SCORE == 0.35
