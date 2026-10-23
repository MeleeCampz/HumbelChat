"""Tests for the German-query RAG fixes (see docs/internal/RAG_ANALYSIS_2026-09-27.md).

Covers:
- kb.chunker.html_tables_to_plain_text — HTML table → plain text conversion
- kb.retrievers._looks_german — non-English query detection that forces the
  LLM rewrite/expansion path for German game queries.
"""
from __future__ import annotations

import pytest

from kb.chunker import Chunker, html_tables_to_plain_text
from kb.retrievers import _looks_german


# ───────────────────────── HTML table → plain text ─────────────────────────

ARMOR_TABLE = """**Armor**

<table>
  <thead>
    <tr>
      <th>Armor</th>
      <th>Armor Class (AC)</th>
      <th>Strength</th>
      <th>Stealth</th>
      <th>Weight</th>
      <th>Cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th colspan="6"><em>Light Armor (1 Minute to Don or Doff)</em></th>
    </tr>
    <tr>
      <td>Padded Armor</td>
      <td>11 + Dex modifier</td>
      <td>\u2014</td>
      <td>Disadvantage</td>
      <td>4 lb.</td>
      <td>5 GP</td>
    </tr>
    <tr>
      <td>Chain Mail</td>
      <td>16</td>
      <td>\u2014</td>
      <td>Disadvantage</td>
      <td>55 lb.</td>
      <td>75 GP</td>
    </tr>
  </tbody>
</table>
"""


class TestHtmlTablesToPlainText:
    def test_no_tables_is_noop(self):
        text = "# Heading\n\nSome plain markdown.\n"
        assert html_tables_to_plain_text(text) == text

    def test_header_row_becomes_column_legend(self):
        out = html_tables_to_plain_text(ARMOR_TABLE)
        assert "Columns: Armor, Armor Class (AC), Strength, Stealth, Weight, Cost" in out
        assert "<th>" not in out and "</tr>" not in out

    def test_rows_keep_item_names_as_clean_tokens(self):
        out = html_tables_to_plain_text(ARMOR_TABLE)
        assert "Chain Mail | 16 | \u2014 | Disadvantage | 55 lb. | 75 GP" in out

    def test_colspan_category_row_kept_as_bracketed_label(self):
        out = html_tables_to_plain_text(ARMOR_TABLE)
        assert "[Light Armor (1 Minute to Don or Doff)]" in out

    def test_stat_block_table_with_empty_headers_and_bold_cells(self):
        sb = (
            "<table>\n"
            "<thead><tr><th></th><th></th><th>MOD</th><th>SAVE</th></tr></thead>\n"
            "<tbody>\n"
            "<tr><td><strong>STR</strong></td><td>21</td><td>+5</td><td>+5</td></tr>\n"
            "</tbody>\n"
            "</table>"
        )
        out = html_tables_to_plain_text(sb)
        assert "STR | 21 | +5 | +5" in out
        # empty header cells are dropped, remaining labels still form a legend
        assert "Columns: MOD, SAVE" in out

    def test_chunker_converts_tables_before_splitting(self, tmp_path):
        doc = tmp_path / "armor.md"
        body = "# Equipment\n\n" + ARMOR_TABLE + "\n\n## Next Section\n\n" + "x" * 200 + "\n"
        doc.write_text(body, encoding="utf-8", newline="\n")
        chunks = Chunker.split_file_sync(doc)
        contents = "\n".join(c.content for c in chunks)
        assert "<table>" not in contents
        assert "Chain Mail | 16" in contents


# ───────────────────────── German query detection ─────────────────────────

class TestLooksGerman:
    @pytest.mark.parametrize(
        "query",
        [
            "Gib mir die Stat block vons verschiedneen Rüstungen",
            "Was ist die Rüstungsklasse einer Kettenrüstung?",
            "Wie viel kostet eine Plattenrüstung?",
            "Zeig mir die Magier Tabelle",          # no umlaut, stopwords: mir/die
            "Was kann ein Krieger in Stufe 5?",     # was/kann/stufe
            "Erkläre die Regeln für Heimlichkeit",
        ],
    )
    def test_german_queries_detected(self, query):
        assert _looks_german(query)

    @pytest.mark.parametrize(
        "query",
        [
            "armor stat block",
            "What is the armor class of chain mail?",
            "show me the wizzard character table",
            "goblin statblock",
            "healing potion price",
        ],
    )
    def test_english_queries_not_flagged(self, query):
        assert not _looks_german(query)


class TestRewriteAllQueriesFlag:
    """RAG_REWRITE_ALL_QUERIES gates the rewrite-all-queries behaviour."""

    @pytest.fixture(autouse=True)
    def _restore_settings(self):
        yield
        import config.settings as settings
        import importlib

        importlib.reload(settings)  # recompute from (restored) env after each test

    def test_defaults_to_true(self, monkeypatch):
        import config.settings as settings
        import importlib

        monkeypatch.delenv("RAG_REWRITE_ALL_QUERIES", raising=False)
        importlib.reload(settings)
        assert settings.RAG_REWRITE_ALL_QUERIES is True

    def test_env_zero_disables(self, monkeypatch):
        import config.settings as settings
        import importlib

        monkeypatch.setenv("RAG_REWRITE_ALL_QUERIES", "0")
        importlib.reload(settings)
        assert settings.RAG_REWRITE_ALL_QUERIES is False


# ─────────────── Diacritic normalization (BM25 + keyword fallback) ───────────────
# The old tokenizers used [a-z0-9']+ / [a-zA-Z_]{3,} which SPLIT on umlauts:
# "Kettenrüstung" → ["kettenr", "stung"] — German terms with umlauts could never
# exact-match (e.g. against the German session notes). Both tokenizers now strip
# diacritics first (kb.lexical.strip_diacritics), so both sides normalize to
# "kettenrustung" and match.

class TestDiacriticTokenization:
    def test_tokenize_strips_umlauts(self):
        from kb.lexical import tokenize

        assert tokenize("Kettenrüstung") == ["kettenrustung"]
        assert tokenize("für") == ["fur"]
        assert tokenize("Heiltrank für Stufe 3") == ["heiltrank", "fur", "stufe", "3"]

    def test_tokenize_untouched_ascii_and_apostrophes(self):
        from kb.lexical import tokenize

        assert tokenize("don't stop") == ["don't", "stop"]
        assert tokenize("Chain Mail AC 16") == ["chain", "mail", "ac", "16"]

    def test_bm25_matches_umlaut_bearing_terms(self):
        from kb.lexical import BM25

        docs = [
            "Die Kettenrüstung hat Rüstungsklasse 14 und wiegt 55 Pfund.",
            "A plain english row about padded armor.",
        ]
        scores = BM25(docs).scores("Kettenrüstung")
        assert scores[0] > 0, "umlaut term must score the German doc"
        assert scores[0] > scores[1]

    def test_reader_normalize_query_normalizes_umlauts(self):
        from kb.reader import _normalize_query

        terms = _normalize_query("Was ist die Kettenrüstung wert?")
        assert "kettenrustung" in terms
        # No term may contain a non-ASCII (un-normalized) character.
        assert all(t.isascii() for t in terms)


# ─────────────────── Parallel rewrite (latency optimization) ───────────────────
# With RAG_REWRITE_ALL_QUERIES the LLM expansion depends on nothing from the dense
# ranking, so _retrieve_vector starts it as a task BEFORE embedding/ranking the
# query. This test proves the overlap (expand() is entered while
# query_with_embeddings is still in flight) and that the expansion-derived chunks
# still reach the final selection.

class TestParallelRewrite:
    async def test_rewrite_overlaps_dense_ranking(self, monkeypatch):
        import asyncio

        import kb.retrievers as R

        monkeypatch.setattr("config.settings.RAG_REWRITE_ALL_QUERIES", True)
        monkeypatch.setattr("config.settings.RAG_QUERY_REWRITER", True)
        monkeypatch.setattr("config.settings.RAG_HYBRID_ENABLED", False)  # isolate
        monkeypatch.setattr("config.settings.RERANK_ENABLED", False)
        monkeypatch.setattr("config.settings.RAG_MIN_ATTACH_SCORE", 0.0)
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR", 0.0)

        events: dict[str, bool] = {"expand_called": False}

        class FakeIdx:
            def is_empty(self):
                return False

            async def query_with_embeddings(self, text, top_n=5):
                await asyncio.sleep(0.2)  # simulate the embed+rank window
                events["expand_seen_during_rank"] = events["expand_called"]
                return [("a.md [Sec]", "alpha content", 0.6)], [0.1]

            def embedding_map(self):
                return {}  # no embeddings → near-duplicate dedup is a no-op

            async def rank_texts(self, texts, top_n=5):
                assert texts == ["armor table english"]
                return [[(("b.md [Exp]", "beta content", 0.7))]]

        class FakeStore:
            def get_index(self):
                return FakeIdx()

        monkeypatch.setattr(R, "_index_store", FakeStore())

        class FakeRewriter:
            async def expand(self, q):
                events["expand_called"] = True
                await asyncio.sleep(0.1)  # simulate the LLM round-trip
                return [q, "armor table english"]

        monkeypatch.setattr(
            "kb.query_rewriter.create_query_rewriter",
            lambda model_slug="": FakeRewriter(),
        )

        docs = await R._retrieve_vector("Kettenrüstung?", "/nonexistent-kb", top_n=4)
        names = [n for n, _ in docs]

        assert events["expand_called"], "rewriter must be invoked (all-queries mode)"
        assert events["expand_seen_during_rank"], (
            "rewrite LLM call must START before the dense ranking completes"
        )
        assert any("b.md" in n for n in names), (
            "expansion-derived chunk must survive RRF merge into the selection"
        )

    async def test_no_parallel_task_when_all_queries_off(self, monkeypatch):
        """Kill-switch path: with RAG_REWRITE_ALL_QUERIES=0 and a confident
        non-German query, no rewrite happens at all (old serial behaviour)."""
        import asyncio

        import kb.retrievers as R

        monkeypatch.setattr("config.settings.RAG_REWRITE_ALL_QUERIES", False)
        monkeypatch.setattr("config.settings.RAG_QUERY_REWRITER", True)
        monkeypatch.setattr("config.settings.RAG_HYBRID_ENABLED", False)
        monkeypatch.setattr("config.settings.RERANK_ENABLED", False)
        monkeypatch.setattr("config.settings.RAG_MIN_ATTACH_SCORE", 0.0)
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR", 0.0)

        calls: list[str] = []

        class FakeIdx:
            def is_empty(self):
                return False

            async def query_with_embeddings(self, text, top_n=5):
                await asyncio.sleep(0.01)
                return [("a.md [Sec]", "alpha content", 0.9)], [0.1]  # confident

            def embedding_map(self):
                return {}  # no embeddings → near-duplicate dedup is a no-op

            async def rank_texts(self, texts, top_n=5):
                calls.append("rank_texts")
                return []

        class FakeStore:
            def get_index(self):
                return FakeIdx()

        monkeypatch.setattr(R, "_index_store", FakeStore())

        class FakeRewriter:
            async def expand(self, q):
                calls.append("expand")
                return [q]

        monkeypatch.setattr(
            "kb.query_rewriter.create_query_rewriter",
            lambda model_slug="": FakeRewriter(),
        )

        docs = await R._retrieve_vector("fireball spell", "/nonexistent-kb", top_n=4)
        assert calls == [], f"confident English query must not trigger a rewrite, got {calls}"
        # select_ranked_chunks collapses to per-file stems (no section suffix).
        assert [n for n, _ in docs] == ["a.md"]


# ─────────────── Rerank against the LLM expansion (cross-lingual rescue) ───────────────
# bge-reranker-v2-m3 scores a German question vs English KB chunks near-noise
# (TQ case Oct 15: right chunk rank 25 @0.003 with the German original, but
# rank 1 @0.963 against the rewriter's English expansion). _retrieve_vector
# therefore scores candidates against BOTH and keeps the max per chunk.

class TestRerankExpansionMaxFusion:
    def _patch_pipeline(self, monkeypatch):
        import kb.retrievers as R

        monkeypatch.setattr("config.settings.RAG_REWRITE_ALL_QUERIES", True)
        monkeypatch.setattr("config.settings.RAG_QUERY_REWRITER", True)
        monkeypatch.setattr("config.settings.RAG_HYBRID_ENABLED", False)  # isolate
        monkeypatch.setattr("config.settings.RERANK_TOP_K", 10)
        monkeypatch.setattr("config.settings.RAG_MIN_ATTACH_SCORE", 0.0)
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR", 0.0)

        class FakeIdx:
            def is_empty(self):
                return False

            async def query_with_embeddings(self, text, top_n=5):
                # Dense (and thus the pre-rerank order) favours the DECOY —
                # exactly like production, where stealth-item near-duplicates
                # outscored Trickster's Quiver on the raw German question.
                return [
                    (("decoy.md [Hood]", "decoy content", 0.62)),
                    ("tq.md [Quiver]", "tq content", 0.57),
                ], [0.1]

            def embedding_map(self):
                return {}  # no embeddings → near-duplicate dedup is a no-op

            async def rank_texts(self, texts, top_n=5):
                return []  # expansions add no new chunks — only the rerank query

        class FakeStore:
            def get_index(self):
                return FakeIdx()

        monkeypatch.setattr(R, "_index_store", FakeStore())

        class FakeRewriter:
            async def expand(self, q):
                return [q, "quiver hidden attack no reveal"]

        monkeypatch.setattr(
            "kb.query_rewriter.create_query_rewriter",
            lambda model_slug="": FakeRewriter(),
        )
        return R

    def _fake_cross_encoder(self, calls: list[str]):
        """Mimics the measured production behaviour per query language."""

        async def fake_rerank(q, chunks, *, top_n=None):
            calls.append(q)
            if q == "quiver hidden attack no reveal":  # English expansion
                table = {"tq content": 0.963, "decoy content": 0.126}
            else:  # German original — near-noise scores, decoy on top
                table = {"decoy content": 0.018, "tq content": 0.003}
            return sorted(((c, table[c]) for c in chunks), key=lambda t: -t[1])

        return fake_rerank

    async def test_max_fusion_rescues_cross_lingual_chunk(self, monkeypatch):
        import kb.reranker as KR

        R = self._patch_pipeline(monkeypatch)
        calls: list[str] = []
        monkeypatch.setattr(KR, "is_available", lambda: True)
        monkeypatch.setattr(KR, "rerank", self._fake_cross_encoder(calls))

        docs = await R._retrieve_vector(
            "Gibt es ein Item dass mich nicht aufdeckt?", "/nonexistent-kb", top_n=4
        )
        names = [n for n, _ in docs]
        assert calls == [
            "Gibt es ein Item dass mich nicht aufdeckt?",
            "quiver hidden attack no reveal",
        ], f"rerank must run against original AND first expansion, got {calls}"
        # max(0.003, 0.963)=0.963 > max(0.018, 0.126)=0.126 → TQ file first.
        assert names == ["tq.md", "decoy.md"], f"max-fusion must rescue TQ, got {names}"

    async def test_disabled_flag_uses_original_only(self, monkeypatch):
        import kb.reranker as KR

        R = self._patch_pipeline(monkeypatch)
        monkeypatch.setattr("config.settings.RERANK_USE_EXPANSION", False)
        calls: list[str] = []
        monkeypatch.setattr(KR, "is_available", lambda: True)
        monkeypatch.setattr(KR, "rerank", self._fake_cross_encoder(calls))

        docs = await R._retrieve_vector(
            "Gibt es ein Item dass mich nicht aufdeckt?", "/nonexistent-kb", top_n=4
        )
        names = [n for n, _ in docs]
        assert calls == ["Gibt es ein Item dass mich nicht aufdeckt?"], (
            "RERANK_USE_EXPANSION=0 must rerank against the original only"
        )
        # German-only scores: decoy 0.018 > TQ 0.003 — reproduces the failure.
        assert names == ["decoy.md", "tq.md"]

    async def test_no_expansion_means_single_pass(self, monkeypatch):
        import kb.reranker as KR

        R = self._patch_pipeline(monkeypatch)
        # Confident English query with all-queries rewrite OFF → no expansions.
        monkeypatch.setattr("config.settings.RAG_REWRITE_ALL_QUERIES", False)
        calls: list[str] = []
        monkeypatch.setattr(KR, "is_available", lambda: True)
        monkeypatch.setattr(KR, "rerank", self._fake_cross_encoder(calls))

        docs = await R._retrieve_vector("fireball spell", "/nonexistent-kb", top_n=4)
        assert calls == ["fireball spell"], f"single pass expected, got {calls}"
        assert len(docs) > 0

    async def test_both_passes_failing_keeps_candidate_order(self, monkeypatch):
        import kb.reranker as KR

        R = self._patch_pipeline(monkeypatch)

        async def failing_rerank(q, chunks, *, top_n=None):
            return [(c, 0.0) for c in chunks]  # the contract's failure fallback

        monkeypatch.setattr(KR, "is_available", lambda: True)
        monkeypatch.setattr(KR, "rerank", failing_rerank)

        docs = await R._retrieve_vector(
            "Gibt es ein Item dass mich nicht aufdeckt?", "/nonexistent-kb", top_n=4
        )
        # All scores 0.0 → stable sort keeps the pre-rerank (dense) order.
        assert [n for n, _ in docs] == ["decoy.md", "tq.md"]
