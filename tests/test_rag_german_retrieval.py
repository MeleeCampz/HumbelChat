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
