"""Backlog #75 (builds on pipeline review #74 F1/F2): session-note duplication.

Session notes.md files carry three layers: live ``## Notes`` bullets, the
end-of-session ``## Overview``, and — when the session ended with a successful
merge — the canonical ``## Session Log (combined from all uploads —
AI-merged)``, which *replaces* the raw content by design. Indexed as-is, the
same events reach RAG twice (raw + merged) while F1's near-dupe dedup can't
catch them (the merged text differs). Three rules are pinned here:

1. Chunker: merged session files get their superseded ``## Notes`` region
   stripped before splitting; unmerged/active sessions keep their notes.
2. Store load: stale Notes rows from before the change are purged from the
   cache (embeddings reused, zero re-embedding; rows pruned via the upsert).
3. Raw drafts (``.attachments/`` etc.) never enter the index (dot-prefix
   hidden-path rule).

Plus the F1 fix itself: ``embedding_map()`` is keyed by the *retrieval* name,
so selection-time dedup finally matches nested production files.
"""
from __future__ import annotations

import pathlib

import pytest

import kb.chunker as C
import kb.index as I
import config.settings as S
from kb.vector_db import KBVectorIndex
from kb.retrievers import select_ranked_chunks


_FILLER = " ".join(
    f"Barlin anecdote {i}: the Alderheart chronicle records a tavern scene."
    for i in range(130)
)  # ~10.7k chars — keeps the file above the 7 500 full-doc threshold so the
   # header-split path runs (a tiny notes.md would stay one full-doc chunk,
   # which would make the section assertions below vacuous).

MERGED_SESSION = f"""# Session: To the roots

- Started: 28.10.2026 20:00
- Ended: 28.10.2026 23:25

## Notes

- (20:11) Rohit said hello to Barlin.
- (20:30) Purchased at the shop: 2x Potion of Healing, 1x Antitoxin,
  1x Rider's Knee Boots and 1x Saddle Bag for Rolo — live scratch notes
  typed during the session, kept verbatim.

## Overview (written when the session ended)

The party met Barlin, bought gear and left for the roots.

{_FILLER}

## Session Log (combined from all uploads — AI-merged)

# To the roots

## Wiedersehen in Alderheart

Rohit greeted Barlin warmly; the party then visited the shop and bought
two healing potions, an antitoxin, Rider's Knee Boots for Rolo and a
saddle bag — the merged canonical log, cleaned across all uploads.

{_FILLER}
"""

# Same file but ended WITHOUT a successful merge: mechanical document sections
# instead of the canonical header → the live notes are the primary content.
UNMERGED_SESSION = MERGED_SESSION.replace(
    "## Session Log (combined from all uploads — AI-merged)",
    "## Uploaded documents",
)


class TestChunkerStrip:
    def test_merged_file_drops_the_superseded_notes_region(self, tmp_path):
        d = tmp_path / "SessionLogs" / "s1"
        d.mkdir(parents=True)
        (d / "notes.md").write_text(MERGED_SESSION, encoding="utf-8")
        chunks = C.Chunker.split_file_sync(d / "notes.md")
        joined = "\n".join(c.content for c in chunks)
        assert "raw scratch" not in joined and "live scratch notes" not in joined
        assert "Rohit said hello" not in joined          # Notes bullets gone
        assert "The party met Barlin" in joined          # Overview kept
        assert "merged canonical log" in joined          # merged log kept
        # No chunk may claim a bare `Notes` section any more.
        assert not any(seg.strip() == "Notes"
                       for c in chunks for seg in c.section_path.split(" -> "))

    def test_unmerged_and_non_session_files_keep_their_notes(self, tmp_path):
        d = tmp_path / "SessionLogs" / "s2"
        d.mkdir(parents=True)
        (d / "notes.md").write_text(UNMERGED_SESSION, encoding="utf-8")
        joined = "\n".join(c.content for c in C.Chunker.split_file_sync(d / "notes.md"))
        assert "Rohit said hello" in joined              # active/unmerged: notes stay
        chunks = C.Chunker.split_file_sync(d / "notes.md")
        assert any(seg.strip() == "Notes"
                   for c in chunks for seg in c.section_path.split(" -> "))
        book = tmp_path / "adventure.md"
        book.write_text("## Notes\n\nA section literally named Notes in a "
                        "regular book file stays indexed as usual.\n",
                        encoding="utf-8")
        joined2 = "\n".join(c.content for c in C.Chunker.split_file_sync(book))
        assert "literally named Notes" in joined2

    def test_strip_helper_is_anchor_exact(self):
        out = C.strip_merged_session_notes(MERGED_SESSION)
        assert "## Notes" not in out
        assert "## Overview" in out and "## Session Log (combined" in out
        assert C.strip_merged_session_notes(UNMERGED_SESSION) == UNMERGED_SESSION

    def test_notes_inside_the_merged_log_body_survive(self):
        """A session with NO live notes whose merged log body happens to contain
        a `## Notes` subsection must keep it — only the live region BEFORE the
        canonical header is superseded."""
        body_only = (
            "# Session: No live notes\n\n- Started: x\n\n"
            "## Session Log (combined from all uploads — AI-merged)\n\n"
            "# Chronicle\n\n## Notes\n\nA section that the MERGE itself wrote — "
            "it is part of the canonical chronicle and stays indexed.\n"
        )
        assert C.strip_merged_session_notes(body_only) == body_only


class TestStorePurge:
    """Pre-change cache rows are purged on load WITHOUT re-embedding."""

    class FakeEmbedder:
        def __init__(self):
            self.calls = 0

        async def encode(self, texts):
            self.calls += 1
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def _make_store(self, kb: pathlib.Path):
        store = I.KBIndexStore(kb)
        store._embedder = self.FakeEmbedder()
        return store

    @pytest.mark.asyncio
    async def test_stale_notes_rows_purged_from_memory_and_db(self, tmp_path, monkeypatch):
        kb = tmp_path / "kb75"
        sess = kb / "SessionLogs" / "2026-10-28_01"
        sess.mkdir(parents=True)
        (sess / "notes.md").write_text(MERGED_SESSION, encoding="utf-8")
        (kb / "rules.md").write_text("# Rules\n\n" + "Armor class basics. " * 30,
                                     encoding="utf-8")

        # 1. Build the cache with the strip DISABLED → simulates the pre-fix
        #    state where the Notes region produced chunks.
        monkeypatch.setattr(C, "strip_merged_session_notes", lambda t: t)
        store = self._make_store(kb)
        idx1 = await store.load()
        names1 = [display for display, _ in idx1.iter_chunks()]
        notes_chunks = [n for n in names1 if I._is_raw_session_notes_chunk(n)]
        assert notes_chunks, "precondition: Notes chunks present in the old cache"

        # 2. Reload with the REAL chunker: file bytes unchanged → hash-valid
        #    rows reused (0 re-embedding) and the Notes rows are purged.
        monkeypatch.undo()
        store2 = self._make_store(kb)
        idx2 = await store2.load()
        assert store2._embedder.calls == 0, "purge must not re-embed anything"
        names2 = [display for display, _ in idx2.iter_chunks()]
        assert not [n for n in names2 if I._is_raw_session_notes_chunk(n)]
        assert len(names2) == len(names1) - len(notes_chunks)
        assert any("Overview" in n for n in names2), "curated sections survive"

        # 3. The DB rows themselves are pruned via the changed-upsert path.
        import sqlite3
        conn = sqlite3.connect(str(store2.db_path))
        rows = [r[0] for r in conn.execute(
            "SELECT doc_name FROM document_index")
            if I._is_raw_session_notes_chunk(r[0])]
        conn.close()
        assert rows == [], "purged rows must be gone from the SQLite cache too"

    @pytest.mark.asyncio
    async def test_unmerged_session_keeps_its_live_notes(self, tmp_path):
        kb = tmp_path / "kb75b"
        sess = kb / "SessionLogs" / "active"
        sess.mkdir(parents=True)
        (sess / "notes.md").write_text(UNMERGED_SESSION, encoding="utf-8")
        store = self._make_store(kb)
        idx = await store.load()
        assert any("Rohit said hello" in text for _display, text in idx.iter_chunks())

    @pytest.mark.asyncio
    async def test_dot_prefixed_draft_dirs_are_never_indexed(self, tmp_path):
        kb = tmp_path / "kb75c"
        sess = kb / "SessionLogs" / "s3"
        (sess / ".attachments").mkdir(parents=True)
        (sess / ".attachments" / "draft.md").write_text(
            "Raw draft with a totally unique marker ZZQXK.\n", encoding="utf-8")
        (sess / "notes.md").write_text(MERGED_SESSION, encoding="utf-8")
        store = self._make_store(kb)
        idx = await store.load()
        joined = "\n".join(text for _display, text in idx.iter_chunks())
        assert "ZZQXK" not in joined, ".attachments drafts must stay out of the index"
        assert "merged canonical log" in joined


class TestEmbeddingMapRetrievalKeys:
    """#74 F1: map keyed by retrieval_name → dedup works end-to-end on nested files."""

    def test_keys_are_retrieval_names(self):
        idx = KBVectorIndex.from_entries(
            [("rules.md [Armor]", "plate armor AC 16", "DnDRules/rules.md")],
            [[1.0, 0.0]],
        )
        assert list(idx.embedding_map()) == ["DnDRules/rules.md [Armor]"]

    def test_nested_near_dupe_is_deduped_end_to_end(self, monkeypatch):
        """The exact production wiring (idx.embedding_map() + retrieval_name
        ranked tuples) must drop the duplicate — the vacuous hand-built-dict
        tests could not catch F1."""
        monkeypatch.setattr(S, "RAG_DEDUP_ENABLED", True)
        monkeypatch.setattr(S, "RAG_DEDUP_COSINE_TAU", 0.92)
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.999, 0.0436, 0.0, 0.0]          # cos ≈ 0.999 > tau
        idx = KBVectorIndex.from_entries(
            [("rules.md [Armor]", "plate armor AC 16", "DnDRules/rules.md"),
             ("rules2.md [Armor copy]", "plate armour copy AC 16", "DnDRules/rules2.md")],
            [v1, v2],
        )
        ranked = [(d.retrieval_name(), d.passage_text(), 0.9) for d in idx._docs]
        docs = select_ranked_chunks(ranked, top_n=4,
                                    embeddings_by_name=idx.embedding_map())
        joined = "\n".join(text for _stem, text in docs)
        assert joined.count("plate armor AC 16") == 1
        assert "Near-duplicate entry omitted from attachment" in joined
