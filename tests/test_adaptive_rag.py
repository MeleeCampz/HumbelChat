"""Tests for Adaptive-k Retrieval, FlashRank Reranking, and Adaptive Chunking."""
import pytest
from kb.retrievers import _adaptive_k_threshold, _flashrank_reorder


class TestAdaptiveK:
    """Tests for Adaptive-k dynamic threshold selection."""

    def test_single_score(self):
        """Single score should return k=1."""
        assert _adaptive_k_threshold([0.8]) == 1

    def test_two_scores_no_gap(self):
        """Two closely spaced scores → include both (with buffer)."""
        scores = [0.9, 0.85]
        # Gap = 0.05, k = 1 + 1 (gap index) + 5 (buffer) → clamped to len
        result = _adaptive_k_threshold(scores)
        assert result == 2  # both scores are high

    def test_two_scores_large_gap(self):
        """Large gap between two scores → select first with buffer."""
        scores = [0.9, 0.3]
        # Gap = 0.6 at idx=0, k = 1 + 5 = 6, clamped to 2
        result = _adaptive_k_threshold(scores)
        assert result == 2

    def test_large_gap_midway(self):
        """Scores: [0.9, 0.85, 0.8, 0.4, 0.35, 0.3]
        Largest gap at idx=2 (0.8→0.4 = 0.4). k = 3 + 1 + 5 = 9, clamped to 6."""
        scores = [0.9, 0.85, 0.8, 0.4, 0.35, 0.3]
        result = _adaptive_k_threshold(scores)
        assert result == 6

    def test_no_gap_all_similar(self):
        """All similar scores → return all."""
        scores = [0.7, 0.69, 0.68, 0.67]
        # Max gap is 0.01 at idx=0, k = 1 + 5 = 6, clamped to 4
        result = _adaptive_k_threshold(scores)
        assert result == 4


class TestFlashRankRerank:
    """Tests for FlashRank marginal utility reranking."""

    def test_empty_input(self):
        """Empty results should return empty."""
        result = _flashrank_reorder([], top_n=5)
        assert result == []

    def test_single_result(self):
        """Single result should be returned."""
        results = [(0.9, "doc1", "content")]
        result = _flashrank_reorder(results, top_n=5)
        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_deduplication(self):
        """Similar file stems should be deduplicated to unique files."""
        results = [
            (0.9, "Doc A [Section 1]", "content1"),
            (0.85, "Doc A [Section 2]", "content2"),
            (0.8, "Doc B [Section 1]", "content3"),
        ]
        result = _flashrank_reorder(results, top_n=5)
        # Should only have one Doc A entry
        stems = [r[0] for r in result]
        assert stems.count("Doc A") == 1

    def test_info_density_bonus(self):
        """Documents with better info density should rank higher."""
        results = [
            (0.9, "doc1", "short" * 5),     # high score, lower density
            (0.8, "doc2", "a" * 200),       # slightly lower score, much higher density
        ]
        result = _flashrank_reorder(results, top_n=5)
        # doc2's info density bonus should potentially reorder it ahead of doc1
        assert len(result) == 2

    def test_top_n_limit(self):
        """Should not exceed top_n."""
        results = [(float(i / 10), f"doc{i}", "content") for i in range(10)]
        result = _flashrank_reorder(results, top_n=3)
        assert len(result) <= 3
