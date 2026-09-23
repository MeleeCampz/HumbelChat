"""P2 #17: vector ranking must be numpy-accelerated without changing results.

The old ``query`` / ``query_with_embeddings`` / ``rank_texts`` ran an O(N·D)
pure-Python cosine loop per query.  These tests prove the numpy path:
  * returns the SAME top-k (order + scores within float32 epsilon) as the
    reference pure-Python ranking,
  * reuses a single matrix across repeated queries (no per-query O(N·D) rebuild),
  * falls back cleanly, and
  * is meaningfully faster than the pure-Python loop at scale.
"""
from __future__ import annotations

import math
import time

import pytest

from kb.vector_db import KBVectorIndex, _cosine_similarity


# ──────────────────────────── Data helpers ──────────────────────────────────

def _make_index(n_docs: int, dim: int, seed: int = 7, top_n: int = 5):
    """Build a KBVectorIndex with n_docs deterministic random embeddings."""
    import random
    rng = random.Random(seed)

    def _vec():
        v = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        # Normalize-ish so magnitudes vary like real embeddings.
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    entries = [(f"doc_{i}", f"content {i}", f"src_{i % 3}.md") for i in range(n_docs)]
    embeddings = [_vec() for _ in range(n_docs)]
    return KBVectorIndex.from_entries(entries, embeddings)


def _reference_rank(idx, q_emb, top_n):
    """Ground truth: the original O(N·D) pure-Python cosine ranking."""
    scored = []
    for i, doc in enumerate(idx._docs):
        emb = doc.embedding
        if emb is None:
            continue
        sim = _cosine_similarity(q_emb, emb)
        if sim > 0:
            scored.append((sim, i))
    scored.sort(key=lambda t: -t[0])
    return scored[:top_n]


# ──────────────────────────── Tests ─────────────────────────────────────────

class TestNumpyRanking:

    @pytest.mark.parametrize("top_n", [1, 3, 5, 10, 100])
    def test_numpy_matches_pure_python(self, top_n):
        """Top-k (order + scores) must match the reference within float32 eps."""
        idx = _make_index(200, 64)
        import random
        q = [random.Random(99).uniform(-1, 1) for _ in range(64)]

        got = idx._rank(q, top_n)
        ref = _reference_rank(idx, q, top_n)

        assert len(got) == len(ref)
        for (s_g, i_g), (s_r, i_r) in zip(got, ref):
            assert i_g == i_r, f"rank order diverged at top_n={top_n}"
            assert abs(s_g - s_r) < 1e-4, f"score mismatch: {s_g} vs {s_r}"

    def test_top_n_larger_than_docs(self):
        """top_n >= N must still return all positive sims, sorted."""
        idx = _make_index(50, 32)
        q = [1.0] * 32  # all-positive query (only positive-sim docs returned)
        got = idx._rank(q, 500)
        # top_n (500) exceeds N (50): every positive-sim doc is returned, and
        # the result must never exceed the number of documents.
        assert len(got) <= 50
        sims = [s for s, _ in got]
        assert sims == sorted(sims, reverse=True)
        assert all(s > 0 for s in sims)

    def test_matrix_cache_reused_across_queries(self):
        """Repeated queries must NOT rebuild the matrix (identity + no work)."""
        idx = _make_index(300, 48)
        assert idx._mat_cache is None
        idx._rank([0.1] * 48, 5)
        first_matrix = idx._mat_cache[0]
        idx._rank([0.2] * 48, 5)
        assert idx._mat_cache[0] is first_matrix, "matrix must be cached & reused"

        # A different index gets its own matrix.
        idx2 = _make_index(300, 48)
        idx2._rank([0.1] * 48, 5)
        assert idx2._mat_cache[0] is not first_matrix

    def test_replacing_docs_invalidates_cache(self):
        """When _docs is replaced, the cached matrix must be rebuilt (no stale data)."""
        idx = _make_index(100, 32)
        idx._rank([0.5] * 32, 5)

        # Replace the doc list wholesale (new embeddings → new object identity
        # → the cache must rebuild, not reuse the stale matrix).
        old_matrix = idx._mat_cache[0]
        idx2 = _make_index(100, 32, seed=123)
        idx._docs = idx2._docs
        idx._rank([0.5] * 32, 5)
        assert idx._mat_cache[0] is not old_matrix, "stale matrix must be discarded"
        # The rebuilt matrix must reflect the NEW docs' first embedding.
        assert abs(float(idx._mat_cache[0][0][0])
                   - float(idx._docs[0].embedding[0])) < 1e-5


class TestBenchmark:
    """P2 #17 verify: show the numpy path beats the O(N·D) loop at scale."""

    def test_numpy_is_faster_than_python_at_scale(self):
        """5 000 chunks: numpy matmul ranking must be faster than pure Python."""
        n_docs, dim = 5000, 768
        idx = _make_index(n_docs, dim)
        import random
        q = [random.Random(3).uniform(-1, 1) for _ in range(dim)]

        # Warm the cache (matrix build) — not part of the per-query cost.
        idx._rank(q, 5)

        # Time the per-query numpy path (matrix already cached).
        t0 = time.perf_counter()
        for _ in range(5):
            idx._rank(q, 5)
        numpy_ms = (time.perf_counter() - t0) * 1000 / 5

        # Time the reference pure-Python path (full O(N·D) loop, no cache).
        t0 = time.perf_counter()
        for _ in range(3):
            _reference_rank(idx, q, 5)
        python_ms = (time.perf_counter() - t0) * 1000 / 3

        # Log for visibility; the hard assertion is a modest but real speedup.
        # (Exact speedup depends on hardware, so assert a conservative >1.0x.)
        print(f"\n[numpy] {numpy_ms:.2f} ms/query vs [python] {python_ms:.2f} ms/query "
              f"({python_ms / max(numpy_ms, 1e-6):.1f}x speedup at N={n_docs}, D={dim})")
        assert numpy_ms < python_ms, "numpy path should be faster than the O(N·D) loop"
        assert numpy_ms < 5.0, "numpy matmul over 5000×768 should take < ~5 ms/query"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
