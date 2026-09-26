"""Lightweight in-memory BM25 lexical scorer for hybrid retrieval.

Provides a cheap exact-term / keyword signal to complement dense vector search. This is
the "lexical leg" of the hybrid pipeline: dense embeddings (bge-m3) capture semantic
similarity, while BM25 captures exact term overlap — which matters a lot for a worldbuilding
KB full of proper names, stat values, and spell/ability terms that paraphrase poorly.

The index is built once per set of chunks and cached by the caller (see
``kb.retrievers._get_bm25``), so per-query cost is just scoring the query against the
precomputed term frequencies.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Lowercase alphanumerics + apostrophes (keeps "don't" / names with ' intact).
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase *text* into a list of BM25 tokens."""
    return _TOKEN_RE.findall((text or "").lower())


class BM25:
    """Okapi BM25 over a fixed corpus of documents (chunk contents).

    Parameters mirror the classic defaults; ``b=0.75`` discounts long documents so a big
    chunk isn't rewarded just for containing a query term somewhere in it.
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_tf = [Counter(toks) for toks in self.doc_tokens]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

        df: dict[str, int] = {}
        for tf in self.doc_tf:
            for term in tf:
                df[term] = df.get(term, 0) + 1
        # +1 keeps IDF non-negative (a term in every doc still scores slightly).
        self.idf = {t: math.log((self.n - df[t] + 0.5) / (df[t] + 0.5)) + 1.0 for t in df}

    def scores(self, query: str) -> list[float]:
        """Return a BM25 score per document (same order as the corpus)."""
        out = [0.0] * self.n
        qterms = tokenize(query)
        if not qterms or self.avgdl <= 0:
            return out
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for i, tf in enumerate(self.doc_tf):
            dl = self.doc_len[i]
            if dl == 0:
                continue
            s = 0.0
            for term in qterms:
                f = tf.get(term)
                if not f:
                    continue
                idf = self.idf.get(term)
                if idf is None:
                    continue
                denom = f + k1 * (1 - b + b * dl / avgdl)
                s += idf * (f * (k1 + 1)) / denom
            out[i] = s
        return out
