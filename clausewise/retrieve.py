"""Lexical retrievers implemented from scratch: BM25, TF-IDF, and their fusion."""
import math
import re
from collections import Counter

import numpy as np

STOP = set("""a an the of to in on for and or by with as at be is are was were this that these those
any all such shall will may its it from under into upon other than not no""".split())
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return [w for w in _WORD.findall(text.lower()) if w not in STOP and len(w) > 1]


def corpus_idf(token_lists):
    """Inverse document frequency over a whole collection of chunks."""
    df = Counter()
    for toks in token_lists:
        df.update(set(toks))
    n = len(token_lists)
    return {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}, n


class BM25:
    name = "BM25"

    def __init__(self, chunk_tokens, idf, k1=1.5, b=0.75):
        self.tf = [Counter(t) for t in chunk_tokens]
        self.len = np.array([len(t) for t in chunk_tokens], dtype=float)
        self.avg = self.len.mean() if len(self.len) else 0.0
        self.idf, self.k1, self.b = idf, k1, b

    def scores(self, query_tokens):
        out = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.len / (self.avg or 1))
        for w in set(query_tokens):
            idf = self.idf.get(w)
            if idf is None:
                continue
            f = np.array([tf.get(w, 0) for tf in self.tf], dtype=float)
            out += idf * f * (self.k1 + 1) / (f + norm)
        return out


class TfIdf:
    name = "TF-IDF"

    def __init__(self, chunk_tokens, idf):
        self.idf = idf
        self.vecs = [self._vec(t) for t in chunk_tokens]

    def _vec(self, toks):
        c = Counter(toks)
        v = {w: (1 + math.log(n)) * self.idf.get(w, 0.0) for w, n in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def scores(self, query_tokens):
        q = self._vec(query_tokens)
        return np.array([sum(q[w] * v.get(w, 0.0) for w in q) for v in self.vecs])


def rrf(*score_arrays, k=60):
    """Reciprocal rank fusion: combine rankings without having to calibrate scores."""
    fused = np.zeros(len(score_arrays[0]))
    for s in score_arrays:
        ranks = np.empty(len(s), dtype=int)
        ranks[np.argsort(-s, kind="stable")] = np.arange(len(s))
        fused += 1.0 / (k + ranks + 1)
    return fused
