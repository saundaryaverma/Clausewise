"""Measure how often the retriever surfaces the clause a lawyer highlighted, and how well
it can say a clause isn't in the contract at all."""
from dataclasses import dataclass

import numpy as np

from clausewise.chunking import chunk, overlaps
from clausewise.retrieve import BM25, TfIdf, corpus_idf, rrf, tokenize


def query_text(q, mode):
    return q.category if mode == "category" else f"{q.category}. {q.description}"


@dataclass
class Indexed:
    contract: object
    spans: list
    bm25: BM25
    tfidf: TfIdf


def index_all(contracts, size=600, overlap=100):
    spans = [chunk(c.text, size, overlap) for c in contracts]
    tokens = [[tokenize(c.text[s:e]) for s, e in sp] for c, sp in zip(contracts, spans)]
    idf, _ = corpus_idf([t for doc in tokens for t in doc])
    return [Indexed(c, sp, BM25(tok, idf), TfIdf(tok, idf)) for c, sp, tok in zip(contracts, spans, tokens)]


def score(ix, q_tokens, method):
    if method == "bm25":
        return ix.bm25.scores(q_tokens)
    if method == "tfidf":
        return ix.tfidf.scores(q_tokens)
    if method == "hybrid":
        return rrf(ix.bm25.scores(q_tokens), ix.tfidf.scores(q_tokens))
    raise ValueError(f"unknown method {method!r}")


def retrieval_metrics(indexed, method="bm25", mode="category+description", ks=(1, 3, 5)):
    """Only questions whose clause exists are scored here; absent clauses are handled by abstention."""
    first_hits, chars_read = [], []
    for ix in indexed:
        lengths = np.array([e - s for s, e in ix.spans])
        for q in ix.contract.questions:
            if not q.spans:
                continue
            s = score(ix, tokenize(query_text(q, mode)), method)
            order = np.argsort(-s, kind="stable")
            hits = [r for r, i in enumerate(order) if overlaps(ix.spans[i], q.spans)]
            first_hits.append(hits[0] if hits else np.inf)
            chars_read.append(lengths[order[:3]].sum())
    first = np.array(first_hits)
    out = {f"recall@{k}": float((first < k).mean()) for k in ks}
    out["mrr"] = float(np.where(np.isfinite(first), 1.0 / (first + 1), 0.0).mean())
    out["chars_read@3"] = float(np.mean(chars_read))
    out["questions"] = len(first)
    return out


def top_scores(indexed, mode="category+description"):
    """(category, best BM25 score, clause actually present) for every question."""
    rows = []
    for ix in indexed:
        for q in ix.contract.questions:
            s = ix.bm25.scores(tokenize(query_text(q, mode)))
            rows.append((q.category, float(s.max()) if len(s) else 0.0, bool(q.spans)))
    return rows


def fit_thresholds(rows):
    """Per clause type, pick the score cutoff that best separates present from absent.
    'Governing Law' and 'Non-Compete' use very different words, so one global cutoff fails."""
    by_cat = {}
    for cat, sc, present in rows:
        by_cat.setdefault(cat, []).append((sc, present))
    thresholds = {}
    for cat, items in by_cat.items():
        scores = np.array([s for s, _ in items])
        labels = np.array([p for _, p in items])
        candidates = np.unique(np.r_[scores, -np.inf])
        acc = [((scores > t) == labels).mean() for t in candidates]
        thresholds[cat] = float(candidates[int(np.argmax(acc))])
    return thresholds


def abstention_metrics(rows, thresholds):
    pred = np.array([sc > thresholds.get(cat, -np.inf) for cat, sc, _ in rows])
    truth = np.array([p for _, _, p in rows])
    said_absent = ~pred
    return {
        "accuracy": float((pred == truth).mean()),
        "always_answer_accuracy": float(truth.mean()),
        "majority_per_type_accuracy": _majority_baseline(rows),
        "absent_precision": float((~truth[said_absent]).mean()) if said_absent.any() else 0.0,
        "absent_recall": float(said_absent[~truth].mean()) if (~truth).any() else 0.0,
        "questions": len(rows),
    }


def _majority_baseline(rows):
    by_cat = {}
    for cat, _, present in rows:
        by_cat.setdefault(cat, []).append(present)
    correct = sum(max(sum(v), len(v) - sum(v)) for v in by_cat.values())
    return correct / len(rows)
