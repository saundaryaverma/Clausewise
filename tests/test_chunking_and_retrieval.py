import numpy as np
import pytest

from clausewise.chunking import chunk, overlaps
from clausewise.retrieve import BM25, TfIdf, corpus_idf, rrf, tokenize
from tests.conftest import CONTRACT


def test_chunks_cover_the_text_with_overlap_and_no_split_words():
    spans = chunk(CONTRACT, size=200, overlap=40)
    assert spans[0][0] == 0 and spans[-1][1] == len(CONTRACT)
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s2 < e1  # consecutive chunks overlap
        assert e1 == len(CONTRACT) or CONTRACT[e1] == " "


def test_chunk_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        chunk("text", size=100, overlap=100)


def test_overlaps():
    assert overlaps((0, 10), [(5, 20)])
    assert not overlaps((0, 10), [(10, 20)])


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The Governing Law of the State!") == ["governing", "law", "state"]


def corpus():
    docs = ["governing law state new york", "term five years effective date", "invoices packaging deliveries"]
    tokens = [tokenize(d) for d in docs]
    idf, _ = corpus_idf(tokens)
    return tokens, idf


def test_bm25_and_tfidf_rank_the_matching_chunk_first():
    tokens, idf = corpus()
    q = tokenize("governing law")
    assert np.argmax(BM25(tokens, idf).scores(q)) == 0
    assert np.argmax(TfIdf(tokens, idf).scores(q)) == 0


def test_unknown_query_words_score_zero():
    tokens, idf = corpus()
    assert BM25(tokens, idf).scores(["zebra"]).sum() == 0


def test_rrf_rewards_agreement_between_rankers():
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([1.0, 3.0, 2.0])
    assert np.argmax(rrf(a, b)) == 1
