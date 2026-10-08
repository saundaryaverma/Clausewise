from clausewise.data import load, parse_question
from clausewise.evaluate import (abstention_metrics, fit_thresholds, index_all, query_text,
                                 retrieval_metrics, top_scores)


def test_parse_question_extracts_category_description_and_spans(cuad_dir):
    contracts = load(cuad_dir / "test.json")
    q = contracts[0].questions[0]
    assert q.category == "Governing Law"
    assert q.description.startswith("Which state")
    s, e = q.spans[0]
    assert contracts[0].text[s:e] == "shall be governed by the laws of the State of New York"
    assert contracts[0].questions[2].spans == []


def test_query_modes():
    q = parse_question({"question": 'related to "Cap On Liability" Details: Is liability capped?', "answers": []})
    assert query_text(q, "category") == "Cap On Liability"
    assert query_text(q, "category+description") == "Cap On Liability. Is liability capped?"


def test_retrieval_finds_annotated_clauses(cuad_dir):
    indexed = index_all(load(cuad_dir / "test.json"), size=200, overlap=40)
    for method in ("bm25", "tfidf", "hybrid"):
        r = retrieval_metrics(indexed, method)
        assert r["questions"] == 6  # 2 answerable questions x 3 contracts; absent clauses excluded
        assert r["recall@3"] == 1.0 and r["mrr"] > 0.5
        assert r["chars_read@3"] > 0


def test_absent_clauses_are_learned_and_detected(cuad_dir):
    train = top_scores(index_all(load(cuad_dir / "train_separate_questions.json"), 200, 40))
    thresholds = fit_thresholds(train)
    m = abstention_metrics(top_scores(index_all(load(cuad_dir / "test.json"), 200, 40)), thresholds)
    assert m["accuracy"] == 1.0 and m["absent_precision"] == 1.0
    assert m["always_answer_accuracy"] == 2 / 3


def test_abstention_counts_wrong_absent_calls():
    rows = [("A", 5.0, True), ("A", 0.5, True), ("A", 0.1, False)]
    m = abstention_metrics(rows, {"A": 1.0})
    assert m["absent_precision"] == 0.5 and m["accuracy"] == 2 / 3
