"""clausewise eval                         benchmark retrieval and abstention on CUAD
clausewise ask CONTRACT.txt "QUESTION"   cited answer for one contract"""
import argparse
import json
import sys
from pathlib import Path

from clausewise import data as cuad
from clausewise.answer import answer, openai_complete
from clausewise.evaluate import (abstention_metrics, fit_thresholds, index_all,
                                 retrieval_metrics, top_scores)

CONFIGS = [(300, 50), (600, 100), (1200, 200)]
METHODS = [("bm25", "BM25"), ("tfidf", "TF-IDF"), ("hybrid", "Hybrid (RRF)")]


def run_eval(args):
    folder = Path(args.data_path) if args.data_path else cuad.download(args.data_dir)
    test = cuad.load(folder / "test.json", args.limit)
    train = cuad.load(folder / "train_separate_questions.json", args.limit)
    print(f"CUAD test split: {len(test)} contracts, "
          f"{sum(len(c.questions) for c in test)} clause questions\n")

    lines = ["| Chunk size | Query | Method | Recall@1 | Recall@3 | Recall@5 | MRR | Chars read (top 3) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    results = []
    for size, overlap in CONFIGS:
        indexed = index_all(test, size, overlap)
        for mode in ("category", "category+description"):
            for key, label in METHODS:
                r = retrieval_metrics(indexed, key, mode)
                results.append({"size": size, "query": mode, "method": label, **r})
                lines.append(f"| {size} | {'name + definition' if mode != 'category' else 'name only'} | {label} | "
                             f"{r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['recall@5']:.3f} | "
                             f"{r['mrr']:.3f} | {r['chars_read@3']:,.0f} |")
    table = "\n".join(lines)

    thresholds = fit_thresholds(top_scores(index_all(train, 600, 100)))
    ab = abstention_metrics(top_scores(index_all(test, 600, 100)), thresholds)
    abstain = (f"Abstention (is the clause in this contract at all?), thresholds tuned on {len(train)} train contracts:\n"
               f"- Accuracy {ab['accuracy']:.1%} vs {ab['majority_per_type_accuracy']:.1%} for guessing the most common "
               f"answer per clause type and {ab['always_answer_accuracy']:.1%} for always answering\n"
               f"- When it says a clause is absent, it's right {ab['absent_precision']:.1%} of the time, "
               f"and it catches {ab['absent_recall']:.1%} of absent clauses")
    print(table + "\n\n" + abstain)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(table + "\n\n" + abstain + "\n")
    (out / "results.json").write_text(json.dumps({"retrieval": results, "abstention": ab}, indent=2))
    return 0


def run_ask(args):
    text = Path(args.contract).read_text(errors="ignore")
    complete = openai_complete(args.model) if args.model else None
    result = answer(text, args.question, complete, args.k)
    if result["answer"]:
        print(result["answer"])
        check = result["check"]
        print("\nCitation check:", "PASSED" if check["valid"] else "FAILED")
        for p in check["problems"]:
            print("  -", p)
    print("\nSources:")
    for c in result["citations"]:
        print(f"[{c.id}] characters {c.start}-{c.end}: {' '.join(c.text.split())[:300]}...")
    return 0 if not result["check"] or result["check"]["valid"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="clausewise")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--data-dir", default="data")
    ev.add_argument("--data-path", help="folder with test.json and train_separate_questions.json")
    ev.add_argument("--limit", type=int, help="only use the first N contracts (for quick runs)")
    ev.add_argument("--out", default="results")
    ask = sub.add_parser("ask")
    ask.add_argument("contract")
    ask.add_argument("question")
    ask.add_argument("--model", help="OpenAI model for a written answer, e.g. gpt-4o-mini")
    ask.add_argument("-k", type=int, default=3)
    args = parser.parse_args(argv)
    return run_eval(args) if args.command == "eval" else run_ask(args)


if __name__ == "__main__":
    sys.exit(main())
