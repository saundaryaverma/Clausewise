# Clausewise

Find the clauses that matter in a contract, cite exactly where they are, and say so when a clause isn't there.

![CI](https://github.com/saundaryaverma/clausewise/actions/workflows/ci.yml/badge.svg)

Contract review starts with retrieval: before anyone can answer "Can either party terminate for convenience?", the right paragraph has to be found in a 50-page agreement. Clausewise benchmarks that step on **CUAD**, a dataset of real commercial contracts where lawyers highlighted the exact text for 41 clause types, and adds a cited-answer mode that refuses to show an answer unless every quote checks out against the source.

## Results on the CUAD test set

102 contracts (about 47,000 characters each), 4,182 clause questions. 1,244 of them have a clause the lawyers highlighted; the rest are clauses the contract doesn't contain. A retrieval counts as correct when a returned chunk overlaps the lawyers' highlighted text.

| Chunk size | Query | Method | Recall@1 | Recall@3 | Recall@5 | Chars read (top 3) |
| --- | --- | --- | --- | --- | --- | --- |
| 600 | clause name only | TF-IDF | 0.350 | 0.499 | 0.586 | 1,768 |
| 600 | name + definition | BM25 | 0.350 | 0.553 | 0.623 | 1,769 |
| 600 | name + definition | **TF-IDF** | **0.389** | **0.568** | **0.650** | 1,766 |
| 600 | name + definition | Hybrid (RRF) | 0.373 | 0.571 | 0.639 | 1,769 |
| 300 | name + definition | TF-IDF | 0.359 | 0.527 | 0.614 | 881 |
| 1200 | name + definition | TF-IDF | 0.419 | 0.597 | 0.694 | 3,511 |

`clausewise eval` prints all 18 combinations (3 chunk sizes x 2 query styles x 3 methods).

What I learned:

- **Saying what you mean helps more than switching algorithms.** Adding CUAD's one-line definition of each clause type to the query raised Recall@3 from 0.499 to 0.568 at the same chunk size. Swapping BM25 for TF-IDF moved it by about 0.015.
- **Bigger chunks find more but cost more.** Going from 600 to 1,200 characters raises Recall@3 by 0.03 but doubles the text an LLM would have to read for every question. From 300 to 600 the gain is 0.04 for the same doubling. I use 600 as the default, since that's where the gains start shrinking relative to cost.
- **Fusing two rankers didn't help here.** Reciprocal rank fusion of BM25 and TF-IDF landed between the two. Both rankers rely on the same word overlap, so combining them adds little. A dense embedding model would bring a different signal and is the next thing to try.
- **About 40% of clauses still aren't in the top 3.** Contracts often phrase a concept in words the query doesn't use ("assign" vs "transfer"), which is exactly where lexical search falls short.

### Knowing when a clause isn't there

Seventy percent of CUAD questions ask about clauses the contract doesn't contain. An assistant that always returns *something* will be wrong most of the time. Clausewise learns a score cutoff per clause type from 408 training contracts, then on the test set:

- decides correctly whether the clause exists **84.2%** of the time, compared with 81.9% for guessing the most common answer for each clause type and 29.7% for always answering,
- is right **89.8%** of the time when it says a clause is absent, and catches 87.4% of absent clauses.

The improvement over the per-type baseline is real but small, which is honest information: lexical scores alone are a weak signal for absence.

## Cited answers, with a citation check

```bash
clausewise ask contract.txt "Governing Law. Which state's law governs the contract?" --model gpt-4o-mini
```

Clausewise retrieves the top passages, labels them `[C1]`, `[C2]`, and so on, and tells the model to quote exact contract language with a source id after each quote, or to reply "Not found in this contract." Before showing the answer, it verifies that:

- every cited id was actually provided,
- every quote appears in the passage it cites (ignoring whitespace and case),
- the answer doesn't make claims without any cited quote.

If any check fails, the answer is flagged and the command exits with an error. Without `--model`, `ask` returns the top passages with their exact character positions, which needs no API key.

On a real CUAD licensing agreement, asking about governing law returns two passages, both stating New York law, with their character offsets in the 140,000-character document.

## Run it

```bash
git clone https://github.com/saundaryaverma/clausewise.git
cd clausewise
pip install -e .
clausewise eval                     # downloads CUAD (about 18 MB) and runs the full benchmark in under a minute
clausewise ask my_contract.txt "Termination for convenience"
pip install -e ".[llm]"             # optional, for --model answers (needs OPENAI_API_KEY)
```

## How it works

- `data.py` loads CUAD and turns each lawyer highlight into character offsets.
- `chunking.py` splits contracts into overlapping chunks without cutting words, keeping offsets so every result can be cited.
- `retrieve.py` implements BM25, TF-IDF, and reciprocal rank fusion from scratch, with IDF computed across all chunks in the collection.
- `evaluate.py` scores retrieval against the highlights and learns per-clause-type cutoffs for abstention on the training split only.
- `answer.py` builds the citation prompt and verifies every quote.

## Tests

```bash
pip install -e ".[dev]"
pytest --cov=clausewise
```

23 tests (93% coverage) run on a small synthetic contract in CUAD's format. They cover chunk boundaries and overlap, both rankers, rank fusion, question parsing, retrieval metrics, abstention thresholds, and the citation checker: quotes that aren't in the source, quotes attributed to the wrong source, invented source ids, and uncited claims. CI runs the tests on Python 3.10 and 3.12 and then the full CUAD benchmark.

## Next

- **Dense retrieval:** add an embedding model and compare it in the same table. It should help most where contracts use different words than the query.
- **Grounded-answer benchmark:** run `ask` with a model across CUAD and measure how often answers pass the citation check and match the lawyers' highlights.

## Data and license

CUAD is from The Atticus Project and licensed CC BY 4.0. Clausewise downloads it on first run and doesn't redistribute it. Code is MIT licensed.
