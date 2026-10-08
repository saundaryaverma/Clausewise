"""Answer a question about one contract with citations, and verify every citation.

A legal answer is only as good as its sources, so the model may only quote text from the
retrieved chunks, must cite the chunk for each quote, and the verifier checks that every
quote really appears in the chunk it cites before the answer is shown."""
import re
from dataclasses import dataclass

import numpy as np

from clausewise.chunking import chunk
from clausewise.retrieve import BM25, TfIdf, corpus_idf, rrf, tokenize

NOT_FOUND = "Not found in this contract."


@dataclass
class Citation:
    id: str
    start: int
    end: int
    text: str


def retrieve(text, question, k=3, size=600, overlap=100):
    spans = chunk(text, size, overlap)
    tokens = [tokenize(text[s:e]) for s, e in spans]
    idf, _ = corpus_idf(tokens)
    q = tokenize(question)
    scores = rrf(BM25(tokens, idf).scores(q), TfIdf(tokens, idf).scores(q))
    top = np.argsort(-scores, kind="stable")[:k]
    return [Citation(f"C{n + 1}", spans[i][0], spans[i][1], text[spans[i][0]:spans[i][1]]) for n, i in enumerate(top)]


def build_prompt(question, citations):
    sources = "\n\n".join(f"[{c.id}] (characters {c.start}-{c.end})\n{c.text}" for c in citations)
    return (
        "You review contracts. Answer the question using only the sources below.\n"
        "Rules: quote the exact contract language in double quotes, and put the source id right after "
        "each quote, like \"...\" [C1]. If the sources don't answer the question, reply exactly: "
        f"{NOT_FOUND}\n\nQuestion: {question}\n\nSources:\n{sources}"
    )


_QUOTE_CITE = re.compile(r'"([^"]{3,})"\s*\[(C\d+)\]')
_CITE = re.compile(r"\[(C\d+)\]")


def _norm(s):
    return " ".join(s.split()).lower()


def verify(answer, citations):
    """Check that every cited id exists and every quote appears verbatim in its cited chunk."""
    by_id = {c.id: c for c in citations}
    problems = []
    for cid in _CITE.findall(answer):
        if cid not in by_id:
            problems.append(f"cites [{cid}], which wasn't provided")
    quotes = _QUOTE_CITE.findall(answer)
    for quote, cid in quotes:
        if cid in by_id and _norm(quote) not in _norm(by_id[cid].text):
            problems.append(f"quote not found in [{cid}]: \"{quote[:60]}\"")
    if answer.strip() != NOT_FOUND and not quotes:
        problems.append("answer makes claims without any cited quote")
    return {"valid": not problems, "problems": problems, "quotes_checked": len(quotes)}


def answer(text, question, complete=None, k=3):
    """With a model (complete(prompt) -> str), returns a verified cited answer.
    Without one, returns the top passages as an extractive answer."""
    citations = retrieve(text, question, k)
    if complete is None:
        return {"answer": None, "citations": citations, "check": None}
    reply = complete(build_prompt(question, citations)).strip()
    return {"answer": reply, "citations": citations, "check": verify(reply, citations)}


def openai_complete(model):
    from openai import OpenAI

    client = OpenAI()

    def complete(prompt):
        res = client.chat.completions.create(model=model, temperature=0,
                                             messages=[{"role": "user", "content": prompt}])
        return res.choices[0].message.content or ""

    return complete
