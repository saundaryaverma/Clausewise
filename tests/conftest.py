import json

import pytest

CONTRACT = (
    "SUPPLY AGREEMENT. This agreement is made between Acme Corp and Beta LLC. "
    "1. Term. The term of this agreement is five years from the effective date. "
    + "Filler text about deliveries, invoices, and packaging requirements. " * 12
    + "9. Governing Law. This agreement shall be governed by the laws of the State of New York. "
    + "More filler about notices and amendments to the schedules. " * 12
    + "12. Termination. Either party may terminate for convenience on ninety days notice."
)


def qa(category, description, answer_text):
    answers = []
    if answer_text:
        answers = [{"text": answer_text, "answer_start": CONTRACT.index(answer_text)}]
    return {"question": f'Highlight the parts (if any) of this contract related to "{category}" that should be '
                        f"reviewed by a lawyer. Details: {description}", "answers": answers, "id": category}


QAS = [
    qa("Governing Law", "Which state or country's law governs interpretation of the contract?",
       "shall be governed by the laws of the State of New York"),
    qa("Termination For Convenience", "Can a party terminate this contract without cause?",
       "Either party may terminate for convenience on ninety days notice"),
    qa("Non-Compete", "Is there a restriction on competing with the counterparty?", None),
]


def write_cuad(path, n=3):
    data = {"version": "test", "data": [
        {"title": f"Contract {i}", "paragraphs": [{"context": CONTRACT, "qas": QAS}]} for i in range(n)]}
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def cuad_dir(tmp_path):
    write_cuad(tmp_path / "test.json")
    write_cuad(tmp_path / "train_separate_questions.json", n=4)
    return tmp_path
