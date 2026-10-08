import json

from clausewise.answer import NOT_FOUND, Citation, answer, build_prompt, retrieve, verify
from clausewise.cli import main
from tests.conftest import CONTRACT

CITES = [Citation("C1", 0, 60, "This agreement shall be governed by the laws of the State of New York."),
         Citation("C2", 60, 120, "Either party may terminate for convenience on ninety days notice.")]


def test_verify_accepts_exact_quotes_with_valid_citations():
    reply = 'New York law applies: "governed by the laws of the State of New York" [C1].'
    assert verify(reply, CITES) == {"valid": True, "problems": [], "quotes_checked": 1}


def test_verify_ignores_whitespace_and_case_differences():
    assert verify('"GOVERNED  BY the laws" [C1]', CITES)["valid"]


def test_verify_catches_quotes_that_are_not_in_the_cited_source():
    check = verify('"governed by the laws of California" [C1]', CITES)
    assert not check["valid"] and "not found in [C1]" in check["problems"][0]


def test_verify_catches_quotes_attributed_to_the_wrong_source():
    assert not verify('"terminate for convenience" [C1]', CITES)["valid"]


def test_verify_catches_invented_source_ids_and_uncited_claims():
    assert "wasn't provided" in verify('"terminate" [C9]', CITES)["problems"][0]
    assert not verify("New York law applies.", CITES)["valid"]


def test_not_found_is_a_valid_answer():
    assert verify(NOT_FOUND, CITES)["valid"]


def test_retrieve_returns_cited_spans_from_the_original_text():
    cites = retrieve(CONTRACT, "governing law state", k=2, size=200, overlap=40)
    assert cites[0].id == "C1" and "Governing Law" in cites[0].text
    assert CONTRACT[cites[0].start:cites[0].end] == cites[0].text


def test_prompt_contains_rules_question_and_sources():
    prompt = build_prompt("Which law governs?", CITES)
    assert "[C1]" in prompt and "Which law governs?" in prompt and NOT_FOUND in prompt


def test_answer_with_a_model_is_verified():
    good = answer(CONTRACT, "governing law", complete=lambda p: '"the laws of the State of New York" [C1]')
    assert good["check"]["valid"]
    bad = answer(CONTRACT, "governing law", complete=lambda p: '"the laws of Delaware" [C1]')
    assert not bad["check"]["valid"]
    assert answer(CONTRACT, "governing law")["answer"] is None


def test_cli_eval_end_to_end(cuad_dir, tmp_path, capsys):
    out = tmp_path / "results"
    assert main(["eval", "--data-path", str(cuad_dir), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Recall@3" in printed and "Abstention" in printed
    saved = json.loads((out / "results.json").read_text())
    assert len(saved["retrieval"]) == 18 and saved["abstention"]["questions"] == 9


def test_cli_ask_without_a_model_prints_sources(tmp_path, capsys):
    path = tmp_path / "contract.txt"
    path.write_text(CONTRACT)
    assert main(["ask", str(path), "governing law"]) == 0
    assert "[C1] characters" in capsys.readouterr().out
