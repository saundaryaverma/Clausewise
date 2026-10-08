"""Load CUAD: real commercial contracts with lawyer-annotated clause locations.
CUAD is released by The Atticus Project under CC BY 4.0."""
import io
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

URL = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data.zip"


def download(data_dir="data"):
    target = Path(data_dir)
    if (target / "test.json").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as res:
        zipfile.ZipFile(io.BytesIO(res.read())).extractall(target)
    return target


@dataclass
class Question:
    category: str          # e.g. "Governing Law"
    description: str       # CUAD's plain-English definition of the category
    spans: list            # [(start, end)] character offsets lawyers highlighted; empty if absent


@dataclass
class Contract:
    title: str
    text: str
    questions: list = field(default_factory=list)


_CATEGORY = re.compile(r'related to "([^"]+)"')


def parse_question(q):
    text = q["question"]
    match = _CATEGORY.search(text)
    category = match.group(1) if match else text
    description = text.split("Details:", 1)[1].strip() if "Details:" in text else ""
    spans = sorted({(a["answer_start"], a["answer_start"] + len(a["text"])) for a in q["answers"]})
    return Question(category, description, spans)


def load(path, limit=None):
    data = json.loads(Path(path).read_text())["data"]
    contracts = []
    for doc in data[:limit]:
        for para in doc["paragraphs"]:
            contracts.append(Contract(doc["title"], para["context"],
                                      [parse_question(q) for q in para["qas"]]))
    return contracts
