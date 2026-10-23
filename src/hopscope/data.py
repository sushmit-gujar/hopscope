"""Dataset loading. HotpotQA (distractor setting) is the default benchmark.

Each HotpotQA question ships with 10 paragraphs: 2 gold "supporting" paragraphs and
8 distractors mined to be lexically similar. We pool the paragraphs of N questions
into one shared corpus (deduplicated by Wikipedia title), so every retriever has to
find the 2 gold paragraphs among thousands of hard negatives.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HF_REPO = "hotpotqa/hotpot_qa"
HF_FILE = "distractor/validation-00000-of-00001.parquet"


@dataclass(frozen=True)
class Passage:
    pid: int
    title: str
    text: str

    @property
    def full(self) -> str:
        return f"{self.title}: {self.text}"


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    qtype: str  # "bridge" | "comparison"
    level: str
    gold_titles: list[str]
    gold_pids: list[int] = field(default_factory=list)

    @property
    def is_yes_no(self) -> bool:
        return self.answer.strip().lower() in {"yes", "no"}


@dataclass
class Benchmark:
    passages: list[Passage]
    questions: list[Question]

    def title_to_pid(self) -> dict[str, int]:
        return {p.title: p.pid for p in self.passages}


def _norm_record(r: dict[str, Any]) -> dict[str, Any]:
    """Accept both the HF parquet layout and the original HotpotQA JSON layout."""
    ctx = r["context"]
    sf = r["supporting_facts"]
    if isinstance(ctx, dict):
        paras = list(zip(ctx["title"], ctx["sentences"]))
    else:
        paras = [(t, s) for t, s in ctx]
    if isinstance(sf, dict):
        gold = list(dict.fromkeys(sf["title"]))
    else:
        gold = list(dict.fromkeys(t for t, _ in sf))
    return {
        "id": r.get("id") or r.get("_id"),
        "question": r["question"],
        "answer": r["answer"],
        "type": r.get("type", "unknown"),
        "level": r.get("level", "unknown"),
        "paras": [(t, " ".join(s).strip()) for t, s in paras],
        "gold": gold,
    }


def build_benchmark(records: Iterable[dict[str, Any]], n: int | None = None,
                    seed: int = 13, offset: int = 0) -> Benchmark:
    """Shuffle once with ``seed`` and take questions [offset, offset+n).

    Disjoint slices (e.g. test = [0, 1000), dev = [1000, 1300)) never share questions.
    """
    recs = [_norm_record(r) for r in records]
    rng = random.Random(seed)
    rng.shuffle(recs)
    recs = recs[offset:] if n is None else recs[offset:offset + n]
    passages: list[Passage] = []
    by_title: dict[str, int] = {}
    questions: list[Question] = []
    for r in recs:
        for title, text in r["paras"]:
            if title not in by_title:
                by_title[title] = len(passages)
                passages.append(Passage(len(passages), title, text))
        q = Question(r["id"], r["question"], r["answer"], r["type"], r["level"], r["gold"])
        q.gold_pids = [by_title[t] for t in q.gold_titles if t in by_title]
        questions.append(q)
    return Benchmark(passages, questions)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.read_table(path).to_pylist())
    with path.open() as f:
        return list(json.load(f))


def download_hotpotqa(data_dir: str | Path = "data") -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", local_dir=str(data_dir)))
