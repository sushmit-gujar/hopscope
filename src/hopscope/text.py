"""Text normalisation, tokenisation and LLM-free entity extraction."""

from __future__ import annotations

import re
import string
from collections import Counter

STOP = frozenset(
    "a an the of and or in on at to for from by with as is was were are be been being "
    "which who whom whose what when where why how that this these those it its his her "
    "their did does do has have had than then also not no yes both either same more most"
    .split()
)

_TOKEN = re.compile(r"[A-Za-z0-9]+")
# Capitalised spans, allowing a few lowercase connectors inside ("Bank of America").
_ENTITY = re.compile(
    r"\b(?:[A-Z][\w'’\-\.]*|[0-9]{3,4})(?:\s+(?:of|the|de|del|la|von|van|and|for|&)?\s*"
    r"(?:[A-Z][\w'’\-\.]*|[0-9]{1,4}))*"
)
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_SENT_START_STOP = frozenset({"the", "a", "an", "in", "on", "it", "he", "she", "they", "this",
                              "his", "her", "after", "during", "when", "while", "however"})


def tokenize(text: str) -> list[str]:
    return [t for t in (m.lower() for m in _TOKEN.findall(text)) if t not in STOP]


def normalize_entity(name: str) -> str:
    name = _PAREN.sub("", name).lower()
    name = name.translate(str.maketrans("", "", string.punctuation + "’"))
    toks = name.split()
    while toks and toks[0] in {"the", "a", "an"}:
        toks = toks[1:]
    return " ".join(toks)


def extract_entities(text: str, title: str | None = None) -> list[str]:
    """Return normalised entity strings mentioned in ``text``.

    A deliberately simple, deterministic extractor (no LLM, no NER model): capitalised
    spans + the passage title. HippoRAG uses an LLM for OpenIE; this ablation shows how
    far a zero-cost extractor gets you.
    """
    out: list[str] = []
    if title:
        out.append(normalize_entity(title))
    for m in _ENTITY.finditer(text):
        ent = normalize_entity(m.group(0))
        if not ent or ent in _SENT_START_STOP or ent in STOP or len(ent) < 2:
            continue
        out.append(ent)
    return list(dict.fromkeys(out))


# --- official HotpotQA answer normalisation -------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)
