"""Retrieval, answer and attribution metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .text import exact_match, f1_score


def evidence_recall(retrieved: Sequence[int], gold: Sequence[int], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return sum(g in top for g in gold) / len(gold)


def full_support(retrieved: Sequence[int], gold: Sequence[int], k: int) -> float:
    """1 if *every* gold passage is in the top-k: the multi-hop chain is complete."""
    top = set(retrieved[:k])
    return float(bool(gold) and all(g in top for g in gold))


def mrr(retrieved: Sequence[int], gold: Sequence[int]) -> float:
    for i, p in enumerate(retrieved):
        if p in gold:
            return 1.0 / (i + 1)
    return 0.0


def answer_scores(pred: str, gold: str) -> tuple[float, float]:
    return exact_match(pred, gold), f1_score(pred, gold)


def attribution(source_pids: Sequence[int], gold: Sequence[int]) -> float:
    """Share of cited/source passages that are gold supporting passages."""
    if not source_pids:
        return 0.0
    return sum(p in gold for p in source_pids) / len(source_pids)


def pct(xs: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q)) if len(xs) else 0.0
