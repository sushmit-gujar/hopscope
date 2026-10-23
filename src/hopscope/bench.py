"""Run the benchmark: build indexes, retrieve, read, score, time."""

from __future__ import annotations

import json
import platform
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .data import Benchmark
from .encoders import Encoder
from .metrics import answer_scores, attribution, evidence_recall, full_support, mrr, pct
from .readers import Reader
from .retrievers import (
    BM25Retriever,
    DenseRetriever,
    EntityGraph,
    GraphExpandRetriever,
    GraphPPRRetriever,
    HybridRetriever,
    Retriever,
)

ALL_METHODS = ["bm25", "dense", "hybrid", "graph_ppr", "graph_ppr+dense", "graph_expand"]


# Chosen on a disjoint dev slice (see `hopscope tune`), never on the reported test slice.
# Dev slice = questions [1000, 1300) of the seed-13 shuffle; test = [0, 1000).
DEFAULT_PARAMS: dict[str, float] = {"damping": 0.7, "dense_weight": 0.75, "expand_seeds": 5,
                                    "expand_alpha": 0.25}


@dataclass
class RunResult:
    config: dict[str, Any]
    index_stats: dict[str, Any]
    summary: dict[str, dict[str, float]]
    by_type: dict[str, dict[str, dict[str, float]]]
    per_question: list[dict[str, Any]] = field(default_factory=list)
    paired_vs_dense: dict[str, dict[str, float]] = field(default_factory=dict)


def paired_bootstrap(per_q: list[dict[str, Any]], methods: list[str], baseline: str = "dense",
                     metrics: tuple[str, ...] = ("full_support@5", "recall@5"),
                     n_boot: int = 2000, seed: int = 0) -> dict[str, dict[str, float]]:
    """Paired bootstrap 95% CI of (method - baseline) per metric over the same questions."""
    if baseline not in methods or not per_q:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_q), size=(n_boot, len(per_q)))
    out: dict[str, dict[str, float]] = {}
    for m in methods:
        if m in (baseline, "oracle"):
            continue
        row: dict[str, float] = {}
        for met in metrics:
            d = np.array([q["methods"][m][met] - q["methods"][baseline][met] for q in per_q])
            boots = d[idx].mean(axis=1)
            row[f"{met}_delta"] = round(float(d.mean()), 4)
            row[f"{met}_ci_lo"] = round(float(np.percentile(boots, 2.5)), 4)
            row[f"{met}_ci_hi"] = round(float(np.percentile(boots, 97.5)), 4)
        out[m] = row
    return out


def build_retrievers(bench: Benchmark, encoder: Encoder, methods: list[str],
                     params: dict[str, float] | None = None, cache_dir: str | None = None
                     ) -> tuple[dict[str, Retriever], dict[str, Any]]:
    prm = {**DEFAULT_PARAMS, **(params or {})}
    stats: dict[str, Any] = {"passages": len(bench.passages), "questions": len(bench.questions)}
    out: dict[str, Retriever] = {}
    t = time.perf_counter()
    bm25 = BM25Retriever(bench.passages)
    stats["bm25_build_s"] = round(time.perf_counter() - t, 2)
    dense = None
    if any(m in methods for m in ("dense", "hybrid", "graph_ppr+dense", "graph_ppr",
                                  "graph_expand")):
        t = time.perf_counter()
        dense = DenseRetriever(bench.passages, encoder, cache_dir=cache_dir)
        stats["dense_build_s"] = round(time.perf_counter() - t, 2)
    graph = None
    if any(m.startswith("graph") for m in methods):
        t = time.perf_counter()
        graph = EntityGraph(bench.passages)
        stats["graph_build_s"] = round(time.perf_counter() - t, 2)
        stats["graph_entities"] = graph.n_ent
        stats["graph_edges"] = graph.n_edges
    for m in methods:
        if m == "oracle":
            out[m] = bm25  # placeholder; bench.run feeds gold passages directly
        elif m == "bm25":
            out[m] = bm25
        elif m == "dense":
            assert dense is not None
            out[m] = dense
        elif m == "hybrid":
            assert dense is not None
            out[m] = HybridRetriever(bm25, dense)
        elif m == "graph_ppr":
            assert graph is not None
            # dense is only a fallback when the query has no matched entity
            out[m] = GraphPPRRetriever(graph, dense, dense_weight=0.0, name="graph_ppr",
                                       damping=prm["damping"])
        elif m == "graph_ppr+dense":
            assert graph is not None and dense is not None
            out[m] = GraphPPRRetriever(graph, dense, dense_weight=prm["dense_weight"],
                                       damping=prm["damping"])
        elif m == "graph_expand":
            assert graph is not None and dense is not None
            out[m] = GraphExpandRetriever(graph, dense, seeds=int(prm["expand_seeds"]),
                                          alpha=prm["expand_alpha"])
        else:
            raise ValueError(f"unknown method {m}")
    return out, stats


def run(bench: Benchmark, encoder: Encoder, reader: Reader | None, methods: list[str],
        k_read: int = 5, ks: tuple[int, ...] = (2, 5, 10), dataset: str = "hotpotqa",
        seed: int = 13, progress: bool = True, params: dict[str, float] | None = None,
        cache_dir: str | None = None, offset: int = 0) -> RunResult:
    retrievers, stats = build_retrievers(bench, encoder, methods, params, cache_dir)
    depth = max(max(ks), k_read)
    per_q: list[dict[str, Any]] = []
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    agg_t: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for qi, q in enumerate(bench.questions):
        rec: dict[str, Any] = {"qid": q.qid, "question": q.question, "answer": q.answer,
                               "type": q.qtype, "gold": q.gold_titles, "methods": {}}
        for name, r in retrievers.items():
            t = time.perf_counter()
            if name == "oracle":  # reader ceiling: gold passages only
                pids = list(q.gold_pids)
            else:
                pids = [h.pid for h in r.retrieve(q.question, depth)]
            lat = (time.perf_counter() - t) * 1000
            row: dict[str, float] = {"latency_ms": lat, "mrr": mrr(pids, q.gold_pids)}
            for k in ks:
                row[f"recall@{k}"] = evidence_recall(pids, q.gold_pids, k)
                row[f"full_support@{k}"] = full_support(pids, q.gold_pids, k)
            mrec: dict[str, Any] = {"top": [bench.passages[p].title for p in pids[:k_read]]}
            if reader is not None and not q.is_yes_no:
                t = time.perf_counter()
                ans = reader.answer(q.question, [bench.passages[p] for p in pids[:k_read]])
                row["reader_ms"] = (time.perf_counter() - t) * 1000
                em, f1 = answer_scores(ans.text, q.answer)
                att = attribution(ans.source_pids, q.gold_pids)
                row.update({"em": em, "f1": f1, "attribution": att,
                            "grounded_em": float(em == 1.0 and att == 1.0),
                            "ungrounded_answer": float(bool(ans.text) and att == 0.0)})
                mrec["pred"] = ans.text
            mrec.update({key: round(v, 4) for key, v in row.items()})
            rec["methods"][name] = mrec
            for key, v in row.items():
                agg[name][key].append(v)
                agg_t[q.qtype][name][key].append(v)
        per_q.append(rec)
        if progress and (qi + 1) % 100 == 0:
            print(f"  {qi + 1}/{len(bench.questions)} questions", flush=True)

    def summarise(d: dict[str, list[float]]) -> dict[str, float]:
        s = {k: round(float(np.mean(v)), 4) for k, v in d.items() if k != "latency_ms"}
        s["n"] = len(d["mrr"])
        s["n_answerable_span"] = len(d.get("em", []))
        s["latency_p50_ms"] = round(pct(d["latency_ms"], 50), 2)
        s["latency_p95_ms"] = round(pct(d["latency_ms"], 95), 2)
        if "reader_ms" in d:
            s["reader_ms"] = round(float(np.mean(d["reader_ms"])), 1)
        return s

    paired = paired_bootstrap(per_q, list(retrievers), baseline="dense")
    config = {"dataset": dataset, "split": "validation (distractor)", "seed": seed,
              "n_questions": len(bench.questions), "offset": offset, "methods": methods,
              "k_read": k_read, "params": {**DEFAULT_PARAMS, **(params or {})},
              "encoder": encoder.name, "reader": reader.name if reader else None,
              "hopscope": __version__, "python": platform.python_version(),
              "machine": f"{platform.system()} {platform.machine()}, CPU only"}
    return RunResult(
        config=config,
        index_stats=stats,
        summary={m: summarise(d) for m, d in agg.items()},
        by_type={t: {m: summarise(d) for m, d in v.items()} for t, v in agg_t.items()},
        per_question=per_q,
        paired_vs_dense=paired,
    )


def to_markdown(res: RunResult) -> str:
    cols = [("recall@2", "R@2"), ("recall@5", "R@5"), ("full_support@5", "Full@5"),
            ("full_support@10", "Full@10"), ("mrr", "MRR"), ("em", "EM"), ("f1", "F1"),
            ("attribution", "Attrib."), ("ungrounded_answer", "Ungrounded"),
            ("latency_p50_ms", "p50 ms"), ("latency_p95_ms", "p95 ms")]
    lines = ["| Method | " + " | ".join(c[1] for c in cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for m, s in res.summary.items():
        vals = []
        for key, _ in cols:
            v = s.get(key)
            if v is None:
                vals.append("–")
            elif key.endswith("_ms"):
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v:.3f}")
        lines.append(f"| `{m}` | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save(res: RunResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = {"config": res.config, "index_stats": res.index_stats, "summary": res.summary,
            "by_type": res.by_type, "paired_vs_dense": res.paired_vs_dense}
    (out / "summary.json").write_text(json.dumps(body, indent=2))
    (out / "per_question.json").write_text(json.dumps(res.per_question))
    (out / "summary.md").write_text(to_markdown(res) + "\n")
    return out
