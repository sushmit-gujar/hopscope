from pathlib import Path

import numpy as np
import pytest

from hopscope.bench import run, save
from hopscope.cli import main
from hopscope.data import Passage, build_benchmark, load_records
from hopscope.encoders import HashingEncoder
from hopscope.readers import LexicalReader
from hopscope.retrievers import (
    BM25Retriever,
    DenseRetriever,
    EntityGraph,
    GraphPPRRetriever,
    HybridRetriever,
)

FIX = Path(__file__).parent / "fixtures" / "hotpot_mini.json"

# A toy 2-hop world: the answer needs passage 0 (bridge) + passage 1 (answer).
TOY = [
    Passage(0, "Acme Rocket", "Acme Rocket was designed by Jane Doe at Orbital Labs."),
    Passage(1, "Jane Doe", "Jane Doe was born in Springfield and studied physics."),
    Passage(2, "Springfield", "Springfield is a city with a large public library."),
    Passage(3, "Beta Rocket", "Beta Rocket was designed by John Roe."),
    Passage(4, "John Roe", "John Roe was born in Shelbyville."),
]


@pytest.fixture(scope="module")
def bench():
    return build_benchmark(load_records(FIX), n=None)


def test_build_benchmark_pools_and_dedups(bench):
    assert len(bench.questions) == 30
    titles = [p.title for p in bench.passages]
    assert len(titles) == len(set(titles))
    for q in bench.questions:
        assert len(q.gold_pids) == len(q.gold_titles) == 2


def test_bm25_and_dense_rank_exact_passage_first():
    q = "Where was Jane Doe born?"
    assert BM25Retriever(TOY).retrieve(q, 1)[0].pid == 1
    assert DenseRetriever(TOY, HashingEncoder()).retrieve(q, 1)[0].pid == 1


def test_graph_ppr_reaches_second_hop():
    g = EntityGraph(TOY, min_max_df=5)
    # The query only names the rocket; PPR must walk rocket -> "jane doe" -> passage 1.
    hits = GraphPPRRetriever(g, None, dense_weight=0.0).retrieve(
        "Where was the designer of Acme Rocket born?", 3)
    pids = [h.pid for h in hits]
    assert pids[0] == 0
    assert 1 in pids[:2]
    assert 4 not in pids[:2]


def test_ppr_is_a_probability_distribution():
    g = EntityGraph(TOY, min_max_df=5)
    seeds = np.zeros(g.n)
    seeds[0] = 1.0
    r = g.ppr(seeds)
    assert r.min() >= 0
    assert abs(r.sum() - 1.0) < 1e-6


def test_hybrid_fuses_both_rankers():
    bm, de = BM25Retriever(TOY), DenseRetriever(TOY, HashingEncoder())
    hits = HybridRetriever(bm, de).retrieve("Jane Doe Springfield", 3)
    assert {h.pid for h in hits} >= {1}


def test_end_to_end_run_produces_sane_metrics(bench, tmp_path):
    methods = ["bm25", "dense", "hybrid", "graph_ppr", "graph_ppr+dense", "oracle"]
    res = run(bench, HashingEncoder(), LexicalReader(), methods, progress=False)
    assert set(res.summary) == set(methods)
    for s in res.summary.values():
        assert 0.0 <= s["recall@5"] <= 1.0
        assert s["recall@10"] >= s["recall@5"] >= s["recall@2"]
        assert s["full_support@10"] >= s["full_support@5"]
    assert res.summary["oracle"]["full_support@2"] == 1.0
    assert res.summary["bm25"]["recall@10"] > 0.5  # tiny corpus: lexical search works
    out = save(res, tmp_path)
    assert (out / "summary.md").read_text().startswith("| Method |")


def test_cli_runs_offline(tmp_path, capsys):
    rc = main(["run", "--data", str(FIX), "-n", "10", "--encoder", "hashing", "--reader",
               "lexical", "--out", str(tmp_path / "r")])
    assert rc == 0
    assert (tmp_path / "r" / "summary.json").exists()
    assert "graph_ppr+dense" in capsys.readouterr().out
