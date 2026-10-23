"""Command-line entry point: ``hopscope download`` / ``hopscope run``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .bench import ALL_METHODS, run, save, to_markdown
from .data import Benchmark, build_benchmark, download_hotpotqa, load_records
from .encoders import Encoder, get_encoder
from .readers import get_reader


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hopscope", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch HotpotQA (distractor, validation) from HF Hub")
    d.add_argument("--data-dir", default="data")
    r = sub.add_parser("run", help="run the benchmark")
    r.add_argument("--data", default=None, help="parquet/json file (default: download)")
    r.add_argument("-n", "--n-questions", type=int, default=500)
    r.add_argument("--seed", type=int, default=13)
    r.add_argument("--methods", default=",".join(ALL_METHODS),
                   help=f"comma list from {ALL_METHODS} (+ 'oracle' = gold passages)")
    r.add_argument("--encoder", default="BAAI/bge-small-en-v1.5",
                   help="fastembed model name, or 'hashing' for the dependency-free fallback")
    r.add_argument("--reader", default="onnx", choices=["onnx", "lexical", "llm", "none"])
    r.add_argument("--k-read", type=int, default=5)
    r.add_argument("--out", default="results/latest")
    r.add_argument("--offset", type=int, default=0, help="start of the question slice")
    r.add_argument("--cache-dir", default="data/cache", help="embedding cache ('' = off)")
    t = sub.add_parser("tune", help="grid-search graph params on a disjoint DEV slice")
    t.add_argument("--data", default=None)
    t.add_argument("-n", "--n-questions", type=int, default=300)
    t.add_argument("--offset", type=int, default=1000)
    t.add_argument("--seed", type=int, default=13)
    t.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    t.add_argument("--cache-dir", default="data/cache")
    t.add_argument("--out", default="results/tune-dev.json")
    a = ap.parse_args(argv)
    if a.cmd == "download":
        print(download_hotpotqa(a.data_dir))
        return 0
    path = a.data or download_hotpotqa()
    bench = build_benchmark(load_records(path), n=a.n_questions, seed=a.seed, offset=a.offset)
    if a.cmd == "tune":
        return tune(bench, get_encoder(a.encoder), a.cache_dir or None, a.out)
    print(f"corpus: {len(bench.passages)} passages, {len(bench.questions)} questions")
    reader = None if a.reader == "none" else get_reader(a.reader)
    res = run(bench, get_encoder(a.encoder), reader, a.methods.split(","), k_read=a.k_read,
              seed=a.seed, cache_dir=a.cache_dir or None, offset=a.offset)
    out = save(res, a.out)
    print(to_markdown(res))
    for m, pv in res.paired_vs_dense.items():
        print(f"{m} vs dense Full@5: {pv['full_support@5_delta']:+.3f} "
              f"[{pv['full_support@5_ci_lo']:+.3f}, {pv['full_support@5_ci_hi']:+.3f}]")
    print(f"\nindex: {res.index_stats}\nwrote {out}/")
    return 0


def tune(bench: Benchmark, encoder: Encoder, cache_dir: str | None, out: str) -> int:
    """Grid search on the dev slice; objective = full_support@5 (complete evidence chains)."""
    grid: list[tuple[str, dict[str, float]]] = []
    for d in (0.3, 0.5, 0.7, 0.85):
        for w in (0.25, 0.5, 0.75):
            grid.append(("graph_ppr+dense", {"damping": d, "dense_weight": w}))
    for s in (2, 3, 5):
        for al in (0.25, 0.5, 1.0):
            grid.append(("graph_expand", {"expand_seeds": s, "expand_alpha": al}))
    rows: list[dict[str, Any]] = []
    base = run(bench, encoder, None, ["dense"], cache_dir=cache_dir, progress=False)
    rows.append({"method": "dense", "params": {}, **base.summary["dense"]})
    for m, prm in grid:
        res = run(bench, encoder, None, [m], params=prm, cache_dir=cache_dir, progress=False)
        rows.append({"method": m, "params": prm, **res.summary[m]})
        s_ = res.summary[m]
        print(f"{m:16s} {prm} R@5={s_['recall@5']:.3f} Full@5={s_['full_support@5']:.3f}",
              flush=True)
    best = {}
    for m in ("graph_ppr+dense", "graph_expand"):
        best[m] = max((r for r in rows if r["method"] == m), key=lambda r: r["full_support@5"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"slice": {"n_questions": len(bench.questions),
                                               "note": "disjoint dev slice"},
                                     "dense": rows[0], "best": best, "grid": rows}, indent=2))
    print("dense", rows[0]["full_support@5"], "best:",
          {m: (b["params"], b["full_support@5"]) for m, b in best.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
