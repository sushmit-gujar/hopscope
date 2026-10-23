"""Retrievers: BM25, dense, hybrid (RRF) and graph (entity KG + Personalized PageRank)."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from .data import Passage
from .encoders import Encoder
from .text import extract_entities, normalize_entity, tokenize


@dataclass
class Hit:
    pid: int
    score: float


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> list[Hit]: ...


def _topk(scores: np.ndarray, k: int) -> list[Hit]:
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [Hit(int(i), float(scores[i])) for i in idx]


class BM25Retriever:
    name = "bm25"

    def __init__(self, passages: list[Passage], k1: float = 1.2, b: float = 0.75) -> None:
        docs = [tokenize(p.full) for p in passages]
        self.n = len(docs)
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        avg = float(lengths.mean()) if self.n else 1.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        for i, d in enumerate(docs):
            tf = Counter(d)
            norm = k1 * (1 - b + b * lengths[i] / avg)
            for t, c in tf.items():
                j = vocab.setdefault(t, len(vocab))
                idf = math.log(1 + (self.n - df[t] + 0.5) / (df[t] + 0.5))
                rows.append(i)
                cols.append(j)
                vals.append(idf * c * (k1 + 1) / (c + norm))
        self.vocab = vocab
        self.mat = sp.csr_matrix((vals, (rows, cols)), shape=(self.n, len(vocab))).tocsc()

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not cols:
            return np.zeros(self.n, dtype=np.float32)
        return np.asarray(self.mat[:, cols].sum(axis=1)).ravel()

    def retrieve(self, query: str, k: int) -> list[Hit]:
        return _topk(self.scores(query), k)


class DenseRetriever:
    name = "dense"

    def __init__(self, passages: list[Passage], encoder: Encoder,
                 cache_dir: str | None = None) -> None:
        self.encoder = encoder
        texts = [p.full for p in passages]
        cache = None
        if cache_dir:
            h = hashlib.sha256((encoder.name + "\x00" + "\x00".join(texts)).encode()).hexdigest()
            cache = Path(cache_dir) / f"emb-{h[:24]}.npy"
            if cache.exists():
                self.emb = np.load(cache)
                return
        self.emb = encoder.encode(texts)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.emb)

    def scores(self, query: str) -> np.ndarray:
        q = self.encoder.encode([query], query=True)[0]
        return self.emb @ q

    def retrieve(self, query: str, k: int) -> list[Hit]:
        return _topk(self.scores(query), k)


class HybridRetriever:
    """Reciprocal-rank fusion of BM25 and dense rankings."""

    name = "hybrid"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60,
                 depth: int = 100) -> None:
        self.bm25, self.dense, self.rrf_k, self.depth = bm25, dense, rrf_k, depth

    def retrieve(self, query: str, k: int) -> list[Hit]:
        fused: dict[int, float] = defaultdict(float)
        for r in (self.bm25, self.dense):
            for rank, h in enumerate(r.retrieve(query, self.depth)):
                fused[h.pid] += 1.0 / (self.rrf_k + rank + 1)
        best = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [Hit(p, s) for p, s in best]


class EntityGraph:
    """Passage–entity bipartite knowledge graph built without an LLM.

    Nodes: passages + normalised entities (capitalised spans, Wikipedia titles).
    Edges: passage —mentions→ entity (weight 1), passage —is-about→ title entity (weight 2).
    Two passages are 2 hops apart iff they share an entity, which is exactly the
    "bridge" structure multi-hop questions need.
    """

    def __init__(self, passages: list[Passage], max_df_ratio: float = 0.01,
                 min_max_df: int = 20) -> None:
        self.n_pass = len(passages)
        mentions: list[list[str]] = [extract_entities(p.text, p.title) for p in passages]
        df: Counter[str] = Counter()
        for ents in mentions:
            df.update(ents)
        max_df = max(min_max_df, int(max_df_ratio * self.n_pass))
        self.ent_index: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for pid, (p, ents) in enumerate(zip(passages, mentions)):
            title_ent = normalize_entity(p.title)
            for e in ents:
                if df[e] > max_df and e != title_ent:
                    continue  # too generic ("american", "january") to be a useful bridge
                eid = self.ent_index.setdefault(e, len(self.ent_index))
                rows.append(pid)
                cols.append(eid)
                vals.append(2.0 if e == title_ent else 1.0)
        self.n_ent = len(self.ent_index)
        # title entity -> passages that are *about* it (Wikipedia-style hyperlink targets)
        self.title_of: dict[int, list[int]] = defaultdict(list)
        for pid, p in enumerate(passages):
            tid = self.ent_index.get(normalize_entity(p.title))
            if tid is not None:
                self.title_of[tid].append(pid)
        self.mentions = mentions
        self.df = np.zeros(self.n_ent, dtype=np.float32)
        b = sp.csr_matrix((vals, (rows, cols)), shape=(self.n_pass, self.n_ent))
        self.df = np.asarray((b > 0).sum(axis=0)).ravel().astype(np.float32)
        n = self.n_pass + self.n_ent
        adj = sp.bmat([[None, b], [b.T, None]], format="csr")
        deg = np.asarray(adj.sum(axis=0)).ravel()
        deg[deg == 0] = 1.0
        self.trans = (adj @ sp.diags(1.0 / deg)).tocsr()  # column-stochastic
        self.n = n
        self.max_ngram = 6
        self.n_edges = int(b.nnz)

    def query_entities(self, query: str) -> dict[int, float]:
        """Match known entities inside the query (n-gram lookup); weight by specificity."""
        toks = normalize_entity(query).split()
        found: dict[int, float] = {}
        for n in range(self.max_ngram, 0, -1):
            for i in range(len(toks) - n + 1):
                gram = " ".join(toks[i:i + n])
                eid = self.ent_index.get(gram)
                if eid is not None and (n > 1 or gram not in _COMMON_SINGLE):
                    found[eid] = 1.0 / max(self.df[eid], 1.0)
        return found

    def ppr(self, seeds: np.ndarray, damping: float = 0.5, iters: int = 30,
            tol: float = 1e-6) -> np.ndarray:
        p = seeds / seeds.sum()
        r = p.copy()
        for _ in range(iters):
            nxt = (1 - damping) * p + damping * (self.trans @ r)
            if np.abs(nxt - r).sum() < tol:
                r = nxt
                break
            r = nxt
        return r


_COMMON_SINGLE = frozenset({"which", "what", "who", "when", "where", "how", "american", "film",
                            "album", "band", "city", "county", "state", "university"})


class GraphPPRRetriever:
    """HippoRAG-style retrieval: seed PPR with query entities (+ optional dense passages).

    ``dense_weight=0`` gives pure entity-seeded graph retrieval; ``dense_weight>0`` blends
    dense passage scores into the personalisation vector (HippoRAG 2 style).
    """

    def __init__(self, graph: EntityGraph, dense: DenseRetriever | None = None,
                 dense_weight: float = 0.5, dense_seeds: int = 10, damping: float = 0.5,
                 name: str | None = None) -> None:
        self.g, self.dense = graph, dense
        self.dense_weight, self.dense_seeds, self.damping = dense_weight, dense_seeds, damping
        self.name = name or ("graph_ppr+dense" if dense is not None and dense_weight > 0
                             else "graph_ppr")

    def retrieve(self, query: str, k: int) -> list[Hit]:
        seeds = np.zeros(self.g.n, dtype=np.float64)
        ents = self.g.query_entities(query)
        if ents:
            w = np.array(list(ents.values()))
            seeds[[self.g.n_pass + e for e in ents]] = (1 - self.dense_weight) * w / w.sum()
        dense_scores = None
        if self.dense is not None and (self.dense_weight > 0 or not ents):
            dense_scores = self.dense.scores(query)
            top = _topk(dense_scores, self.dense_seeds)
            s = np.array([max(h.score, 0.0) for h in top]) + 1e-9
            s = np.exp((s - float(np.max(s))) / 0.05)  # sharpen: trust the dense top hits most
            wt = self.dense_weight if ents else 1.0
            seeds[[h.pid for h in top]] += wt * s / s.sum()
        if seeds.sum() == 0:
            return []
        r = self.g.ppr(seeds, damping=self.damping)[: self.g.n_pass]
        if dense_scores is not None:
            r = r + 1e-6 * dense_scores  # deterministic tie-break
        return _topk(r, k)


class GraphExpandRetriever:
    """Dense seeds + 1-hop entity-link expansion (the HotpotQA "bridge" pattern).

    For each of the top ``seeds`` dense hits, every passage whose *title entity* is
    mentioned in the seed (forward link) or that mentions the seed's title (backward link)
    becomes a candidate. Final score = dense(p) + alpha * max over linking seeds of dense(seed).
    Only title-anchored links are used, which are far less noisy than generic co-mentions.
    """

    name = "graph_expand"

    def __init__(self, graph: EntityGraph, dense: DenseRetriever, seeds: int = 3,
                 alpha: float = 0.5, pool: int = 50) -> None:
        self.g, self.dense, self.seeds, self.alpha, self.pool = graph, dense, seeds, alpha, pool
        # backward links: title entity of p -> passages mentioning it
        self.mentioned_by: dict[int, list[int]] = defaultdict(list)
        for pid, ents in enumerate(graph.mentions):
            for e in ents:
                eid = graph.ent_index.get(e)
                if eid is not None and graph.title_of.get(eid) and pid not in graph.title_of[eid]:
                    self.mentioned_by[eid].append(pid)
        self.title_eid = {pid: eid for eid, pids in graph.title_of.items() for pid in pids}

    def retrieve(self, query: str, k: int) -> list[Hit]:
        ds = self.dense.scores(query)
        top = _topk(ds, max(self.pool, k))
        score = {h.pid: h.score for h in top}
        for seed in top[: self.seeds]:
            linked: set[int] = set()
            for e in self.g.mentions[seed.pid]:
                eid = self.g.ent_index.get(e)
                if eid is not None:
                    linked.update(self.g.title_of.get(eid, []))
            te = self.title_eid.get(seed.pid)
            if te is not None:
                linked.update(self.mentioned_by.get(te, []))
            linked.discard(seed.pid)
            for p in linked:
                bonus = float(ds[p]) + self.alpha * seed.score
                score[p] = max(score.get(p, float(ds[p])), bonus)
        best = sorted(score.items(), key=lambda x: -x[1])[:k]
        return [Hit(p, s) for p, s in best]
