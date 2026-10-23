# hopscope

[![CI](https://github.com/sushmit-gujar/hopscope/actions/workflows/ci.yml/badge.svg)](https://github.com/sushmit-gujar/hopscope/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![Python 3.10–3.13](https://img.shields.io/badge/python-3.10–3.13-blue)
**[Live results explorer →](https://sushmit-gujar.github.io/hopscope/)**

**When does knowledge-graph retrieval (GraphRAG) actually beat vector RAG?** hopscope is a
reproducible, key-free benchmark harness that answers this on multi-hop questions. It measures
evidence recall, whether the *complete* reasoning chain was retrieved, answer accuracy, answer
attribution to gold evidence, and latency. It also reports paired bootstrap confidence intervals,
so "graph is better" is a claim with error bars.

Everything runs locally on CPU: ONNX embeddings, an ONNX extractive reader, and an LLM-free
knowledge-graph build. A hosted LLM reader (OpenAI / Anthropic) is an optional plug-in.

## Why

Teams adopt GraphRAG because "it's better for multi-hop", then can't tell whether it helped
*their* questions or just added latency and an indexing pipeline. Multi-hop questions fail when
retrieval finds one hop but not the bridge. Top-k recall hides that, so hopscope reports
**Full@k**, the share of questions where **every** gold passage is in the top-k, and splits
results by question type.

## Headline results

Setup: HotpotQA distractor validation set. The questions are shuffled once (seed 13).
**Test = questions 0–999.** Their 10 paragraphs each are pooled and de-duplicated into one corpus
of **9,771 passages**, so each question competes against every other question's hard
distractors. Graph hyper-parameters were tuned **only** on a disjoint **dev slice (questions
1000–1299)**. Embeddings: `BAAI/bge-small-en-v1.5` (fastembed ONNX). Reader:
`distilbert-base-cased-distilled-squad` (int8 ONNX), top-5 passages. Hardware: 8-vCPU Linux box,
CPU only, while other jobs were running, so absolute latencies are pessimistic.

| Method | R@2 | R@5 | **Full@5** | Full@10 | MRR | EM | F1 | Attrib. | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.560 | 0.717 | 0.466 | 0.703 | 0.866 | 0.203 | 0.303 | 0.529 | 0.5 | 0.6 |
| `dense` (vector RAG) | 0.710 | 0.844 | 0.699 | 0.808 | 0.933 | 0.232 | 0.349 | 0.625 | 25.6 | 45.0 |
| `hybrid` (BM25+dense RRF) | 0.660 | 0.844 | 0.696 | 0.845 | 0.926 | 0.243 | 0.350 | 0.589 | 27.0 | 46.2 |
| `graph_ppr` (entity KG only) | 0.344 | 0.546 | 0.265 | 0.527 | 0.592 | 0.141 | 0.223 | 0.524 | 18.5 | 32.0 |
| **`graph_ppr+dense`** (HippoRAG-style) | **0.718** | **0.867** | **0.746** | 0.844 | **0.937** | 0.232 | 0.350 | **0.630** | 60.9 | 80.8 |
| `graph_expand` (dense + 1-hop links) | 0.634 | 0.826 | 0.681 | **0.874** | 0.863 | **0.252** | **0.362** | 0.618 | 25.3 | 43.9 |
| `oracle` (gold passages → reader) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.376 | 0.540 | 1.000 | – | – |

R@k = share of gold supporting passages in the top-k. Full@k = all gold passages in the top-k.
EM/F1 use official HotpotQA normalization over the 922 span-answer questions; the 78 yes/no
questions are excluded because an extractive reader can't produce "yes"/"no". Attrib. = share of
answers extracted from a gold passage. Latency = retrieval only.

**Paired difference vs `dense` on Full@5** (bootstrap, 2,000 resamples, 95% CI):

| Method | All (n=1000) | Bridge (n=773) | Comparison (n=227) |
|---|---|---|---|
| `graph_ppr+dense` | **+4.7 pts [+2.9, +6.5]** | **+5.8 [+3.6, +8.3]** | +0.9 [−0.5, +2.6] |
| `graph_expand` | −1.8 [−5.0, +1.4] | **+7.8 [+4.5, +11.0]** | **−34.4 [−40.5, −28.2]** |
| `hybrid` | −0.3 [−2.8, +2.3] | +0.9 [−2.3, +4.0] | −4.4 [−7.9, −1.3] |

### What the numbers say

1. **A graph helps only as a complement to vectors.** The LLM-free entity graph on its own
   (`graph_ppr`) loses 43 points of Full@5 to dense retrieval. Seeding Personalized PageRank with
   dense hits *and* query entities (`graph_ppr+dense`) is the best retriever. It recovers the full
   evidence chain on 4.7% more questions (CI excludes 0), and the whole gain comes from **bridge**
   questions.
2. **Graph traversal is query-type dependent.** One-hop link expansion is the best method for
   bridge questions (+7.8 pts) but destroys comparison questions (−34 pts). Comparison questions
   name both entities and need no hop, so expansion crowds out the second entity's page. **A
   query-type router is worth more than any single retriever.** This is the next experiment.
3. **The graph costs latency.** `graph_ppr+dense` is 2.4× dense at p50 (61 vs 26 ms), because PPR
   runs over 86k nodes in numpy. Graph construction is cheap: 0.9 s for 76k entities and 119k
   edges with regex extraction.
4. **Better retrieval ≠ better answers here.** The local extractive reader caps out at F1 0.54
   even with gold passages (`oracle`), so retrieval gains translate into only about 1 point of F1.
   That's the case for measuring retrieval and generation separately. Swap in an LLM reader
   (below) to see how much of the gap is the reader.
5. **BM25+dense fusion didn't help** on this corpus: Full@5 −0.3 [−2.8, +2.3]. Hybrid isn't a free
   win either.

Dev-slice tuning (`results/tune-dev.json`): dense Full@5 0.800; the best `graph_ppr+dense`
(damping 0.7, dense weight 0.75) scored 0.817, and the best `graph_expand` (5 seeds, α 0.25)
scored 0.780. Those parameters were frozen before the test run.

## Architecture

```mermaid
flowchart LR
    D[(HotpotQA<br/>distractor dev)] --> C[Pooled corpus<br/>9,771 passages]
    C --> B[BM25<br/>sparse matrix]
    C --> E[Dense index<br/>bge-small ONNX]
    C --> G[Entity graph<br/>passage↔entity, no LLM]
    Q[Question] --> B & E
    Q --> X[Query entities<br/>n-gram match]
    E --> H[Hybrid RRF]
    B --> H
    X --> P[Personalized PageRank]
    E -- dense seeds --> P
    G --> P
    E --> L[1-hop title-link<br/>expansion]
    G --> L
    B & E & H & P & L --> R[Top-5 passages] --> RD[Reader<br/>DistilBERT ONNX · or LLM]
    RD --> M[Metrics<br/>R@k · Full@k · EM/F1 · attribution<br/>latency · paired bootstrap]
```

* **Entity graph without an LLM.** Capitalized spans plus Wikipedia titles become entity nodes.
  Passages link to the entities they mention (weight 1) and to their own title entity (weight 2).
  Two passages are two hops apart iff they share an entity, which is the HotpotQA "bridge"
  structure. Very common entities (df > max(20, 1% of corpus)) are dropped as non-informative
  bridges.
* **`graph_ppr+dense`** follows HippoRAG (2): the PPR personalization vector mixes query-entity
  seeds, weighted by specificity 1/df, with sharpened dense passage scores. Rank = PPR mass on
  passage nodes.
* **`graph_expand`**: the top-s dense passages pull in passages connected by forward links (a
  passage about an entity the seed mentions) or backward links (passages mentioning the seed's
  title). Score = dense(p) + α·dense(seed).

## Quickstart

```bash
pip install "hopscope[local] @ git+https://github.com/sushmit-gujar/hopscope"
hopscope run -n 1000 --out results/hotpotqa-1000    # downloads HotpotQA + ONNX models (~200 MB)
hopscope tune -n 300 --offset 1000                   # dev-slice grid search
python scripts/make_demo_data.py results/hotpotqa-1000   # refresh the Pages explorer
```

Faster smoke run, with no model downloads and no network if you pass `--data`:

```bash
hopscope run --data tests/fixtures/hotpot_mini.json -n 30 --encoder hashing --reader lexical
```

Optional hosted-LLM reader (answers + citations → attribution is computed from the citations):

```bash
export HOPSCOPE_LLM_PROVIDER=openai OPENAI_API_KEY=...        # or anthropic + ANTHROPIC_API_KEY
export HOPSCOPE_LLM_MODEL=gpt-4o-mini                          # optional
hopscope run -n 200 --reader llm
```

The LLM reader is unit-tested with mocked HTTP. No numbers above use it.

## Design decisions and trade-offs

* **Full@k as the headline metric.** Multi-hop QA needs every hop. Recall@k rewards finding half a chain.
* **Pooled corpus instead of per-question 10-paragraph contexts.** The per-question setting is
  nearly solved by any retriever. Pooling ~10k passages creates realistic hard negatives.
* **Disjoint dev/test slices and paired bootstrap CIs.** Graph methods have knobs (damping, seed
  weights). Tuning them on the test set would inflate exactly the comparison this tool exists to
  make.
* **Local-first models.** bge-small and a quantized DistilBERT reader keep the whole benchmark
  free, deterministic and CI-friendly. The trade-off is a weak reader (oracle F1 0.54). The
  pluggable LLM reader exists to close that gap when a key is available.
* **Regex entity extraction instead of LLM OpenIE.** It's zero-cost and reproducible, and it
  shows how far a cheap KG gets you. The pure-graph row shows its limits. LLM- or NER-based
  extraction is the obvious next ablation.
* **Known limitations:** HotpotQA only so far (MuSiQue and 2WikiMultiHopQA are next). 1,000 test
  questions. Latencies measured on a shared, loaded machine.

## Tests

```bash
pip install -e ".[dev]" && pytest -q && ruff check . && mypy src
```

The tests cover metrics, HotpotQA normalization, BM25/dense/hybrid ranking, a toy 2-hop world
where PPR must reach the second hop, PPR as a probability distribution, the end-to-end
benchmark, the offline CLI, and OpenAI/Anthropic readers via mocked transports. CI runs Python
3.10–3.13 plus an offline smoke benchmark.

Data: HotpotQA (Yang et al., 2018), CC BY-SA 4.0. References:
[HippoRAG](https://arxiv.org/abs/2405.14831) · [RAG vs GraphRAG: A Systematic Evaluation](https://arxiv.org/abs/2502.11371).

MIT © 2026 Sushmit Gujar
