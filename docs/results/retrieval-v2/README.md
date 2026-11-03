# Retrieval v2: hybrid search (NFR-7)

**Why:** in baseline-v1 the source document of a held-out question was in the top 8 for only 46% of questions. When it was, the model answered correctly 88% of the time, so retrieval was the main cause of the 70% accuracy.

**What changed** (`retrieval/keyword.py`, `retrieval/dense.py`, `retrieval/search.py`):
- **BM25 keyword index** over the same rows as the vector index. Numpy only, saved in the dense snapshot; uploads are added without a rebuild.
- **Fusion:** the best 20 vector candidates and the best 20 BM25 candidates, both under the same permission and validity masks (so the filter stays inside the search, CLAUDE.md rule 2), fused by reciprocal rank (k = 60), with keyword ranks weighted 2.
- **Query instruction:** questions are embedded with BGE's query instruction ("Represent this sentence for searching relevant passages: "); passages are not.

## Measurements on the v1 corpus (539,406 chunks, 98 answerable held-out questions)

Jobs 1514 and 1518 (`logs/askiisc_rbench_*.out`). doc@8 = source document in the top 8; chunk@8 = the exact source chunk in the top 8; MRR = mean reciprocal rank of the source document.

| Search | doc@8 | chunk@8 | doc MRR |
|---|---|---|---|
| Vector (baseline-v1) | 0.469 | 0.337 | 0.300 |
| Vector with query instruction | 0.459 | 0.367 | 0.318 |
| BM25 only | 0.643 | 0.449 | 0.371 |
| **Hybrid, 20 + 20, keyword weight 2 (chosen)** | **0.633** | **0.459** | **0.406** |
| Hybrid, 50 + 50, keyword weight 3 | 0.653 | 0.449 | 0.385 |
| Hybrid + collapsing near-duplicate notices | 0.582–0.622 | 0.408–0.449 | 0.38–0.41 |
| Hybrid + cross-encoder rerank of the top 16–32 (bge-reranker-base) | 0.561–0.633 | 0.429–0.510 | 0.38–0.42 |

**Not adopted:**
- **Reranking** gave no consistent gain over hybrid search alone on this set. It also costs 2.4 s per query on CPU, or 53 ms on a GPU shared with vLLM.
- **Collapsing near-duplicate notices** removed the copy the question was written from as often as it removed noise.

Search time stays well inside NFR-4: hybrid search adds one BM25 pass, a few milliseconds, to the 18 ms vector search.

## On the v2 corpus (568,817 chunks) and held-out set v2 (150 answerable)

Job 1541, `eval/retrieval_bench.py` (the serving `DenseIndex`), results in `/data4/muthangiv/askiisc/results/retrieval/v2/bench.json`.

| Search | doc@8 | chunk@8 | doc MRR |
|---|---|---|---|
| Vector | 0.480 | 0.367 | 0.318 |
| Vector with query instruction | 0.493 | 0.373 | 0.331 |
| **Hybrid (serving)** | **0.600** | **0.467** | **0.417** |

The gain holds on new questions it was not tuned on: +12 points doc@8, +10 points chunk@8.

**Filtering and speed on the v2 index** (job 1539, `eval/probes/recall.py`): recall@8 1.000 over 10,000 probes as 20 users, 0 forbidden or expired chunks, search p95 19.8 ms.
