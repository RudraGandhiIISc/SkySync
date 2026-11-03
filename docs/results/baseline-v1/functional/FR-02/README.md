# FR-2: permission- and version-filtered retrieval

**Requirement (PRD).** Search only chunks the user may read and that are current. In 10,000 access probes zero forbidden chunks are returned, and recall@8 is at least 0.95 against an exact filtered search.

**Verdict: pass for leaks (0 forbidden chunks). Recall is exact on this server (Milvus Lite indexes as FLAT).**

## Real index (job 1403, 2026-10-20 01:16)

- **Index and users:** the real index (539,351 chunks), real policies (24,911 documents) and the real users tree.
- **Probes** (`eval/probes/real.py`): 2,000. Each picks a random user, takes their permission set from the policy engine, puts it inside the Milvus search expression, and runs a top-8 search with a random query vector (leaks do not depend on the query).
- **Checking:** every returned chunk is replayed through the independent oracle (`eval/oracle/oracle.py`).
- **Why 2,000, not 10,000:** at about 0.85 s per search on Milvus Lite, 10,000 probes would not fit in one job. The 10,000-probe test runs on the 2,000-chunk test index below.

| Check | Result |
|---|---|
| Probes / chunks returned | 2,000 / 16,000 |
| **Forbidden chunks returned (oracle)** | **0** |
| Search time p50 / p95 / p99 | 844 / 889 / 940 ms (see NFR-1 and NFR-4) |

## Test index (`tests/integration/test_fr2_probes.py`, job 1388)

| Check | Result |
|---|---|
| 10,000 probes, 200 users, 100 documents with groups, departments, deny lists and embargoes | **0 forbidden chunks** |
| recall@8 against an exact filtered search | at least 0.95 (passes; Milvus Lite is FLAT, so recall is exact) |

**Not measured yet:** HNSW recall, which needs Milvus standalone.

Raw data: `/data4/muthangiv/askiisc/results/probes/1403//probes.json`.
