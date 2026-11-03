# FR-5: document versions and stale content

**Requirement (PRD).** Keep every document version and stop using outdated chunks. As-of-date queries are correct at all 60 revision checkpoints; superseded chunks leave search within 15 minutes; no answer uses them without a warning flag.

**Verdict: pass in simulation (60/60 checkpoints). Live: 18/20 as-of answers correct, and the stale flag right in 19/20.**

## Live run on the real index

**Setup.**
- 10 test circulars (`corpus/restricted/institute/circ-*`) were ingested as v1, then changed (one fact each) and ingested again.
- The reconciler detected the content change and recorded v2. Both versions are kept: v1 got `valid_to`, and nothing was deleted. See jobs 1369 (v1) and 1374 (v2).
- Each circular was then asked twice through the gateway (job 1376), as of v1's start time and as of v2's start time (the `as_of` field in `/v1/ask`).

| Check | Target | Result |
|---|---|---|
| As-of answer gives that version's fact | all | **18/20** |
| v1 answers carry the stale flag; v2 answers do not | all | **19/20** |
| Versions kept (v1 and v2 rows in `versions.db`) | all | 10/10 documents |

The 2 misses:
- **r001** is correct but scored wrong: it answered "10:00 PM" for the reference "10 pm".
- **r003** is a retrieval miss: a fee-structure page outranked the gym circular, and the answer came from that page.

## Unit and simulation evidence

`tests/unit/test_versions_reconciler.py`:
- **60 checkpoints:** 60 revisions across 20 documents, both with ETags and with content hashes; the as-of version is right at all 60 checkpoints.
- **Atomic switch:** at every instant exactly one version is searchable, never both, never neither.
- **Retired chunks** never appear in current results.
- **Crash safety:** a failed ingest leaves the document unrecorded, so it is retried.

**Not measured:** the 15-minute demotion bound in a running system, because the 5-minute reconciler loop is not yet inside the stack (see NFR-8).

Raw data: `/data4/muthangiv/askiisc/results/stack/1376/heldout_results.json` (ids r001 to r020), `grade/graded.json`, `/data4/muthangiv/askiisc/results/revisions/`.
