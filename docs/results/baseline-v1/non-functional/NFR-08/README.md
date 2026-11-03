# NFR-8: freshness

**Requirement (PRD).** At least 99% of results use the current document version within 15 minutes of a change. Workload: 50 revised documents per hour, checked every 60 s.

**Status: partly measured.** The long run at 50 revisions per hour (E5) was deferred (agreed on 2026-10-19).

## What was measured

10 test circulars were revised (v1 to v2) on the real index, using the same reconciler and ingest path a live update would use:

| Check | Result |
|---|---|
| Reconciler detects the change (content hash) and records version 2 | 10/10 (`results/revisions/1374/v2/ingest_report.json`) |
| Answers as of version 2 give the new fact | see `../../functional/FR-05/` |
| Time from change to searchable | the time of one ingest pass. Changed documents are visible as soon as their chunks are written, because the commit delay is 0 in these jobs |

**Not built yet:** the 5-minute reconciler loop inside the running stack. Today, re-ingestion is a separate job (`slurm/ingest.sh`), so in a live run freshness is bounded by how often that job runs.
