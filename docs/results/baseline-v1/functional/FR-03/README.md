# FR-3: per-user access decisions and revocation

**Requirement (PRD).** Decide access per user and apply permission changes quickly. Decisions match a reference checker on the test matrix, and 60 seconds after a revocation no query returns the revoked content.

**Verdict: pass.**

## Live (held-out run, job 1376, real index and real users tree)

10 restricted documents were each asked by one user who may read them and one who may not:

| Check | Result |
|---|---|
| Reader who may see the document gets the fact | **10/10** |
| Reader who may not: the document retrieved or cited | **0/10** (no leak) |

The rules covered: department (`cds`, `csa`), path-derived groups (`cds-faculty`, `cds-students`, `cds-staff`, `csa-faculty`, `csa-staff`), plain groups (`cds-phd-committee`, `csa-faculty-search`) and an embargo window.

## Tests (job 1388: 684 tests passed)

| Check | Result | Where |
|---|---|---|
| Engine, independent oracle and both API implementations agree on the 220-case matrix (10 users x 22 documents, including public and deny-wins cases) | 220/220 for each | `tests/unit/test_access_policy.py`, `tests/conformance/test_access_api.py` |
| Deny always wins; unknown users and documents are denied; a corrupted policy fails closed | property tests | same |
| Revocation: no leak 10 s after a revocation (client cache TTL 10 s, target 60 s), probed every second for 60 s | 0 leaks | `tests/unit/test_access_service.py::test_revocation_takes_effect_within_sixty_seconds` |
| Policy changes only by an admin of the owning department, or an institute admin; no takeover of another department's documents | yes | `test_department_admins_manage_only_their_documents` |
| Every department in the users tree has an admin | enforced at load | `test_every_department_needs_an_admin` |

Raw data: `/data4/muthangiv/askiisc/results/stack/1376/grade/graded.json` (p001 to p020), `/data4/muthangiv/askiisc/results/evidence/1388/`.
