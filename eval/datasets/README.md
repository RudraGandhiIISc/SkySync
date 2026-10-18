# Sample corpus (synthetic)

`sample_corpus/` is a **synthetic** three-tenant corpus for smoke tests and the
M1 end-to-end run. The documents are written for testing and are **not** real
IISc regulations; do not quote them as such. The held-out question set
(README Phase 5.3) must be built from the real corpus and checked by a human;
it does not exist yet.

| File | Readers (sample_policies.json) |
|---|---|
| `inst-attendance.txt` | everyone in the directory (students, faculty, staff) |
| `inst-hostel-fees.txt` | students and staff |
| `cds-thesis.txt` | department `cds` |
| `ece-lab-safety.txt` | department `ece` |
| `staff-travel.txt` | group `staff` |
| `fac-promotion.txt` | role `faculty`; also contains a planted prompt injection |

`sample_directory.json` lists the test users; `sample_questions.json` the
smoke questions with the expected document (or none, for refusals).
