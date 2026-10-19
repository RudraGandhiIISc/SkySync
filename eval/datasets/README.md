# Datasets

`sample/` is a **mock** copy of the real data layout on `/data4`, with fake
users and mock documents only. Tests and the default stack run use it; real
data never enters git.

| Mock (in git) | Real (on the server only, mode 700) |
|---|---|
| `sample/users/` | `/data4/muthangiv/askiisc/private/users/` |
| `sample/corpus/` | `/data4/muthangiv/askiisc/corpus/` |
| `sample/questions.json` | the held-out question set (not built yet) |

## Layout

```
users/<department>/<students|faculty|staff|admins>.json
    [{"user_id": "...", "groups": [...]}]
    department and role come from the path; each user also gets the
    group "<department>-<file stem>", e.g. "cds-faculty"
corpus/
  public/<institute|cds|csa>/<admissions|academic-policy|course-catalogue|handbooks|department>/<doc_id>.{html,pdf,txt}
      visibility "public": every IISc user in the directory may read it
  restricted/<institute|cds|csa>/<doc_id>.{txt,html,pdf}
  restricted/<institute|cds|csa>/<doc_id>.policy.json
      visibility "restricted": at least one of groups, departments, roles
      or allow_users; optional deny_users, valid_from, valid_until
  manifest.jsonl
      one line per public document: doc_id, path, url, category,
      department, visibility, content_type, etag, last_modified, sha256
```

`slurm/crawl.sh` writes the real `corpus/public/` and `manifest.jsonl`;
`python -m askiisc.ingest.corpus` rebuilds `policies.json` from both folders.

## Mock access rules

| Document | Rule | Who can read it (mock users) |
|---|---|---|
| 7 public documents | public | everyone in `sample/users/` |
| `cds-thesis` | department cds | stu_cds, stu_cds_phd, fac_cds, staff_cds |
| `cds-faculty-meeting-minutes` | group cds-faculty | fac_cds |
| `csa-internal-seminar-calendar` | department csa | the csa users |
| `csa-faculty-search-shortlist` | group csa-faculty-search | fac_csa |
| `staff-travel` | role staff or admin | staff_cds, staff_csa, staff_acc, admin_csa |
| `fac-promotion` | role faculty; contains a planted prompt injection | fac_cds, fac_csa |

`questions.json` has 11 smoke questions with the expected document, or
`null` when the user may not read it or no document answers it.
