# FR-8: nine models through one API

**Requirement (PRD).** All nine models answer the full test set through the same client; each 4-bit model has at least 2x the KV-cache room of its 8B parent; switching models needs only a config change.

**Verdict: partial (single-model round).** This round is the single-model baseline (Qwen3-4B-Instruct-2507). The nine-model sweep is Phase 13.

| Check | Result | Where |
|---|---|---|
| Nine models registered: three each of 4B, 8B and 8B-AWQ | yes | `src/askiisc/llm/models.yaml`, `tests/unit/test_rag.py::test_registry_has_nine_models_in_three_classes` |
| Switching `ASKIISC_MODEL` changes only the request's `model` field | yes | `test_switching_model_changes_only_the_model_field` |
| A second model served through the same client with no code change | **yes, live:** Granite-3.3-8B-Instruct ran as the NFR-7 judge (job 1387) via `MODEL=granite-3.3-8b-instruct`, through the same OpenAI-compatible API | `slurm/llm_job.sh` |
| All nine answer the full test set | not yet (Phase 13) | |
| 4-bit KV-cache room at least 2x the 8B parent | not yet (needs the AWQ checkpoints) | |
