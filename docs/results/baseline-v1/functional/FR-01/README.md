# FR-1: cited answers and honest refusals

**Requirement (PRD).** Answer only from retrieved text, cite sources, and say "I don't know" when the text does not support an answer. At least 95% of answers on the test set carry a correct citation, and at least 90% of unanswerable questions get an explicit "I don't know".

**Verdict: refusals pass (90%). Citations pass by the judge, fail by the strict rubric.**

## How it was run

- **Model and run:** Qwen3-4B-Instruct-2507 (BF16, vLLM 0.11, 8,192-token context) on one H200 (GPU 2). Job 1376, 2026-10-19 23:10.
- **Questions:** the held-out set, `eval/datasets/heldout/questions.jsonl`, 168 items. Each is asked once through the real gateway (`ASKIISC_EXPERIMENT=heldout`) by its test user.
- **Index:** the real one, 24,889 documents and 539,351 chunks. The answer cache was off.
- **Graders:**
  - the rubric (`eval/grader/rubric.py`): a cited chunk contains the key fact;
  - the Granite-3.3-8B judge (`eval/grader/judge.py`): the cited text supports the answer.

Rerun: `bash slurm/experiments.sh` (heldout entry), then the rubric and judge commands in `NFR-07/README.md`.

## Results

| Metric | Target | Result |
|---|---|---|
| Unanswerable questions answered with the exact refusal | at least 90% | **90.0%** (27/30) |
| Answers (not refusals) whose citations support them, judge | at least 95% | **96.2%** |
| Answers with at least one citation whose chunk contains the reference fact, strict rubric | at least 95% | 51.3% |
| Answers to answerable questions that were refusals | | 20.4% (20/98) |

The strict rubric undercounts: it needs the reference phrase verbatim in the cited chunk, but many answers cite a sibling document that states the same fact (for example a different year's notice). The judge reads the cited text itself.

Raw data: `/data4/muthangiv/askiisc/results/stack/1376/` (`heldout_results.json`, `grade/graded.json`, `grade/report.json`, `grade/judge.json`, `grade/disagreements.csv`).
