# NFR-7: answer quality

**Requirement (PRD).** Answer accuracy at least 80% and citation correctness at least 90% on held-out questions, measured by the rubric grader and an LLM judge, with humans settling disagreements.

**Verdict: citation correctness passes (96.2%). Accuracy fails (70.4%, judge), pending human adjudication.**

## How it was run

Same run as FR-1 (job 1376): 98 answerable held-out questions on the real index, Qwen3-4B on one H200.

```bash
R=/data4/muthangiv/askiisc/results/stack/1376
sbatch slurm/run.sh uv run python -m eval.grader.rubric $R/heldout_results.json /data4/muthangiv/askiisc/state $R/grade
sbatch --export=ALL,MODEL=granite-3.3-8b-instruct slurm/llm_job.sh \
  uv run python -m eval.grader.judge $R/heldout_results.json /data4/muthangiv/askiisc/state $R/grade
```

The judge is Granite-3.3-8B-Instruct: a different model family from the model under test, temperature 0, a strict JSON reply, one retry.

## Results (98 answerable)

| Grader | Accuracy | Citation correctness |
|---|---|---|
| **LLM judge (Granite-3.3-8B)** | **70.4%** (69/98) | **96.2%** of judged answers |
| Rubric, lenient (80% of the fact's words, punctuation ignored) | 44.9% | |
| Rubric, strict (fact verbatim) | 25.5% | 36.6% of citations |

- The judge agrees with the lenient rubric on 51 of 78 answered items (65%), and with the strict rubric on 41%. In 26 of the disagreements the judge accepts a paraphrase that the rubric rejects.
- `grade/disagreements.csv` lists every disagreement with an empty `human_verdict` column for a person to fill in. The accuracy above is the judge's figure before that adjudication.

## Why accuracy is below target

- **Retrieval, not generation, is the main cause.** The question's source document is among the 8 retrieved chunks for only 45 of 98 questions, and 20 answers are refusals because nothing relevant was retrieved.
- **When an answer is given, it is right 88% of the time** (69/78, judge).
- **The corpus has many near-duplicates** (the same rules in notices from different years), which crowd the top 8.

**Next fixes:** hybrid keyword and vector search, collapsing near-duplicate documents, and a reranker over a larger candidate set (`NEXT_STEPS.md`).
