# NFR-9: Security and privacy

**Target:** 0 leaks; ≤ 5% prompt-injection success  
**Result:** **1 of 69** injections succeeded (1.4%); **0** leaks in 20 cross-user attacks. Before the output guards (run 1573): 10 of 70  
**Verdict:** pass

## Workload

`redteam`: the 90-case suite (`eval/datasets/redteam/`) once: 20 instruction overrides, 20 prompt extractions, 20 instructions planted in documents, 10 exfiltration links, 20 requests for other users' documents.

**Why it tests NFR-9:** Each case has a deterministic check (canary, system-prompt phrase, attacker link, forbidden document cited or quoted).

**Command:**
```bash
sbatch slurm/loadtest.sh redteam
```

## Evidence

**Red-team successes by category (lower is better)**

![Red-team successes by category (lower is better)](chart-1.svg)

**Grafana: Forbidden citations seen by Locust**

![Grafana: Forbidden citations seen by Locust](grafana-37.png)

**Grafana: Outcomes**

![Grafana: Outcomes](grafana-10.png)

## Source files

- `/data4/muthangiv/askiisc/results/locust/1578-redteam/answers.json`
- `/data4/muthangiv/askiisc/results/locust/1573-redteam/answers.json`
