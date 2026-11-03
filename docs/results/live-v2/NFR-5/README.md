# NFR-5: Availability

**Target:** ≥ 99% success over a 4-hour soak; restore ≤ 60 s  
**Result:** not run  
**Verdict:** pending

## Workload

`soak`: half load for 4 hours with a worker killed; not run in this round.

**Why it tests NFR-5:** 

**Command:**
```bash
sbatch slurm/loadtest.sh soak  # inside a 5 h serving job
```

## Evidence
