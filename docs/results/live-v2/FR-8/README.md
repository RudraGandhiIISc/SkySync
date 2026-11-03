# FR-8: Nine models, one API

**Target:** all nine models; config-only switch  
**Result:** the switch by `ASKIISC_MODEL` is verified; the other eight models are Phase 13 (docs/STATUS.md)  
**Verdict:** pending

## Workload

Not run in this round (Phase 13).

**Why it tests FR-8:** 

**Command:**
```bash
sbatch --export=ALL,ASKIISC_MODEL=<key> slurm/serve.sh
```

## Evidence
