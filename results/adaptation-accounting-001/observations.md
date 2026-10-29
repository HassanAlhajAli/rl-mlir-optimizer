# Per-method accounting and reset verification

Follow-up at source `460ef02`, using the unchanged integration-003 pretrained
policy and compiler. This checks the reporting repair, not a new performance
claim. Original integration-003 files were not altered.

```sh
.venv/bin/python scripts/run_adaptation.py --policy runs/integration-003/pretrain/policy.pt --output runs/adaptation-accounting-001 --workload-id b492eae50d11c001 --reuse 10000 --budgets 8 16 --methods frozen grpo ppo --hours 0.1
```

All six comparisons completed, with 33 distinct accepted numerical measurements
and six correct fresh deployments. Each context began with the same initial
policy fingerprint. Frozen policies and all eight-trial policies remained
unchanged. GRPO and PPO each changed their policy at 16 trials, with one update
between groups and no extra evaluations. The shared initial policy stayed
unchanged.

Each method's tuning counters exclude earlier methods and deployments. Each
separate deployment counter contains one fresh compilation and 10,000 actual
invocations. Candidate requests are exactly eight or 16, including duplicates.

The verification read every result, checked the conditions above and preserved
the numerical outputs, counts, update metrics and identities in [raw.zip](raw.zip).
See the [audit](audit.json) and [archive/file hashes](manifest.json). The observed
schedule choices agree with the original small integration example. This check
does not replace the planned multi-program, multi-seed comparison.
