# Interrupted main PPO campaign

This attempt used compiler revision `6a436e9` and stopped on a compiler crash.
It is preserved as an incomplete study, not a performance comparison with the
subsequently repaired backend. Queue elapsed time was 6,372.59 seconds.

| Objective | Seed | State | Candidate requests | Distinct outcomes | Optimizer epochs |
|---|---:|---|---:|---:|---:|
| Reuse | 7 | Request limit reached | 8,192 | 1,349 | 460 |
| Runtime | 7 | Distinct-outcome limit reached | 1,759 | 1,536 | 210 |
| Reuse | 17 | Request limit reached | 8,192 | 1,302 | 460 |
| Runtime | 17 | Compiler SIGSEGV | 831 | 833, including one failure | 100 |
| Reuse | 27 | Not started | 0 | 0 | 0 |
| Runtime | 27 | Not started | 0 | 0 | 0 |
| Reuse | 37 | Not started | 0 | 0 | 0 |
| Runtime | 37 | Not started | 0 | 0 | 0 |
| Reuse | 47 | Not started | 0 | 0 | 0 |
| Runtime | 47 | Not started | 0 | 0 | 0 |

Distinct outcomes include reference measurements. A failed new measurement
still consumes its slot. Candidate requests also count cache hits, so these
columns need not match. Different stopping conditions are retained, not
equalized after observing the results.

The read-only [audit](audit.json) checked all 5,020 records. All 5,019 accepted
records have successful numerical checks and matching saved IR hashes. All
18,972 completed trajectories have legal recorded actions, accepted candidate
and reference outcomes, and recomputed costs/rewards matching the records.
All four final policies have finite, changed weights and nonzero optimizer
updates. Each run respects its request and distinct-outcome limits. The failed
candidate did not become a training reward. The audit retains 527 originally
flagged timing blocks and 127 unresolved drift flags.

These checks establish evidence consistency and actual training, not better
optimization. No validation-selected checkpoint or held-out result exists for
this campaign. The failure diagnosis and limitations are in the
[compiler repair report](../../docs/compiler-repair-002.md).

## Reproduce the evidence audit

The four ZIP files preserve every measurement record and IR artifact, all
completed trajectories, the initial/latest policy of each attempted run, run
metadata and logs. [manifest.json](manifest.json) hashes every packaged file.
Intermediate `behavior-*.zip` checkpoints remain in the original local run
directories and are not duplicated in this snapshot.

Extract all four ZIP files into one empty directory and copy [queue.json](queue.json)
there. With the documented Python dependencies installed:

```sh
.venv/bin/python scripts/audit_ppo_campaign.py /path/to/extracted-evidence \
  --output /path/to/audit.json
```

To inspect the exact original implementation, check out `6a436e9` separately.
Do not execute the old compiler on arbitrary schedules without resource limits:
the preserved diagnostic demonstrates excessive memory consumption.

The original local source is `runs/ppo-campaign-001`. The complete ten-job
definition remains in queue.json. No failed stage was silently retried and no
budget was reset. All costs remain charged to the cumulative 48-hour allowance.
