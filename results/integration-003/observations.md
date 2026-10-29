# Development integration: PPO, imitation and adaptation

All six stages completed in 1,053.55 seconds on the pinned Intel/WSL setup.
This is a pipeline integration result, not the main performance comparison.
No held-out workload was measured. Source revision:
`5e8abb01c5fdfa908007525c7c11fb3cecc5e432`.

| Stage | Executed evidence |
|---|---|
| Reuse-aware PPO | 512 action steps, 152 episodes, 224 distinct outcomes, 20 optimizer epochs |
| Execution-only PPO | 512 action steps, 145 episodes, 219 distinct outcomes, 20 optimizer epochs |
| Mixed experience | 72 training and 36 validation trajectories from fixed, heuristic and random choices |
| Pretraining | 255 recheck requests, 220 distinct outcomes; 143 training and 12 validation demonstration contexts |
| Adaptation | Five methods, budgets 8 and 16, one validation matvec of shape 33 by 210, reuse 10,000 |

Both PPO checkpoints changed from their common initial weights. Imitation used
training examples for actor fitting and validation for early stopping. Its
critic used training trajectories only. The source trajectory hashes,
demonstration file and policy hash were checked. One selected demonstration
was near-tied; three had unresolved timing drift. These are retained and must
not be interpreted as confidently optimal labels.

Across the full queue, all 787 measurement records passed independent numerical
checks. There were 52 flagged original timing blocks; 14 remained flagged after
the prescribed repeat. The audit retains these observations. Correctness and
clean bookkeeping do not remove timing uncertainty.

## Adaptation observation

The following are individual development measurements, not repeated final
estimates or statistically significant method rankings.

| Method | Trials | Distinct schedules | Updates | Tuning, ms | Fresh deployment, ms | Total, ms |
|---|---:|---:|---:|---:|---:|---:|
| Frozen sampling | 8 | 4 | 0 | 3,467.33 | 35.36 | 3,502.69 |
| GRPO | 8 | 4 | 0 | 3,498.63 | 35.65 | 3,534.28 |
| PPO adaptation | 8 | 4 | 0 | 3,455.27 | 37.09 | 3,492.36 |
| Random search | 8 | 8 | 0 | 6,976.59 | 36.29 | 7,012.88 |
| Mutation search | 8 | 4 | 0 | 3,509.41 | 35.33 | 3,544.73 |
| Frozen sampling | 16 | 7 | 0 | 6,063.79 | 28.82 | 6,092.60 |
| GRPO | 16 | 7 | 1 | 6,308.91 | 29.22 | 6,338.13 |
| PPO adaptation | 16 | 7 | 1 | 6,079.36 | 29.58 | 6,108.95 |
| Random search | 16 | 15 | 0 | 13,037.62 | 35.13 | 13,072.75 |
| Mutation search | 16 | 8 | 0 | 6,983.90 | 28.55 | 7,012.46 |

Frozen sampling, GRPO and PPO sampled identical action sequences at each budget
with the fixed seed. At 16 trials, the updates had finite nonzero gradients and
changed probabilities, but did not change the sampled choices. Their winner was
tiling the reduction dimension by four. At eight trials, every method selected
the unchanged schedule. Random search also selected that schedule at 16 trials.

The extra trials exposed a useful candidate here; learning from the first group
did not demonstrate an additional benefit. Tuning cost seconds, while compiling
and executing the selected code 10,000 times cost about 28 to 37 ms. These costs
show why final code speed alone is an inadequate adaptation metric. They do not
establish a general break-even reuse count or a universal method ranking.

## Counter correction and reset verification

Audit found that historical `worker_counts` were lifetime totals for the shared
worker. Later adaptation entries included earlier methods' operations and
deployments. Do not sum those totals or treat them as per-method work. Logical
trial counts, per-record hardware deltas and elapsed timing boundaries were
separate and correct. Original evidence is preserved without rewriting it.

Commit `460ef02` scopes counters to evaluator creation, records initialization
and fresh-deployment operations separately, and adds policy fingerprints plus a
guard against mutation of the initial policy. Six new comparisons verified the
repair in [adaptation-accounting-001](../adaptation-accounting-001/observations.md).
The relevant Python suite passed all 16 tests. No compiler pipeline, observation,
reward, numerical tolerance or search budget changed.

## Reproduction and evidence

The original queue command, from the repository in Ubuntu:

```sh
.venv/bin/python scripts/run_queue.py --config configs/integration-queue.json --output runs/integration-003 --hours 2
```

Use the recorded source revision to reproduce the original measurement. To
inspect the archived output without running kernels, extract `raw.zip` into
`runs/integration-003`, then run the current auditor:

```sh
.venv/bin/python scripts/audit_integration.py runs/integration-003 --output runs/integration-003-audit.json
```

- [Audit summary](audit.json): costs/rewards, exact budgets, provenance, legal
  actions, termination, checkpoint changes, first-group controls and IR hashes.
- [Raw archive](raw.zip): 841 original files, including queue commands/logs,
  stage metadata, all measurements and compressed IR, complete trajectories,
  demonstration provenance, policies, update metrics and fresh deployment costs.
- [Archive and file hashes](manifest.json): SHA-256 integrity manifest.

The main ten-run PPO campaign may proceed because integration is correct and
bounded, not because this pilot demonstrated a learning advantage. Validation
checkpoint selection remains separate; final held-out evaluation remains sealed.
