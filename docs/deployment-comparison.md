# Complete deployment comparison controller

`scripts/compare_deployments.py` compares frozen schedule selectors on canonical
training or validation workloads. It does not permit held-out performance access
or perform learning updates. Final method freezing, the full baseline registry,
adaptation orchestration and final statistical analysis remain unfinished.

The controller accepts the fixed reference and locality heuristic, plus named
compatible actor checkpoints, an optional compatible cost predictor and an
optional [training-selected fixed schedule](fixed-baseline.md). Model
load time is recorded separately. Frozen selectors receive an analysis-only
worker interface: they cannot make compilation or performance queries while
choosing their schedule. The caller measures the entire selector callback,
including heuristic/predictor candidate generation and analysis overhead.

## Measurement boundaries

Each of five rounds visits the complete workload/method/reuse grid in seeded,
shuffled order. Each case selects a schedule, checks independently generated
inputs at seeds7 and17, and records inspectable generated IR. Calibration targets
120ms for the candidate and reference; measured steady batches must reach100ms.
The complete deployment then performs a new compilation, installation/buffer
setup and exactly N calls through the existing native `deploy` command.

The charged selection-plus-deployment total is the outer callback time plus
the native deployment total. Compilation, installation and execution must sum
to that native total; the response must report a fresh compilation and exactly
N calls. A separate first-call observation and steady-state batches are saved.
Research validation, calibration, reference checks, transport and input/reference
generation are excluded from that deployment objective but remain in complete
case elapsed time and hardware counters. Service initialization and model loading
are separately recorded. Warm code-cache deployment is not measured by this
controller yet.

Each deployment/steady block is surrounded by reference batches. Drift above5%
or a steady/reference batch below100ms flags the block. Repeat once with the
same selected schedule, retaining both original and repeat. The selector is
called again in each independent round, not again for a drift repeat. No
favorable-only timing summary is produced.

Wrong output, nonfinite/bad timing, wrong call counts, unexpected warm deployment,
worker failure and handle cleanup failure stop the run. Starts and final outcomes
are appended separately; failures preserve partial attempts and wrong outputs.
The controller never retries a failed run or automatically resumes its output
directory. A drift repeat is the sole within-case retry and is visible.

The wall deadline is checked before each worker request, including selector
analysis. An in-flight request may run to the existing120s worker timeout; handle
cleanup still runs after the deadline. The3minute integration ceiling below is
a nominal controller bound with that documented in-flight allowance, not a
promise of immediate process preemption.

## Bounded integration protocol, recorded before running

`configs/comparison-development.json` uses only the already-selected validation
matvec33x210 (`b492eae50d11c001`), reference and heuristic selectors, reuse counts
1/100/10000, five rounds and order seed971. It contains30 planned cases, with
at most one flagged-block repeat per case. It introduces no PPO training or
adaptation request allocation, and consumes no held-out performance.

First verify the controller using simulated responses, including deliberately
wrong output, invalid timings, drift, deadline and cleanup failure. Then use a
fresh `runs/comparison-controller-integration-001` for one native integration:

```sh
.venv/bin/python scripts/compare_deployments.py \
  --config configs/comparison-development.json --check-only
.venv/bin/python scripts/compare_deployments.py \
  --config configs/comparison-development.json \
  --output runs/comparison-controller-integration-001 --hours 0.05
```

This is a new controller integration check within the existing48hour allowance;
its actual elapsed time will be charged separately. It does not restart the
interrupted PPO queue, replenish any seed's request budget, or establish that
the original compiler crash is resolved. It stops at the first failure. Do not
edit/build/test concurrently with this measured stage. This one-program check
cannot establish learned-policy performance or generalization.

The [completed integration](../results/comparison-controller-integration-001/observations.md)
records30correct cases,31fresh deployments and one retained unresolved timing
flag. Use a new directory for any explicitly justified future run; do not overwrite
or automatically repeat this evidence. `scripts/audit_comparison.py` checks saved
coverage, hashes, costs, drift handling and hardware counters without execution.
