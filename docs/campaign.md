# Bounded continuation and campaign allocation

Recorded before the full five-seed training study. The repaired landscape pilot
completed100 correct candidates in328.1s, with6/100 original timing blocks
flagged and1/6 repeats still flagged. It found different estimated reuse choices
on8/20 workloads. The small real PPO run used219 distinct measurements in318s,
about1.45s per new outcome including calibration, checks and learning. These
development figures motivate a larger trial; they do not establish RL benefit.

First run `configs/integration-queue.json`. It connects both512-step PPO
objectives, small fixed/heuristic/random pools, demonstration rechecks,
supervised imitation and measured8/16-request adaptation on one preselected
validation matvec33x210. No held-out program can be measured by these scripts.
Inspect every stage before authorizing the main queue. A negative performance
comparison is acceptable; a crash, incorrect output or budget violation is not.

The main PPO queue uses both objectives and seeds7/17/27/37/47. Uniform limits
are1536 distinct outcomes (including references),8192 logical candidate
requests,16384 requested action steps and45minutes per run. PPO action-step
counts may round to a full rollout; hardware request limits are checked before
another evaluation and cannot round up. The reduced1536 cap, from the proposed
4096 maximum, reserves capacity for pretraining rechecks, baselines, adaptation
and final validation within the initial48-hour measurement allocation.

At the observed rate,10 such runs would take roughly6.2hours, with a7.5-hour
sum of per-run ceilings. This estimate is uncertain because future schedules
can compile much more slowly. Allocate at most8hours to the first unattended
queue; retain at least24hours of the overall campaign for adaptation and final
measurement. Before those stages, use their development cost to freeze their
remaining allocation. Track standalone pilots separately from queue elapsed
time to avoid counting child runs twice. Do not launch additional queues that
would exceed48 cumulative hardware hours without an explicit revised budget.

Commands, from the repository in Ubuntu:

```sh
.venv/bin/python scripts/run_queue.py --config configs/integration-queue.json --output runs/integration-003 --hours 2
# Only after the integration output has been inspected:
.venv/bin/python scripts/run_queue.py --config configs/ppo-campaign.json --output runs/ppo-campaign-001 --hours 8
```

The queue uses a process lock, records commands and logs, preserves completed
stages across restarts, rejects changed code/compiler/corpus, and stops at the
first failed stage. All measured entry points share an exclusive measurement
lease. Never build or run other CPU-heavy checks while a measured stage runs.
Read-only status inspection is fine. A queue deadline first interrupts the
training process so it can save a checkpoint, then terminates its process group
if it does not exit. The120s individual worker timeout and150s checkpoint grace
mean an in-flight operation can extend the nominal deadline; this is reported.

For an interrupted PPO run, `train_ppo.py --resume <previous-run> --output
<new-run>` preserves weights/optimizer and accepted measurements while
subtracting previous requests, distinct outcomes, elapsed time and action steps
from the same total limits. Compiler, corpus, objective, seed and observation
version must match. Partial episodes/rollouts are discarded. This is an explicit
continuation, not a promise of bitwise equivalence with uninterrupted training.
The queue never silently retries a failed stage or enlarges its limits.

Observation version2 counts used structured-op loads, including an accumulator
and excluding unused inputs. Earlier version1 used nominal input/store counts.
Old checkpoints remain reproducible at their recorded source revisions; current
policy loading rejects them rather than silently changing their observation
meaning. This changes features, not the repaired code-generation pipeline.

The supervised cost predictor learns log compilation/setup and execution costs
from the same training pool. Its deterministic legal candidate pool is capped
at256 unique schedules and1024 generation attempts; small action spaces may
yield fewer. Generation and inference are charged. A lower prediction error
does not itself establish better deployment performance.
