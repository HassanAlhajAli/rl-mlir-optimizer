# Adaptation deployment and interruption evidence

`scripts/run_adaptation.py` uses `development-adaptation-v2`. This is still a
development controller for training or validation workloads. It cannot open
held-out performance and does not establish that a learning method wins.

## What is accepted

Each method/budget case starts with an empty measurement cache and a copy of the
same initial policy. A successful tuning result must have the expected program,
reuse count, method, seed, policy identity and exact logical request budget. The
reported counters must agree with the evaluator. The returned schedule must be
the best candidate actually observed within that budget. Duplicate schedules
still consume requests.

Deployment uses the worker's fresh-compilation operation. Its output must match
the independent reference, `cache_hit` must be false, and its integer call count
must equal the requested reuse count. Compile, installation, execution and total
costs must be finite and nonnegative. Their components must sum to the reported
total. The shared check in `rlmlir.measurement.check_fresh_deployment` is also used
by the frozen-selector comparison controller.

The reported adapted cost remains tuning elapsed time plus a fresh deployment's
compile, installation and complete N-call execution time. Model loading, worker
initialization, total case/controller time and hardware counters are separate
fields. The overall time ceiling includes model loading. A deadline check before
deployment prevents starting another compilation after tuning has exhausted it.
An already running worker operation can extend the nominal deadline up to its
configured timeout; the actual elapsed time remains visible.

## Evidence retained on failure

`run.json` is written before model loading and updated between cases. It records
the full seeded method/budget order, pending/completed/failed states, process IDs,
input policy hash and successful summaries. A failed case stops the controller;
it does not silently retry or use a larger request budget.

Each case has:

- `events.jsonl`: context start, request start/completion, update start, group
  completion and context completion/failure. Start events are independent
  snapshots, so subsequent mutation cannot change the recorded attempt.
- `measurements/`: the evaluator's original measurements and compiler artifacts.
- `result.json`: initial status, complete tuning result before deployment, and
  final success or failure with counters. A rejected deployment retains its
  complete response, expected output and failure reason.

JSON-lines events are appended and closed before the next trial. Summary files
are replaced atomically. Nonfinite diagnostic values are represented explicitly
as objects such as `{"invalid_float": "nan"}` so the failure file remains standard
JSON. This protects evidence across ordinary process failures; it is not a
guarantee against disk failure or power loss.

Logging performed during tuning is included in tuning elapsed time. The final
completion event and summary serialization occur outside that timer and remain
within case/controller time. These are research recording costs, disclosed here
rather than mixed silently with older measurements. Version-1 results are not
relabeled as version 2. Compiler transforms, rewards, update equations and trial
budgets are unchanged.

## Verification scope

`tests/test_adaptation_controller.py` uses synthetic workers to check successful
accounting, partial failures, invalid cache/count/timing/output reports, exact
budgets, shared-policy identity, deadlines and early configuration rejection.
`tests/test_adaptation.py` checks actual policy update mechanics and event order,
including preservation of requests before an interrupted trial or update.
These tests are correctness evidence, not a performance experiment.

The main PPO campaign remains stopped at its preserved compiler failure. This
controller improvement does not authorize a restart or transfer old measurements
to the revised compiler binary.
