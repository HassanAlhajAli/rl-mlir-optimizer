# Traceable validation selection and campaign coverage

Every PPO objective/seed combination needs its own validation-selected checkpoint.
A trained checkpoint is not automatically the chosen policy for final comparison.
Version 2 of `validate_ppo.py` makes that selection traceable and rejects incomplete
evidence before exporting a winner. Held-out performance remains sealed.

## Selection protocol

The numerical criterion is unchanged. Select up to four evenly spaced saved
behavior checkpoints plus the latest checkpoint, then evaluate each on all 32
canonical validation workloads at reuse counts 1/100/10000. Selection is
deterministic, with the trained objective: runtime-only policies use execution
time; reuse-aware policies include selection, compilation, installation and
estimated repeated execution. Average equally across families. Exact ties follow
the predeclared checkpoint order. These are validation estimates, not final
five-round deployment findings.

Before the first worker starts, the controller requires compatible provenance
and a successfully finished bounded training run. `budget_reached` is a legitimate
completion, while `stopped` is not eligible. The saved plan includes the training
metadata hash and every candidate checkpoint's path and hash. It records process
IDs, startup, a wall limit, counters and partial summaries. Candidate bytes are
checked before loading and again before exporting the winner. Changed training
artifacts stop selection rather than silently changing the experiment.

Each outcome records its checkpoint hash, canonical workload, reuse, selected
schedule, candidate measurement key and reference measurement key. Before export,
an independent aggregation routine checks the complete Cartesian product of
checkpoints, validation workloads and reuse counts, verifies measurement joins,
and recomputes costs and rewards. Missing or duplicate cases cannot produce a
winner. Measurement errors preserve the run and counters with status `stopped`.
The underlying measurement deadline is checked between measurements; the queue's
external process ceiling remains necessary for an in-flight compiler operation.

The exported policy and both evidence logs are hashed. `rlmlir.validation.audit`
checks their identities, exact coverage, aggregates, provenance and request
accounting without executing native code or updating a model. It does not rerun
numerical validation or inspect machine code. The compiler correctness checks and
full saved-measurement audit remain separate requirements. Historical version-1
results remain historical and are not silently relabeled as version 2.

## Ten-run coverage report

The read-only mode below extends the existing campaign auditor:

```sh
.venv/bin/python scripts/audit_ppo_campaign.py runs/ppo-campaign-001 \
  --readiness-only --config configs/ppo-campaign.json \
  --output runs/checkpoint-readiness-001/report.json
```

It requires both objectives and seeds 7/17/27/37/47 exactly once in the campaign
definition, equal declared allocations, and agreement between the config,
resolved queue commands and child metadata. It reports all ten combinations,
including unstarted and failed jobs. It checks cumulative candidate/distinct
limits, source compatibility and whether a separate validation selection exists
for each eligible training run. Supply `--validations mapping.json` to map job
names to validation directories; each supplied selection is audited independently
and must point back to the correct training run.

`all_ten_selections_accounted_for` only means these metadata and selection checks
passed. The lightweight mode does not load training models or prove optimizer
updates. The existing full campaign auditor checks saved trajectories, numerical
evidence and changed policy tensors. Neither report freezes the entire study or
enables held-out access: the other baselines, adaptation controls, compiler
continuation, data allocation and statistical protocol still need their evidence.
The output always says `whole_study_frozen=false` and `heldout_access=false`.

The current campaign is stopped. No validation runs are started by this command,
and a negative coverage result is expected. A historical run from the previous
compiler remains visible but is ineligible for the current compiler's final study.

## Tests and current limits

The tests use synthetic checkpoints and deterministic fake measurements to run
the complete two-checkpoint, 192-case selection/export/audit path. They reject
missing/duplicate cases, wrong costs, changed checkpoint bytes, wrong workload
joins, modified exports/counters and failed collection. Coverage tests retain all
ten jobs, reject missing/duplicate seeds and unequal budgets, and prevent one
run's validation directory from being substituted for another's.

No current-backend policy has been selected by a real validation campaign yet.
These implementation checks are not evidence of policy quality or RL superiority.

The targeted 20-check suite passed in 10.63 seconds. After the final changes, the
complete Python suite passed 116 checks in 21.42 seconds. CI includes the new
validation tests. These are correctness checks with synthetic data, not a native
validation campaign.

The [saved campaign coverage audit](../results/checkpoint-readiness-001/observations.md)
confirms three completed, one failed and six unstarted jobs. All started jobs
use the old compiler, and all ten separate validation selections are missing.
