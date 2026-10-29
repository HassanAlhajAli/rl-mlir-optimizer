# Training-selected fixed schedule

The learned selectors need a comparison with one schedule chosen using training
data and then applied unchanged to every later program and reuse hint. This is
different from choosing the best measured schedule separately for each test
program, which would give the baseline additional test-time search.

The implementation is available. **No current-backend grid has been measured and
no trained fixed-schedule artifact has been produced.** The stopped PPO campaign
and its compiler continuation gate are unchanged. Old compiler measurements
cannot be imported into a current artifact.

## Predeclared candidates and objective

`configs/fixed-training-grid.json` specifies all 128 canonical training workloads,
reuse counts 1/100/10000, order seed 627 and 16 literal schedules:

- The untransformed reference.
- Tile dimension 0 by 4, 8, 16, 32 or 64.
- Tile dimension 1 by those same five sizes.
- Tile dimensions 0 and 1 by the same size, using those five sizes.

These dimensions exist in the supported families. The compiler still checks
legality and correctness for each case. No per-family fallback or shape-dependent
replacement is allowed. These candidates are a restricted, shared action subset,
not every possible fixed schedule. They do not include vectorization: a literal
vectorization sequence legal across different loop counts needs additional
constraints. The separate locality/vectorization heuristic and cost predictor
remain required comparisons. Do not describe this baseline as globally optimal
or use beating it alone to claim RL superiority.

Collect every workload/schedule pair exactly once, giving 2,048 planned candidate
requests and distinct measurements. The reference counts as a candidate in this
grid. Order is shuffled across the complete grid. Existing paired measurement
rules check two input seeds and retain timing flags and their bounded repeats.
One C/T observation supports all reuse estimates; repeating the same cached
observation is not a new independent sample.

For each candidate, compute log(reference cost / candidate cost) at each reuse
count. Average across reuse counts within each program, programs within each
family, then families equally. Cost includes the measured schedule-copy overhead,
compilation, installation and N times steady execution. Exact score ties follow
the declared candidate order, with the reference first. Timing flags remain in
selection and are reported, never filtered to improve the score. This training
proxy does not replace fresh complete-deployment evaluation.

The candidate list and selection rule are recorded before this grid is measured.
Earlier development and interrupted PPO evidence has already been inspected;
this baseline is part of ongoing development, not a retroactive preregistration
of the entire project. Held-out performance remains sealed.

## Evidence and failure boundaries

`collect_fixed_grid.py` saves the configuration, seeded order, process IDs, raw
measurements, IR, started/completed trial pairs, elapsed time and request/hardware
counters. It uses the exclusive measurement lease and requires a clean tracked
tree. Limits are exactly the declared grid size, with a supplied wall ceiling.
The wall limit is checked between measurements; an in-flight measurement contains
multiple worker calls, each with the existing 120-second timeout, and can extend
the nominal ceiling. The eventual campaign must also impose its external process
deadline. An error preserves the incomplete run and stops; there is no auto-retry,
automatic budget enlargement or partial-grid winner.

`fit_fixed_baseline.py` executes no kernels. It requires a complete collection
with matching compiler/corpus/protocol identities and an exact one-to-one join
between trials and measurements. It verifies source and IR hashes, correctness
flags, input seeds, raw timing summaries, drift handling and exact request counts.
Every candidate must cover every declared training program. Validation IDs,
changed workload identities, failed results, missing cells and duplicates are
rejected. The artifact contains all cost cells, the leaderboard and source hashes.
Reduced synthetic/development grids are explicitly marked as incomplete training
coverage; the final study must require full coverage.

`compare_deployments.py` accepts an optional `best_fixed` artifact path. Loading
checks compatibility and recomputes the selection from its saved grid. Selection
then simply copies the stored schedule, ignoring program identity and reuse hint.
The comparison records the artifact hash and charges its actual callback time.
The saved-evidence auditor rejects a fixed schedule that changes across cases.
Older comparison configurations remain valid when this optional field is absent.

## Commands and current gate

The following inspection starts no worker and grants no campaign allocation:

```sh
.venv/bin/python scripts/collect_fixed_grid.py \
  --config configs/fixed-training-grid.json --check-only
.venv/bin/python -m pytest tests/test_fixed_baseline.py tests/test_comparison.py -q
```

After the compiler continuation and cumulative allocation are explicitly revised,
the collection interface is:

```sh
.venv/bin/python scripts/collect_fixed_grid.py \
  --config configs/fixed-training-grid.json --output runs/fixed-training-001 --hours 1
.venv/bin/python scripts/fit_fixed_baseline.py \
  --run runs/fixed-training-001 --output models/best-fixed.json
```

The one-hour argument is an upper limit, not evidence that the full grid fits it.
Do not launch this collection from the commands alone, restart an incomplete
directory, or count synthetic tests as measured optimization results. Shared
training access for the learned methods and predictor must be recorded in the
revised data-allocation protocol before the grid is used in the full comparison.

## Implementation checks, 2026-10-29

The full Python correctness suite passed 96 checks in 18.24 seconds. The new
synthetic checks cover equal family weighting, compilation/reuse tradeoffs,
complete grid coverage, training-only membership, compiler/IR/measurement joins,
raw timing summaries, preserved collection failures, fixed selection across
programs, and comparison audit rejection of a changed fixed schedule. These
checks execute no new native performance campaign. The CI workflow includes them.
