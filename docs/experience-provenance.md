# Shared initial experience: integrity and identity

Imitation pretraining and the supervised cost predictor use the same initial
experience loader. This specification strengthens that input boundary before
another training campaign. No compiler, reward, split or evaluation budget is
changed, and no failed campaign is restarted.

Each source must have a terminal successful/budget-limited status, compatible
compiler/corpus/observation/measurement settings, unique accepted measurement
identities, and complete trajectories joined to the exact measured workload and
schedule. Reject failed source runs and failed measurement records rather than
silently filtering them into an apparently successful pool. Budget-limited runs
may supply their completed trajectories; this is not a claim of full corpus
coverage.

The loader checks finite numerical observations and costs, action/mask agreement,
connected state transitions and terminal placement. It recomputes measurement
keys and complete deployment costs from source records. These checks do not
re-execute compiler legality or numerical output. Native correctness remains the
responsibility of the original experiment and its evidence audit.

Record hashes of run metadata, measurements and trajectories, plus counts and
split/context coverage. Reject repeated source directories and byte-identical
source copies, which could accidentally multiply training weight. Preserve the
caller-specified source order because it can affect seeded training. A shared
initial-pool fingerprint includes ordered content hashes and compatibility
settings, excluding local path names so an unchanged relocated pool has the same
identity.

Both learners record that fingerprint. An optional saved manifest pins the exact
input bytes before training; a mismatch fails before worker/model training starts.
The manifest tool is read-only with respect to the source runs and starts no
compiler or training process. Compatibility with the current binary is still
required when fitting a model; a historical audit cannot grant transfer access.

## Additional rechecks: specification before implementation

Imitation's extra checks will become a separately completed, hashed evidence
bundle. For each program/reuse context, collapse identical histories using median
initial total and selection costs, then recheck the first two histories and all
others within 5% of the first. Preserve the original source order for ties. Export
every requested leader, including losing candidates, with its source trajectory
indices, accepted measurement key and recomputed total cost. The initial manifest
binds those indices to complete observations and actions; they are not fabricated
replacement trajectories.

Record each completed request as it happens. Publish a completion manifest only
after every planned request has an accepted outcome, all measurements are joined,
and request/distinct/cache counts agree. Preserve partial logs on failure, but
reject incomplete bundles for fitting. Measurement completion is separate from
whether subsequent actor training succeeds. The consumer checks exact initial
pool identity, compiler/corpus compatibility, byte hashes, full leader coverage,
schedule joins, costs and split membership before constructing a model.

The predictor may explicitly consume this same bundle. Each independently logged
measurement contributes once per program/history cost target; repeated cached
trajectories and reuse contexts do not multiply a timing's weight. Targets use
the median of initial and recheck compilation/setup and execution measurements.
Both learners record the same measurement-access fingerprint when supplied the
same complete bundle. Validation data remain exclusively for selection/validation.
The actor selects rechecked leaders; its critic retains the initial training pool.
Shared access does not require identical weighting or learning algorithms.

Existing historical files will not be rewritten or represented as if they had
this new export. Synthetic tests will establish the new boundary before any
authorized measurement/training run. No budgets or failed runs are restarted.

## Fairness limitation

Matching the initial-pool fingerprint proves identical initial source access.
It does not establish identical learned models, training algorithms or additional
measurement access. Imitation performs separate leader rechecks before selecting
demonstrations. Sharing those additional outcomes with the cost predictor, and
auditing the complete measurement access of the final methods, remains unfinished.
Do not describe the full baseline comparison as fair solely because these initial
fingerprints match. No full-study superiority or held-out access follows.

## Commands and verification scope

Create a local manifest from the preserved development integration sources:

```sh
.venv/bin/python scripts/freeze_experience.py \
  --runs runs/integration-003/ppo-reuse runs/integration-003/ppo-runtime \
         runs/integration-003/experience-train runs/integration-003/experience-validation \
  --output runs/initial-experience-audit-001/manifest.json
```

Pass the same `--pool-manifest <path>` and ordered `--runs ...` to `pretrain.py`
and `fit_predictor.py` when those training stages are authorized. Both save
`initial-pool-manifest.json` and `initial_pool_fingerprint` in their model metadata.
Existing commands without a pin remain available for development, but still use
the stricter loader and record a fingerprint. Fingerprints must be compared when
declaring equal initial access.

The historical manifest command checks sources against their own recorded
compiler contract. It does not make an old pool compatible with the current
compiler; both learning commands retain that separate guard. Positive compile
setup and execution costs are required because the models use log-cost targets.
The byte-level checks do not re-audit saved IR artifacts, regenerate observations
or verify masks against the native compiler. That stronger evidence comes from
the source experiment's own audit and must remain available.

Older measurements may lack a per-record corpus hash. They are accepted only
through the compatible parent run's corpus hash plus exact canonical workload
validation. The manifest counts every such record explicitly. A present but
incorrect per-record corpus hash is rejected; the loader never rewrites older
records or claims that missing fields were originally recorded.

The [historical integration audit](../results/initial-experience-audit-001/observations.md)
verified499 accepted measurements and405 complete trajectories without executing
native code or training a model. It explicitly reports missing per-record corpus
hashes and the additional-recheck fairness gap.
