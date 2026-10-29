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

## Fairness limitation

Matching the initial-pool fingerprint proves identical initial source access.
It does not establish identical learned models, training algorithms or additional
measurement access. Imitation performs separate leader rechecks before selecting
demonstrations. Sharing those additional outcomes with the cost predictor, and
auditing the complete measurement access of the final methods, remains unfinished.
Do not describe the full baseline comparison as fair solely because these initial
fingerprints match. No full-study superiority or held-out access follows.
