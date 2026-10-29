# Shared recheck evidence: development verification

The additional timings used to select imitation demonstrations now have a
complete export that the supervised predictor can explicitly consume. This
addresses an implementation gap in access to measurement evidence. It is not a
trained-model performance comparison.

Specification commit: `728b764`. Implementation and audit revision:
`207dace37a1501e2ea2b9e510bfe88fc0a2f143c`.

## Executed checks

The full local correctness suite passed **199 tests in 22.90 seconds**. The new
synthetic cases check complete leader coverage, losing-candidate availability,
independent training/validation roles, cache-hit deduplication, partial failure
preservation, relocated bundles, both learner entry points, and rejection of
changed initial pools, costs, counts, source indices, schedules, splits, compiler
identity, failed measurements and missing corpus/drift fields. No native timing
is collected by the new recheck tests; other existing integration tests exercise
the worker for correctness.

A separate read-only audit used the original `integration-003` sources:

| Check | Observed result |
|---|---:|
| Initial trajectories | 405 |
| Program/reuse contexts | 155 |
| Planned leader requests | 255 |
| Leader key sets matching the original demonstrations | All 155 contexts |
| Initial predictor program/history examples | 273 |
| Training / validation histories | 254 / 19 |
| Distinct source-measurement contributions to those examples | 351 |

Thus the extracted leader-selection logic preserves the candidates used by the
original integration, including its rule to recheck two leaders and other near
ties. The predictor example constructor also accepts the real saved observations.
Repeated trajectories do not multiply one saved timing's weight. Different
histories and independent source measurements remain visible.

The machine-readable [summary](summary.json) pins the initial pool and original
demonstration bytes. The [initial experience audit](../initial-experience-audit-001/observations.md)
records all source-file hashes and the legacy corpus-metadata limitation.

## Reproduce the read-only candidate comparison

From the repository root, with the preserved local run directories available:

```sh
.venv/bin/python - <<'PY'
import json, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, 'python')
from rlmlir.experience import load_pool, pool_manifest, digest
from rlmlir.rechecks import planned_rechecks
from rlmlir.predictor import cost_examples
base = Path('runs/integration-003')
runs = [base / n for n in ('ppo-reuse', 'ppo-runtime',
                          'experience-train', 'experience-validation')]
meta = json.loads((runs[0] / 'run.json').read_bytes())
pool, sources = load_pool(runs, meta)
plan = planned_rechecks(pool)
raw = (base / 'pretrain/demonstrations.json').read_bytes()
demos = json.loads(raw)['demonstrations']
expected = {}
for p in plan:
    context = (p['workload_id'], p['reuse'])
    key = pool[p['source_indices'][0]]['outcome']['measurement_key']
    expected.setdefault(context, []).append(key)
observed = {(d['workload']['id'], d['reuse']):
            [c['key'] for c in d['rechecked_candidates']] for d in demos}
assert len(observed) == len(demos) and set(expected) == set(observed)
assert all(Counter(expected[k]) == Counter(observed[k]) for k in expected)
examples = cost_examples(pool)
print(pool_manifest(pool, sources, meta)['fingerprint'], digest(raw))
print(len(pool), len(expected), len(plan), len(examples))
print(Counter(e['split'] for e in examples))
print(sum(e['distinct_measurements'] for e in examples))
PY
```

This audit does not generate a new-format recheck bundle from the old run. The
old artifacts remain unchanged. No actor or predictor was trained, no native
performance was queried, and no held-out data were opened. Full current-backend
training, equal-access artifact checks and final evaluation remain pending.
The main PPO queue remains stopped at its preserved compiler failure.
