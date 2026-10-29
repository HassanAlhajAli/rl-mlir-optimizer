# Historical initial-experience integrity audit

The preserved integration's four initial experience sources pass the strengthened
loader under their own recorded compiler contract. This is a read-only source
audit, not new training, native execution or evidence of current-compiler speed.

| Source | Accepted measurement records | Complete trajectories | Split |
|---|---:|---:|---|
| ppo-reuse | 224 | 152 | Training |
| ppo-runtime | 219 | 145 | Training |
| experience-train | 37 | 72 | Training |
| experience-validation | 19 | 36 | Validation |
| Total | 499 | 405 | 369 training and 36 validation trajectories |

These trajectories cover155 distinct program/reuse contexts. The earlier
pretraining record selected143 training and12 validation demonstration contexts.
All four trajectory hashes still match those recorded by that pretraining run.
Multiple trajectories can reuse one measurement, and reference measurements can
exist without being a selected trajectory; these counts are not interchangeable.

The loader verified canonical workload identities, compatible source settings,
accepted measurement status, recomputed measurement keys, trajectory/measurement
joins, finite observations and costs, action/mask agreement, connected states,
terminal placement and recorded objective costs. It reconstructs complete
deployment costs for execution-only source trajectories before reuse-aware
pretraining. It does not change the original execution-only outcome fields.

All499 historical measurement records lack a local corpus-hash field. Their
parent run metadata contains the correct frozen-corpus hash, and each workload
passes exact canonical identity validation. The manifest explicitly counts the
missing fields. No record was rewritten or presented as if it originally had the
new field. This audit does not regenerate observations or native masks, execute
numerical correctness checks, or repeat the separate saved-IR audit.

## Reproduce and inspect

Specification commit: `53f3ab4`. Implementation and audit source:
`b6f801024762938dcb0c657fd57ea8cd393a0db9`.

```sh
.venv/bin/python scripts/freeze_experience.py \
  --runs runs/integration-003/ppo-reuse runs/integration-003/ppo-runtime \
         runs/integration-003/experience-train runs/integration-003/experience-validation \
  --output runs/initial-experience-audit-001/manifest.json
```

Use a new output path when this one exists. The [manifest](manifest.json) hashes
the exact run metadata, measurement and trajectory bytes, preserving source
order. Its initial-pool fingerprint is
`688f6a8f9138802e42f245be573a77753b89b24e8892cab736448b9b32cadc50`.
The entire manifest file has SHA-256
`0daba0a2727e25580ed711db8277acfd1618456ce6e617633041eff4c26e4748`.

The full Python suite passed181checks in21.93seconds, including corruption tests
and checks that both learners reject a changed pinned pool before starting model
training or a compiler worker. The records remain tied to old compiler binary
`63d174c5ca0e41f6d7b2a5f076087d558526c9f383b95374385da85a9b3e466a`.
The current-binary compatibility guard is unchanged.

## What this does not establish

Matching initial-pool fingerprints would establish identical initial source
access. Imitation's additional leader rechecks are **not included** here, and
sharing those measurements with the cost predictor remains unfinished. This is
not proof of equal access to all study data, a trained predictor's performance,
or an RL advantage. The main compiler failure is unresolved, the original PPO
campaign remains stopped, and held-out performance remains sealed.
