# Frozen split and experience integrity checks

Revision `1840cdf` strengthens the development measurement boundary without
opening held-out performance or changing the compiler binary, action space,
reward calculation or request caps. The main PPO campaign remains stopped.

The earlier evaluator trusted a caller's `train` or `validation` label. It now
checks the complete workload identity against the hashed frozen corpus before
using a cache entry or contacting the compiler. Experience loading also requires
each trajectory to refer to a correct measurement of that same workload,
schedule and compiler. PPO resume and checkpoint selection use the complete
provenance compatibility contract.

All **46 Python checks passed**, including 22 new corpus/experience cases. The
new checks reject relabeled held-out IDs, altered shapes, duplicate identities,
corrupted corpus hashes, incorrect cached outcomes and mismatched trajectory
joins. These rejection tests do not execute or time held-out programs.

Read-only audits then checked the saved evidence:

- The development integration's 787 accepted measurements passed the strengthened
  manifest checks, together with its trajectory, demonstration and adaptation
  accounting checks.
- All 5,020 records from the interrupted PPO campaign retained their canonical
  training identities. The original failed record remains a failure; the other
  5,019 were accepted. No old measurements were replaced.
- All 19,377 saved trajectories across the four integration experience sources
  and four attempted main jobs joined to the same correct workload and schedule.
  There were 19,341 training and 36 validation trajectories, with no held-out
  entries. This check used the historical compiler identity and did not import
  those measurements into the revised backend's training data.

This found no split contamination in the audited records. It does not establish
generalization or RL superiority, explain the original crash, or constitute a
final method freeze. The [evaluation-seal document](../../docs/evaluation-seal.md)
lists the remaining prerequisites.

`raw.zip` contains the test log and three read-only audit reports. `manifest.json`
records hashes for the archive, every member and the code revision. The original
experiment artifacts remain in their earlier result directories. No new hardware
measurement campaign ran; these development checks fall within the existing
conservative debugging/check reserve.

```sh
.venv/bin/python -m pytest tests/test_learning.py tests/test_adaptation.py \
  tests/test_predictor.py tests/test_queue.py tests/test_corpus.py -q
.venv/bin/python scripts/audit_integration.py runs/integration-003 \
  --output runs/new-split-audit/integration.json
.venv/bin/python scripts/audit_ppo_campaign.py runs/ppo-campaign-001 \
  --output runs/new-split-audit/main.json
```
