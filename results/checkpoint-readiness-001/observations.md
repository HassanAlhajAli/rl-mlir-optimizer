# PPO checkpoint-selection coverage audit

This is a read-only audit of the interrupted campaign, performed with source
`153dcc2788e148075b0e9e68a38520c4baf96aae`. It starts no compiler worker, trains no
policy and measures no kernel. The [complete report](report.json) retains all ten
declared objective/seed combinations and hashes the source configuration, queue,
campaign and available child metadata.

| Seed | Reuse objective | Runtime objective |
|---|---|---|
| 7 | Completed at request limit | Completed at distinct-measurement limit |
| 17 | Completed at request limit | Failed during compilation |
| 27 | Not started | Not started |
| 37 | Not started | Not started |
| 47 | Not started | Not started |

All four started jobs used the previous compiler binary. Their saved outcomes
remain valid historical records, but are incompatible with the current compiler
contract. No separately audited validation selection exists for any of the ten
jobs. The three successful bounded runs do not establish a completed study.

The original failure is unchanged: runtime/17 stopped when the worker exited
with signal 11. The queue's original elapsed cost remains 6,372.591520136001
seconds, charged previously. This audit does not repair that failure, reset
requests, transfer old rewards or allocate replacement runs.

The report says `all_ten_selections_accounted_for=false`,
`whole_study_frozen=false` and `heldout_access=false`. Its lightweight mode does
not recheck optimizer updates; the earlier full campaign audit retains that
separate evidence. The new validation controller and coverage checker passed
the complete 116-test Python suite, including a synthetic 192-case validation
selection and deliberately corrupted evidence. No real current-backend
checkpoint validation was performed in this check.

Reproduce against the original saved campaign, using a new output path:

```sh
.venv/bin/python scripts/audit_ppo_campaign.py runs/ppo-campaign-001 \
  --readiness-only --config configs/ppo-campaign.json \
  --output runs/checkpoint-readiness-recheck/report.json
```

The [validation protocol](../../docs/checkpoint-validation.md) explains the new
hashes, complete-coverage requirement and evidence limits. No changes to the
48-hour allowance or remaining adaptation/final reservation were made.
