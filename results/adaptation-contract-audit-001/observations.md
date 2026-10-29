# Read-only audit of historical adaptation deployments

On 2026-10-29, source `809e1822761c5af7a355f8d46cbb36732250c400`
checked the saved native results against the shared deployment contract.
No compiler, training or benchmark process was started by this audit.

| Saved campaign | Cases checked | Deployment/count/cost and logical-budget checks | Reset digests and separate deployment counters |
|---|---:|---|---|
| integration-003/adaptation | 10 | Passed | Not recorded in this earlier version |
| adaptation-accounting-001 | 6 | Passed | Present and consistent |

All 16 cases report a fresh compilation, exactly 10,000 deployment calls,
finite nonnegative components summing to deployment cost, and tuning plus
deployment summing to adapted total. Each case has its declared 8 or 16 logical
requests, and its selected candidate is the best observed within those requests.
The six later cases share an initial policy digest and independently report one
deployment compilation and 10,000 invocations. No-update controls preserve that
policy digest. Both complete method/budget grids are checked.

These are record-consistency checks. Numerical output was not executed again;
the audit checks the saved correctness flags. Neither historical campaign has
the new event log. Missing older evidence is not reconstructed. The original
integration's lifetime hardware counters must still not be summed as if they
were per-method totals.

Both campaigns used the old compiler binary
`63d174c5ca0e41f6d7b2a5f076087d558526c9f383b95374385da85a9b3e466a`.
This does not validate performance under the revised compiler, demonstrate an
RL advantage, or resolve the stopped main campaign's original crash.

Reproduce from the repository in Ubuntu, using a new output path:

```sh
.venv/bin/python scripts/audit_adaptation_deployments.py runs/integration-003/adaptation runs/adaptation-accounting-001 --output runs/adaptation-contract-audit-001/report.json
```

The [report](report.json) hashes the original result and run files and identifies
exactly which evidence fields exist. Its SHA-256 is
`4f5d6a5dd4803c41adbce3355901c80048dfac5055d58fa2e78622afe1087a85`.
The source integration evidence is documented in
[integration-003](../integration-003/observations.md).

Separately, the updated controller passed the complete Python correctness suite:
143 checks in 20.21 seconds. Those checks use synthetic deployment failures and
existing native compiler-analysis checks; they are not a new performance study.
