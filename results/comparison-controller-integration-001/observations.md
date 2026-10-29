# Complete-deployment controller integration

The controller protocol was committed at `28ea316` before this native run. It
used the previously selected validation matvec33x210, the reference and locality
heuristic, reuse counts1/100/10000 and five seeded, shuffled rounds. The main PPO
campaign remained stopped, and held-out performance stayed sealed.

All **30 planned cases completed correctly**, including both input seeds during
validation. There were **31 fresh deployments**: one block was flagged, repeated
once, and remained flagged. Both attempts are retained. No failed or unfavorable
timing was discarded. The worker stage took24.46385755seconds; selector-loading
setup took0.589232184seconds separately. The three-minute nominal stage limit was
not reached.

The read-only auditor verified:

- Exact five-round grid coverage and seeded order.
- Canonical development workloads, source identities and every saved IR hash.
- Fresh compilation and exactly N calls for every deployment.
- Compilation + installation + execution equaling each native total, with the
  complete outer selection callback added once to the reported deployment cost.
- Retained drift attempts, one-repeat limit and the unresolved flag.
- All8,520,870 invocations,77 actual compilations and595 worker requests, including
  research checks/calibration and initialization, reconciling with saved counters.
- Release of both handles after every case, with no cleanup failures.

The complete Python suite passed63checks before this run. The final controller
and auditor suite subsequently passed21checks, including four new saved-evidence
tests that detect altered costs, counters and missing cases. The controller tests
use simulated responses to provoke failures; these are not performance results.

This establishes a working **development measurement controller**. It does not
establish RL benefit, generalization or a solution to the original compiler
SIGSEGV. No learned actor was evaluated in this integration. Full method freezing,
the remaining baselines, adaptation evaluation, warm-cache deployment and final
statistical analysis still need work. The repaired compiler's original interrupted
training data remain separate.

`raw.zip` contains the exact run configuration, event log, run metadata,30 IR
artifacts, verification logs and native audit. `manifest.json` hashes every member
and the auditor source. Recorded setup+stage cost25.053089734seconds is charged
once to the existing48hour ledger. Python startup and independent test/audit work
remain covered by the existing conservative development reserve.

Commands and measurement boundaries are in
[deployment-comparison.md](../../docs/deployment-comparison.md). The saved run can
be audited without executing generated code:

```sh
.venv/bin/python scripts/audit_comparison.py \
  runs/comparison-controller-integration-001 --output runs/new-comparison-audit.json
```
