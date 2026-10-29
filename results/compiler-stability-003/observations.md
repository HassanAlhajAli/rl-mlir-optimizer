# Long-lived request-path replay

The bounded diagnostic at revision `30b48d3` completed the saved runtime/17
measurement history on the current `native-inner-tile-v2` backend. It used one
worker, the original order and saved calibration/benchmark invocation counts.
All 832 originally accepted records passed their numerical checks. The 833rd
candidate, row reduction 336x205, compiled successfully at the original crash
point. The worker made 13,271 successful requests, including 2,496 invokes and
6,611 benchmarks covering 599,529,512 total invocations. Those invoke and
benchmark counts match the original stopped job. The original job also made
`analyze` requests interleaved with PPO, which this diagnostic cannot reconstruct.

The run took 926.87 seconds against a 1,800-second overall ceiling. The worker
had a 120-second per-request timeout and 4 GiB address-space limit. Recorded
peak resident memory was 423,464 KiB and peak virtual memory was 607,532 KiB.
Worker stderr was empty. The [read-only audit](audit.json) verifies paired
start/completion entries for every request, counters, numerical record coverage,
the original failure point, resource observations and stderr length.

This is stronger stability evidence for the long-lived request path than the
earlier correctness-only replays. It does **not** explain the original SIGSEGV
or prove future stability. It does not reproduce policy computation, `analyze`
interleaving, exact machine state or the old binary. The new batch times were
not retained as optimization measurements. The original failure, 5,020
campaign records and prior costs remain preserved.

[raw.zip](raw.zip) contains the full per-request trail, record outcomes,
`run.json` and worker stderr. [manifest.json](manifest.json) hashes each file
and the archive. Reproduce the diagnostic with the command in
[compiler-stability-003.md](../../docs/compiler-stability-003.md); audit with:

```sh
.venv/bin/python scripts/audit_measurement_history.py runs/compiler-stability-003
```

This pass permits the separately bounded current-backend development pilot
specified in [current-backend-restart.md](../../docs/current-backend-restart.md).
It does not authorize automatic restart of the old PPO queue or opening
held-out performance.
