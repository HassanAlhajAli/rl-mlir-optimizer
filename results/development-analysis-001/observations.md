# Exploratory summary of the saved deployment integration

The locality/vectorization heuristic was slower than the fixed reference at all
three measured reuse levels on validation matvec33x210. No learned policy was
tested. This is one development program, not a general compiler or RL result.

| Invocations N | Mean reference total (ms) | Mean heuristic total (ms) | Paired geometric speedup (reference / heuristic) |
|---:|---:|---:|---:|
| 1 | 11.36 | 25.39 | 0.448 |
| 100 | 11.65 | 26.60 | 0.438 |
| 10,000 | 33.21 | 84.66 | 0.392 |

A speedup below one means the heuristic loses. These totals include schedule
selection, fresh compilation, installation and exactly N invocations. The two
mean-cost columns average the complete costs; the speedup column averages paired
round log ratios, so it is not defined as the ratio of the two displayed means.

The heuristic selected `TILE(0,8), VECTORIZE`. Its mean compilation cost was
about23.5-23.7ms, compared with11.3-11.4ms for the reference. At10,000 invocations,
its mean execution batch took59.30ms versus21.75ms. Selection added about1.8ms
versus0.009ms in that context. Thus the observed loss is not explained solely by
the extra schedule-selection overhead. Identifying the exact generated-code
bottleneck would require a separate inspection; this report does not assert one.

All30 planned cases and31 recorded attempts remain included. One original block
was flagged for timing drift, repeated once and remained flagged. Both attempts
contribute equally to that case's cost estimate. The unresolved warning affects
the10,000-invocation comparison and is visible in the report. No unfavorable
attempt or workload was removed.

There is only one frozen shape group. The summarizer therefore omits a confidence
interval for generalization, even though five rounds were measured. Repeating
this program cannot establish performance on other programs. This analysis was
specified after development data had already been inspected and is explicitly
exploratory. It does not meet the full study's superiority criteria.

## Evidence and reproduction

The [analysis specification](../../docs/comparison-analysis.md) was committed as
`d5de1d9` before implementation. The implementation at
`154caf9e2527ad5916d36ea0ae265c1663545f29` first audited the saved configuration,
case coverage, IR hashes, costs, drift handling and hardware counters, then
summarized the complete grid. No native code, policy training or new timing ran.
The original measurements and their limitations remain in the
[controller integration report](../comparison-controller-integration-001/observations.md).

```sh
.venv/bin/python scripts/summarize_comparison.py \
  runs/comparison-controller-integration-001 --baseline reference \
  --output runs/development-analysis-001/report.json
```

Use a new output path if the report already exists. The [JSON report](report.json)
includes all paired costs, workload/family estimates, warnings, source hashes
and the saved-evidence audit. Its SHA-256 is
`76da3872cad775d65b5e09757e47cbe101e6f3d9bce3c1a7d67a6aa93c10f795`.

The full Python correctness suite passed159checks in21.96seconds. Known-answer
tests cover family balance, paired ratios, retained drift repeats and clustered
uncertainty; corruption controls reject missing or invalid evidence. These tests
validate the analysis implementation, not a performance claim. The main PPO
campaign remains stopped and held-out performance remains sealed.
