# Development comparison analysis specification

Recorded before implementing the summarizer. Existing development measurements
have already been inspected, so applying this rule to them is exploratory. It
does not freeze the full study or open held-out performance.

## Inclusion and cost rules

Analyze only a complete, audited frozen-selector comparison. A failed, missing,
duplicate or incompatible case prevents an aggregate; do not summarize only
successful cases. Keep the full declared workload/method/reuse/round grid.

For a case with a drift repeat, take the arithmetic mean of **both** recorded
deployment attempts. Neither the original nor the repeat is selected because it
is faster. Selection cost is charged once in each attempt's deployment objective,
as it was measured for that case. This average estimates one deployment; the
cost of collecting repeats stays in campaign hardware/time accounting.

Pair a method with the explicitly named baseline at the same workload, reuse
count and round. For each round compute the log of baseline total divided by
method total, then average those log ratios across rounds. Exponentiating gives
that workload's geometric-mean speedup. A ratio below one is a regression.
These are paired round blocks, not simultaneous measurements; pairing cannot
remove every source of drift. Do not pool different reuse counts.

Report every workload's paired rounds, selection/compile/install/execution
costs, steady-state runtime, speedup, regressions and flagged/unresolved counts.
Timing warnings remain included in the primary estimate. Report a separate
warning-free count, not a selectively filtered headline estimate.

## Weighting and uncertainty

Within a family, average workload log ratios equally. Across represented
families, average family log ratios equally. This prevents a family with more
generated shapes from dominating. Report the families actually represented;
the summary does not imply coverage of absent families or unseen programs.

Use a stratified cluster bootstrap, with frozen corpus `group` as the resampling
unit inside each family. All shapes and all paired rounds in a selected group
stay together. Each replicate samples the original number of groups with
replacement per family, computes the mean over the sampled workloads, then
combines family means equally. Use 2,000 replicates, seed 20261029 and a 95%
percentile interval. Store these settings in the report. The interval is
conditional on these methods, recorded rounds and sampled workload groups; it
does not include training-seed variation, independent-machine variation or
multiple-comparison correction.

If any represented family has fewer than two groups, omit the aggregate
confidence interval and state why. Fewer than ten groups in any family is flagged
as a small-cluster exploratory estimate. Repeating one program many times does
not create evidence of generalization. Each named policy remains separate; do
not pool different policy seeds into fictitious independent programs.

## Reporting boundary

The caller names a baseline; the tool does not choose whichever baseline makes
a result look best. It reports development comparisons, never an automatic
superiority decision. The required strongest non-RL baseline selection, all five
training seeds, full method freeze, independent confirmation and final analysis
remain separate requirements. Adaptation costs require their own comparison;
this tool summarizes frozen-selector complete deployments only.

Read-only analysis must retain source hashes and the integrity-audit result. It
must not load a policy, execute native code, edit input evidence or start timing.

## Implemented entry point

`python/rlmlir/analysis.py` implements the rules above. The command first runs
the saved-evidence integrity audit, then verifies the complete grid again before
calculating an estimate. Use a new output path:

```sh
.venv/bin/python scripts/summarize_comparison.py \
  runs/comparison-controller-integration-001 \
  --baseline reference --output runs/development-analysis-001/report.json
```

The JSON report contains every paired round and attempt cost, workload and family
estimates, equal-family aggregates and uncertainty limitations. Tests use known
answers and deliberate corruptions, including a family-count imbalance that
would produce a misleading gain under simple pooling. The statistics do not
repair incomplete evidence or convert development measurements into a final
confirmatory study.
