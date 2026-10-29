# Optimization pilot protocol

Recorded before measurements. This pilot determines whether the compiler choices produce meaningful code differences and whether measurements are stable enough to justify learning. It is not an RL comparison.

`configs/pilot.json` fixes20 development workloads and five schedules per workload:100 distinct source/schedule pairs. All are training/development examples; future held-out manifests must exclude their shape groups. Matrix and vector families are balanced. No result is a global optimum.

Every candidate compiles with the same strict FP64/backend settings, passes independent reference checks for seeds7/17/27, and exports source/transformed/lowered IR, optimized LLVM IR and actual JIT object/disassembly. Output reset is checked with repeated invocation. Record first call and compilation separately. Compilation includes source parsing, schedule application, backend lowering and code generation; JSON orchestration is separately visible as request time.

Compiler initialization is measured once using a trivial development kernel and excluded from the already-initialized-service objective. Fresh candidates must be cache misses. Finish compilation/disassembly before each workload's timed block. Calibrate steady-state counts to300ms, require at least100ms; five rounds interleave candidate methods in seeded order. Bracket each block with baseline anchors. Preserve and flag >5% drift or too-short batches, repeat the complete block once. Keep continuing instability explicit. Do not select only favorable repetitions.

Primary pilot summaries use original blocks (attempt0), with repeat attempts separately available for sensitivity. Report every workload and failed attempt. Any C+N*T totals are estimates, not final measured deployment latency. A meaningful signal can motivate the next stage, but noisy small differences do not establish a method's superiority.

Run, after committing the implementation:

```sh
python3 scripts/optimization_pilot.py --output runs/optimization-pilot-001
```

The run refuses to overwrite evidence, stops on correctness/compiler failure, limits individual worker requests, and writes progress plus a stop reason. This initial pilot is checkpointed by completed workloads but does not silently resume or merge runs. It uses only local CPU execution. No full training or held-out evaluation starts automatically.

## Targeted confirmation, recorded after the pilot

The pilot completed100 correct candidates in381seconds.25/100 original blocks and6/25 repeats were flagged. Its observations motivate fresh complete deployment measurements for matvec128x256 (large observed vector gain) and matmul64x64x64 (vector regression/high compilation cost). This selection is post-pilot, not a held-out or preregistered representative claim. `configs/confirmation.json` fixes the two cases, three schedules, three reuse counts and five randomized rounds before the new measurements.

`deploy` bypasses the code cache. Its native clock spans compilation, installation/owned-buffer construction and all N invocations, including the first call. Transport/JSON receive, service initialization, independent reference validation and code teardown are outside that boundary. This is a broader boundary than the pilot's C+N*T estimate, with allocation cost explicit. No policy-selection cost exists yet. A targeted confirmation supports examples only, not broad superiority. Full latency batches may be short at N=1; they are not steady-state throughput samples and are not padded with extra calls.

Run `python3 scripts/confirm_deployment.py --output runs/deployment-confirmation-001` after committing. Preserve all90 deployments. This diagnostic does not replace the final study's anchor/drift protocol.
