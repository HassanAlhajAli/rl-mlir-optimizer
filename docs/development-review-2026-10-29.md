# Development review: 29 October 2026

The compiler, JIT and learning pipeline have working development evidence. The
main research comparison is incomplete. Its first PPO campaign stopped at a
compiler failure; a later compiler repair changes generated code, so the earlier
performance results cannot be treated as results for the current backend.

## Implemented and exercised

| Component | Evidence | Remaining limit |
|---|---|---|
| MLIR transformations and LLVM code generation | Six numerical families; verification and conservative action masks | Restricted kernels and schedules, not a general compiler optimizer |
| C++ JIT ownership, cache and execution | Native correctness suites; ordinary and protected-buffer replays | Replays do not establish absence of compiler/runtime defects |
| Both PPO objectives, imitation and budgeted adaptation | Six-stage [integration](../results/integration-003/observations.md) completed with real measurements and policy updates | Older backend; small development integration, not the full study |
| Validation selection, fixed baseline and complete deployment comparison | Synthetic coverage, plus a [30-case reference/heuristic integration](../results/comparison-controller-integration-001/observations.md) | Current-backend fixed grid and learned-policy comparison remain pending |
| Shared learning evidence | [Initial source audit](../results/initial-experience-audit-001/observations.md) and [shared recheck verification](../results/shared-recheck-audit-001/observations.md) | No current-backend fitted imitation/predictor pair yet |
| Reporting and experiment controls | Exact budgets, split guards, provenance, retained failures/drift and [paired analysis](comparison-analysis.md) | Full final-method freeze and held-out evaluator are unfinished |

The latest code verification passed 199 local Python tests and
[GitHub correctness CI](https://github.com/HassanAlhajAli/rl-mlir-optimizer/actions/runs/40857620939).
Passing tests is engineering evidence, not a performance finding.

## Research observations so far

The earlier [100-candidate pilot](../results/optimization-pilot-002/observations.md)
passed numerical checks and showed different estimated preferred schedules at
different reuse counts on 8 of 20 workloads. Those are selected development
estimates on the earlier backend. They neither confirm the main hypothesis nor
describe the repaired backend's optimization landscape.

In the older integration's single validation example, GRPO and PPO adaptation
updated their parameters but sampled the same schedules as frozen-policy
sampling. No additional learning benefit was demonstrated. Tuning took seconds;
fresh deployment of the chosen code for 10,000 invocations took tens of
milliseconds. Extra exploration was therefore expensive for that reuse context.

The current backend's [reference/heuristic comparison](../results/development-analysis-001/observations.md)
retains a heuristic regression on one validation matrix-vector program. There
are five rounds but only one program group, so no cross-program uncertainty
interval or broad method ranking is justified. No learned policy was compared.

## Interrupted campaign and next decision

The original ten-job protocol remains visible: three jobs completed their
recorded limits, the fourth failed, and six never started. All ten separate
validation selections are missing. The [coverage audit](../results/checkpoint-readiness-001/observations.md)
also records that all four started jobs use the previous compiler binary.

Two failure observations must remain distinct. A large-vector LLVM resource
problem was reproduced and repaired with native-width inner tiling. The original
row-reduction SIGSEGV did not reproduce in fresh tests and remains unexplained.
A Clang build also failed once. The ordinary and guarded 833-record replays pass,
but do not reproduce the original calibration/benchmark call counts. See the
[diagnostic evidence](compiler-repair-002.md).

Before a new main campaign:

1. Define the next bounded compiler diagnostic around the remaining uncertainty:
   the long-lived request history and execution/calibration path. Preserve
   per-request failure context, stderr and resource observations. Stop on a failure;
   do not turn it into a low reward, skip it, or keep retrying for a passing run.
2. Record the stability conclusion and any unresolved limitation. Freeze a
   current-backend development pilot and allocation before new timing. The old
   pilot's tradeoffs cannot automatically authorize large training on changed code.
3. Revise the campaign protocol explicitly, with both objectives and all five
   seeds visible, prior costs still charged, and a serial queue. Do not reuse old
   timing caches or present old checkpoints as a same-backend continuation.
4. After eligible training, select each checkpoint on validation; fit the fixed,
   supervised and imitation baselines with audited evidence access; compare
   frozen sampling, PPO, GRPO and simpler search at equal request budgets.
5. Freeze all methods, artifact identities and statistical rules before opening
   held-out performance. Run fresh complete deployments and report regressions,
   failures, tuning cost and uncertainty, even if simpler methods win.

This review launches no diagnostic or replacement campaign and allocates no new
requests. The [ledger](compute-ledger.json) records 8,926.02 seconds, plus a
7,200-second conservative development reserve: 16,126.02 seconds charged, about
4.48 hours of the 48-hour allowance. At least 24 hours remain reserved for
adaptation/final measurements; 19.52 hours remain unallocated. These accounting
boundaries differ from elapsed overnight agent work and are stated in the ledger.

All findings concern the pinned Intel/WSL environment. No Arm performance,
novelty, or RL-superiority claim is established. Issue #1 remains open.

## Later stability and allocation update

The [long-lived replay](../results/compiler-stability-003/observations.md)
completed the saved 833-record request path, including the originally failed
compile, with 13,271 successful requests and no worker stderr. It repeated the
original 599,529,512 measured invocations but could not reconstruct PPO/analyze
interleaving or the old machine state. The SIGSEGV cause remains unknown.
The [current-backend restart revision](current-backend-restart.md) now freezes a
small pilot and a conditional fresh ten-job allocation. The ledger charges this
diagnostic at 926.87 seconds; 17,052.89 seconds are now charged overall before
the pilot. No replacement training or held-out performance has started.
