# Implementation status

This file distinguishes the approved scope from completed work. Issue #1 tracks the full study.

| Gate | State | Evidence required |
|---|---|---|
| A: environment/JIT pilot | Passed for bounded development; final noise checks remain required | Correct native code; repaired landscape resolves large effects while retaining drift flags |
| B: compiler/runtime | Revised backend passes ordinary and protected-memory 833-record replays; original crash cause unresolved | Six families; 88 schedule cases plus family/runtime suites; memory-guard fault controls; native-width vector lowering addresses a reproduced resource failure |
| C: optimization pilot | Passed for training | Repaired pilot:100/100 correct,6 flagged original blocks,1 repeat still flagged; estimated reuse choices differ on8/20 workloads; no RL finding |
| D: PPO | Main campaign stopped; three jobs completed, fourth failed, six unstarted | Original checkpoints/results preserved; compiler failure diagnosis and revised-backend verification required before continuation |
| E: pretraining | Measured development integration passed; full study pending | Hashed mixed trajectories; 143 training and 12 validation demonstration contexts; separate training-only critic fitting |
| F: adaptation | Measured development integration passed; full study pending | Five methods at 8/16 trials on one validation case; exact budgets, fresh deployment, finite updates; per-method counter repair verified separately; no learning advantage established |
| G: evaluation | Partial development controller verified on30cases; final study pending | Final method freeze, full baseline/adaptation orchestration, ablations and statistics remain unfinished |
| H: release | Pending | Reproduction, report, focused explanation |

Do not check a gate solely because its code exists. The gate requires executed checks and inspectable evidence. No training or final evaluation is authorized by an unpassed prerequisite.

The [integration report](../results/integration-003/observations.md) records the
evidence and limitations authorizing the bounded main PPO queue. Held-out
performance remains sealed. Issue #1 remains open for the full project.

The [main-campaign interruption report](compiler-repair-002.md) supersedes the
earlier ready-to-run status. The revised compiler must not inherit the old
backend's performance conclusions. No full-study superiority is established.

The [evaluation seal](evaluation-seal.md) verifies corpus membership at the
measurement and experience-loading boundaries. Strengthened checks passed on
the saved integration and interrupted campaign. This does not open held-out
performance or resolve the compiler continuation gate.

The [complete-deployment controller integration](../results/comparison-controller-integration-001/observations.md)
passed30native cases, with one unresolved timing flag retained. It compares the
reference and heuristic only; no new learned-policy performance claim follows.

The [training-selected fixed baseline](fixed-baseline.md) now has a complete-grid
collector, evidence checks, selection rule and deployment integration. It has
synthetic correctness coverage only. The 2,048-cell current-backend training grid
has not run and no fitted fixed artifact exists. This does not reopen training.
