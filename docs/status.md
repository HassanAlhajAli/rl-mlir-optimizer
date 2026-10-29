# Implementation status

This file distinguishes the approved scope from completed work. Issue #1 tracks the full study.

| Gate | State | Evidence required |
|---|---|---|
| A: environment/JIT pilot | Passed for bounded development; final noise checks remain required | Correct native code; repaired landscape resolves large effects while retaining drift flags |
| B: compiler/runtime | Core implemented and locally checked | Six families; 84 schedule cases plus family/runtime suites; stack-growth regression repaired; LLVM IR and assembly inspected |
| C: optimization pilot | Passed for training | Repaired pilot:100/100 correct,6 flagged original blocks,1 repeat still flagged; estimated reuse choices differ on8/20 workloads; no RL finding |
| D: PPO | Both objectives passed integration; main campaign ready | Integration-003: 512 steps each, 152/145 episodes, changed weights and 20 optimizer epochs each; five-seed training and validation selection outstanding |
| E: pretraining | Measured development integration passed; full study pending | Hashed mixed trajectories; 143 training and 12 validation demonstration contexts; separate training-only critic fitting |
| F: adaptation | Measured development integration passed; full study pending | Five methods at 8/16 trials on one validation case; exact budgets, fresh deployment, finite updates; per-method counter repair verified separately; no learning advantage established |
| G: evaluation | Pending | Sealed splits, ablations, fresh measurements |
| H: release | Pending | Reproduction, report, focused explanation |

Do not check a gate solely because its code exists. The gate requires executed checks and inspectable evidence. No training or final evaluation is authorized by an unpassed prerequisite.

The [integration report](../results/integration-003/observations.md) records the
evidence and limitations authorizing the bounded main PPO queue. Held-out
performance remains sealed. Issue #1 remains open for the full project.
