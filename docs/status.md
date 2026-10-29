# Implementation status

This file distinguishes the approved scope from completed work. Issue #1 tracks the full study.

| Gate | State | Evidence required |
|---|---|---|
| A: environment/JIT pilot | Correctness passed; timing stability under investigation | Two preserved pilots show host variability |
| B: compiler/runtime | Core implemented and locally checked | 81 scheduled kernels, strict math, legality, rollback and cache tests pass; machine-code inspection follows |
| C: optimization pilot | Pending | Repeatable training-only schedule tradeoffs |
| D: PPO | Pending | Frozen corpus, learnable diagnostic, two objectives |
| E: pretraining | Pending | Provenance-complete trajectories and imitation validation |
| F: adaptation | Pending | GRPO/PPO/search, exact budgets/resets/costs |
| G: evaluation | Pending | Sealed splits, ablations, fresh measurements |
| H: release | Pending | Reproduction, report, focused explanation |

Do not check a gate solely because its code exists. The gate requires executed checks and inspectable evidence. No training or final evaluation is authorized by an unpassed prerequisite.
