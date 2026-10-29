# Implementation status

This file distinguishes the approved scope from completed work. Issue #1 tracks the full study.

| Gate | State | Evidence required |
|---|---|---|
| A: environment/JIT pilot | Correctness passed; timing stability under investigation | Two preserved pilots show host variability |
| B: compiler/runtime | Core implemented and locally checked | Six families; 84 schedule cases plus family/runtime suites; stack-growth regression repaired; LLVM IR and assembly inspected |
| C: optimization pilot | Development evidence collected | 100 candidates, 90 fresh deployments; old backend results preserved; repeat the landscape on repaired backend before main training |
| D: PPO | Bounded integration passed; main study pending | Frozen 224-workload corpus; learnable diagnostic; 512 real actions/144 episodes completed on repaired compiler; two objectives/five seeds and validation selection outstanding |
| E: pretraining | Implementation in progress | Balanced imitation and separate critic fitting; mixed-data collection, rechecks and measured validation still required |
| F: adaptation | Implementation in progress | Group controller and GRPO/PPO/search mechanics; verified pretrained input and measured comparisons still required |
| G: evaluation | Pending | Sealed splits, ablations, fresh measurements |
| H: release | Pending | Reproduction, report, focused explanation |

Do not check a gate solely because its code exists. The gate requires executed checks and inspectable evidence. No training or final evaluation is authorized by an unpassed prerequisite.
