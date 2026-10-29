# Implementation status

This file distinguishes the approved scope from completed work. Issue #1 tracks the full study.

| Gate | State | Evidence required |
|---|---|---|
| A: environment/JIT pilot | Passed for bounded development; final noise checks remain required | Correct native code; repaired landscape resolves large effects while retaining drift flags |
| B: compiler/runtime | Core implemented and locally checked | Six families; 84 schedule cases plus family/runtime suites; stack-growth regression repaired; LLVM IR and assembly inspected |
| C: optimization pilot | Passed for training | Repaired pilot:100/100 correct,6 flagged original blocks,1 repeat still flagged; estimated reuse choices differ on8/20 workloads; no RL finding |
| D: PPO | Bounded integration passed; main study pending | Frozen 224-workload corpus; learnable diagnostic; 512 real actions/144 episodes completed on repaired compiler; two objectives/five seeds and validation selection outstanding |
| E: pretraining | Code and diagnostics implemented; measured integration pending | Balanced imitation and separate critic fitting, mixed-data collection and recheck entry points |
| F: adaptation | Code and diagnostics implemented; measured integration pending | Exact-budget GRPO/PPO/frozen/random/mutation; fresh deployment accounting; pretrained measured comparison still required |
| G: evaluation | Pending | Sealed splits, ablations, fresh measurements |
| H: release | Pending | Reproduction, report, focused explanation |

Do not check a gate solely because its code exists. The gate requires executed checks and inspectable evidence. No training or final evaluation is authorized by an unpassed prerequisite.
