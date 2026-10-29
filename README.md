# RL for MLIR optimization

A research project studying compiler schedule selection and limited-feedback adaptation. The planned system combines MLIR/LLVM transformations, a C++ JIT runtime, and PPO/imitation/GRPO comparisons.

**Status: compiler/JIT core implemented and tested. No trained policy or performance finding yet.** Matrix multiplication and matrix-vector multiplication support tiling, parallel-loop interchange and masked vectorization. Timing stability is being investigated before training.

The question is whether learning improves total deployment cost, including compilation and reuse, and whether adapting to a new program repays its measurement cost. A simpler baseline winning is a valid outcome.

- [Research protocol](docs/protocol.md)
- [Milestones](docs/status.md)
- [Build and run checks](docs/build.md)
- [Compiler/runtime design](docs/compiler-runtime.md)
- [Original protocol issue](https://github.com/HassanAlhajAli/rl-mlir-optimizer/issues/1)

Scope: restricted numerical kernels, FP64, LLVM/MLIR 18.1.3, x86-64-v3 on Intel under Ubuntu/WSL2. These are not Arm results. This repository is separate from the earlier gem5 and bytecode-fusion projects.

Development is assisted by OpenAI Codex. Implementation, checks, corrections and evidence are recorded in commits. AI-generated code is subject to the same correctness and experiment requirements as other code.
