# RL for MLIR optimization

A research project studying compiler schedule selection and limited-feedback adaptation. The planned system combines MLIR/LLVM transformations, a C++ JIT runtime, and PPO/imitation/GRPO comparisons.

**Status: the main PPO campaign stopped after three completed jobs and a compiler crash in the fourth.** A revised vector backend passes local correctness checks and an 833-case persistent-worker replay. The original crash remains unexplained, and training is stopped. Six numerical families are supported with conservative legality masks. Earlier bounded PPO, imitation and adaptation integration remains recorded at its original compiler version. The small adaptation example did not demonstrate an advantage from learning over frozen-policy sampling. The main comparison is incomplete.

The question is whether learning improves total deployment cost, including compilation and reuse, and whether adapting to a new program repays its measurement cost. A simpler baseline winning is a valid outcome.

- [Research protocol](docs/protocol.md)
- [Milestones](docs/status.md)
- [Build and run checks](docs/build.md)
- [Compiler/runtime design](docs/compiler-runtime.md)
- [Training and observation design](docs/learning.md)
- [100-candidate development pilot](results/optimization-pilot-001/observations.md)
- [Repaired compiler pilot](results/optimization-pilot-002/observations.md)
- [Fresh deployment diagnostic](results/deployment-confirmation-001/observations.md)
- [Compiler failure and repair](docs/compiler-repair-001.md)
- [Main campaign interruption and vector repair](docs/compiler-repair-002.md)
- [Interrupted PPO campaign evidence](results/ppo-campaign-001/observations.md)
- [Budgeted adaptation design](docs/adaptation-design.md)
- [Evaluation seal and remaining prerequisites](docs/evaluation-seal.md)
- [Complete deployment comparison controller](docs/deployment-comparison.md)
- [Training-selected fixed schedule baseline](docs/fixed-baseline.md)
- [PPO, imitation and adaptation integration](results/integration-003/observations.md)
- [Bounded campaign and continuation](docs/campaign.md)
- [Original protocol issue](https://github.com/HassanAlhajAli/rl-mlir-optimizer/issues/1)

Scope: restricted numerical kernels, FP64, LLVM/MLIR 18.1.3, x86-64-v3 on Intel under Ubuntu/WSL2. These are not Arm results. This repository is separate from the earlier gem5 and bytecode-fusion projects.

Development is assisted by OpenAI Codex. Implementation, checks, corrections and evidence are recorded in commits. AI-generated code is subject to the same correctness and experiment requirements as other code.
