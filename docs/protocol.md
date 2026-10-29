# Research protocol v1

Recorded before implementation on 2026-10-29. This is the public experimental specification approved by the project owner. Revisions must state their reason and whether any relevant evaluation evidence was already inspected. The initial issue is #1.

## Questions and hypotheses

Can a reusable policy choose legal compiler transformations that improve total compilation plus execution cost? Can experience pretraining and limited online adaptation improve new-program choices relative to equal-budget simpler search?

H1: compilation cost and reuse change preferred schedules. H2: useful generalization to unseen shapes. H3: pretraining reduces new measurements. H4: GRPO's advantage over warm PPO and frozen sampling is uncertain. H5: tuning does not repay its cost for every workload. H6: inaccurate reuse hints and unfamiliar structures can hurt. These are hypotheses, not findings or predictions attributed to Hassan.

## Related work

- [MLGO](https://research.google/blog/mlgo-a-machine-learning-framework-for-compiler-optimization/): LLVM learned decisions.
- [MLIR RL](https://arxiv.org/abs/2409.11068): RL over MLIR loop transformations.
- [Pearl](https://arxiv.org/abs/2506.01880): transformation sequences, representations and generalization.
- [Offline imitation from multiple baselines](https://mlforsystems.org/assets/papers/neurips2024/paper30.pdf): learning from mixed experience.
- [WarmTuner](https://arxiv.org/abs/2607.25831): pretraining then GRPO compiler-flag adaptation.
- [CompilerDream](https://arxiv.org/abs/2404.16077): model-based compiler optimization.

Tasks, machines and metrics differ. Do not transfer speedups or claim novelty for using MLIR, RL or GRPO.

## System contract

Ubuntu 24.04 under WSL2, Intel Core Ultra 9 275HX; LLVM/MLIR 18.1.3 (record complete Debian package versions), C++17/CMake, Python 3.12, SB3 and SB3-Contrib initially 2.9.2 with a resolved lock before training. One consistent x86-64-v3 target. CPU training initially; benchmark activity separated. No Arm claims.

Restricted static-shaped Linalg kernels with contiguous non-aliasing buffers, strict FP64 and explicit output reset semantics. Families: matmul, matvec, row reduction, 2D elementwise, 2D stencil, direct convolution. Establish matmul/matvec first. FP32 is outside v1.

C++ components: input validation, structural features, schedule application, fixed LLVM lowering, verification and IR dumps. Actions STOP, interchange of eligible parallel dimensions, tile one dimension by 4/8/16/32/64, masked vectorization when legal. VECTORIZE terminates selection. Never reorder reductions. At most six transformation actions, two tile levels per original dimension, second smaller, vector volume at most 1024. Stable logical dimension identities and state-dependent legal masks. Clone before mutation and publish only verified transformations. Shared target, optimization level and numeric settings for every method.

Persistent C++ ExecutionEngine/LLVM JIT: owned callable handles, shape/dtype/layout/alias guards, 32-entry LRU compiled cache, no eviction of code still held by invocations, structured errors. Compiled identity covers kernel/shape/dtype/schedule/target/compiler; decision cache adds policy version and reuse. JSON-lines interface for load/apply/compile/validate/benchmark/invoke/release; diagnostics on stderr. Primary deployment is an already initialized service. Startup/model load measured separately.

## Learning contract

Underlying state is transformed IR plus action history. Observation includes bounds/iterator types, ranks/access strides, arithmetic/load/store counts, tile/vector state, history/remaining action count and log reuse. Exclude identifiers, family/split labels, held-out times and oracle actions. Features may be partially observable.

Episode: choose, stop, compile, verify, measure. Terminal reward, gamma=1. E(s,N)=C(s)+N*T(s), R=log(E_reference/E_candidate). C counts policy selection, transformation, compile and install. T counts dispatch and execution. Record research validation separately, include deployed checks. No extra action-count penalty. Incorrect output receives no accepted speed reward; repeated unexpected failure stops the campaign.

### PPO

Execution-only and reuse-aware MaskablePPO. MLP 128x128, lr 3e-4, clip .2, GAE .95, entropy .01, rollout 256, batch 64, epochs 10, grad norm .5. Seeds 7/17/27/37/47. Select checkpoints using validation, report all seeds. Freeze any revised defaults before final evaluation.

### Imitation pretraining

Mix fixed/heuristic, random legal, PPO and intermediate trajectories. Save observations, masks, actions, behavior probabilities when available, next observations, termination and outcome. Recheck near-tied winners. Imitate best verified demonstrations per training program/reuse; balance programs/reuse rather than long/frequent trajectories. Same actor, lr 3e-4, batch128, at most100epochs, validation patience10. This is supervised imitation, not IQL or full offline RL. Frozen pretrained actor is mandatory baseline.

### Adaptation

Reset actor independently for every program/reuse context. Sample groups of8 complete schedules. Evaluate budgets8/16/32; update only when another group remains. Return best verified observed candidate. At K=8 no update can improve already sampled candidates. Same initial actor, representation, action space and budgets for GRPO/PPO.

GRPO: within-group normalized reward, clipped sequence policy objective, clip .2, lr1e-4, four epochs, grad norm .5, KL .02 to initial actor, no critic. Skip effectively zero-variance groups. Document sequence normalization and differences from reference GRPO. Log duplicates, diversity, spread, KL and update time. Warm PPO critic fitted using training trajectories only. Both consume identical complete trajectory groups; controller enforces exact budget, never generic train-call rollout overshoot.

Separate tuning and fresh deployment cache. E_adapt=D_tuning+C_deployment+N*T_deployment. Charge all trial/update elapsed costs and fresh deployment compile. Report final kernel quality and complete cost. Duplicates consume request slots; within-method measured results can be cached with actual time/hits recorded. No held-out measurement sharing across methods or cross-program learning.

IQL, graph observations, world models, adaptive trial count and LLM post-training are later extensions. STOP ends a schedule, not the adaptation campaign.

## Evaluation

Freeze/hash manifest before full training: 128 train +32 validation +32 unseen-shape workloads from first four families;32 unseen-structure workloads from stencil/conv. Group near-duplicates. Correctness can cover supported operations, but all held-out performance sealed until core and advanced methods frozen. Label interim validation as development. N=1/100/10000; pilot may lower10000 to1000 solely for budget before training, uniformly. Test hints four times high/low.

Zero-exploration methods: fixed pipeline, best fixed training schedule, locality/vector heuristic, supervised C/T predictor, execution-only PPO, reuse-aware PPO, frozen imitation. Predictor gets same measurement pool and deterministic256 legal candidates; charge generation/inference. Adaptation controls: frozen policy best-of-K, warm PPO, GRPO, random legal, mutation initialized from heuristic/pretrained choices. Same requests, disclose wall time. Diagnostic subset up to512 legal schedules is best observed, not an oracle. Single-thread OpenBLAS matmul is external context.

Independent references: known/zero/unit/odd/remainder/rectangular, deterministic random seeds, signs/cancellation, bounded large/small finite values, repeated invocation/output reset, invalid layouts/alias, cache hits/eviction/lifetime. Start FP64 atol1e-10/rtol1e-9, validate before training, never loosen for fast failing code. Verify IR after every transformation; inspect representative assembly.

Learning tests: zero invalid-action probability, termination, hand-checked reward examples, exact requests, independent adaptation resets, zero-variance groups, checkpoint identity, learnable known-choice diagnostic.

Timing: one worker, no concurrent training/build/benchmarks, fixed guest affinity with WSL physical isolation limitation. Record selection/transform/compile/install, first call, steady dispatch+execution, complete N-call batches, warm-cache, updates, startup. Training proxy C+NT allowed; final total claims require complete measured batches. Steady batches at least100ms; five independent final rounds, seeded interleaving. Anchor drift>5%: retain/flag/repeat once, disclose continuing instability. No selective deletion of bad times.

Each record: revision/executable hash, kernel/shape/inputseed/reuse, schedule/IR hashes, compiler/target/machine/OS, correctness, raw timings/costs, policy/seed/cache, failures/timeouts. Report per-workload/family, equal family aggregate, grouped uncertainty for related shapes. Include latency vs reuse, runtime speedup and compile cost, improvement vs trials/wall time, break-even reuse, unseen shapes/structures, seeds/regressions/failures, training and campaign cost, mechanistic success/failure.

Broad superiority requires >=5% aggregate improvement over strongest relevant non-RL validation-selected baseline, uncertainty supporting positive difference, fresh confirmation, disclosed regressions. Smaller/uncertain differences are inconclusive. No success-only aggregation.

## Gates, budget and governance

A environment/JIT; B two-family compiler/runtime; C training-only landscape; D corpus/PPO; E imitation; F adaptation; G sealed study; H release. Initial estimate10-20focused days plus measurement, revised by pilot. Repair toolchain before scope expansion. Equivalent machine code requires pipeline inspection. Fixed dominance requires action/workload investigation. If compile cost doesn't matter revise hypothesis before final test. Debug diagnostic before new algorithms. GRPO vs frozen null result and tuning break-even are valid results. Preserve bugs/evidence; affected final runs exploratory, new confirmation needs revised frozen protocol.

Local only, no paid cloud.100 distinct pilot measurements estimate allocation of48hardware hours including validation/final. Initial max4096distinct kernel/schedule requests per PPO run, uniformly reduce before training if budget requires. Track logical/distinct/cachehits/compiles/executions/training and total time. Resumable bounded queues; repeated correctness failure stops unattended work. Overnight only a verified fixed stage.

Issue before code, protocol first commit. Coherent commits with explanations, C++ RAII, explicit configs/errors, versioned records, separate compiler/runtime/learning/orchestration/reporting. CI deterministic correctness only. Private plan/decision/experiment/code-reading/AI-use records outside repository. Completion means reproducible build, visible transformation, JIT execution, train/load policy, frozen/adaptation comparisons, raw evidence, and clear attribution to built/reused/AI-assisted components.
