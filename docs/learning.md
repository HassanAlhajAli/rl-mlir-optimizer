# Learning implementation and boundaries

The frozen corpus contains128 training,32 validation,32 unseen-shape and32 unseen-structure workloads. Family/shape neighborhoods are grouped around16-element anchors; pilot groups occur only in training. The manifest hash is checked on load. Held-out measurement access is rejected by the current training evaluator. Correctness-only tests cover all six families.

The observation is235 bounded numerical features: four loop descriptions, stable logical order, access strides, buffer shapes/ranks, arithmetic counts, structured-op load/store counts, vector/step state, log reuse and six action-history slots. It contains no program IDs or split/family labels. There are28 categorical actions: STOP, VECTORIZE,20 tile choices and six parallel interchange pairs. The compiler, not a mirrored Python model, supplies legal actions. Version2 counts used scalar operands, including a read accumulator and excluding unused inputs; old version1 checkpoints must use their recorded source revision.

The environment reparses/replays a prefix to obtain actual transformed state. This is a simple initial implementation, not an optimized incremental compiler session. Its measured analysis/request time is charged as selection overhead. PPO policy-forward time is separately charged through hooks; this includes the actor/value forward implementation used in this development run. Training uses measured C+N*T estimates and terminal log relative cost; final deployment conclusions still require whole native batches with the deployed selector.

The evaluator runs synchronously with model updates: there is no training/build activity during a timing call. It validates independent reference inputs, calibrates >=100ms batches, brackets measurements with baseline anchors and retains flagged/repeated samples. Timing uncertainty remains visible. Compressed IR artifacts, raw timing blocks, compilation costs, actual compiler/invocation counts and failed attempts are stored. New measurement, logical request and wall-time limits are enforced before invoking additional compilation. Baseline calibration is counted as actual hardware work, even when it is not a candidate request.

Cache keys include the executable hash, source, schedule and measurement protocol. Cached measurements do not imply zero deployment compilation cost. Model seeds and policy probabilities are retained in complete trajectories. An exhausted budget or failure saves the latest policy and stop reason. A checkpoint saved before any optimizer update is not a trained result. Rollout-boundary behavior snapshots make PPO experience inspectable.

Install CPU training dependencies in an isolated Ubuntu environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pytest tests/test_learning.py -q
```

The dependency lock records the versions actually installed, including SB3/SB3-Contrib2.9.2. The known-choice diagnostic must learn different actions for different contexts, assign zero probability to an illegal action, and restore identical checkpoint weights.

First bounded integration run (development only):

```sh
.venv/bin/python scripts/train_ppo.py --output runs/ppo-integration-001 --seed 7 --objective reuse --max-new 160 --max-requests 256 --hours 0.5 --timesteps 512
```

The first run stopped before learning because of stack growth in generated code. The repaired second run completed512 steps,144 episodes and20 optimizer epochs; its checkpoint and evidence are under `results/ppo-integration-002`. This is not the five-seed study and does not select a final checkpoint.

Balanced imitation and training-only critic fitting are implemented in `imitation.py`. The group controller and gradient updates in `adaptation.py` implement frozen sampling, GRPO, warm PPO, random legal sampling and prefix-mutation search. Nine Python checks cover learning, weights, exact budgets, reset isolation and zero-variance updates. These mechanics need measured comparisons after the demonstration dataset is built. The supervised cost predictor, complete baseline evaluation and sealed final study remain outstanding.

`collect_experience.py` records fixed, heuristic and random trajectories. `pretrain.py` accepts only matching compiler executables and training/validation data, remeasures the best two histories plus all candidates within5% of the current leader, and retains near-tie and drift flags. Actor contexts are equally weighted; critic fitting draws one trajectory/state per training context per epoch. Validation selects the actor by demonstration likelihood, with measured deployment validation still required before final release. Pretraining is supervised learning, not offline RL.

`run_adaptation.py` starts independent actors and measurement caches, enforces exact candidate budgets and measures a fresh native deployment of the selected schedule. Warm PPO requires a critic trained on absolute negative log cost. The detailed update definition is in `adaptation-design.md`. The script currently accepts development splits only. A single development run is not the final five-round study.

Measurement protocol v2 includes native compile setup and owned-buffer installation in C and records first invocation separately. Original v1 measurements remain archived with their original hashes and boundaries.
