# Learning implementation and boundaries

The frozen corpus contains128 training,32 validation,32 unseen-shape and32 unseen-structure workloads. Family/shape neighborhoods are grouped around16-element anchors; pilot groups occur only in training. The manifest hash is checked on load. Held-out measurement access is rejected by the current training evaluator. Correctness-only tests cover all six families.

The observation is235 bounded numerical features: four loop descriptions, stable logical order, access strides, buffer shapes/ranks, arithmetic counts, nominal structured-op load/store counts, vector/step state, log reuse and six action-history slots. It contains no program IDs or split/family labels. There are28 categorical actions: STOP, VECTORIZE,20 tile choices and six parallel interchange pairs. The compiler, not a mirrored Python model, supplies legal actions.

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

This is not the five-seed study and does not select a final checkpoint. Imitation, GRPO, warm-PPO adaptation, supervised/search baselines and sealed final evaluation remain separate required stages. Do not describe them as implemented merely because they appear in the research protocol.
