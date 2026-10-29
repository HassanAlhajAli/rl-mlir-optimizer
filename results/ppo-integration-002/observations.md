# Measured PPO integration, development only

After the stack-allocation repair, seed7 completed512 action steps and144
complete compiler episodes without a rejected numerical result or worker
crash. It requested144 candidates and144 reference outcomes;219 distinct
measurements were made and69 requests reused their own saved measurements.
Total evaluator elapsed time was317.997seconds, including calibration,
independent correctness checks, compiler activity and model learning.

This run exercises real MLIR transformations, machine-code generation, native
execution, legal masks, terminal measured rewards and PPO updates. It does not
establish that the policy is useful on unseen programs or better than simpler
choices. There is only one seed, two rollouts and no validation selection.
Rollout rewards were mostly negative relative to the fixed reference.

`run.json` preserves the source/executable hashes and counters.
`measurements.jsonl` contains accepted timing records including flagged drift.
`trajectories.zip` contains complete observations, masks, actions, behavior
probabilities and outcomes. `policy.pt` is a portable development checkpoint;
`policy.json` records its hash, optimizer epochs and changed parameter tensors.
Compressed IR artifacts and original SB3 checkpoints remain in the local run
directory. The published checkpoint is not selected as the final policy.

This run uses measurement protocol paired-training-v1. Subsequent v2 recording
includes the native compile wrapper and owned-buffer installation in C, plus a
separate first-invocation sample. Do not merge old and new costs silently.

Reproduce on the recorded revision:

```sh
.venv/bin/python scripts/train_ppo.py --output runs/ppo-integration-reproduction --seed 7 --objective reuse --max-new 360 --max-requests 512 --hours 0.5 --timesteps 512
```

Exact timing is machine-dependent. Reproduction on a later revision is a new
experiment with its own source and executable hash.
