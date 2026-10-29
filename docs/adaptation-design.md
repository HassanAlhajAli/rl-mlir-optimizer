# Adaptation implementation decisions, recorded before evaluation

The controller owns the 8/16/32 request budget, samples complete groups of eight,
and updates only when another group remains. Each call copies the original
actor and uses a new measurement cache. The incumbent can only be a verified
candidate actually requested by that method. A duplicate consumes a slot.

Rewards for adaptation are `-log(C + selection + installation + N*T)`, in
milliseconds. Subtracting a constant reference log cost has no effect on GRPO
group advantages. This removes an otherwise unbudgeted reference evaluation.
The PPO adaptation critic is fitted to this same absolute log-cost return on
training trajectories, not to the reference-relative PPO training reward.

GRPO uses the population standard deviation within the group plus 1e-8, and
skips groups with standard deviation below 1e-8. Each action receives its
sequence's normalized advantage. We average the clipped token/action surrogate
and the exact categorical KL across actions within each sequence, then average
across eight sequences. This gives each schedule equal loss weight regardless
of length. It uses per-action probability ratios, not the product of ratios for
an entire schedule. This is our explicitly defined GRPO variant for short
compiler schedules, not a claim to reproduce an LLM training implementation.

KL is `KL(current || initial actor)` on the sampled observations with the same
legal masks. The old behavior log probabilities are fixed for all four epochs.
The initial actor remains fixed throughout a context. There is no value loss in
GRPO. PPO uses a separate critic, terminal rewards, gamma=1, GAE=.95, a clipped
policy loss, value coefficient .5 and entropy coefficient .01. Both adaptations
start at learning rate1e-4, clip.2, four epochs and gradient norm.5.

Frozen sampling, GRPO and PPO use the same random seed and first group. At K=8
they therefore request the same candidates before any update. Real timings are
measured independently for each method, so measured winners can differ due to
noise. This control must never be described as evidence of a learning benefit.

Diagnostic tests of the update mechanics and a small development run do not
constitute the five-seed compiler study. Held-out performance remains sealed.

## Compatibility before an experiment

Measured adaptation requires the policy's compiler executable hash, corpus hash,
observation version, measurement protocol, compiler release and CPU target to
match the current experiment. Missing fields are rejected. The same check
applies to experience pools used by imitation and the supervised predictor.
Documentation-only source revisions may differ when this complete contract is
unchanged. Cross-compiler transfer may be studied later under an explicit
protocol; it is not an implicit fallback after a compiler repair.
