# Main campaign interruption and vector lowering repair

On 2026-10-29, `ppo-campaign-001` stopped at the fourth of ten jobs. Three
jobs reached their recorded limits normally. Execution-only PPO, seed 17,
lost its compiler worker to SIGSEGV during compilation of row reduction
336x205 with `TILE(1,8), TILE(0,64), TILE(0,16)`. The failing record and latest
checkpoint remain in the original run. The other six jobs were not started.
Held-out performance remains sealed. This is not a completed five-seed study.

The exact row-reduction input compiled successfully in a fresh process under
GDB. Its original failure is therefore **not yet explained**. Replaying the
compilation history exposed a separate reproducible resource problem: a
previously accepted matmul 32x128x8 schedule consumed about 17 GB RSS inside
LLVM InstCombine before the diagnostic was interrupted. That schedule had
already taken 54.8 seconds to compile in the original campaign:

```text
TILE(0,16), TILE(0,8), TILE(2,32), TILE(2,16), VECTORIZE
```

A fresh bounded reproduction exceeded 3 GiB of virtual memory. The backtrace
involves shuffle-mask construction in InstCombine. Extra canonicalization/CSE
did not fix it. Unrolling the already-created vectors helped the exact shape
but still exhausted memory on a 17x127x9 remainder case. That attempted repair
also initially broke masked-reduction regions; its failing test logs are kept.
None of these failures becomes a low reward or a silently skipped candidate.

The implemented repair constructs a small inner Linalg tile **before**
vectorization. The innermost parallel dimension gets at most four FP64 lanes,
matching the fixed AVX2 target; other dimensions get one. Reductions retain
their original order and remainder accesses retain masks. The agent's selected
outer tiles, action vocabulary, action limits and 1024-element eligibility bound
are unchanged. This is a change to VECTORIZE's implementation and to generated
code, so all new performance comparisons require the same repaired backend.
The worker reports `native-inner-tile-v2`, and binary hashes prevent reuse of
old timing caches or resuming a checkpoint as if its compiler were unchanged.

The Python worker also imposes a 4 GiB virtual-address-space limit before
accepting compilation requests. The existing 120-second request timeout still
applies. LLVM signal diagnostics and an optional `RLMLIR_DEBUG_LLVM` output
preserve evidence for future failures. These guards stop excessive resource
consumption; they do not convert failed compilations into accepted results.

Local verification: all four CTest suites passed, including 88 scheduled-kernel
cases, the exact large-vector input, odd/remainder variants, the original
row-reduction input, strict floating-point checks and executable ownership.
All 17 Python tests passed, including the child-process memory limit. One
parallel build also crashed in Clang's optimizer; a saved serial retry and the
final serial build succeeded. Its cause is not established either.

The [longer-lived correctness replay](../results/compiler-repair-002/observations.md)
completed all 833 records, with 3,332 numerical comparisons, 1,538 actual
compilations and a recorded peak RSS of about 417 MiB. It took 126.82 seconds.
The script below
recompiles every saved training record in order, including the originally
failed one, checks candidate and anchor against two independent reference
inputs, and records process memory. It deliberately does not reproduce the
original calibration counts or claim to reproduce performance:

```sh
.venv/bin/python scripts/replay_correctness.py \
  --records runs/ppo-campaign-001/ppo-runtime-17/measurements/measurements.jsonl \
  --output runs/compiler-replay-002 --seconds 1200
```

The main campaign remains stopped. Old results and checkpoints must stay
separate from any revised backend study. No request limit or hardware allowance
has been increased, and there has been no automatic restart.

The [interrupted campaign evidence](../results/ppo-campaign-001/observations.md)
preserves all ten objective/seed statuses, all 5,020 measurement records, saved
trajectories and the initial/latest checkpoints of each attempted run. A
read-only audit verified hashes, numerical acceptance, rewards, budgets and
policy updates. That audit does not select checkpoints or establish RL benefit.

## Follow-up: distinguish answer correctness from buffer safety

The [completed bounded diagnostic](../results/compiler-memory-replay-001/observations.md)
adds `--guard-memory` to the replay command and uses the fresh
`runs/compiler-memory-replay-001` output. It checks both buffer
placements against inaccessible boundary pages, read-only inputs and output
padding sentinels. Each candidate/anchor is checked with both input seeds,
both placements, and three invocations. The 1,200-second overall and existing
worker resource limits remain in force. Any failure stops this diagnostic.

The guard implementation is tested by deliberately faulty functions in isolated
child processes before replay. This work changes only the diagnostic runtime
and worker command, not the compiler transformation or timing paths. Compare
the new saved LLVM hashes to `compiler-replay-002` to verify that the generated
IR is unchanged. All 833 records and 6,664 numerical comparisons passed;
all 1,666 LLVM hashes matched. No training or performance finding follows from
this check. The original SIGSEGV remains unexplained.
