# Protected-memory replay of the interrupted job

This correctness-only diagnostic at revision `730e7d7` replayed all 833 saved
training records from the interrupted runtime/17 job, including its failed
row-reduction input. The main PPO campaign remains stopped.

For every candidate and anchor, both input seeds ran with buffers aligned first
against the leading guard page and then against the trailing guard page. Inputs
were read-only; output padding was checked after each of three invocations.
The worker retained its 4 GiB address-space limit and 120-second request timeout;
the overall diagnostic cap was 1,200 seconds. No performance reward was recorded.

- All 833 records and 6,664 numerical comparisons passed.
- 1,538 actual compilations and 19,992 guarded invocations completed.
- All 1,666 candidate/anchor LLVM hashes matched the preceding ordinary replay.
- Recorded elapsed time was 178.78 seconds; peak worker RSS was 422,716 KiB.

The matching LLVM hashes verify that this diagnostic tested the same generated
IR as `compiler-replay-002`, with different buffer placement and protection.
It does not establish the cause of the original SIGSEGV or rule out failures
inside compiler libraries, JIT temporaries, or untested execution histories.
The original failure and every earlier repair attempt remain preserved.

The guard itself was tested using deliberately faulty native functions in
isolated child processes: out-of-bounds reads/writes, writes to read-only
inputs, and output-padding overwrites were detected. Valid repeated and empty
calls passed. The additional-family suite also checked real generated kernels
with empty dimensions, odd sizes and remainders in both guarded placements.
The five CTest suites and original 17 Python checks passed when the diagnostic
was introduced. Fourteen targeted adaptation/predictor checks then passed after
the seven new provenance tests were added.

Separately, measured adaptation previously lacked a compiler-identity check
when loading a policy. Commit `e3f4298` requires matching compiler executable,
corpus, observation, measurement protocol, compiler release and target, and
applies the same contract to pretraining/predictor experience pools. A command
using the old integration policy was rejected before creating an output run
or launching a worker. The two previously published adaptation experiments
were checked and already used matching compiler identities.

[audit.json](audit.json) contains the replay hashes and counters.
[raw.zip](raw.zip) preserves the per-record checks, metadata and diagnostic/test
logs; [manifest.json](manifest.json) hashes every member. Reproduce with:

```sh
.venv/bin/python scripts/replay_correctness.py \
  --records runs/ppo-campaign-001/ppo-runtime-17/measurements/measurements.jsonl \
  --output runs/new-guarded-replay --seconds 1200 --guard-memory
```

This result is compiler/runtime validation, not evidence of an RL performance
advantage. No new training budget or held-out performance access was used.
