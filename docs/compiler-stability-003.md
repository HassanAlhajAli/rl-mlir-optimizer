# Bounded long-lived worker diagnostic

Recorded before execution on 2026-10-29. The original row-reduction SIGSEGV is
unexplained. The 833-record ordinary and guarded replays established numerical
and buffer checks, but skipped the original calibration and benchmark request
path. This diagnostic addresses that gap; it does not generate training rewards
or update the optimization landscape.

`scripts/replay_measurement_history.py` uses the saved ordered runtime/17
measurement file, one pinned current-backend worker and one guest CPU. For each
of the 832 accepted records, it clears the cache, compiles the candidate,
checks the original three invocation calls, replays every saved candidate and
anchor calibration count, then replays each saved anchor/candidate/anchor timing
triplet and releases both handles. The final record stops after its candidate
compile, the point where the earlier worker died. It does not silently resume
PPO, reuse old timing caches or accept a failed compilation as a low reward.

Every request has a flushed start and completion/failure entry with record key,
phase, command, count, elapsed time and worker virtual/resident/peak memory.
Worker stderr is saved separately. The script stops at the first numerical,
worker or deadline failure, with a 120-second per-request worker timeout, 4 GiB
address-space limit and 1,800-second overall ceiling. The existing exclusive
measurement lease prevents overlap. Charge actual elapsed time to the cumulative
48-hour ledger; do not count this ceiling as spent time.

The saved log does not reconstruct PPO policy computation, `analyze` interleaving,
exact machine state or the old compiler binary. Historic counts are replayed
on the repaired backend, so their new batch times are diagnostic only. A pass
shows this bounded current-backend request history completed. It cannot identify
the original SIGSEGV's cause or prove future compiler stability. A failure
requires saving its per-request evidence and stopping further measurement until
diagnosed. If it passes, a separately frozen, smaller current-backend pilot may
start with the same stop-on-failure rule; a replacement training allocation still
requires its own explicit revision and pilot review.

Run after committing the diagnostic and checking that no other measurement is
active:

```sh
.venv/bin/python scripts/replay_measurement_history.py \
  --records runs/ppo-campaign-001/ppo-runtime-17/measurements/measurements.jsonl \
  --output runs/compiler-stability-003 --seconds 1800
```

The output directory contains `run.json`, `requests.jsonl`, `records.jsonl`
and `worker.stderr`. Its inputs and executable are identified by SHA-256.
