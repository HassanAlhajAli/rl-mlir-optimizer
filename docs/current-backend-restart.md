# Current-backend pilot and replacement campaign revision

Recorded on 2026-10-29 before any current-backend optimization pilot or
replacement PPO timing. Earlier development performance, including the old
100-candidate pilot and a current-backend reference/heuristic regression, has
been inspected. Held-out performance remains sealed. This is an explicit
revision to the development sequence and allocation, not a claim that the
original row-reduction SIGSEGV has been explained.

## Stability decision

The [long-lived request-path diagnostic](../results/compiler-stability-003/observations.md)
completed 833 records, 13,271 successful worker requests and 599,529,512
invocations within its limits. The originally failed compile succeeded. Earlier
ordinary and guarded correctness replays also passed. These results justify a
small, serial current-backend pilot with a stop on unexpected compiler or
numerical failure. They do not prove the compiler will remain stable during
new training. All future measurement entry points retain their worker timeout,
memory bound and exclusive lease. No failure becomes a reward or is retried
until diagnosed.

## Frozen pilot

Use the byte-identical `configs/pilot.json` (`SHA-256
2e776657195e32ce31d0b44a53e7aa1c5f057579849b3297b5b3dc8989f457bc`)
with `scripts/optimization_pilot.py`, the current worker executable and fresh
`runs/optimization-pilot-003`. This is the same 20 matmul/matvec shapes, five
literal schedules and 100 distinct candidates as the older pilot, with three
independent correctness inputs, five interleaved timing rounds, anchor drift
checks and retained repeats. The identical design makes backend changes easier
to inspect; it does not cover row reduction or elementwise performance. The
script refuses an existing output and tracked source edits. A 1,200-second
external wall ceiling interrupts the run; charge actual elapsed time, including
an interrupted run, to the 48-hour allowance.

Run only with no other measurement, build, test or training process active:

```sh
timeout -s INT -k 130s 1200s .venv/bin/python -u scripts/optimization_pilot.py \
  --config configs/pilot.json --output runs/optimization-pilot-003
```

Audit all 100 numerical outcomes, compilation failures, original and repeated
drift blocks, compiler/version hashes, per-workload timing and estimated
`C + N*T` choices for N=1/100/10000. Do not transfer old-backend timing or
combine the two pilots into one estimate. The pilot is descriptive and selected
on its own development data; it cannot establish a final deployment advantage.
If the pilot fails, stop and diagnose. If all 100 pass but compilation cost no
longer affects the estimated choice, revise the H1 interpretation before large
training. If the selected schedules or compile costs differ substantially,
state that explicitly and keep all methods on the new binary.

## Conditional replacement training allocation

After reviewing the pilot, use a **fresh** ten-job queue at a new output path.
Both PPO objectives and seeds 7/17/27/37/47 remain required. The per-job caps
stay at 1,536 distinct outcomes including references, 8,192 logical candidate
requests, 16,384 requested action steps and 45 minutes. Keep the serial queue
ceiling at eight hours. These are newly allocated current-backend resources
within the original cumulative 48-hour allowance; they do not replenish or
resume the old job's counters. Preserve all old checkpoints and costs as
historical evidence. Do not import old timing caches or claim old checkpoints
are compatible with the changed compiler. A job reaching a cap is reported as
such; a compiler failure stops the queue and all later jobs.

Before this pilot, the ledger charges 17,052.89 seconds including the stopped
campaign, both prior replays, this diagnostic and the conservative 7,200-second
development reserve. Reserve at most 1,200 seconds for the pilot, 28,800 for
the conditional replacement queue and 86,400 for adaptation/final work. This
leaves 39,347.11 seconds unallocated within 172,800 seconds. Reservations
are ceilings, not spent time. Replace each ceiling with actual elapsed usage
when the stage ends and reassess the remaining budget before starting the next.

This revision changes the backend boundary and fresh resource allocation.
It does not change the action vocabulary, objectives, reward, validation split,
five-seed requirement, final statistical rules or the sealed held-out set.
Separate validation checkpoint selection and all baseline/adaptation method
freezes remain prerequisites for final evaluation.
