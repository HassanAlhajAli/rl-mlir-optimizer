# Training-only pilot observations

Exploratory measurements on one Intel/WSL2 laptop. No policy was trained.

All 100 candidates passed independent output checks. 12/100 original timing blocks were flagged; 4/12 repeat blocks remained flagged.

The table uses medians of all five original blocks, including flagged blocks. Repeats remain in raw records. Best-observed schedules are chosen on these development observations, so their ratios are descriptive and selection-biased, not confirmatory gains.

| Workload | Baseline us | Lowest median among five | Descriptive ratio | Estimated choices N=1/100/10000 |
|---|---:|---|---:|---|
| matmul-127x63x33 | 22.747 | row_tile4 | 1.01x | baseline / baseline / row_tile4 |
| matmul-128x32x64 | 53.689 | parallel_tile16_vector | 1.77x | baseline / baseline / parallel_tile16_vector |
| matmul-16x16x16 | 0.376 | baseline | 1.00x | baseline / baseline / baseline |
| matmul-31x65x17 | 2.476 | baseline | 1.00x | baseline / baseline / baseline |
| matmul-32x128x8 | 1.579 | baseline | 1.00x | row_tile16 / row_tile16 / row_tile16 |
| matmul-32x32x32 | 2.669 | row_tile4 | 1.15x | baseline / baseline / row_tile4 |
| matmul-48x96x24 | 6.951 | row_tile16 | 1.20x | baseline / baseline / row_tile4 |
| matmul-64x64x64 | 53.614 | parallel_tile8_vector | 1.58x | baseline / baseline / parallel_tile8_vector |
| matmul-64x8x32 | 3.007 | parallel_tile8_vector | 1.41x | baseline / baseline / baseline |
| matmul-8x64x128 | 19.499 | parallel_tile16_vector | 1.63x | row_tile4 / row_tile4 / parallel_tile8_vector |
| matvec-128x256 | 10.851 | parallel_tile8_vector | 2.60x | baseline / baseline / parallel_tile8_vector |
| matvec-128x32 | 0.419 | baseline | 1.00x | row_tile4 / row_tile4 / row_tile4 |
| matvec-16x128 | 0.537 | parallel_tile8_vector | 2.36x | baseline / baseline / parallel_tile16_vector |
| matvec-256x256 | 21.574 | parallel_tile16_vector | 2.29x | baseline / baseline / parallel_tile8_vector |
| matvec-256x64 | 3.518 | parallel_tile8_vector | 1.85x | baseline / baseline / parallel_tile8_vector |
| matvec-31x65 | 0.430 | row_tile4 | 1.01x | baseline / baseline / baseline |
| matvec-512x128 | 17.438 | parallel_tile8_vector | 1.91x | baseline / baseline / parallel_tile8_vector |
| matvec-512x16 | 1.028 | parallel_tile8_vector | 1.01x | row_tile4 / row_tile4 / baseline |
| matvec-64x128 | 2.149 | parallel_tile8_vector | 2.15x | baseline / baseline / parallel_tile8_vector |
| matvec-8x64 | 0.113 | parallel_tile8_vector | 1.98x | parallel_tile8_vector / parallel_tile8_vector / parallel_tile8_vector |

The C+N*T choice estimates include measured transformations/backend compilation; no learned selector was used in this pilot. Compile cost was measured once per candidate, and complete deployment batches were not measured. These estimates cannot establish H1 yet.

Reproduce with `python3 scripts/summarize_pilot.py <run-directory>`. Source revision, protocol hash, executable hash, package versions, affinity, all calibration samples and every block are retained beside this report.

## Current-backend review

This run used `native-inner-tile-v2` at revision `cecbfa3`, the frozen pilot
configuration, and 337.81 seconds of local hardware time. All 20 workloads
and 100 distinct candidate compilations completed with correct outputs on
input seeds 7, 17 and 27. Twelve of 100 original blocks were flagged; four
repeat blocks remained flagged, affecting matmul 32x128x8, matvec 512x128 and
matvec 8x64. Every flag and repeat remains in `records.jsonl`; the table above
does not select a cleaner attempt. Guest affinity does not isolate a physical
CPU under WSL.

Estimated `C + N*T` choices differ across N=1/100/10000 on 13 of 20 workloads.
That remains true on 12 of the 17 workloads without unresolved drift. This is
development evidence that compilation cost can affect the *estimated* choice
on the repaired backend, not a complete-batch confirmation of H1. The lowest
median runtime schedule differs from the older-backend pilot on 13 of 20
workloads, and the estimated choice sequence differs on nine. These are
separate development runs with different generated code and measurement noise;
their timings are not pooled or treated as a controlled speedup comparison.

The raw record, summary, configuration, metadata and pilot log are beside this
report. [artifacts.zip](artifacts.zip) preserves every IR, object and assembly
file; [artifact-manifest.json](artifact-manifest.json) hashes all 600 members.
No held-out program, PPO checkpoint or learned selector was evaluated here.
