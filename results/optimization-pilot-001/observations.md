# Training-only pilot observations

Exploratory measurements on one Intel/WSL2 laptop. No policy was trained.

All 100 candidates passed independent output checks. 25/100 original timing blocks were flagged; 6/25 repeat blocks remained flagged.

The table uses medians of all five original blocks, including flagged blocks. Repeats remain in raw records. Best-observed schedules are chosen on these development observations, so their ratios are descriptive and selection-biased, not confirmatory gains.

| Workload | Baseline us | Lowest median among five | Descriptive ratio | Estimated choices N=1/100/10000 |
|---|---:|---|---:|---|
| matmul-127x63x33 | 24.971 | row_tile4 | 1.01x | baseline / baseline / baseline |
| matmul-128x32x64 | 60.402 | baseline | 1.00x | baseline / baseline / baseline |
| matmul-16x16x16 | 0.404 | baseline | 1.00x | row_tile16 / row_tile16 / baseline |
| matmul-31x65x17 | 2.514 | row_tile16 | 1.00x | baseline / baseline / baseline |
| matmul-32x128x8 | 2.820 | row_tile16 | 1.00x | row_tile16 / row_tile16 / row_tile16 |
| matmul-32x32x32 | 2.136 | baseline | 1.00x | row_tile4 / row_tile4 / baseline |
| matmul-48x96x24 | 9.635 | row_tile16 | 1.01x | baseline / baseline / baseline |
| matmul-64x64x64 | 58.110 | baseline | 1.00x | baseline / baseline / baseline |
| matmul-64x8x32 | 3.462 | row_tile4 | 1.03x | baseline / baseline / baseline |
| matmul-8x64x128 | 19.219 | parallel_tile8_vector | 1.26x | row_tile16 / row_tile16 / parallel_tile8_vector |
| matvec-128x256 | 11.594 | parallel_tile16_vector | 2.96x | baseline / baseline / parallel_tile8_vector |
| matvec-128x32 | 0.589 | parallel_tile16_vector | 1.24x | row_tile4 / row_tile4 / row_tile4 |
| matvec-16x128 | 0.535 | parallel_tile8_vector | 2.46x | parallel_tile16_vector / parallel_tile16_vector / parallel_tile8_vector |
| matvec-256x256 | 23.668 | parallel_tile16_vector | 2.21x | baseline / baseline / parallel_tile8_vector |
| matvec-256x64 | 3.964 | parallel_tile16_vector | 1.57x | baseline / baseline / parallel_tile8_vector |
| matvec-31x65 | 0.500 | row_tile4 | 1.03x | baseline / baseline / baseline |
| matvec-512x128 | 17.037 | parallel_tile16_vector | 2.03x | baseline / baseline / parallel_tile8_vector |
| matvec-512x16 | 1.118 | parallel_tile8_vector | 1.02x | row_tile4 / row_tile4 / baseline |
| matvec-64x128 | 2.446 | parallel_tile16_vector | 2.38x | baseline / baseline / parallel_tile8_vector |
| matvec-8x64 | 0.118 | parallel_tile16_vector | 1.96x | parallel_tile8_vector / parallel_tile8_vector / parallel_tile8_vector |

The C+N*T choice estimates include measured transformations/backend compilation but no learned policy selection, which does not exist yet. Compile cost was measured once per candidate, and complete deployment batches were not measured. These estimates cannot establish H1 yet.

Reproduce with `python3 scripts/summarize_pilot.py <run-directory>`. Source revision, protocol hash, executable hash, package versions, affinity, all calibration samples and every block are retained beside this report.
