# Training-only pilot observations

Exploratory measurements on one Intel/WSL2 laptop. No policy was trained.

All 100 candidates passed independent output checks. 6/100 original timing blocks were flagged; 1/6 repeat blocks remained flagged.

The table uses medians of all five original blocks, including flagged blocks. Repeats remain in raw records. Best-observed schedules are chosen on these development observations, so their ratios are descriptive and selection-biased, not confirmatory gains.

| Workload | Baseline us | Lowest median among five | Descriptive ratio | Estimated choices N=1/100/10000 |
|---|---:|---|---:|---|
| matmul-127x63x33 | 21.724 | row_tile4 | 1.00x | baseline / baseline / baseline |
| matmul-128x32x64 | 58.313 | parallel_tile8_vector | 1.28x | baseline / baseline / parallel_tile8_vector |
| matmul-16x16x16 | 0.222 | row_tile16 | 1.01x | row_tile16 / row_tile16 / row_tile16 |
| matmul-31x65x17 | 2.706 | baseline | 1.00x | baseline / baseline / baseline |
| matmul-32x128x8 | 2.632 | baseline | 1.00x | row_tile16 / row_tile16 / row_tile16 |
| matmul-32x32x32 | 2.874 | baseline | 1.00x | baseline / baseline / baseline |
| matmul-48x96x24 | 9.220 | row_tile16 | 1.01x | baseline / baseline / baseline |
| matmul-64x64x64 | 55.670 | row_tile16 | 1.00x | baseline / baseline / baseline |
| matmul-64x8x32 | 3.095 | baseline | 1.00x | baseline / baseline / baseline |
| matmul-8x64x128 | 20.201 | parallel_tile8_vector | 1.55x | row_tile16 / row_tile16 / parallel_tile8_vector |
| matvec-128x256 | 11.231 | parallel_tile8_vector | 2.77x | baseline / baseline / parallel_tile8_vector |
| matvec-128x32 | 0.466 | parallel_tile16_vector | 1.02x | row_tile4 / row_tile4 / row_tile4 |
| matvec-16x128 | 0.561 | parallel_tile8_vector | 2.48x | parallel_tile8_vector / parallel_tile8_vector / parallel_tile8_vector |
| matvec-256x256 | 22.586 | parallel_tile8_vector | 2.34x | baseline / baseline / parallel_tile8_vector |
| matvec-256x64 | 3.679 | parallel_tile16_vector | 1.90x | baseline / baseline / parallel_tile8_vector |
| matvec-31x65 | 0.448 | row_tile16 | 1.01x | baseline / baseline / baseline |
| matvec-512x128 | 17.776 | parallel_tile16_vector | 2.00x | baseline / baseline / parallel_tile8_vector |
| matvec-512x16 | 0.987 | baseline | 1.00x | row_tile4 / row_tile4 / baseline |
| matvec-64x128 | 2.248 | parallel_tile16_vector | 2.48x | baseline / baseline / parallel_tile8_vector |
| matvec-8x64 | 0.111 | parallel_tile8_vector | 1.99x | parallel_tile16_vector / parallel_tile16_vector / parallel_tile16_vector |

The C+N*T choice estimates include measured transformations/backend compilation; no learned selector was used in this pilot. Compile cost was measured once per candidate, and complete deployment batches were not measured. These estimates cannot establish H1 yet.

Reproduce with `python3 scripts/summarize_pilot.py <run-directory>`. Source revision, protocol hash, executable hash, package versions, affinity, all calibration samples and every block are retained beside this report.
