# Stack allocation repair before PPO training

The first measured PPO integration attempt stopped after 95 actions and 25
completed episodes, before its first 256-step update. It is not a trained model
result. The worker crashed while invoking a verified-IR schedule for matmul
96x78x98: TILE(axis1,4), TILE(axis2,16), TILE(axis2,8), VECTORIZE.

The debugger stopped at exhausted stack memory. Optimized LLVM IR contained
fixed-size `alloca` instructions inside the innermost reduction loop. The
vector-to-SCF conversion had introduced temporary buffers whose allocation
repeated on every iteration, with release only when the function returned.
IR verification alone does not establish safe resource usage.

The shared backend now runs upstream `buffer-loop-hoisting` immediately after
`convert-vector-to-scf`, before LLVM lowering. This moves eligible temporaries
outside the loop while retaining strict arithmetic settings. All methods use
the repaired backend. The original case and two nearby odd-sized cases are
numerical regression tests with repeated invocation.

The failed run metadata and failing measurement are preserved in
`results/ppo-integration-001`. Earlier environment, optimization and deployment
pilots remain exploratory evidence from their recorded executable revisions.
Their timings must not be combined with new measurements as if the backend
were unchanged. The corpus and held-out performance remain sealed as before.
