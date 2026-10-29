# Fresh deployment diagnostic, before the stack-allocation repair

These two workloads were deliberately selected after the optimization pilot:
one apparent vectorization success and one failure. This is a diagnostic, not
an unseen-program generalization study. All 90 deployments matched independent
references. Each cell below is the median of five fresh native deployments,
including compilation, owned-buffer installation and all N invocations.

| Workload | Calls | Baseline ms | Tile8 + vector ms | Tile16 + vector ms |
|---|---:|---:|---:|---:|
| Matvec128x256 | 1 | 11.278 | 15.915 | 22.521 |
| Matvec128x256 | 100 | 12.826 | 16.641 | 22.804 |
| Matvec128x256 | 10000 | 126.024 | 57.248 | 63.422 |
| Matmul64x64x64 | 1 | 13.959 | 39.628 | 976.819 |
| Matmul64x64x64 | 100 | 19.065 | 45.439 | 1010.120 |
| Matmul64x64x64 | 10000 | 594.341 | 652.788 | 1625.293 |

For this matvec case, the cheapest choice among the three changes with reuse.
The matmul example also shows that vectorization can increase compilation cost
without repaying it in execution. Neither example says whether RL helps.

Commands and selection protocol are in `protocol.json` and
`../../docs/optimization-pilot.md`; raw deployments are `records.jsonl`, source
revision and executable hash are `metadata.json`. The whole run took28.920s.
Initialization of the compiler service, transport, independent reference
checking and teardown are outside the deployment clock. This diagnostic lacks
the anchor-block protocol required of the final study and retains WSL noise.

After this experiment, PPO exploration exposed loop-local stack allocation in
the backend. See `../../docs/compiler-repair-001.md`. These are historical
measurements from the old executable, not results for the repaired compiler.
