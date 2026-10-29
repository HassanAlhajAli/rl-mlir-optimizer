# Correctness replay after native-width vector lowering

At revision `08e0568`, the revised compiler replayed all 833 records from the
failed runtime/17 job, in order, including the previously failed record.
Each record compiled the selected schedule and the fixed anchor and checked
both with input seeds 7 and 17, using three repeated invocations per check.

- 833/833 records passed; 3,332 numerical comparisons matched their references.
- 1,666 compile requests produced 1,538 actual compilations; 128 were cache hits.
- 9,996 invocations completed in the persistent worker.
- Recorded elapsed time: 126.82 seconds, within the 1,200-second diagnostic cap.
- Worker peak RSS: 427,116 KiB, about 417 MiB; address-space ceiling: 4 GiB.
- All four local CTest suites and all 17 Python tests passed before this replay.

The [audit summary](audit.json) records the executable hash, exact source-record
hash, compiler revision, backend identity and counters. [raw.zip](raw.zip)
contains each per-record correctness result, the original failure/reproducer,
GDB and bounded-memory diagnostics, rejected-repair evidence, and verification
logs. [manifest.json](manifest.json) hashes every member. The old executable
and the large generated replay input remain local; their source revision and
inputs are identified in the original campaign evidence.

This is a correctness and stability diagnostic, **not a performance replay**.
It does not repeat calibration, long timed batches, every policy observation,
or the original compiler behavior. No speedup is inferred from elapsed time.
The original row-reduction SIGSEGV did not reproduce in isolation and remains
unexplained. The diagnosed vector-related resource failure is separately
reproducible, and the repaired cases pass, but this does not establish that the
two failures had the same cause.

See the [implementation and reproduction instructions](../../docs/compiler-repair-002.md).
The main training campaign remains stopped; held-out performance stays sealed.
