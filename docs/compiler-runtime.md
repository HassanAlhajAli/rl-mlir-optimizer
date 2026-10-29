# Compiler and runtime milestone

The C++ driver accepts a deliberately restricted MLIR function named `kernel` with three contiguous static-shaped FP64 buffers. The first two are inputs, the third is output. It requires an output fill followed by one Linalg generic operation and at most four loop dimensions. Six generators are supported: matmul, matvec, row reduction, elementwise calculation, a nine-point 2D averaging stencil with an input halo, and direct1D cross-correlation (the usual ML convolution convention).

Projected-permutation and bounded nonnegative affine input accesses are supported. Every affine index is checked against static buffer extents. The output map must uniquely cover the parallel dimensions, so changing their order cannot create conflicting writes. Multiple reduction dimensions retain their original nesting and cannot be tiled individually. Stencil/convolution indexing is conservatively excluded from vectorization in this implementation; tiling and parallel interchange remain available. This restriction is visible in the policy's legal-action mask.

Each schedule action runs on a cloned module. Only verified success replaces the previous state. Logical axis numbers remain attached to original dimensions, even after interchange. Tiling may create loop-bound remainders; masked vectorization handles those lanes. Since the second compiler repair, the vector action creates an inner tile with at most four FP64 elements on the innermost parallel axis and one on every other axis. Selected outer tiles still control locality. Reductions remain scalar and ordered. It never enables fast math or a horizontal reduction across multiple accumulation steps.

Lowering masks precedes expansion of vector reductions. Expanding a reduction inside a single-operation `vector.mask` region produces invalid IR. This ordering was established by an actual failing integration test and corrected before measurements.

All methods use the same backend: vector mask/reduction/transfer lowering, canonicalization, Linalg loops, vector-to-SCF, strided metadata expansion, affine/control-flow/memref/arithmetic/function conversion to LLVM, and LLVM O3 for x86-64-v3. Optimized LLVM IR and the actual JIT object can be exported to inspect what ran.

The runtime owns buffers and memref ABI descriptors. The JSON interface copies input arrays into distinct contiguous allocations, so external aliases/strides cannot reach execution. This is not yet a zero-copy external-array API. Shape lengths and finite inputs are checked before constructing an invocation. An invocation holds a shared executable owner; eviction cannot invalidate it. The cache retains at most32 variants and evicts the least-recently-used unused entry. A cache full of live handles returns an explicit error. `release` invalidates a handle; `clear_cache` releases only unused entries.

Cache hits report zero new lowering/code-generation time. Their dispatch/lookup time must still be measured by the caller. A fresh deployment measurement explicitly clears unused code or uses a fresh worker. Process-local compiler version/target/math settings, vector-lowering version and source/shape/schedule form compiled identity. Policy decision caching is not implemented; the policies themselves are implemented and have bounded integration evidence.

The serial worker supports `info`, `analyze`, `compile`, `invoke`, `benchmark`, `release`, `clear_cache`, and `dump_object`. JSON on stdout, diagnostics on stderr. This is a trusted local research worker, not a hardened service for arbitrary untrusted MLIR. Calls allocate/validate buffers outside the execution timer; repeated invocations use the already-validated owned buffers. This boundary is part of the running-service deployment model.

## Protected-memory diagnostic

`check_memory` is a separate correctness-only command. It invokes an already
compiled handle with independent page-protected buffers, marks both inputs
read-only, and checks sentinel padding around the output after each invocation.
The required `alignment` is `start` or `end`: replay both placements to test
accesses at each exact boundary. Counts are limited to 1..1000. Empty buffers
are supported. Executable ownership is retained throughout the call.

Boundary violations intentionally terminate the worker with a native fault;
padding writes return an error. The caller must preserve the current record
and stop, as it does for other unexpected failures. This command returns no
timing fields and is never used to assign a performance reward. Normal
`invoke`, `benchmark` and `deploy` allocation/timing paths are unchanged.

The `memory_guards` CTest deliberately executes bad functions in child
processes. It verifies actual traps for input writes and boundary reads/writes,
plus detection of padding corruption, valid repeated calls, and empty buffers.
Passing this diagnostic is additional evidence, not a proof of memory safety:
it does not cover all possible wild addresses, internal JIT temporaries, or
compiler-library allocations.
