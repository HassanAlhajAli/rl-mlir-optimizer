# Compiler and runtime milestone

The C++ driver accepts a deliberately restricted MLIR function named `kernel` with three contiguous static-shaped FP64 buffers. The first two are inputs, the third is output. It requires an output fill followed by one Linalg generic operation, projected-permutation access maps, at most four loop dimensions and one reduction. Matrix multiplication and matrix-vector multiplication are the current generators. Stencil/convolution support still needs explicit legality work.

Each schedule action runs on a cloned module. Only verified success replaces the previous state. Logical axis numbers remain attached to original dimensions, even after interchange. Tiling may create loop-bound remainders; masked vectorization handles those lanes. The vector action first tiles the reduction axis to one so the vector work runs across independent outputs. It never enables fast math or a horizontal reduction across multiple accumulation steps.

Lowering masks precedes expansion of vector reductions. Expanding a reduction inside a single-operation `vector.mask` region produces invalid IR. This ordering was established by an actual failing integration test and corrected before measurements.

All methods use the same backend: vector mask/reduction/transfer lowering, canonicalization, Linalg loops, vector-to-SCF, strided metadata expansion, affine/control-flow/memref/arithmetic/function conversion to LLVM, and LLVM O3 for x86-64-v3. Optimized LLVM IR and the actual JIT object can be exported to inspect what ran.

The runtime owns buffers and memref ABI descriptors. The JSON interface copies input arrays into distinct contiguous allocations, so external aliases/strides cannot reach execution. This is not yet a zero-copy external-array API. Shape lengths and finite inputs are checked before constructing an invocation. An invocation holds a shared executable owner; eviction cannot invalidate it. The cache retains at most32 variants and evicts the least-recently-used unused entry. A cache full of live handles returns an explicit error. `release` invalidates a handle; `clear_cache` releases only unused entries.

Cache hits report zero new lowering/code-generation time. Their dispatch/lookup time must still be measured by the caller. A fresh deployment measurement explicitly clears unused code or uses a fresh worker. Process-local compiler version/target/math settings and source/shape/schedule form compiled identity. Policy decision caching is not implemented yet because policies are not implemented yet.

The serial worker supports `info`, `analyze`, `compile`, `invoke`, `benchmark`, `release`, `clear_cache`, and `dump_object`. JSON on stdout, diagnostics on stderr. This is a trusted local research worker, not a hardened service for arbitrary untrusted MLIR. Calls allocate/validate buffers outside the execution timer; repeated invocations use the already-validated owned buffers. This boundary is part of the running-service deployment model.
