# Build and inspect the first JIT

Use Ubuntu 24.04 with the pinned LLVM/MLIR packages. Installation is explicit:

```sh
sudo bash scripts/setup_ubuntu.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-18
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/jit_pilot.py --output runs/jit-pilot-001
```

The pilot refuses to overwrite an existing run. Run timing without concurrent builds or training. The generated source and lowered MLIR are saved beside measurements. Initial matrices are development examples, not the final corpus.

`cpp/compiler.cpp` parses/verifies MLIR, lowers it, optimizes LLVM at O3, and forces code generation by resolving the C ABI entry. `cpp/runtime.cpp` owns buffers and the executable handle. `cpp/worker.cpp` exposes a serial JSON-lines protocol. `tests/test_jit.py` checks actual generated code against a separate scalar reference, not a second compilation of the same source.

The current core also includes schedule actions, six kernel families and a32-entry executable cache. See `learning.md` for Python training dependencies and checks. Compilation and execution are timed separately; Python/JSON orchestration and input allocation are excluded from the steady execution clock. The `deploy` command separately times fresh compilation, owned-buffer setup and complete invocation batches. Startup is reported separately for the already-running-service objective.
