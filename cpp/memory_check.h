#pragma once
#include "compiler.h"

namespace rlmlir {
// Correctness diagnostic only. Runs with inaccessible boundary pages and
// read-only inputs; a boundary violation intentionally terminates the worker.
std::vector<double> checkMemory(const std::shared_ptr<CompiledKernel> &kernel,
    const std::vector<std::vector<double>> &inputs, int64_t count, bool alignEnd);
} // namespace rlmlir
