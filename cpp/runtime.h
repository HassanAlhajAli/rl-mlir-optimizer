#pragma once
#include "compiler.h"
#include <cstdint>
#include <vector>

namespace rlmlir {
// Dynamic storage for the ranked memref ABI: two pointers, offset, sizes, strides.
// The allocation and descriptor outlive every call; buffers are independently owned.
class Buffer {
public:
  Buffer(const Shape &shape, std::vector<double> values);
  void *descriptor() { return abi.data(); }
  const std::vector<double> &values() const { return storage; }
private:
  std::vector<double> storage;
  std::vector<intptr_t> abi;
};

class Invocation {
public:
  Invocation(std::shared_ptr<CompiledKernel> kernel,
             const std::vector<std::vector<double>> &inputs);
  void run();
  const std::vector<double> &output() const { return buffers[2].values(); }
private:
  std::shared_ptr<CompiledKernel> kernel;
  std::vector<Buffer> buffers;
};
} // namespace rlmlir
