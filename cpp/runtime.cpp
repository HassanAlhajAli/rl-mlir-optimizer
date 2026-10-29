#include "runtime.h"
#include <cmath>
#include <stdexcept>

namespace rlmlir {
Buffer::Buffer(const Shape &shape, std::vector<double> values)
    : storage(std::move(values)), abi(3 + 2 * shape.size()) {
  static_assert(sizeof(intptr_t) == sizeof(void *));
  int64_t count = 1;
  for (auto size : shape) count *= size;
  if (count != static_cast<int64_t>(storage.size()))
    throw std::runtime_error("input length does not match the compiled shape");
  for (auto value : storage)
    if (!std::isfinite(value)) throw std::runtime_error("inputs must be finite");
  abi[0] = abi[1] = reinterpret_cast<intptr_t>(storage.data());
  abi[2] = 0;
  int64_t stride = 1;
  for (size_t index = shape.size(); index-- > 0;) {
    abi[3 + index] = shape[index];
    abi[3 + shape.size() + index] = stride;
    stride *= shape[index];
  }
}

Invocation::Invocation(std::shared_ptr<CompiledKernel> code,
                       const std::vector<std::vector<double>> &inputs)
    : kernel(std::move(code)) {
  if (inputs.size() != 3) throw std::runtime_error("expected exactly three buffers");
  buffers.reserve(3);
  for (size_t index = 0; index < 3; ++index)
    buffers.emplace_back(kernel->shapes[index], inputs[index]);
}

void Invocation::run() {
  kernel->entry(buffers[0].descriptor(), buffers[1].descriptor(), buffers[2].descriptor());
}
} // namespace rlmlir
