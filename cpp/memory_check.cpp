#include "memory_check.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace rlmlir {
namespace {
constexpr unsigned char sentinel = 0xa5;

class Mapping {
public:
  explicit Mapping(size_t bytes) : size(bytes) {
    address = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (address == MAP_FAILED) throw std::runtime_error("guard allocation failed");
  }
  ~Mapping() { munmap(address, size); }
  Mapping(const Mapping &) = delete;
  Mapping &operator=(const Mapping &) = delete;
  void *address;
  size_t size;
};

size_t pageSize() {
  auto size = sysconf(_SC_PAGESIZE);
  if (size <= 0) throw std::runtime_error("cannot determine guard page size");
  return static_cast<size_t>(size);
}

class GuardedBuffer {
public:
  GuardedBuffer(const Shape &shape, const std::vector<double> &values,
                bool readOnly, bool alignEnd)
      : bytes(values.size() * sizeof(double)), page(pageSize()),
        usable(std::max(page, ((bytes + page - 1) / page) * page)),
        mapping(usable + 2 * page), abi(3 + 2 * shape.size()) {
    int64_t elements = 1;
    for (auto size : shape) elements *= size;
    if (elements != static_cast<int64_t>(values.size()))
      throw std::runtime_error("input length does not match the compiled shape");
    if (!std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }))
      throw std::runtime_error("inputs must be finite");
    middle = static_cast<unsigned char *>(mapping.address) + page;
    if (mprotect(middle, usable, PROT_READ | PROT_WRITE))
      throw std::runtime_error("cannot enable guarded buffer pages");
    std::memset(middle, sentinel, usable);
    data = middle + (alignEnd ? usable - bytes : 0);
    if (bytes) std::memcpy(data, values.data(), bytes);
    if (readOnly && mprotect(middle, usable, PROT_READ))
      throw std::runtime_error("cannot protect diagnostic input");
    abi[0] = abi[1] = reinterpret_cast<intptr_t>(data);
    abi[2] = 0;
    int64_t stride = 1;
    for (size_t index = shape.size(); index-- > 0;) {
      abi[3 + index] = shape[index];
      abi[3 + shape.size() + index] = stride;
      stride *= shape[index];
    }
  }
  void *descriptor() { return abi.data(); }
  void verifyPadding() const {
    // The opposite edge can have up to one page of alignment padding. Check
    // writes there as well; two placements test reads at both exact boundaries.
    auto intact = [](unsigned char value) { return value == sentinel; };
    if (!std::all_of(middle, data, intact) ||
        !std::all_of(data + bytes, middle + usable, intact))
      throw std::runtime_error("generated code wrote outside its output buffer");
  }
  std::vector<double> values() const {
    if (!bytes) return {};
    auto *begin = reinterpret_cast<const double *>(data);
    return {begin, begin + bytes / sizeof(double)};
  }
private:
  size_t bytes, page, usable;
  Mapping mapping;
  unsigned char *middle = nullptr, *data = nullptr;
  std::vector<intptr_t> abi;
};
} // namespace

std::vector<double> checkMemory(const std::shared_ptr<CompiledKernel> &kernel,
    const std::vector<std::vector<double>> &inputs, int64_t count, bool alignEnd) {
  if (!kernel || !kernel->entry || kernel->shapes.size() != 3 || inputs.size() != 3)
    throw std::runtime_error("memory check requires three buffers and live code");
  if (count < 1 || count > 1000) throw std::runtime_error("diagnostic count outside 1..1000");
  std::array<std::unique_ptr<GuardedBuffer>, 3> buffers;
  for (size_t i = 0; i < buffers.size(); ++i)
    buffers[i] = std::make_unique<GuardedBuffer>(kernel->shapes[i], inputs[i], i < 2, alignEnd);
  for (int64_t i = 0; i < count; ++i) {
    kernel->entry(buffers[0]->descriptor(), buffers[1]->descriptor(), buffers[2]->descriptor());
    buffers[2]->verifyPadding();
  }
  return buffers[2]->values();
}
} // namespace rlmlir
