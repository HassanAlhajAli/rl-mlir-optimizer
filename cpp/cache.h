#pragma once
#include "compiler.h"
#include <list>
#include <string>
#include <utility>

namespace rlmlir {
// Cache owns one shared reference. Handles/invocations own additional references.
// Only unused entries can be evicted. Pinned code cannot become a dangling pointer.
class ExecutableCache {
public:
  std::shared_ptr<CompiledKernel> find(const std::string &key);
  void reserveSlot();
  void insert(std::string key, std::shared_ptr<CompiledKernel> code);
  size_t size() const { return entries.size(); }
  void clearUnused();
private:
  static constexpr size_t capacity = 32;
  std::list<std::pair<std::string, std::shared_ptr<CompiledKernel>>> entries;
};
} // namespace rlmlir
