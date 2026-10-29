#include "cache.h"
#include <stdexcept>

namespace rlmlir {
std::shared_ptr<CompiledKernel> ExecutableCache::find(const std::string &key) {
  for (auto it = entries.begin(); it != entries.end(); ++it) {
    if (it->first == key) {
      auto code = it->second;
      entries.splice(entries.begin(), entries, it);
      return code;
    }
  }
  return {};
}

void ExecutableCache::reserveSlot() {
  if (entries.size() < capacity) return;
  for (auto it = entries.end(); it != entries.begin();) {
    --it;
    if (it->second.use_count() == 1) {
      entries.erase(it);
      return;
    }
  }
  throw std::runtime_error("compiled cache is full of active handles; release a handle first");
}

void ExecutableCache::insert(std::string key, std::shared_ptr<CompiledKernel> code) {
  entries.emplace_front(std::move(key), std::move(code));
}

void ExecutableCache::clearUnused() {
  for (auto it = entries.begin(); it != entries.end();) {
    if (it->second.use_count() == 1) it = entries.erase(it);
    else ++it;
  }
}
} // namespace rlmlir
