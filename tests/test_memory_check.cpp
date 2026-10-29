#include "memory_check.h"
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
double *data(void *descriptor) {
  return reinterpret_cast<double *>(static_cast<intptr_t *>(descriptor)[1]);
}
void correct(void *a, void *b, void *out) { data(out)[0] = data(a)[0] * data(b)[0]; }
void nothing(void *, void *, void *) {}
void overflow(void *, void *, void *out) { data(out)[1] = 42; }
void underflow(void *, void *, void *out) { data(out)[-1] = 42; }
void inputWrite(void *a, void *, void *) { data(a)[0] = 42; }
void inputReadPast(void *a, void *, void *) { volatile double value = data(a)[1]; (void)value; }
void inputReadBefore(void *a, void *, void *) { volatile double value = data(a)[-1]; (void)value; }

using Entry = void (*)(void *, void *, void *);
std::shared_ptr<rlmlir::CompiledKernel> kernel(Entry entry) {
  auto code = std::make_shared<rlmlir::CompiledKernel>();
  code->entry = entry;
  code->shapes = {{1}, {1}, {1}};
  return code;
}
const std::vector<std::vector<double>> inputs = {{2}, {3}, {99}};

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void mustFault(Entry entry, bool alignEnd) {
  // Isolate deliberately invalid code; the parent must observe a real fault.
  pid_t child = fork();
  require(child >= 0, "fork failed");
  if (child == 0) {
    const rlimit noCore{0, 0};
    if (setrlimit(RLIMIT_CORE, &noCore)) _exit(3);
    try { rlmlir::checkMemory(kernel(entry), inputs, 1, alignEnd); }
    catch (...) { _exit(2); }
    _exit(0);
  }
  int status = 0;
  require(waitpid(child, &status, 0) == child, "waitpid failed");
  require(WIFSIGNALED(status) && WTERMSIG(status) == SIGSEGV, "guard failed to trap invalid memory access");
}
void mustReject(Entry entry, bool alignEnd) {
  try { rlmlir::checkMemory(kernel(entry), inputs, 1, alignEnd); }
  catch (const std::runtime_error &) { return; }
  throw std::runtime_error("padding overwrite was not detected");
}
} // namespace

int main() {
  for (bool end : {false, true}) {
    require(rlmlir::checkMemory(kernel(correct), inputs, 3, end) == std::vector<double>{6}, "valid output changed");
    mustFault(inputWrite, end);
    auto empty = kernel(nothing);
    empty->shapes = {{0}, {0}, {0}};
    require(rlmlir::checkMemory(empty, {{}, {}, {}}, 3, end).empty(), "empty output changed");
  }
  mustFault(overflow, true);
  mustFault(underflow, false);
  mustFault(inputReadPast, true);
  mustFault(inputReadBefore, false);
  mustReject(overflow, false);
  mustReject(underflow, true);
  try {
    rlmlir::checkMemory(kernel(correct), {{2,4}, {3}, {0}}, 1, false);
    throw std::logic_error("wrong input length accepted");
  } catch (const std::runtime_error &) {}
  std::cout << "Boundary reads/writes, input protection, padding, repeated and zero-size checks passed\n";
}
