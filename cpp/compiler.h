#pragma once

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <string>
#include <vector>

namespace rlmlir {
using Shape = std::vector<int64_t>;

// The function pointer is valid only while its owning engine is alive.
struct CompiledKernel {
  std::unique_ptr<mlir::ExecutionEngine> engine;
  void (*entry)(void *, void *, void *) = nullptr;
  std::vector<Shape> shapes;
  std::string sourceIR;
  std::string loweredIR;
  std::string transformedIR;
  std::string llvmIR;
  double loweringMs = 0;
  double codegenMs = 0;
};

class Compiler {
public:
  Compiler();
  std::shared_ptr<CompiledKernel> compile(const std::string &source, const llvm::json::Array &actions = {});
  llvm::json::Object analyze(const std::string &source, const llvm::json::Array &actions);

private:
  mlir::MLIRContext context;
};
} // namespace rlmlir
