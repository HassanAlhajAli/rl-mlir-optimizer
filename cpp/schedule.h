#pragma once
#include "compiler.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/JSON.h"

namespace rlmlir {
struct PreparedKernel {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<Shape> shapes;
  llvm::json::Object observation;
};
PreparedKernel prepare(mlir::MLIRContext &context, const std::string &source,
                       const llvm::json::Array &actions);
} // namespace rlmlir
