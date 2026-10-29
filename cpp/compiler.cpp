#include "compiler.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/Module.h"
#include <chrono>
#include <stdexcept>

namespace rlmlir {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::string printIR(mlir::Operation *op) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  op->print(stream);
  return result;
}
} // namespace

Compiler::Compiler() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  context.appendDialectRegistry(registry);
  context.disableMultithreading();
  mlir::registerAllPasses();
}

std::shared_ptr<CompiledKernel> Compiler::compile(const std::string &source) {
  const auto start = Clock::now();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module || mlir::failed(mlir::verify(*module)))
    throw std::runtime_error("input MLIR failed parsing or verification");
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  if (!function || function.getNumArguments() != 3 || function.getNumResults() != 0)
    throw std::runtime_error("expected kernel with exactly three memref arguments and no result");
  auto result = std::make_shared<CompiledKernel>();
  for (auto type : function.getArgumentTypes()) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
    if (!memref || !memref.hasStaticShape() || !memref.getElementType().isF64() ||
        !memref.getLayout().isIdentity() || memref.getRank() < 1 || memref.getRank() > 4)
      throw std::runtime_error("only static contiguous FP64 memrefs of rank 1..4 are supported");
    int64_t elements = 1;
    for (auto dimension : memref.getShape()) {
      if (dimension > 10000000 || (dimension && elements > 10000000 / dimension))
        throw std::runtime_error("buffer exceeds the ten-million-element safety limit");
      elements *= dimension;
    }
    result->shapes.emplace_back(memref.getShape().begin(), memref.getShape().end());
  }
  function->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(&context));
  result->sourceIR = printIR(*module);
  mlir::PassManager passes(&context);
  passes.enableVerifier(true);
  const char *pipeline =
      "func.func(convert-linalg-to-loops,lower-affine),"
      "convert-scf-to-cf,convert-vector-to-llvm,finalize-memref-to-llvm,"
      "convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts";
  if (mlir::failed(mlir::parsePassPipeline(pipeline, passes)) ||
      mlir::failed(passes.run(*module)))
    throw std::runtime_error("fixed MLIR lowering pipeline failed");
  result->loweredIR = printIR(*module);
  result->loweringMs = elapsed(start);

  const auto codegenStart = Clock::now();
  std::string error;
  auto triple = llvm::sys::getDefaultTargetTriple();
  auto *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) throw std::runtime_error(error);
  llvm::TargetOptions targetOptions;
  targetOptions.AllowFPOpFusion = llvm::FPOpFusion::Strict;
  auto machine = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      triple, "x86-64-v3", "", targetOptions, llvm::Reloc::PIC_, std::nullopt,
      llvm::CodeGenOptLevel::Aggressive));
  if (!machine) throw std::runtime_error("failed to create x86-64-v3 target");
  auto optimizer = mlir::makeOptimizingTransformer(3, 0, machine.get());
  mlir::ExecutionEngineOptions options;
  options.transformer = optimizer;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  options.enableGDBNotificationListener = false;
  options.enablePerfNotificationListener = false;
  auto engine = mlir::ExecutionEngine::create(*module, options, std::move(machine));
  if (!engine) throw std::runtime_error(llvm::toString(engine.takeError()));
  result->engine = std::move(*engine);
  // Lookup forces lazy code generation before the execution timer can start.
  auto symbol = result->engine->lookup("_mlir_ciface_kernel");
  if (!symbol) throw std::runtime_error(llvm::toString(symbol.takeError()));
  result->entry = reinterpret_cast<void (*)(void *, void *, void *)>(*symbol);
  result->codegenMs = elapsed(codegenStart);
  return result;
}
} // namespace rlmlir
