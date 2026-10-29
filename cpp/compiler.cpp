#include "compiler.h"
#include "schedule.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
#include <cstdlib>
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
struct TracePasses : mlir::PassInstrumentation {
  void runBeforePass(mlir::Pass *pass, mlir::Operation *) override {
    llvm::errs() << "starting pass: " << pass->getName() << '\n';
  }
};
} // namespace

Compiler::Compiler() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  context.appendDialectRegistry(registry);
  context.disableMultithreading();
  // Functional transformation APIs do not have a pass manager to load dependencies.
  context.loadAllAvailableDialects();
  mlir::registerAllPasses();
}

llvm::json::Object Compiler::analyze(const std::string &source, const llvm::json::Array &actions) {
  auto prepared = prepare(context, source, actions);
  auto result = std::move(prepared.observation);
  result["transformed_ir"] = printIR(*prepared.module);
  return result;
}

std::shared_ptr<CompiledKernel> Compiler::compile(const std::string &source, const llvm::json::Array &actions) {
  const auto start = Clock::now();
  auto prepared = prepare(context, source, actions);
  auto module = std::move(prepared.module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  if (!function || function.getNumArguments() != 3 || function.getNumResults() != 0)
    throw std::runtime_error("expected kernel with exactly three memref arguments and no result");
  auto result = std::make_shared<CompiledKernel>();
  result->shapes = std::move(prepared.shapes);
  function->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(&context));
  result->sourceIR = source;
  result->transformedIR = printIR(*module);
  // Mask regions may contain exactly one operation. Lower masks before expanding
  // a reduction into multiple operations; the reverse order creates invalid IR.
  mlir::PassManager maskPasses(&context);
  maskPasses.enableVerifier(true);
  if (mlir::failed(mlir::parsePassPipeline("func.func(lower-vector-mask)", maskPasses)) ||
      mlir::failed(maskPasses.run(*module)))
    throw std::runtime_error("vector mask lowering failed");
  mlir::RewritePatternSet vectorPatterns(&context);
  mlir::vector::populateVectorMultiReductionLoweringPatterns(
      vectorPatterns, mlir::vector::VectorMultiReductionLowering::InnerParallel);
  mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(vectorPatterns);
  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(*module, std::move(vectorPatterns))))
    throw std::runtime_error("fixed vector lowering patterns did not converge");
  if (mlir::failed(mlir::verify(*module))) throw std::runtime_error("vector lowering produced invalid IR");
  mlir::PassManager passes(&context);
  if (std::getenv("RLMLIR_DEBUG_PASSES")) passes.addInstrumentation(std::make_unique<TracePasses>());
  passes.enableVerifier(true);
  const char *pipeline =
      "canonicalize,func.func(convert-linalg-to-loops,convert-vector-to-scf),"
      "expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-vector-to-llvm,"
      "finalize-memref-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,"
      "convert-cf-to-llvm,reconcile-unrealized-casts";
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
  auto captureLLVM = [&](llvm::Module *llvmModule) -> llvm::Error {
    if (auto error = optimizer(llvmModule)) return error;
    llvm::raw_string_ostream stream(result->llvmIR);
    llvmModule->print(stream, nullptr);
    return llvm::Error::success();
  };
  mlir::ExecutionEngineOptions options;
  options.transformer = captureLLVM;
  options.enableObjectDump = true;
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
