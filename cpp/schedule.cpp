#include "schedule.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace rlmlir {
namespace {
mlir::linalg::GenericOp findGeneric(mlir::ModuleOp module) {
  mlir::linalg::GenericOp found;
  module.walk([&](mlir::linalg::GenericOp op) { found = op; });
  return found;
}

std::vector<Shape> validate(mlir::ModuleOp module) {
  auto function = module.lookupSymbol<mlir::func::FuncOp>("kernel");
  if (!function || function.getNumArguments() != 3 || function.getNumResults() ||
      !llvm::hasSingleElement(function.getBody()))
    throw std::runtime_error("expected one defined kernel with three buffers and no result");
  int functions = 0, generics = 0, fills = 0;
  module.walk([&](mlir::Operation *op) {
    if (!mlir::isa<mlir::ModuleOp, mlir::func::FuncOp, mlir::func::ReturnOp,
                   mlir::linalg::GenericOp, mlir::linalg::FillOp, mlir::linalg::YieldOp,
                   mlir::arith::ConstantOp, mlir::arith::AddFOp,
                   mlir::arith::SubFOp, mlir::arith::MulFOp>(op))
      throw std::runtime_error("unsupported input operation: " + op->getName().getStringRef().str());
    functions += mlir::isa<mlir::func::FuncOp>(op);
    generics += mlir::isa<mlir::linalg::GenericOp>(op);
    fills += mlir::isa<mlir::linalg::FillOp>(op);
    if (auto flags = op->getAttrOfType<mlir::arith::FastMathFlagsAttr>("fastmath"))
      if (flags.getValue() != mlir::arith::FastMathFlags::none)
        throw std::runtime_error("fast-math flags are outside the strict FP64 protocol");
  });
  if (functions != 1 || generics != 1 || fills != 1)
    throw std::runtime_error("expected one function, one output fill and one generic calculation");
  auto generic = findGeneric(module);
  if (generic.getNumDpsInputs() != 2 || generic.getNumDpsInits() != 1 ||
      generic.getDpsInputOperand(0)->get() != function.getArgument(0) ||
      generic.getDpsInputOperand(1)->get() != function.getArgument(1) ||
      generic.getDpsInitOperand(0)->get() != function.getArgument(2))
    throw std::runtime_error("calculation must read buffers 0/1 and write buffer 2");
  if (!llvm::all_of(generic.getIndexingMapsArray(), [](mlir::AffineMap map) { return map.isProjectedPermutation(); }))
    throw std::runtime_error("v1 pilot accepts only projected-permutation indexing maps");
  module.walk([&](mlir::linalg::FillOp fill) {
    if (fill.getOutputs()[0] != function.getArgument(2) ||
        fill->getBlock() != generic->getBlock() || !fill->isBeforeInBlock(generic))
      throw std::runtime_error("output initialization must precede the calculation");
  });
  if (generic.getNumLoops() > 4 || generic.getNumReductionLoops() > 1)
    throw std::runtime_error("pilot supports at most four loops and one reduction axis");
  std::vector<Shape> shapes;
  for (auto type : function.getArgumentTypes()) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
    if (!memref || !memref.hasStaticShape() || !memref.getElementType().isF64() ||
        !memref.getLayout().isIdentity() || memref.getRank() < 1 || memref.getRank() > 4 ||
        memref.getMemorySpaceAsInt() != 0)
      throw std::runtime_error("only static contiguous host FP64 memrefs of rank 1..4 are supported");
    int64_t count = 1;
    for (auto d : memref.getShape()) {
      if (d > 10000000 || (d && count > 10000000 / d))
        throw std::runtime_error("buffer exceeds ten million elements");
      count *= d;
    }
    shapes.emplace_back(memref.getShape().begin(), memref.getShape().end());
  }
  return shapes;
}

mlir::linalg::GenericOp tile(mlir::IRRewriter &rewriter,
                             mlir::linalg::GenericOp op,
                             llvm::ArrayRef<int64_t> sizes) {
  rewriter.setInsertionPoint(op);
  mlir::linalg::LinalgTilingOptions options;
  options.setTileSizes(sizes);
  auto tiled = mlir::linalg::tileLinalgOp(rewriter, op, options);
  if (mlir::failed(tiled)) throw std::runtime_error("legal tiling unexpectedly failed");
  rewriter.eraseOp(op);
  return mlir::cast<mlir::linalg::GenericOp>(tiled->op.getOperation());
}
} // namespace

PreparedKernel prepare(mlir::MLIRContext &context, const std::string &source,
                       const llvm::json::Array &actions) {
  PreparedKernel result;
  result.module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!result.module || mlir::failed(mlir::verify(*result.module)))
    throw std::runtime_error("input MLIR failed parsing or verification");
  result.shapes = validate(*result.module);
  auto original = findGeneric(*result.module);
  auto originalBounds = original.getStaticLoopRanges();
  auto types = original.getIteratorTypesArray();
  std::vector<int64_t> bounds(originalBounds.begin(), originalBounds.end());
  std::vector<unsigned> order(bounds.size());
  std::iota(order.begin(), order.end(), 0);
  std::vector<std::vector<int64_t>> tiles(bounds.size());
  bool terminal = false, vectorized = false;
  unsigned steps = 0;
  const auto canVectorize = [&]() {
    int64_t volume = 1;
    for (unsigned i = 0; i < bounds.size(); ++i) {
      if (bounds[i] <= 0) return false;
      if (types[i] == mlir::utils::IteratorType::parallel) {
        if (bounds[i] > 1024 / volume) return false;
        volume *= bounds[i];
      }
    }
    return true;
  };
  for (const auto &value : actions) {
    auto *action = value.getAsObject();
    if (!action) throw std::runtime_error("each action must be an object");
    auto kind = action->getString("kind");
    if (!kind || terminal) throw std::runtime_error("missing action kind or action after termination");
    if (*kind == "STOP") { terminal = true; continue; }
    if (steps >= 6) throw std::runtime_error("six-transformation budget exceeded");
    // Every action mutates a clone. Failed transformations cannot poison the caller's IR.
    auto candidate = mlir::OwningOpRef<mlir::ModuleOp>(result.module->clone());
    auto op = findGeneric(*candidate);
    mlir::IRRewriter rewriter(&context);
    if (*kind == "TILE") {
      auto axis = action->getInteger("axis"), size = action->getInteger("size");
      if (!axis || !size || *axis < 0 || *axis >= static_cast<int64_t>(bounds.size()) ||
          (*size != 4 && *size != 8 && *size != 16 && *size != 32 && *size != 64))
        throw std::runtime_error("invalid tile axis or size");
      auto &history = tiles[*axis];
      if (bounds[*axis] <= 0 || history.size() >= 2 || (!history.empty() && *size >= history.back()))
        throw std::runtime_error("tile requires a positive dimension and decreasing levels (at most two)");
      llvm::SmallVector<int64_t> sizes(bounds.size(), 0);
      auto position = std::find(order.begin(), order.end(), *axis) - order.begin();
      sizes[position] = *size;
      tile(rewriter, op, sizes);
      history.push_back(*size);
      bounds[*axis] = std::min(bounds[*axis], *size);
    } else if (*kind == "INTERCHANGE") {
      auto a = action->getInteger("a"), b = action->getInteger("b");
      if (!a || !b || *a < 0 || *b < 0 || *a >= static_cast<int64_t>(bounds.size()) ||
          *b >= static_cast<int64_t>(bounds.size()) || *a == *b ||
          types[*a] != mlir::utils::IteratorType::parallel || types[*b] != mlir::utils::IteratorType::parallel)
        throw std::runtime_error("interchange requires two distinct parallel logical axes");
      auto pa = std::find(order.begin(), order.end(), *a) - order.begin();
      auto pb = std::find(order.begin(), order.end(), *b) - order.begin();
      llvm::SmallVector<unsigned> permutation(bounds.size());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::swap(permutation[pa], permutation[pb]);
      rewriter.setInsertionPoint(op);
      if (mlir::failed(mlir::linalg::interchangeGenericOp(rewriter, op, permutation)))
        throw std::runtime_error("legal interchange unexpectedly failed");
      std::swap(order[pa], order[pb]);
    } else if (*kind == "VECTORIZE") {
      if (!canVectorize()) throw std::runtime_error("vector tile is empty or exceeds 1024 elements");
      llvm::SmallVector<int64_t> scalarReduction(bounds.size(), 0), vectorSizes;
      for (unsigned position = 0; position < order.size(); ++position) {
        auto axis = order[position];
        const bool reduction = types[axis] == mlir::utils::IteratorType::reduction;
        scalarReduction[position] = reduction ? 1 : 0;
        vectorSizes.push_back(reduction ? 1 : bounds[axis]);
      }
      // Keep each reduction step scalar and in order. SIMD is across independent outputs.
      if (llvm::is_contained(scalarReduction, 1)) op = tile(rewriter, op, scalarReduction);
      rewriter.setInsertionPoint(op);
      llvm::SmallVector<bool> scalable(vectorSizes.size(), false);
      if (mlir::failed(mlir::linalg::vectorize(rewriter, op, vectorSizes, scalable)))
        throw std::runtime_error("masked vectorization failed");
      terminal = true;
      vectorized = true;
    } else throw std::runtime_error("unknown schedule action");
    if (mlir::failed(mlir::verify(*candidate))) throw std::runtime_error("transformed IR failed verification");
    result.module = std::move(candidate);
    ++steps;
  }
  llvm::json::Array dimensions, permutation, tileHistory, legal;
  legal.push_back(llvm::json::Object{{"kind", "STOP"}});
  for (unsigned i = 0; i < bounds.size(); ++i) {
    dimensions.push_back(llvm::json::Object{{"bound", originalBounds[i]}, {"tile_bound", bounds[i]},
                         {"parallel", types[i] == mlir::utils::IteratorType::parallel}});
    permutation.push_back(order[i]);
    llvm::json::Array history;
    for (auto size : tiles[i]) history.push_back(size);
    tileHistory.push_back(std::move(history));
    if (terminal || steps >= 6) continue;
    for (int64_t size : {4, 8, 16, 32, 64})
      if (bounds[i] > 0 && tiles[i].size() < 2 && (tiles[i].empty() || size < tiles[i].back()))
        legal.push_back(llvm::json::Object{{"kind", "TILE"}, {"axis", i}, {"size", size}});
    for (unsigned j = i + 1; j < bounds.size(); ++j)
      if (types[i] == mlir::utils::IteratorType::parallel && types[j] == mlir::utils::IteratorType::parallel)
        legal.push_back(llvm::json::Object{{"kind", "INTERCHANGE"}, {"a", i}, {"b", j}});
  }
  if (!terminal && steps < 6 && canVectorize()) legal.push_back(llvm::json::Object{{"kind", "VECTORIZE"}});
  result.observation = llvm::json::Object{{"dimensions", std::move(dimensions)},
      {"order", std::move(permutation)}, {"tiles", std::move(tileHistory)},
      {"steps", steps}, {"terminal", terminal || steps == 6}, {"vectorized", vectorized},
      {"legal_actions", std::move(legal)}};
  return result;
}
} // namespace rlmlir
