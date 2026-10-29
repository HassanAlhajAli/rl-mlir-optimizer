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

std::vector<int64_t> coefficients(mlir::AffineExpr expression, unsigned dimensions) {
  std::vector<int64_t> result(dimensions + 1, 0);
  if (auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
    result.at(dimension.getPosition()) = 1;
  } else if (auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression)) {
    if (constant.getValue() < 0 || constant.getValue() > 10000000)
      throw std::runtime_error("unsupported affine offset");
    result.back() = constant.getValue();
  } else if (auto binary = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression)) {
    auto left = coefficients(binary.getLHS(), dimensions);
    auto right = coefficients(binary.getRHS(), dimensions);
    if (binary.getKind() == mlir::AffineExprKind::Add) {
      for (unsigned i = 0; i <= dimensions; ++i) result[i] = left[i] + right[i];
    } else if (binary.getKind() == mlir::AffineExprKind::Mul &&
               llvm::all_of(llvm::ArrayRef<int64_t>(right).drop_back(), [](auto c) { return c == 0; }) && right.back() <= 1024) {
      for (unsigned i = 0; i <= dimensions; ++i) result[i] = left[i] * right.back();
    } else throw std::runtime_error("only bounded nonnegative affine additions/scaling are supported");
  } else throw std::runtime_error("unsupported affine expression");
  if (llvm::any_of(result, [](auto c) { return c > 10000000; }))
    throw std::runtime_error("affine coefficient exceeds the supported bound");
  return result;
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
  module.walk([&](mlir::linalg::FillOp fill) {
    if (fill.getOutputs()[0] != function.getArgument(2) ||
        fill->getBlock() != generic->getBlock() || !fill->isBeforeInBlock(generic))
      throw std::runtime_error("output initialization must precede the calculation");
  });
  if (generic.getNumLoops() > 4)
    throw std::runtime_error("at most four loop dimensions are supported");
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
  const auto ranges = generic.getStaticLoopRanges();
  if (ranges.size() != generic.getNumLoops() || llvm::any_of(ranges, [](auto d) { return d < 0 || d > 10000000; }))
    throw std::runtime_error("could not establish bounded static loop ranges");
  const auto maps = generic.getIndexingMapsArray();
  const auto iterators = generic.getIteratorTypesArray();
  std::vector<unsigned> outputUses(ranges.size(), 0);
  for (auto expression : maps.back().getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension) throw std::runtime_error("output map must contain only parallel dimensions");
    ++outputUses[dimension.getPosition()];
  }
  for (unsigned i = 0; i < ranges.size(); ++i)
    if (outputUses[i] != (iterators[i] == mlir::utils::IteratorType::parallel ? 1u : 0u))
      throw std::runtime_error("output indexing must uniquely cover the parallel dimensions");
  for (unsigned operand = 0; operand < maps.size(); ++operand) {
    for (unsigned resultIndex = 0; resultIndex < maps[operand].getNumResults(); ++resultIndex) {
      auto linear = coefficients(maps[operand].getResult(resultIndex), ranges.size());
      int64_t maximum = linear.back();
      for (unsigned i = 0; i < ranges.size(); ++i) maximum += linear[i] * std::max<int64_t>(0, ranges[i] - 1);
      if (!llvm::is_contained(ranges, 0) && maximum >= shapes[operand][resultIndex])
        throw std::runtime_error("indexing map can access beyond an input/output buffer");
    }
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
  const bool projected = llvm::all_of(original.getIndexingMapsArray(), [](mlir::AffineMap map) { return map.isProjectedPermutation(); });
  const auto reductionCount = original.getNumReductionLoops();
  llvm::json::Array accessStrides, shapes, arithmetic;
  for (unsigned operand = 0; operand < result.shapes.size(); ++operand) {
    const auto &shape = result.shapes[operand];
    llvm::json::Array shapeJson;
    for (auto dimension : shape) shapeJson.push_back(dimension);
    shapes.push_back(std::move(shapeJson));
    std::vector<int64_t> strides(originalBounds.size(), 0);
    int64_t memoryStride = 1;
    auto map = original.getIndexingMapsArray()[operand];
    for (size_t i = shape.size(); i-- > 0;) {
      auto linear = coefficients(map.getResult(i), originalBounds.size());
      for (unsigned axis = 0; axis < strides.size(); ++axis) strides[axis] += linear[axis] * memoryStride;
      memoryStride *= shape[i];
    }
    llvm::json::Array strideJson;
    for (auto stride : strides) strideJson.push_back(stride);
    accessStrides.push_back(std::move(strideJson));
  }
  for (llvm::StringRef name : {"arith.addf", "arith.mulf", "arith.subf"}) {
    int64_t count = 0;
    original.walk([&](mlir::Operation *op) { count += op->getName().getStringRef() == name; });
    arithmetic.push_back(count);
  }
  int64_t usedScalarLoads = 0;
  for (auto argument : original.getBody()->getArguments())
    usedScalarLoads += !argument.use_empty();
  std::vector<int64_t> bounds(originalBounds.begin(), originalBounds.end());
  std::vector<unsigned> order(bounds.size());
  std::iota(order.begin(), order.end(), 0);
  std::vector<std::vector<int64_t>> tiles(bounds.size());
  bool terminal = false, vectorized = false;
  unsigned steps = 0;
  const auto canVectorize = [&]() {
    if (!projected || reductionCount > 1) return false;
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
      if (reductionCount > 1 && types[*axis] == mlir::utils::IteratorType::reduction)
        throw std::runtime_error("multi-axis reductions retain their original loop nesting");
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
      // Legal schedule tiles describe locality, not physical SIMD width. Build
      // a small inner tile before vectorization so LLVM never receives enormous
      // transpose/shuffle expressions. x86-64-v3 has four FP64 lanes per AVX2
      // vector. Only the innermost parallel axis uses those lanes; reductions
      // remain scalar and ordered. Remainders use the existing masked transfers.
      auto laneAxis = std::find_if(order.rbegin(), order.rend(), [&](unsigned axis) {
        return types[axis] == mlir::utils::IteratorType::parallel;
      });
      llvm::SmallVector<int64_t> nativeTile(bounds.size(), 1), vectorSizes(bounds.size(), 1);
      for (unsigned position = 0; position < order.size(); ++position) {
        auto axis = order[position];
        if (laneAxis != order.rend() && axis == *laneAxis)
          nativeTile[position] = vectorSizes[position] = std::min<int64_t>(bounds[axis], 4);
      }
      op = tile(rewriter, op, nativeTile);
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
      if (bounds[i] > 0 && !(reductionCount > 1 && types[i] == mlir::utils::IteratorType::reduction) &&
          tiles[i].size() < 2 && (tiles[i].empty() || size < tiles[i].back()))
        legal.push_back(llvm::json::Object{{"kind", "TILE"}, {"axis", i}, {"size", size}});
    for (unsigned j = i + 1; j < bounds.size(); ++j)
      if (types[i] == mlir::utils::IteratorType::parallel && types[j] == mlir::utils::IteratorType::parallel)
        legal.push_back(llvm::json::Object{{"kind", "INTERCHANGE"}, {"a", i}, {"b", j}});
  }
  if (!terminal && steps < 6 && canVectorize()) legal.push_back(llvm::json::Object{{"kind", "VECTORIZE"}});
  result.observation = llvm::json::Object{{"dimensions", std::move(dimensions)},
      {"buffer_shapes", std::move(shapes)}, {"access_strides", std::move(accessStrides)},
      {"arithmetic_counts", std::move(arithmetic)}, {"projected_indexing", projected},
      {"memory_counts", llvm::json::Array{usedScalarLoads, 1}}, {"observation_version", 2},
      {"order", std::move(permutation)}, {"tiles", std::move(tileHistory)},
      {"steps", steps}, {"terminal", terminal || steps == 6}, {"vectorized", vectorized},
      {"legal_actions", std::move(legal)}};
  return result;
}
} // namespace rlmlir
