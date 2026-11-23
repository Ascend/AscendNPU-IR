//===- VectorizeLinalg.cpp - Vectorize lifted linalg functions ------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/TransformOps/VectorizeLinalg.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

namespace {
constexpr StringLiteral kTargetLabelPrefix = "hivm-vectorize-target-";
constexpr int64_t kBitsPerByte = 8;
constexpr int64_t kMinAlignedLastAxisBytes = 32;

SmallVector<ShapedType> getShapedOperandAndResultTypes(Operation *op) {
  SmallVector<ShapedType> shapedTypes;
  for (Type type :
       llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes()))
    if (auto shapedType = dyn_cast<ShapedType>(type))
      shapedTypes.push_back(shapedType);
  return shapedTypes;
}

SmallVector<int64_t> getMaxShape(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (lhs.size() < rhs.size())
    return llvm::to_vector(rhs);
  if (lhs.size() > rhs.size())
    return llvm::to_vector(lhs);
  SmallVector<int64_t> result;
  for (auto [lhsSize, rhsSize] : llvm::zip_equal(lhs, rhs))
    result.push_back(std::max(lhsSize, rhsSize));
  return result;
}

SmallVector<int64_t> getOpMaxShape(Operation *op) {
  SmallVector<int64_t> maxShape;
  for (ShapedType shapedType : getShapedOperandAndResultTypes(op))
    maxShape = getMaxShape(maxShape, shapedType.getShape());
  return maxShape;
}

SmallVector<int64_t> computeSingleAxisTileSize(int64_t rank,
                                               int64_t vectorLengthInBytes,
                                               int64_t elemWidthInBytes) {
  if (rank == 0)
    return {};
  SmallVector<int64_t> tileSizes(rank, 1);
  tileSizes[rank - 1] = vectorLengthInBytes / elemWidthInBytes;
  return tileSizes;
}

SmallVector<int64_t> computeMultiAxisTileSize(ArrayRef<int64_t> maxOpShape,
                                              int64_t vectorLengthInBytes,
                                              int64_t elemWidthInBytes,
                                              int64_t maxAxisNum,
                                              int64_t loopRank) {
  int64_t rank = static_cast<int64_t>(maxOpShape.size());
  if (rank <= 1 || maxAxisNum == 1)
    return computeSingleAxisTileSize(std::max(rank, loopRank),
                                     vectorLengthInBytes, elemWidthInBytes);

  int64_t remainBytes = vectorLengthInBytes;
  int64_t allocAxisNum = 0;
  SmallVector<int64_t> tileSizes(maxOpShape.size(), 1);
  for (int64_t i = rank - 1; i >= 0; --i) {
    if (ShapedType::isDynamic(maxOpShape[i]))
      return computeSingleAxisTileSize(rank, vectorLengthInBytes,
                                       elemWidthInBytes);
    allocAxisNum++;
    int64_t elementsLeft = std::max<int64_t>(remainBytes / elemWidthInBytes, 1);
    if (allocAxisNum == maxAxisNum) {
      tileSizes[i] = elementsLeft;
      break;
    }
    int64_t dim = std::max<int64_t>(maxOpShape[i], 1);
    tileSizes[i] = std::min(elementsLeft, dim);
    remainBytes /= tileSizes[i];
  }
  return tileSizes;
}

SmallVector<Type> getNestedDataTypes(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp->getNumRegions() == 0)
    return SmallVector<Type>(op->getOperandTypes());

  SmallVector<Type> result;
  linalgOp->getRegion(0).front().walk([&result](Operation *nestedOp) {
    llvm::append_range(result, getNestedDataTypes(nestedOp));
  });
  return result;
}

unsigned getMaxElemBitWidth(Operation *op) {
  unsigned maxWidth = 0;
  for (Type type : getNestedDataTypes(op)) {
    Type elemType = getElementTypeOrSelf(type);
    if (elemType.isIndex())
      continue;
    maxWidth = std::max(maxWidth, elemType.getIntOrFloatBitWidth());
  }
  return maxWidth;
}

bool hasReductionIterator(Operation *op) {
  if (isa<linalg::ReduceOp>(op))
    return true;
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic)
    return false;
  return llvm::any_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType type) {
                        return linalg::isReductionIterator(type);
                      });
}

bool hasNotAlignedLastAxis(Operation *op, int64_t alignBytes) {
  for (Value operand : op->getOperands()) {
    auto shapedType = dyn_cast<ShapedType>(operand.getType());
    if (!shapedType || shapedType.getRank() < 1)
      continue;
    int64_t lastSize = shapedType.getShape().back();
    int64_t lastBits = lastSize * shapedType.getElementTypeBitWidth();
    int64_t lastBytes = llvm::divideCeil(lastBits, kBitsPerByte);
    if (lastBytes % alignBytes != 0)
      return true;
  }
  return false;
}

bool hasImplicitBroadcast(linalg::GenericOp generic) {
  unsigned numLoops = generic.getNumLoops();
  for (OpOperand *input : generic.getDpsInputOperands()) {
    AffineMap map = generic.getMatchingIndexingMap(input);
    if (map.getNumResults() < numLoops || !map.isProjectedPermutation())
      return true;

    llvm::SmallBitVector projectedDims(numLoops);
    for (AffineExpr expr : map.getResults())
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
        projectedDims.set(dimExpr.getPosition());
    if (static_cast<unsigned>(projectedDims.count()) < numLoops)
      return true;
  }
  return false;
}

bool supportsMultiAxisVectorize(ArrayRef<Operation *> scope) {
  if (llvm::any_of(scope,
                   [](Operation *op) { return hasReductionIterator(op); }))
    return false;
  if (llvm::any_of(scope, [](Operation *op) {
        return hasNotAlignedLastAxis(op, kMinAlignedLastAxisBytes);
      }))
    return false;
  return llvm::none_of(scope, [](Operation *op) {
    auto generic = dyn_cast<linalg::GenericOp>(op);
    return generic && hasImplicitBroadcast(generic);
  });
}

struct TransposedInput {
  AffineMap map;
  ArrayRef<int64_t> shape;
};

std::optional<TransposedInput> findTransposedInput(linalg::GenericOp generic) {
  for (OpOperand *input : generic.getDpsInputOperands()) {
    AffineMap map = generic.getMatchingIndexingMap(input);
    if (!map.isPermutation() || map.isIdentity())
      continue;
    auto shapedType = dyn_cast<ShapedType>(input->get().getType());
    if (!shapedType)
      continue;
    return TransposedInput{map, shapedType.getShape()};
  }
  return std::nullopt;
}

SmallVector<int64_t>
computeTransposedTileSize(linalg::GenericOp generic, TransposedInput input,
                          int64_t vectorLengthInBytes, int64_t elemWidthInBytes,
                          int64_t maxAxisNum, int64_t loopRank) {
  SmallVector<int64_t> inputTiles = computeMultiAxisTileSize(
      input.shape, vectorLengthInBytes, elemWidthInBytes, maxAxisNum, loopRank);
  SmallVector<int64_t> iterationTiles(generic.getNumLoops(), 1);
  for (auto [index, expr] : llvm::enumerate(input.map.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return inputTiles;
    iterationTiles[dimExpr.getPosition()] = inputTiles[index];
  }
  return iterationTiles;
}

SmallVector<int64_t> computeRawTileSize(linalg::LinalgOp op,
                                        ArrayRef<int64_t> maxShape,
                                        int64_t vectorLengthInBytes,
                                        int64_t elemWidthInBytes,
                                        int64_t maxAxisNum) {
  if (auto generic = dyn_cast<linalg::GenericOp>(op.getOperation()))
    if (auto transpose = findTransposedInput(generic))
      return computeTransposedTileSize(generic, *transpose, vectorLengthInBytes,
                                       elemWidthInBytes, maxAxisNum,
                                       op.getNumLoops());
  return computeMultiAxisTileSize(maxShape, vectorLengthInBytes,
                                  elemWidthInBytes, maxAxisNum,
                                  op.getNumLoops());
}

SmallVector<int64_t> clampToIterationDomain(linalg::LinalgOp op,
                                            ArrayRef<int64_t> tileSizes) {
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  if (loopRanges.size() != tileSizes.size())
    return llvm::to_vector(tileSizes);

  SmallVector<int64_t> result;
  for (auto [tileSize, range] : llvm::zip_equal(tileSizes, loopRanges)) {
    bool clampable = !ShapedType::isDynamic(range) && range > 0;
    result.push_back(clampable ? std::min(tileSize, range) : tileSize);
  }
  return result;
}

bool isSupportedTileProduct(ArrayRef<int64_t> tileSizes) {
  int64_t product = 1;
  for (int64_t size : tileSizes) {
    product *= size;
    if (product > 128)
      return false;
  }
  static const llvm::DenseSet<int64_t> kSupported = {1,  2,  3,  4,  8,
                                                     16, 32, 64, 128};
  return kSupported.contains(product);
}

SmallVector<int64_t> computeTileSizes(linalg::LinalgOp op,
                                      ArrayRef<Operation *> scope,
                                      const VectorizeOptions &options) {
  unsigned elemBitWidth = std::max(getMaxElemBitWidth(op.getOperation()), 1u);
  if (elemBitWidth == 64 &&
      hacc::utils::isRegBasedArch(op->getParentOfType<ModuleOp>())) {
    SmallVector<int64_t> tiles(op.getNumLoops(), 1);
    if (!tiles.empty())
      tiles.back() = 64;
    return tiles;
  }
  if (elemBitWidth == 1) {
    SmallVector<int64_t> predicateTiles(op.getNumLoops(), 1);
    if (!predicateTiles.empty())
      predicateTiles.back() = 64;
    return predicateTiles;
  }
  int64_t elemWidthInBytes =
      std::max<int64_t>(llvm::divideCeil(elemBitWidth, kBitsPerByte), 1);
  if (elemBitWidth == 32 && !hasReductionIterator(op) &&
      hacc::utils::isRegBasedArch(op->getParentOfType<ModuleOp>())) {
    auto generic = dyn_cast<linalg::GenericOp>(op.getOperation());
    if (!generic || !findTransposedInput(generic))
      return computeSingleAxisTileSize(op.getNumLoops(), options.vectorLength,
                                       elemWidthInBytes);
  }
  int64_t maxAxisNum =
      supportsMultiAxisVectorize(scope) ? options.maxVectorizeAxes : 1;
  if (maxAxisNum == 0)
    maxAxisNum = 1;

  SmallVector<int64_t> maxShape = getOpMaxShape(op.getOperation());
  SmallVector<int64_t> tileSizes = computeRawTileSize(
      op, maxShape, options.vectorLength, elemWidthInBytes, maxAxisNum);
  tileSizes = clampToIterationDomain(op, tileSizes);
  if (maxAxisNum != 1 && !isSupportedTileProduct(tileSizes)) {
    tileSizes = computeRawTileSize(op, maxShape, options.vectorLength,
                                   elemWidthInBytes, 1);
    tileSizes = clampToIterationDomain(op, tileSizes);
  }
  if (hasReductionIterator(op.getOperation())) {
    auto ranges = op.getStaticLoopRanges();
    for (auto [tile, range] : llvm::zip_equal(tileSizes, ranges)) {
      if (ShapedType::isDynamic(range))
        continue;
      while (tile > 1 && range % tile != 0)
        --tile;
    }
  }
  return tileSizes;
}

SmallVector<int64_t>
computeReductionVectorSizes(linalg::LinalgOp op, ArrayRef<int64_t> tileSizes,
                            const VectorizeOptions &options) {
  auto sizes = llvm::to_vector(tileSizes);
  if (sizes.empty() || sizes.back() <= 1 ||
      getMaxElemBitWidth(op.getOperation()) != 32 ||
      !hacc::utils::isRegBasedArch(op->getParentOfType<ModuleOp>()))
    return sizes;
  SmallVector<unsigned> dimensions;
  op.getReductionDims(dimensions);
  if (dimensions.size() != 1 || dimensions.front() != sizes.size() - 1)
    return sizes;
  sizes.back() = std::max(sizes.back(), options.vectorLength / 4);
  return sizes;
}

struct FusionGroup {
  SmallVector<Operation *> members;
  bool fused = false;
};

SmallVector<linalg::LinalgOp> collectLinalgOps(func::FuncOp function) {
  SmallVector<linalg::LinalgOp> ops;
  function.walk([&](linalg::LinalgOp op) { ops.push_back(op); });
  return ops;
}

bool isIdentityMappedInput(linalg::LinalgOp consumer, OpOperand *input) {
  auto generic = dyn_cast<linalg::GenericOp>(consumer.getOperation());
  return generic && generic.getMatchingIndexingMap(input).isIdentity();
}

bool hasIterationSizedResults(linalg::LinalgOp op) {
  unsigned numLoops = op.getNumLoops();
  return llvm::all_of(op->getResultTypes(), [&](Type type) {
    auto shaped = dyn_cast<ShapedType>(type);
    return shaped && shaped.getRank() == static_cast<int64_t>(numLoops);
  });
}

SmallVector<SmallVector<unsigned>>
buildFusionAdjacency(ArrayRef<linalg::LinalgOp> ops) {
  DenseMap<Operation *, unsigned> ordinals;
  for (auto [index, op] : llvm::enumerate(ops)) {
    linalg::LinalgOp linalgOp = op;
    ordinals[linalgOp.getOperation()] = index;
  }

  SmallVector<SmallVector<unsigned>> adjacency(ops.size());
  for (auto [index, op] : llvm::enumerate(ops)) {
    linalg::LinalgOp linalgOp = op;
    for (OpOperand *input : linalgOp.getDpsInputOperands()) {
      Operation *producer = input->get().getDefiningOp();
      auto found = producer ? ordinals.find(producer) : ordinals.end();
      if (found == ordinals.end() || !isIdentityMappedInput(linalgOp, input))
        continue;
      if (!hasIterationSizedResults(ops[found->second]))
        continue;
      adjacency[index].push_back(found->second);
      adjacency[found->second].push_back(index);
    }
  }
  return adjacency;
}

SmallVector<unsigned>
collectComponent(unsigned seed, ArrayRef<SmallVector<unsigned>> adjacency,
                 MutableArrayRef<bool> visited) {
  SmallVector<unsigned> component;
  SmallVector<unsigned> worklist{seed};
  visited[seed] = true;
  while (!worklist.empty()) {
    unsigned current = worklist.pop_back_val();
    component.push_back(current);
    for (unsigned neighbor : adjacency[current]) {
      if (visited[neighbor])
        continue;
      visited[neighbor] = true;
      worklist.push_back(neighbor);
    }
  }
  llvm::sort(component);
  return component;
}

DenseSet<Operation *> toOperationSet(ArrayRef<unsigned> component,
                                     ArrayRef<linalg::LinalgOp> ops) {
  DenseSet<Operation *> members;
  for (unsigned index : component) {
    linalg::LinalgOp op = ops[index];
    members.insert(op.getOperation());
  }

  return members;
}

bool hasExternalUsers(linalg::LinalgOp op,
                      const DenseSet<Operation *> &members) {
  return llvm::any_of(op->getResults(), [&](Value result) {
    return llvm::any_of(result.getUsers(), [&](Operation *user) {
      return !members.contains(user);
    });
  });
}

bool consumesResultOf(linalg::LinalgOp consumer, linalg::LinalgOp producer) {
  if (!hasIterationSizedResults(producer))
    return false;
  for (OpOperand *input : consumer.getDpsInputOperands()) {
    if (!isIdentityMappedInput(consumer, input))
      continue;
    if (input->get().getDefiningOp() == producer.getOperation())
      return true;
  }
  return false;
}

bool canFuseGroup(ArrayRef<unsigned> component,
                  ArrayRef<linalg::LinalgOp> ops) {
  if (component.size() < 2)
    return false;
  if (llvm::any_of(component, [&](unsigned index) {
        linalg::LinalgOp op = ops[index];
        return hasReductionIterator(op.getOperation());
      }))
    return false;

  DenseSet<Operation *> members = toOperationSet(component, ops);
  for (unsigned index : component) {
    auto op = ops[index];
    for (Value destination : op.getDpsInits())
      if (members.contains(destination.getDefiningOp()))
        return false;
  }
  unsigned sinkCount = 0;
  for (unsigned index : component) {
    linalg::LinalgOp op = ops[index];
    bool isSink = llvm::none_of(component, [&](unsigned other) {
      return other != index && consumesResultOf(ops[other], op);
    });
    if (isSink) {
      ++sinkCount;
      if (index != component.back())
        return false;
      continue;
    }
    if (hasExternalUsers(op, members))
      return false;
  }
  return sinkCount == 1;
}

SmallVector<FusionGroup> buildFusionGroups(func::FuncOp function) {
  SmallVector<linalg::LinalgOp> ops = collectLinalgOps(function);
  if (ops.empty())
    return {};

  SmallVector<SmallVector<unsigned>> adjacency = buildFusionAdjacency(ops);
  SmallVector<bool> visited(ops.size(), false);
  SmallVector<FusionGroup> groups;
  for (unsigned seed = 0; seed < ops.size(); ++seed) {
    if (visited[seed])
      continue;
    SmallVector<unsigned> component =
        collectComponent(seed, adjacency, visited);
    FusionGroup group;
    for (unsigned index : component) {
      linalg::LinalgOp op = ops[index];
      group.members.push_back(op.getOperation());
    }
    group.fused = canFuseGroup(component, ops);
    groups.push_back(std::move(group));
  }
  return groups;
}

struct VectorizeTarget {
  Value handle;
  SmallVector<int64_t> vectorSizes;
};

class ScheduleBuilder {
public:
  ScheduleBuilder(func::FuncOp function, const VectorizeOptions &options)
      : function(function), options(options), builder(function.getContext()) {
    builder.setInsertionPointAfter(function);
    sequence = builder.create<transform::SequenceOp>(
        function.getLoc(), TypeRange(),
        transform::FailurePropagationMode::Propagate,
        builder.getType<transform::AnyOpType>(),
        [](OpBuilder &nestedBuilder, Location loc, Value) {
          nestedBuilder.create<transform::YieldOp>(loc, ValueRange());
        });
    builder.setInsertionPointToStart(sequence.getBodyBlock());
  }

  void addGroup(const FusionGroup &group);
  void emitVectorizeOps();

  transform::SequenceOp getSequence() const { return sequence; }

private:
  Value matchOperation(Operation *op);
  void addFusedGroup(ArrayRef<Operation *> members);
  void addSingleOperation(Operation *member);
  void emitCleanup();

  func::FuncOp function;
  VectorizeOptions options;
  OpBuilder builder;
  transform::SequenceOp sequence;
  SmallVector<VectorizeTarget> targets;
  int64_t labelCounter = 0;
};

Value ScheduleBuilder::matchOperation(Operation *op) {
  std::string label = (kTargetLabelPrefix + llvm::Twine(++labelCounter)).str();
  op->setAttr(label, UnitAttr::get(op->getContext()));

  DictionaryAttr labelAttr = builder.getDictionaryAttr(
      builder.getNamedAttr(label, builder.getUnitAttr()));
  return builder
      .create<transform::MatchOp>(
          sequence.getLoc(), builder.getType<transform::AnyOpType>(),
          sequence.getBodyBlock()->getArguments().front(), ArrayAttr(),
          transform::MatchInterfaceEnumAttr{}, labelAttr, DictionaryAttr{},
          TypeAttr{}, ArrayAttr{})
      .getResult();
}

void ScheduleBuilder::emitCleanup() {
  Location loc = sequence.getLoc();
  auto loopLikeAttr = transform::MatchInterfaceEnumAttr::get(
      builder.getContext(), transform::MatchInterfaceEnum::LoopLikeInterface);
  Value loopLike = builder.create<transform::MatchOp>(
      loc, builder.getType<transform::AnyOpType>(),
      sequence.getBodyBlock()->getArguments().front(), ArrayAttr(),
      loopLikeAttr, DictionaryAttr(), DictionaryAttr{}, TypeAttr{},
      ArrayAttr{});
  builder.create<transform::ApplyLoopInvariantCodeMotionOp>(loc, loopLike);

  auto bodyBuilder = [](OpBuilder &nestedBuilder, Location nestedLoc) {
    nestedBuilder.create<transform::ApplyCanonicalizationPatternsOp>(nestedLoc);
    nestedBuilder
        .create<transform::ApplyMergeConsecutiveInsertExtractSlicePatternsOp>(
            nestedLoc);
  };
  transform::ApplyPatternsOp applyPatterns =
      builder.create<transform::ApplyPatternsOp>(
          loc, sequence.getBodyBlock()->getArguments().front(), bodyBuilder);
  applyPatterns.setApplyCse(true);
  applyPatterns.setDisablePatternsAttr(builder.getArrayAttr(
      SmallVector<Attribute>{builder.getStringAttr("SimplifyTrivialLoops")}));
}

void ScheduleBuilder::addFusedGroup(ArrayRef<Operation *> members) {
  if (members.empty())
    return;

  Operation *sink = members.back();
  SmallVector<int64_t> tileSizes =
      computeTileSizes(cast<linalg::LinalgOp>(sink), members, options);
  if (tileSizes.empty()) {
    for (Operation *member : members)
      addSingleOperation(member);
    return;
  }

  Location loc = sequence.getLoc();
  Value loop;
  for (Operation *member : llvm::reverse(members)) {
    Value handle = matchOperation(member);
    if (member == sink) {
      transform::TileUsingForOp tiled =
          builder.create<transform::TileUsingForOp>(loc, handle, tileSizes);
      handle = tiled.getTiledLinalgOp();
      loop = tiled.getLoops().front();
    } else {
      transform::FuseIntoContainingOp fused =
          builder.create<transform::FuseIntoContainingOp>(loc, handle, loop);
      handle = fused.getFusedOp();
      loop = fused.getNewContainingOp();
    }
    emitCleanup();
    targets.push_back({handle, tileSizes});
  }
}

void ScheduleBuilder::addSingleOperation(Operation *member) {
  SmallVector<int64_t> tileSizes =
      computeTileSizes(cast<linalg::LinalgOp>(member), {member}, options);
  auto vectorSizes = computeReductionVectorSizes(cast<linalg::LinalgOp>(member),
                                                 tileSizes, options);
  Value handle = matchOperation(member);
  if (!tileSizes.empty()) {
    transform::TileUsingForOp tiled = builder.create<transform::TileUsingForOp>(
        sequence.getLoc(), handle, tileSizes);
    handle = tiled.getTiledLinalgOp();
    emitCleanup();
  }
  targets.push_back({handle, vectorSizes});
}

void ScheduleBuilder::addGroup(const FusionGroup &group) {
  if (group.fused) {
    addFusedGroup(group.members);
    return;
  }
  for (Operation *member : group.members)
    addSingleOperation(member);
}

void ScheduleBuilder::emitVectorizeOps() {
  for (const VectorizeTarget &target : targets) {
    builder.create<transform::VectorizeOp>(
        sequence.getLoc(), target.handle, SmallVector<Value>(),
        target.vectorSizes, nullptr,
        SmallVector<bool>(target.vectorSizes.size(), false));
  }
}

void removeTargetLabels(func::FuncOp function) {
  function.walk([&](Operation *op) {
    SmallVector<StringRef> labels;
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().strref().starts_with(kTargetLabelPrefix))
        labels.push_back(attr.getName().strref());
    for (StringRef label : labels)
      op->removeAttr(label);
  });
}

LogicalResult applySchedule(func::FuncOp function,
                            transform::SequenceOp sequence) {
  ModuleOp module = function->getParentOfType<ModuleOp>();
  LogicalResult result = transform::applyTransformNamedSequence(
      function, sequence, module, transform::TransformOptions());
  sequence->erase();
  return result;
}

std::string getFunctionName(func::FuncOp function) {
  return function.getSymName().str();
}

bool isSingleElementWideGeneric(linalg::GenericOp generic) {
  return generic.hasPureTensorSemantics() &&
         getMaxElemBitWidth(generic) == 64 &&
         llvm::all_of(generic.getStaticLoopRanges(),
                      [](int64_t size) { return size == 1; });
}

void scalarizeSingleElementWideGeneric(linalg::GenericOp generic,
                                       IRRewriter &rewriter) {
  rewriter.setInsertionPoint(generic);
  auto loc = generic.getLoc();
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  IRMapping mapping;
  for (auto [argument, operand] : llvm::zip_equal(
           generic.getBody()->getArguments(), generic->getOperands())) {
    Value scalar = operand;
    if (auto tensor = dyn_cast<RankedTensorType>(operand.getType()))
      scalar = rewriter.create<tensor::ExtractOp>(
          loc, operand, SmallVector<Value>(tensor.getRank(), zero));
    mapping.map(argument, scalar);
  }
  for (auto &operation : generic.getBody()->without_terminator()) {
    if (auto index = dyn_cast<linalg::IndexOp>(operation))
      mapping.map(index.getResult(), zero);
    else
      rewriter.clone(operation, mapping);
  }
  SmallVector<Value> results;
  for (auto [value, output] :
       llvm::zip_equal(generic.getBody()->getTerminator()->getOperands(),
                       generic.getDpsInits())) {
    auto type = cast<RankedTensorType>(output.getType());
    results.push_back(rewriter.create<tensor::InsertOp>(
        loc, mapping.lookupOrDefault(value), output,
        SmallVector<Value>(type.getRank(), zero)));
  }
  rewriter.replaceOp(generic, results);
}

void scalarizeSingleElementWideGenerics(func::FuncOp function) {
  SmallVector<linalg::GenericOp> singletons;
  function.walk([&](linalg::GenericOp generic) {
    if (isSingleElementWideGeneric(generic))
      singletons.push_back(generic);
  });
  IRRewriter rewriter(function.getContext());
  for (auto generic : singletons)
    scalarizeSingleElementWideGeneric(generic, rewriter);
}

void guardInactiveIntegerDivisors(func::FuncOp function) {
  IRRewriter rewriter(function.getContext());
  function.walk([&](Operation *operation) {
    if (!isa<arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp>(
            operation))
      return;
    auto type = dyn_cast<VectorType>(operation->getResult(0).getType());
    if (!type)
      return;
    Value mask;
    for (auto operand : operation->getOperands())
      if (auto masked = operand.getDefiningOp<vector::MaskOp>())
        mask = masked.getMask();
    if (!mask)
      return;
    rewriter.setInsertionPoint(operation);
    auto one = rewriter.create<arith::ConstantOp>(
        operation->getLoc(),
        DenseElementsAttr::get(
            type, rewriter.getIntegerAttr(type.getElementType(), 1)));
    auto divisor = rewriter.create<arith::SelectOp>(
        operation->getLoc(), mask, operation->getOperand(1), one);
    operation->setOperand(1, divisor);
  });
}

LogicalResult vectorizeFusionGroups(func::FuncOp function,
                                    ArrayRef<FusionGroup> groups,
                                    const VectorizeOptions &options) {
  ScheduleBuilder schedule(function, options);
  for (const FusionGroup &group : groups)
    schedule.addGroup(group);
  schedule.emitVectorizeOps();

  LogicalResult applied = applySchedule(function, schedule.getSequence());
  removeTargetLabels(function);
  return applied;
}

LogicalResult vectorizeFunction(func::FuncOp function,
                                const VectorizeOptions &options,
                                std::string &error) {
  scalarizeSingleElementWideGenerics(function);
  auto groups = buildFusionGroups(function);
  if (!groups.empty() &&
      failed(vectorizeFusionGroups(function, groups, options))) {
    error = "failed to vectorize function '" + getFunctionName(function) + "'";
    return failure();
  }
  guardInactiveIntegerDivisors(function);
  if (failed(flattenVectorComputations(function))) {
    error = "failed to flatten vector computations in function '" +
            getFunctionName(function) + "'";
    return failure();
  }
  if (failed(verify(function))) {
    error = "verification failed after vectorizing function '" +
            getFunctionName(function) + "'";
    return failure();
  }
  return success();
}

} // namespace

namespace {
struct LowerUnitReduction : OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                PatternRewriter &rewriter) const override {
    auto maskable = cast<vector::MaskableOpInterface>(op.getOperation());
    if (maskable.isMasked())
      return failure();
    auto type = op.getSourceVectorType();
    if (!llvm::all_of(op.getReductionDims(), [&](Attribute axis) {
          return type.getDimSize(cast<IntegerAttr>(axis).getInt()) == 1;
        }))
      return failure();
    Value source = op.getSource();
    if (auto dest = dyn_cast<VectorType>(op.getDestType()))
      source = rewriter.create<vector::ShapeCastOp>(op.getLoc(), dest, source);
    else {
      auto flat = VectorType::get({1}, type.getElementType());
      source = rewriter.create<vector::ShapeCastOp>(op.getLoc(), flat, source);
      source = rewriter.create<vector::ExtractOp>(op.getLoc(), source,
                                                  ArrayRef<int64_t>{0});
    }
    auto result = vector::makeArithReduction(rewriter, op.getLoc(),
                                             op.getKind(), source, op.getAcc());
    rewriter.replaceOp(op, result);
    return success();
  }
};

bool canCastIndexVector(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    return isa<DenseIntElementsAttr>(constant.getValue());
  if (auto broadcast = value.getDefiningOp<vector::BroadcastOp>())
    return broadcast.getSourceType().isIndex();
  auto *operation = value.getDefiningOp();
  return operation &&
         isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(operation) &&
         llvm::all_of(operation->getOperands(), canCastIndexVector);
}

FailureOr<Value> castIndexVector(Value value, VectorType target,
                                 PatternRewriter &rewriter) {
  auto loc = value.getLoc();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
    if (!dense)
      return failure();
    SmallVector<APInt> integers;
    for (auto number : dense.getValues<APInt>())
      integers.push_back(number.sextOrTrunc(target.getElementTypeBitWidth()));
    return rewriter
        .create<arith::ConstantOp>(loc,
                                   DenseIntElementsAttr::get(target, integers))
        .getResult();
  }
  if (auto broadcast = value.getDefiningOp<vector::BroadcastOp>()) {
    if (!broadcast.getSourceType().isIndex())
      return failure();
    auto scalar = rewriter.create<arith::IndexCastOp>(
        loc, target.getElementType(), broadcast.getSource());
    return rewriter.create<vector::BroadcastOp>(loc, target, scalar)
        .getResult();
  }
  auto *operation = value.getDefiningOp();
  if (!operation ||
      !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(operation))
    return failure();
  auto left = castIndexVector(operation->getOperand(0), target, rewriter);
  if (failed(left))
    return failure();
  auto right = castIndexVector(operation->getOperand(1), target, rewriter);
  if (failed(right))
    return failure();
  OperationState state(loc, operation->getName());
  state.addOperands({*left, *right});
  state.addTypes(target);
  return rewriter.create(state)->getResult(0);
}

struct FoldIndexVectorCast : OpRewritePattern<arith::IndexCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    auto source = dyn_cast<VectorType>(op.getIn().getType());
    auto target = dyn_cast<VectorType>(op.getType());
    if (!source || !target || !source.getElementType().isIndex() ||
        !canCastIndexVector(op.getIn()))
      return failure();
    auto converted = castIndexVector(op.getIn(), target, rewriter);
    if (failed(converted))
      return failure();
    rewriter.replaceOp(op, *converted);
    return success();
  }
};

struct LowerMaskedFloatingExtrema : OpRewritePattern<vector::MaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskOp op,
                                PatternRewriter &rewriter) const override {
    auto reduction = dyn_cast_or_null<vector::ReductionOp>(op.getMaskableOp());
    if (!reduction || (reduction.getKind() != vector::CombiningKind::MAXIMUMF &&
                       reduction.getKind() != vector::CombiningKind::MINIMUMF))
      return failure();
    auto type = reduction.getSourceVectorType();
    auto element = dyn_cast<FloatType>(type.getElementType());
    if (!element)
      return failure();
    auto identity = rewriter.getFloatAttr(
        element, APFloat::getInf(element.getFloatSemantics(),
                                 reduction.getKind() ==
                                     vector::CombiningKind::MAXIMUMF));
    auto neutral = rewriter.create<arith::ConstantOp>(
        op.getLoc(), DenseElementsAttr::get(type, identity));
    auto selected = rewriter.create<arith::SelectOp>(
        op.getLoc(), op.getMask(), reduction.getVector(), neutral);
    rewriter.replaceOpWithNewOp<vector::ReductionOp>(
        op, reduction.getKind(), selected, reduction.getAcc(),
        reduction.getFastmath());
    return success();
  }
};

} // namespace

LogicalResult mlir::hivm::flattenVectorComputations(func::FuncOp function) {
  auto *context = function.getContext();
  RewritePatternSet units(context);
  units.add<LowerUnitReduction, FoldIndexVectorCast>(context);
  vector::populateVectorMaskLoweringPatternsForSideEffectingOps(units);
  if (failed(applyPatternsGreedily(function, std::move(units))))
    return failure();
  RewritePatternSet reductions(context);
  vector::populateVectorMultiReductionLoweringPatterns(
      reductions, vector::VectorMultiReductionLowering::InnerReduction);
  reductions.add<LowerMaskedFloatingExtrema>(context);
  if (failed(applyPatternsGreedily(function, std::move(reductions))))
    return failure();
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  vector::populateVectorLinearizeTypeConversionsAndLegality(
      typeConverter, patterns, target, std::numeric_limits<unsigned>::max());
  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    return failure();
  RewritePatternSet folding(context);
  vector::populateShapeCastFoldingPatterns(folding);
  return applyPatternsGreedily(function, std::move(folding));
}

LogicalResult mlir::hivm::vectorizeFunctions(ArrayRef<func::FuncOp> functions,
                                             const VectorizeOptions &options,
                                             std::string &error) {
  error.clear();
  for (func::FuncOp function : functions)
    if (failed(vectorizeFunction(function, options, error)))
      return failure();
  return success();
}

DiagnosedSilenceableFailure HIVMVectorizeOp::apply(TransformRewriter &,
                                                   TransformResults &results,
                                                   TransformState &state) {
  SmallVector<func::FuncOp> functions;
  for (Operation *payloadOp : state.getPayloadOps(getVfs())) {
    auto function = dyn_cast<func::FuncOp>(payloadOp);
    if (!function)
      return emitDefiniteFailure() << "expected only function operations";
    functions.push_back(function);
  }

  VectorizeOptions options;
  options.vectorLength = getVectorLength();
  options.maxVectorizeAxes = getMaxVectorizeAxes();

  std::string error;
  if (failed(vectorizeFunctions(functions, options, error)))
    return emitDefiniteFailure() << error;

  results.set(cast<OpResult>(getResult()),
              SmallVector<Operation *>(functions.begin(), functions.end()));
  return DiagnosedSilenceableFailure::success();
}

void HIVMVectorizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getVfsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}
