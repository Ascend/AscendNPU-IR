//===--------- AutoVectorize.cpp - Auto vectorization pass ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Transforms.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <cassert>

#define DEBUG_TYPE "hfusion-auto-vectorize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_AUTOVECTORIZE
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static thread_local bool archisAscend950{false};

void applyCleanUp(OpBuilder &builder, func::FuncOp func,
                  transform::SequenceOp seqOp) {
  auto loopLikeAttr = transform::MatchInterfaceEnumAttr::get(
      builder.getContext(), transform::MatchInterfaceEnum::LoopLikeInterface);
  Value loopLikeHandle = builder
                             .create<transform::MatchOp>(
                                 builder.getInsertionPoint()->getLoc(),
                                 builder.getType<transform::AnyOpType>(),
                                 seqOp.getBodyBlock()->getArguments().front(),
                                 ArrayAttr(), loopLikeAttr, DictionaryAttr(),
                                 DictionaryAttr{}, TypeAttr{}, ArrayAttr{})
                             .getResults();
  builder.create<transform::ApplyLoopInvariantCodeMotionOp>(
      loopLikeHandle.getLoc(), loopLikeHandle);

  Value funcHandle = builder.create<transform::MatchOp>(
      builder.getInsertionPoint()->getLoc(),
      seqOp.getBodyBlock()->getArguments().front(),
      ArrayRef<StringRef>({func::FuncOp::getOperationName()}));
  auto bodyBuilder = [](OpBuilder &innerBuilder, Location loc) {
    innerBuilder.create<transform::ApplyCanonicalizationPatternsOp>(loc);
    innerBuilder
        .create<transform::ApplyMergeConsecutiveInsertExtractSlicePatternsOp>(
            loc);
  };
  transform::ApplyPatternsOp applyPatternsOp =
      builder.create<transform::ApplyPatternsOp>(funcHandle.getLoc(),
                                                 /*target=*/funcHandle,
                                                 /*bodyBuilder=*/bodyBuilder);
  applyPatternsOp.setApplyCse(true);
  applyPatternsOp.setDisablePatternsAttr(builder.getArrayAttr(
      SmallVector<Attribute>{builder.getStringAttr("SimplifyTrivialLoops")}));
}

// vectorizableOp centralizes which ops the pass treats as "vectorizable".
bool vectorizableOp(Operation *op) {
  return isa<linalg::LinalgOp>(op) || isa<hfusion::InterleaveOp>(op) ||
         isa<hfusion::DeinterleaveOp>(op);
}

bool isNonVectorizableOp(Operation *op) {
  return isa<hfusion::LoadOp, hfusion::StoreOp, tensor::ExtractOp,
             tensor::DimOp, tensor::InsertSliceOp, tensor::ExtractSliceOp,
             hfusion::GatherOp>(op) ||
         hfusion::isMatmulOps(op) || isa<hfusion::ReduceWithIndexOp>(op) ||
         hfusion::opCanFuseIntoMatmul(op) || isa<hfusion::PrintOp>(op);
}

// do not vectorize linalg within scf.for_all
// since simd_vf cannot be launched inside simt_vf
// Note: up until now scf.for_all accouts for simt
// later on in the bishengir-compile pass pipeline, it may refer to multi-core
// But better annotate scf.for_all differently as clear indicators
bool isInsideForAll(Operation *op) {
  return op->getParentOfType<scf::ForallOp>() != nullptr;
}

bool opIsOnlyVecUser(Operation *cur, Operation *prev) {
  int numUser = 0;
  bool curIsUser = false;

  for (auto user : prev->getUsers()) {
    Operation *currentUser = user;
    SmallVector<Operation *> worklist;
    worklist.push_back(currentUser);
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (vectorizableOp(op)) {
        numUser++;
        if (op == cur)
          curIsUser = true;
      } else if (!isNonVectorizableOp(op) && cur->isAncestor(user)) {
        // traverse through vectorizable ops
        for (auto result : op->getResults()) {
          for (auto indirectUser : result.getUsers()) {
            worklist.push_back(indirectUser);
          }
        }
      } else {
        return false;
      }
    }
  }
  return curIsUser && numUser == 1;
}

bool hasReductionIterator(Operation *op) {
  if (isa<linalg::ReduceOp>(op)) {
    return true;
  }
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp) {
    return false;
  }
  const auto iteratorTypes = genericOp.getIteratorTypesArray();
  return llvm::any_of(iteratorTypes, [](utils::IteratorType type) {
    return linalg::isReductionIterator(type);
  });
}

bool isValidToFuseTranspose(SmallVector<Operation *> &curVecScope,
                            linalg::TransposeOp transpose) {
  if (curVecScope.empty()) {
    return true;
  }

  Value input = transpose.getInput();
  int64_t vectorizableUsersNum = llvm::count_if(
      input.getUsers(), [](Operation *user) { return vectorizableOp(user); });
  if (vectorizableUsersNum != 1) {
    return false;
  }

  Operation *defOp = input.getDefiningOp();
  if (!defOp || curVecScope.back() != defOp) {
    return false;
  }

  for (Operation *op : curVecScope) {
    // 1. Check if the operation is a linalg::GenericOp
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp) {
      return false;
    }

    // 2. Check for elementwise semantics
    // This checks if all indexing maps are projected permutations
    // and all iterator types are parallel.
    if (!linalg::isElementwise(genericOp)) {
      return false;
    }
  }

  return true;
}

// group vectorizable ops by their scopes
void computeInitialVecScopeRanges(
    func::FuncOp func, SmallVector<SmallVector<Operation *>> &vecScopes) {
  Operation *prev = nullptr;

  func.walk([&](Operation *op) {
    // TODO: Move VF to HIVM
    if (mlir::hfusion::isInCubeScope(op))
      return;

    // If op is a linalg op, check the "inside forall" condition using the
    // linalg API.
    bool insideForAll = false;
    if (auto lOp = dyn_cast<linalg::LinalgOp>(op))
      insideForAll = isInsideForAll(lOp);

    // If op has the simt label attribute (same as original), break scope
    if (insideForAll || hivm::util::isSIMTVF(op)) {
      prev = nullptr;
      return;
    }

    if (vectorizableOp(op) && !isNonVectorizableOp(op)) {
      if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
        if (vecScopes.empty() ||
            !isValidToFuseTranspose(vecScopes.back(), transpose)) {
          vecScopes.push_back({op});
          prev = nullptr; // transpose op does not fuse with later ops
          return;
        }
        vecScopes.back().push_back(op);
        return;
      }

      // do not update prev, put fill op in a separate vec scope
      if (mlir::hfusion::isFillOp(op)) {
        // separate single fill and transpose in vec scope
        vecScopes.insert(vecScopes.begin(), {op});
        prev = nullptr;
        return;
      }
      if (!prev || prev->getParentOp() != op->getParentOp() ||
          !opIsOnlyVecUser(op, prev)) {
        vecScopes.push_back(SmallVector<Operation *>());
        prev = op;
      }
      // always put op to the last group
      auto &curScope = vecScopes[vecScopes.size() - 1];
      curScope.push_back(op);
    } else if (isNonVectorizableOp(op)) {
      // ensures the next checked vectorizable op is in a new group.
      // producer consumer fusion can skip load ops in between.
      prev = nullptr;
    }
  });
}

// try to make vec scope smaller to include less unnecessary scalar operations
void shrinkVecScope(SmallVector<SmallVector<Operation *>> &vecScopes) {
  // Linalg ops on single element tensors at the beginning and end of the vec
  // scope can be moved out. For non-linalg ops we conservatively keep them.
  for (SmallVector<Operation *> &vecScope : vecScopes) {
    size_t i = 0;
    while (i < vecScope.size()) {
      Operation *op = vecScope[i];
      // only call hfusion helpers for linalg ops
      if (auto lOp = dyn_cast<linalg::LinalgOp>(op)) {
        if (hfusion::isSingleElementLinalgOp(lOp) &&
            !isa<linalg::GenericOp>(lOp) && !isa<linalg::MapOp>(lOp))
          i++;
        else
          break;
      } else {
        break;
      }
    }

    vecScope.erase(vecScope.begin(), vecScope.begin() + i);

    // trim end
    if (vecScope.empty())
      continue;

    size_t j = vecScope.size();
    while (j > 0) {
      Operation *op = vecScope[j - 1];
      if (auto lOp = dyn_cast<linalg::LinalgOp>(op)) {
        if (hfusion::isSingleElementLinalgOp(lOp) &&
            !isa<linalg::GenericOp>(lOp) && !isa<linalg::MapOp>(lOp))
          j--;
        else
          break;
      } else {
        break;
      }
    }

    if (j < vecScope.size())
      vecScope.erase(vecScope.begin() + j, vecScope.end());
  }

  vecScopes.erase(std::remove_if(vecScopes.begin(), vecScopes.end(),
                                 [](SmallVector<Operation *> &vecScope) {
                                   return vecScope.empty();
                                 }),
                  vecScopes.end());
}

/// Tile_reduction_using_for will fail or cause bugs in some context, see issue:
/// https://codehub-y.huawei.com/CompilerKernel/BiShengCompiler/AscendNPU-IR/issues/307
/// So we still use tile_using_for instead for these context.
static bool shouldUseTileReductionUsingFor(Operation *op, OpBuilder &builder,
                                           SmallVector<int64_t> vecShape) {
  if (!isa<linalg::LinalgOp>(op))
    return false;
  auto linalgOp = cast<linalg::LinalgOp>(op);
  if (linalgOp.getNumReductionLoops() != 1)
    return false;
  // if there is more than one outputs and they are dependent with each other
  // tile-reduction-using-for will cause coredump.
  if (linalgOp.getRegionOutputArgs().size() > 1)
    return false;

  SmallVector<unsigned> reductionDims;
  linalgOp.getReductionDims(reductionDims);
  auto tilingInterface = cast<TilingInterface>(op);
  SmallVector<Range> iterationDomain =
      tilingInterface.getIterationDomain(builder);
  for (auto i : reductionDims) {
    // There is a bug if reduction dims is in the middle, don't use
    // tile-reduction-using-for
    if (i > 0 && i < linalgOp.getNumLoops() - 1)
      return false;
  }
  return true;
}

/// For reduction op, tile_reduction_using_for has better performance than
/// tile_using_for. Firstly we should tile parallel axis by tile_using_for,
/// then tile reduction axis by tile_reduction_using_for.
void tileReductionOp(
    transform::SequenceOp seqOp, OpBuilder &builder, Operation *op,
    Value &linalgOpHandle, Value &loopToBeOutlined, int64_t &consumerLoopSize,
    bool opIsLeafNode, SmallVector<int64_t> vecShape,
    SmallVector<std::pair<Value, SmallVector<int64_t>>> &linalgOpHandles) {
  assert(isa<linalg::LinalgOp>(op));
  auto reductionOp = cast<linalg::LinalgOp>(op);
  assert(reductionOp.getNumReductionLoops() > 0);
  // get parallel axis and tile.
  if (reductionOp.getNumParallelLoops() > 0) {
    SmallVector<unsigned> parallelDims;
    reductionOp.getParallelDims(parallelDims);
    SmallVector<int64_t> parallelAxisTileSize(reductionOp.getNumLoops(), 0);
    for (auto i : parallelDims) {
      parallelAxisTileSize[i] = vecShape[i];
    }
    transform::TileUsingForOp parallelAxisTilingResult =
        builder.create<transform::TileUsingForOp>(
            seqOp->getLoc(), linalgOpHandle, parallelAxisTileSize);
    linalgOpHandle = parallelAxisTilingResult.getTiledLinalgOp();
    if (opIsLeafNode) {
      // IsLeafNode means there may be other ops fused into current reduction
      // op, otherwise current reduction op will be fused into other ops.
      loopToBeOutlined = parallelAxisTilingResult.getLoops().front();
      consumerLoopSize =
          static_cast<int64_t>(reductionOp.getNumParallelLoops());
    }
  } else {
    // Tile a reduction op by tile_reduction_using_for will get a fillOp, a
    // ForOp containing splitLinalgOp and a combiningLinalgOp, if this reduction
    // op has only reduction axis, the fillOp and combiningLinalgOp will not be
    // wrapped by a loop, then they will not be outlined to a VF. So here we
    // build a fake loop that executes only once for those reduction ops which
    // have only reduction axis.
    auto tilingInterface = cast<TilingInterface>(op);
    SmallVector<Range> iterationDomain =
        tilingInterface.getIterationDomain(builder);
    Range range = iterationDomain[0];
    auto lb = getConstantIntValue(range.offset);
    auto ub = getConstantIntValue(range.size);
    assert(lb && ub);
    transform::TileUsingForOp fakeTilingResult =
        builder.create<transform::TileUsingForOp>(
            seqOp->getLoc(), linalgOpHandle, SmallVector<int64_t>{*ub - *lb});
    linalgOpHandle = fakeTilingResult.getTiledLinalgOp();
    loopToBeOutlined = fakeTilingResult.getLoops().front();
  }

  // get reduction axis and tile.
  SmallVector<unsigned> reductionDims;
  reductionOp.getReductionDims(reductionDims);
  SmallVector<int64_t> reductionAxisTileSize(reductionOp.getNumLoops(), 0);
  for (auto i : reductionDims) {
    reductionAxisTileSize[i] = vecShape[i];
  }
  transform::TileReductionUsingForOp reductionAxisTilingResult =
      builder.create<transform::TileReductionUsingForOp>(
          seqOp->getLoc(), linalgOpHandle, reductionAxisTileSize);
  linalgOpHandle = reductionAxisTilingResult.getSplitLinalgOp();
  // fillOp and combiningLinalgOp should also be vectorized, here we add
  // them into linalgOpHandles which will be vectorized after tiling and
  // fusing.
  for (auto fillOp : reductionAxisTilingResult.getFillOp())
    linalgOpHandles.push_back(std::make_pair(fillOp, vecShape));
  // If combiningLinalgOp(is a linalg.reduce op) has dyncamic shape, it
  // cannot be vectorized, so we convert it to a linalg.generic op.
  Value generalizedCombiningLinalgOp = builder.create<transform::GeneralizeOp>(
      seqOp->getLoc(), builder.getType<transform::AnyOpType>(),
      reductionAxisTilingResult.getCombiningLinalgOp());
  linalgOpHandles.push_back(
      std::make_pair(generalizedCombiningLinalgOp, vecShape));
  builder.create<transform::AnnotateOp>(seqOp.getLoc(),
                                        reductionAxisTilingResult.getForOp(),
                                        "reductionLoop", nullptr);
}

static const std::string vectorizeLabelPrefix =
    "hfusion-auto-vectorize-target-";

class AutoVectorize : public impl::AutoVectorizeBase<AutoVectorize> {
public:
  explicit AutoVectorize(const AutoVectorizeOptions &options)
      : AutoVectorizeBase(options) {}
  void runOnOperation() override;

private:
  int linalgOpCount = 0;

  Value getOpTransformHandle(Operation *op, OpBuilder &builder,
                             transform::SequenceOp seq);

  LogicalResult buildAndApplyVectorizationSchedule(
      OpBuilder &builder, func::FuncOp func,
      SmallVector<SmallVector<Operation *>> &vecScopes);

public:
  // compute tile size for `op`.
  // Use `multiAxisVectorizable` to enable multi axis vectorize. This is useful
  // when some special op in vector scope not support multi axis vectorization,
  // e.g. transpose
  SmallVector<int64_t> computeTileSize(Operation *op, unsigned elemWidthInBits,
                                       bool multiAxisVectorizable);

  Value tileAndFuseElemwiseWithTranspose(
      SmallVector<Operation *> &vecScope, transform::SequenceOp seqOp,
      SmallVector<std::pair<Value, SmallVector<int64_t>>> &linalgOpHandles,
      OpBuilder &builder);
};

Value AutoVectorize::getOpTransformHandle(Operation *op, OpBuilder &builder,
                                          transform::SequenceOp seqOp) {
  // name the linalg op uniquely so that it can be matched later
  std::string label = vectorizeLabelPrefix + std::to_string(++linalgOpCount);
  op->setAttr(label, UnitAttr::get(&getContext()));

  DictionaryAttr opAttr = builder.getDictionaryAttr(
      builder.getNamedAttr(label, builder.getUnitAttr()));
  Value linalgOpHandle =
      builder
          .create<transform::MatchOp>(
              seqOp.getLoc(), builder.getType<transform::AnyOpType>(),
              seqOp.getBodyBlock()->getArguments().front(), ArrayAttr(),
              transform::MatchInterfaceEnumAttr{}, opAttr, DictionaryAttr{},
              TypeAttr{}, ArrayAttr{})
          .getResults();

  return linalgOpHandle;
}

template <typename OpType>
bool containsOpType(SmallVector<Operation *> &vecScope) {
  return llvm::any_of(vecScope, [](Operation *op) { return isa<OpType>(op); });
}

enum class TransposeKind { Outer, Inner };

template <TransposeKind kind>
bool isTranspose(Operation *op) {
  auto transposeOp = dyn_cast<linalg::TransposeOp>(op);
  if (!transposeOp)
    return false;
  auto perm = transposeOp.getPermutation();
  int64_t rank = static_cast<int64_t>(perm.size());
  if constexpr (kind == TransposeKind::Inner) {
    return perm[rank - 1] != static_cast<long>(rank - 1);
  } else {
    return perm[rank - 1] == static_cast<long>(rank - 1);
  }
}

bool isMultiAxisVectorizableTranspose(linalg::TransposeOp op) {
  auto inputType = dyn_cast<ShapedType>(op.getInput().getType());
  if (!inputType || !inputType.hasStaticShape()) {
    return false;
  }
  // Rule 1: Should not be inner axis transpose
  if (isTranspose<TransposeKind::Inner>(op)) {
    return false;
  }

  ArrayRef<int64_t> shape = inputType.getShape();
  int64_t rank = inputType.getRank();

  // Safety check: Needs at least 2 axes to be a multi-axis candidate
  if (rank < 2)
    return false;

  // Calculate element size in bytes
  unsigned bitWidth = inputType.getElementType().getIntOrFloatBitWidth();
  uint64_t bytesPerElement = llvm::divideCeil(bitWidth, utils::kBitsToByte);

  // Rule 2: Last axis should fit in exactly 32 bytes
  int64_t lastDim = shape[rank - 1];
  if (lastDim * bytesPerElement != 32) {
    return false;
  }

  // Rule 3: The last two axes should not exceed 256 bytes
  // This helps prevent register pressure or cache line thrashing
  int64_t secondLastDim = shape[rank - 2];
  return (lastDim * secondLastDim * bytesPerElement) <= 256;
}

bool isMultiAxisVectorizableScopeWithTranspose(
    SmallVector<Operation *> &vecScope) {
  SmallVector<linalg::TransposeOp> transposes;
  for (Operation *op : vecScope) {
    if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
      transposes.push_back(transpose);
    }
  }
  if (transposes.size() != 1) {
    return false;
  }
  return isMultiAxisVectorizableTranspose(transposes.front());
}

bool hasImplicitBroadcastGeneric(const SmallVector<Operation *> &vecScope) {
  for (Operation *op : vecScope) {
    // 1. Only analyze linalg.generic operations
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp)
      continue;

    unsigned numLoops = genericOp.getNumLoops();

    // 2. Iterate through "input" operands (ignore "outputs/init" for broadcast
    // detection)
    for (OpOperand *inputOperand : genericOp.getDpsInputOperands()) {
      AffineMap map = genericOp.getMatchingIndexingMap(inputOperand);

      // 3. A broadcast is present if:
      // Case A: The rank of the operand is less than the number of loops.
      if (map.getNumResults() < numLoops) {
        return true;
      }

      // Case B: The rank is the same, but a dimension is "projected" out
      // (e.g., a map like (d0, d1) -> (0, d1) which broadcasts a row).
      // We check if the map is a permutation of a subset of dimensions.
      if (!map.isProjectedPermutation()) {
        // This catches non-standard indexing that usually implies
        // broadcasting or complex data access.
        return true;
      }

      // Case C: Explicit check for dropped dimensions in projected
      // permutations. Even if map.getNumResults() == numLoops, a dimension
      // could be replaced by a constant, which is a form of broadcasting.
      llvm::SmallBitVector projectedDims(numLoops);
      for (AffineExpr expr : map.getResults()) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
          projectedDims.set(dimExpr.getPosition());
        }
      }

      if (static_cast<unsigned>(projectedDims.count()) < numLoops) {
        // At least one loop dimension is not used to index this input.
        return true;
      }
    }
  }
  return false;
}

// it's hard for AVE to support multi-axis vectorization when last axis is not
// 32 bytes aligned
bool hasNotAlignedLastAxis(Operation *op, int64_t alignBytes) {
  for (Value operand : op->getOperands()) {
    auto shapedType = dyn_cast<ShapedType>(operand.getType());
    if (!shapedType) {
      continue;
    }
    if (shapedType.getRank() < 1) {
      continue;
    }
    ArrayRef<int64_t> shape = shapedType.getShape();
    int64_t lastSize = shape.back();
    int64_t lastBits = lastSize * shapedType.getElementTypeBitWidth();
    int64_t lastBytes = llvm::divideCeil(lastBits, utils::kBitsToByte);
    if (lastBytes % alignBytes != 0) {
      return true;
    }
  }
  return false;
}

bool supportMultiAxisVectorize(SmallVector<Operation *> &vecScope) {
  // never tile multi axes if has reduce iterator
  if (llvm::any_of(vecScope,
                   [](Operation *op) { return hasReductionIterator(op); })) {
    return false;
  }

  // never tile multi axes if last axis is not 32 bytes aligned
  if (llvm::any_of(vecScope, [](Operation *op) {
        return hasNotAlignedLastAxis(op, 32);
      })) {
    return false;
  }

  bool supported = true;
  if (containsOpType<linalg::TransposeOp>(vecScope)) {
    supported &= isMultiAxisVectorizableScopeWithTranspose(vecScope);
  }

  if (containsOpType<linalg::GenericOp>(vecScope)) {
    supported &= (!hasImplicitBroadcastGeneric(vecScope));
  }
  return supported;
}

bool isElemwiseWithTransposePattern(SmallVector<Operation *> &vecScope) {
  if (vecScope.size() <= 1) {
    return false;
  }
  // last op should be transpose
  if (!isa<linalg::TransposeOp>(vecScope.back())) {
    return false;
  }
  // previous op should be linalg::generic without reduction iterator
  return llvm::all_of(llvm::drop_end(vecScope), [&](Operation *op) {
    return isa<linalg::GenericOp>(op) && !hasReductionIterator(op);
  });
}

// get all data types for op, and if op is linalg, recursively get data types
// for inner op. It's necessary to check inner op because we have to make sure
// not to miss max element type when has type casts.
SmallVector<Type> getDataTypes(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp->getNumRegions() == 0) {
    return SmallVector<Type>(op->getOperandTypes());
  }

  SmallVector<Type> result;
  Block *body = &linalgOp->getRegion(0).front();
  body->walk(
      [&result](Operation *innerOp) { result.append(getDataTypes(innerOp)); });
  return result;
}

unsigned getMaxElemBitWidth(Operation *op) {

  unsigned maxWidth = 0;
  for (Type type : getDataTypes(op)) {
    Type elemTy = getElementTypeOrSelf(type);
    if (elemTy.isIndex()) {
      continue;
    }
    unsigned width = elemTy.getIntOrFloatBitWidth();
    maxWidth = std::max(maxWidth, width);
  }

  // 300/310 does not support 64-bit types, using 32-bit instead
  return (maxWidth == 64) ? 32 : maxWidth;
}

bool hasManyNonUnitTileSizes(ArrayRef<int64_t> tileSizes) {
  int n = 0;
  for (int64_t s : tileSizes) {
    if (s > 1) {
      n++;
    }
  }
  return n > 1;
}

Value AutoVectorize::tileAndFuseElemwiseWithTranspose(
    SmallVector<Operation *> &vecScope, transform::SequenceOp seqOp,
    SmallVector<std::pair<Value, SmallVector<int64_t>>> &linalgOpHandles,
    OpBuilder &builder) {
  bool multiAxisVectorizable = supportMultiAxisVectorize(vecScope);

  auto transpose = llvm::cast<linalg::TransposeOp>(vecScope.back());
  SmallVector<linalg::GenericOp> generics;
  for (Operation *op : llvm::drop_end(vecScope)) {
    generics.push_back(llvm::cast<linalg::GenericOp>(op));
  }

  unsigned maxElemWidth = 0;
  for (Operation *op : vecScope) {
    maxElemWidth = std::max(maxElemWidth, getMaxElemBitWidth(op));
  }

  SmallVector<int64_t> tileSize = AutoVectorize::computeTileSize(
      transpose, maxElemWidth, multiAxisVectorizable);
  SmallVector<int64_t> genericTileSize;
  for (int64_t permIdx : transpose.getPermutation()) {
    genericTileSize.push_back(tileSize[permIdx]);
  }
  LDBG("get tile size for transpose: " << utils::debugger::to_string(tileSize));
  LDBG("get tile size for generic: "
       << utils::debugger::to_string(genericTileSize));

  // get handler for last transpose op
  Value linalgOpHandle = getOpTransformHandle(transpose, builder, seqOp);

  SmallVector<int64_t> interchange;
  int64_t size = tileSize.size();
  if (tileSize.size() >= 2 && hasManyNonUnitTileSizes(tileSize)) {
    interchange.push_back(1);
    interchange.push_back(0);
    for (int64_t i = 0; i < size - 2; ++i) {
      interchange.push_back(i + 2);
    }
  }
  LDBG("interchange: " << utils::debugger::to_string(interchange));

  auto tilingResult = builder.create<transform::TileUsingForOp>(
      seqOp->getLoc(), linalgOpHandle, tileSize, interchange);
  linalgOpHandle = tilingResult.getTiledLinalgOp();
  linalgOpHandles.push_back(std::make_pair(linalgOpHandle, tileSize));

  if (interchange.size() >= 2) {
    Value vsstbLoop = tilingResult.getLoops()[1];
    builder.create<transform::AnnotateOp>(
        seqOp->getLoc(), vsstbLoop, builder.getStringAttr("unroll_for_vsstb"),
        nullptr);
  }

  Value loopToBeOutlined;
  int64_t consumerLoopSize = tilingResult.getLoops().size();
  if (consumerLoopSize > 0) {
    loopToBeOutlined = tilingResult.getLoops().front();
  }

  Value genericHandle;
  for (linalg::GenericOp op : llvm::reverse(generics)) {
    genericHandle = getOpTransformHandle(op, builder, seqOp);
    auto fuseIntoOp = builder.create<transform::FuseIntoContainingOp>(
        seqOp.getLoc(), genericHandle, loopToBeOutlined);
    loopToBeOutlined = fuseIntoOp.getNewContainingOp();
    genericHandle = fuseIntoOp.getFusedOp();
    linalgOpHandles.push_back(std::make_pair(genericHandle, genericTileSize));
  }
  return loopToBeOutlined;
}

LogicalResult AutoVectorize::buildAndApplyVectorizationSchedule(
    OpBuilder &builder, func::FuncOp func,
    SmallVector<SmallVector<Operation *>> &vecScopes) {
  builder.setInsertionPointAfter(func);

  // generate transform sequence op
  transform::SequenceOp seqOp = builder.create<transform::SequenceOp>(
      func.getLoc(), TypeRange(), transform::FailurePropagationMode::Propagate,
      builder.getType<transform::AnyOpType>(),
      [](OpBuilder &b, Location nested, Value rootH) {
        b.create<transform::YieldOp>(nested, ValueRange());
      });

  builder.setInsertionPointToStart(seqOp.getBodyBlock());

  // Memorize all generated vector loops for outlining
  SmallVector<Value> vectorLoopHandles;
  SmallVector<Value> loopPeelTargets;
  SmallVector<std::pair<Value, SmallVector<int64_t>>> linalgOpHandles;

  // Match and vectorize each vectorizable op
  for (SmallVector<Operation *> &vecScope : vecScopes) {
    if (isElemwiseWithTransposePattern(vecScope)) {
      Value fusedLoop = tileAndFuseElemwiseWithTranspose(
          vecScope, seqOp, linalgOpHandles, builder);
      applyCleanUp(builder, func, seqOp);
      vectorLoopHandles.push_back(fusedLoop);
      continue;
    }

    // record a possible target for loop peeling
    Value loopPeelingTarget;
    int64_t consumerLoopSize = 0;
    bool multiAxisVectorizable = supportMultiAxisVectorize(vecScope);
    Value loopToBeOutlined;
    // look at the vectorizable ops from bottom up, so that consumer is handled
    // before the producer
    for (Operation *curOp : llvm::reverse(vecScope)) {
      unsigned elemWidth = getMaxElemBitWidth(curOp);
      SmallVector<int64_t> tileSize =
          computeTileSize(curOp, elemWidth, multiAxisVectorizable);
      Value linalgOpHandle = getOpTransformHandle(curOp, builder, seqOp);

      // 1. Capture whether this is the start of the chain (Leaf Node)
      // We must capture this before modifying loopToBeOutlined in the fusion
      // step.
      bool isFirstOp = !loopToBeOutlined;
      // 2. Linearize the Fusion Step
      // If there is a loop, fuse into it. Update handles accordingly.
      if (!isFirstOp) {
        auto fuseIntoOp = builder.create<transform::FuseIntoContainingOp>(
            seqOp.getLoc(), linalgOpHandle, loopToBeOutlined);
        loopToBeOutlined = fuseIntoOp.getNewContainingOp();
        linalgOpHandle = fuseIntoOp.getFusedOp();
      }

      // 3. Determine Tiling Strategy
      // We always tile if it's the first op. If it's fused, we only tile if
      // specific dimension/reduction conditions are met.
      bool useReduction =
          shouldUseTileReductionUsingFor(curOp, builder, tileSize);
      bool needsStandardTiling = isFirstOp;

      if (!isFirstOp && !useReduction) {
        int64_t producerLoopSize = static_cast<int64_t>(tileSize.size());
        bool hasReduction =
            isa<linalg::LinalgOp>(curOp) &&
            cast<linalg::LinalgOp>(curOp).getNumReductionLoops() > 0;

        if (producerLoopSize > consumerLoopSize || hasReduction) {
          needsStandardTiling = true;
        }
      }

      // 4. Execute Unified Tiling Logic
      if (useReduction) {
        tileReductionOp(seqOp, builder, curOp, linalgOpHandle, loopToBeOutlined,
                        consumerLoopSize, /*opIsLeafNode=*/isFirstOp, tileSize,
                        linalgOpHandles);

        // If we fused and then did reduction tiling, disable peeling
        if (!isFirstOp)
          loopPeelingTarget = nullptr;

      } else if (needsStandardTiling) {
        auto tilingResult = builder.create<transform::TileUsingForOp>(
            seqOp->getLoc(), linalgOpHandle, tileSize);
        linalgOpHandle = tilingResult.getTiledLinalgOp();

        if (isFirstOp) {
          // Logic specific to the start of the chain (creating the initial loop
          // nest)
          consumerLoopSize =
              static_cast<int64_t>(tilingResult.getLoops().size());
          if (consumerLoopSize > 0)
            loopToBeOutlined = tilingResult.getLoops().front();
          if (consumerLoopSize > 1)
            loopPeelingTarget = tilingResult.getLoops().back();
        } else {
          // Logic specific to fused ops (cleanup peeling targets)
          loopPeelingTarget = nullptr;
        }
      }

      // 5. Unified Cleanup
      applyCleanUp(builder, func, seqOp);

      // only collect true linalg ops for later vectorization. hfusion ops
      // like Interleave/Deinterleave will still be tiled/fused/outlined but
      // dont have VectorizeOp.
      if (isa<linalg::LinalgOp>(curOp))
        linalgOpHandles.push_back(std::make_pair(linalgOpHandle, tileSize));
    }
    assert(loopToBeOutlined &&
           "Unexpected empty group of linalg op for vectorization!");
    vectorLoopHandles.push_back(loopToBeOutlined);
    if (loopPeelingTarget)
      loopPeelTargets.push_back(loopPeelingTarget);
  }

  // at the end, vectorize all linalg ops together. This ensures linalg ops
  // don't get destroied when we still need them for producer-consumer
  // relationship
  for (auto &[handle, vecSize] : linalgOpHandles) {
    if (llvm::all_of(vecSize, [](int64_t s) { return s == 1; })) {
      continue;
    }
    // vectorize the handle
    // Note: Vectorization will fail if the provided vector sizes is
    // smaller than the original input tensor sizes
    builder.create<transform::VectorizeOp>(
        seqOp.getLoc(), handle, SmallVector<Value>(), vecSize, nullptr,
        SmallVector<bool>(vecSize.size(), false));
  }

  // After vectorization but before outline, we can do some loop peeling to
  // eliminate some masks
  if (peelLoops) {
    for (Value loop : loopPeelTargets) {
      Value castedOp = builder.create<transform::CastOp>(
          seqOp.getLoc(),
          transform::OperationType::get(&getContext(),
                                        scf::ForOp::getOperationName()),
          loop);
      builder.create<transform::LoopPeelOp>(
          seqOp.getLoc(), builder.getType<transform::AnyOpType>(),
          builder.getType<transform::AnyOpType>(), castedOp);
    }
  }

  // each vf needs to be named differently
  int idx = 0;
  std::string vfNamePrefix = func.getSymName().str() + "_outlined_vf_";
  for (Value loop : vectorLoopHandles) {
    auto outlineOp = builder.create<transform::LoopOutlineOp>(
        seqOp.getLoc(),
        SmallVector<Type>{builder.getType<transform::AnyOpType>(),
                          builder.getType<transform::AnyOpType>()},
        loop, vfNamePrefix + std::to_string(idx++));
    builder.create<transform::AnnotateOp>(
        seqOp.getLoc(), outlineOp.getFunction(),
        mlir::hivm::VectorFunctionAttr::name, nullptr);
    builder.create<transform::AnnotateOp>(
        seqOp.getLoc(), outlineOp.getFunction(), "no_inline", nullptr);
    builder.create<transform::AnnotateOp>(seqOp.getLoc(), outlineOp.getCall(),
                                          mlir::hivm::VectorFunctionAttr::name,
                                          nullptr);
    builder.create<transform::AnnotateOp>(seqOp.getLoc(), outlineOp.getCall(),
                                          "no_inline", nullptr);
  }

  LLVM_DEBUG(llvm::dbgs() << "Dumping kernel and schedule:\n");
  LLVM_DEBUG(func.dump());
  LLVM_DEBUG(seqOp.dump());

  LogicalResult result = transform::applyTransformNamedSequence(
      func, seqOp, func->getParentOfType<ModuleOp>(),
      transform::TransformOptions());

  seqOp->erase();
  return result;
}

SmallVector<int64_t> getMaxShape(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (lhs.size() < rhs.size()) {
    return llvm::to_vector(rhs);
  }
  if (lhs.size() > rhs.size()) {
    return llvm::to_vector(lhs);
  }
  SmallVector<int64_t> result;
  for (auto [s1, s2] : llvm::zip(lhs, rhs)) {
    result.push_back(std::max(s1, s2));
  }
  return result;
}

SmallVector<int64_t> computeSingleAxisTileSize(int64_t rank,
                                               int64_t vectorLengthInBytes,
                                               int64_t elemWidthInBytes) {
  if (rank == 0) {
    return {};
  }
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
  if (rank <= 1 || maxAxisNum == 1) {
    // make full usage of VL when there is only one dimension vectorize
    // TODO: have to support this, otherwise AVE will fail in some cases.
    //       e.g. tiling special shapes like <23xi8> or <1xi16>.
    //       Should figure out why and fix it.
    return computeSingleAxisTileSize(std::max(rank, loopRank),
                                     vectorLengthInBytes, elemWidthInBytes);
  }
  // TODO: need fix tileSizes when loopRank is not equal to rank
  int64_t remainBytes = vectorLengthInBytes;
  int64_t allocAxisNum = 0;
  SmallVector<int64_t> tileSizes(maxOpShape.size(), 1);
  for (int64_t i = rank - 1; i >= 0; --i) {
    if (ShapedType::isDynamic(maxOpShape[i])) {
      // fallback to single axis tiling if exists dynamic shape in multi-axis
      // tiling
      return computeSingleAxisTileSize(rank, vectorLengthInBytes,
                                       elemWidthInBytes);
    }
    allocAxisNum++;
    if (allocAxisNum == maxAxisNum) {
      tileSizes[i] = remainBytes / elemWidthInBytes;
      break;
    } else {
      tileSizes[i] = std::min(remainBytes / elemWidthInBytes, maxOpShape[i]);
      remainBytes /= tileSizes[i];
    }
  }
  return tileSizes;
}

SmallVector<int64_t> computeTransposedMultiAxisTileSize(
    linalg::TransposeOp transpose, ArrayRef<int64_t> maxOpShape,
    int64_t vectorLengthInBytes, int64_t elemWidthInBytes, int64_t maxAxisNum,
    int64_t loopRank) {
  ArrayRef<int64_t> perm = transpose.getPermutation();
  Value input = transpose.getInput();
  ArrayRef<int64_t> shape = llvm::cast<ShapedType>(input.getType()).getShape();
  SmallVector<int64_t> outputTileSizes = computeMultiAxisTileSize(
      shape, vectorLengthInBytes, elemWidthInBytes, maxAxisNum, loopRank);
  SmallVector<int64_t> inputTileSizes;
  for (int64_t idx : perm) {
    inputTileSizes.push_back(outputTileSizes[idx]);
  }
  LDBG("transposed tile sizes: " << utils::debugger::to_string(inputTileSizes));
  return inputTileSizes;
}

SmallVector<ShapedType> getShapedOperandAndResultTypes(Operation *op) {
  SmallVector<Type> allTypes;
  allTypes.append(llvm::to_vector(op->getOperandTypes()));
  allTypes.append(llvm::to_vector(op->getResultTypes()));

  SmallVector<ShapedType> shapedTypes;
  for (Type type : allTypes) {
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
      shapedTypes.push_back(shapedType);
    }
  }
  return shapedTypes;
}

bool isSupportTileSizeForPGE(ArrayRef<int64_t> tileSizes) {
  int64_t accum = 1;
  for (int64_t s : tileSizes) {
    accum = accum * s;
  }
  // Currently the mult-axes transpose should be lower to
  DenseSet<int64_t> supportedSizes = {1, 2, 3, 4, 8, 16, 32, 64, 128};
  return supportedSizes.contains(accum);
}

SmallVector<int64_t>
AutoVectorize::computeTileSize(Operation *op, unsigned elemWidthInBits,
                               bool multiAxisVectorizable) {
  // Determine max shape for op.
  SmallVector<int64_t> opMaxShape;
  for (ShapedType shapedType : getShapedOperandAndResultTypes(op)) {
    opMaxShape = getMaxShape(opMaxShape, shapedType.getShape());
  }

  LDBG("-----------------------------");
  LDBG("compute tile sizes for op: " << *op);
  LDBG("opMaxShape: " << utils::debugger::to_string(opMaxShape));
  LDBG("elem bit width for op: " << elemWidthInBits);
  assert(elemWidthInBits > 0 && "Invalid elem width!");

  int64_t elemWidthInBytes =
      llvm::divideCeil(elemWidthInBits, mlir::utils::INTR_BITS_PER_BYTE);
  int64_t maxAxisNum = (multiAxisVectorizable ? maxVectorizeAxes : 1);
  LDBG("maxAxisNum: " << maxAxisNum);

  // Use getNumLoops to calc tileSizes when broadcast and reduce are fused in
  // one LinalgOp
  int64_t loopRank = -1;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (linalgOp) {
    loopRank = linalgOp.getNumLoops();
  }
  SmallVector<int64_t> tileSizes;
  if (isTranspose<TransposeKind::Outer>(op)) {
    // for outer-axis transpose, compute tile size based on input instead of
    // output, because we want to ensure the input is continuous on memory
    auto transpose = llvm::cast<linalg::TransposeOp>(op);
    tileSizes = computeTransposedMultiAxisTileSize(
        transpose, opMaxShape, vectorLength, elemWidthInBytes, maxAxisNum,
        loopRank);
  } else {
    tileSizes = computeMultiAxisTileSize(
        opMaxShape, vectorLength, elemWidthInBytes, maxAxisNum, loopRank);
  }

  if (maxAxisNum != 1 && !isSupportTileSizeForPGE(tileSizes)) {
    // TODO: remove this logic after full support for PLT in AVE
    tileSizes = computeMultiAxisTileSize(
        opMaxShape, vectorLength, elemWidthInBytes, /*maxAxisNum=*/1, loopRank);
  }

  LDBG("tileSizes: " << utils::debugger::to_string(tileSizes));
  return tileSizes;
}

bool isCubeFunc(func::FuncOp func) {
  auto fusionKind = mlir::hfusion::tryGetFusionKind(func);
  return fusionKind.has_value() &&
         (fusionKind.value() == mlir::hfusion::FusionKind::ShallowCV ||
          fusionKind.value() == mlir::hfusion::FusionKind::SingleCube);
}

SmallVector<func::FuncOp> collectVectorizableFunc(ModuleOp module) {
  SmallVector<func::FuncOp> funcList;
  module->walk([&funcList](func::FuncOp func) {
    if (!hacc::utils::isDevice(func)) {
      return WalkResult::advance();
    }
    if (mlir::scope::utils::isManualVFScope(func)) {
      LDBG("skip manual vec scope func: " << func.getName());
      return WalkResult::skip();
    }
    if (isCubeFunc(func)) {
      // Skip vectorization for cube func
      LDBG("skip cube func: " << func.getName());
      return WalkResult::skip();
    }
    funcList.push_back(func);
    return WalkResult::advance();
  });
  return funcList;
}

void AutoVectorize::runOnOperation() {
  ModuleOp op = getOperation();
  archisAscend950 = hacc::utils::isAscend950(op);
  MLIRContext *context = op->getContext();
  OpBuilder builder(context);

  LLVM_DEBUG(llvm::dbgs() << "Dumping kernel before scheduling:");
  LLVM_DEBUG(op->dump(););

  if (treeReduce) {
    RewritePatternSet treeReducePatterns(context);
    mlir::GreedyRewriteConfig config;
    config.maxIterations = 1;
    config.maxNumRewrites = 1;
    hfusion::populateTreeReducePattern(treeReducePatterns);
    (void)applyPatternsGreedily(op, std::move(treeReducePatterns), config);
  }
  for (func::FuncOp func : collectVectorizableFunc(op)) {
    // Split all vectorizable op in the function into groups representing vec
    // scopes
    SmallVector<SmallVector<Operation *>> vecScopes;
    computeInitialVecScopeRanges(func, vecScopes);
    shrinkVecScope(vecScopes);

    // Generate transform ops to vectorize each recorded op
    // apply the transforms in one shot
    if (failed(buildAndApplyVectorizationSchedule(builder, func, vecScopes)))
      func->emitError("Failed to vectorize the function.");
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createHFusionAutoVectorizePass(
    const AutoVectorizeOptions &options) {
  return std::make_unique<AutoVectorize>(options);
}
