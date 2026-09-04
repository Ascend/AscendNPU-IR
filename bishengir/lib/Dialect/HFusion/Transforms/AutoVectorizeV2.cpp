//===--------- AutoVectorizeV2.cpp - Auto vectorization pass
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFStackInfo.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/TilingUtils.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Attrs.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/PlanContext.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Verify.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Utils/Utils.h"
#include "bishengir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>

#define DEBUG_TYPE "hfusion-auto-vectorize-v2"

namespace mlir {
#define GEN_PASS_DEF_AUTOVECTORIZEV2
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::analysis;
using namespace mlir::hfusion;
using mlir::hfusion::RetriedOptions;

namespace {

/// Restore static shape information lost after MCF cleanup:
///
///   static tensor -> tensor.cast -> dynamic tensor
///                 -> scf.for iter_arg -> tensor.insert_slice
///
/// Limit the workaround to the observed softmax tree-reduction accumulators:
/// tensor<1xNx32xf32> -> tensor<1xNx?xf32>, where N is 16, 8, 4, or 2,
/// updated at [0, iv, 0] inside the 19x32 row loop. General tiling and
/// single-iteration-loop cleanup remain unchanged.
static constexpr StringLiteral kRefinedLoopCarriedTensorShape =
    "__hfusion_av2_refined_loop_carried_tensor_shape";
static constexpr int64_t kTargetTreeReduceStageSizes[] = {16, 8, 4, 2};

static bool isRefinedLoopCarriedTensorShape(Operation *op) {
  return op && op->hasAttr(kRefinedLoopCarriedTensorShape);
}

static bool isInsideTargetTreeReduceRowLoop(scf::ForOp forOp) {
  auto parent = forOp->getParentOfType<scf::ForOp>();
  if (!parent || parent.getNumRegionIterArgs() != 1 ||
      !isConstantIntValue(parent.getLowerBound(), 0) ||
      !isConstantIntValue(parent.getUpperBound(), 19) ||
      !isConstantIntValue(parent.getStep(), 1))
    return false;
  auto type =
      dyn_cast<RankedTensorType>(parent.getRegionIterArgs().front().getType());
  return type && type.getRank() == 2 && type.getElementType().isF32() &&
         type.getDimSize(0) == 19 && type.getDimSize(1) == 32;
}

static bool isTargetStaticSubvectorLoop(scf::ForOp forOp,
                                        RankedTensorType staticType,
                                        RankedTensorType dynamicType,
                                        tensor::InsertSliceOp yieldInsert) {
  if (staticType.getRank() != 3 || dynamicType.getRank() != 3 ||
      !staticType.getElementType().isF32() || staticType.getDimSize(0) != 1 ||
      dynamicType.getDimSize(0) != 1 ||
      staticType.getDimSize(1) != dynamicType.getDimSize(1) ||
      !llvm::is_contained(kTargetTreeReduceStageSizes,
                          staticType.getDimSize(1)) ||
      staticType.getDimSize(2) != 32 || !dynamicType.isDynamicDim(2) ||
      llvm::count_if(dynamicType.getShape(),
                     [](int64_t dim) { return ShapedType::isDynamic(dim); }) !=
          1 ||
      !isConstantIntValue(forOp.getLowerBound(), 0) ||
      !isConstantIntValue(forOp.getUpperBound(), staticType.getDimSize(1)) ||
      !isConstantIntValue(forOp.getStep(), 1) ||
      !isInsideTargetTreeReduceRowLoop(forOp))
    return false;

  auto sourceType = dyn_cast<RankedTensorType>(yieldInsert.getSourceType());
  if (!sourceType || sourceType.getRank() != 3 ||
      sourceType.getDimSize(0) != 1 || sourceType.getDimSize(1) != 1 ||
      (!sourceType.isDynamicDim(2) &&
       sourceType.getDimSize(2) != staticType.getDimSize(2)))
    return false;

  SmallVector<OpFoldResult> offsets = yieldInsert.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = yieldInsert.getMixedSizes();
  SmallVector<OpFoldResult> strides = yieldInsert.getMixedStrides();
  return offsets.size() == 3 && isConstantIntValue(offsets[0], 0) &&
         offsets[1].dyn_cast<Value>() == forOp.getInductionVar() &&
         isConstantIntValue(offsets[2], 0) && sizes.size() == 3 &&
         isConstantIntValue(sizes[0], 1) && isConstantIntValue(sizes[1], 1) &&
         isConstantIntValue(sizes[2], staticType.getDimSize(2)) &&
         strides.size() == 3 && llvm::all_of(strides, [](OpFoldResult stride) {
           return isConstantIntValue(stride, 1);
         });
}

struct RefineStaticLoopCarriedTensorPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  static bool isLoopAccumulatorUse(OpOperand &use, scf::ForOp forOp,
                                   tensor::InsertSliceOp yieldInsert,
                                   unsigned &extractCount) {
    Operation *owner = use.getOwner();
    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(owner)) {
      SmallVector<OpFoldResult> offsets = extractOp.getMixedOffsets();
      SmallVector<OpFoldResult> sizes = extractOp.getMixedSizes();
      SmallVector<OpFoldResult> strides = extractOp.getMixedStrides();
      bool isTarget =
          extractOp.getSource() == use.get() && offsets.size() == 3 &&
          isConstantIntValue(offsets[0], 0) &&
          offsets[1].dyn_cast<Value>() == forOp.getInductionVar() &&
          isConstantIntValue(offsets[2], 0) && sizes.size() == 3 &&
          isConstantIntValue(sizes[0], 1) && isConstantIntValue(sizes[1], 1) &&
          isConstantIntValue(sizes[2], 32) && strides.size() == 3 &&
          llvm::all_of(strides, [](OpFoldResult stride) {
            return isConstantIntValue(stride, 1);
          });
      extractCount += isTarget;
      return isTarget;
    }
    auto insertOp = dyn_cast<tensor::InsertSliceOp>(owner);
    return insertOp == yieldInsert && insertOp.getDest() == use.get();
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (forOp.getNumRegionIterArgs() != 1)
      return failure();
    auto initCast = forOp.getInitArgs().front().getDefiningOp<tensor::CastOp>();
    if (!initCast || !initCast.getResult().hasOneUse())
      return failure();

    auto staticType =
        dyn_cast<RankedTensorType>(initCast.getSource().getType());
    auto dynamicType = dyn_cast<RankedTensorType>(initCast.getType());
    if (!staticType || !dynamicType || !staticType.hasStaticShape() ||
        dynamicType.hasStaticShape() ||
        !tensor::preservesStaticInformation(dynamicType, staticType))
      return failure();

    BlockArgument iterArg = forOp.getRegionIterArgs().front();
    auto yieldInsert =
        forOp.getYieldedValues().front().getDefiningOp<tensor::InsertSliceOp>();
    unsigned extractCount = 0;
    if (!yieldInsert || !yieldInsert.getResult().hasOneUse() ||
        yieldInsert.getDest() != iterArg ||
        !isTargetStaticSubvectorLoop(forOp, staticType, dynamicType,
                                     yieldInsert) ||
        llvm::any_of(iterArg.getUses(),
                     [&](OpOperand &use) {
                       return !isLoopAccumulatorUse(use, forOp, yieldInsert,
                                                    extractCount);
                     }) ||
        extractCount != 1)
      return failure();

    rewriter.modifyOpInPlace(forOp, [&] {
      forOp.getInitArgsMutable()[0].set(initCast.getSource());
      iterArg.setType(staticType);
      forOp.getResult(0).setType(staticType);
      forOp->setAttr(kRefinedLoopCarriedTensorShape, rewriter.getUnitAttr());
    });
    rewriter.modifyOpInPlace(yieldInsert, [&] {
      yieldInsert.getResult().setType(staticType);
      yieldInsert->setAttr(kRefinedLoopCarriedTensorShape,
                           rewriter.getUnitAttr());
    });
    rewriter.eraseOp(initCast);
    return success();
  }
};

struct FoldRefinedLoopCarriedTensorDimPattern
    : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(dimOp.getSource().getType());
    std::optional<int64_t> dim = dimOp.getConstantIndex();
    if (!isRefinedLoopCarriedTensorShape(dimOp.getSource().getDefiningOp()) ||
        !type || !dim || type.isDynamicDim(*dim))
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(dimOp,
                                                        type.getDimSize(*dim));
    return success();
  }
};

struct RefineLoopCarriedTensorInsertSlicePattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (!isRefinedLoopCarriedTensorShape(insertOp) &&
        !isRefinedLoopCarriedTensorShape(
            insertOp.getSource().getDefiningOp()) &&
        !isRefinedLoopCarriedTensorShape(insertOp.getDest().getDefiningOp()))
      return failure();
    auto destType = dyn_cast<RankedTensorType>(insertOp.getDest().getType());
    auto sourceType =
        dyn_cast<RankedTensorType>(insertOp.getSource().getType());
    if (!destType || !sourceType || !destType.hasStaticShape() ||
        static_cast<size_t>(sourceType.getRank()) !=
            insertOp.getMixedSizes().size())
      return failure();

    bool changed = insertOp.getType() != destType;
    SmallVector<OpFoldResult> sizes(insertOp.getMixedSizes());
    for (auto [dim, size] : llvm::enumerate(sizes)) {
      if (sourceType.isDynamicDim(dim) || isa<Attribute>(size))
        continue;
      if (std::optional<int64_t> constant = getConstantIntValue(size)) {
        size = rewriter.getIndexAttr(*constant);
        changed = true;
      }
    }
    if (!changed)
      return failure();

    auto newInsert = rewriter.create<tensor::InsertSliceOp>(
        insertOp.getLoc(), insertOp.getSource(), insertOp.getDest(),
        insertOp.getMixedOffsets(), sizes, insertOp.getMixedStrides());
    newInsert->setAttr(kRefinedLoopCarriedTensorShape, rewriter.getUnitAttr());
    rewriter.replaceOp(insertOp, newInsert);
    return success();
  }
};

struct RefineLoopCarriedTensorExpandShapePattern
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    if (!isRefinedLoopCarriedTensorShape(expandOp.getSrc().getDefiningOp()))
      return failure();
    auto sourceType = dyn_cast<RankedTensorType>(expandOp.getSrc().getType());
    RankedTensorType resultType = expandOp.getResultType();
    if (!sourceType || !sourceType.hasStaticShape() ||
        resultType.hasStaticShape())
      return failure();

    SmallVector<int64_t> shape;
    for (OpFoldResult dim : expandOp.getMixedOutputShape()) {
      std::optional<int64_t> constant = getConstantIntValue(dim);
      if (!constant)
        return failure();
      shape.push_back(*constant);
    }
    auto staticType = RankedTensorType::get(shape, resultType.getElementType(),
                                            resultType.getEncoding());
    if (tensor::CollapseShapeOp::inferCollapsedType(
            staticType, expandOp.getReassociationMaps())
            .getShape() != sourceType.getShape())
      return failure();
    rewriter.modifyOpInPlace(expandOp, [&] {
      expandOp.getOutputShapeMutable().assign(ValueRange{});
      expandOp.setStaticOutputShape(shape);
      expandOp.getResult().setType(staticType);
    });
    return success();
  }
};

static LogicalResult refineStaticLoopCarriedTensorTypes(func::FuncOp func) {
  RewritePatternSet patterns(func.getContext());
  patterns.add<RefineStaticLoopCarriedTensorPattern,
               FoldRefinedLoopCarriedTensorDimPattern,
               RefineLoopCarriedTensorInsertSlicePattern,
               RefineLoopCarriedTensorExpandShapePattern>(func.getContext());
  GreedyRewriteConfig config;
  config.fold = false;
  config.cseConstants = false;
  config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
  LogicalResult result =
      applyPatternsGreedily(func, std::move(patterns), config);
  func.walk(
      [](Operation *op) { op->removeAttr(kRefinedLoopCarriedTensorShape); });
  return result;
}

static bool isCubeScopeOp(Operation *op) {
  auto scopeOp = dyn_cast<scope::ScopeOp>(op);
  if (!scopeOp)
    return false;

  auto attr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (!attr)
    return false;

  return attr.getTcoretype() == mlir::hivm::TCoreType::CUBE;
}

/// If two fusable ops are conflict with each other, they cannot be fused into
/// the same VF:
/// 1. The producer(upstream op) and consumer(downstream op) of
///    NonVectorizableOp are confilict with each other. For example:
///       A(FusableOp)
///       B(NonVectorizableOp, use A)
///       C(FusableOp, use B)
///    Then A and C are confilict with each other
/// 2. The previous and following of scf.for/hivm.hir.sync_block_wait/
///    hivm.hir.sync_block_set are confilict with each other. For example:
///       A(FusableOp)
///       hivm.hir.sync_block_wait
///       B(FusableOp)
///    Then A and B are confilict with each other

// A fusable op is output node when its all users is NonVectorizableOp or
// terminator op.
static bool isFusableOutputNode(Operation *op, Block *block) {
  if (!isFusableOp(op))
    return false;
  if (op->getUsers().empty())
    return true;
  for (Operation *user : op->getUsers()) {
    if (isa<hfusion::AssertOp>(user))
      continue;
    if (isa<annotation::MarkOp>(user))
      continue;
    if (!(isNonVectorizableOp(user) || !isOpInBlock(user, block) ||
          isa<scf::YieldOp, func::ReturnOp, scope::ReturnOp>(user) ||
          isa<bufferization::BufferizationDialect>(user->getDialect())))
      return false;
  }
  return true;
}

static void interchangeForLeafNodes(SmallVector<int64_t> commonAxis,
                                    FusableOpInfo &leafNodeInfo) {
  leafNodeInfo.tileInterchange = commonAxis;
  for (unsigned i = 0; i < leafNodeInfo.numLoops; i++) {
    if (!llvm::is_contained(commonAxis, i)) {
      leafNodeInfo.tileInterchange.push_back(i);
    }
  }
}

static SmallVector<unsigned> getTreeReductionParallelDims(Operation *op,
                                                          bool treeReduce) {
  if (!treeReduce || !hfusion::shouldUseTreeReduction(op))
    return {};

  SmallVector<unsigned> parallelDims;
  cast<linalg::LinalgOp>(op).getParallelDims(parallelDims);
  return parallelDims;
}

/// Tree reduction first tiles only its parallel dimensions and returns that
/// loop nest for sibling fusion.  A regular elementwise sibling normally tiles
/// in iteration-space order instead.  Keep such siblings in one fused node
/// only when they can expose the same parallel loop nest.
static bool
canShareTreeReductionSiblingLoop(Operation *candidate,
                                 const std::shared_ptr<FusedNode> &node,
                                 PlanContext &ctx, bool treeReduce) {
  bool candidateUsesRegisterTree =
      treeReduce && hfusion::shouldUseRegisterTreeReduction(candidate);
  bool nodeUsesRegisterTree =
      treeReduce && llvm::any_of(node->ops(), [](Operation *op) {
        return hfusion::shouldUseRegisterTreeReduction(op);
      });

  // The direct register lowering replaces its containing tiled loop.  Keep
  // that loop dedicated to the reduction so no sibling computation can be
  // dropped by the replacement.  This is a structural rule, independent of
  // the concrete reduction shape.
  if (candidateUsesRegisterTree || nodeUsesRegisterTree)
    return false;

  Operation *treeReduction = nullptr;
  SmallVector<unsigned> candidateTreeDims =
      getTreeReductionParallelDims(candidate, treeReduce);
  for (Operation *fusedOp : node->ops()) {
    if (!getTreeReductionParallelDims(fusedOp, treeReduce).empty()) {
      treeReduction = fusedOp;
      break;
    }
  }
  // Downstream lowering is not correctness-safe when two distinct
  // split-reduction trees share one containing loop. Keep at most one tree in
  // a fused node; compatible elementwise siblings may still share its loop.
  if (treeReduction && !candidateTreeDims.empty())
    return false;
  if (!treeReduction && !candidateTreeDims.empty())
    treeReduction = candidate;
  if (!treeReduction)
    return true;

  SmallVector<unsigned> parallelDims =
      getTreeReductionParallelDims(treeReduction, treeReduce);
  const FusableOpInfo &treeInfo = ctx.getInfo(treeReduction);
  auto isCompatible = [&](Operation *op) {
    const FusableOpInfo &info = ctx.getInfo(op);
    if (info.numLoops != treeInfo.numLoops)
      return false;
    for (unsigned dim : parallelDims) {
      if (info.shape[dim] != treeInfo.shape[dim])
        return false;
    }

    SmallVector<unsigned> opParallelDims =
        getTreeReductionParallelDims(op, treeReduce);
    if (!opParallelDims.empty())
      return opParallelDims == parallelDims;

    // tile_reduction_using_for also exposes a fixed parallel-only loop nest
    // and does not consume tileInterchange.  It can share the tree sibling's
    // loop only when both reductions expose the same parallel dimensions.
    if (hfusion::shouldUseTileReductionUsingForV2(op)) {
      SmallVector<unsigned> fixedParallelDims;
      cast<linalg::LinalgOp>(op).getParallelDims(fixedParallelDims);
      return fixedParallelDims == parallelDims;
    }
    return true;
  };

  return isCompatible(candidate) && llvm::all_of(node->leafOps(), isCompatible);
}

static void
alignTreeReductionSiblingLoops(const std::shared_ptr<FusedNode> &node,
                               PlanContext &ctx, bool treeReduce) {
  SmallVector<unsigned> parallelDims;
  for (Operation *leaf : node->leafOps()) {
    parallelDims = getTreeReductionParallelDims(leaf, treeReduce);
    if (!parallelDims.empty())
      break;
  }
  if (parallelDims.empty())
    return;

  SmallVector<int64_t> commonAxes(parallelDims.begin(), parallelDims.end());
  for (Operation *leaf : node->leafOps())
    interchangeForLeafNodes(commonAxes, ctx.getInfo(leaf));
}

static bool isProducerConsumedImpl(Operation *target, Operation *source,
                                   DenseSet<Value> &visited) {
  if (!target || !source) {
    return false;
  }
  for (Value targetResult : target->getResults()) {
    if (visited.contains(targetResult)) {
      continue;
    }
    visited.insert(targetResult);
    // dfs check if any of the target result users is consumed by source op
    bool consumed =
        llvm::any_of(targetResult.getUsers(), [&](Operation *resultUser) {
          return source->isAncestor(resultUser) ||
                 isProducerConsumedImpl(resultUser, source, visited);
        });
    if (consumed) {
      return true;
    }
  }
  return false;
}

static bool isProducerConsumed(Operation *target, Operation *source) {
  DenseSet<Value> visited;
  if (target->isBeforeInBlock(source)) {
    return isProducerConsumedImpl(target, source, visited);
  } else {
    return isProducerConsumedImpl(source, target, visited);
  }
}

// fuse sibling will clone all users of those front siblings behind fused loop
// which will cause existing handle lost or IR order changed. So we move those
// front siblings and their users to their new positions after fuse. For
// example: we have 4 nodes in order A B C(use A) D, and A and D should fuse
// sibling, then we move A before D and move C hehind D, after moving the order
// will be B A D C(use A).

/// Returns true if an op's results are used by "many" distinct users.
/// We count distinct owning operations across all result values.
static bool hasManyUsers(Operation *op, unsigned threshold = 2) {
  if (!op)
    return false;

  DenseSet<Operation *> users;
  for (Value res : op->getResults()) {
    for (OpOperand &use : res.getUses())
      users.insert(use.getOwner());
  }
  return users.size() >= threshold;
}

/// Pre validation for fusion opportunity of Linalg's
/// tileAndFuseFirstExtractUse.
///
/// Returns true if all consumers of the producer will fuse into a single loop.
/// When consumers fuse into different loops (different fusedNode labels), the
/// producer has no valid fusion opportunity and should remain a standalone op.
static bool hasFusionOpportunity(Operation *producer, PlanContext &ctx) {
  if (!producer)
    return false;

  auto tileableProducer = dyn_cast<TilingInterface>(producer);
  if (!tileableProducer)
    return false;

  if (llvm::any_of(tileableProducer->getUsers(), [&](Operation *user) {
        return !ctx.hasOpInfo(user) || !ctx.getInfo(user).fusedNode;
      })) {
    // Sanity check for nullptr
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "======== FusionOpportunity ========\n");
  LLVM_DEBUG(llvm::dbgs() << "producer: " << *producer << "\n");
  // If all consumers share the same fusedNode (loop), producer can fuse there.
  std::set<std::string> consumerLabels;
  llvm::for_each(tileableProducer->getUsers(), [&](Operation *user) {
    const std::string &fusedNodeLabel = ctx.getInfo(user).fusedNode->getLabel();
    LLVM_DEBUG(llvm::dbgs() << "fusedNodeLabel: " << fusedNodeLabel << "\n");
    consumerLabels.insert(fusedNodeLabel);
  });
  return consumerLabels.size() == 1;
}

// Among the producer's fusable users, pick the FusedNode whose ops
// appear earliest in the block and that passes all plan-level checks.
// Choosing the earliest minimizes disruption to the original op order;
// skipping non-fusable users avoids missing any viable fusion chance.
static std::shared_ptr<FusedNode>
findBestFusedNodeForProducer(Block *block, Operation *producer,
                             PlanContext &ctx, int64_t vectorLength) {
  // Since the producers and consumers of a transpose op have opposite axes, we
  // cannot fuse them into the same fusedNode. For vsstb pattern transpose, we
  // fuse the producers into this transpose op; for other pattern transpose, we
  // fuse this transpose op into its consumer ops. For example: if we have user
  // chain: op1 -> vsstb transpose -> op2, we will fuse op1 into transpose op;
  // if we have user chain: op1 -> normal transpose -> op2, we will fuse
  // transpose op into op2
  if (isVsstbPatternTransposeOp(producer))
    return nullptr;

  // Multi-consumer fusion (hasManyUsers && !hasFusionOpportunity) requires the
  // containing loop to yield the tiled result back to outside consumers, but
  // upstream replaceForWithNewSignature only supports linalg::GenericOp.
  //
  // Allowing fusion for a non-GenericOp producer leaves both the untiled
  // original and the tiled clone alive with the same label, causing vectorize
  // to abort on the untiled one (its iteration space no longer matches the tile
  // sizes).
  //
  // So bail out for non-GenericOp. Remove this guard once
  // replaceForWithNewSignature supports more producer types. The producer
  // stays in place regardless.
  if ((!ctx.getEnableMultipleConsumerFusion() ||
       !isa<linalg::GenericOp>(producer)) &&
      hasManyUsers(producer) && !hasFusionOpportunity(producer, ctx)) {
    return nullptr;
  }

  std::shared_ptr<FusedNode> bestFusedNode = nullptr;
  // Returns true when bestFusedNode is unset (first candidate) or `user` is in
  // a node appearing earlier in the block.
  // Relies on ops within a node being contiguous after consolidation, so any op
  // from bestFusedNode serves as a valid proxy for its block position.
  auto isBetter = [&bestFusedNode](auto *user) {
    if (!bestFusedNode)
      return true;
    return !bestFusedNode->contains(user) &&
           user->isBeforeInBlock(*bestFusedNode->ops().begin());
  };
  for (auto *user : producer->getUsers()) {
    if (!isOpInBlock(user, block) || !isFusableOp(user) || !isBetter(user))
      continue;
    if (auto node = ctx.getInfo(user).fusedNode;
        node->canAccept(producer, AcceptContext::Producer) &&
        node->canFuseProducer(producer))
      bestFusedNode = node;
  }
  if (!bestFusedNode)
    return nullptr;

  FusableOpInfo &producerInfo = ctx.getInfo(producer);
  int numUsersInBestFusedNode = 0;
  for (auto user : DenseSet<Operation *>(producer->getUsers().begin(),
                                         producer->getUsers().end())) {
    if (bestFusedNode->contains(user)) {
      numUsersInBestFusedNode++;
      if (isa<linalg::TransposeOp>(user) && !isVsstbPatternTransposeOp(user))
        return nullptr;
    }
  }

  if (numUsersInBestFusedNode > 1) {
    AffineMap map;
    for (OpOperand &use : producer->getUses()) {
      Operation *user = use.getOwner();
      if (bestFusedNode->contains(user)) {
        if (auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(user)) {
          if (!map) {
            map = consumerLinalgOp.getMatchingIndexingMap(&use);
          } else if (map != consumerLinalgOp.getMatchingIndexingMap(&use)) {
            return nullptr;
          }
        }
      }
    }
  }
  // If producer is reduction and user is broadcast, only fuse last-axis
  // reduction producer into last-axis broadcast op. Other fusion context will
  // generate memref.alloc inside VF and cannot be outlined, so give up fusing,
  // see issue:
  // https://codehub-y.huawei.com/CompilerKernel/BiShengCompiler/AscendNPU-IR/issues/638
  if (producerInfo.numReductionLoops > 0) {
    bool producerIsNonLastAxisReduce = false;
    auto producerLinalgOp = dyn_cast<linalg::LinalgOp>(producer);
    SmallVector<unsigned> reductionDims;
    producerLinalgOp.getReductionDims(reductionDims);
    if (reductionDims[0] != producerInfo.numLoops - 1)
      producerIsNonLastAxisReduce = true;

    for (OpOperand &use : producer->getUses()) {
      Operation *user = use.getOwner();
      if (bestFusedNode->contains(user)) {
        if (auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(user)) {
          AffineMap map = consumerLinalgOp.getMatchingIndexingMap(&use);
          if (map.getNumResults() < map.getNumDims()) {
            // last-axis broadcast, the affine map will be
            // `affine_map<(d0,d1)->(d0)>` non-last-axis broadcast, the affine
            // map will be `affine_map<(d0,d1)->(d1)>`
            for (auto indexAndResult : llvm::enumerate(map.getResults()))
              if (auto d = dyn_cast<AffineDimExpr>(indexAndResult.value()))
                if (d.getPosition() != indexAndResult.index())
                  return nullptr;
            if (producerIsNonLastAxisReduce)
              return nullptr;
          }
        }
      }
    }
  }

  return bestFusedNode;
}

static Value tileReductionOpTreeReduce(
    OpBuilder &builder, transform::SequenceOp seqOp, Operation *op,
    Value &linalgOpHandle, SmallVector<int64_t> tileSize, std::string label,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps) {
  assert(isa<linalg::LinalgOp>(op));
  auto reductionOp = cast<linalg::LinalgOp>(op);
  assert(reductionOp.getNumParallelLoops() > 0);
  assert(reductionOp.getNumReductionLoops() == 1);
  auto loc = seqOp->getLoc();
  // get parallel axis and tile.
  // get reduction axis and tile.
  SmallVector<unsigned> reductionDims;
  reductionOp.getReductionDims(reductionDims);
  auto reductionDim = reductionDims[0];
  auto reductionTileSize = tileSize[reductionDim];

  int64_t reductionSize = reductionOp.getStaticLoopRanges()[reductionDim];
  assert(reductionSize > 0 &&
         "tree reduction requires a positive static extent");

  // Creates new Vector with tile sizes
  // where "amount" 1s are added to idx position
  auto createExtendedTileSizes = [&](size_t idx, size_t amount) {
    SmallVector<int64_t> res = tileSize;
    res.insert(res.begin() + idx, amount, 1);

    return res;
  };

  auto getIterationTilingIndicator = [&](size_t iterCnt) {
    SmallVector<int64_t> tilingIndicator(reductionOp.getNumLoops() + iterCnt,
                                         0);
    tilingIndicator[reductionDim + iterCnt] = 1;

    return tilingIndicator;
  };

  // Processes (annotate + add to other vectorizable ops vector for futher
  // vectorization) on of auxilary ops (fill or splited reduce) created in tree
  // reduce algo
  auto processAuxOpTreeReduce = [&](Value opHandle, std::string opName,
                                    size_t opIdx) {
    auto vectorizeTileSize = createExtendedTileSizes(reductionDim, opIdx);
    std::string splitOpLabel =
        label + "_" + opName + "_" + std::to_string(opIdx);
    builder.create<transform::AnnotateOp>(loc, opHandle, splitOpLabel, nullptr);
    otherVectorizableOps.push_back(
        std::make_pair(splitOpLabel, vectorizeTileSize));
  };

  // Tile parallel axes and patch first reduce operation to be that reduce
  // which was generated by parallel tilining (its new operation, we can't use
  // old one)
  // NOTE: reductionTileSizePairs guaranteed not empty
  SmallVector<unsigned> parallelDims;
  reductionOp.getParallelDims(parallelDims);
  SmallVector<int64_t> parallelAxisTileSize(reductionOp.getNumLoops(), 0);
  for (auto i : parallelDims)
    parallelAxisTileSize[i] = tileSize[i];
  // Tile all parallel dims
  transform::TileUsingForOp parallelAxisTilingResult =
      builder.create<transform::TileUsingForOp>(loc, linalgOpHandle,
                                                parallelAxisTileSize);
  linalgOpHandle = parallelAxisTilingResult.getTiledLinalgOp();

  // Pad non-power-of-two reductions with the neutral element instead of
  // reducing a power-of-two prefix and merging an unbalanced tail.  Besides
  // keeping the reduction tree balanced, this also makes its reassociation
  // deterministic for numerically sensitive floating-point reductions.
  auto alignedRedSize = static_cast<int64_t>(
      llvm::PowerOf2Ceil(static_cast<uint64_t>(reductionSize)));
  if (reductionSize != alignedRedSize) {
    SmallVector<int64_t> reductionAxisTileSize(reductionOp.getNumLoops(), 0);
    reductionAxisTileSize[reductionDim] = alignedRedSize;
    auto tileReductionRes = builder.create<transform::TileReductionUsingForOp>(
        loc, linalgOpHandle, reductionAxisTileSize);

    linalgOpHandle = tileReductionRes.getCombiningLinalgOp();
    Value fillOpHandle = tileReductionRes.getFillOp().front();
    auto splitedReductionOp = tileReductionRes.getSplitLinalgOp();

    if (alignedRedSize > reductionTileSize) {
      auto tilingIndicator = getIterationTilingIndicator(0);
      fillOpHandle = builder
                         .create<transform::TileUsingForOp>(loc, fillOpHandle,
                                                            tilingIndicator)
                         .getTiledLinalgOp();
      splitedReductionOp = builder
                               .create<transform::TileUsingForOp>(
                                   loc, splitedReductionOp, tilingIndicator)
                               .getTiledLinalgOp();
    }
    processAuxOpTreeReduce(splitedReductionOp, "tile_reduce", 0);
    processAuxOpTreeReduce(fillOpHandle, "tile_fill", 0);
  }

  SmallVector<int64_t, 4> splitedRedDims;

  // For small reductions, preserve one global pairwise tree across the whole
  // reduction axis.  Splitting a 64-element reduction directly by 32 creates
  // the pairs (0, 32), (1, 33), ... that are expected from a balanced tree.
  // The first split materializes at most 32 partial values per output lane.
  // Keep the bounded radix-16 hierarchy for larger reductions to avoid an
  // unbounded temporary tensor.
  constexpr int64_t maxGlobalTreeReductionSize = 64;
  if (alignedRedSize <= maxGlobalTreeReductionSize) {
    splitedRedDims.push_back(alignedRedSize);
  } else {
    int64_t redRemain = alignedRedSize;
    for (; redRemain > 16; redRemain /= 16)
      splitedRedDims.push_back(16);
    splitedRedDims.push_back(redRemain);
  }

  auto iterAmount = splitedRedDims.size();
  for (auto [iterIdx, curSplitDim] :
       llvm::enumerate(llvm::reverse(splitedRedDims))) {
    auto curIterTilingIndicator = getIterationTilingIndicator(iterIdx);
    auto nextIterTilingIndicator = getIterationTilingIndicator(iterIdx + 1);

    Value nextLinalgOpHandle;
    if (iterIdx != iterAmount - 1) {
      auto splitReductionResult = builder.create<transform::SplitReductionOp>(
          loc, linalgOpHandle, curSplitDim, reductionDim + iterIdx);

      auto tiledSplitFillOp =
          builder
              .create<transform::TileUsingForOp>(
                  loc, splitReductionResult.getFillOp(), curIterTilingIndicator)
              .getTiledLinalgOp();
      processAuxOpTreeReduce(tiledSplitFillOp, "split_fill", iterIdx);

      nextLinalgOpHandle = builder
                               .create<transform::TileUsingForOp>(
                                   loc, splitReductionResult.getSplitLinalgOp(),
                                   curIterTilingIndicator)
                               .getTiledLinalgOp();

      linalgOpHandle = splitReductionResult.getCombiningLinalgOp();
    }

    for (; curSplitDim > 2; curSplitDim /= 2) {
      auto reductionTilingResult = builder.create<transform::SplitReductionOp>(
          loc, linalgOpHandle, curSplitDim / 2, reductionDim + iterIdx, true);

      auto tiledFill = builder
                           .create<transform::TileUsingForOp>(
                               loc, reductionTilingResult.getFillOp(),
                               curIterTilingIndicator)
                           .getTiledLinalgOp();
      processAuxOpTreeReduce(tiledFill, "block_fill", iterIdx);

      auto nextIterTilingResult = builder.create<transform::TileUsingForOp>(
          loc, reductionTilingResult.getSplitLinalgOp(),
          nextIterTilingIndicator);
      auto curIterTilingResult = builder.create<transform::TileUsingForOp>(
          loc, nextIterTilingResult.getTiledLinalgOp(), curIterTilingIndicator);
      processAuxOpTreeReduce(curIterTilingResult.getTiledLinalgOp(),
                             "block_split_reduce", iterIdx + 1);

      linalgOpHandle = reductionTilingResult.getCombiningLinalgOp();
    }

    auto curIterTilingResult = builder.create<transform::TileUsingForOp>(
        loc, linalgOpHandle, curIterTilingIndicator);
    processAuxOpTreeReduce(curIterTilingResult.getTiledLinalgOp(),
                           "block_combine_reduce", iterIdx);

    linalgOpHandle = nextLinalgOpHandle;
  }

  return parallelAxisTilingResult.getLoops().front();
}

/// Tile a canonical RA reduction normally, but mark its dedicated loop for
/// the post-vectorization direct register-tree rewrite.  Unlike
/// tileReductionOpTreeReduce this does not create tensor.empty/fill/split
/// intermediates for the pairwise levels.
static Value
tileReductionOpRegisterTree(OpBuilder &builder, transform::SequenceOp seqOp,
                            Value linalgOpHandle, ArrayRef<int64_t> tileSize,
                            ArrayRef<int64_t> tileInterchange = {}) {
  auto tilingResult = builder.create<transform::TileUsingForOp>(
      seqOp.getLoc(), linalgOpHandle, tileSize, tileInterchange);
  Value loop = tilingResult.getLoops().front();
  builder.create<transform::AnnotateOp>(
      seqOp.getLoc(), loop,
      builder.getStringAttr(hfusion::kRegisterTreeReductionLoopAttr), nullptr);
  return loop;
}

/// For reduction op, tile_reduction_using_for has better performance than
/// tile_using_for. Firstly we should tile parallel axis by tile_using_for,
/// then tile reduction axis by tile_reduction_using_for.
static Value
tileReductionOp(OpBuilder &builder, transform::SequenceOp seqOp, Operation *op,
                Value &linalgOpHandle, SmallVector<int64_t> tileSize,
                std::string label,
                SmallVector<std::pair<std::string, SmallVector<int64_t>>>
                    &otherVectorizableOps) {
  assert(isa<linalg::LinalgOp>(op));
  auto reductionOp = cast<linalg::LinalgOp>(op);
  assert(reductionOp.getNumParallelLoops() > 0);
  assert(reductionOp.getNumReductionLoops() == 1);
  auto loc = seqOp->getLoc();
  // get parallel axis and tile.
  SmallVector<unsigned> parallelDims;
  reductionOp.getParallelDims(parallelDims);
  SmallVector<int64_t> parallelAxisTileSize(reductionOp.getNumLoops(), 0);
  for (auto i : parallelDims)
    parallelAxisTileSize[i] = tileSize[i];
  transform::TileUsingForOp parallelAxisTilingResult =
      builder.create<transform::TileUsingForOp>(loc, linalgOpHandle,
                                                parallelAxisTileSize);
  // get reduction axis and tile.
  SmallVector<unsigned> reductionDims;
  reductionOp.getReductionDims(reductionDims);
  SmallVector<int64_t> reductionAxisTileSize(reductionOp.getNumLoops(), 0);
  for (auto i : reductionDims)
    reductionAxisTileSize[i] = tileSize[i];
  transform::TileReductionUsingForOp reductionAxisTilingResult =
      builder.create<transform::TileReductionUsingForOp>(
          loc, parallelAxisTilingResult.getTiledLinalgOp(),
          reductionAxisTileSize);
  builder.create<transform::AnnotateOp>(
      loc, reductionAxisTilingResult.getForOp(), "reductionLoop", nullptr);
  // fillOp and combiningLinalgOp should also be vectorized, here we add them
  // into otherVectorizableOps which will be vectorized after tiling and
  // fusing.
  for (auto it : llvm::enumerate(reductionAxisTilingResult.getFillOp())) {
    Value fillOp = it.value();
    std::string fillOpLabel = label + "_fill_" + std::to_string(it.index());
    builder.create<transform::AnnotateOp>(loc, fillOp, fillOpLabel, nullptr);
    otherVectorizableOps.push_back(std::make_pair(fillOpLabel, tileSize));
  }
  std::string splitLinalgOpLabel = label + "_split";
  builder.create<transform::AnnotateOp>(
      loc, reductionAxisTilingResult.getSplitLinalgOp(), splitLinalgOpLabel,
      nullptr);
  otherVectorizableOps.push_back(std::make_pair(splitLinalgOpLabel, tileSize));
  // If combiningLinalgOp(is a linalg.reduce op) has dyncamic shape, it
  // cannot be vectorized, so we convert it to a linalg.generic op.
  Value generalizedCombiningLinalgOp = builder.create<transform::GeneralizeOp>(
      loc, builder.getType<transform::AnyOpType>(),
      reductionAxisTilingResult.getCombiningLinalgOp());
  std::string combiningLinalgOpLabel = label + "_combining";
  builder.create<transform::AnnotateOp>(loc, generalizedCombiningLinalgOp,
                                        combiningLinalgOpLabel, nullptr);
  otherVectorizableOps.push_back(
      std::make_pair(combiningLinalgOpLabel, tileSize));
  return parallelAxisTilingResult.getLoops().front();
}

static void
collectFusableFuncInModule(ModuleOp moduleOp,
                           SmallVector<func::FuncOp> &fusableFuncList) {
  moduleOp->walk([&](func::FuncOp func) {
    auto fusionKind = mlir::hfusion::tryGetFusionKind(func);
    if (hacc::utils::isDevice(func)) {
      if (fusionKind.has_value() &&
          (fusionKind.value() == mlir::hfusion::FusionKind::ShallowCV ||
           fusionKind.value() == mlir::hfusion::FusionKind::SingleCube)) {
        // Skip this for now
      } else
        fusableFuncList.push_back(func);
    }
  });
}

class AutoVectorizeV2 : public impl::AutoVectorizeV2Base<AutoVectorizeV2> {
public:
  explicit AutoVectorizeV2(const AutoVectorizeV2Options &options)
      : AutoVectorizeV2Base(options) {}
  void runOnOperation() override;

private:
  void buildTileAndFuseTransformSequenceForBlock(
      OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps,
      PlanContext &result);
  void buildVectorizeTransformSequence(
      OpBuilder &builder, transform::SequenceOp seqOp,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps,
      PlanContext &result);
  Value getOpTransformHandle(std::string label, OpBuilder &builder,
                             transform::SequenceOp seqOp);
  void planFuseSiblingForLeafNodes(Block *block, PlanContext &result);
  void planFuseProducersIntoConsumers(Block *block, PlanContext &result);
  void planFuseProducerIntoFusedNode(Block *block, Operation *producer,
                                     PlanContext &result);
  void tileAndFuseSiblingForLeafNodes(
      OpBuilder &builder, transform::SequenceOp seqOp, PlanContext &result,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps);
  void fuseProducersIntoConsumers(
      OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
      PlanContext &result,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps);
  void applyCleanUp(OpBuilder &builder, transform::SequenceOp seqOp);
  void sortFunc(func::FuncOp func);
  transform::SequenceOp buildTransformSequence(func::FuncOp func,
                                               RetriedOptions &retryCtx,
                                               OpBuilder &builder);
  void emitTransformSequenceIR(func::FuncOp func, RetriedOptions &retryCtx,
                               OpBuilder &builder);
  LogicalResult runAttempt(func::FuncOp func, RetriedOptions &retryCtx,
                           OpBuilder &builder, IRRewriter &rewriter);
  LogicalResult vectorize(func::FuncOp func, RetriedOptions &retryCtx,
                          OpBuilder &builder);
};

void AutoVectorizeV2::planFuseSiblingForLeafNodes(Block *block,
                                                  PlanContext &ctx) {
  // Collect leafNodes in the block
  SmallVector<Operation *> leafNodes;
  block->walk([&](Operation *op) {
    if (isOpInBlock(op, block) && isFusableOutputNode(op, block)) {
      leafNodes.push_back(op);
    }
  });
  if (leafNodes.empty())
    return;
  // Group leafNodes, all leafNodes in the same group will be fused siblings
  for (auto leafNode : leafNodes) {
    if (ctx.size() == 0) {
      ctx.add(leafNode);
      continue;
    }
    if (isMemrefLinalgOp(leafNode)) {
      ctx.add(leafNode);
      continue;
    }

    bool isInserted = false;
    for (auto &node : ctx.nodes()) {
      if (!node->canAccept(leafNode))
        continue;
      if (!canShareTreeReductionSiblingLoop(leafNode, node, ctx, treeReduce))
        continue;
      node->addLeaf(leafNode);
      isInserted = true;
      break;
    }

    if (!isInserted) {
      ctx.add(leafNode);
    }
  }
}

// Determine how to fuse producers layer by layer.
void AutoVectorizeV2::planFuseProducersIntoConsumers(Block *block,
                                                     PlanContext &ctx) {
  std::queue<Operation *> queue;
  for (auto &node : ctx.nodes()) {
    for (Operation *leafNode : node->leafOps()) {
      queue.push(leafNode);
    }
  }
  while (!queue.empty()) {
    auto consumer = queue.front();
    queue.pop();
    for (Value operand : consumer->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (isFusableOp(producer)) {
        // It means that current producer has been handled if the value of
        // ctx.getInfo(producer).fusedNode is not nullptr.
        if (!isOpInBlock(producer, block) || ctx.getInfo(producer).fusedNode)
          continue;

        // If all users of the producer has been handled, we can fuse this
        // producer into proper fused node.
        if (llvm::all_of(producer->getUsers(), [&](Operation *user) {
              if (isFusableOp(user))
                return ctx.getInfo(user).fusedNode != nullptr;
              return true;
            })) {
          planFuseProducerIntoFusedNode(block, producer, ctx);
          queue.push(producer);
        }
      }
    }
  }
}

void AutoVectorizeV2::planFuseProducerIntoFusedNode(Block *block,
                                                    Operation *producer,
                                                    PlanContext &ctx) {
  FusableOpInfo &producerInfo = ctx.getInfo(producer);
  std::shared_ptr<FusedNode> bestFusedNode =
      findBestFusedNodeForProducer(block, producer, ctx, vectorLength);
  // Apply the same one-tree-per-node invariant when a reduction is accepted as
  // a producer rather than entering the fallback-leaf path below.
  if (bestFusedNode) {
    bool producerUsesRegisterTree =
        treeReduce && hfusion::shouldUseRegisterTreeReduction(producer);
    bool nodeUsesRegisterTree =
        treeReduce && llvm::any_of(bestFusedNode->ops(), [](Operation *op) {
          return hfusion::shouldUseRegisterTreeReduction(op);
        });
    if (producerUsesRegisterTree || nodeUsesRegisterTree ||
        (!getTreeReductionParallelDims(producer, treeReduce).empty() &&
         llvm::any_of(bestFusedNode->ops(), [&](Operation *fusedOp) {
           return !getTreeReductionParallelDims(fusedOp, treeReduce).empty();
         })))
      bestFusedNode.reset();
  }
  if (bestFusedNode) {
    bestFusedNode->addProducer(producer);

    // consumer leafNodes should interchange when tiling because of reduction
    // producer.
    if (producerInfo.numReductionLoops) {
      for (OpOperand &use : producer->getUses()) {
        Operation *user = use.getOwner();
        if (bestFusedNode->containsLeaf(user)) {
          // FIXME: here only find common axis for LinalgOp, also should find
          // for non LinalgOp(interleave, deinterleave...)
          if (auto consumer = dyn_cast<linalg::LinalgOp>(user)) {
            SmallVector<int64_t> commonAxis;
            AffineMap indexingMap = consumer.getMatchingIndexingMap(&use);
            for (AffineExpr e : indexingMap.getResults()) {
              if (auto d = dyn_cast<AffineDimExpr>(e)) {
                commonAxis.push_back(d.getPosition());
              }
            }
            for (Operation *leafNode : bestFusedNode->leafOps())
              interchangeForLeafNodes(commonAxis, ctx.getInfo(leafNode));
            break;
          }
        }
      }
    }
  } else {
    // expand_shape has no tile transform of its own — only vsstb-fused paths
    // can tile it.
    if (isExpandShapeOpCanFuseIntoVsstbPatternTranspose(producer))
      return;

    bool isInserted = false;
    for (auto &node : ctx.nodes()) {
      if (!node->canAccept(producer, AcceptContext::FallbackLeaf))
        continue;
      if (!canShareTreeReductionSiblingLoop(producer, node, ctx, treeReduce))
        continue;
      if (llvm::any_of(node->leafOps(), [&](Operation *otherLeafNode) {
            return isProducerConsumed(producer, otherLeafNode);
          }))
        continue;
      node->addLeaf(producer);
      isInserted = true;
      break;
    }
    if (!isInserted) {
      ctx.add(producer);
    }
  }
}

Value AutoVectorizeV2::getOpTransformHandle(std::string label,
                                            OpBuilder &builder,
                                            transform::SequenceOp seqOp) {
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

void AutoVectorizeV2::tileAndFuseSiblingForLeafNodes(
    OpBuilder &builder, transform::SequenceOp seqOp, PlanContext &ctx,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps) {
  auto loc = seqOp->getLoc();
  for (auto &node : ctx.nodes()) {
    alignTreeReductionSiblingLoops(node, ctx, treeReduce);
    SmallVector<Value> tiledLoopHandles;
    bool hasFillOp = false;
    for (Operation *leafNode : node->leafOps()) {
      if (mlir::hfusion::isFillOp(leafNode))
        hasFillOp = true;
      FusableOpInfo &leafNodeInfo = ctx.getInfo(leafNode);
      Value leafNodeHandle =
          getOpTransformHandle(leafNodeInfo.label, builder, seqOp);
      if (hfusion::shouldUseTileReductionUsingForV2(leafNode)) {
        tiledLoopHandles.push_back(tileReductionOp(
            builder, seqOp, leafNode, leafNodeHandle, leafNodeInfo.tileSize,
            leafNodeInfo.label, otherVectorizableOps));
      } else if (treeReduce &&
                 hfusion::shouldUseRegisterTreeReduction(leafNode)) {
        tiledLoopHandles.push_back(tileReductionOpRegisterTree(
            builder, seqOp, leafNodeHandle, leafNodeInfo.tileSize,
            leafNodeInfo.tileInterchange));
      } else if (treeReduce && hfusion::shouldUseTreeReduction(leafNode)) {
        tiledLoopHandles.push_back(tileReductionOpTreeReduce(
            builder, seqOp, leafNode, leafNodeHandle, leafNodeInfo.tileSize,
            leafNodeInfo.label, otherVectorizableOps));
      } else {
        transform::TileUsingForOp tilingResult =
            builder.create<transform::TileUsingForOp>(
                loc, leafNodeHandle, leafNodeInfo.tileSize,
                leafNodeInfo.tileInterchange);
        tiledLoopHandles.push_back(tilingResult.getLoops().front());
        if (isVsstbPatternTransposeOp(leafNode)) {
          assert(tilingResult.getLoops().size() >= 2);
          builder.create<transform::AnnotateOp>(
              loc, tilingResult.getLoops()[1],
              builder.getStringAttr("unroll_for_vsstb"), nullptr);
        }
      }
    }
    assert(!std::empty(tiledLoopHandles) && "Should fuse more than one loops");
    applyCleanUp(builder, seqOp);
    Value fusedLoopHandle = tiledLoopHandles.front();
    for (Value nextLoopHandle : llvm::drop_begin(tiledLoopHandles)) {
      fusedLoopHandle =
          builder
              .create<transform::LoopFuseSiblingOp>(
                  loc, builder.getType<transform::AnyOpType>(),
                  /*target=*/fusedLoopHandle, /*source=*/nextLoopHandle)
              .getFusedLoop();
      fusedLoopHandle = builder
                            .create<transform::LoopFuseNestedSiblingsOp>(
                                loc, builder.getType<transform::AnyOpType>(),
                                fusedLoopHandle, /*recursive=*/true)
                            .getTransformed();
    }
    if (hasFillOp)
      builder.create<transform::AnnotateOp>(loc, fusedLoopHandle,
                                            "outlinedLoopWithFill", nullptr);
    builder.create<transform::AnnotateOp>(loc, fusedLoopHandle,
                                          node->getLabel(), nullptr);
    if (tiledLoopHandles.size() > 1)
      applyCleanUp(builder, seqOp);
  }
}

void AutoVectorizeV2::fuseProducersIntoConsumers(
    OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
    PlanContext &ctx,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps) {
  auto loc = seqOp.getLoc();
  for (auto &node : ctx.nodes())
    for (Operation *producer : node->producerOps()) {
      FusableOpInfo &producerInfo = ctx.getInfo(producer);
      Value producerHandle =
          getOpTransformHandle(producerInfo.label, builder, seqOp);
      std::shared_ptr<FusedNode> fusedNode = producerInfo.fusedNode;
      Value containingLoopHandle =
          getOpTransformHandle(fusedNode->getLabel(), builder, seqOp);
      builder.create<transform::ApplyPatternsOp>(
          loc, containingLoopHandle, [](OpBuilder &innerBuilder, Location loc) {
            innerBuilder.create<transform::ApplyCanonicalizationPatternsOp>(
                loc);
          });
      builder.create<transform::MergeProducerExtractUsesOp>(
          loc, producerHandle, containingLoopHandle);
      transform::FuseIntoContainingOp fuseIntoOp =
          builder.create<transform::FuseIntoContainingOp>(
              loc, builder.getType<transform::AnyOpType>(),
              builder.getType<transform::AnyOpType>(), producerHandle,
              containingLoopHandle);
      Value fusedOp = fuseIntoOp.getFusedOp();
      Value newContainingLoopHandle = fuseIntoOp.getNewContainingOp();
      if (hfusion::shouldUseTileReductionUsingForV2(producer)) {
        tileReductionOp(builder, seqOp, producer, fusedOp,
                        producerInfo.tileSize, producerInfo.label,
                        otherVectorizableOps);
      } else if (treeReduce &&
                 hfusion::shouldUseRegisterTreeReduction(producer)) {
        tileReductionOpRegisterTree(builder, seqOp, fusedOp,
                                    producerInfo.tileSize);
      } else if (treeReduce && hfusion::shouldUseTreeReduction(producer)) {
        tileReductionOpTreeReduce(builder, seqOp, producer, fusedOp,
                                  producerInfo.tileSize, producerInfo.label,
                                  otherVectorizableOps);
      } else if (!isa<tensor::ExpandShapeOp>(producer)) {
        builder.create<transform::TileUsingForOp>(loc, fusedOp,
                                                  producerInfo.tileSize);
      }
      builder.create<transform::ApplyPatternsOp>(
          loc, newContainingLoopHandle,
          [](OpBuilder &innerBuilder, Location loc) {
            innerBuilder.create<transform::ApplyCanonicalizationPatternsOp>(
                loc);
          });
      Value funcHandle = builder.create<transform::MatchOp>(
          loc, seqOp.getBodyBlock()->getArguments().front(),
          ArrayRef<StringRef>({func::FuncOp::getOperationName()}));
      builder.create<transform::ApplyRegisteredPassOp>(
          loc, builder.getType<transform::AnyOpType>(), funcHandle,
          builder.getStringAttr("eliminate-single-iteration-scf-for"));
      applyCleanUp(builder, seqOp);
    }
}

void AutoVectorizeV2::buildTileAndFuseTransformSequenceForBlock(
    OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps,
    PlanContext &ctx) {
  ctx.resetForBlock();
  planFuseSiblingForLeafNodes(block, ctx);
  planFuseProducersIntoConsumers(block, ctx);
  ctx.computeTileSize();
#ifndef NDEBUG
  LLVM_DEBUG(llvm::dbgs() << "========Dumping LeafNodeGroups begin========\n");
  for (auto &node : ctx.nodes()) {
    LLVM_DEBUG(llvm::dbgs() << "========Dumping group========\n");
    for (auto op : node->leafOps()) {
      LLVM_DEBUG(llvm::dbgs() << *op);
      LLVM_DEBUG(llvm::dbgs() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "----shape:[");
      for (auto i : ctx.getInfo(op).shape)
        LLVM_DEBUG(llvm::dbgs() << i << ",");
      LLVM_DEBUG(llvm::dbgs() << "]----\n");
      LLVM_DEBUG(llvm::dbgs() << "----tilesize:[");
      for (auto i : ctx.getInfo(op).tileSize)
        LLVM_DEBUG(llvm::dbgs() << i << ",");
      LLVM_DEBUG(llvm::dbgs() << "]----\n");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "====Dumping ProducersToBeFusedInto begin====\n");
  for (auto &node : ctx.nodes()) {
    for (auto op : node->producerOps()) {
      LLVM_DEBUG(llvm::dbgs() << *op);
      LLVM_DEBUG(llvm::dbgs() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "----shape:[");
      for (auto i : ctx.getInfo(op).shape)
        LLVM_DEBUG(llvm::dbgs() << i << ",");
      LLVM_DEBUG(llvm::dbgs() << "]----\n");
      LLVM_DEBUG(llvm::dbgs() << "----tilesize:[");
      for (auto i : ctx.getInfo(op).tileSize)
        LLVM_DEBUG(llvm::dbgs() << i << ",");
      LLVM_DEBUG(llvm::dbgs() << "]----\n");
    }
  }
#endif
  tileAndFuseSiblingForLeafNodes(builder, seqOp, ctx, otherVectorizableOps);
  fuseProducersIntoConsumers(builder, seqOp, block, ctx, otherVectorizableOps);
}

void AutoVectorizeV2::buildVectorizeTransformSequence(
    OpBuilder &builder, transform::SequenceOp seqOp,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps,
    PlanContext &result) {
  auto loc = seqOp.getLoc();
  for (auto info : result.opInfos()) {
    // Here only vectorize LinalgOp, interleave/deinterleave will be vectorized
    // in convert-hfusion-to-hivmave pass.
    if (isa<linalg::LinalgOp>(info.first)) {
      FusableOpInfo &opInfo = info.second;
      builder.create<transform::VectorizeOp>(
          loc, getOpTransformHandle(opInfo.label, builder, seqOp),
          SmallVector<Value>(), opInfo.tileSize, nullptr,
          SmallVector<bool>(opInfo.tileSize.size(), false));
    }
  }
  for (auto vectorizableOp : otherVectorizableOps) {
    builder.create<transform::VectorizeOp>(
        loc, getOpTransformHandle(vectorizableOp.first, builder, seqOp),
        SmallVector<Value>(), vectorizableOp.second, nullptr,
        SmallVector<bool>(vectorizableOp.second.size(), false));
  }
}

void AutoVectorizeV2::applyCleanUp(OpBuilder &builder,
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

transform::SequenceOp AutoVectorizeV2::buildTransformSequence(
    func::FuncOp func, RetriedOptions &retryCtx, OpBuilder &builder) {
  analysis::VFStackInfoBuilder vfStack{retryCtx.enableVFStackLimit};
  PlanContext ctx(retryCtx, vfStack);
  ctx.initFusableOpInfoFrom(func);
  SmallVector<std::pair<std::string, SmallVector<int64_t>>>
      otherVectorizableOps;

  transform::SequenceOp seqOp = buildTransformSequenceOp(builder, func);
  func.walk([&](Block *block) {
    if (scope::utils::isInCubeScope(block->getParentOp()) ||
        isCubeScopeOp(block->getParentOp()))
      return;

    if (isa<func::FuncOp, scf::ForOp, scf::IfOp, scf::WhileOp, scope::ScopeOp>(
            block->getParentOp()))
      buildTileAndFuseTransformSequenceForBlock(builder, seqOp, block,
                                                otherVectorizableOps, ctx);
  });
  buildVectorizeTransformSequence(builder, seqOp, otherVectorizableOps, ctx);
  return seqOp;
}

void AutoVectorizeV2::emitTransformSequenceIR(func::FuncOp func,
                                              RetriedOptions &retryCtx,
                                              OpBuilder &builder) {
  // Emit mode materializes the payload tags and transform sequence for
  // debugging/replay only. It intentionally does not call the transform
  // interpreter, so the payload IR is left unvectorized.
  transform::SequenceOp seqOp = buildTransformSequence(func, retryCtx, builder);
  StringRef funcName = func.getSymName();
  func->setAttr(transform::TransformDialect::kTargetTagAttrName,
                builder.getStringAttr(
                    hfusion::auto_vectorize::getPayloadRootTag(funcName)));
  seqOp->setAttr(transform::TransformDialect::kTargetTagAttrName,
                 builder.getStringAttr(
                     hfusion::auto_vectorize::getTransformRootTag(funcName)));
}

LogicalResult AutoVectorizeV2::vectorize(func::FuncOp func,
                                         RetriedOptions &retryCtx,
                                         OpBuilder &builder) {
  transform::SequenceOp seqOp = buildTransformSequence(func, retryCtx, builder);

  transform::TransformOptions transformOptions;
  transformOptions.enableExpensiveChecks(false);
  LogicalResult result = transform::applyTransformNamedSequence(
      func, seqOp, func->getParentOfType<ModuleOp>(), transformOptions);
  seqOp->erase();

  if (succeeded(result))
    result = refineStaticLoopCarriedTensorTypes(func);

  hfusion::AutoVectorizeVerifier verifier;
  return failure(
      failed(result) ||
      failed(verifier.verifyFreeVectorRegion(true).emitDiagnostics(false).check(
          func)));
}

void AutoVectorizeV2::sortFunc(func::FuncOp func) {
  func.walk([&](Block *block) {
    if (isa<func::FuncOp, scf::ForOp, scf::WhileOp, scf::IfOp, scope::ScopeOp>(
            block->getParentOp()))
      sortTopologically(block);
  });
}

LogicalResult AutoVectorizeV2::runAttempt(func::FuncOp func,
                                          RetriedOptions &retryCtx,
                                          OpBuilder &builder,
                                          IRRewriter &rewriter) {
  std::string funcName = func.getSymName().str();

  builder.setInsertionPointAfter(func);
  func::FuncOp cloned = cast<func::FuncOp>(builder.clone(*func));
  SymbolTable::setSymbolName(cloned, "cloned_" + funcName);

  if (succeeded(vectorize(cloned, retryCtx, builder))) {
    rewriter.eraseOp(func);
    SymbolTable::setSymbolName(cloned, funcName);
    sortFunc(cloned);
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "========Failed========\n");
  rewriter.eraseOp(cloned);
  return failure();
}

static bool isSIMDFunc(func::FuncOp func) {
  if (auto pmAttr = func->getAttrOfType<StringAttr>("parallel_mode"))
    return pmAttr.getValue() == "simd";
  // Fallback: scan for SIMT ops via standard codebase APIs.
  bool hasSIMT = false;
  func.walk([&](Operation *inner) {
    if (hfusion::isSimtOps(inner) || hivm::util::isSIMTVF(inner)) {
      hasSIMT = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !hasSIMT;
}

// Minimum tensor-element count above which an intermediate result between
// VFs is considered large enough to risk UB overflow when multiplied by
// tt.num_stages.
static constexpr int64_t kLargeTensorMinElements = 4096;

// Inline private _fused_ calls whose return tensors are large enough to
// overflow UB if left as cross-VF intermediate buffers.
static DenseSet<func::FuncOp>
inlineFusedCalls(func::FuncOp func, ModuleOp moduleOp, IRRewriter &rewriter) {
  SmallVector<func::CallOp> callsToInline;
  func.walk([&](func::CallOp callOp) {
    auto callee = moduleOp.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee || !callee.isPrivate() ||
        !callee.getSymName().contains("_fused_")) {
      return;
    }
    bool hasLargeResult = false;
    for (Type t : callee.getResultTypes()) {
      if (auto tensorTy = dyn_cast<RankedTensorType>(t)) {
        if (tensorTy.hasStaticShape() &&
            ShapedType::getNumElements(tensorTy.getShape()) >
                kLargeTensorMinElements) {
          hasLargeResult = true;
          break;
        }
      }
    }
    if (hasLargeResult) {
      callsToInline.push_back(callOp);
    }
  });

  DenseSet<func::FuncOp> inlinedFuncs;
  for (auto callOp : callsToInline) {
    auto callee = moduleOp.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee)
      continue;

    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(callOp);
    IRMapping mapping;
    for (auto [arg, operand] :
         llvm::zip(callee.getArguments(), callOp.getOperands()))
      mapping.map(arg, operand);

    for (auto &op : callee.getBody().front()) {
      if (isa<func::ReturnOp>(op)) {
        for (auto [result, retVal] :
             llvm::zip(callOp->getResults(), op.getOperands()))
          result.replaceAllUsesWith(mapping.lookup(retVal));
      } else {
        rewriter.clone(op, mapping);
      }
    }
    rewriter.eraseOp(callOp);
    inlinedFuncs.insert(callee);
  }
  return inlinedFuncs;
}

void AutoVectorizeV2::runOnOperation() {
  ModuleOp op = getOperation();
  MLIRContext *context = op->getContext();
  IRRewriter rewriter(context);
  OpBuilder builder(context);

  SmallVector<func::FuncOp> fusableFuncList;
  collectFusableFuncInModule(op, fusableFuncList);

  // Inline VFFusionPass _fused_ sub-functions with large return tensors.
  // TODO: remove it after better solution.
  DenseSet<func::FuncOp> allInlined;
  for (func::FuncOp func : fusableFuncList) {
    if (hacc::utils::isDevice(func) && isSIMDFunc(func)) {
      auto inlined = inlineFusedCalls(func, op, rewriter);
      allInlined.insert(inlined.begin(), inlined.end());
    }
  }
  llvm::erase_if(fusableFuncList,
                 [&](func::FuncOp f) { return allInlined.contains(f); });
  for (auto callee : allInlined)
    rewriter.eraseOp(callee);

  for (func::FuncOp func : fusableFuncList) {
    RetriedOptions retryCtx{maxFusedOps, enableMultipleConsumerFusion,
                            enableCrossIfFusion, enableVFStackLimit,
                            vectorLength};
    if (emitTransformSequence) {
      // Emit payload IR with transform sequence.
      emitTransformSequenceIR(func, retryCtx, builder);
      // Return early to avoid applying the transform sequence.
      continue;
    }

    LogicalResult result = runAttempt(func, retryCtx, builder, rewriter);
    if (failed(result)) {
      func.emitWarning() << "AutoVectorizeV2 failed;";
      signalPassFailure();
    }
  }
  // Cost-selection attributes are internal communication between VFFusion
  // and AutoVectorizeV2.  Do not leak them into the lowered module.
  op->removeAttr(hfusion::kTreeReductionSelectionFrozenAttr);
  op->removeAttr(hfusion::kRegularTreeReductionScopeAttr);
  op.walk([](Operation *nestedOp) {
    nestedOp->removeAttr(hfusion::kRegisterTreeReductionSelectedAttr);
    nestedOp->removeAttr(hfusion::kRegularTreeReductionSelectedAttr);
  });
}

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createHFusionAutoVectorizeV2Pass(
    const AutoVectorizeV2Options &options) {
  return std::make_unique<AutoVectorizeV2>(options);
}
