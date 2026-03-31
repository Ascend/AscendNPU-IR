//===- Pattern.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/Pattern.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Transforms/Transforms.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <utility>

#define DEBUG_TYPE "common-pattern-bubble-up-extract-slice"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace mlir::hivm::detail {

static bool areOperandsUpperLevel(tensor::ExtractSliceOp sliceOp) {
  // can bubble up if all of the dependencies are on the equal or ancestor
  // of the source op
  auto *sliceParentRegion = sliceOp.getSource().getParentRegion();
  assert(sliceParentRegion && sliceParentRegion->getParentOp() &&
         "sliceOp should have a parent region");
  auto *op = sliceParentRegion->getParentOp();
  if (!op)
    return false;
  return llvm::all_of(sliceOp.getOperands(), [&](Value oprVal) {
    auto *targetPar = oprVal.getParentRegion()->getParentOp();
    if (!targetPar)
      return false;
    return targetPar->isAncestor(op);
  });
}

static bool isDynamicSlice(OffsetSizeAndStrideOpInterface op) {
  return ShapedType::isDynamicShape(op.getStaticSizes());
}

// This function create new parentOp after bubble up

// For example:
// %ParentOp = op %src
// %ChildOp = slice %ParentOp
// ->
// %ChildOp' = slice %src
// %ParentOp' = op %ChildOp'
// This function is creating %ChildOp'
template <typename OpTy, typename OpTy2>
static FailureOr<OpTy>
createNewParentOpAfterBubbledUp(RewriterBase &rewriter, size_t tilingDim,
                                OpTy childOp, OpTy2 parentOp) {
  if (!isa<OffsetSizeAndStrideOpInterface>(childOp.getOperation()) ||
      !isa<OffsetSizeAndStrideOpInterface>(parentOp.getOperation())) {
    return failure();
  }
  SmallVector<OpFoldResult, 4> newSrcStrides;
  SmallVector<OpFoldResult, 4> newSrcOffsets;
  SmallVector<OpFoldResult, 4> newSrcSizes;
  SmallVector<int64_t, 4> newSrcShape;
  rewriter.setInsertionPoint(childOp);
  auto maybeSubBlockLoop = findContainingSubblockLoop(childOp);
  if (failed(maybeSubBlockLoop))
    return failure();

  // We have an assumption here that HIVMBubbleUp is only serving
  // HIVMTileAndBindSubBlock 1:2. Since we only work on marked extractSlice,
  // it's safe for now.
  auto size =
      getSingleTileSize(rewriter, childOp->getLoc(), parentOp.getSource(),
                        tilingDim, maybeSubBlockLoop.value());
  if (failed(size))
    return failure();

  rewriter.setInsertionPointToStart(maybeSubBlockLoop.value().getBody());
  auto offsetAtTileDim = calculateOffsetAtTilingDim(
      rewriter, childOp->getLoc(), maybeSubBlockLoop.value(), size.value());

  auto rankType = cast<ShapedType>(childOp.getSourceType());
  if (failed(findCorrespondingSizesOffsetsStrides(
          rewriter, rankType, tilingDim, offsetAtTileDim, size.value(),
          newSrcStrides, newSrcOffsets, newSrcSizes, newSrcShape)))
    return failure();

  rewriter.setInsertionPoint(childOp);
  auto newSrc =
      rewriter.create<OpTy>(childOp->getLoc(), parentOp.getSource(),
                            newSrcOffsets, newSrcSizes, newSrcStrides);
  markCreatedExtractSliceOp(rewriter, newSrc);
  return newSrc;
}

// This function create new childOp after bubble up

// For example:
// %ParentOp = op %src
// %ChildOp = slice %ParentOp
// ->
// %ChildOp' = slice %src
// %ParentOp' = op %ChildOp'
// This function is creating %ParentOp'
template <typename OpTy, typename OpTy2, typename... Arg>
static FailureOr<OpTy>
createNewChildOpAfterBubbledUp(RewriterBase &rewriter, size_t tilingDim,
                               OpTy childOp, OpTy2 parentOp,
                               OpTy createdNewParent, Arg &&... args) {
  if (!isa<OffsetSizeAndStrideOpInterface>(childOp.getOperation()) ||
      !isa<OffsetSizeAndStrideOpInterface>(parentOp.getOperation())) {
    return failure();
  }
  SmallVector<OpFoldResult, 4> newViewStrides;
  SmallVector<OpFoldResult, 4> newViewOffsets;
  SmallVector<OpFoldResult, 4> newViewSizes;
  SmallVector<int64_t, 4> newViewShape;
  auto newSize = getSingleTileSize(
      rewriter, childOp->getLoc(), createdNewParent->getResult(0), tilingDim,
      childOp->template getParentOfType<scf::ForOp>());
  if (failed(newSize))
    return failure();

  rewriter.setInsertionPointToStart(
      childOp->template getParentOfType<scf::ForOp>().getBody());
  auto newOffsetAtTileDim = calculateOffsetAtTilingDim(
      rewriter, childOp->getLoc(),
      childOp->template getParentOfType<scf::ForOp>(), newSize.value());

  auto rankType = cast<ShapedType>(childOp.getSourceType());
  if (failed(findCorrespondingSizesOffsetsStrides(
          rewriter, rankType, tilingDim, newOffsetAtTileDim, newSize.value(),
          newViewStrides, newViewOffsets, newViewSizes, newViewShape)))
    return failure();

  rewriter.setInsertionPoint(childOp);

  return rewriter.create<OpTy2>(childOp->getLoc(), createdNewParent,
                                std::forward(args)..., newViewOffsets,
                                newViewSizes, parentOp.getMixedStrides());
}

LogicalResult
BubbleUpPattern::matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                 PatternRewriter &rewriter) const {
  Value source = sliceOp.getSource();

  if (!sliceOp.hasUnitStride())
    return rewriter.notifyMatchFailure(sliceOp, "expected unit stride");

  int extractSliceCount = 0;
  bool allAllowedOperationUsage =
      llvm::all_of(source.getUsers(), [&extractSliceCount](Operation *user) {
        if (isa<tensor::ExtractSliceOp>(user)) {
          extractSliceCount++;
        }
        return isa<tensor::ExtractSliceOp>(user) ||
               isa<annotation::MarkOp>(user);
      });
  if (!allAllowedOperationUsage)
    return rewriter.notifyMatchFailure(sliceOp,
                                       "not all usages are extract slice");

  // TODO: if it's not one use, operation cloning need to be done
  if (extractSliceCount != 1)
    return rewriter.notifyMatchFailure(
        sliceOp, "source has more than one usage beside extract slice.");
  auto *sourceDefiningOp = source.getDefiningOp();
  if (sourceDefiningOp && !areOperandsUpperLevel(sliceOp))
    return failure();

  // Try each strategy
  for (const auto &strategy : bubbleUpStrategies) {
    if (isMarkedExtractSliceOp(sliceOp) &&
        strategy->isSupportedOperation(sliceOp)) {
      LDBG("Picked strategy for sliceOp " << source);
      return strategy->execute(sliceOp, rewriter);
    }
  }

  return failure();
}

LogicalResult
BubbleUpSubviewFromTiling::matchAndRewrite(memref::SubViewOp subviewOp,
                                           PatternRewriter &rewriter) const {
  if (!subviewOp->hasAttrOfType<UnitAttr>(toBeBubbleUpSlice))
    return failure();

  if (isDynamicSlice(subviewOp))
    return failure();

  auto parentViewOp = subviewOp.getSource().getDefiningOp<memref::SubViewOp>();
  if (!parentViewOp || !createdByTiling(parentViewOp))
    return failure();

  auto extractDims = getExtractOrInsertDim(subviewOp);
  if (extractDims.size() != 1)
    return failure();
  auto tilingDim = *extractDims.begin();

  auto maybeNewSrc = createNewParentOpAfterBubbledUp(rewriter, tilingDim,
                                                     subviewOp, parentViewOp);
  if (failed(maybeNewSrc))
    return failure();
  auto newSrc = maybeNewSrc.value();

  auto maybeNewSubviewOp = createNewChildOpAfterBubbledUp(
      rewriter, tilingDim, subviewOp, parentViewOp, newSrc);
  if (failed(maybeNewSubviewOp))
    return failure();

  rewriter.replaceOp(subviewOp, maybeNewSubviewOp.value());
  return success();
}

Operation *BubbleUpPattern::getDefOpForInsertionPoint(OpOperand &opr) const {
  if (auto blockArg = dyn_cast<BlockArgument>(opr.get()))
    return &blockArg.getOwner()->front();
  return opr.get().getDefiningOp();
}

bool BroadcastBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<hivm::VBrcOp>(sourceOp) && !isDynamicSlice(sliceOp);
}

LogicalResult
BroadcastBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                   PatternRewriter &rewriter) const {
  auto broadcastOp =
      dyn_cast<hivm::VBrcOp>(sliceOp.getSource().getDefiningOp());
  if (!broadcastOp)
    return failure();

  auto outputType =
      dyn_cast<RankedTensorType>(broadcastOp.getResult().front().getType());

  // Get the positions of the input dimensions in the output
  auto broadcastDimMask =
      utils::arrayToMask(broadcastOp.getBroadcastDims(), outputType.getRank());

  // Get the offsets and sizes from the slice operation
  auto outputOffsets = sliceOp.getMixedOffsets();
  auto outputSizes = sliceOp.getMixedSizes();

  // Compute the input offsets and sizes
  SmallVector<OpFoldResult> inputOffsets, inputSizes;

  // Construct the new input offset, size and stride tuple
  for (int position = 0; position < outputType.getRank(); position++) {
    if (!broadcastDimMask[position]) {
      inputOffsets.push_back(outputOffsets[position]);
      inputSizes.push_back(outputSizes[position]);
    } else {
      inputOffsets.push_back(rewriter.getIndexAttr(0));
      inputSizes.push_back(rewriter.getIndexAttr(1));
    }
  }

  SmallVector<OpFoldResult> inputStrides(broadcastDimMask.size(),
                                         rewriter.getIndexAttr(1));
  Location loc = broadcastOp.getLoc();
  rewriter.setInsertionPoint(broadcastOp);
  if (broadcastOp.getNumDpsInits() != 1)
    return rewriter.notifyMatchFailure(broadcastOp,
                                       "dps init is more than one.");

  SmallVector<Value> newOperands;
  if (isa<RankedTensorType>(broadcastOp.getSrc().getType())) {
    rewriter.setInsertionPoint(broadcastOp);
    auto newSlicedInput = rewriter.create<tensor::ExtractSliceOp>(
        loc, broadcastOp.getSrc(), inputOffsets, inputSizes, inputStrides);
    markCreatedExtractSliceOp(rewriter, newSlicedInput);
    newOperands.push_back(newSlicedInput.getResult());
  } else {
    newOperands.push_back(broadcastOp.getSrc());
  }
  auto newSlicedInit = rewriter.create<tensor::ExtractSliceOp>(
      loc, broadcastOp.getDpsInits().front(), sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, newSlicedInit);

  newOperands.push_back(newSlicedInit);

  // Create the new BroadcastOp with the tiled input
  rewriter.setInsertionPointAfter(broadcastOp);
  Operation *newOp =
      clone(rewriter, broadcastOp, {sliceOp.getType()}, newOperands);
  rewriter.replaceAllUsesWith(sliceOp, newOp->getResult(0));

  return success();
}

bool ReduceBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<hivm::VReduceOp>(sourceOp) && !isDynamicSlice(sliceOp);
}

LogicalResult ReduceBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                              PatternRewriter &rewriter) const {
  auto reduceOp = cast<hivm::VReduceOp>(sliceOp.getSource().getDefiningOp());
  if (!reduceOp)
    return failure();

  // Build a map of reduction dimensions
  auto inputType = cast<RankedTensorType>(reduceOp.getSrc().getType());
  auto rank = inputType.getRank();

  BitVector isReductionDim =
      utils::arrayToMask(reduceOp.getReduceDims(), inputType.getRank());

  // Get the offsets and sizes from the slice operation
  auto sliceOffsets = sliceOp.getMixedOffsets();
  auto sliceSizes = sliceOp.getMixedSizes();

  if (reduceOp.getNumDpsInits() != 1)
    return rewriter.notifyMatchFailure(
        reduceOp, "doesn't support bubble up on multiple inits of vreduce");
  // Compute the input offsets and sizes

  auto inputShape = inputType.getShape();
  if (ShapedType::isDynamicShape(inputShape))
    return rewriter.notifyMatchFailure(reduceOp,
                                       "better dynamic analysis is needed");

  auto inputSizes = sliceSizes;
  for (unsigned i = 0; i < rank; ++i) {
    if (isReductionDim[i]) {
      inputSizes[i] = rewriter.getIndexAttr(inputShape[i]);
    }
  }

  rewriter.setInsertionPoint(reduceOp);
  SmallVector<OpFoldResult> inputStrides(rank, rewriter.getIndexAttr(1));
  auto newSlicedInput = rewriter.create<tensor::ExtractSliceOp>(
      reduceOp.getLoc(), reduceOp.getSrc(), sliceOffsets, inputSizes,
      inputStrides);
  markCreatedExtractSliceOp(rewriter, newSlicedInput);

  auto initReduce = reduceOp.getDpsInitOperand(0)->get();
  rewriter.setInsertionPoint(reduceOp);
  auto newSlicedInit = rewriter.create<tensor::ExtractSliceOp>(
      initReduce.getLoc(), initReduce, sliceOffsets, sliceSizes,
      sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, newSlicedInit);

  // Create the new ReduceOp with tiled operands
  SmallVector<Value> newOperands = {newSlicedInput.getResult(),
                                    newSlicedInit.getResult()};

  Operation *newOp =
      clone(rewriter, reduceOp, newSlicedInit.getType(), newOperands);
  rewriter.replaceOp(sliceOp, newOp->getResults());

  return success();
}

/// returns the index of the shape which has the non unit, returns -1 if all of
/// them is 1
static std::optional<int64_t> findOnlyNonUnit(ArrayRef<int64_t> shape) {
  int64_t rank = static_cast<int64_t>(shape.size());
  /// -1 means index not found
  int64_t ret = -1;
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] != 1) {
      if (ret != -1)
        return std::nullopt;
      ret = i;
    }
  }
  return ret;
}

bool ExpandBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<tensor::ExpandShapeOp>(sourceOp) &&
         !isDynamicSlice(sliceOp);
}

LogicalResult ExpandBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                              PatternRewriter &rewriter) const {
  auto expandOp =
      dyn_cast<tensor::ExpandShapeOp>(sliceOp.getSource().getDefiningOp());
  if (!expandOp)
    return failure();

  auto outputType = expandOp.getResultType();
  auto outputShape = outputType.getShape();
  auto inputRankType = cast<RankedTensorType>(expandOp.getSrc().getType());

  // The function findOnlyNonUnit only supports the tiling dimension that is
  // non-unit
  auto nonUnitOutput = findOnlyNonUnit(outputShape);
  auto nonUnitInput = findOnlyNonUnit(expandOp.getSrcType().getShape());
  if (!nonUnitOutput.has_value() || !nonUnitInput.has_value()) {
    /// This part deals with non-unit tensor.expand_shape, e.g.,
    /// tensor<32x32> into tensor<2x16x2x16>

    // Get the offsets and sizes from slice operation. This is the target output
    // of newExpandOp
    auto outputSizes =
        sliceOp.getMixedSizes(); // SmallVector<OpFoldResult> Type
    auto outputOffsets = sliceOp.getMixedOffsets();

    // Convert SmallVector<OpFoldResult> into SmallVector<int64_t>
    SmallVector<int64_t> outputSizesInt64, outputOffsetsInt64;
    for (const auto outputSize : outputSizes) {
      if (auto attr = dyn_cast<Attribute>(outputSize)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          outputSizesInt64.push_back(intAttr.getInt());
        }
      }
    }

    auto reassociation = expandOp.getReassociationIndices();

    // The shape, offsets, strides for the newSliceOp that will bubble up before
    // the expandOp
    SmallVector<OpFoldResult> inputShape, inputOffsets, inputStrides;

    /// find the inputShape, and the tilingDim for the newSliceOp
    /// We infer the input shape from reassociation [[0, 1], [2, 3]] and
    /// outputshape [4, 16, 4, 16], to get the inputShape [64, 64]. Here
    /// subGroup would be [0, 1] and [2, 3], groupProduct would be 64,
    int64_t tilingDim = -1;
    for (int64_t groupIdx = 0;
         groupIdx < static_cast<int64_t>(reassociation.size()); ++groupIdx) {
      auto subGroup = reassociation[groupIdx];
      int64_t groupProduct = 1; // groupProduct is the size before expandOp
      for (int64_t i = 0; i < static_cast<int64_t>(subGroup.size()); ++i) {
        groupProduct *= outputSizesInt64[subGroup[i]];
        if (outputOffsets[subGroup[i]]
                .is<Value>()) { // To find out the dynamic shape dimension, it
                                // is the tilingDim
          tilingDim = groupIdx;
        }
      }
      inputShape.push_back(
          getAsIndexOpFoldResult(rewriter.getContext(), groupProduct));
      inputStrides.push_back(rewriter.getIndexAttr(1));
    }

    // Calculate the offset at tilingDim
    auto maybeContainingLoop = findContainingSubblockLoop(expandOp);
    if (tilingDim == -1 || failed(maybeContainingLoop)) {
      return failure();
    }

    auto containingLoop = maybeContainingLoop.value();
    auto maybeSingleTileSize =
        getSingleTileSize(rewriter, expandOp.getLoc(), expandOp.getSrc(),
                          tilingDim, containingLoop);
    if (failed(maybeSingleTileSize)) {
      return failure();
    }

    auto offsetAtTileDim =
        calculateOffsetAtTilingDim(rewriter, expandOp.getLoc(), containingLoop,
                                   maybeSingleTileSize.value());

    // Calculate inputOffsets
    for (int64_t i = 0; i < inputRankType.getRank(); i++) {
      if (i != tilingDim) {
        inputOffsets.push_back(rewriter.getIndexAttr(0));
      } else {
        inputOffsets.push_back(offsetAtTileDim);
      }
    }

    rewriter.setInsertionPoint(sliceOp);
    Location loc = expandOp.getLoc();
    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, expandOp.getSrc(), inputOffsets, inputShape, inputStrides);

    markCreatedExtractSliceOp(rewriter, newSliceOp);

    auto newExpandOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, sliceOp.getResultType(), newSliceOp,
        expandOp.getReassociationIndices());

    rewriter.replaceOp(sliceOp, newExpandOp);
    if (expandOp->use_empty())
      rewriter.eraseOp(expandOp);
    return success();
  } else {
    /// This part deals with the unit tensor.expand_shape, e.g.,
    /// tensor<32> into tensor<32x1>

    // Get the offsets and sizes from the slice operation
    auto outputOffsets = sliceOp.getMixedOffsets();
    auto outputSizes = sliceOp.getMixedSizes();

    auto inputRank = expandOp.getSrcType().getRank();
    // Compute the input offsets and sizes
    SmallVector<OpFoldResult> inputOffsets(inputRank),
        inputSizes(inputRank, rewriter.getIndexAttr(1)),
        inputStrides(inputRank, rewriter.getIndexAttr(1));

    inputOffsets[nonUnitInput.value()] = outputOffsets[nonUnitOutput.value()];
    inputSizes[nonUnitInput.value()] = outputSizes[nonUnitOutput.value()];

    // Create the extract_slice of the input
    rewriter.setInsertionPoint(sliceOp);
    Location loc = expandOp.getLoc();
    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, expandOp.getSrc(), inputOffsets, inputSizes, inputStrides);
    markCreatedExtractSliceOp(rewriter, newSliceOp);

    auto newExpandOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, sliceOp.getResultType(), newSliceOp,
        expandOp.getReassociationIndices());
    rewriter.replaceOp(sliceOp, newExpandOp);
    if (expandOp->use_empty())
      rewriter.eraseOp(expandOp);
    return success();
  }
}

bool ExtractSliceBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  if (!sourceOp) {
    return false;
  }
  auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(sourceOp);
  if (!extractSliceOp)
    return false;
  if (!extractSliceOp.hasUnitStride())
    return false;
  return !isDynamicSlice(extractSliceOp) && !isDynamicSlice(sliceOp);
}

static LogicalResult
handleExtractRankReducedCase(tensor::ExtractSliceOp sliceOp,
                             PatternRewriter &rewriter) {
  auto parentSliceOp =
      cast<tensor::ExtractSliceOp>(sliceOp.getSource().getDefiningOp());
  auto parentSizes = parentSliceOp.getStaticSizes();
  // Currently we only try to handle the following ranked-reduced case,
  // which is safe to bubble up. other scenarios might not be safe to bubble up.
  // or it can be handled by mergeConsecutiveInsertExtractSlice Pattern.
  //
  // extract A x B x C -> B x C
  // extract B x C -> B' x C'
  // ->
  // extract A x B x C -> A x B' x C'
  // extract  A x B' x C' ->  B' x C'
  //
  // Parent is a ranked-reduce extract on first dimension.
  if ((parentSliceOp.getSource().getType().getRank() -
           parentSliceOp.getResultType().getRank() !=
       1) ||
      parentSizes[0] != 1) {
    return failure();
  }

  // and parent does not extract on any other dimension
  for (size_t i = 1; i < parentSizes.size(); i++) {
    if (parentSizes[i] != parentSliceOp.getSource().getType().getDimSize(i))
      return failure();
  }
  // TODO:: This can be enhance to support more rank-reduced scenario.

  // Safe to bubble up.
  auto parentMixedOffset = parentSliceOp.getMixedOffsets();
  auto childSizes = sliceOp.getMixedSizes();
  SmallVector<OpFoldResult> newStrides = parentSliceOp.getMixedStrides();
  SmallVector<OpFoldResult> newParentSizes;
  SmallVector<OpFoldResult> newSizes;

  newSizes.push_back(
      rewriter.getIndexAttr(parentSliceOp.getSourceType().getDimSize(0)));
  newParentSizes.push_back(rewriter.getIndexAttr(1));
  for (auto size : childSizes) {
    newSizes.push_back(size);
    newParentSizes.push_back(size);
  }

  auto childOffsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> newOffsets;
  newOffsets.push_back(rewriter.getIndexAttr(0));
  for (auto offset : childOffsets) {
    newOffsets.push_back(offset);
  }

  auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), parentSliceOp.getSource(), newOffsets, newSizes,
      newStrides);
  markCreatedExtractSliceOp(rewriter, newSliceOp);

  auto newParentSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), newSliceOp, parentSliceOp.getMixedOffsets(),
      newParentSizes, parentSliceOp.getMixedStrides());
  rewriter.replaceOp(parentSliceOp, newSliceOp);

  rewriter.modifyOpInPlace(newParentSliceOp, [&]() {
    newParentSliceOp->getResult(0).setType(sliceOp->getResult(0).getType());
  });

  rewriter.replaceOp(sliceOp, newParentSliceOp);
  return success();
}

static LogicalResult
handleExtractOfExtractSameDimCase(tensor::ExtractSliceOp sliceOp,
                                  PatternRewriter &rewriter) {
  // This function is handling such cases
  // extract A x B -> A/N x B
  // extract A/N x B -> A/2N x B
  // ->
  // extract A x B -> A/2 x B
  // extract  A/2 x B ->  A/2N x B

  auto parentSliceOp =
      sliceOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  auto extractDims = getExtractOrInsertDim(sliceOp);
  // Note: be extremely careful when handling such case, and not all cases
  // can be bubbled up.
  if (getExtractOrInsertDim(parentSliceOp).size() != 1 ||
      extractDims.size() != 1)
    // We are being very conservative that, only handling the case when
    // parentExtract is extracting single dim, and it overlaps with child
    // extract dim. It probably can be enhanced, but need to be very careful.
    return failure();
  auto tilingDim = *extractDims.begin();

  // If this insertSlice is not created by Tiling, it's very dangerous for us
  // to bubbled up, because the semantic may not be guaranteed to be the same.
  if (!createdByTiling(parentSliceOp))
    return failure();

  // We have an assumption here that HIVMBubbleUp is only serving
  // HIVMTileAndBindSubBlock 1:2. Since we only work on marked extractSlice,
  // it's safe for now.

  auto maybeNewSrc = createNewParentOpAfterBubbledUp(rewriter, tilingDim,
                                                     sliceOp, parentSliceOp);
  if (failed(maybeNewSrc))
    return failure();
  auto newSrc = maybeNewSrc.value();

  auto maybeNewSliceOp = createNewChildOpAfterBubbledUp(
      rewriter, tilingDim, sliceOp, parentSliceOp, newSrc);
  rewriter.replaceOp(sliceOp, maybeNewSliceOp.value());

  return success();
}

LogicalResult
ExtractSliceBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                      PatternRewriter &rewriter) const {
  auto parentSliceOp =
      cast<tensor::ExtractSliceOp>(sliceOp.getSource().getDefiningOp());
  // Handle Rank-reduced extract slice scenario.
  if (sliceOp.getDroppedDims().any() || parentSliceOp.getDroppedDims().any()) {
    return handleExtractRankReducedCase(sliceOp, rewriter);
  }

  // Handle the case when both extracts are extracting same dim.
  if (!llvm::set_intersection(getExtractOrInsertDim(sliceOp),
                              getExtractOrInsertDim(parentSliceOp))
           .empty()) {
    return handleExtractOfExtractSameDimCase(sliceOp, rewriter);
  }

  // TODO: Handle the case when both extracts are extracting different dims.

  return failure();
}

bool InsertSliceBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  if (!sourceOp)
    return false;
  auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(sourceOp);
  if (!insertSliceOp)
    return false;
  if (!insertSliceOp.hasUnitStride())
    return false;
  return !isDynamicSlice(insertSliceOp) && !isDynamicSlice(sliceOp);
}

static LogicalResult
handleInsertRankedReduceCase(tensor::ExtractSliceOp sliceOp,
                             PatternRewriter &rewriter) {
  auto parentInsertOp =
      cast<tensor::InsertSliceOp>(sliceOp.getSource().getDefiningOp());
  auto staticChildSize = sliceOp.getStaticSizes();
  // Currently we only try to handle the following ranked-reduced case,
  // which is safe to bubble up. other scenarios might not be safe to bubble
  // up. or it can be handled by mergeConsecutiveInsertExtractSlice Pattern.
  //
  // insert A x B -> C x A x B
  // extract C x A x B -> C x A' x B'
  // ->
  // extract A x B -> A' x B'
  // insert  A' x B' -> C x A' x B'
  //
  // If it's inserting not to first dimension and not extracting from the first
  // dimension
  if (staticChildSize[0] != sliceOp.getSource().getType().getDimSize(0) ||
      parentInsertOp.getStaticSizes()[0] != 1 ||
      parentInsertOp.getResultType().getRank() -
              parentInsertOp.getSource().getType().getRank() !=
          1) {
    // TODO:: this can be enhance to any dimension.
    return failure();
  }

  // Safe to bubble up.
  SmallVector<OpFoldResult> newStrides = sliceOp.getMixedStrides();
  newStrides.erase(newStrides.begin());
  SmallVector<OpFoldResult> newOffsets = sliceOp.getMixedOffsets();
  newOffsets.erase(newOffsets.begin());
  SmallVector<OpFoldResult> newSizes = sliceOp.getMixedSizes();
  newSizes.erase(newSizes.begin());
  auto newSrc = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), parentInsertOp.getSource(), newOffsets, newSizes,
      newStrides);
  markCreatedExtractSliceOp(rewriter, newSrc);
  auto newDst = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), parentInsertOp.getDest(), sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, newDst);

  newSizes.insert(newSizes.begin(), rewriter.getIndexAttr(1));
  auto newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
      sliceOp->getLoc(), newSrc, newDst, parentInsertOp.getMixedOffsets(),
      newSizes, parentInsertOp.getMixedStrides());
  rewriter.replaceOp(sliceOp, newInsertSliceOp);
  return success();
}

static FailureOr<tensor::InsertSliceOp>
createNewInsertForExtractOfInsertSameDim(RewriterBase &rewriter,
                                         size_t tilingDim,
                                         tensor::ExtractSliceOp sliceOp,
                                         tensor::InsertSliceOp parentInsertOp,
                                         tensor::ExtractSliceOp newSrc,
                                         tensor::ExtractSliceOp newDst) {
  SmallVector<OpFoldResult, 4> newInsertStrides;
  SmallVector<OpFoldResult, 4> newInsertOffsets;
  SmallVector<OpFoldResult, 4> newInsertSizes;
  SmallVector<int64_t, 4> newInsertShape;
  auto maybeSubBlockLoop = findContainingSubblockLoop(sliceOp);
  if (failed(maybeSubBlockLoop))
    return failure();
  auto size =
      getSingleTileSize(rewriter, sliceOp->getLoc(), parentInsertOp.getSource(),
                        tilingDim, maybeSubBlockLoop.value());
  if (failed(size))
    return failure();
  auto rankType = cast<ShapedType>(parentInsertOp.getSourceType());

  rewriter.setInsertionPointToStart(
      sliceOp->getParentOfType<scf::ForOp>().getBody());
  auto newOffsetAtTileDim = calculateOffsetAtTilingDim(
      rewriter, sliceOp->getLoc(), sliceOp->getParentOfType<scf::ForOp>(),
      size.value());
  if (failed(findCorrespondingSizesOffsetsStrides(
          rewriter, rankType, tilingDim, newOffsetAtTileDim, size.value(),
          newInsertStrides, newInsertOffsets, newInsertSizes, newInsertShape)))
    return failure();

  rewriter.setInsertionPoint(sliceOp);
  auto newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
      sliceOp->getLoc(), newSrc, newDst, newInsertOffsets, newInsertSizes,
      newInsertStrides);
  return newInsertSliceOp;
}

static LogicalResult
handleExtractOfInsertSameDimCase(tensor::ExtractSliceOp sliceOp,
                                 PatternRewriter &rewriter) {
  // This function is handling such cases
  // insert A x C into B x C
  // extract B x C -> B/2 x C
  // ->
  // extract A x C -> A/2 x C
  // extract B x C-> B/2 x C
  // insert A/2 x C into B/2 x C

  // We slice both src and dst becuase the aim of tile and bind sub block is
  // to split memory usage into multiple sub blocks.

  auto parentInsertOp =
      cast<tensor::InsertSliceOp>(sliceOp.getSource().getDefiningOp());
  // Note: be extremely careful when handling such case, and not all cases
  // can be bubbled up.
  if (parentInsertOp.getStaticSizes().size() != 1)
    // We are being very conservative that, only handling the case when
    // inserting to single dim, and it's overlaps with extract dim.
    // It probably can be enhanced, but need to be very careful.
    return failure();

  // If this insertSlice is not created by Tiling, it's very dangerous for us
  // to bubbled up, because the semantic may not be guaranteed to be the same.
  if (!createdByTiling(parentInsertOp)) {
    return failure();
  }

  auto extractDims = getExtractOrInsertDim(sliceOp);
  if (extractDims.size() != 1)
    return failure();
  auto tilingDim = *extractDims.begin();

  // We have an assumption here that HIVMBubbleUp is only serving
  // HIVMTileAndBindSubBlock 1:2. Since we only work on marked extractSlice,
  // it's safe for now.

  auto newDst = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), parentInsertOp.getDest(), sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, newDst);

  auto maybeNewSrc = createNewParentOpAfterBubbledUp(rewriter, tilingDim,
                                                     sliceOp, parentInsertOp);
  if (failed(maybeNewSrc))
    return failure();
  auto newSrc = maybeNewSrc.value();

  auto maybeNewInsertOp = createNewInsertForExtractOfInsertSameDim(
      rewriter, tilingDim, sliceOp, parentInsertOp, newSrc, newDst);
  if (failed(maybeNewInsertOp))
    return failure();

  rewriter.replaceOp(sliceOp, maybeNewInsertOp.value());
  return success();
}

LogicalResult
InsertSliceBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                     PatternRewriter &rewriter) const {
  auto parentInsertOp =
      cast<tensor::InsertSliceOp>(sliceOp.getSource().getDefiningOp());
  if (parentInsertOp->hasAttrOfType<UnitAttr>(toBeBubbleUpSlice)) {
    return failure();
  }

  // Handle ranked-reduce case.
  if ((parentInsertOp.getResultType().getRank() -
           parentInsertOp.getSource().getType().getRank() >
       0)) {
    return handleInsertRankedReduceCase(sliceOp, rewriter);
  }

  // Handle extract and insert on same dimension case.
  if (!llvm::set_intersection(getExtractOrInsertDim(parentInsertOp),
                              getExtractOrInsertDim(sliceOp))
           .empty()) {
    return handleExtractOfInsertSameDimCase(sliceOp, rewriter);
  }

  // TODO:: Handle extract and insert on different dimension case.

  return failure();
}

bool CollapseBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<tensor::CollapseShapeOp>(sourceOp) &&
         !isDynamicSlice(sliceOp);
}

LogicalResult
CollapseBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                  PatternRewriter &rewriter) const {
  auto collapseOp =
      dyn_cast<tensor::CollapseShapeOp>(sliceOp.getSource().getDefiningOp());
  if (!collapseOp)
    return failure();

  // Build a map of collapsed dimensions
  auto inputType = dyn_cast<RankedTensorType>(collapseOp.getSrc().getType());
  if (!inputType) {
    return failure();
  }
  auto outputType =
      dyn_cast<RankedTensorType>(collapseOp.getResult().getType());
  if (!outputType) {
    return failure();
  }

  auto reassociation = collapseOp.getReassociationIndices();

  auto inputRank = inputType.getRank();
  auto outputOffsets = sliceOp.getMixedOffsets();
  auto outputSizes = sliceOp.getMixedSizes();

  // Find the collapse dim
  BitVector isCollapseDim(inputRank, false);
  for (int64_t groupIdx = 0;
       groupIdx < static_cast<int64_t>(reassociation.size()); groupIdx++) {
    auto subGroup = reassociation[groupIdx];
    if (subGroup.size() != 1) { // the collapse dim must be in the group with size > 1
      int64_t outputSize = outputType.getDimSize(groupIdx);
      for (int64_t i = 0; i < static_cast<int64_t>(subGroup.size()); i++) {
        auto InputIndex = subGroup[i];
        int64_t inputSize = inputType.getDimSize(InputIndex);
        if (inputSize == 1 || inputSize != outputSize) {
          /// We count two cases as collapse dim:
          /// 1) input size = 1;
          /// 2) the inputsize != outputSize;
          isCollapseDim[InputIndex] = true;
        }
      }
    }
  }

  // Get the inputOffset and inputSize
  auto mixedSizeFinal = tensor::getMixedSizes(rewriter, collapseOp.getLoc(),
                                              collapseOp->getOperand(0));
  SmallVector<OpFoldResult> inputOffsets(inputRank);
  SmallVector<OpFoldResult> inputSizes(inputRank);

  unsigned outIdx = 0;
  for (int64_t i = 0; i < inputRank; i++) {
    if (isCollapseDim[i]) {
      inputOffsets[i] = rewriter.getIndexAttr(0);
      inputSizes[i] = (inputType.isDynamicDim(i))
                          ? mixedSizeFinal[i]
                          : rewriter.getIndexAttr(inputType.getDimSize(i));
    } else {
      inputOffsets[i] = outputOffsets[outIdx];
      inputSizes[i] = outputSizes[outIdx];
      ++outIdx;
    }
  }

  SmallVector<OpFoldResult> inputStrides(inputRank, rewriter.getIndexAttr(1));

  rewriter.setInsertionPoint(collapseOp);
  auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      collapseOp.getLoc(), collapseOp->getOperand(0), inputOffsets, inputSizes,
      inputStrides);

  markCreatedExtractSliceOp(rewriter, newSliceOp);

  auto staticOutputShape = decomposeMixedValues(outputSizes);
  auto newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
      collapseOp.getLoc(), newSliceOp, collapseOp.getReassociationIndices());

  rewriter.replaceOp(sliceOp, newCollapseOp);
  if (collapseOp->use_empty())
    rewriter.eraseOp(collapseOp);
  return success();
}

bool LoopBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<scf::ForOp>(sourceOp) && !isDynamicSlice(sliceOp);
}

LogicalResult LoopBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                            PatternRewriter &rewriter) const {
  auto forOp = dyn_cast<scf::ForOp>(sliceOp.getSource().getDefiningOp());
  if (!forOp)
    return rewriter.notifyMatchFailure(sliceOp, "source failed to bind");

  Value oldStep = forOp.getStep();
  auto oldStepAsIndexOp = oldStep.getDefiningOp<arith::ConstantIndexOp>();
  if (oldStepAsIndexOp && oldStepAsIndexOp.value() != 1) {
    bishengir::normalizeLoop(rewriter, forOp, oldStep);
    return success();
  }

  auto yieldIndex = cast<OpResult>(sliceOp.getSource()).getResultNumber();
  auto oldResultType = sliceOp.getSource().getType();
  LDBG("Processing result of " << yieldIndex << " from for op " << forOp);
  auto valueToSlice = forOp.getYieldedValues()[yieldIndex];
  Operation *yieldOp = forOp.getRegion().getBlocks().rbegin()->getTerminator();
  rewriter.setInsertionPoint(yieldOp);
  auto newMovedInSlice = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(),
      /* resultType */ cast<RankedTensorType>(sliceOp.getType()),
      /* src */ valueToSlice, sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, newMovedInSlice);

  LDBG(valueToSlice);
  rewriter.modifyOpInPlace(
      forOp, [&]() { forOp.getResult(yieldIndex).setType(sliceOp.getType()); });
  rewriter.replaceAllUsesWith(sliceOp, forOp->getResult(yieldIndex));
  rewriter.modifyOpInPlace(yieldOp, [&]() {
    auto &yieldValueOpr = yieldOp->getOpOperand(yieldIndex);
    yieldValueOpr.assign(newMovedInSlice.getResult());
  });

  BlockArgument regionIterArg = forOp.getRegionIterArg(yieldIndex);
  regionIterArg.setType(sliceOp.getType());
  rewriter.setInsertionPointAfterValue(regionIterArg);
  auto tmpEmpty = rewriter.create<tensor::EmptyOp>(forOp.getLoc(),
                                                   oldResultType, ValueRange{});
  auto argumentInsert = rewriter.create<tensor::InsertSliceOp>(
      forOp.getLoc(), regionIterArg, tmpEmpty, sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  rewriter.replaceAllUsesExcept(regionIterArg, argumentInsert.getResult(),
                                argumentInsert);

  OpOperand &forOpInit = forOp.getInitsMutable()[yieldIndex];
  rewriter.setInsertionPoint(forOp);
  auto slicedInit = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(),
      /* resultType */ cast<RankedTensorType>(sliceOp.getType()),
      /* src */ forOpInit.get(), sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, slicedInit);

  forOpInit.set(slicedInit.getResult());

  return success();
}

bool LoopArgsBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  return false;
}

LogicalResult
LoopArgsBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                  PatternRewriter &rewriter) const {
  llvm_unreachable("This should not happen anymore");
  auto forOp = sliceOp->getParentOfType<scf::ForOp>();
  if (!forOp) {
    return failure();
  }

  BlockArgument blockArg = dyn_cast<BlockArgument>(sliceOp.getSource());
  if (!blockArg)
    return failure();

  auto blockArgIdx = blockArg.getArgNumber() - 1;

  rewriter.setInsertionPoint(forOp);
  auto movedOutSlice = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), cast<RankedTensorType>(sliceOp.getType()),
      forOp.getInitArgsMutable()[blockArgIdx].get(), sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, movedOutSlice);

  blockArg.setType(sliceOp.getType());
  rewriter.replaceAllUsesWith(sliceOp, blockArg);
  forOp.getInitArgsMutable()[blockArgIdx].set(movedOutSlice);

  return success();
}

bool VTransposeBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  // Check if source is a block argument of a for loop
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  if (!sourceOp) {
    return false;
  }
  bool isVTranspose =
      isa_and_nonnull<hivm::VTransposeOp>(sourceOp) && !isDynamicSlice(sliceOp);
  return isVTranspose;
}

LogicalResult
VTransposeBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                    PatternRewriter &rewriter) const {
  auto VTransOp =
      dyn_cast<hivm::VTransposeOp>(sliceOp.getSource().getDefiningOp());
  if (!VTransOp)
    return failure();

  auto inputVector = VTransOp.getOperand(0);
  auto inputType = dyn_cast<RankedTensorType>(inputVector.getType());

  auto resultType = dyn_cast<RankedTensorType>(sliceOp.getResult().getType());
  auto resultRank = resultType.getRank();

  ArrayRef<int64_t> targetShape = resultType.getShape();
  ArrayRef<int64_t> perm = VTransOp.getPermutation();

  // Set the outputShape of newVTransOp
  SmallVector<int64_t> outputShape;
  for (int64_t i = 0; i < resultRank; ++i) {
    outputShape.push_back(targetShape[i]);
  }

  rewriter.setInsertionPoint(VTransOp);
  Location loc = VTransOp.getLoc();
  auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
      loc, outputShape, inputType.getElementType());

  SmallVector<OpFoldResult> outputOffsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> inputOffsets(resultRank);
  SmallVector<OpFoldResult> inputSizes = sliceOp.getMixedSizes();
  SmallVector<OpFoldResult> inputStrides = sliceOp.getMixedStrides();

  // Infer the sizes and offsets of newSliceOp by perm
  for (int64_t i = 0; i < resultRank; ++i) {
    inputSizes[perm[i]] = rewriter.getIndexAttr(targetShape[i]);
    inputOffsets[perm[i]] = outputOffsets[i];
  }

  rewriter.setInsertionPoint(VTransOp);
  auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      VTransOp.getLoc(), VTransOp.getSrc(), inputOffsets, inputSizes,
      inputStrides);

  markCreatedExtractSliceOp(rewriter, newSliceOp);

  auto newVTransOp = rewriter.create<hivm::VTransposeOp>(
      VTransOp.getLoc(), sliceOp.getResultType(), newSliceOp.getResult(),
      newEmptyOp.getResult(), rewriter.getDenseI64ArrayAttr(perm));

  rewriter.replaceOp(sliceOp, newVTransOp);
  if (VTransOp->use_empty())
    rewriter.eraseOp(VTransOp);

  return success();
}

bool VarangeBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  if (!sourceOp) {
    return false;
  }
  bool isVarangeOp = dyn_cast<hivm::VArangeOp>(sourceOp);
  return isVarangeOp;
}

LogicalResult
VarangeBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                 PatternRewriter &rewriter) const {
  auto varangeOp =
      dyn_cast<hivm::VArangeOp>(sliceOp.getSource().getDefiningOp());
  if (!varangeOp) {
    return failure();
  }

  auto loc = varangeOp.getLoc();

  // Extract slice parameters
  auto offsets = sliceOp.getMixedOffsets();
  auto sizes = sliceOp.getMixedSizes();
  auto strides = sliceOp.getMixedStrides();

  rewriter.setInsertionPoint(varangeOp);
  auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, varangeOp.getDst(), offsets, sizes, strides);

  markCreatedExtractSliceOp(rewriter, newSliceOp);

  rewriter.setInsertionPointAfter(varangeOp);
  auto newVarangeOp = rewriter.create<hivm::VArangeOp>(loc, sliceOp.getType(),
                                                       newSliceOp.getResult());

  rewriter.replaceOp(sliceOp, newVarangeOp.getResult());
  rewriter.eraseOp(varangeOp);

  return success();
}

bool VInterleaveBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  if (!sourceOp) {
    return false;
  }
  bool isVInterleaveOp = dyn_cast<hivm::VInterleaveOp>(sourceOp);
  return isVInterleaveOp;
}

LogicalResult
VInterleaveBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                     PatternRewriter &rewriter) const {
  auto vinterleaveOp =
      dyn_cast<hivm::VInterleaveOp>(sliceOp.getSource().getDefiningOp());
  if (!vinterleaveOp)
    return failure();

  SmallVector<OpFoldResult> sliceOffsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sliceSizes = sliceOp.getMixedSizes();
  SmallVector<OpFoldResult> sliceStrides = sliceOp.getMixedStrides();

  SmallVector<OpFoldResult> newSizes(sliceSizes.begin(), sliceSizes.end() - 1);
  SmallVector<OpFoldResult> newSizesEmpty(sliceSizes.begin(),
                                          sliceSizes.end() - 1);
  newSizes.push_back(rewriter.getIndexAttr(1));
  newSizesEmpty.push_back(rewriter.getIndexAttr(2));

  // Create the new sliceOp for every input of vinterleaveOp
  SmallVector<Value> slicedInputs;
  for (Value input : vinterleaveOp.getSrc()) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());

    rewriter.setInsertionPoint(vinterleaveOp);
    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(), input, sliceOffsets, newSizes, sliceStrides);

    markCreatedExtractSliceOp(rewriter, newSliceOp);
    slicedInputs.push_back(newSliceOp);
  }

  auto interleaveType =
      dyn_cast<RankedTensorType>(vinterleaveOp.getOperand(0).getType());

  rewriter.setInsertionPoint(vinterleaveOp);
  auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
      sliceOp.getLoc(), newSizesEmpty, interleaveType.getElementType());

  rewriter.setInsertionPoint(vinterleaveOp);
  auto newInterleaveOp = rewriter.create<hivm::VInterleaveOp>(
      vinterleaveOp.getLoc(), sliceOp.getResultType(), ValueRange(slicedInputs),
      newEmptyOp, hfusion::InterleaveOp::getInterLeaveChannelNums());

  rewriter.replaceOp(sliceOp, newInterleaveOp);

  if (vinterleaveOp->use_empty())
    rewriter.eraseOp(vinterleaveOp);

  return success();
}

bool BufferizationBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<bufferization::ToTensorOp>(sourceOp) &&
         !isDynamicSlice(sliceOp);
}

LogicalResult
BufferizationBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                       PatternRewriter &rewriter) const {
  auto ToTensorOp =
      dyn_cast<bufferization::ToTensorOp>(sliceOp.getSource().getDefiningOp());

  if (!ToTensorOp)
    return failure();

  // Utilize the offsets, sizes, and strides from ExtractSliceOp
  auto offsets = sliceOp.getMixedOffsets();
  auto sizes = sliceOp.getMixedSizes();
  auto strides = sliceOp.getMixedStrides();

  Value srcMemref = ToTensorOp.getMemref();

  // Pattern 1: This deals with the pattern: memref.alloc() ->
  // bufferization.to_tensor, with Load Op
  for (Operation *userOp :
       srcMemref.getUsers()) { // srcMemref is the source of ToTensorOp
    if (auto LoadOp = dyn_cast<hivm::LoadOp>(userOp)) {

      // For Operand(0): if the Operand(0) of LoadOP is memref.reinterpret_cast
      auto castOp = dyn_cast<memref::ReinterpretCastOp>(
          LoadOp.getOperand(0).getDefiningOp());
      if (castOp) {
        rewriter.setInsertionPoint(LoadOp);
        auto castSubviewOp = rewriter.create<memref::SubViewOp>(
            castOp.getLoc(), castOp.getResult(), offsets, sizes, strides);

        rewriter.modifyOpInPlace(
            LoadOp, [&]() { LoadOp.setOperand(0, castSubviewOp.getResult()); });
      }

      // For Operand(1): if the Operand(1) of LoadOP is memref.alloc()
      auto AllocOp =
          dyn_cast<memref::AllocOp>(LoadOp.getOperand(1).getDefiningOp());
      if (AllocOp) {

        rewriter.setInsertionPoint(AllocOp);
        Location loc = AllocOp.getLoc();

        auto resultType =
            dyn_cast<RankedTensorType>(sliceOp.getResult().getType());
        ArrayRef<int64_t> shape = resultType.getShape();
        SmallVector<int64_t> staticShape;

        for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
          int64_t dim = shape[i];
          staticShape.push_back(dim);
        }

        auto memrefType =
            MemRefType::get(staticShape, resultType.getElementType());
        // Create a newAllocOp
        auto newAllocOp = rewriter.create<memref::AllocOp>(loc, memrefType);

        rewriter.modifyOpInPlace(
            LoadOp, [&]() { LoadOp.setOperand(1, newAllocOp.getResult()); });

        rewriter.setInsertionPoint(sliceOp);
        // Create a newsubViewOp, (it is used to create newToTensorOp)
        auto subviewOp = rewriter.create<memref::SubViewOp>(
            sliceOp.getLoc(), newAllocOp.getResult(), offsets, sizes, strides);

        // Create a new ToTensorOp
        auto newToTensorOp = rewriter.create<bufferization::ToTensorOp>(
            sliceOp.getLoc(), subviewOp.getResult(), true, true);

        rewriter.modifyOpInPlace(newToTensorOp, [&]() {
          newToTensorOp->setOperand(0, newAllocOp.getResult());
        });

        rewriter.replaceOp(sliceOp, newToTensorOp);
        if (ToTensorOp->use_empty())
          rewriter.eraseOp(ToTensorOp);
        if (subviewOp->use_empty())
          rewriter.eraseOp(subviewOp);
      }
      return success();
    }
    // Pattern 2: This deals with the pattern: memref.alloc() ->
    // bufferization.to_tensor, with Subview Op + Load Op
    if (auto subViewOp = dyn_cast<memref::SubViewOp>(userOp)) {
      if (!subViewOp->hasOneUse())
        continue;
      auto loadOp = dyn_cast<hivm::LoadOp>(*subViewOp->user_begin());
      if (!loadOp)
        continue;
      auto subViewOpGM = loadOp.getSrc().getDefiningOp<memref::SubViewOp>();
      // TODO : support compile-triton-nsa-topk-bwd-dkdv.mlir
      if (!subViewOpGM || !subViewOpGM.hasZeroOffset() ||
          !subViewOpGM.hasUnitStride() || !subViewOpGM->hasOneUse())
        continue;
      auto castOp =
          subViewOpGM.getSource().getDefiningOp<memref::ReinterpretCastOp>();
      if (!castOp)
        continue;
      auto allocOp = subViewOp.getSource().getDefiningOp<memref::AllocOp>();
      if (!allocOp)
        continue;
      LDBG("Pattern 2:\n"
           << castOp << "\n"
           << subViewOpGM << "\n"
           << allocOp << "\n"
           << subViewOp);
      rewriter.setInsertionPoint(subViewOpGM);

      // Compute new size
      auto newSizes = subViewOpGM.getMixedSizes();
      auto extractDims = getExtractOrInsertDim(sliceOp);
      if (extractDims.size() != 1)
        continue;
      auto tilingDim = *extractDims.begin();
      auto offsetVal = getValueOrCreateConstantIndexOp(
          rewriter, sliceOp.getLoc(), offsets[tilingDim]);
      auto sizeVal = getValueOrCreateConstantIndexOp(rewriter, sliceOp.getLoc(),
                                                     newSizes[tilingDim]);
      rewriter.setInsertionPointAfterValue(sizeVal);
      auto tilingSize = getValueOrCreateConstantIndexOp(
          rewriter, sliceOp.getLoc(), sizes[tilingDim]);
      offsetVal = rewriter.create<arith::MinSIOp>(offsetVal.getLoc(), offsetVal,
                                                  sizeVal);
      sizeVal =
          rewriter.create<arith::SubIOp>(sizeVal.getLoc(), sizeVal, offsetVal);
      sizeVal = rewriter.create<arith::MinSIOp>(sizeVal.getLoc(), sizeVal,
                                                tilingSize);
      newSizes[tilingDim] = sizeVal;

      // Rewrite operations
      rewriter.setInsertionPoint(subViewOpGM);
      auto newCastValue = rewriter.create<memref::SubViewOp>(
          castOp.getLoc(), castOp, offsets, sizes, strides);
      auto newSubViewOpGM = rewriter.create<memref::SubViewOp>(
          subViewOpGM.getLoc(), newCastValue, subViewOpGM.getMixedOffsets(),
          newSizes, subViewOpGM.getMixedStrides());
      rewriter.setInsertionPoint(allocOp);
      auto resultType =
          dyn_cast<RankedTensorType>(sliceOp.getResult().getType());
      allocOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(
          allocOp,
          MemRefType::get(resultType.getShape(), resultType.getElementType()));
      rewriter.setInsertionPoint(subViewOp);
      auto newSubViewOp = rewriter.create<memref::SubViewOp>(
          subViewOp.getLoc(), allocOp, subViewOp.getMixedOffsets(), newSizes,
          subViewOp.getMixedStrides());

      rewriter.modifyOpInPlace(loadOp, [&]() {
        loadOp.getSrcMutable().set(newSubViewOpGM);
        loadOp.getDstMutable().set(newSubViewOp);
      });

      rewriter.setInsertionPoint(sliceOp);
      rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
          sliceOp, allocOp.getResult(), true, true);

      if (subViewOp->use_empty())
        rewriter.eraseOp(subViewOp);
      if (subViewOpGM->use_empty())
        rewriter.eraseOp(subViewOpGM);
      if (ToTensorOp->use_empty())
        rewriter.eraseOp(ToTensorOp);
      return success();
    }
  }

  // Pattern 3: This deals with the pattern: memref.alloc() ->
  // memref.memory_space_cast -> bufferization.to_tensor
  auto sourceOp =
      srcMemref.getDefiningOp(); // srcMemref is the source of ToTensorOp, it is
                                 // bufferization.to_tensor
  if (auto memorySpaceCastOp = dyn_cast<memref::MemorySpaceCastOp>(sourceOp)) {
    if (auto UbAllocOp = dyn_cast<memref::AllocOp>(
            memorySpaceCastOp.getSource().getDefiningOp())) {
      auto addrSpace = cast<hivm::AddressSpaceAttr>(
          UbAllocOp.getResult().getType().getMemorySpace());
      if (addrSpace.getAddressSpace() ==
          hivm::AddressSpace::UB) { // If alloc is UB, tile it
        // get the alloc result shape from sliceOp
        auto resultType =
            dyn_cast<RankedTensorType>(sliceOp.getResult().getType());
        ArrayRef<int64_t> shape = resultType.getShape();
        SmallVector<int64_t> staticShape;

        for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
          int64_t dim = shape[i];
          staticShape.push_back(dim);
        }

        auto originalType =
            dyn_cast<MemRefType>(UbAllocOp.getResult().getType());
        auto newType = MemRefType::get(
            staticShape, originalType.getElementType(),
            originalType.getLayout(), originalType.getMemorySpace());

        Location loc = UbAllocOp.getLoc();

        rewriter.setInsertionPoint(UbAllocOp);
        auto newUbAllocOp = rewriter.create<memref::AllocOp>(loc, newType);

        // deal with the annotation.mark Op
        for (Operation *userOp :
             llvm::make_early_inc_range(UbAllocOp.getResult().getUsers())) {
          if (auto mark = dyn_cast<annotation::MarkOp>(userOp)) {
            rewriter.modifyOpInPlace(
                mark, [&]() { mark->setOperand(0, newUbAllocOp.getResult()); });
          }
        }

        rewriter.setInsertionPoint(memorySpaceCastOp);

        Value sourceUbmemref = newUbAllocOp.getResult();
        auto ubType = mlir::cast<MemRefType>(sourceUbmemref.getType());
        auto defaulType = MemRefType::get(
            ubType.getShape(), ubType.getElementType(), ubType.getLayout());

        auto newMemorySpaceCastOp = rewriter.create<memref::MemorySpaceCastOp>(
            memorySpaceCastOp.getLoc(), defaulType, sourceUbmemref);

        rewriter.setInsertionPoint(ToTensorOp);
        auto newToTensorOp = rewriter.create<bufferization::ToTensorOp>(
            ToTensorOp.getLoc(), newMemorySpaceCastOp.getResult(), true, true);

        rewriter.replaceOp(sliceOp, newToTensorOp);
        if (ToTensorOp->use_empty())
          rewriter.eraseOp(ToTensorOp);
        if (memorySpaceCastOp->use_empty())
          rewriter.eraseOp(memorySpaceCastOp);
        if (UbAllocOp->use_empty())
          rewriter.eraseOp(UbAllocOp);

        return success();
      }

      if (addrSpace.getAddressSpace() ==
          hivm::AddressSpace::L1) { // If alloc is L1, don't tile it
        rewriter.setInsertionPoint(ToTensorOp);
        auto newsubViewOp = rewriter.create<memref::SubViewOp>(
            ToTensorOp.getLoc(), memorySpaceCastOp.getResult(), offsets, sizes,
            strides);

        auto newToTensorOp = rewriter.create<bufferization::ToTensorOp>(
            ToTensorOp.getLoc(), newsubViewOp.getResult(), true, true);

        rewriter.replaceOp(sliceOp, newToTensorOp);
        if (ToTensorOp->use_empty())
          rewriter.eraseOp(ToTensorOp);

        return success();
      }
    }
  }

  // Pattern 4: This deals with the pattern: memref.alloc() ->
  // memref.memory_space_cast -> memref.expand_shape -> bufferization.to_tensor
  if (auto memExpandOp = dyn_cast<memref::ExpandShapeOp>(sourceOp)) {
    if (auto memorySpaceCastOp = dyn_cast<memref::MemorySpaceCastOp>(
            memExpandOp.getSrc().getDefiningOp())) {
      if (auto UbAllocOp = dyn_cast<memref::AllocOp>(
              memorySpaceCastOp.getSource().getDefiningOp())) {

        // get the toTensorOp result shape from sliceOp
        auto resultType =
            dyn_cast<RankedTensorType>(sliceOp.getResult().getType());
        ArrayRef<int64_t> shape = resultType.getShape();
        SmallVector<int64_t> resultShape;

        for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
          int64_t dim = shape[i];
          resultShape.push_back(dim);
        }

        // Infer the memExpandOp input shape
        SmallVector<int64_t> ExpandInputShape;
        auto reassociation = memExpandOp.getReassociationIndices();
        for (int64_t groupIdx = 0;
             groupIdx < static_cast<int64_t>(reassociation.size());
             groupIdx++) {
          auto subGroup = reassociation[groupIdx];
          int64_t groupProduct =
              1; // groupProduct is the dim size before expandOp
          for (int64_t i = 0; i < static_cast<int64_t>(subGroup.size()); i++) {
            groupProduct *= resultShape[subGroup[i]];
          }
          ExpandInputShape.push_back(groupProduct);
        }

        auto originalType =
            dyn_cast<MemRefType>(UbAllocOp.getResult().getType());
        auto newType = MemRefType::get(
            ExpandInputShape, originalType.getElementType(),
            originalType.getLayout(), originalType.getMemorySpace());

        Location loc = UbAllocOp.getLoc();

        rewriter.setInsertionPoint(UbAllocOp);
        auto newUbAllocOp = rewriter.create<memref::AllocOp>(loc, newType);

        /// deal with the annotation.mark Op
        for (Operation *userOp :
             llvm::make_early_inc_range(UbAllocOp.getResult().getUsers())) {
          if (auto mark = dyn_cast<annotation::MarkOp>(userOp)) {
            rewriter.modifyOpInPlace(
                mark, [&]() { mark->setOperand(0, newUbAllocOp.getResult()); });
          }
        }

        /// create the new Ops with 1:2 shape
        rewriter.setInsertionPoint(memorySpaceCastOp);
        Value sourceUbmemref = newUbAllocOp.getResult();
        auto ubType = mlir::cast<MemRefType>(sourceUbmemref.getType());
        auto defaulType = MemRefType::get(
            ubType.getShape(), ubType.getElementType(), ubType.getLayout());

        auto newMemorySpaceCastOp = rewriter.create<memref::MemorySpaceCastOp>(
            memorySpaceCastOp.getLoc(), defaulType, sourceUbmemref);

        auto ExpandType = MemRefType::get(resultShape, ubType.getElementType());

        rewriter.setInsertionPoint(memExpandOp);
        auto newMemExpandOp = rewriter.create<memref::ExpandShapeOp>(
            memExpandOp.getLoc(), ExpandType, newMemorySpaceCastOp,
            reassociation);

        rewriter.setInsertionPoint(ToTensorOp);
        auto newToTensorOp = rewriter.create<bufferization::ToTensorOp>(
            ToTensorOp.getLoc(), newMemExpandOp.getResult(), true, true);

        rewriter.replaceOp(sliceOp, newToTensorOp);
        if (ToTensorOp->use_empty())
          rewriter.eraseOp(ToTensorOp);
        if (memExpandOp->use_empty())
          rewriter.eraseOp(memExpandOp);
        if (memorySpaceCastOp->use_empty())
          rewriter.eraseOp(memorySpaceCastOp);
        if (UbAllocOp->use_empty())
          rewriter.eraseOp(UbAllocOp);

        return success();
      }
    }
  }

  // If it is not the above patterns, return failure()
  return failure();
}

bool FixpipeBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<hivm::FixpipeOp>(sourceOp) && !isDynamicSlice(sliceOp);
}

LogicalResult
FixpipeBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                 PatternRewriter &rewriter) const {
  auto fixpipeOp =
      dyn_cast<hivm::FixpipeOp>(sliceOp.getSource().getDefiningOp());

  if (!fixpipeOp)
    return failure();

  // Utilize the offsets, sizes, and strides from ExtractSliceOp
  auto offsets = sliceOp.getMixedOffsets();
  auto sizes = sliceOp.getMixedSizes();
  auto strides = sliceOp.getMixedStrides();

  auto originalType =
      dyn_cast<RankedTensorType>(fixpipeOp.getResult(0).getType());
  if (!originalType) {
    return failure();
  }
  ArrayRef<int64_t> originalShape = originalType.getShape();

  auto slicedType = cast<RankedTensorType>(sliceOp.getResult().getType());
  if (!slicedType) {
    return failure();
  }
  ArrayRef<int64_t> slicedShape = slicedType.getShape();

  // Initialize the split mode
  auto splitMode = hivm::FixpipeDualDstMode::ROW_SPLIT;

  // Try to match the split mode by comparing the shapes of sliceOp and
  // fixpipeOp
  if (slicedShape[0] < originalShape[0]) {
    splitMode = hivm::FixpipeDualDstMode::ROW_SPLIT;
  } else if (slicedShape[1] < originalShape[1]) {
    splitMode = hivm::FixpipeDualDstMode::COLUMN_SPLIT;
  } else {
    return failure();
  }

  rewriter.setInsertionPoint(fixpipeOp);
  auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp.getLoc(), fixpipeOp.getOperand(1), offsets, sizes, strides);

  markCreatedExtractSliceOp(rewriter, newSliceOp);

  auto dualAttr =
      hivm::FixpipeDualDstModeAttr::get(rewriter.getContext(), splitMode);
  NamedAttrList attrs(fixpipeOp->getAttrs());
  attrs.set(fixpipeOp.getDualDstModeAttrName(), dualAttr);
  auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
      sliceOp.getLoc(), TypeRange{sliceOp.getType()},
      ValueRange{fixpipeOp.getSrc(), newSliceOp.getResult()}, attrs.getAttrs());

  rewriter.replaceOp(sliceOp, newFixpipeOp);
  if (fixpipeOp->use_empty())
    rewriter.eraseOp(fixpipeOp);

  return success();
}

bool BitcastBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<hivm::BitcastOp>(sourceOp);
}

LogicalResult
BitcastBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                 PatternRewriter &rewriter) const {
  auto bitcastOp =
      dyn_cast<hivm::BitcastOp>(sliceOp.getSource().getDefiningOp());
  if (!bitcastOp)
    return failure();

  auto loc = bitcastOp.getLoc();

  // Extract slice parameters
  auto offsets = sliceOp.getMixedOffsets();
  auto sizes = sliceOp.getMixedSizes();
  auto strides = sliceOp.getMixedStrides();

  // 1. Slice the bitcast source
  rewriter.setInsertionPoint(bitcastOp);
  auto newSlicedInput = rewriter.create<tensor::ExtractSliceOp>(
      loc, bitcastOp.getSrc(), offsets, sizes, strides);
  markCreatedExtractSliceOp(rewriter, newSlicedInput);

  // 2. Create a new bitcast on sliced input
  rewriter.setInsertionPointAfter(bitcastOp);
  auto newBitcast = rewriter.create<hivm::BitcastOp>(
      loc, sliceOp.getType(), newSlicedInput.getResult());

  // 3. Replace the original extract_slice
  rewriter.replaceOp(sliceOp, newBitcast.getResult());

  return success();
}

bool IfBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<scf::IfOp>(sourceOp) && !isDynamicSlice(sliceOp);
}

LogicalResult IfBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                          PatternRewriter &rewriter) const {
  auto ifOp = dyn_cast<scf::IfOp>(sliceOp.getSource().getDefiningOp());
  if (!ifOp)
    return rewriter.notifyMatchFailure(sliceOp,
                                       "source failed to bind to scf.if");

  auto yieldIndex = cast<OpResult>(sliceOp.getSource()).getResultNumber();
  LDBG("Processing result of " << yieldIndex << " from if op " << ifOp);

  // then block
  {
    Block &thenBlock = ifOp.getThenRegion().front();
    auto *yieldOp = thenBlock.getTerminator();
    rewriter.setInsertionPoint(yieldOp);

    auto thenYieldVal = yieldOp->getOperand(yieldIndex);
    auto newThenSlice = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(), cast<RankedTensorType>(sliceOp.getType()),
        thenYieldVal, sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
        sliceOp.getMixedStrides());
    markCreatedExtractSliceOp(rewriter, newThenSlice);

    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp->setOperand(yieldIndex, newThenSlice.getResult());
    });
  }

  // else block (if present)
  if (!ifOp.getElseRegion().empty()) {
    Block &elseBlock = ifOp.getElseRegion().front();
    auto *yieldOp = elseBlock.getTerminator();
    rewriter.setInsertionPoint(yieldOp);

    auto elseYieldVal = yieldOp->getOperand(yieldIndex);
    auto newElseSlice = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(), cast<RankedTensorType>(sliceOp.getType()),
        elseYieldVal, sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
        sliceOp.getMixedStrides());
    markCreatedExtractSliceOp(rewriter, newElseSlice);

    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp->setOperand(yieldIndex, newElseSlice.getResult());
    });
  }

  // update ifOp result type
  rewriter.modifyOpInPlace(
      ifOp, [&]() { ifOp.getResult(yieldIndex).setType(sliceOp.getType()); });

  rewriter.replaceAllUsesWith(sliceOp, ifOp.getResult(yieldIndex));

  return success();
}

bool SelectBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {

  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<arith::SelectOp>(sourceOp) &&
         mlir::isa<RankedTensorType>(sliceOp.getSource().getType());
}

LogicalResult SelectBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                              PatternRewriter &rewriter) const {

  auto selectOp =
      dyn_cast<arith::SelectOp>(sliceOp.getSource().getDefiningOp());
  if (!selectOp)
    return failure();

  // only support tensor select
  if (!mlir::isa<RankedTensorType>(sliceOp.getSource().getType()))
    return failure();

  auto loc = selectOp.getLoc();

  auto offsets = sliceOp.getMixedOffsets();
  auto sizes = sliceOp.getMixedSizes();
  auto strides = sliceOp.getMixedStrides();

  Value cond = selectOp.getCondition();
  Value trueVal = selectOp.getTrueValue();
  Value falseVal = selectOp.getFalseValue();

  rewriter.setInsertionPoint(selectOp);

  auto slicedTrue = rewriter.create<tensor::ExtractSliceOp>(
      loc, cast<RankedTensorType>(sliceOp.getType()), trueVal, offsets, sizes,
      strides);
  markCreatedExtractSliceOp(rewriter, slicedTrue);

  auto slicedFalse = rewriter.create<tensor::ExtractSliceOp>(
      loc, cast<RankedTensorType>(sliceOp.getType()), falseVal, offsets, sizes,
      strides);
  markCreatedExtractSliceOp(rewriter, slicedFalse);

  auto newSelect = rewriter.create<arith::SelectOp>(
      loc, sliceOp.getType(), cond, slicedTrue.getResult(),
      slicedFalse.getResult());

  rewriter.replaceOp(sliceOp, newSelect.getResult());

  return success();
}

} // namespace mlir::hivm::detail
