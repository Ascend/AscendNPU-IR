//===- Pattern.cpp --------------------------------------------------------===//
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
  assert(sliceParentRegion->getParentOp() &&
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
                               OpTy createdNewParent, Arg &&...args) {
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
  // Get first non unit

  auto outputShape = outputType.getShape();
  auto nonUnitOutput = findOnlyNonUnit(outputShape);
  auto nonUnitInput = findOnlyNonUnit(expandOp.getSrcType().getShape());
  if (!nonUnitOutput.has_value())
    return failure();
  if (!nonUnitInput.has_value())
    return failure();
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
  rewriter.eraseOp(expandOp);
  return success();
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
  auto collapseDims = collapseOp.getReassociationIndices();
  // Get and check the collapse dimensions
  // We only support bubble up for simple collapse: Ax1 or 1xA -> A
  if (!inputType || inputType.getRank() > 2 || collapseDims.size() > 1)
    return failure();

  auto inputRank = inputType.getRank();
  BitVector isCollapseDim(inputRank, false);

  if (inputType.getDimSize(0) == 1) {
    isCollapseDim[0] = true;
  } else if (inputType.getDimSize(1) == 1) {
    isCollapseDim[1] = true;
  } else {
    return failure();
  }

  // Get the offsets and sizes from the slice operation
  auto outputOffsets = sliceOp.getMixedOffsets();
  auto outputSizes = sliceOp.getMixedSizes();

  // Compute the input offsets and sizes
  unsigned outIdx = 0;
  SmallVector<OpFoldResult> inputOffsets(inputRank);
  SmallVector<OpFoldResult> inputSizes(inputRank);
  auto inputCollapse = collapseOp->getOperand(0);
  auto mixedSizeFinal =
      tensor::getMixedSizes(rewriter, collapseOp.getLoc(), inputCollapse);

  for (unsigned inIdx = 0; inIdx < inputRank; ++inIdx) {
    if (isCollapseDim[inIdx]) {
      inputOffsets[inIdx] = rewriter.getIndexAttr(0);
      inputSizes[inIdx] =
          (inputType.isDynamicDim(inIdx))
              ? mixedSizeFinal[inIdx]
              : rewriter.getIndexAttr(inputType.getDimSize(inIdx));
    } else {
      inputOffsets[inIdx] = outputOffsets[outIdx];
      inputSizes[inIdx] = outputSizes[outIdx];
      ++outIdx;
    }
  }

  SmallVector<OpFoldResult> inputStrides(inputRank, rewriter.getIndexAttr(1));
  rewriter.setInsertionPoint(collapseOp);
  auto tiledInput = rewriter.create<tensor::ExtractSliceOp>(
      inputCollapse.getLoc(), inputCollapse, inputOffsets, inputSizes,
      inputStrides);
  markCreatedExtractSliceOp(rewriter, tiledInput);

  auto staticOutputShape = decomposeMixedValues(outputSizes);
  auto newCollapse = rewriter.create<tensor::CollapseShapeOp>(
      collapseOp.getLoc(), tiledInput, collapseOp.getReassociationIndices());
  rewriter.replaceOp(sliceOp, newCollapse->getResults());
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

} // namespace mlir::hivm::detail
