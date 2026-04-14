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
#include "bishengir/Dialect/Scope/IR/Scope.h"
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

  auto newOp = rewriter.create<OpTy2>(childOp->getLoc(), createdNewParent,
      std::forward(args)..., newViewOffsets,
      newViewSizes, parentOp.getMixedStrides());
  for (auto attr : childOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()) && 
        attr.getName() != toBeBubbleUpSlice)
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  return newOp;
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
  return isa_and_nonnull<hivm::VBrcOp>(sourceOp);
}

LogicalResult
BroadcastBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                   PatternRewriter &rewriter) const {
  auto broadcastOp =
      dyn_cast<hivm::VBrcOp>(sliceOp.getSource().getDefiningOp());
  if (!broadcastOp)
    return failure();

  auto outputType = dyn_cast<RankedTensorType>(broadcastOp.getDst().getType());

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
  return isa_and_nonnull<hivm::VReduceOp>(sourceOp);
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
  if (ret < 0)
    return std::nullopt;
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
  SmallVector<OpFoldResult> inputOffsets(inputRank, rewriter.getIndexAttr(0)),
      inputSizes(inputRank, rewriter.getIndexAttr(1)),
      inputStrides(inputRank, rewriter.getIndexAttr(1));

  const int64_t inIdx = *nonUnitInput;
  const int64_t outIdx = *nonUnitOutput;
  const int64_t irank = static_cast<int64_t>(inputRank);
  const int64_t orank = static_cast<int64_t>(outputOffsets.size());
  if (inIdx < 0 || outIdx < 0 || inIdx >= irank || outIdx >= orank ||
      static_cast<int64_t>(outputSizes.size()) != orank)
    return failure();

  inputOffsets[static_cast<size_t>(inIdx)] =
      outputOffsets[static_cast<size_t>(outIdx)];
  inputSizes[static_cast<size_t>(inIdx)] =
      outputSizes[static_cast<size_t>(outIdx)];

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
  for (auto attr : parentSliceOp->getAttrs()) {
    if (!newParentSliceOp->hasAttr(attr.getName()) && 
        attr.getName() != toBeBubbleUpSlice)
      newParentSliceOp->setAttr(attr.getName(), attr.getValue());
  }
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
  return !isDynamicSlice(sliceOp);
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
handleSliceInsertDimOnly(tensor::ExtractSliceOp sliceOp,
                         tensor::InsertSliceOp parentInsertOp, size_t tilingDim,
                         PatternRewriter &rewriter) {
  auto newDst = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp.getLoc(), parentInsertOp.getDest(), sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, newDst);
  auto newOffsets = newDst.getMixedOffsets();
  auto newSizes = newDst.getMixedSizes();
  newOffsets[tilingDim] = parentInsertOp.getMixedOffsets()[tilingDim];
  newSizes[tilingDim] = parentInsertOp.getMixedSizes()[tilingDim];
  rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
      sliceOp, parentInsertOp.getSource(), newDst, newOffsets, newSizes,
      parentInsertOp.getMixedStrides());
  return success();
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
  auto extractDims = getExtractOrInsertDim(sliceOp);
  if (extractDims.size() != 1)
    return failure();

  auto tilingDim = *extractDims.begin();
  extractDims = getExtractOrInsertDim(parentInsertOp);
  // Note: be extremely careful when handling such case, and not all cases
  // can be bubbled up.
  if (extractDims.size() != 1 || tilingDim != *extractDims.begin()) {
    // We are being very conservative that, only handling the case when
    // inserting to single dim, and it's overlaps with extract dim.
    // It probably can be enhanced, but need to be very careful.
    return failure();
  }

  // If this insertSlice is not created by Tiling, it's very dangerous for us
  // to bubbled up, because the semantic may not be guaranteed to be the same.
  if (!createdByTiling(parentInsertOp)) {
    if (parentInsertOp.getSourceType().getDimSize(tilingDim) == 1) {
      // This case is equivalent to handleInsertRankedReduceCase
      return handleSliceInsertDimOnly(sliceOp, parentInsertOp, tilingDim,
                                      rewriter);
    }
    return failure();
  }

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

static LogicalResult
handleExtractInsertExtractSameDimCase(tensor::ExtractSliceOp sliceOp,
                                      PatternRewriter &rewriter) {
  auto parentInsertOp =
      sliceOp.getSource().getDefiningOp<tensor::InsertSliceOp>();
  auto srcExtractOp =
      parentInsertOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();

  auto extractDims = getExtractOrInsertDim(sliceOp);
  if (extractDims.size() != 1)
    return failure();
  auto tilingDim = *extractDims.begin();

  rewriter.setInsertionPoint(parentInsertOp);
  auto newDst = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), parentInsertOp.getDest(), sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, newDst);

  rewriter.setInsertionPoint(srcExtractOp);
  auto newSrcSrc = rewriter.create<tensor::ExtractSliceOp>(
      srcExtractOp->getLoc(), srcExtractOp.getSource(),
      sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
      sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, newDst);

  auto sizes = srcExtractOp.getMixedSizes();
  auto offsetVal = getValueOrCreateConstantIndexOp(
      rewriter, sliceOp.getLoc(), sliceOp.getMixedOffsets()[tilingDim]);
  auto sizeVal = getValueOrCreateConstantIndexOp(rewriter, sliceOp.getLoc(),
                                                 sizes[tilingDim]);
  auto tilingSize = getValueOrCreateConstantIndexOp(
      rewriter, sliceOp.getLoc(), sliceOp.getMixedSizes()[tilingDim]);
  offsetVal =
      rewriter.create<arith::MinSIOp>(offsetVal.getLoc(), offsetVal, sizeVal);
  sizeVal =
      rewriter.create<arith::SubIOp>(sizeVal.getLoc(), sizeVal, offsetVal);
  sizeVal =
      rewriter.create<arith::MinSIOp>(sizeVal.getLoc(), sizeVal, tilingSize);
  sizes[tilingDim] = sizeVal;

  srcExtractOp = rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
      srcExtractOp, newSrcSrc, srcExtractOp.getMixedOffsets(), sizes,
      srcExtractOp.getMixedStrides());
  rewriter.setInsertionPoint(parentInsertOp);
  rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
      sliceOp, srcExtractOp, newDst, parentInsertOp.getMixedOffsets(), sizes,
      parentInsertOp.getMixedStrides());
  rewriter.eraseOp(parentInsertOp);
  return success();
}

static LogicalResult
handleExtractOfInsertDifferentDimCase(tensor::ExtractSliceOp sliceOp,
                                      PatternRewriter &rewriter) {
  auto parentSliceOp =
      sliceOp.getSource().getDefiningOp<tensor::InsertSliceOp>();
  auto srcType = parentSliceOp.getSourceType();
  if (ShapedType::isDynamicShape(srcType.getShape()))
    return failure();
  auto parentExtractDims = getExtractOrInsertDim(parentSliceOp);
  auto extractDims = getExtractOrInsertDim(sliceOp);
  auto parentOffsets = parentSliceOp.getMixedOffsets();
  auto parentSizes = parentSliceOp.getMixedSizes();
  auto offsets = sliceOp.getMixedOffsets();
  auto sizes = sliceOp.getMixedSizes();
  auto strides = sliceOp.getMixedStrides();
  auto newDst = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), parentSliceOp.getDest(), offsets, sizes, strides);
  markCreatedExtractSliceOp(rewriter, newDst);
  for (auto dim : getExtractOrInsertDim(parentSliceOp)) {
    std::swap(parentOffsets[dim], offsets[dim]);
    sizes[dim] = parentSizes[dim];
    parentSizes[dim] = rewriter.getIndexAttr(srcType.getDimSize(dim));
  }
  for (auto dim : getExtractOrInsertDim(sliceOp)) {
    std::swap(parentOffsets[dim], offsets[dim]);
    parentSizes[dim] = sizes[dim];
  }

  auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), parentSliceOp.getSource(), parentOffsets, parentSizes,
      strides);
  markCreatedExtractSliceOp(rewriter, newSliceOp);

  auto newParentSliceOp = rewriter.create<tensor::InsertSliceOp>(
      sliceOp->getLoc(), newSliceOp, newDst, offsets, sizes, strides);

  for (auto attr : parentSliceOp->getAttrs()) {
    if (!newParentSliceOp->hasAttr(attr.getName()))
      newParentSliceOp->setAttr(attr.getName(), attr.getValue());
  }

  rewriter.replaceOp(sliceOp, newParentSliceOp);
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
       0) &&
      !isDynamicSlice(parentInsertOp)) {
    return handleInsertRankedReduceCase(sliceOp, rewriter);
  }

  // Handle extract and insert on same dimension case.
  if (!llvm::set_intersection(getExtractOrInsertDim(parentInsertOp),
                              getExtractOrInsertDim(sliceOp))
           .empty()) {
    // handling special case
    // ex)
    // %extracted_slice = tensor.extract_slice %src[0] [%size] [1] :
    // tensor<16xf32> to tensor<?xf32> %inserted_slice = tensor.insert_slice
    // %extracted_slice into %10[0] [%size] [1] : tensor<?xf32> into
    // tensor<16xf32> %to_bubble_up = tensor.extract_slice
    // %inserted_slice[%offset] [8] [1] {to_be_bubbled_slice} : tensor<16xf32>
    // to tensor<8xf32>
    if (auto srcExtractOp =
            parentInsertOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
        srcExtractOp && srcExtractOp->hasOneUse() &&
        srcExtractOp.getSource().getType() == sliceOp.getSource().getType() &&
        (!sliceOp.hasZeroOffset() || !srcExtractOp.hasZeroOffset() ||
         !sliceOp.hasUnitStride() || !srcExtractOp.hasUnitStride())) {
      return handleExtractInsertExtractSameDimCase(sliceOp, rewriter);
    }
    if (!isDynamicSlice(parentInsertOp))
      return handleExtractOfInsertSameDimCase(sliceOp, rewriter);
  } else {
    return handleExtractOfInsertDifferentDimCase(sliceOp, rewriter);
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
  auto inputType = collapseOp.getSrcType();
  auto outputType = collapseOp.getType();
  auto collapseDims = collapseOp.getReassociationIndices();
  auto extractDims = getExtractOrInsertDim(sliceOp);
  if (extractDims.size() != 1)
    return failure();
  auto tilingDim = *extractDims.begin();
  int64_t newTilingDim = -1;
  // Get and check the collapse dimensions
  for (auto idx : collapseDims[tilingDim]) {
    auto dimSize = inputType.getDimSize(idx);
    if (dimSize != 1) {
      // We only support bubble up for simple collapse: Ax1 or 1xA -> A
      if (newTilingDim != -1)
        return failure();
      newTilingDim = idx;
    }
  }

  if (newTilingDim == -1)
    return failure();

  for (auto [idx, dimSize] : llvm::enumerate(inputType.getShape())) {
    if (static_cast<int64_t>(idx) != newTilingDim && dimSize != 1)
      return failure();
  }

  auto inputRank = inputType.getRank();
  auto outputRank = outputType.getRank();

  // Get the offsets and sizes from the slice operation
  auto outputOffsets = sliceOp.getMixedOffsets();
  auto outputSizes = sliceOp.getMixedSizes();

  // Compute the input offsets and sizes
  SmallVector<OpFoldResult> inputOffsets(inputRank);
  SmallVector<OpFoldResult> inputSizes(inputRank);
  auto inputCollapse = collapseOp->getOperand(0);
  auto mixedSizeFinal =
      tensor::getMixedSizes(rewriter, collapseOp.getLoc(), inputCollapse);

  for (unsigned outIdx = 0; outIdx < outputRank; ++outIdx) {
    for (auto inIdx : collapseDims[outIdx]) {
      if (inIdx != newTilingDim) {
        inputOffsets[inIdx] = rewriter.getIndexAttr(0);
        inputSizes[inIdx] =
            (inputType.isDynamicDim(inIdx))
                ? mixedSizeFinal[inIdx]
                : rewriter.getIndexAttr(inputType.getDimSize(inIdx));
      } else {
        inputOffsets[inIdx] = outputOffsets[outIdx];
        inputSizes[inIdx] = outputSizes[outIdx];
      }
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
  return isa_and_nonnull<scf::ForOp, scf::WhileOp>(sourceOp) &&
         !isDynamicSlice(sliceOp);
}

static void sliceRegionIterArg(BlockArgument regionIterArg,
                               tensor::ExtractSliceOp sliceOp, Location loc,
                               PatternRewriter &rewriter) {
  regionIterArg.setType(sliceOp.getType());
  rewriter.setInsertionPointAfterValue(regionIterArg);
  auto tmpEmpty = rewriter.create<tensor::EmptyOp>(loc, sliceOp.getSourceType(),
                                                   ValueRange{});
  auto argumentInsert = rewriter.create<tensor::InsertSliceOp>(
      loc, regionIterArg, tmpEmpty, sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  rewriter.replaceAllUsesExcept(regionIterArg, argumentInsert.getResult(),
                                argumentInsert);
}

LogicalResult LoopBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                            PatternRewriter &rewriter) const {
  auto *defOp = sliceOp.getSource().getDefiningOp();
  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    Value oldStep = forOp.getStep();
    auto oldStepAsIndexOp = oldStep.getDefiningOp<arith::ConstantIndexOp>();
    if (oldStepAsIndexOp && oldStepAsIndexOp.value() != 1) {
      bishengir::normalizeLoop(rewriter, forOp, oldStep);
      return success();
    }

    auto yieldIndex = cast<OpResult>(sliceOp.getSource()).getResultNumber();
    LDBG("Processing result of " << yieldIndex << " from for op " << forOp);
    auto valueToSlice = forOp.getYieldedValues()[yieldIndex];
    Operation *yieldOp =
        forOp.getRegion().getBlocks().rbegin()->getTerminator();
    rewriter.setInsertionPoint(yieldOp);
    auto newMovedInSlice = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp->getLoc(),
        /* resultType */ cast<RankedTensorType>(sliceOp.getType()),
        /* src */ valueToSlice, sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
    markCreatedExtractSliceOp(rewriter, newMovedInSlice);

    LDBG(valueToSlice);
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      auto &yieldValueOpr = yieldOp->getOpOperand(yieldIndex);
      yieldValueOpr.assign(newMovedInSlice.getResult());
    });

    BlockArgument regionIterArg = forOp.getRegionIterArg(yieldIndex);
    sliceRegionIterArg(regionIterArg, sliceOp, forOp.getLoc(), rewriter);

    OpOperand &forOpInit = forOp.getInitsMutable()[yieldIndex];
    rewriter.setInsertionPoint(forOp);
    auto slicedInit = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp->getLoc(),
        /* resultType */ cast<RankedTensorType>(sliceOp.getType()),
        /* src */ forOpInit.get(), sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
    markCreatedExtractSliceOp(rewriter, slicedInit);

    forOpInit.set(slicedInit.getResult());
    rewriter.modifyOpInPlace(forOp, [&]() {
      forOp.getResult(yieldIndex).setType(sliceOp.getType());
    });
    rewriter.replaceOp(sliceOp, forOp->getResult(yieldIndex));
    return success();
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
    auto yieldIndex = cast<OpResult>(sliceOp.getSource()).getResultNumber();
    auto conditionOp = whileOp.getConditionOp();
    auto valueToSlice = conditionOp.getArgs()[yieldIndex];

    rewriter.setInsertionPoint(conditionOp);
    auto newMovedInSlice = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp->getLoc(),
        /* resultType */ cast<RankedTensorType>(sliceOp.getType()),
        /* src */ valueToSlice, sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
    markCreatedExtractSliceOp(rewriter, newMovedInSlice);
    rewriter.modifyOpInPlace(conditionOp, [&]() {
      auto &yieldValueOpr = conditionOp.getArgsMutable()[yieldIndex];
      yieldValueOpr.assign(newMovedInSlice.getResult());
    });

    BlockArgument regionIterArg = whileOp.getAfterArguments()[yieldIndex];
    sliceRegionIterArg(regionIterArg, sliceOp, whileOp.getLoc(), rewriter);
    rewriter.modifyOpInPlace(whileOp, [&]() {
      whileOp.getResult(yieldIndex).setType(sliceOp.getType());
    });
    rewriter.replaceOp(sliceOp, whileOp->getResult(yieldIndex));
    return success();
  }
  return failure();
}

bool LoopArgsBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto blockArg = dyn_cast<BlockArgument>(sliceOp.getSource());
  if (!blockArg)
    return false;
  auto whileOp = dyn_cast_or_null<scf::WhileOp>(sliceOp->getParentOp());
  if (!whileOp)
    return false;
  return llvm::any_of(
      whileOp.getBeforeArguments(),
      [&blockArg](auto beforeArg) { return blockArg == beforeArg; });
}

LogicalResult
LoopArgsBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                  PatternRewriter &rewriter) const {
  auto whileOp = sliceOp->getParentOfType<scf::WhileOp>();
  if (!whileOp) {
    return failure();
  }

  BlockArgument blockArg = dyn_cast<BlockArgument>(sliceOp.getSource());
  if (!blockArg)
    return failure();

  auto blockArgIdx = blockArg.getArgNumber();

  rewriter.setInsertionPoint(whileOp);
  auto movedOutSlice = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), cast<RankedTensorType>(sliceOp.getType()),
      whileOp.getInits()[blockArgIdx], sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, movedOutSlice);

  blockArg.setType(sliceOp.getType());
  rewriter.replaceAllUsesWith(sliceOp, blockArg);
  rewriter.modifyOpInPlace(whileOp, [&]() {
    whileOp.getInitsMutable()[blockArgIdx].set(movedOutSlice);
  });

  auto yieldOp = whileOp.getYieldOp();
  rewriter.setInsertionPoint(yieldOp);
  movedOutSlice = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), cast<RankedTensorType>(sliceOp.getType()),
      yieldOp.getResults()[blockArgIdx], sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, movedOutSlice);

  rewriter.modifyOpInPlace(yieldOp, [&]() {
    auto &yieldValueOpr = yieldOp.getResultsMutable()[blockArgIdx];
    yieldValueOpr.assign(movedOutSlice.getResult());
  });

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

bool BufferizationBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<bufferization::ToTensorOp>(sourceOp) &&
         !isDynamicSlice(sliceOp);
}

LogicalResult
BufferizationBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                       PatternRewriter &rewriter) const {
  auto toTensorOp =
      dyn_cast<bufferization::ToTensorOp>(sliceOp.getSource().getDefiningOp());

  if (!toTensorOp)
    return failure();

  LDBG("BufferizationBubbleUpStrategy\n" << toTensorOp);

  // Utilize the offsets, sizes, and strides from ExtractSliceOp
  auto offsets = sliceOp.getMixedOffsets();
  auto sizes = sliceOp.getMixedSizes();
  auto strides = sliceOp.getMixedStrides();

  auto srcMemref = toTensorOp.getMemref();

  for (Operation *userOp : srcMemref.getUsers()) {
    // Pattern 1: This deals with the pattern: memref.alloc() ->
    // bufferization.to_tensor, with Load Op
    if (auto loadOp = dyn_cast<hivm::LoadOp>(userOp)) {
      LDBG("Pattern 1:\n" << loadOp);

      // For Operand(0): if the Operand(0) of LoadOP is memref.reinterpret_cast
      auto castOp = dyn_cast<memref::ReinterpretCastOp>(
          loadOp.getOperand(0).getDefiningOp());
      if (castOp) {
        rewriter.setInsertionPoint(loadOp);
        auto castSubviewOp = rewriter.create<memref::SubViewOp>(
            castOp.getLoc(), castOp.getResult(), offsets, sizes, strides);

        rewriter.modifyOpInPlace(
            loadOp, [&]() { loadOp.setOperand(0, castSubviewOp.getResult()); });
      }

      // For Operand(1): if the Operand(1) of LoadOP is memref.alloc()
      auto AllocOp =
          dyn_cast<memref::AllocOp>(loadOp.getOperand(1).getDefiningOp());
      if (AllocOp) {
        rewriter.setInsertionPoint(AllocOp);
        Location loc = AllocOp.getLoc();

        auto resultType =
            dyn_cast<RankedTensorType>(sliceOp.getResult().getType());

        auto memrefType =
            MemRefType::get(resultType.getShape(), resultType.getElementType());
        // Create a newAllocOp
        auto newAllocOp = rewriter.create<memref::AllocOp>(loc, memrefType);

        rewriter.modifyOpInPlace(
            loadOp, [&]() { loadOp.setOperand(1, newAllocOp.getResult()); });

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
        rewriter.replaceOp(toTensorOp, newToTensorOp);
        rewriter.eraseOp(subviewOp);
      }
      return success();
    }
    // Pattern 2: This deals with the pattern: memref.alloc() ->
    // bufferization.to_tensor, with Subview Op + Load Op
    if (auto subViewOp = dyn_cast<memref::SubViewOp>(userOp)) {
      if (!subViewOp->hasOneUse() || !subViewOp.hasUnitStride())
        continue;
      auto loadOp = dyn_cast<hivm::LoadOp>(*subViewOp->user_begin());
      if (!loadOp)
        continue;
      auto subViewOpGM = loadOp.getSrc().getDefiningOp<memref::SubViewOp>();
      if (!subViewOpGM || !subViewOpGM.hasUnitStride())
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
           << subViewOp << "\n"
           << loadOp);
      rewriter.setInsertionPoint(subViewOpGM);

      // Compute new size
      auto newOffsets = subViewOp.getMixedOffsets();
      auto newOffsetsGM = subViewOpGM.getMixedOffsets();
      auto newSizes = subViewOp.getMixedSizes();
      auto newSizesGM = subViewOpGM.getMixedSizes();
      auto extractDims = getExtractOrInsertDim(sliceOp);
      if (extractDims.size() != 1)
        continue;
      auto tilingDim = *extractDims.begin();
      rewriter.setInsertionPoint(subViewOp);
      handleExtractOfExtract(newOffsets[tilingDim], newSizes[tilingDim],
                             offsets[tilingDim], sizes[tilingDim],
                             subViewOp.getLoc(), rewriter);

      // Rewrite operations
      rewriter.setInsertionPoint(subViewOpGM);
      handleExtractOfExtract(newOffsetsGM[tilingDim], newSizesGM[tilingDim],
                             offsets[tilingDim], sizes[tilingDim],
                             subViewOpGM.getLoc(), rewriter);
      auto newCastValue = rewriter.create<memref::SubViewOp>(
          castOp.getLoc(), castOp, offsets, sizes, strides);
      auto newSubViewOpGM = rewriter.create<memref::SubViewOp>(
          subViewOpGM.getLoc(), newCastValue, newOffsetsGM, newSizesGM,
          subViewOpGM.getMixedStrides());
      rewriter.setInsertionPoint(allocOp);
      auto resultType =
          dyn_cast<RankedTensorType>(sliceOp.getResult().getType());
      allocOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(
          allocOp,
          MemRefType::get(resultType.getShape(), resultType.getElementType()));
      rewriter.setInsertionPoint(subViewOp);
      auto newSubViewOp = rewriter.create<memref::SubViewOp>(
          subViewOp.getLoc(), allocOp, newOffsets, newSizes,
          subViewOp.getMixedStrides());

      rewriter.modifyOpInPlace(loadOp, [&]() {
        loadOp.getSrcMutable().set(newSubViewOpGM);
        loadOp.getDstMutable().set(newSubViewOp);
      });

      rewriter.setInsertionPoint(sliceOp);
      auto newToTensorOp =
          rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
              sliceOp, allocOp.getResult(), true, true);

      rewriter.replaceOp(toTensorOp, newToTensorOp);

      LDBG("After Pattern 2:\n"
           << newCastValue << "\n"
           << newSubViewOpGM << "\n"
           << allocOp << "\n"
           << newSubViewOp << "\n"
           << loadOp);
      LDBG(newSubViewOp->getParentOfType<func::FuncOp>());
      return success();
    }
  }

  // If it is not the above patterns, return failure()
  return failure();
}

bool VTransposeBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  // Check if source is a block argument of a for loop
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  if (!sourceOp)
    return false;
  return isa_and_nonnull<hivm::VTransposeOp>(sourceOp) &&
         !isDynamicSlice(sliceOp);
}

LogicalResult
VTransposeBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                    PatternRewriter &rewriter) const {
  auto transOp =
      dyn_cast<hivm::VTransposeOp>(sliceOp.getSource().getDefiningOp());
  if (!transOp)
    return failure();

  auto resultType = cast<RankedTensorType>(sliceOp.getType());
  auto resultRank = resultType.getRank();

  auto resultShape = llvm::to_vector(resultType.getShape());
  auto perm = transOp.getPermutation();

  rewriter.setInsertionPoint(transOp);
  auto dstOffsets = sliceOp.getMixedOffsets();
  auto srcOffsets = SmallVector<OpFoldResult>(resultRank);
  auto srcSizes = llvm::to_vector(sliceOp.getMixedSizes());
  auto srcStrides = sliceOp.getMixedStrides();

  // Infer the sizes and offsets of newSliceOp by perm
  for (int64_t i = 0; i < resultRank; ++i) {
    srcSizes[perm[i]] = rewriter.getIndexAttr(resultShape[i]);
    srcOffsets[perm[i]] = dstOffsets[i];
  }

  rewriter.setInsertionPoint(sliceOp);
  auto newSrc = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp.getLoc(), transOp.getSrc(), srcOffsets, srcSizes, srcStrides);

  auto newDst = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp.getLoc(), transOp.getDst(), dstOffsets, sliceOp.getMixedSizes(),
      sliceOp.getMixedStrides());

  markCreatedExtractSliceOp(rewriter, newSrc);
  markCreatedExtractSliceOp(rewriter, newDst);

  if (!utils::isAlignedInUB(newSrc.getType()) ||
      !utils::isAlignedInUB(newDst.getType())) {
    rewriter.eraseOp(newSrc);
    rewriter.eraseOp(newDst);
    return failure();
  }

  auto newTransOp = rewriter.create<hivm::VTransposeOp>(
      sliceOp.getLoc(), sliceOp.getResultType(), newSrc, newDst,
      rewriter.getDenseI64ArrayAttr(perm));

  rewriter.replaceOp(sliceOp, newTransOp);
  rewriter.eraseOp(transOp);

  return success();
}

bool IfBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto sourceOp = sliceOp.getSource().getDefiningOp<scf::IfOp>();
  return sourceOp && !isDynamicSlice(sliceOp);
}

LogicalResult IfBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                          PatternRewriter &rewriter) const {
  auto src = cast<OpResult>(sliceOp.getSource());
  auto ifOp = src.getDefiningOp<scf::IfOp>();
  if (!ifOp)
    return failure();

  auto yieldIndex = src.getResultNumber();
  for (auto yieldOp : {ifOp.thenYield(), ifOp.elseYield()}) {
    rewriter.setInsertionPoint(yieldOp);
    auto newMovedInSlice = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp->getLoc(), yieldOp.getResults()[yieldIndex],
        sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
        sliceOp.getMixedStrides());
    markCreatedExtractSliceOp(rewriter, newMovedInSlice);
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      auto &yieldValueOpr = yieldOp->getOpOperand(yieldIndex);
      yieldValueOpr.assign(newMovedInSlice.getResult());
    });
  }

  rewriter.modifyOpInPlace(
      ifOp, [&]() { ifOp.getResult(yieldIndex).setType(sliceOp.getType()); });
  rewriter.replaceAllUsesWith(sliceOp, ifOp->getResult(yieldIndex));

  return success();
}

bool ScopeBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto sourceOp = sliceOp.getSource().getDefiningOp<scope::ScopeOp>();
  return sourceOp && !isDynamicSlice(sliceOp);
}

LogicalResult ScopeBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                             PatternRewriter &rewriter) const {
  auto src = cast<OpResult>(sliceOp.getSource());
  auto scopeOp = src.getDefiningOp<scope::ScopeOp>();
  if (!scopeOp)
    return failure();

  auto returnIndex = src.getResultNumber();
  auto returnOp =
      cast<scope::ReturnOp>(scopeOp.getRegion().front().getTerminator());
  rewriter.setInsertionPoint(returnOp);
  auto newMovedInSlice = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp->getLoc(), returnOp.getResults()[returnIndex],
      sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
      sliceOp.getMixedStrides());
  markCreatedExtractSliceOp(rewriter, newMovedInSlice);
   auto &returnValueOpr = returnOp->getOpOperand(returnIndex);
   rewriter.modifyOpInPlace(returnOp, [&returnValueOpr, &newMovedInSlice]() {
     returnValueOpr.assign(newMovedInSlice.getResult());
   });

  rewriter.modifyOpInPlace(scopeOp, [&]() {
    scopeOp->getResult(returnIndex).setType(sliceOp.getType());
  });
  rewriter.replaceAllUsesWith(sliceOp, scopeOp->getResult(returnIndex));

  return success();
}

} // namespace mlir::hivm::detail
