//===- BufferizationPropagateOp.cpp - Propagate patterns for bubble-up ----===//
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

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/BubbleUpUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/BufferizationBubbleUp.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "bufferization-bubble-up-propagate-op"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: " << X << "\n")

namespace mlir::hivm::detail {
namespace {

/// Find the tiling dimension by comparing propagate_up UCC input/output shapes.
static FailureOr<int64_t>
findTilingDimFromPropagatorShapes(MemRefType parentType,
                                  MemRefType slicedType) {
  int64_t tilingDim = -1;
  for (int64_t dim = 0; dim < slicedType.getRank(); ++dim) {
    int64_t parentDim = parentType.getDimSize(dim);
    int64_t slicedDim = slicedType.getDimSize(dim);
    if (!ShapedType::isDynamic(slicedDim) &&
        !ShapedType::isDynamic(parentDim) && parentDim == slicedDim)
      continue;
    if (tilingDim != -1)
      return failure();
    tilingDim = dim;
  }
  if (tilingDim == -1)
    return failure();
  return tilingDim;
}

/// Derive subview offsets/sizes/strides from propagate_up UCC shapes: locate
/// tilingDim from the UCC result type, then reuse sub-block tiling helpers.
FailureOr<std::tuple<SmallVector<OpFoldResult, 4>, SmallVector<OpFoldResult, 4>,
                     SmallVector<OpFoldResult, 4>>>
getSubViewParamsFromPropagatorResult(UnrealizedConversionCastOp propagateOp,
                                     PatternRewriter &rewriter) {
  auto parentType = dyn_cast<MemRefType>(propagateOp.getInputs()[0].getType());
  auto slicedType = dyn_cast<MemRefType>(propagateOp.getResult(0).getType());
  if (!parentType || !slicedType)
    return failure();
  if (parentType.getRank() != slicedType.getRank())
    return failure();

  FailureOr<int64_t> maybeTilingDim =
      findTilingDimFromPropagatorShapes(parentType, slicedType);
  if (failed(maybeTilingDim))
    return failure();
  int64_t tilingDim = maybeTilingDim.value();

  FailureOr<scf::ForOp> maybeLoop =
      findContainingSubblockLoop(propagateOp.getOperation());
  if (failed(maybeLoop))
    return failure();
  scf::ForOp containingLoop = maybeLoop.value();

  Location loc = propagateOp.getLoc();
  Value parentValue = propagateOp.getInputs()[0];

  FailureOr<OpFoldResult> maybeTileSize =
      getSingleTileSize(rewriter, loc, parentValue, tilingDim, containingLoop);
  if (failed(maybeTileSize))
    return failure();

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(containingLoop.getBody());
  OpFoldResult offsetAtTileDim = calculateOffsetAtTilingDim(
      rewriter, loc, containingLoop, parentValue, tilingDim);

  SmallVector<OpFoldResult, 4> strides;
  SmallVector<OpFoldResult, 4> offsets;
  SmallVector<OpFoldResult, 4> sizes;
  SmallVector<int64_t, 4> newShape;
  if (failed(findCorrespondingSizesOffsetsStrides(
          rewriter, parentType, tilingDim, offsetAtTileDim,
          maybeTileSize.value(), strides, offsets, sizes, newShape)))
    return failure();

  return std::make_tuple(std::move(offsets), std::move(sizes),
                         std::move(strides));
}

} // namespace

// propagate up MemorySpaceCast Op
LogicalResult BufferizationPropagateUpPattern::propagateUpMemorySpaceCast(
    memref::MemorySpaceCastOp castOp, UnrealizedConversionCastOp propagateOp,
    PatternRewriter &rewriter) const {
  auto propagateResultType =
      dyn_cast<MemRefType>(propagateOp.getResult(0).getType());
  if (!propagateResultType)
    return failure();
  auto slicedTensorType = RankedTensorType::get(
      propagateResultType.getShape(), propagateResultType.getElementType());

  auto slicedCastResultType = getSlicedMemRefType(
      cast<MemRefType>(castOp.getResult().getType()), slicedTensorType);

  Value castSource = castOp.getSource();
  auto oldSourceType = dyn_cast<MemRefType>(castSource.getType());
  if (!oldSourceType)
    return failure();
  auto slicedSourceType = getSlicedMemRefType(oldSourceType, slicedTensorType);

  Value slicedSource;
  auto tilingDimInfo = getTilingDimInfo(propagateOp);
  UnrealizedConversionCastOp upOnSource = createBubblePropagatorUpLink(
      castSource, slicedSourceType, tilingDimInfo.offset, tilingDimInfo.size,
      tilingDimInfo.tilingDim, rewriter);
  slicedSource = upOnSource.getResult(0);
  // if (llvm::any_of(castSource.getUsers(), [](Operation *user) {
  //       return user->hasAttr(kBubbleUpPropagateUp);
  //     })) {
  //   for (Operation *user : castSource.getUsers()) {
  //     if (!user->hasAttr(kBubbleUpPropagateUp))
  //       continue;
  //     slicedSource = cast<UnrealizedConversionCastOp>(user).getResult(0);
  //     break;
  //   }
  // }

  if (!slicedSource)
    return failure();

  rewriter.setInsertionPoint(castOp);
  auto newCastOp = rewriter.create<memref::MemorySpaceCastOp>(
      castOp.getLoc(), slicedCastResultType, slicedSource);
  resolveUpLinksForOldValue(castOp.getResult(), newCastOp.getResult(),
                            rewriter);

  LDBG("Propagated up through memory_space_cast, new cast op is:\n "
       << newCastOp);
  return success();
}

LogicalResult BufferizationPropagateUpPattern::propagateUpAlloc(
    memref::AllocOp allocOp, UnrealizedConversionCastOp propagateOp,
    PatternRewriter &rewriter) const {

  auto propagateResultType =
      dyn_cast<MemRefType>(propagateOp.getResult(0).getType());
  if (!propagateResultType)
    return failure();

  auto maybeNewAlloc =
      createSlicedAllocLike(propagateResultType, allocOp, rewriter);
  if (failed(maybeNewAlloc)) {
    LDBG("propagateUpAlloc createSlicedAllocLike failed for " << allocOp);
    return failure();
  }

  memref::AllocOp newAllocOp = maybeNewAlloc.value();
  markTiledTightlyCoupledAllocIfNeeded(rewriter, allocOp.getResult());

  auto [tilingDim, tiledOffset, tiledSize] = getTilingDimInfo(propagateOp);
  SmallVector<OpOperand *> uses;
  for (auto &use : allocOp->getUses())
    uses.push_back(&use);
  for (auto *use : uses) {
    auto *user = use->getOwner();
    auto newPropagateOp =
        createBubblePropagatorDown(use->get(), newAllocOp.getResult(),
                                   tiledOffset, tiledSize, tilingDim, rewriter);
    rewriter.modifyOpInPlace(user,
                             [&]() { use->set(newPropagateOp->getResult(0)); });
  }
  LDBG("Propagated up to alloc, the new alloc is:\n " << newAllocOp);
  return success();
}

LogicalResult BufferizationPropagateUpPattern::propagateUpSubView(
    memref::SubViewOp subViewOp, UnrealizedConversionCastOp propagateOp,
    PatternRewriter &rewriter) const {
  auto [tilingDim, tiledOffset, tiledSize] = getTilingDimInfo(propagateOp);
  if (tilingDim == -1)
    return failure();

  auto loc = subViewOp.getLoc();
  auto offsets = subViewOp.getMixedOffsets();
  auto sizes = subViewOp.getMixedSizes();
  auto strides = subViewOp.getMixedStrides();

  rewriter.setInsertionPoint(subViewOp);
  handleExtractOfExtract(offsets[tilingDim], sizes[tilingDim], tiledOffset,
                         tiledSize, loc, rewriter);

  auto slicedType = getSlicedMemRefType(subViewOp.getSourceType(), tilingDim);
  auto srcUp =
      createBubblePropagatorUpLink(subViewOp.getSource(), slicedType,
                                   tiledOffset, tiledSize, tilingDim, rewriter);
  auto newOp = rewriter.create<memref::SubViewOp>(loc, srcUp->getResult(0),
                                                  offsets, sizes, strides);
  rewriter.replaceOp(propagateOp, newOp);
  return success();
}

LogicalResult BufferizationPropagateUpPattern::propagateUpReinterpretCast(
    memref::ReinterpretCastOp castOp, UnrealizedConversionCastOp propagateOp,
    PatternRewriter &rewriter) const {

  auto maybeParams =
      getSubViewParamsFromPropagatorResult(propagateOp, rewriter);
  if (failed(maybeParams))
    return failure();
  auto &[offsets, sizes, strides] = *maybeParams;

  rewriter.setInsertionPointAfter(castOp);
  auto newSubViewOp = rewriter.create<memref::SubViewOp>(
      castOp.getLoc(), castOp.getResult(), offsets, sizes, strides);
  resolveUpLinksForOldValue(castOp.getResult(), newSubViewOp.getResult(),
                            rewriter);

  LDBG("Propagated up through reinterpret_cast " << castOp);
  return success();
}

// PropagateUp pattern should be rewritten here.
LogicalResult BufferizationPropagateUpPattern::matchAndRewrite(
    UnrealizedConversionCastOp propagateOp, PatternRewriter &rewriter) const {
  if (!propagateOp->hasAttr(kBubbleUpPropagateUp))
    return failure();

  Value input = propagateOp.getInputs()[0];
  auto *defOp = input.getDefiningOp();
  if (!defOp)
    return failure();

  return TypeSwitch<Operation *, LogicalResult>(defOp)
      .Case([&](UnrealizedConversionCastOp op) {
        if (!op->hasAttr(kBubbleUpPropagateDown))
          return failure();
        if (getTilingDimInfo(propagateOp).tilingDim !=
            getTilingDimInfo(op).tilingDim)
          return failure();
        rewriter.replaceOp(propagateOp, op.getInputs()[0]);
        return success();
      })
      .Case([&](memref::MemorySpaceCastOp op) {
        return propagateUpMemorySpaceCast(op, propagateOp, rewriter);
      })
      .Case([&](memref::SubViewOp op) {
        return propagateUpSubView(op, propagateOp, rewriter);
      })
      .Case([&](memref::AllocOp op) {
        return propagateUpAlloc(op, propagateOp, rewriter);
      })
      .Case([&](memref::ReinterpretCastOp op) {
        return propagateUpReinterpretCast(op, propagateOp, rewriter);
      })
      .Default([&](Operation *) { return failure(); });
}

LogicalResult BufferizationPropagateDownPattern::propagateDownMarkOp(
    annotation::MarkOp markOp, UnrealizedConversionCastOp propagateOp,
    OpOperand &use, PatternRewriter &rewriter) const {

  Value newValue = propagateOp.getInputs()[0];
  rewriter.modifyOpInPlace(markOp, [&]() { use.set(newValue); });
  LDBG("Propagated down through annotation.mark " << markOp);
  return success();
}

LogicalResult BufferizationPropagateDownPattern::propagateDownLoadOp(
    hivm::LoadOp loadOp, UnrealizedConversionCastOp propagateOp,
    PatternRewriter &rewriter) const {
  if (loadOp.getDst() != propagateOp.getResult(0))
    return failure();

  // Subview dst slicing is handled by propagateDownSubView; only handle direct
  // alloc (or similar root) down links here.
  if (!traceDefOp<memref::AllocOp>(propagateOp.getInputs()[0]))
    return failure();

  Value newDst = propagateOp.getInputs()[0];
  auto propagateInputType = dyn_cast<MemRefType>(newDst.getType());
  if (!propagateInputType)
    return failure();

  rewriter.modifyOpInPlace(loadOp,
                           [&]() { loadOp.getDstMutable().set(newDst); });

  auto slicedTensorType = RankedTensorType::get(
      propagateInputType.getShape(), propagateInputType.getElementType());

  Value loadSrc = loadOp.getSrc();
  auto oldSrcType = dyn_cast<MemRefType>(loadSrc.getType());
  auto slicedSrcType = getSlicedMemRefType(oldSrcType, slicedTensorType);

  UnrealizedConversionCastOp srcUp;
  if (auto *defOp = loadSrc.getDefiningOp();
      defOp && defOp->hasAttr(kBubbleUpPropagateUp)) {
    srcUp = cast<UnrealizedConversionCastOp>(defOp);
  } else {
    auto tilingDimInfo = getTilingDimInfo(propagateOp);
    srcUp = createBubblePropagatorUpLink(
        loadSrc, slicedSrcType, tilingDimInfo.offset, tilingDimInfo.size,
        tilingDimInfo.tilingDim, rewriter);
  }
  rewriter.modifyOpInPlace(
      loadOp, [&]() { loadOp.getSrcMutable().set(srcUp.getResult(0)); });

  if (propagateOp->use_empty())
    clearBubblePropagatorAttrs(propagateOp, rewriter);
  LDBG("Propagated down through hivm.load " << loadOp);
  return success();
}

LogicalResult BufferizationPropagateDownPattern::propagateDownSubView(
    memref::SubViewOp subViewOp, UnrealizedConversionCastOp propagateOp,
    PatternRewriter &rewriter) const {
  if (subViewOp.getSource() != propagateOp.getResult(0))
    return failure();

  auto oldSourceType = dyn_cast<MemRefType>(propagateOp.getResult(0).getType());
  // Sliced memref on the propagate-down input side (e.g. tiled alloc).
  Value slicedSource = propagateOp.getInputs()[0];
  auto slicedSourceType = dyn_cast<MemRefType>(slicedSource.getType());
  if (!oldSourceType || !slicedSourceType)
    return failure();
  if (oldSourceType.getRank() != slicedSourceType.getRank())
    return failure();

  auto resultType = dyn_cast<MemRefType>(subViewOp.getResult().getType());
  if (!resultType || resultType.hasStaticShape())
    return failure();

  auto [tilingDim, tiledOffset, tiledSize] = getTilingDimInfo(propagateOp);
  auto newTilingDim = tilingDim;
  if (tilingDim == -1)
    return failure();
  auto newOffsets = subViewOp.getMixedOffsets();
  auto newSizes = subViewOp.getMixedSizes();
  auto newStrides = subViewOp.getMixedStrides();

  auto droppedDims = subViewOp.getDroppedDims();

  for (int64_t i = 0; i < tilingDim; i++) {
    if (droppedDims[i])
      newTilingDim--;
  }

  rewriter.setInsertionPoint(subViewOp);
  mlir::hivm::handleExtractOfExtract(newOffsets[tilingDim], newSizes[tilingDim],
                                     tiledOffset, tiledSize, subViewOp.getLoc(),
                                     rewriter);

  auto fullShape =
      cast<MemRefType>(memref::SubViewOp::inferResultType(
                           slicedSourceType, newOffsets, newSizes, newStrides))
          .getShape();

  SmallVector<int64_t> reducedShape;
  for (size_t i = 0; i < fullShape.size(); i++) {
    if (!droppedDims[i])
      reducedShape.push_back(fullShape[i]);
  }

  auto newSubViewType = memref::SubViewOp::inferRankReducedResultType(
      reducedShape, slicedSourceType, newOffsets, newSizes, newStrides);

  auto newOp = rewriter.create<memref::SubViewOp>(
      subViewOp.getLoc(), cast<MemRefType>(newSubViewType), slicedSource,
      newOffsets, newSizes, newStrides);
  for (auto attr : subViewOp->getAttrs()) {
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());
  }
  SmallVector<OpOperand *> uses;
  for (auto &use : subViewOp->getUses())
    uses.push_back(&use);
  for (auto *use : uses) {
    auto *user = use->getOwner();
    auto newPropagateOp = createBubblePropagatorDown(
        use->get(), newOp, tiledOffset, tiledSize, newTilingDim, rewriter);
    rewriter.modifyOpInPlace(user,
                             [&]() { use->set(newPropagateOp->getResult(0)); });
  }
  rewriter.eraseOp(subViewOp);
  rewriter.eraseOp(propagateOp);
  LDBG("Propagated down through dynamic subview " << newOp);
  return success();
}

// PropagateDown pattern should be written here.
LogicalResult BufferizationPropagateDownPattern::matchAndRewrite(
    UnrealizedConversionCastOp propagateOp, PatternRewriter &rewriter) const {
  if (!propagateOp->hasAttr(kBubbleUpPropagateDown))
    return failure();
  if (propagateOp.use_empty())
    return failure();
  assert(propagateOp.getResult(0).hasOneUse());

  OpOperand &use = *propagateOp.getResult(0).use_begin();
  Operation *userOp = use.getOwner();
  if (userOp->hasAttr(kBubbleUpPropagateUp) ||
      userOp->hasAttr(kBubbleUpPropagateDown))
    return failure();

  return TypeSwitch<Operation *, LogicalResult>(userOp)
      .Case([&](annotation::MarkOp op) {
        return propagateDownMarkOp(op, propagateOp, use, rewriter);
      })
      .Case([&](hivm::LoadOp op) {
        return propagateDownLoadOp(op, propagateOp, rewriter);
      })
      .Case([&](memref::SubViewOp op) {
        return propagateDownSubView(op, propagateOp, rewriter);
      })
      // .Case([&](memref::MemorySpaceCastOp castOp) {
      //   if (mappedValue == castOp.getResult())
      //     return failure();
      //   auto oldResultType =
      //   dyn_cast<MemRefType>(castOp.getResult().getType()); auto
      //   newSourceType = dyn_cast<MemRefType>(mappedValue.getType()); if
      //   (!oldResultType || !newSourceType)
      //     return failure();
      //   auto newResultType = MemRefType::get(
      //       newSourceType.getShape(), newSourceType.getElementType(),
      //       oldResultType.getLayout(), oldResultType.getMemorySpace());
      //   rewriter.setInsertionPoint(castOp);
      //   auto newCastOp = rewriter.create<memref::MemorySpaceCastOp>(
      //       castOp.getLoc(), newResultType, mappedValue);
      //   rewriter.replaceOpUsesWithIf(castOp, newCastOp.getResult(),
      //                                [&](OpOperand &opr) {
      //                                  return opr.getOwner() !=
      //                                  newCastOp;
      //                                });
      //   if (castOp->use_empty())
      //     rewriter.eraseOp(castOp);
      //   return success();
      // })
      // .Case([&](bufferization::ToTensorOp otherToTensorOp) {
      //   if (use->get() != propagateOp.getResult(0))
      //     return failure();
      //   rewriter.modifyOpInPlace(otherToTensorOp, [&]() {
      //     otherToTensorOp.getMemrefMutable().set(mappedValue);
      //   });
      //   return success();
      // })
      .Default([&](Operation *) { return failure(); });
}

} // namespace mlir::hivm::detail
