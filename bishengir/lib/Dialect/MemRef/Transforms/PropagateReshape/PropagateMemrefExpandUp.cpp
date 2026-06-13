//===- PropagateMemrefExpandUp.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Propagate expand up will try to bubble up the expandshape operation to the
//  top
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "bishengir/Dialect/MemRef/Transforms/PropagateReshape.h"

#define DEBUG_TYPE "propagate-memref-reshape"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
using namespace mlir::utils::debugger;

namespace mlir {
namespace memref {
using namespace mlir::utils::debugger;

namespace {

LogicalResult handleAllocOp(memref::ExpandShapeOp expandOp,
                            PatternRewriter &rewriter, Operation *definingOp) {
  rewriter.setInsertionPointAfter(definingOp);
  SmallVector<Value, 4> newOperands;
  auto allocOp = cast<memref::AllocOp>(definingOp);
  auto reassociation = expandOp.getReassociation();
  auto collapsedRes = rewriter.create<memref::CollapseShapeOp>(
      expandOp->getLoc(), definingOp->getResults()[0].getType(),
      definingOp->getResults()[0], reassociation);
  rewriter.replaceAllUsesExcept(definingOp->getResults()[0],
                                collapsedRes.getResult(), collapsedRes);
  rewriter.modifyOpInPlace(definingOp, [&]() {
    definingOp->getResult(0).setType(expandOp.getResultType());
  });
  rewriter.replaceAllUsesWith(expandOp.getResult(), allocOp);
  return success();
}

static OpFoldResult multiplyOFR(PatternRewriter &rewriter, Location loc,
                                OpFoldResult a, OpFoldResult b) {
  auto aConstant = getConstantIntValue(a);
  auto bConstant = getConstantIntValue(b);

  if (aConstant && bConstant)
    return rewriter.getIndexAttr(*aConstant * *bConstant);

  Value aVal = getValueOrCreateConstantIndexOp(rewriter, loc, a);
  Value bVal = getValueOrCreateConstantIndexOp(rewriter, loc, b);
  return rewriter.create<arith::MulIOp>(loc, aVal, bVal).getResult();
}

LogicalResult handleReinterpretCast(memref::ExpandShapeOp expandOp,
                                    PatternRewriter &rewriter,
                                    Operation *definingOp) {
  auto reinterpretCast = cast<memref::ReinterpretCastOp>(definingOp);
  auto expandResType = mlir::cast<MemRefType>(expandOp.getResult().getType());

  auto reassociation = expandOp.getReassociationIndices();

  SmallVector<OpFoldResult> offsetOfr = reinterpretCast.getMixedOffsets();
  SmallVector<OpFoldResult> oldStrides = reinterpretCast.getMixedStrides();
  SmallVector<OpFoldResult> sizesOfr = getMixedValues(
      expandOp.getStaticOutputShape(), expandOp.getOutputShape(), rewriter);

  SmallVector<OpFoldResult> newStridesOfr;

  rewriter.setInsertionPoint(reinterpretCast);
  for (auto [idx, group] : llvm::enumerate(reassociation)) {
    OpFoldResult currentStride = oldStrides[idx];
    SmallVector<OpFoldResult> groupStrides;
    for (int i = (int)group.size() - 1; i >= 0; --i) {
      groupStrides.push_back(currentStride);
      if (i > 0) {
        currentStride = multiplyOFR(rewriter, reinterpretCast.getLoc(),
                                    currentStride, sizesOfr[group[i]]);
      }
    }
    std::reverse(groupStrides.begin(), groupStrides.end());
    newStridesOfr.append(groupStrides.begin(), groupStrides.end());
  }
  expandOp->moveAfter(reinterpretCast);
  rewriter.setInsertionPointAfterValue(expandOp);
  auto newReinterpret = rewriter.create<memref::ReinterpretCastOp>(
      reinterpretCast->getLoc(), expandResType, reinterpretCast.getSource(),
      offsetOfr, sizesOfr, newStridesOfr);

  auto origResultType = reinterpretCast.getResult().getType();
  auto collapsedType = memref::CollapseShapeOp::computeCollapsedType(
      cast<MemRefType>(newReinterpret.getType()), reassociation);

  auto newCollapse = rewriter.create<memref::CollapseShapeOp>(
      reinterpretCast->getLoc(), collapsedType, newReinterpret, reassociation);

  if (collapsedType == origResultType) {
    rewriter.replaceAllUsesExcept(reinterpretCast, newCollapse, expandOp);
  } else {
    auto oldSizes = reinterpretCast.getMixedSizes();
    auto castOp = rewriter.create<memref::ReinterpretCastOp>(
        reinterpretCast->getLoc(), origResultType, newCollapse.getResult(),
        offsetOfr, oldSizes, oldStrides);
    rewriter.replaceAllUsesExcept(reinterpretCast, castOp, expandOp);
  }

  rewriter.replaceOp(expandOp, newReinterpret);
  LDBG(
      (definingOp->getParentOp() ? *(definingOp->getParentOp()) : *definingOp));
  rewriter.eraseOp(reinterpretCast);
  return success();
}

LogicalResult handleSubView(memref::ExpandShapeOp expandOp,
                            PatternRewriter &rewriter, Operation *definingOp) {
  auto reassociation = expandOp.getReassociationIndices();
  auto subviewOp = dyn_cast<memref::SubViewOp>(definingOp);
  SmallVector<OpFoldResult> newMixedOffsets;
  SmallVector<OpFoldResult> newMixedSizes;
  SmallVector<OpFoldResult> newMixedStrides;

  SmallVector<OpFoldResult> expandOutputShape;
  auto res = tensor::reshape_utils::getSubviewModifyingOp(
      rewriter, subviewOp, reassociation,
      tensor::reshape_utils::getMixedSizesOrOutputShape(rewriter,
                                                        expandOp.getResult()),
      /* subview */ true, newMixedOffsets, newMixedSizes, newMixedStrides,
      expandOutputShape);
  if (res.failed())
    return failure();
  auto staticOutputShape = decomposeMixedValues(expandOutputShape);
  auto loc = definingOp->getLoc();
  auto srcType = memref::ExpandShapeOp::computeExpandedType(
      subviewOp.getSourceType(), staticOutputShape.first, reassociation);
  if (failed(srcType))
    return failure();
  auto expandedNewSrc = rewriter.create<memref::ExpandShapeOp>(
      loc, srcType.value(), subviewOp.getSource(), reassociation,
      expandOutputShape);
  auto newSubviewOp = rewriter.create<memref::SubViewOp>(
      loc, expandedNewSrc.getResult(), newMixedOffsets, newMixedSizes,
      newMixedStrides);
  auto newCollapse = rewriter.create<memref::CollapseShapeOp>(
      newSubviewOp.getLoc(), subviewOp.getResult().getType(),
      newSubviewOp.getResult(), reassociation);
  rewriter.replaceAllUsesExcept(subviewOp, newCollapse, expandOp);
  Value subviewOpReplacement = newSubviewOp.getResult();
  auto targetType = cast<MemRefType>(expandOp.getResult().getType());
  
  rewriter.setInsertionPointAfterValue(expandOp);
  if (newSubviewOp.getResult().getType() != targetType) {
    if (memref::CastOp::areCastCompatible(
            cast<MemRefType>(newSubviewOp.getResult().getType()), targetType)) {
      subviewOpReplacement = rewriter.create<memref::CastOp>(
          loc, targetType, newSubviewOp.getResult());
      rewriter.replaceOp(expandOp, subviewOpReplacement);
    } else {
      auto stridedMetadata = rewriter.create<memref::ExtractStridedMetadataOp>(
          loc, expandOp.getResult());
      subviewOpReplacement =
          rewriter
              .create<memref::ReinterpretCastOp>(
                  loc, targetType, newSubviewOp.getResult(),
                  getAsOpFoldResult(stridedMetadata.getOffset()),
                  getAsOpFoldResult(stridedMetadata.getSizes()),
                  getAsOpFoldResult(stridedMetadata.getStrides()))
              .getResult();
      rewriter.replaceAllUsesExcept(expandOp, subviewOpReplacement, stridedMetadata);
    }
  } else {
    rewriter.replaceOp(expandOp, newSubviewOp.getResult());
  }
  LDBG(
      (definingOp->getParentOp() ? *(definingOp->getParentOp()) : *definingOp));
  return success();
}

} // namespace

LogicalResult
PropagateMemrefExpandUp::matchAndRewrite(memref::ExpandShapeOp expandOp,
                                         PatternRewriter &rewriter) const {
  Value source = expandOp.getSrc();
  Operation *definingOp = source.getDefiningOp();
  if (!definingOp)
    return failure();
  if (definingOp->getParentOp() != expandOp->getParentOp())
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "-- Found definingOp: " << *definingOp << "\n";);
  LLVM_DEBUG(llvm::dbgs() << "Ok rewriting\n";);
  LLVM_DEBUG(llvm::dbgs() << (definingOp->getParentOp()
                                  ? *(definingOp->getParentOp())
                                  : *definingOp)
                          << "\n";);
  if (isa<memref::AllocOp>(definingOp)) {
    LDBG("[AllocOp] Ok in here");
    return handleAllocOp(expandOp, rewriter, definingOp);
  }
  if (isa<memref::ReinterpretCastOp>(definingOp)) {
    LDBG("[Reinterpret cast] Ok in here");
    return handleReinterpretCast(expandOp, rewriter, definingOp);
  }
  if (isa<memref::SubViewOp>(definingOp)) {
    LDBG("[SubView] Ok in here");
    return handleSubView(expandOp, rewriter, definingOp);
  }
  return failure();
}
} // namespace memref
} // namespace mlir