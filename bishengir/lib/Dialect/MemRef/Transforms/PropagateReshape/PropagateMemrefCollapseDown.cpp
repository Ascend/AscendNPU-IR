//===- PropagateMemrefCollapseDown.cpp ------------------------------------===//
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
using namespace mlir::utils::debugger;


namespace mlir::memref {

namespace {

LogicalResult handleCopyOp(memref::CollapseShapeOp collapseOp,
                           PatternRewriter &rewriter, Operation *userOp) {
  auto resultRank = collapseOp.getResult().getType().getRank();
  SmallVector<Value> newOperands = getNewOperands(
      collapseOp, rewriter, userOp, resultRank);
  rewriter.modifyOpInPlace(userOp, [&]() { userOp->setOperands(newOperands); });
  return success();
}

LogicalResult handleSubviewOp(memref::CollapseShapeOp collapseOp,
                           PatternRewriter &rewriter, Operation *userOp) {

  auto reassociation = collapseOp.getReassociationIndices();
  auto subviewOp = dyn_cast<memref::SubViewOp>(userOp);
  SmallVector<OpFoldResult> newMixedOffsets;
  SmallVector<OpFoldResult> newMixedSizes;
  SmallVector<OpFoldResult> newMixedStrides;
  SmallVector<OpFoldResult> dummyExpand;
  auto res = tensor::reshape_utils::getSubviewModifyingOp(
      rewriter, subviewOp, reassociation,
      tensor::reshape_utils::getMixedSizesOrOutputShape(rewriter, collapseOp.getSrc()),
      /*superview*/ false, newMixedOffsets, newMixedSizes, newMixedStrides,
      dummyExpand);
  if (res.failed())
    return failure();
  auto loc = userOp->getLoc();
  auto newSubviewOp = rewriter.create<memref::SubViewOp>(
      loc, collapseOp.getSrc(), newMixedOffsets, newMixedSizes,
      newMixedStrides);
  auto subviewOpResultShape =
      utils::getShape(subviewOp.getResult().getType());

  auto newCollapse = rewriter.create<memref::CollapseShapeOp>(
      newSubviewOp.getLoc(), subviewOp.getResult().getType(),
      newSubviewOp.getResult(), reassociation);
  rewriter.replaceAllUsesWith(subviewOp, newCollapse);
  rewriter.eraseOp(subviewOp);
  return success();
}

LogicalResult handleFillOp(memref::CollapseShapeOp collapseOp,
                           PatternRewriter &rewriter, Operation *userOp) {
  auto fillOp = cast<linalg::FillOp>(userOp);
  auto resultRank = cast<ShapedType>(
      fillOp.getDpsInitOperand(0)->get().getType()).getRank();
  SmallVector<Value> newOperands = getNewOperands(
      collapseOp, rewriter, userOp, resultRank);
  rewriter.modifyOpInPlace(userOp, [&]() { userOp->setOperands(newOperands); });
  return success();
}

LogicalResult handleMarkOp(memref::CollapseShapeOp collapseOp,
                           PatternRewriter &rewriter, Operation *userOp) {
  auto markOp = cast<annotation::MarkOp>(userOp);
  auto resultRank = cast<ShapedType>(markOp.getSrc().getType()).getRank();
  SmallVector<Value> newOperands = getNewOperands(
      collapseOp, rewriter, userOp, resultRank);
  rewriter.modifyOpInPlace(userOp, [&]() {
    userOp->setOperands(newOperands);
  });
  return success();
}


} // namespace

LogicalResult
PropagateMemrefCollapseDown::matchAndRewrite(memref::CollapseShapeOp collapseOp,
                                             PatternRewriter &rewriter) const {
  Value result = collapseOp.getResult();
  auto userRange = result.getUsers();
  SmallVector<Operation *> users(userRange.begin(), userRange.end());
  // Propagate one by one, to be safe
  auto *src = collapseOp.getSrc().getDefiningOp();
  if (!src)
    return failure();

  for (Operation *userOp : users) {
    if (isa<memref::CopyOp>(userOp)) {
      return handleCopyOp(collapseOp, rewriter, userOp);
    }
    if (isa<memref::SubViewOp>(userOp)) {
      return handleSubviewOp(collapseOp, rewriter, userOp);
    }
    if (isa<linalg::FillOp>(userOp)) {
      return handleFillOp(collapseOp, rewriter, userOp);
    }
    if (isa<annotation::MarkOp>(userOp)) {
      return handleMarkOp(collapseOp, rewriter, userOp);
    }
  }
  return failure();
}
} // namespace mlir::memref
