//===- SwapCollapseExpand.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Swap collapse and expand order so collapse can be put down and expand can be
// put up
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/SwapCollapseExpand.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "propagate-reshape"
namespace mlir {
namespace tensor {
using namespace mlir::hfusion;
using namespace mlir::tensor::reshape_utils;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;

// %b = collapse %a
// <AxBxCxDxExFxG> -> <AxBCDxExFG>
// [[0], [1, 2, 3], [4], [5, 6]]
// %c = expand %b
// <AxBCDxExFG> -> <AxBCDxE1xE2xFG>
// [[0], [1], [2, 3], [4]]
//
// |
// v
//
// %tmp = expand %a
// <AxBxCxDxExFxG> -> <AxBxCxDxE1xE2xFxG>
// [[0], [1], [2], [3], [4, 5], [6], [7]]
// %c = collapse %tmp
// <AxBxCxDxE1xE2xFxG> -> <AxBCDxE1xE2xFG>
// [[0], [1, 2, 3], [4], [5], [6, 7]]
namespace {

bool areReassociationsCompatible(
    ArrayRef<ReassociationIndices> collapseReassoc,
    ArrayRef<ReassociationIndices> expandReassoc,
    SmallVector<ReassociationIndices> &supposedExpand,
    SmallVector<ReassociationIndices> &supposedCollapse,
    ArrayRef<int64_t> collapseSourceShape, ArrayRef<int64_t> expandShapeResult,
    SmallVector<int64_t> &newExpandShape) {
  // Check if collapse and expand reassociations are inverses of each other
  if (collapseReassoc.size() != expandReassoc.size())
    return false;
  for (size_t i = 0; i < collapseReassoc.size(); ++i) {
    bool isCollapsing = collapseReassoc[i].size() > 1;
    bool isExpanding = expandReassoc[i].size() > 1;
    if (isCollapsing && isExpanding) {
      return false;
    }
    if (isExpanding) {
      for (auto el : expandReassoc[i]) {
        assert(el >= 0 && static_cast<size_t>(el) < expandShapeResult.size());
        newExpandShape.push_back(expandShapeResult[el]);
        supposedCollapse.push_back({-1});
      }
      supposedExpand.push_back(expandReassoc[i]);
    } else {
      for (auto el : collapseReassoc[i]) {
        newExpandShape.push_back(collapseSourceShape[el]);
        supposedExpand.push_back({-1});
      }
      supposedCollapse.push_back(collapseReassoc[i]);
    }
  }
  return true;
}
} // namespace

LogicalResult
SwapCollapseExpand::matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                    PatternRewriter &rewriter) const {
  auto collapseOp = expandOp.getSrc().getDefiningOp<tensor::CollapseShapeOp>();
  if (!collapseOp)
    return failure();
  auto *definedCollapse = collapseOp.getSrc().getDefiningOp();
  if (!definedCollapse || isStopPropagatable(definedCollapse))
    return failure();
  if (llvm::all_of(expandOp->getUsers(),
                   [&](Operation *op) { return isOutOp(op); })) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Trying to swap collapse expand here\n";);
  auto collapseReassoc = collapseOp.getReassociationIndices();
  auto expandReassoc = expandOp.getReassociationIndices();
  SmallVector<ReassociationIndices> newReassociationExpand;
  SmallVector<ReassociationIndices> newReassociationCollapse;
  auto collapseSourceShape = utils::getShape(collapseOp.getSrc().getType());
  auto expandShapeResult = utils::getShape(expandOp.getResult().getType());
  SmallVector<int64_t> newExpandShape;
  bool reassociationsDone = false;
  if (!areReassociationsCompatible(
          collapseReassoc, expandReassoc, newReassociationExpand,
          newReassociationCollapse, collapseSourceShape, expandShapeResult,
          newExpandShape)) {
    newExpandShape.clear();
    newReassociationExpand.clear();
    newReassociationCollapse.clear();
    LLVM_DEBUG(llvm::dbgs() << "Fixed reassociations fail\n";);
  } else
    reassociationsDone = true;

  if (!reassociationsDone &&
      !areLooseReassociationsCompatible(
          newReassociationExpand, newReassociationCollapse, collapseSourceShape,
          expandShapeResult, newExpandShape)) {
    LLVM_DEBUG(llvm::dbgs() << "Loose reassociations fail\n";);
    return failure();
  }

  renumberReassociation(newReassociationExpand);
  renumberReassociation(newReassociationCollapse);
  rewriter.setInsertionPointAfter(expandOp);
  auto newExpandType =
      RankedTensorType::get(newExpandShape, getElementTypeOrSelf(expandOp));
  auto newExpandOp = rewriter.create<tensor::ExpandShapeOp>(
      collapseOp.getLoc(), newExpandType, collapseOp.getSrc(),
      newReassociationExpand);
  auto newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
      expandOp.getLoc(), expandOp.getResult().getType(),
      newExpandOp.getResult(), newReassociationCollapse);
  rewriter.replaceAllUsesWith(expandOp, newCollapseOp.getResult());
  return success();
}
} // namespace tensor
} // namespace mlir
