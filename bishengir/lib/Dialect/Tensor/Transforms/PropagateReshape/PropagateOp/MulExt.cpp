//===- MulExt.cpp - hfusion::MulExt propagate implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagatableOp.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"

#define DEBUG_TYPE "propagate-mul-ext-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir::tensor {
using namespace mlir::hfusion;
using namespace mlir::tensor::reshape_utils;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;

LogicalResult
PropagatableMulExt::matchAndRewriteExpand(PatternRewriter &rewriter,
                                          Operation *op,
                                          tensor::ExpandShapeOp expandOp) {
  rewriter.setInsertionPointAfter(op);
  auto loc = expandOp.getLoc();
  SmallVector<Value, 4> newOperands;
  auto sourceRank = utils::getShapeRank(expandOp.getSrc());
  for (Value operand : op->getOperands()) {
    rewriter.setInsertionPointAfterValue(operand);
    auto shapeRank = utils::getShapeRank(operand);
    if (shapeRank.has_value() && shapeRank.value() == sourceRank.value()) {
      Operation *newReshapeOp =
          createNewExpandOpFromExpandOp(expandOp, rewriter, loc, operand);
      LLVM_DEBUG(llvm::dbgs() << "Created " << *newReshapeOp << "\n";);
      newOperands.push_back(newReshapeOp->getResult(0));
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Can't collapse inequal rank " << shapeRank
          << " : " << sourceRank << "\n";);
      newOperands.push_back(operand);
    }
  }
  auto oldShape = expandOp.getSrc().getType();
  auto expandResultNumber = cast<OpResult>(expandOp.getSrc()).getResultNumber();
  auto reassociation = expandOp.getReassociation();
  auto newMulExtOp = rewriter.create<hfusion::MulExtOp>(
      op->getLoc(), newOperands);
  for (size_t i = 0; i < op->getNumResults(); ++i) {
    auto collapsedResult = rewriter.create<tensor::CollapseShapeOp>(
        op->getLoc(), oldShape,
        newMulExtOp->getResult(i), reassociation);
    rewriter.replaceAllUsesWith(op->getResult(i), collapsedResult);
  }
  rewriter.replaceOp(expandOp, newMulExtOp.getResult(expandResultNumber));
  rewriter.eraseOp(op);
  return success();
}

LogicalResult
PropagatableMulExt::matchAndRewriteCollapse(PatternRewriter &rewriter,
                                            Operation *op,
                                            tensor::CollapseShapeOp
                                            collapseOp) {
  rewriter.setInsertionPointAfter(op);
  auto loc = collapseOp.getLoc();

  auto resultRank = utils::getShapeRank(collapseOp.getResult()).value_or(0);
  SmallVector<Value> newOperands;

  for (Value operand : op->getOperands()) {
    rewriter.setInsertionPointAfterValue(operand);
    auto shapeRank = utils::getShapeRank(operand);
    // Check in case its scalar elemwise
    if (shapeRank.has_value() && shapeRank.value() == resultRank) {
      Operation *newExpandedOperand =
          createNewExpandOpFromCollapseOp(collapseOp, rewriter, loc, operand);
      LLVM_DEBUG(llvm::dbgs() << "Created " << *newExpandedOperand << "\n";);
      newOperands.push_back(newExpandedOperand->getResult(0));
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Can't expand inequal rank " << shapeRank
          << " : " << resultRank << "\n";);
      newOperands.push_back(operand);
    }
  }
  auto newMulExtOp = rewriter.create<hfusion::MulExtOp>(
      op->getLoc(), newOperands);
  auto oldShape = collapseOp.getResult().getType();
  auto reassociation = collapseOp.getReassociation();
  for (size_t i = 0; i < op->getNumResults(); ++i) {
    auto collapsedResult = rewriter.create<tensor::CollapseShapeOp>(
        op->getLoc(), oldShape,
        newMulExtOp->getResult(i), reassociation);
    rewriter.replaceAllUsesWith(op->getResult(i), collapsedResult);
  }
  rewriter.eraseOp(op);
  return success();
}

} // namespace mlir::tensor