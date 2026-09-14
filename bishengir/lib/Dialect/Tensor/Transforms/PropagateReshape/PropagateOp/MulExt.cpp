//===- MulExt.cpp - hfusion::MulExt reshape propagation -------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025~2026. All rights reserved.
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

#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagatableOp.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"

namespace mlir::tensor {

namespace {

template <typename ReshapeOp, typename CreateReshape>
SmallVector<Value> reshapeOperands(PatternRewriter &rewriter, Operation *op,
                                   ReshapeOp reshapeOp, int64_t expectedRank,
                                   CreateReshape createReshape) {
  SmallVector<Value> operands;
  operands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    auto rank = utils::getShapeRank(operand);
    if (!rank || static_cast<int64_t>(*rank) != expectedRank) {
      operands.push_back(operand);
      continue;
    }

    rewriter.setInsertionPointAfterValue(operand);
    Operation *newReshape = createReshape(operand);
    operands.push_back(newReshape->getResult(0));
  }
  return operands;
}

/// Build high-rank operands for collapse propagation. Prefer the collapse
/// source when the operand is the collapse being propagated, so the collapse
/// can be erased when it has no other users.
SmallVector<Value> expandCollapseOperands(PatternRewriter &rewriter,
                                          Operation *op,
                                          tensor::CollapseShapeOp collapseOp,
                                          int64_t resultRank) {
  SmallVector<Value> operands;
  operands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    if (operand == collapseOp.getResult()) {
      operands.push_back(collapseOp.getSrc());
      continue;
    }
    auto rank = utils::getShapeRank(operand);
    if (!rank || static_cast<int64_t>(*rank) != resultRank) {
      operands.push_back(operand);
      continue;
    }
    rewriter.setInsertionPointAfterValue(operand);
    Operation *expanded = createNewExpandOpFromCollapseOp(
        collapseOp, rewriter, collapseOp.getLoc(), operand);
    operands.push_back(expanded->getResult(0));
  }
  return operands;
}

template <typename MulExtOpTy>
void collapseResults(PatternRewriter &rewriter, Operation *oldOp,
                     MulExtOpTy newOp, Type collapsedType,
                     ArrayRef<ReassociationIndices> reassociation) {
  for (auto [oldResult, newResult] :
       llvm::zip(oldOp->getResults(), newOp->getResults())) {
    auto collapsed = rewriter.create<tensor::CollapseShapeOp>(
        oldOp->getLoc(), collapsedType, newResult, reassociation);
    rewriter.replaceAllUsesWith(oldResult, collapsed);
  }
}

} // namespace

LogicalResult PropagatableMulExt::matchAndRewriteExpand(
    PatternRewriter &rewriter, Operation *op, tensor::ExpandShapeOp expandOp) {
  const auto sourceRank = utils::getShapeRank(expandOp.getSrc());
  if (!sourceRank)
    return failure();
  auto sourceResult = dyn_cast<OpResult>(expandOp.getSrc());
  if (!sourceResult)
    return failure();

  SmallVector<Value> operands =
      reshapeOperands(rewriter, op, expandOp, *sourceRank, [&](Value operand) {
        return createNewExpandOpFromExpandOp(expandOp, rewriter,
                                             expandOp.getLoc(), operand);
      });
  rewriter.setInsertionPointAfter(op);

  if (isa<hfusion::MulExtOp>(op)) {
    auto newMulExt = rewriter.create<hfusion::MulExtOp>(op->getLoc(), operands);
    collapseResults(rewriter, op, newMulExt, expandOp.getSrc().getType(),
                    expandOp.getReassociationIndices());
    rewriter.replaceOp(expandOp,
                       newMulExt.getResult(sourceResult.getResultNumber()));
  } else if (isa<hfusion::MulExtUiOp>(op)) {
    auto newMulExt =
        rewriter.create<hfusion::MulExtUiOp>(op->getLoc(), operands);
    collapseResults(rewriter, op, newMulExt, expandOp.getSrc().getType(),
                    expandOp.getReassociationIndices());
    rewriter.replaceOp(expandOp,
                       newMulExt.getResult(sourceResult.getResultNumber()));
  } else {
    return failure();
  }
  rewriter.eraseOp(op);
  return success();
}

LogicalResult PropagatableMulExt::matchAndRewriteCollapse(
    PatternRewriter &rewriter, Operation *op,
    tensor::CollapseShapeOp collapseOp) {
  const auto resultRank = utils::getShapeRank(collapseOp.getResult());
  if (!resultRank)
    return failure();

  SmallVector<Value> operands =
      expandCollapseOperands(rewriter, op, collapseOp, *resultRank);
  rewriter.setInsertionPointAfter(op);

  if (isa<hfusion::MulExtOp>(op)) {
    auto newMulExt = rewriter.create<hfusion::MulExtOp>(op->getLoc(), operands);
    collapseResults(rewriter, op, newMulExt, collapseOp.getResult().getType(),
                    collapseOp.getReassociationIndices());
  } else if (isa<hfusion::MulExtUiOp>(op)) {
    auto newMulExt =
        rewriter.create<hfusion::MulExtUiOp>(op->getLoc(), operands);
    collapseResults(rewriter, op, newMulExt, collapseOp.getResult().getType(),
                    collapseOp.getReassociationIndices());
  } else {
    return failure();
  }
  rewriter.eraseOp(op);
  // Other handlers call collapseAndReplace on results; for MulExt the
  // collapsed results already replace old uses. Erase the original collapse
  // when this user was its last consumer.
  if (collapseOp.getResult().use_empty())
    rewriter.eraseOp(collapseOp);
  return success();
}

} // namespace mlir::tensor
