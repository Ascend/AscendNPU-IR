//===- PropagateNearExpandDown.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Propagate expand down will try to bring the expand shape operation to the
// end
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagateNearEndExpandDown.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"

#define DEBUG_TYPE "propagate-reshape"
namespace mlir {
namespace tensor {
using namespace mlir::hfusion;
using namespace mlir::tensor::reshape_utils;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;

namespace {

// Given the old expand op, this will create a new expand op based on the
// shape of the final result
Operation *createNewCollapseOpFromExpandOp(tensor::ExpandShapeOp expandOp,
                                           PatternRewriter &rewriter,
                                           Location loc, Value operand) {
  auto reassociation = expandOp.getReassociationIndices();
  auto currentShape = utils::getShape(expandOp.getSrc().getType());
  auto resultType =
      RankedTensorType::get(currentShape, getElementTypeOrSelf(operand));
  return rewriter.create<tensor::CollapseShapeOp>(loc, resultType, operand,
                                                  reassociation);
}

LogicalResult handleElementwiseOp(tensor::ExpandShapeOp expandOp,
                                  PatternRewriter &rewriter,
                                  Operation *userOp) {
  rewriter.setInsertionPointAfter(userOp);

  LLVM_DEBUG(llvm::dbgs() << "Trying " << expandOp << " to an elemwise "
                          << *userOp << "\n";);

  auto resultRank = utils::getShapeRank(expandOp.getResult()).value_or(0);
  SmallVector<Value> newOperands;
  auto dpsOp = cast<DestinationStyleOpInterface>(userOp);
  auto oldDpsInits = dpsOp.getDpsInits();
  auto loc = userOp->getLoc();
  for (auto &opOperand : dpsOp->getOpOperands()) {
    auto operand = opOperand.get();
    rewriter.setInsertionPointAfterValue(operand);
    auto shapeRank = utils::getShapeRank(operand).value_or(0);
    bool isProperRank = shapeRank == resultRank;
    // Check in case its scalar elemwise
    if (isProperRank && dpsOp.isDpsInput(&opOperand)) {
      newOperands.push_back(expandOp.getSrc());
    } else if (isProperRank && dpsOp.isDpsInit(&opOperand)) {
      Operation *newReshapeOp =
          createNewCollapseOpFromExpandOp(expandOp, rewriter, loc, operand);
      newOperands.push_back(newReshapeOp->getResult(0));
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Can't expand inequal rank " << shapeRank
                              << " : " << resultRank << "\n";);
      newOperands.push_back(operand);
    }
  }
  updateDefiningOp(userOp, rewriter, newOperands);
  // Expand the results and the inits

  auto resultReassociation = expandOp.getReassociationIndices();
  auto resultShape = utils::getShape(expandOp.getResult().getType());
  for (auto res : userOp->getResults()) {
    expandAndReplace(rewriter, resultReassociation, resultShape, res, userOp);
  }
  for (auto oldInit : oldDpsInits) {
    expandAndReplace(rewriter, resultReassociation, resultShape, oldInit,
                     userOp);
  }
  return success();
}

bool isEndingOrSelfUsers(Operation *op, Operation *self) {
  if (op == self)
    return true;
  if (isReturnOp(op))
    return true;
  LLVM_DEBUG(llvm::dbgs() << "What " << *op << "\n";);
  return false;
}

bool areValidUsers(Value res, Operation *userOp) {
  return std::all_of(res.getUsers().begin(), res.getUsers().end(),
                     [userOp](Operation *userInit) {
                       return isEndingOrSelfUsers(userInit, userOp);
                     });
}

} // namespace

LogicalResult
PropagateNearEndExpandDown::matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                            PatternRewriter &rewriter) const {
  Value result = expandOp.getResult();
  auto userRange = result.getUsers();
  SmallVector<Operation *> users(userRange.begin(), userRange.end());
  // Propagate one by one, to be safe
  if (expandOp.getReassociationIndices().empty()) {
    return failure();
  }
  auto *src = expandOp.getSrc().getDefiningOp();
  if (!src)
    return failure();
  if (isStopPropagatable(src))
    return failure();
  if (llvm::all_of(expandOp->getUsers(),
                   [&](Operation *op) { return isOutOp(op); })) {
    return failure();
  }

  for (Operation *userOp : users) {
    if (isReturnOp(userOp))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "Here\n";);
    bool firstRequirements = isMarkedAsElementwiseOp(userOp) ||
                             mlir::hivm::detail::isElemwiseNaryOpImpl(userOp) ||
                             isa<hivm::CopyOp>(userOp) ||
                             isa<hivm::StoreOp>(userOp) ||
                             isa<hivm::LoadOp>(userOp);
    LLVM_DEBUG(llvm::dbgs() << "checking " << *userOp << "\n";);
    if (!firstRequirements)
      return failure();
    LLVM_DEBUG(llvm::dbgs()
                   << "First requirement satisfied " << *userOp << " \n";);
    // Check if the marked
    // has ok res
    auto dpsOp = cast<DestinationStyleOpInterface>(userOp);
    auto oldDpsInits = dpsOp.getDpsInits();
    // Only check for one dps inputs for now
    if (dpsOp.getNumDpsInputs() != 1)
      return failure();
    for (auto res : oldDpsInits) {
      if (!res.getDefiningOp())
        continue;
      if (isArgOp(res.getDefiningOp()))
        continue;
      return failure();
    }
    for (auto res : userOp->getResults()) {
      if (!areValidUsers(res, userOp)) {
        LLVM_DEBUG(llvm::dbgs() << res << " res is invalid\n";);
        return failure();
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "--> Ok valid processing\n";);
  for (Operation *userOp : users) {
    if (isMarkedAsElementwiseOp(userOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Propagate expand down - Elemwise\n";);
      return handleElementwiseOp(expandOp, rewriter, userOp);
    }
    if (mlir::hivm::detail::isElemwiseNaryOpImpl(userOp) ||
        isa<hivm::CopyOp>(userOp) || isa<hivm::StoreOp>(userOp) ||
        isa<hivm::LoadOp>(userOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Propagate expand down - HIVM Elemwise\n";);
      return handleElementwiseOp(expandOp, rewriter, userOp);
    }
  }
  return failure();
}

} // namespace tensor
} // namespace mlir