//===- PropagateMemrefCollapseDown.cpp ------------------------------------===//
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

#include "bishengir/Dialect/Utils/Util.h"
using namespace mlir::utils::debugger;

namespace mlir {
namespace memref {

namespace {
Operation *createNewExpandOpFromCollapseOp(memref::CollapseShapeOp &collapseOp,
                                           PatternRewriter &rewriter,
                                           Location loc, Value operand) {
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(operand);
  auto reassociation = collapseOp.getReassociationIndices();
  auto currentShape = utils::getShape(collapseOp.getSrc().getType());
  if (isa<MemRefType>(operand.getType())) {
    return rewriter.create<memref::ExpandShapeOp>(loc, currentShape, operand,
                                                  reassociation);
  }
  auto resultType =
      RankedTensorType::get(currentShape, getElementTypeOrSelf(operand));
  return rewriter.create<tensor::ExpandShapeOp>(loc, resultType, operand,
                                                reassociation);
}

LogicalResult handleLoadOp(memref::CollapseShapeOp collapseOp,
                           PatternRewriter &rewriter, Operation *userOp) {
  auto loadOp = cast<hivm::LoadOp>(userOp);
  auto resultRank = loadOp.getDstOperandType().getRank();
  SmallVector<Value> newOperands;
  auto loc = loadOp.getLoc();
  for (Value operand : userOp->getOperands()) {
    rewriter.setInsertionPointAfterValue(operand);
    auto shapeRank = utils::getShapeRank(operand);
    // Check in case its scalar elemwise
    if (shapeRank.has_value() &&
        static_cast<int64_t>(shapeRank.value()) == resultRank) {
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
  rewriter.modifyOpInPlace(userOp, [&]() { userOp->setOperands(newOperands); });
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
    if (collapseOp->getParentOp() != userOp->getParentOp())
      continue;
    if (isa<hivm::LoadOp>(userOp)) {
      return handleLoadOp(collapseOp, rewriter, userOp);
    }
  }
  return failure();
}
} // namespace memref
} // namespace mlir