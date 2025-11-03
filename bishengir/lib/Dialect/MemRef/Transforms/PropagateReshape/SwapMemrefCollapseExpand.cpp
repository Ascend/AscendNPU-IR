//===- SwapMemrefCollapseExpand.cpp ---------------------------------------===//
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
// Swap collapse and expand order so collapse can be put down and expand can be
// put up
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRef/Transforms/PropagateReshape.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "propagate-reshape"
namespace mlir {
namespace memref {
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

LogicalResult
SwapMemrefCollapseExpand::matchAndRewrite(memref::ExpandShapeOp expandOp,
                                          PatternRewriter &rewriter) const {
  auto collapseOp = expandOp.getSrc().getDefiningOp<memref::CollapseShapeOp>();
  if (!collapseOp)
    return failure();
  auto *definedCollapse = collapseOp.getSrc().getDefiningOp();
  if (!definedCollapse || reshape_utils::isStopPropagatable(definedCollapse))
    return failure();
  if (llvm::all_of(expandOp->getUsers(),
                   [&](Operation *op) { return reshape_utils::isOutOp(op); })) {
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
  if (!reshape_utils::areReassociationsCompatible(
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

  utils::renumberReassociation(newReassociationExpand);
  utils::renumberReassociation(newReassociationCollapse);
  rewriter.setInsertionPointAfter(expandOp);
  auto newExpandType = memref::ExpandShapeOp::computeExpandedType(
      cast<MemRefType>(collapseOp.getSrc().getType()), newExpandShape,
      newReassociationExpand);
  if (failed(newExpandType) ||
      !memref::CollapseShapeOp::isGuaranteedCollapsible(
          *newExpandType, newReassociationCollapse)) {
    LLVM_DEBUG(llvm::dbgs() << "type conversion failed\n";);
    return failure();
  }
  auto newCollapseType = memref::CollapseShapeOp::computeCollapsedType(
      *newExpandType, newReassociationCollapse);
  if (newCollapseType != expandOp.getResult().getType()) {
    LLVM_DEBUG(llvm::dbgs() << "stride is not compatible\n";);
    return failure();
  }
  auto newExpandOp = rewriter.create<memref::ExpandShapeOp>(
      collapseOp.getLoc(), *newExpandType, collapseOp.getSrc(),
      newReassociationExpand);
  auto newCollapseOp = rewriter.create<memref::CollapseShapeOp>(
      expandOp.getLoc(), newCollapseType, newExpandOp.getResult(),
      newReassociationCollapse);
  rewriter.replaceAllUsesWith(expandOp, newCollapseOp.getResult());
  return success();
}
} // namespace memref
} // namespace mlir
