//===- SwapCollapseExpand.cpp ---------------------------------------------===//
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
  if (!::mlir::reshape_utils::areReassociationsCompatible(
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

  if (options.forRegbased &&
      exceedsUnitDimPropagationLimit(newExpandShape,
                                     options.maxUnitDimsForPropagation)) {
    return rewriter.notifyMatchFailure(
        expandOp, "common refinement exceeds the unit-dimension threshold");
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
