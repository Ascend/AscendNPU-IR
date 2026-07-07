//===- AnnotationMark.cpp - Annotation mark implementation ----------------===//
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

#define DEBUG_TYPE "propagate-annotation-mark"
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
PropagatableAnnotationMark::matchAndRewriteCollapse(PatternRewriter &rewriter,
                                                    Operation *op,
                                                    tensor::CollapseShapeOp
                                                    collapseOp) {
  auto markOp = cast<annotation::MarkOp>(op);
  auto resultRank = cast<ShapedType>(markOp.getSrc().getType()).getRank();
  SmallVector<Value> newOperands = getNewOperands(
      collapseOp, rewriter, op, resultRank);
  rewriter.modifyOpInPlace(op, [&]() {
    op->setOperands(newOperands);
  });
  return success();
}

} // namespace mlir::tensor