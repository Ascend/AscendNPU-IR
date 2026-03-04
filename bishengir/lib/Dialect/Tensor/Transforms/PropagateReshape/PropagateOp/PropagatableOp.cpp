//===- PropagatableOp.cpp - Propagatable operation base implementation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagatableOp.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagateExpandUp.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"

#define DEBUG_TYPE "propagatable-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace mlir::tensor {
using namespace mlir::tensor::reshape_utils;
using namespace mlir::utils::debugger;
LogicalResult
PropagatableOp::matchAndRewriteExpand(PatternRewriter &rewriter, Operation *op,
                                      tensor::ExpandShapeOp expandOp) {
  return rewriter.notifyMatchFailure(op, "Expand up is not implemented");
}

LogicalResult
PropagatableOp::matchAndRewriteCollapse(PatternRewriter &rewriter,
                                        Operation *op,
                                        tensor::CollapseShapeOp collapseOp) {
  return rewriter.notifyMatchFailure(op, "Collpase down is not implemented");
}
} // namespace mlir::tensor
