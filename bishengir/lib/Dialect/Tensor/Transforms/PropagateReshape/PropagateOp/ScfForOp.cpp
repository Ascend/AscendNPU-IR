//===- ScfForOp.cpp - scf::forOp propagate implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagatableOp.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"

#define DEBUG_TYPE "propagate-scf-for-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir::tensor {
using namespace mlir::tensor::reshape_utils;
using namespace mlir::utils::debugger;

LogicalResult
PropagatableScfFor::matchAndRewriteExpand(mlir::PatternRewriter &rewriter,
                                          mlir::Operation *op,
                                          tensor::ExpandShapeOp expandOp) {
  scf::ForOp forOp = dyn_cast<scf::ForOp>(op);
  if (!forOp)
    return failure();
  Block *forOpBody = forOp.getBody();
  auto resultNumber = cast<OpResult>(expandOp.getSrc()).getResultNumber();
  // %scfForResult:N = scf.for {
  //
  //    scf.yield #0, #1, ... #N - 1)
  // }
  // resultNumber has the range [0, N)
  Operation *forOpTerminator = forOpBody->getTerminator();
  scf::YieldOp yieldOp = cast<scf::YieldOp>(forOpTerminator);

  OpOperand &yieldOperand = yieldOp->getOpOperand(resultNumber);
  // Expand it
  // ResultType, Src, Reassociation, ExpandShape
  Type oldResultType = yieldOperand.get().getType();
  Type newResultType = expandOp.getResultType();

  // %scfForResult:N = scf.for {
  //   ...
  //   scf.yield #0, #1, ... #N - 1)
  // }
  // %newCollapsedScfResult = tensor.collapse_shape %scfForResult#2  <--- insert
  // %oldExpandOp = tensor.expand_shape %newCollapsedScfResult
  // resultNumber has the range [0, N)
  auto reassociationMap = expandOp.getReassociationIndices();
  rewriter.setInsertionPoint(forOpTerminator);
  tensor::ExpandShapeOp newExpandOp = rewriter.create<tensor::ExpandShapeOp>(
      yieldOp->getLoc(), newResultType, yieldOperand.get(), reassociationMap);

  yieldOperand.set(newExpandOp);
  OpResult forOpResult = forOp->getResult(resultNumber);
  LDBG("Processing result number here " << resultNumber << "!");
  forOpResult.setType(newResultType);
  rewriter.setInsertionPointAfterValue(forOpResult);
  tensor::CollapseShapeOp collapsedForOpResult =
      rewriter.create<tensor::CollapseShapeOp>(yieldOp->getLoc(), oldResultType,
                                               forOpResult, reassociationMap);
  rewriter.replaceAllUsesExcept(forOpResult, collapsedForOpResult.getResult(),
                                collapsedForOpResult);

  // %scfForResult:N = scf.for (arg0, arg1, ... argN - 1){
  //   %collapsedArg = tensor.collapse_shape src(%arg2) <---- Insert this
  //   Replace all usage of %arg2 with collapsedArg
  //   %newExpandYield = tensor.expand_shape(#2)
  //   scf.yield #0, #1, %newExpandYield, ... #N - 1)
  // }
  // %newCollapsedScfResult = tensor.collapse_shape %scfForResult#2
  // %oldExpandOp = tensor.expandShape %newCollapsedScfResult
  // resultNumber has the range [0, N)
  BlockArgument regionIterArg = forOp.getRegionIterArg(resultNumber);
  regionIterArg.setType(newResultType);
  rewriter.setInsertionPointAfterValue(regionIterArg);
  auto argumentCollapsed = rewriter.create<tensor::CollapseShapeOp>(
      forOp.getLoc(), oldResultType, regionIterArg, reassociationMap);
  rewriter.replaceAllUsesExcept(regionIterArg, argumentCollapsed.getResult(),
                                argumentCollapsed);

  // Adjust the init source of the for op
  // %expandedInit = tensor.expand_shape src(%arg2)  <---- Insert this
  // %scfForResult:N = scf.for (arg0, arg1, %expandedInit, ... argN - 1){
  //   %collapsedArg = tensor.collapse_shape src(%arg2)
  //   Replace all usage of %arg2 with collapsedArg
  //   %newExpandYield = tensor.expand_shape(#2)
  //   scf.yield #0, #1, %newExpandYield, ... #N - 1)
  // }
  // %newCollapsedScfResult = tensor.collapse_shape %scfForResult#2
  // %oldExpandOp = tensor.expandShape %newCollapsedScfResult
  // resultNumber has the range [0, N)
  OpOperand &forOpInit = forOp.getInitsMutable()[resultNumber];
  rewriter.setInsertionPointAfterValue(forOpInit.get());
  tensor::ExpandShapeOp expandedInit = rewriter.create<tensor::ExpandShapeOp>(
      forOpInit.get().getLoc(), newResultType, forOpInit.get(),
      reassociationMap);
  forOpInit.set(expandedInit.getResult());
  return success();
}

} // namespace mlir::tensor
