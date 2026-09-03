//===- HIVMVectorizeOps.cpp - hivm op vectorize ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/IR/HIVMVectorize.h"
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMVECTORIZEOPS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-vectorize-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
using namespace mlir;
using namespace mlir::hivm;

namespace {
struct HIVMVectorizeOpsPass
    : public impl::HIVMVectorizeOpsBase<HIVMVectorizeOpsPass> {
  void runOnOperation() override;
};
} // namespace

void HIVMVectorizeOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  if (!hivm::isVF(funcOp))
    return;

  IRRewriter rewriter(&getContext());
  WalkResult result = funcOp.walk([&](VectorizableOpInterface op) {
    if (!canVectorizeHIVMOp(op.getOperation()))
      return WalkResult::advance();
    auto structuredOp = dyn_cast<HIVMStructuredOp>(op.getOperation());
    if (!structuredOp)
      return WalkResult::interrupt();
    FailureOr<SmallVector<int64_t>> vectorSizes =
        computeVectorSizes(structuredOp);
    if (failed(vectorSizes))
      return WalkResult::interrupt();
    LDBG("vectorSizes: " << utils::debugger::to_string(*vectorSizes));
    rewriter.setInsertionPoint(op.getOperation());
    if (failed(op.vectorize(rewriter, *vectorSizes)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createHIVMVectorizeOpsPass() {
  return std::make_unique<HIVMVectorizeOpsPass>();
}
