//===- HIVMVectorizeOps.cpp - hivm op vectorize ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMVECTORIZEOPS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-vectorize-ops"

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct HIVMVectorizeOpsPattern
    : public OpInterfaceRewritePattern<mlir::hivm::VectorizableOpInterface> {
  using OpInterfaceRewritePattern<
      mlir::hivm::VectorizableOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(mlir::hivm::VectorizableOpInterface op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> vectorSizes = {4, 8};
    return op.vectorize(rewriter, vectorSizes);
  }
};

struct HIVMVectorizeOpsPass
    : public impl::HIVMVectorizeOpsBase<HIVMVectorizeOpsPass> {
  void runOnOperation() override;
};
} // namespace

void HIVMVectorizeOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  if (!hivm::isVF(funcOp))
    return;
  RewritePatternSet patterns(&getContext());
  patterns.add<HIVMVectorizeOpsPattern>(&getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createHIVMVectorizeOpsPass() {
  return std::make_unique<HIVMVectorizeOpsPass>();
}