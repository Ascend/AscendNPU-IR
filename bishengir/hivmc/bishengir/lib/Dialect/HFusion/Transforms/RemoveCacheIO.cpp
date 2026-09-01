//===--------- RemoveCacheIO.cpp - Remove CacheIO Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-remove-cache-io"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_REMOVECACHEIO
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {

template<typename OpType>
struct RemoveCacheIO : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0) {
      return failure();
    }
    rewriter.replaceAllUsesWith(op->getResult(0), op.getInputs()[0]);
    return success();
  }
};

struct RemoveCacheIOPass : public impl::RemoveCacheIOBase<RemoveCacheIOPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RemoveCacheIO<hfusion::LoadOp>>(context);
    patterns.add<RemoveCacheIO<hfusion::StoreOp>>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createRemoveCacheIO() {
  return std::make_unique<RemoveCacheIOPass>();
}
