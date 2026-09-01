//===-----DecomposeTensorConcat.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_DECOMPOSETENSORCONCAT
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
using namespace mlir;

struct DecomposeTensorConcatPass
    : public impl::DecomposeTensorConcatBase<DecomposeTensorConcatPass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    tensor::populateDecomposeTensorConcatPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tensor::createDecomposeTensorConcatPass() {
  return std::make_unique<DecomposeTensorConcatPass>();
}
