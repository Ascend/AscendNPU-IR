//===- BubbleUpExtractSlice.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_BUBBLEUPEXTRACTSLICE
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;

namespace {
struct BubbleUpExtractSlicePass
    : public impl::BubbleUpExtractSliceBase<BubbleUpExtractSlicePass> {
  explicit BubbleUpExtractSlicePass(
      const BubbleUpExtractSliceOptions &options)
      : BubbleUpExtractSliceBase(options) {}
  void runOnOperation() override;
};
} // namespace

void BubbleUpExtractSlicePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  RewritePatternSet patterns(funcOp.getContext());
  linalg::BubbleUpExtractSliceOptions linalgOptions;
  linalgOptions.aggressive = this->aggressive;
  linalg::populateBubbleUpExtractSliceOpPatterns(patterns, linalgOptions);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::tensor::createBubbleUpExtractSlicePass(
    const BubbleUpExtractSliceOptions &options) {
  return std::make_unique<BubbleUpExtractSlicePass>(options);
}
