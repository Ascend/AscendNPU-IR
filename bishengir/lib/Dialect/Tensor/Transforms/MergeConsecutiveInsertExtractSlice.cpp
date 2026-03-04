//===- MergeConsecutiveInsertExtractSlice.cpp -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_MERGECONSECUTIVEINSERTEXTRACTSLICE
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;

namespace {
struct MergeConsecutiveInsertExtractSlicePass
    : public impl::MergeConsecutiveInsertExtractSliceBase<
          MergeConsecutiveInsertExtractSlicePass> {
  void runOnOperation() override;
};
} // namespace

void MergeConsecutiveInsertExtractSlicePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  RewritePatternSet patterns(funcOp.getContext());
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass>
mlir::tensor::createMergeConsecutiveInsertExtractSlicePass() {
  return std::make_unique<MergeConsecutiveInsertExtractSlicePass>();
}
