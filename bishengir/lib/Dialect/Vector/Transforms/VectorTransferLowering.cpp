//===- VectorTransferLowering.cpp - Lower Transfer Ops --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_VECTORTRANSFERLOWERING
#include "bishengir/Dialect/Vector/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::vector;

#define DEBUG_TYPE "vector-transfer-lowering"

namespace {

struct VectorTransferLoweringPass
    : public impl::VectorTransferLoweringBase<VectorTransferLoweringPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void VectorTransferLoweringPass::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(op->getContext());
  vector::populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
  vector::populateVectorTransferLoweringPatterns(patterns,
                                                 /*maxTransferRank=*/1);
  if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::vector::createVectorTransferLoweringPass() {
  return std::make_unique<VectorTransferLoweringPass>();
}