//===- AffineToTriton.cpp - conversion from Affine to Triton dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace {
struct AffineApplyOpPattern : public OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    AffineMap map = op.getAffineMap();
    assert(map.getNumResults() == 1 &&
           "affine.apply verifier requires a single-result map");
    ValueRange operands = op.getOperands();
    Value expanded =
        affine::expandAffineExpr(rewriter, op.getLoc(), map.getResult(0),
                                 operands.take_front(map.getNumDims()),
                                 operands.drop_front(map.getNumDims()));
    if (!expanded)
      return op.emitOpError("unsupported affine.apply expression");
    rewriter.replaceOp(op, expanded);
    return success();
  }
};
} // namespace

void mlir::hivm::populateAffineToTritonPatterns(RewritePatternSet &patterns) {
  patterns.add<AffineApplyOpPattern>(patterns.getContext());
}
