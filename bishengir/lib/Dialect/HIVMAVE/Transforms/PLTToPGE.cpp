//===------------------------- PLTToPGE.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_PLTTOPGE
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "ave-plt-to-pge"

using namespace mlir;

namespace {

struct PLTToPGEPass : public impl::PLTToPGEBase<PLTToPGEPass> {
  using PLTToPGEBase<PLTToPGEPass>::PLTToPGEBase;

public:
  void runOnOperation() override;
};

/// Convert VFPltOp with constant true_shape into VFPgeOp.
///
/// VFPltOp produces a mask where the first `true_shape` bits are true and the
/// rest are false.  When `true_shape` is a compile-time constant that matches
/// one of the PgePattern enumerators (VL1, VL2, …, ALL, ALLF), the op can be
/// replaced by VFPgeOp which is a cheaper, hardware-native pattern generator.
struct PLTToPGEPattern : public OpRewritePattern<hivmave::VFPltOp> {
public:
  using OpRewritePattern<hivmave::VFPltOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivmave::VFPltOp op,
                                PatternRewriter &rewriter) const override {
    // PGE does not produce new_true_shape; bail out if it has users.
    if (!op.getNewTrueShape().use_empty())
      return failure();

    // Only convert when true_shape is a compile-time constant.
    IntegerAttr trueShapeAttr;
    if (!matchPattern(op.getTrueShape(), m_Constant(&trueShapeAttr)))
      return failure();

    int64_t trueShape = trueShapeAttr.getInt();
    auto maskType = cast<VectorType>(op.getRes().getType());
    int64_t resultShape = maskType.getNumElements();

    // Map the constant to a PgePattern enumerator.
    auto pgeAttr =
        hivmave::getPgePatternAttr(rewriter, trueShape, resultShape);
    if (failed(pgeAttr))
      return failure();

    // Create VFPgeOp and replace the mask result of PLT.
    auto pge = rewriter.create<hivmave::VFPgeOp>(
        op->getLoc(), op.getRes().getType(), pgeAttr->getValue());
    rewriter.replaceAllUsesWith(op.getRes(), pge.getRes());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void PLTToPGEPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<PLTToPGEPattern>(ctx);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> hivmave::createPLTToPGEPass() {
  return std::make_unique<PLTToPGEPass>();
}
