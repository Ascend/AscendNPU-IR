//===- ArithToAffine.cpp - conversion from Arith to Affine dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOAFFINE
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

template <typename ArithOpTy, AffineExprKind AffineExprTy>
struct BinaryArithOpToAffineApply : public OpRewritePattern<ArithOpTy> {
  using OpRewritePattern<ArithOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ArithOpTy op,
                                PatternRewriter &rewriter) const final {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    if (!lhs.getType().isIndex() || !rhs.getType().isIndex())
      return rewriter.notifyMatchFailure(op, "lhs or rhs is not index typed!");

    AffineExpr lhsExpr = getAffineSymbolExpr(0, rewriter.getContext());
    AffineExpr rhsExpr = getAffineSymbolExpr(1, rewriter.getContext());

    // There is no AffineExprKind::Sub, need special treatment.
    if constexpr (std::is_same_v<ArithOpTy, arith::SubIOp>) {
      rhsExpr = -(rhsExpr);
    }
    AffineExpr result = getAffineBinaryOpExpr(AffineExprTy, lhsExpr, rhsExpr);

    auto applyOp = affine::makeComposedAffineApply(rewriter, op->getLoc(),
                                                   result, {lhs, rhs});

    rewriter.replaceOp(op, applyOp);
    return success();
  }
};

void mlir::arith::populateArithToAffineConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      BinaryArithOpToAffineApply<arith::AddIOp, AffineExprKind::Add>,
      BinaryArithOpToAffineApply<arith::SubIOp, AffineExprKind::Add>,
      BinaryArithOpToAffineApply<arith::MulIOp, AffineExprKind::Mul>,
      BinaryArithOpToAffineApply<arith::CeilDivSIOp, AffineExprKind::CeilDiv>,
      BinaryArithOpToAffineApply<arith::DivSIOp, AffineExprKind::FloorDiv>,
      BinaryArithOpToAffineApply<arith::RemSIOp, AffineExprKind::Mod>>(
      patterns.getContext());
}

namespace {
struct ArithToAffineConversionPass
    : public impl::ConvertArithToAffineBase<ArithToAffineConversionPass> {
  void runOnOperation() override;
};
} // namespace

void ArithToAffineConversionPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<affine::AffineDialect>();
  target.addDynamicallyLegalOp<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                               arith::CeilDivSIOp, arith::DivSIOp,
                               arith::RemSIOp>([](Operation *op) {
    assert(op->getNumOperands() == 2); // candidate arith must have 2 operands
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    return !lhs.getType().isIndex() || !rhs.getType().isIndex();
  });
  RewritePatternSet patterns(&getContext());
  arith::populateArithToAffineConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createArithToAffineConversionPass() {
  return std::make_unique<ArithToAffineConversionPass>();
}
