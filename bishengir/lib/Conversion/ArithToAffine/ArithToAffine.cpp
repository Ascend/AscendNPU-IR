//===- ArithToAffine.cpp - conversion from Arith to Affine dialect --------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
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

template <typename ArithOpTy, typename AffineOpTy>
struct ArithMinMaxToAffine : public OpRewritePattern<ArithOpTy> {
  using OpRewritePattern<ArithOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ArithOpTy op,
                                PatternRewriter &rewriter) const final {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    if (!lhs.getType().isIndex() || !rhs.getType().isIndex())
      return rewriter.notifyMatchFailure(op, "lhs or rhs is not index typed!");

    AffineExpr lhsExpr = getAffineSymbolExpr(0, rewriter.getContext());
    AffineExpr rhsExpr = getAffineSymbolExpr(1, rewriter.getContext());

    OpFoldResult ofr;
    AffineMap map =
        AffineMap::get(0, 2, {lhsExpr, rhsExpr}, rewriter.getContext());
    if constexpr (std::is_same_v<AffineOpTy, affine::AffineMaxOp>)
      ofr = affine::makeComposedFoldedAffineMax(rewriter, op->getLoc(), map,
                                                {lhs, rhs});
    else
      ofr = affine::makeComposedFoldedAffineMin(rewriter, op->getLoc(), map,
                                                {lhs, rhs});

    rewriter.replaceOp(
        op, getValueOrCreateConstantIndexOp(rewriter, op->getLoc(), ofr));
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
      BinaryArithOpToAffineApply<arith::RemSIOp, AffineExprKind::Mod>,
      // For index-typed operands, SI and UI doesn't mean anything. So we can
      // convert them to affine.
      ArithMinMaxToAffine<arith::MaxSIOp, affine::AffineMaxOp>,
      ArithMinMaxToAffine<arith::MaxUIOp, affine::AffineMaxOp>,
      ArithMinMaxToAffine<arith::MinSIOp, affine::AffineMinOp>,
      ArithMinMaxToAffine<arith::MinUIOp, affine::AffineMinOp>>(
      patterns.getContext());
}

namespace {
struct ArithToAffineConversionPass
    : public impl::ConvertArithToAffineBase<ArithToAffineConversionPass> {
  void runOnOperation() override;
};
} // namespace

void ArithToAffineConversionPass::runOnOperation() {
  auto *module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<affine::AffineDialect>();
  target.addDynamicallyLegalOp<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                               arith::CeilDivSIOp, arith::DivSIOp,
                               arith::RemSIOp, arith::MaxSIOp, arith::MinSIOp>(
      [](Operation *op) {
        assert(op->getNumOperands() ==
               2); // candidate arith must have 2 operands
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
