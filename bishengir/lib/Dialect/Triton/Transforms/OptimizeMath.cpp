//===-- OptimizeMath.cpp - Cheap math rewrites ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//   1. exp -> exp2 with log2(e) factor.
//      `math.exp(x)` is rewritten to `math.exp2(arith.mulf(x, splat(log2e)))`.
//      The downstream Canonicalizer constant-folds the new splat with any
//      adjacent constant-splat multiplier already present in `x`, so the
//      flash-attention scale rebase (`qk_scale = scale * log2(e)`) emerges
//      automatically without an FA-specific pattern.  No false negatives:
//      every exp gets the rewrite; if x is *not* a multiply chain, we just
//      pay one extra elementwise mulf, which is still a clear win because
//      software exp internally does a divide-by-ln(2) plus exp2 anyway.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "optimize-math"
#define DBGS() (llvm::dbgs() << "[OptimizeMath] ")

namespace bishengir {
namespace triton {
#define GEN_PASS_DEF_OPTIMIZEMATH
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

/// Build a float constant of `ty` (scalar or splat tensor); nullptr if `ty`
/// has a non-float element type.
static Value buildFloatConst(Type ty, double floatVal, OpBuilder &b,
                              Location loc) {
  Type elemTy = isa<RankedTensorType>(ty)
                    ? cast<RankedTensorType>(ty).getElementType()
                    : ty;
  auto fty = dyn_cast<FloatType>(elemTy);
  if (!fty)
    return nullptr;

  APFloat val(static_cast<float>(floatVal));
  bool losesInfo;
  val.convert(fty.getFloatSemantics(), APFloat::rmNearestTiesToEven,
              &losesInfo);

  if (auto rt = dyn_cast<RankedTensorType>(ty))
    return b.create<arith::ConstantOp>(loc, ty,
                                          DenseElementsAttr::get(rt, val));
  return b.create<arith::ConstantOp>(loc, FloatAttr::get(fty, val));
}

//===----------------------------------------------------------------------===//
// ExpToExp2Pattern — rewrite `math.exp(x)` to `math.exp2(x * log2(e))`.
// Downstream canonicalize fuses the new splat(log2e) with any existing scale
// the kernel was already multiplying by, so the log2-rebase costs nothing.
//===----------------------------------------------------------------------===//

struct ExpToExp2Pattern : public OpRewritePattern<math::ExpOp> {
  using OpRewritePattern<math::ExpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpOp expOp,
                                PatternRewriter &rewriter) const override {
    Location loc = expOp.getLoc();
    Value x = expOp.getOperand();
    Type ty = x.getType();

    Type elemTy = isa<RankedTensorType>(ty)
                      ? cast<RankedTensorType>(ty).getElementType()
                      : ty;
    if (!isa<FloatType>(elemTy))
      return failure();

    Value log2eVal =
        buildFloatConst(ty, /*log2(e)=*/1.4426950408889634, rewriter, loc);
    if (!log2eVal)
      return failure();

    Value scaled = rewriter.create<arith::MulFOp>(loc, x, log2eVal);
    rewriter.replaceOpWithNewOp<math::Exp2Op>(expOp, scaled);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct OptimizeMathPass : public impl::OptimizeMathBase<OptimizeMathPass> {
  void runOnOperation() override {
    auto fn = getOperation();
    LLVM_DEBUG(DBGS() << "running on " << fn.getName() << '\n');

    // math.exp -> math.exp2 with log2(e) factor.
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<ExpToExp2Pattern>(&getContext());
      (void)applyPatternsGreedily(fn, std::move(patterns));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createOptimizeMathPass() {
  return std::make_unique<OptimizeMathPass>();
}

} // namespace triton
} // namespace bishengir
