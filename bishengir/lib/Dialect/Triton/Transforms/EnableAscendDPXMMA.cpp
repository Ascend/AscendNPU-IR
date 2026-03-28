//===- EnableAscendDPXMMA.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Promotes DotOp operands and result to f32 where necessary, enabling the
// Ascend DPX hardware MMA path which requires f32 types.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace bishengir {
namespace triton {

#define GEN_PASS_DEF_ENABLEASCENDDPXMMA
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace triton
} // namespace bishengir

using namespace mlir;
using namespace mlir::triton;

namespace bishengir {
namespace triton {

namespace {

/// Cast a ranked-tensor value's element type to f32 via arith::ExtFOp.
/// Returns the value unchanged if its element type is already f32.
static Value castToF32(PatternRewriter &rewriter, Location loc, Value v) {
  auto tensorTy = cast<RankedTensorType>(v.getType());
  if (tensorTy.getElementType().isF32())
    return v;
  auto f32Ty = cast<RankedTensorType>(tensorTy.clone(rewriter.getF32Type()));
  return rewriter.create<arith::ExtFOp>(loc, f32Ty, v);
}

/// Truncate a ranked-tensor value from f32 down to targetElemTy via
/// arith::TruncFOp.  Returns the value unchanged when the types already match.
static Value castFromF32(PatternRewriter &rewriter, Location loc, Value v,
                         Type targetElemTy) {
  auto tensorTy = cast<RankedTensorType>(v.getType());
  if (tensorTy.getElementType() == targetElemTy)
    return v;
  auto dstTy = cast<RankedTensorType>(tensorTy.clone(targetElemTy));
  return rewriter.create<arith::TruncFOp>(loc, dstTy, v);
}

/// If any operand (A, B, or accumulator C) of a DotOp is not f32, cast all
/// operands to f32, create a new f32 DotOp, then truncate the result back to
/// the original output element type if needed.
class PromoteDotOperandsToF32 : public OpRewritePattern<DotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto elemTy = [](Value v) {
      return cast<RankedTensorType>(v.getType()).getElementType();
    };

    if (elemTy(dotOp.getA()).isF32() && elemTy(dotOp.getB()).isF32() &&
        elemTy(dotOp.getC()).isF32())
      return failure();

    Location loc = dotOp.getLoc();
    Value a = castToF32(rewriter, loc, dotOp.getA());
    Value b = castToF32(rewriter, loc, dotOp.getB());
    Value c = castToF32(rewriter, loc, dotOp.getC());
    auto origResultTy = cast<RankedTensorType>(dotOp.getType());
    auto f32ResultTy =
        cast<RankedTensorType>(origResultTy.clone(rewriter.getF32Type()));
    Value newDot = rewriter.create<DotOp>(loc, f32ResultTy, ValueRange{a, b, c},
                                          dotOp->getAttrs());

    Value result =
        castFromF32(rewriter, loc, newDot, origResultTy.getElementType());

    rewriter.replaceOp(dotOp, result);
    return success();
  }
};

class EnableAscendDPXMMAPass
    : public impl::EnableAscendDPXMMABase<EnableAscendDPXMMAPass> {
public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<PromoteDotOperandsToF32>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createEnableAscendDPXMMAPass() {
  return std::make_unique<EnableAscendDPXMMAPass>();
}

} // namespace triton
} // namespace bishengir
