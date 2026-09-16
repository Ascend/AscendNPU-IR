//===--------- DowngradeFP64CstOp.cpp - Downgrade FP64 to FP32Pass---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <set>

namespace mlir {
#define GEN_PASS_DEF_DOWNGRADEFP64CSTOPPASS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hfusion;

namespace {
struct F64ConstToF32Pattern : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constOp,
                                PatternRewriter &rewriter) const override {
    // Check if const type is F64 and not already processed
    if (isa<FloatAttr>(constOp.getValue())) {
      auto floatAttr = cast<FloatAttr>(constOp.getValue());
      if (floatAttr.getType().isF64()) {
        auto truncValue =
            static_cast<float>(floatAttr.getValue().convertToDouble());
        auto f32Value = FloatAttr::get(FloatType::getF32(rewriter.getContext()),
                                       truncValue);
        auto f64ConstOp = rewriter.create<arith::ConstantOp>(
            constOp.getLoc(), FloatType::getF64(rewriter.getContext()),
            f32Value);
        rewriter.replaceOp(constOp, f64ConstOp.getResult());

        return success();
      }
    }
    return failure();
  }
};

struct PrecisionLossTruncFPattern : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp truncFOp,
                                PatternRewriter &rewriter) const override {
    if (!truncFOp->getOperand(0).getType().isF64()) {
      return failure();
    }
    arith::ConstantOp cstOp = llvm::dyn_cast<arith::ConstantOp>(
        truncFOp->getOperand(0).getDefiningOp());
    if (!cstOp || !cstOp.getValue().getType().isF32())
      return failure();

    auto newCst = rewriter.create<arith::ConstantOp>(
        cstOp.getLoc(), FloatType::getF32(rewriter.getContext()),
        cstOp.getValue());
    auto newTruncFOp = rewriter.create<arith::TruncFOp>(
        truncFOp.getLoc(), truncFOp.getOut().getType(), newCst);

    rewriter.replaceOp(truncFOp, newTruncFOp);
    return success();
  }
};

class DowngradeFP64CstOpPass
    : public impl::DowngradeFP64CstOpPassBase<DowngradeFP64CstOpPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult convertFP64toFP32(func::FuncOp func, OpBuilder &builder) {
    RewritePatternSet patterns(func.getContext());
    patterns.add<F64ConstToF32Pattern>(func.getContext());
    patterns.add<PrecisionLossTruncFPattern>(func.getContext());
    return applyPatternsGreedily(func, std::move(patterns));
  }
};

void DowngradeFP64CstOpPass::runOnOperation() {
  func::FuncOp func = getOperation();
  OpBuilder builder(&getContext());
  if (failed(convertFP64toFP32(func, builder))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createDowngradeFP64CstOpPass() {
  return std::make_unique<DowngradeFP64CstOpPass>();
}
