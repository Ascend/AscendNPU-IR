//===- GPUToHFusion.cpp - conversion from GPU to HFusion dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/GPUToHFusion/GPUToHFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

struct GPUBarrierToHFusionBarrierOp : public OpRewritePattern<gpu::BarrierOp> {
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<hfusion::BarrierOp>(op);
    return success();
  }
};

void mlir::hfusion::populateGPUToHFusionConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<GPUBarrierToHFusionBarrierOp>(patterns.getContext());
}

namespace {
struct GPUToHFusionConversionPass
    : public impl::ConvertGPUToHFusionBase<GPUToHFusionConversionPass> {
  void runOnOperation() override;
};
} // namespace

void GPUToHFusionConversionPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<hfusion::HFusionDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();

  RewritePatternSet patterns(&getContext());
  populateGPUToHFusionConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createGPUToHFusionConversionPass() {
  return std::make_unique<GPUToHFusionConversionPass>();
}
