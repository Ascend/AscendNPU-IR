//===- GPUToHFusion.cpp - conversion from GPU to HFusion dialect ----------===//
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
