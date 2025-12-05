//===- TensorToHFusion.cpp - conversion from Tensor to HFusion dialect ----===//
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

#include "bishengir/Conversion/TensorToHFusion/TensorToHFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

struct TensorSplatOpToLinalgFill : public OpRewritePattern<tensor::SplatOp> {
  using OpRewritePattern<tensor::SplatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::SplatOp op,
                                PatternRewriter &rewriter) const final {
    auto retTensorType = op.getAggregate().getType();
    ValueRange dynamicSizes = op.getDynamicSizes();
    auto cstVal = op.getInput();
    Value emptyTensorOp = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), retTensorType, dynamicSizes);
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{cstVal},
                                                ValueRange{emptyTensorOp});
    return success();
  }
};

void mlir::hfusion::populateTensorToHFusionConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TensorSplatOpToLinalgFill>(patterns.getContext());
}

namespace {
struct TensorToHFusionConversionPass
    : public impl::ConvertTensorToHFusionBase<TensorToHFusionConversionPass> {
  void runOnOperation() override;
};
} // namespace

void TensorToHFusionConversionPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<tensor::TensorDialect, linalg::LinalgDialect,
                         hfusion::HFusionDialect>();
  target.addIllegalOp<tensor::SplatOp>();
  RewritePatternSet patterns(&getContext());
  populateTensorToHFusionConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createTensorToHFusionConversionPass() {
  return std::make_unique<TensorToHFusionConversionPass>();
}
