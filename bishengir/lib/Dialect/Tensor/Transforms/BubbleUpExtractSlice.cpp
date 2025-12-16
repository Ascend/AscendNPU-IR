//===- BubbleUpExtractSlice.cpp ---------------------------------*- C++ -*-===//
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
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_BUBBLEUPEXTRACTSLICE
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;

namespace {

// Additional BubbleUpExtractSlice patterns for simple producer ops
// (e.g. Fill, Cast, elementwise unary) that can be re-created on slices.
template <typename ProducerOp>
struct AggressiveBubbleUpExtractSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto producerOp = sliceOp.getSource().getDefiningOp<ProducerOp>();
    auto sliceType = dyn_cast<RankedTensorType>(sliceOp.getType());
    if (!producerOp || !sliceType)
      return failure();

    auto input = producerOp.getOperands()[0];
    auto result = producerOp.getResult(0);
    constexpr bool isFill = std::is_same_v<ProducerOp, linalg::FillOp>;
    auto inTy = isFill ? nullptr : dyn_cast<RankedTensorType>(input.getType());
    if constexpr (isFill) {
      if (!sliceOp.hasUnitStride() || !isa<FloatType, IntegerType>(input.getType()))
        return failure();
    } else {
      auto outTy = dyn_cast<RankedTensorType>(result.getType());
      if (!inTy || !outTy || inTy.getShape() != outTy.getShape())
        return failure();
    }

    if (!llvm::all_of(result.getUsers(), [](Operation *u) {
          return isa<tensor::ExtractSliceOp>(u);
        }))
      return failure();
    auto sliceLoc = sliceOp.getLoc();
    auto empty = rewriter.create<tensor::EmptyOp>(
        sliceLoc, sliceType.getShape(), sliceType.getElementType());

    // Create a new sliced producer
    Operation *newOp = nullptr;
    if constexpr (isFill) {
      newOp = rewriter.create<linalg::FillOp>(sliceLoc, input, empty.getResult());
    } else {
      auto slicedInputType =
          RankedTensorType::get(sliceType.getShape(), inTy.getElementType());
      auto slicedInput = rewriter.create<tensor::ExtractSliceOp>(
          sliceLoc, slicedInputType, input, sliceOp.getMixedOffsets(),
          sliceOp.getMixedSizes(), sliceOp.getMixedStrides());

      newOp = rewriter.create<ProducerOp>(sliceLoc, empty.getType(),
                                                slicedInput.getResult(),
                                                empty.getResult());
    }
    newOp->setAttrs(producerOp->getAttrs());
    rewriter.replaceOp(sliceOp, newOp->getResult(0));
    return success();
  }
};

struct BubbleUpExtractSlicePass
    : public impl::BubbleUpExtractSliceBase<BubbleUpExtractSlicePass> {
  explicit BubbleUpExtractSlicePass(
      const BubbleUpExtractSliceOptions &options)
      : BubbleUpExtractSliceBase(options) {}
  void runOnOperation() override;
};
} // namespace

void populateAggressiveBubbleUpExtractSlicePatterns(
    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns
      .add<AggressiveBubbleUpExtractSlice<linalg::FillOp>,
           AggressiveBubbleUpExtractSlice<hfusion::CastOp>,
           AggressiveBubbleUpExtractSlice<linalg::ElemwiseUnaryOp>
           >(ctx);
}

void BubbleUpExtractSlicePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  RewritePatternSet patterns(funcOp.getContext());
  linalg::BubbleUpExtractSliceOptions linalgOptions;
  linalgOptions.aggressive = this->aggressive;
  linalg::populateBubbleUpExtractSliceOpPatterns(patterns, linalgOptions);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  if (linalgOptions.aggressive) {
    populateAggressiveBubbleUpExtractSlicePatterns(patterns);
  }
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::tensor::createBubbleUpExtractSlicePass(
    const BubbleUpExtractSliceOptions &options) {
  return std::make_unique<BubbleUpExtractSlicePass>(options);
}
