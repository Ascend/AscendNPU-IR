//===- NormalizeScalar.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"

namespace mlir::hfusion {

template <typename OpType>
struct NormalizeScalarLikeTensorOp : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    bool isConverted = false;
    SmallVector<Value> inputsNew;
    for (auto inp : op.getInputs()) {
      auto inpNew = singleElemDenseTensorToScalar(inp, rewriter);
      if (inpNew.has_value()) {
        inputsNew.push_back(*inpNew);
        isConverted = true;
      } else {
        inputsNew.push_back(inp);
      }
    }

    SmallVector<Value> outputsNew;
    for (auto out : op.getOutputs()) {
      auto outNew = singleElemDenseTensorToScalar(out, rewriter);
      if (outNew.has_value()) {
        outputsNew.push_back(*outNew);
        isConverted = true;
      } else {
        outputsNew.push_back(out);
      }
    }

    if (!isConverted)
      return failure();

    IRMapping mapper;
    mapper.map(op.getInputs(), ValueRange(inputsNew));
    mapper.map(op.getOutputs(), ValueRange(outputsNew));

    Operation *clonedOp = rewriter.clone(*op, mapper);
    rewriter.replaceOp(op, clonedOp);
    return success();
  }
};

/// Convert linalg.broadcast to linalg.fill if input operand only has one elem.
struct NormalizeScalarLikeTensorLinalgBrcOp
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto optInpNew = singleElemDenseTensorToScalar(op.getInput(), rewriter);
    if (!optInpNew.has_value())
      return failure();

    auto fillOp = rewriter.create<linalg::FillOp>(
        op->getLoc(), ValueRange(*optInpNew), op.getInit());
    rewriter.replaceOp(op, fillOp);
    return success();
  }
};

/// Convert linalg.broadcast to linalg.fill if input operand only has one elem.
/// necessary normalization to break cycle on infinite loop of propagate reshape
/// pass.
struct NormalizeScalarLikeTensorLinalgBrcOpNonDense
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput().getDefiningOp<arith::ConstantOp>())
      return failure();
    auto inputShape = op.getInput().getType().getShape();
    if (ShapedType::isDynamicShape(inputShape))
      return failure();
    if (llvm::any_of(inputShape, [](auto &val) { return val != 1; }))
      return failure();
    SmallVector<Value> indices;
    indices.resize(
        inputShape.size(),
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0).getResult());
    auto extractOp = rewriter.create<tensor::ExtractOp>(op->getLoc(),
                                                        op.getInput(), indices);
    auto fillOp = rewriter.create<linalg::FillOp>(
        op->getLoc(), extractOp.getResult(), op.getInit());
    rewriter.replaceOp(op, fillOp);
    return success();
  }
};

/// normalize i8/i32 CompareOp
///   i8 -> f16
///   i32 -> i64 (except vne and veq)
/// e.g.
///   hfusion.compare ins(%src1, %src2 : tensor<6x6xi32>, tensor<6x6xi32>)
/// is normalized to
///   %cast1 = hfusion.cast %src1 : tensor<6x6xi32> to tensor<6x6xi64>
///   %cast2 = hfusion.cast %src2 : tensor<6x6xi32> to tensor<6x6xi64>
///   hfusion.compare ins(%cast1, %cast2 : tensor<6x6xi64>, tensor<6x6xi64>)

void populateNormalizeScalarLikeHFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeScalarLikeTensorOp<hfusion::ElemwiseUnaryOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<hfusion::ElemwiseBinaryOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<hfusion::CompareOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<hfusion::SelectOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<hfusion::CastOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<linalg::ElemwiseUnaryOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<linalg::ElemwiseBinaryOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorLinalgBrcOp>(patterns.getContext());
}

void populateNormalizeFinalScalarPatterns(RewritePatternSet &patterns) {
  if (archIsRegbased)
    patterns.add<NormalizeScalarLikeTensorLinalgBrcOpNonDense>(
        patterns.getContext());
}
} // namespace mlir::hfusion
