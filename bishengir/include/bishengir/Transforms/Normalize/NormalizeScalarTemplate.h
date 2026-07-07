//===- NormalizeScalarTemplate.h ----------------------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZESCALARTEMPLATE_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZESCALARTEMPLATE_H

#include <optional>

#include "bishengir/Transforms/Normalize/Utils/ScalarTemplateHelpers.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
template <typename OpType, typename Traits>
struct NormalizeScalarLikeTensorOpTemplate;

template <typename OpType, typename Traits>
struct NormalizeScalarLikeTensorBrcOpTemplate;

template <typename OpType, typename Traits>
struct NormalizeScalarLikeTensorBrcOpNonDenseTemplate;

template <typename Traits, typename... Ops>
void addNormalizeScalarLikeTensorOpPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeScalarLikeTensorOpTemplate<Ops, Traits>...>(
      patterns.getContext());
}

template <typename Traits, typename... Ops>
void addNormalizeScalarLikeTensorBrcOpPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeScalarLikeTensorBrcOpTemplate<Ops, Traits>...>(
      patterns.getContext());
}

template <typename Traits, typename... Ops>
void addNormalizeScalarLikeTensorBrcOpNonDensePatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<NormalizeScalarLikeTensorBrcOpNonDenseTemplate<Ops, Traits>...>(
          patterns.getContext());
}

// Normalize scalar-like dense tensor constants used by regular DPS ops.
// Example:
//   %cst = arith.constant dense<5.000000e-01> : tensor<1xf32>
//   <op> ins(%arg0, %cst : tensor<1xf32>, tensor<1xf32>) ...
// can become:
//   %cst = arith.constant 5.000000e-01 : f32
//   <op> ins(%arg0, %cst : tensor<1xf32>, f32) ...
//
// Uses DestinationStyleOpInterface so it works for both HFusion and HIVM ops.
template <typename OpType, typename Traits>
struct NormalizeScalarLikeTensorOpTemplate : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dpsOp)
      return failure();

    bool isConverted = false;
    SmallVector<Value> inputsNew;
    for (auto [inputIdx, inp] : llvm::enumerate(dpsOp.getDpsInputs())) {
      if (Traits::shouldKeepInputShaped(op.getOperation(), inputIdx)) {
        inputsNew.push_back(inp);
        continue;
      }

      auto inpNew = singleElemDenseTensorToScalar(inp, rewriter);
      if (inpNew.has_value()) {
        inputsNew.push_back(*inpNew);
        isConverted = true;
      } else {
        inputsNew.push_back(inp);
      }
    }

    SmallVector<Value> outputsNew;
    for (auto out : dpsOp.getDpsInits()) {
      if (Traits::shouldKeepInitShaped()) {
        outputsNew.push_back(out);
        continue;
      }

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
    mapper.map(dpsOp.getDpsInputs(), ValueRange(inputsNew));
    mapper.map(dpsOp.getDpsInits(), ValueRange(outputsNew));

    Operation *clonedOp = rewriter.clone(*op, mapper);
    rewriter.replaceOp(op, clonedOp);
    return success();
  }
};

/// Normalize broadcast-like ops whose source is a dense scalar-like constant.
/// Example:
///   %cst = arith.constant dense<5.000000e-01> : tensor<1xf32>
///   linalg.broadcast ins(%cst : tensor<1xf32>) outs(%dst : tensor<4xf32>)
/// can become an equivalent fill/broadcast from the scalar constant:
///   %cst = arith.constant 5.000000e-01 : f32
///   linalg.fill ins(%cst : f32) outs(%dst : tensor<4xf32>)
///
/// The dialect traits decide whether that rebuilt op is linalg.fill
/// (HFusion) or hivm.hir.vbrc (HIVM).
template <typename OpType, typename Traits>
struct NormalizeScalarLikeTensorBrcOpTemplate : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dpsOp) {
      op->emitError("NormalizeScalarLikeTensorBrcOpTemplate: operation does "
                    "not implement DestinationStyleOpInterface");
      return failure();
    }

    auto inputs = dpsOp.getDpsInputs();
    auto inits = dpsOp.getDpsInits();
    if (inputs.empty() || inits.empty())
      return failure();

    std::optional<Value> optInpNew =
        Traits::allowMultiRankUnitDenseBroadcastInput()
            ? unitDenseTensorToScalar(inputs[0], rewriter)
            : singleElemDenseTensorToScalar(inputs[0], rewriter);
    if (!optInpNew.has_value())
      return failure();

    auto fillOp =
        Traits::createFillOp(rewriter, op->getLoc(), *optInpNew, inits[0]);
    rewriter.replaceOp(op, fillOp);
    return success();
  }
};

/// Normalize broadcast-like ops whose source is scalar-like but non-dense.
/// "NonDense" here means "not produced by arith.constant"; the value can still
/// have a tensor type such as tensor<1xf32> or tensor<1x1xf32>.
/// Example:
///   %src: tensor<1x1xf32>
/// can become:
///   %v = tensor.extract %src[%c0, %c0] : tensor<1x1xf32>
///   linalg.fill/hivm.hir.vbrc from %v
///
/// This is a necessary normalization to break reshape propagation cycles.
template <typename OpType, typename Traits>
struct NormalizeScalarLikeTensorBrcOpNonDenseTemplate : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dpsOp) {
      op->emitError("NormalizeScalarLikeTensorBrcOpNonDenseTemplate: "
                    "operation does not implement "
                    "DestinationStyleOpInterface");
      return failure();
    }

    auto inputs = dpsOp.getDpsInputs();
    auto inits = dpsOp.getDpsInits();
    if (inputs.empty() || inits.empty())
      return failure();

    if (inputs[0].template getDefiningOp<arith::ConstantOp>())
      return failure();
    auto shapedType = dyn_cast<ShapedType>(inputs[0].getType());
    if (!shapedType)
      return failure();
    auto inputShape = shapedType.getShape();
    if (ShapedType::isDynamicShape(inputShape))
      return failure();
    if (llvm::any_of(inputShape, [](auto &val) { return val != 1; }))
      return failure();
    SmallVector<Value> indices;
    indices.resize(
        inputShape.size(),
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0).getResult());
    auto extractOp = rewriter.create<tensor::ExtractOp>(op->getLoc(),
                                                        inputs[0], indices);
    auto fillOp = Traits::createFillOp(rewriter, op->getLoc(),
                                       extractOp.getResult(), inits[0]);
    rewriter.replaceOp(op, fillOp);
    return success();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZESCALARTEMPLATE_H
