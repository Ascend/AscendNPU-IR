//===-------- NormalizeArithmeticTemplate.h -------------------------------===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZEARITHMETICTEMPLATE_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZEARITHMETICTEMPLATE_H

#include <optional>

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Normalize/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {
/// Normalizes `rsqrt(x)` to `rec(sqrt(x))`
template <typename RSqrtOpType, typename Traits>
struct NormalizeRSqrtOpTemplate : public OpRewritePattern<RSqrtOpType> {
public:
  using OpRewritePattern<RSqrtOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(RSqrtOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeRSqrt(op))
      return failure();

    Location loc = op->getLoc();
    Value input = op.getDpsInputs()[0];
    Value sqrtInit = utils::createEmptyOp(rewriter, loc, input);
    Value sqrtValue = Traits::createUnaryOp(rewriter, loc, input, sqrtInit,
                                            UnaryKind::Sqrt);
    Value result = Traits::createUnaryOp(rewriter, loc, sqrtValue,
                                         op.getDpsInits()[0],
                                         UnaryKind::Rec);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Normalizes `mul(rec_like(x), y)` to `div(y, x)`
/// (1/b) * a -> a/b
/// a * (1/b) -> a/b
template <typename MulOpType, typename Traits>
struct NormalizeMulRecOpTemplate : public OpRewritePattern<MulOpType> {
public:
  using OpRewritePattern<MulOpType>::OpRewritePattern;
  using RecOpType = typename Traits::RecOpType;
  using DivOpType = typename Traits::DivOpType;

  LogicalResult matchAndRewrite(MulOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeMulRec(op))
      return failure();

    Value mulLhs = op.getDpsInputs()[0];
    Value mulRhs = op.getDpsInputs()[1];
    Location loc = op->getLoc();

    // (1/b) * a or rec(b) * a -> a/b
    if (std::optional<Value> denominator = matchRecLike(mulLhs)) {
      Value divInit = utils::createEmptyOp(rewriter, loc, *denominator);
      Value result = Traits::createBinaryOp(rewriter, loc, mulRhs, *denominator, divInit, BinaryKind::Div);
      rewriter.replaceOp(op, result);
      return success();
    }

    // a * (1/b) or a * rec(b) -> a/b
    if (std::optional<Value> denominator = matchRecLike(mulRhs)) {
      Value divInit = utils::createEmptyOp(rewriter, loc, *denominator);
      Value result = Traits::createBinaryOp(rewriter, loc, mulLhs, *denominator, divInit, BinaryKind::Div);
      rewriter.replaceOp(op, result);
      return success();
    }

    return failure();
  }

private:
  // Returns true if the given attribute represents a constant value of 1.
  static bool isConstantOne(Attribute value) {
    if (auto constFloatAttr = dyn_cast<FloatAttr>(value)) {
      llvm::APFloat floatOne(constFloatAttr.getValue().getSemantics(), 1);
      return constFloatAttr.getValue() == floatOne;
    }

    if (auto constIntAttr = dyn_cast<IntegerAttr>(value))
      return constIntAttr.getInt() == 1;

    return false;
  }

  // Returns whether the given value is rec-like: RecOp or DivOp with numerator of constant one.
  static std::optional<Value> matchRecLike(Value value) {
    Operation *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;

    if (Traits::matchOp(op, UnaryKind::Rec))
      return cast<RecOpType>(op).getDpsInputs()[0];

    if (!Traits::matchOp(op, BinaryKind::Div))
      return std::nullopt;

    auto divOp = cast<DivOpType>(op);
    Value numerator = divOp.getDpsInputs()[0];
    auto lhsConstOp = dyn_cast_or_null<arith::ConstantOp>(
        numerator.getDefiningOp());
    if (!lhsConstOp || !isConstantOne(lhsConstOp.getValue()))
      return std::nullopt;

    return divOp.getDpsInputs()[1];
  }
};

/// Normalizes `div(1, x)` to `rec(x)`.
template <typename DivOpType, typename Traits>
struct NormalizeDivVSToRecTemplate : public OpRewritePattern<DivOpType> {
public:
  using OpRewritePattern<DivOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(DivOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeDiv(op))
      return failure();

    auto inputs = op.getDpsInputs();

    Type numeratorType = inputs[0].getType();
    if (!numeratorType.isIntOrFloat())
      return failure();

    Type elemType = getElementTypeOrSelf(numeratorType);
    if (elemType.isBF16() || elemType.isF32())
      // rec accuracy is not enough for f32, and bf16 will be cast to f32 finally.
      return failure();

    auto numeratorConstOp =
        dyn_cast_or_null<arith::ConstantOp>(inputs[0].getDefiningOp());
    if (!numeratorConstOp)
      return failure();

    auto constFloatAttr = dyn_cast<FloatAttr>(numeratorConstOp.getValue());
    if (!constFloatAttr)
      return failure();

    llvm::APFloat oneFloat(constFloatAttr.getValue().getSemantics(), 1);
    if (constFloatAttr.getValue() != oneFloat)
      return failure();

    Value result = Traits::createUnaryOp(rewriter, op->getLoc(), inputs[1],
                                         op.getDpsInits()[0], UnaryKind::Rec);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZEARITHMETICTEMPLATE_H
