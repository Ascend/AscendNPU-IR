//===- NormalizeMathTemplate.h ---------------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZEMATHTEMPLATE_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZEMATHTEMPLATE_H

#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include <cmath>

namespace mlir {

/// Shared normalize template that rewrites `2^x` to `exp(ln(2) * x)`.
template <typename OpType, typename Traits>
struct NormalizeExp2OpTemplate : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeExp2(op))
      return failure();

    Location loc = op->getLoc();
    Value src = op.getDpsInputs()[0];
    Value dst = op.getDpsInits()[0];
    Type inputElemType = getElementTypeOrSelf(src.getType());
    if (!inputElemType.isF16() && !inputElemType.isF32())
      return failure();

    if (inputElemType.isF16())
      src = Traits::createCastOp(rewriter, loc, src, rewriter.getF32Type(),
                                 CastRoundKind::Round);

    Type calcElemType = getElementTypeOrSelf(src.getType());
    Value ln2 = rewriter.create<arith::ConstantOp>(
        loc, calcElemType, rewriter.getFloatAttr(calcElemType, std::log(2.0)));
    Value mul =
        Traits::createBinaryOp(rewriter, loc, src, ln2,
                               utils::createEmptyOp(rewriter, loc, src),
                               BinaryKind::Mul);
    Value result =
        Traits::createUnaryOp(rewriter, loc, mul,
                              utils::createEmptyOp(rewriter, loc, mul),
                              UnaryKind::Exp);

    if (inputElemType.isF16())
      result = Traits::createCastOp(rewriter, loc, result,
                                    getElementTypeOrSelf(dst.getType()),
                                    CastRoundKind::Round);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Shared normalize template that rewrites `erf(x)` to a clipped rational
/// polynomial approximation.
template <typename OpType, typename Traits>
struct NormalizeErfOpTemplate : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeErf(op))
      return failure();

    Location loc = op->getLoc();
    Value src = op.getDpsInputs()[0];
    Value dst = op.getDpsInits()[0];
    Type inputElemType = getElementTypeOrSelf(src.getType());
    if (!inputElemType.isF16() && !inputElemType.isF32())
      return failure();

    if (inputElemType.isF16())
      src = Traits::createCastOp(rewriter, loc, src, rewriter.getF32Type(),
                                 CastRoundKind::Round);

    auto createConst = [&](double value) -> Value {
      Type elementType = getElementTypeOrSelf(src.getType());
      return rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getFloatAttr(elementType, value));
    };

    // Step 1: clip the input into the approximation range [-3.92, 3.92].
    Value lower = createConst(-3.92);
    Value upper = createConst(3.92);
    Value clipInit = utils::createEmptyOp(rewriter, loc, src);
    Value minValue =
        Traits::createBinaryOp(rewriter, loc, src, upper, clipInit,
                               BinaryKind::Min);
    Value clippedInput =
        Traits::createBinaryOp(rewriter, loc, minValue, lower, clipInit,
                               BinaryKind::Max);

    // Step 2: compute y = x^2. Both numerator and denominator polynomials are
    // expanded on y instead of directly on x.
    Value square = Traits::createBinaryOp(
        rewriter, loc, clippedInput, clippedInput,
        utils::createEmptyOp(rewriter, loc, clippedInput), BinaryKind::Mul);

    // Step 3: build the numerator polynomial and multiply it by x so the final
    // numerator keeps the original odd-function structure.
    Value numerInit = utils::createEmptyOp(rewriter, loc, clippedInput);
    Value numerSeed = Traits::createBinaryOp(
        rewriter, loc, square, createConst(0.53443748819e-1),
        numerInit, BinaryKind::Mul);

    constexpr double numerCoeff[] = {0.75517016694e1, 0.10162808918e3,
                                     0.13938061484e4, 0.50637915060e4,
                                     0.29639384698e5};
    Value numerPoly = genPolyExpr(rewriter, loc, square, numerSeed,
                                  utils::createEmptyOp(rewriter, loc, numerSeed),
                                  numerCoeff);
    Value numer = Traits::createBinaryOp(
        rewriter, loc, clippedInput, numerPoly, numerInit, BinaryKind::Mul);

    // Step 4: build the denominator polynomial on the same y = x^2 input.
    constexpr double denomCoeff[] = {0.31212858877e2, 0.39856963806e3,
                                     0.30231248150e4, 0.13243365831e5,
                                     0.26267224157e5};
    Value denom = genPolyExpr(
        rewriter, loc, square, square, utils::createEmptyOp(rewriter, loc, square),
        denomCoeff);

    // Step 5: finish the rational approximation numer / denom.
    Value result =
        Traits::createBinaryOp(rewriter, loc, numer, denom,
                               utils::createEmptyOp(rewriter, loc, numer),
                               BinaryKind::Div);

    if (inputElemType.isF16())
      result = Traits::createCastOp(rewriter, loc, result,
                                    getElementTypeOrSelf(dst.getType()),
                                    CastRoundKind::Round);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static Value genPolyExpr(PatternRewriter &rewriter, Location loc,
                           Value squareSrc, Value input, Value tempInit,
                           ArrayRef<double> coeffs) {
    // Materialize each scalar coefficient in the same compute type as y = x^2.
    auto createConst = [&](double value) -> Value {
      Type elementType = getElementTypeOrSelf(squareSrc.getType());
      return rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getFloatAttr(elementType, value));
    };

    // Expand the polynomial one coefficient at a time on y = x^2:
    //   result = (((input + c0) * y + c1) * y + ...) * y + cn
    Value result = input;
    for (auto [idx, coeff] : llvm::enumerate(coeffs)) {
      Value add = Traits::createBinaryOp(
          rewriter, loc, result, createConst(coeff), tempInit, BinaryKind::Add);
      if (idx + 1 == coeffs.size())
        result = add;
      else
        result = Traits::createBinaryOp(rewriter, loc, add, squareSrc, tempInit,
                                        BinaryKind::Mul);
    }
    return result;
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZEMATHTEMPLATE_H
