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
#include "bishengir/Transforms/Normalize/Utils/Kinds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/APFloat.h"

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
        rewriter, loc, square, square,
        utils::createEmptyOp(rewriter, loc, square), denomCoeff);

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

/// Shared normalize template for base-changing logarithms.
/// Normalize logb(x) to ln(x) / ln(b) when log base b is not e
/// eg.
/// y = hfusion elemwise unary {log2} (x)
///  is normalized to
///  y = linalg.elemwise_unary {log}(x) / linalg.elemwise_unary {log}(2)
template <typename LogLikeOpType, typename Traits>
struct NormalizeLogLikeOpTemplate : public OpRewritePattern<LogLikeOpType> {
public:
  using OpRewritePattern<LogLikeOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(LogLikeOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeLogLike(op))
      return failure();

    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    Value input = dpsOp.getDpsInputs()[0];
    Value dst = dpsOp.getDpsInits()[0];
    Type inputElemType = getElementTypeOrSelf(input.getType());
    if (!inputElemType.isF16() && !inputElemType.isF32())
      return failure();

    Location loc = op->getLoc();
    if (inputElemType.isF16())
      input = Traits::createCastOp(rewriter, loc, input, rewriter.getF32Type(),
                                   CastRoundKind::Round);

    Value result = logBaseChange(rewriter, loc, op, input);

    if (inputElemType.isF16())
      result = Traits::castBackLogLikeF16Result(rewriter, loc, result, dst);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  Value logBaseChange(PatternRewriter &rewriter, Location loc,
                      LogLikeOpType op, Value input) const {
    Value lnInit = utils::createEmptyOp(rewriter, loc, input);
    Value outInit = utils::createEmptyOp(rewriter, loc, input);
    Value ln = Traits::createUnaryOp(rewriter, loc, input, lnInit,
                                     UnaryKind::Ln);

    Type elementType = getElementTypeOrSelf(input.getType());
    Value logBaseValue = rewriter.create<arith::ConstantOp>(
        loc, elementType,
        rewriter.getFloatAttr(elementType, Traits::getLogBase(op)));

    Value logBaseTensor =
        Traits::createFillOp(rewriter, loc, logBaseValue, lnInit);
    Value lnBase = Traits::createUnaryOp(rewriter, loc, logBaseTensor, lnInit,
                                         UnaryKind::Ln);
    return Traits::createBinaryOp(rewriter, loc, ln, lnBase, outInit,
                                  BinaryKind::Div);
  }
};

/// Shared normalize template for `log1p(x)`.
/// Normalize vlog1p(x) to vln(x + 1)
/// eg.
/// y = hivm.hir.vlog1p x
///  is normalized to
///  y = hivm.hir.vln (x + 1)
template <typename Log1pOpType, typename Traits>
struct NormalizeLog1pOpTemplate : public OpRewritePattern<Log1pOpType> {
public:
  using OpRewritePattern<Log1pOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(Log1pOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeLog1p(op))
      return failure();

    Location loc = op->getLoc();
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    Value input = dpsOp.getDpsInputs()[0];
    Type inputElemType = getElementTypeOrSelf(input.getType());
    if (!inputElemType.isF16() && !inputElemType.isF32())
      return failure();

    Type elementType = getElementTypeOrSelf(input.getType());
    Value plusValue = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1.0f));
    Value add =
        Traits::createBinaryOp(rewriter, loc, input, plusValue,
                               utils::createEmptyOp(rewriter, loc, input),
                               BinaryKind::Add);
    Value result = Traits::createUnaryOp(rewriter, loc, add,
                                         utils::createEmptyOp(rewriter, loc, add),
                                         UnaryKind::Ln);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Normalizes `expm1(x)` to `exp(x) - 1`.
template <typename ExpM1OpType, typename Traits>
struct NormalizeExpM1OpTemplate : public OpRewritePattern<ExpM1OpType> {
public:
  using OpRewritePattern<ExpM1OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpM1OpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeExpM1(op))
      return failure();

    Location loc = op->getLoc();
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    Value input = dpsOp.getDpsInputs()[0];
    Type inputType = getElementTypeOrSelf(input.getType());
    if (!inputType.isF16() && !inputType.isF32())
      return failure();

    Value output = dpsOp.getDpsInits()[0];
    if (inputType.isF16())
      input = Traits::createCastOp(rewriter, loc, input, rewriter.getF32Type(),
                                   CastRoundKind::Round);

    Value expInit = utils::createEmptyOp(rewriter, loc, input);
    Value exp = Traits::createUnaryOp(rewriter, loc, input, expInit,
                                      UnaryKind::Exp);

    Type elementType = getElementTypeOrSelf(input.getType());
    Value one = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1.0f));
    Value subInit = utils::createEmptyOp(rewriter, loc, input);
    Value result =
        Traits::createBinaryOp(rewriter, loc, exp, one, subInit,
                               BinaryKind::Sub);

    if (inputType.isF16())
      result = Traits::createCastOp(rewriter, loc, result,
                                    getElementTypeOrSelf(output.getType()),
                                    CastRoundKind::Round);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Normalizes `ilogb(x)` to `floor(log2(abs(x)))`.
template <typename IlogbOpType, typename Traits>
struct NormalizeIlogbOpTemplate : public OpRewritePattern<IlogbOpType> {
public:
  using OpRewritePattern<IlogbOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(IlogbOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeIlogb(op))
      return failure();

    Location loc = op->getLoc();
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    Value input = dpsOp.getDpsInputs()[0];
    Type inputType = getElementTypeOrSelf(input.getType());
    if (!inputType.isF16() && !inputType.isF32())
      return failure();

    Value absInit = utils::createEmptyOp(rewriter, loc, input);
    Value abs =
        Traits::createUnaryOp(rewriter, loc, input, absInit, UnaryKind::Abs);

    Value log2Init = utils::createEmptyOp(rewriter, loc, input);
    Value log2 =
        Traits::createUnaryOp(rewriter, loc, abs, log2Init, UnaryKind::Log2);

    Value result = Traits::createIlogbResult(rewriter, loc, log2);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Template for normalizing mod operations across dialects.
template <typename SourceOp, typename Traits>
struct NormalizeModOpTemplate : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  /// Normalize the shared i8 path by widening both operands, computing the
  /// requested mod kind on the widened type, then casting the result back.
  static Value rewriteModType(PatternRewriter &rewriter, SourceOp op, Value x,
                              Value y, Type origType, Type castedType) {
    Location loc = op->getLoc();
    CastSignKind castSignKind = Traits::getCastSignKind(op);
    BinaryKind modKind = Traits::getModKind(op);
    Value dst = utils::createEmptyOpWithTargetElemType(rewriter, loc, x,
                                                       castedType);
    Value xCasted = Traits::createCastOp(rewriter, loc, x, castedType,
                                         CastRoundKind::Default, Value(),
                                         castSignKind);
    Value yCasted = Traits::createCastOp(rewriter, loc, y, castedType,
                                         CastRoundKind::Default, Value(),
                                         castSignKind);
    Value modResult =
        Traits::createBinaryOp(rewriter, loc, xCasted, yCasted, dst, modKind);
    return Traits::createCastOp(rewriter, loc, modResult, origType,
                                CastRoundKind::Default, Value(),
                                castSignKind);
  }

  static Value ensureRankedTensor(OpBuilder *rewriter, Location loc, Value val,
                                  Value shapeV) {
    Type ty = val.getType();
    if (isa<RankedTensorType>(ty))
      return val;

    auto ranked = dyn_cast<RankedTensorType>(shapeV.getType());
    if (!ranked)
      llvm_unreachable("reference tensor is not a ranked tensor");

    RankedTensorType resultTy =
        RankedTensorType::get(ranked.getShape(), val.getType());
    return rewriter->create<tensor::GenerateOp>(
        loc, resultTy, ValueRange{},
        [&](OpBuilder &b, Location genLoc, ValueRange indices) {
          b.create<tensor::YieldOp>(genLoc, val);
        });
  }

  /// Preserve the original HFusion behavior for floating mod: if the modulus
  /// is +/-inf, select NaN instead of the computed remainder.
  static Value handleInfinityModulus(PatternRewriter &rewriter, Location loc,
                                     Value y, Value result) {
    y = ensureRankedTensor(&rewriter, loc, y, result);

    Type elemType = getElementTypeOrSelf(result.getType());
    auto floatTy = cast<FloatType>(elemType);
    Value constNan = rewriter.create<arith::ConstantOp>(
        loc, elemType,
        rewriter.getFloatAttr(
            elemType, APFloat::getNaN(floatTy.getFloatSemantics())));
    Value yIsInf = Traits::createIsInfOp(rewriter, loc, y);
    Value selectDst = utils::createEmptyOp(rewriter, loc, result);
    return Traits::createSelectOp(rewriter, loc, yIsInf, constNan, result,
                                  selectDst);
  }

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalize(op))
      return failure();

    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dpsOp) {
      op->emitError("NormalizeModOpTemplate: operation does not implement "
                    "DestinationStyleOpInterface");
      return failure();
    }

    auto inputs = dpsOp.getDpsInputs();
    if (inputs.size() < 2) {
      op->emitError("NormalizeModOpTemplate: expected at least 2 inputs");
      return failure();
    }

    Value x = inputs[0];
    Value y = inputs[1];
    Value dst = dpsOp.getDpsInits()[0];
    Type elemType = getElementTypeOrSelf(x.getType());
    Location loc = op->getLoc();

    if (!Traits::isSupportedType(elemType))
      return failure();

    if (elemType.isInteger(1)) {
      auto constZero = utils::createConstantOp<bool>(rewriter, loc, elemType, 0);
      auto zeroDst =
          utils::createEmptyOpWithTargetElemType(rewriter, loc, dst, elemType);
      rewriter.replaceOp(op,
                         Traits::createFillOp(rewriter, loc, constZero, zeroDst));
      return success();
    }

    if (elemType.isInteger(8)) {
      rewriter.replaceOp(
          op, rewriteModType(rewriter, op, x, y, elemType, rewriter.getI16Type()));
      return success();
    }

    bool needsCast = elemType.isBF16() || elemType.isF16();
    Value xForDiv = x;
    Value yForDiv = y;
    if (needsCast) {
      Type f32Type = rewriter.getF32Type();
      xForDiv = Traits::createCastOp(rewriter, loc, x, f32Type,
                                     CastRoundKind::Round);
      yForDiv = Traits::createCastOp(rewriter, loc, y, f32Type,
                                     CastRoundKind::Round);
    }

    Value truncDiv = Traits::createDivOpForMod(rewriter, loc, xForDiv, yForDiv,
                                               elemType);
    Value mul = Traits::createBinaryOp(
        rewriter, loc, truncDiv, y, utils::createEmptyOp(rewriter, loc, x),
        BinaryKind::Mul);
    Value rem = Traits::createBinaryOp(
        rewriter, loc, x, mul, utils::createEmptyOp(rewriter, loc, x),
        BinaryKind::Sub);

    rewriter.replaceOp(op, handleInfinityModulus(rewriter, loc, y, rem));
    return success();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZEMATHTEMPLATE_H
