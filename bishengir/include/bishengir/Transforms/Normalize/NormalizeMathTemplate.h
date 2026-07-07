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
#include "bishengir/Transforms/Normalize/Utils/MathTemplateHelpers.h"
#include "bishengir/Transforms/Normalize/Utils/ScalarTemplateHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorHandling.h"

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

    if (elemType.isFloat8E4M3FN() || elemType.isFloat8E5M2()) {
      rewriter.replaceOp(
          op, rewriteModType(rewriter, op, x, y, elemType, rewriter.getF32Type()));
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

/// Shared normalize template that rewrites `ldexp(x, y)` to `x * y`.
///
/// eg.
///   y = ldexp(x, y)
///   is normalized to
///   y = mul(x, y)
template <typename LdexpOpType, typename Traits>
struct NormalizeLdexpOpTemplate : public OpRewritePattern<LdexpOpType> {
public:
  using OpRewritePattern<LdexpOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(LdexpOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeLdexp(op))
      return failure();

    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    Value input = dpsOp.getDpsInputs()[0];
    Value exponent = dpsOp.getDpsInputs()[1];
#ifndef NDEBUG
    Type inputElemType = getElementTypeOrSelf(input.getType());
    if (!inputElemType.isF16() && !inputElemType.isF32())
      llvm::report_fatal_error("only support input Type is f16 or f32");
#endif
    Value result = Traits::createBinaryOp(
        rewriter, op.getLoc(), input, exponent,
        utils::createEmptyOp(rewriter, op.getLoc(), input), BinaryKind::Mul);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Shared normalize template for `powf(x, y)`.
///
/// Constant-exponent paths:
///   powf(base, 0)   -> fill(1)
///   powf(base, 0.5) -> sqrt(base)
///   powf(base, 2)   -> base * base
///   powf(base, 3)   -> base * base * base
///
/// Dynamic-exponent path lowers to exp + log with boundary-case selects.
template <typename PowfOpType, typename Traits>
struct NormalizePowfOpTemplate : public OpRewritePattern<PowfOpType> {
public:
  using OpRewritePattern<PowfOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(PowfOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizePowf(op))
      return failure();

    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    Value base = dpsOp.getDpsInputs()[0];
    Value exponent = dpsOp.getDpsInputs()[1];
    Type inputElemType = getElementTypeOrSelf(base.getType());
    if (!inputElemType.isF16() && !inputElemType.isF32())
      return failure();

    Location loc = op.getLoc();
    // Step 1a. Try to recover the exponent as a scalar constant even when the
    // input IR wrapped it in a broadcast-like container.
    // Example:
    //   - HFusion may feed `powf` with `linalg.fill(0.5)` or
    //     `hfusion.cast(linalg.fill(0.5))`
    //   - HIVM may feed `vpow` with `hivm.hir.vbrc(0.5)`
    arith::ConstantOp exponentConstOp =
        getPowfExponentScalarConstant(exponent, rewriter);

    // Step 1. Try the short constant-exponent rewrites first.
    // Example:
    //   powf(base, 0)   -> fill(1)
    //   powf(base, 0.5) -> sqrt(base)
    //   powf(base, 2)   -> base * base
    //   powf(base, 3)   -> base * base * base
    // If one of these cases matches, we can stop here and avoid building the
    // longer select/log/exp expansion below.
    if (Value result =
            normalizeConstantExponentPowf(rewriter, loc, base, exponentConstOp)) {
      rewriter.replaceOp(op, result);
      return success();
    }

    // Step 2. Normalize the exponent input shape before building the main
    // elementwise rewrite.
    // Example:
    //   base     : tensor<16xf32>
    //   exponent : dense<0.5> : tensor<1xf32>
    // The later abs/ln/mul/exp/select ops want to treat the exponent as a
    // normal elementwise tensor input, so we first materialize that single
    // scalar value into a tensor with the same shape as `base`.
    // This keeps the remaining path uniform instead of carrying a separate
    // "single-element dense tensor exponent" special case through every step.
    Value expTensor = materializeExponentTensor(rewriter, loc, base, exponent);

    // Step 3. Build the two runtime candidates used by the original HFusion
    // control flow, then select between them.
    //   (a) General magnitude path:
    //       exp(exponent * ln(abs(base)))
    //       Example: `powf(2, y)` -> `exp(y * ln(2))`
    //   (b) Negative-base integer-exponent path:
    //       sign_correction(exponent) * exp(exponent * ln(abs(base)))
    //       Example: powf(-2, 3) needs the same magnitude as powf(2, 3), but
    //       the final sign must stay negative because the exponent is odd.
    Value negativeCondition =
        isNegativeBaseAndIntegerExponent(rewriter, loc, base, expTensor);
    Value negativeResult =
        calculateNegativeBaseIntegerExponentPower(rewriter, loc, base, expTensor);
    Value positiveResult =
        calculatePositiveBaseOrNonIntegerExponentPower(rewriter, loc, base,
                                                       exponent);
    Value partialResult0 = createPowfSelect(rewriter, loc, negativeCondition,
                                            negativeResult, positiveResult);

    // Step 4. Apply the special boundary overrides on top of the main result.
    // These are runtime per-element tensor conditions, so the final result is
    // still assembled by chaining selects on top of the already-built main
    // result. The selects are layered in priority order:
    //   (a) abs(base) == 1 && abs(exponent) == inf -> 1
    //       Example: powf(-1, +inf) should return 1 instead of flowing through
    //       the generic log/exp path.
    //   (b) negative finite base with finite non-integer exponent -> nan
    //       Example: powf(-2, 0.5) should return nan.
    //   (c) exponent == 0 -> 1
    //       Example: powf(base, 0) must return 1 for every finite base.
    Value one = createFloatConstant(rewriter, loc, base, 1.0);
    Value boundaryForOne = isAbsOneAndExponentInf(rewriter, loc, base,
                                                  expTensor);
    Value partialResult1 =
        createPowfSelect(rewriter, loc, boundaryForOne, one, partialResult0);

    Value nan = createNanConstant(rewriter, loc, base);
    Value nanCondition = shouldYieldNan(rewriter, loc, base, expTensor);
    Value partialResult2 =
        createPowfSelect(rewriter, loc, nanCondition, nan, partialResult1);

    Value partialResult3 = createPowfSelect(
        rewriter, loc, isZeroExponent(rewriter, loc, exponent), one,
        partialResult2);

    rewriter.replaceOp(op, partialResult3);
    return success();
  }

private:
  /// Rewrites the small constant-exponent cases that do not need the full
  /// select-based powf expansion.
  static Value normalizeConstantExponentPowf(PatternRewriter &rewriter,
                                             Location loc, Value base,
                                             arith::ConstantOp exponentConstOp) {
    if (!exponentConstOp)
      return nullptr;

    auto constFloatAttr = dyn_cast<FloatAttr>(exponentConstOp.getValue());
    if (!constFloatAttr)
      return nullptr;

    APFloat exponentValue = constFloatAttr.getValue();
    Type resultElemType = getElementTypeOrSelf(base.getType());
    Value empty = utils::createEmptyOp(rewriter, loc, base);

    if (exponentValue.isZero()) {
      Value one = rewriter.create<arith::ConstantOp>(
          loc, resultElemType, rewriter.getFloatAttr(resultElemType, 1.0));
      return Traits::createFillOp(rewriter, loc, one, empty);
    }

    APFloat halfValue(exponentValue.getSemantics(), "5e-1");
    if (exponentValue.compare(halfValue) == APFloat::cmpEqual) {
      return Traits::createUnaryOp(rewriter, loc, base, empty, UnaryKind::Sqrt);
    }

    double roundedValue = std::round(exponentValue.convertToDouble());
    if (!exponentValue.isInteger() || roundedValue < 1.0 || roundedValue > 3.0)
      return nullptr;

    return calculatePower(rewriter, loc, base, static_cast<int>(roundedValue));
  }

  /// Builds `base ^ exponent` for small positive integer exponents by
  /// recursive repeated multiply.
  static Value calculatePower(PatternRewriter &rewriter, Location loc,
                              Value base, int exponent) {
    if (exponent <= 1)
      return base;
    Value partial = calculatePower(rewriter, loc, base, exponent - 1);
    return Traits::createBinaryOp(rewriter, loc, base, partial,
                                  utils::createEmptyOp(rewriter, loc, base),
                                  BinaryKind::Mul);
  }

  /// Broadcasts a single-element dense exponent into the shape of `base` so
  /// later elementwise ops can consume a tensor exponent.
  static Value materializeExponentTensor(PatternRewriter &rewriter, Location loc,
                                         Value base, Value exponent) {
    std::optional<Value> scalar = singleElemDenseTensorToScalar(exponent, rewriter);
    if (!scalar.has_value())
      return exponent;
    return Traits::createFillOp(
        rewriter, loc, scalar.value(),
        utils::createEmptyOp(rewriter, loc, base));
  }

  /// Tries to recover the exponent as a scalar constant op.
  /// There are two sources:
  ///   1. a dialect-specific wrapper form such as `linalg.fill(0.5)`,
  ///      `hfusion.cast(linalg.fill(0.5))`, or `hivm.hir.vbrc(0.5)`
  ///   2. a bare `arith.constant` that feeds `powf` directly
  /// Example for case 2:
  ///   `powf(base, %cst)` where `%cst = arith.constant 2.0 : f32`
  static arith::ConstantOp getPowfExponentScalarConstant(
      Value exponent, PatternRewriter &rewriter) {
    if (arith::ConstantOp constOp =
            Traits::getPowfExponentScalarConstant(exponent, rewriter)) {
      return materializeExponentConstant(constOp, exponent.getLoc(), rewriter,
                                        getElementTypeOrSelf(exponent.getType()));
    }

    if (auto constOp =
            dyn_cast_or_null<arith::ConstantOp>(exponent.getDefiningOp())) {
      return materializeExponentConstant(constOp, exponent.getLoc(), rewriter,
                                        getElementTypeOrSelf(exponent.getType()));
    }

    return nullptr;
  }

  /// Converts the recovered exponent constant into a float constant with the
  /// same element type as the exponent input.
  /// Integer constants are intentionally converted to floating-point here.
  /// Example:
  ///   `arith.constant 2 : i32` becomes floating `2.0` when the exponent input
  ///   type for `powf` is `f32`.
  /// The powf normalize logic only reasons about floating-point exponents, so
  /// once we recover a scalar constant we materialize it in the exponent's
  /// floating-point element type before matching the `0`, `0.5`, `1`, `2`,
  /// or `3` fast paths.
  static arith::ConstantOp materializeExponentConstant(
      arith::ConstantOp constOp, Location loc, PatternRewriter &rewriter,
      Type targetType) {
    if (!constOp)
      return nullptr;
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
      return rewriter.create<arith::ConstantOp>(
          loc, targetType,
          rewriter.getFloatAttr(targetType, floatAttr.getValue().convertToDouble()));
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return rewriter.create<arith::ConstantOp>(
          loc, targetType,
          rewriter.getFloatAttr(
              targetType,
              llvm::APIntOps::RoundAPIntToDouble(intAttr.getValue())));
    }
    return nullptr;
  }

  /// Creates a scalar floating-point constant matching `like`'s element type.
  static Value createFloatConstant(PatternRewriter &rewriter, Location loc,
                                   Value like, double value) {
    Type elemType = getElementTypeOrSelf(like.getType());
    return rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, value));
  }

  /// Creates a scalar NaN constant matching `like`'s element type.
  static Value createNanConstant(PatternRewriter &rewriter, Location loc,
                                 Value like) {
    Type elemType = getElementTypeOrSelf(like.getType());
    auto floatType = cast<FloatType>(elemType);
    return rewriter.create<arith::ConstantOp>(
        loc, elemType,
        rewriter.getFloatAttr(elemType,
                              APFloat::getNaN(floatType.getFloatSemantics())));
  }

  /// Creates the floating-point `inf` constant used by the powf boundary
  /// checks.
  static Value createPowfBoundaryInfConstant(PatternRewriter &rewriter,
                                             Location loc, Value like) {
    Type elemType = getElementTypeOrSelf(like.getType());
    if (elemType.isF16()) {
      return rewriter.create<arith::ConstantOp>(
          loc, elemType, rewriter.getFloatAttr(elemType, 0x7C00));
    }
    if (elemType.isF32()) {
      return rewriter.create<arith::ConstantOp>(
          loc, elemType, rewriter.getFloatAttr(elemType, 0x7F800000));
    }
    llvm_unreachable("powf only supports f16/f32");
  }

  /// Checks the `abs(base) == 1 && abs(exponent) == inf` boundary that should
  /// return `1`.
  static Value isAbsOneAndExponentInf(PatternRewriter &rewriter, Location loc,
                                      Value base, Value exponent) {
    Value absBase = Traits::createUnaryOp(
        rewriter, loc, base, utils::createEmptyOp(rewriter, loc, base),
        UnaryKind::Abs);
    Value one = rewriter.create<arith::ConstantOp>(
        loc, getElementTypeOrSelf(base.getType()),
        rewriter.getFloatAttr(getElementTypeOrSelf(base.getType()), 1.0));
    Value isAbsBaseOne =
        Traits::createCmpOp(rewriter, loc, absBase, one, CompareKind::EQ);

    Value absExponent = Traits::createUnaryOp(
        rewriter, loc, exponent, utils::createEmptyOp(rewriter, loc, exponent),
        UnaryKind::Abs);
    Value inf = createPowfBoundaryInfConstant(rewriter, loc, base);
    Value isAbsExponentInf =
        Traits::createCmpOp(rewriter, loc, absExponent, inf, CompareKind::EQ);
    return Traits::createBinaryOp(
        rewriter, loc, isAbsBaseOne, isAbsExponentInf,
        utils::createEmptyOp(rewriter, loc, isAbsBaseOne), BinaryKind::And);
  }

  /// Extracts the sign bit of `base` as an i1 predicate.
  static Value getSignBitAsPredicate(PatternRewriter &rewriter, Location loc,
                                     Value base) {
    Type elemType = getElementTypeOrSelf(base.getType());
    auto shapedType = cast<ShapedType>(base.getType());
    Type intType = rewriter.getIntegerType(elemType.getIntOrFloatBitWidth());
    Type bitcastType = shapedType.clone(intType);
    Value bitcast = Traits::createBitcastOp(rewriter, loc, bitcastType, base);
    Value shiftValue = rewriter.create<arith::ConstantOp>(
        loc, intType,
        rewriter.getIntegerAttr(intType, elemType.getIntOrFloatBitWidth() - 1));
    Value shifted = Traits::createShiftOp(
        rewriter, loc, bitcast, shiftValue,
        utils::createEmptyOp(rewriter, loc, bitcast),
        ShiftKind::RightSigned);
    Value negOne = rewriter.create<arith::ConstantOp>(
        loc, intType, rewriter.getIntegerAttr(intType, -1));
    return Traits::createCmpOp(rewriter, loc, shifted, negOne, CompareKind::EQ);
  }

  /// Checks whether the exponent is an integer by comparing it with floor(x).
  static Value isIntegerValue(PatternRewriter &rewriter, Location loc,
                              Value exponent) {
    Value floorValue = Traits::createFloorOp(
        rewriter, loc, exponent, utils::createEmptyOp(rewriter, loc, exponent));
    return Traits::createCmpOp(rewriter, loc, floorValue, exponent,
                               CompareKind::EQ);
  }

  /// Identifies the `base < 0 && exponent is integer` path that needs sign
  /// correction.
  static Value isNegativeBaseAndIntegerExponent(PatternRewriter &rewriter,
                                                Location loc, Value base,
                                                Value exponent) {
    Value isNegative = getSignBitAsPredicate(rewriter, loc, base);
    Value isInteger = isIntegerValue(rewriter, loc, exponent);
    return Traits::createBinaryOp(
        rewriter, loc, isNegative, isInteger,
        utils::createEmptyOp(rewriter, loc, isNegative), BinaryKind::And);
  }

  /// Builds the sign coefficient for the negative-base integer-exponent path.
  /// It works in three simple steps:
  ///   1. Take `abs(exponent)` so `-3` and `3` follow the same odd/even rule.
  ///   2. Compute `abs(exponent) mod 2` to ask whether the integer exponent is
  ///      odd (`1`) or even (`0`).
  ///   3. Map that parity to the final coefficient with `(-2 * parity) + 1`,
  ///      so odd -> `-1` and even -> `1`.
  /// Example:
  ///   exponent = 3 -> abs(3) = 3 -> 3 mod 2 = 1 -> (-2 * 1) + 1 = -1
  ///   exponent = 4 -> abs(4) = 4 -> 4 mod 2 = 0 -> (-2 * 0) + 1 = 1
  /// That coefficient is multiplied with the shared magnitude result so
  /// `powf(-2, 3)` stays negative while `powf(-2, 4)` becomes positive.
  static Value calculateOddIntegerCoefficient(PatternRewriter &rewriter,
                                              Location loc, Value exponent) {
    Type elemType = getElementTypeOrSelf(exponent.getType());
    Value positiveOne = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, 1.0));
    Value positiveTwo = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, 2.0));
    Value negativeTwo = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, -2.0));

    Value absExponent = Traits::createUnaryOp(
        rewriter, loc, exponent, utils::createEmptyOp(rewriter, loc, exponent),
        UnaryKind::Abs);
    Value mod = Traits::createBinaryOp(
        rewriter, loc, absExponent, positiveTwo,
        utils::createEmptyOp(rewriter, loc, exponent), BinaryKind::Mod);
    return Traits::computeParityCoefficient(
        rewriter, loc, mod, negativeTwo, positiveOne);
  }

  /// Builds the `base < 0 && exponent is integer` candidate:
  /// `sign_correction(exponent) * exp(exponent * ln(abs(base)))`.
  static Value calculateNegativeBaseIntegerExponentPower(
      PatternRewriter &rewriter, Location loc, Value base, Value exponent) {
    Value lnEmpty = utils::createEmptyOp(rewriter, loc, base);
    Value mulEmpty = utils::createEmptyOp(rewriter, loc, base);
    Value coefficient = calculateOddIntegerCoefficient(rewriter, loc, exponent);
    Value absEmpty = utils::createEmptyOp(rewriter, loc, base);
    Value absBase = Traits::createUnaryOp(rewriter, loc, base, absEmpty,
                                          UnaryKind::Abs);
    Value lnBase =
        Traits::createUnaryOp(rewriter, loc, absBase, lnEmpty, UnaryKind::Ln);
    Value mul =
        Traits::createBinaryOp(rewriter, loc, lnBase, exponent, mulEmpty,
                               BinaryKind::Mul);
    Value expEmpty = utils::createEmptyOp(rewriter, loc, base);
    Value magnitude =
        Traits::createUnaryOp(rewriter, loc, mul, expEmpty, UnaryKind::Exp);
    return Traits::createBinaryOp(
        rewriter, loc, magnitude, coefficient,
        utils::createEmptyOp(rewriter, loc, base), BinaryKind::Mul);
  }

  /// Builds the general candidate `exp(exponent * ln(abs(base)))`.
  static Value calculatePositiveBaseOrNonIntegerExponentPower(
      PatternRewriter &rewriter, Location loc, Value base, Value exponent) {
    Value lnEmpty = utils::createEmptyOp(rewriter, loc, base);
    Value mulEmpty = utils::createEmptyOp(rewriter, loc, base);
    Value resultEmpty = utils::createEmptyOp(rewriter, loc, base);
    Value absEmpty = utils::createEmptyOp(rewriter, loc, base);
    Value absBase = Traits::createUnaryOp(rewriter, loc, base, absEmpty,
                                          UnaryKind::Abs);
    Value lnBase =
        Traits::createUnaryOp(rewriter, loc, absBase, lnEmpty, UnaryKind::Ln);
    Value mul =
        Traits::createBinaryOp(rewriter, loc, lnBase, exponent, mulEmpty,
                               BinaryKind::Mul);
    return Traits::createUnaryOp(rewriter, loc, mul, resultEmpty,
                                 UnaryKind::Exp);
  }

  /// Builds one `select` node in the final boundary-resolution chain.
  static Value createPowfSelect(PatternRewriter &rewriter, Location loc,
                                Value predicate, Value trueValue,
                                Value falseValue) {
    Value selectInit = utils::createEmptyOp(rewriter, loc, falseValue);
    return Traits::createTernaryOp(rewriter, loc, predicate, trueValue,
                                   falseValue, selectInit,
                                   TernaryKind::Select);
  }

  /// Checks whether the input is finite by testing `abs(x) != inf`.
  static Value isFinite(PatternRewriter &rewriter, Location loc, Value input) {
    Value absInput = Traits::createUnaryOp(
        rewriter, loc, input, utils::createEmptyOp(rewriter, loc, input),
        UnaryKind::Abs);
    Value inf = rewriter.create<arith::ConstantOp>(
        loc, getElementTypeOrSelf(input.getType()),
        rewriter.getFloatAttr(getElementTypeOrSelf(input.getType()),
                              std::numeric_limits<double>::infinity()));
    Value isInf =
        Traits::createCmpOp(rewriter, loc, absInput, inf, CompareKind::EQ);
    return Traits::createUnaryOp(rewriter, loc, isInf,
                                 utils::createEmptyOp(rewriter, loc, isInf),
                                 UnaryKind::Not);
  }

  /// Checks the boundary that should produce NaN: negative finite base with a
  /// finite non-integer exponent.
  static Value shouldYieldNan(PatternRewriter &rewriter, Location loc,
                              Value base, Value exponent) {
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, getElementTypeOrSelf(base.getType()),
        rewriter.getFloatAttr(getElementTypeOrSelf(base.getType()), 0.0));
    Value isNegative =
        Traits::createCmpOp(rewriter, loc, base, zero, CompareKind::LT);
    Value baseFinite = isFinite(rewriter, loc, base);
    Value negativeFinite = Traits::createBinaryOp(
        rewriter, loc, isNegative, baseFinite,
        utils::createEmptyOp(rewriter, loc, isNegative), BinaryKind::And);

    Value exponentFinite = isFinite(rewriter, loc, exponent);
    Value exponentInteger = isIntegerValue(rewriter, loc, exponent);
    Value exponentNotInteger = Traits::createUnaryOp(
        rewriter, loc, exponentInteger,
        utils::createEmptyOp(rewriter, loc, exponentInteger), UnaryKind::Not);
    Value finiteNonInteger = Traits::createBinaryOp(
        rewriter, loc, exponentFinite, exponentNotInteger,
        utils::createEmptyOp(rewriter, loc, exponentFinite), BinaryKind::And);

    return Traits::createBinaryOp(
        rewriter, loc, negativeFinite, finiteNonInteger,
        utils::createEmptyOp(rewriter, loc, negativeFinite), BinaryKind::And);
  }

  /// Checks the final `exponent == 0` override.
  static Value isZeroExponent(PatternRewriter &rewriter, Location loc,
                              Value exponent) {
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, getElementTypeOrSelf(exponent.getType()),
        rewriter.getFloatAttr(getElementTypeOrSelf(exponent.getType()), 0.0));
    return Traits::createCmpOp(rewriter, loc, exponent, zero, CompareKind::EQ);
  }
};
/// Normalizes ceil/floor ops to dialect-specific cast with ceil/floor round
/// mode.
/// On non-Ascend950 platforms, f16/bf16 same-type inputs require an intermediate
/// f32 cast chain because the hardware only supports ceil/floor on f32:
///   f16/bf16 -> f32(rint) -> f32(ceil/floor) -> outType(ceil/floor)
///
/// eg. (non-950, f16 same-type input)
///   y = elemwise_unary {ceil} (x)   where x and dst are f16
///    is normalized to
///    tmp0 = cast(x, f32, rint)       // widen to f32
///    tmp1 = cast(tmp0, f32, ceil)    // compute ceil in f32
///    y   = cast(tmp1, f16, ceil)     // narrow back to dst type
template <typename OpType, typename Traits>
struct NormalizeCeilandFloorOpTemplate : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeCeilandFloor(op))
      return failure();

    CastRoundKind roundKind = Traits::getCeilOrFloorRoundKind(op);
    Location loc = op->getLoc();
    Value src = Traits::getCeilandFloorInput(op);
    Value dst = Traits::getCeilandFloorOutput(op);
    Type inType = getElementTypeOrSelf(src.getType());
    Type outType = getElementTypeOrSelf(dst.getType());

    if (inType.isInteger())
      return failure();

    if (!Traits::archIsAscend950()) {
      if ((inType.isF16() || inType.isBF16()) && inType == outType) {
        src = Traits::createCastOp(rewriter, loc, src, rewriter.getF32Type(),
                                  CastRoundKind::RInt);
        src = Traits::createCastOp(rewriter, loc, src, rewriter.getF32Type(),
                                  roundKind);
      }
    }

    Value result = Traits::createCastOp(rewriter, loc, src, outType, roundKind,
                                        dst);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Normalizes maxnumf (minnumf) elemwise binary ops to maxf (minf) by masking
/// NaN inputs with +/-inf so the hardware maxf/minf op gives maxnumf/minnumf
/// semantics.
/// NaN is masked via: select(isnan(x), padding_inf, x)
///   maxnumf uses -inf as padding so NaN loses the comparison
///   minnumf uses +inf as padding so NaN loses the comparison
///
/// eg.
///   dst = elemwise_binary {maxnumf} (src0, src1)
///    is normalized to
///    new_src0 = select(isnan(src0), -inf, src0)
///    new_src1 = select(isnan(src1), -inf, src1)
///    dst = elemwise_binary {maxf} (new_src0, new_src1)
template <typename OpType, typename Traits>
struct NormalizeElemwiseMinMaxNumFOpTemplate : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeElemwiseMinMaxNumF(op))
      return failure();

    bool isMax = Traits::isMaxNumF(op);
    double paddingInfValue =
        isMax ? -std::numeric_limits<double>::infinity()
              : std::numeric_limits<double>::infinity();
    BinaryKind targetKind = isMax ? BinaryKind::Max : BinaryKind::Min;

    Location loc = op->getLoc();
    Value src0 = Traits::getMinMaxNumFInput0(op);
    Value src1 = Traits::getMinMaxNumFInput1(op);

    Value newSrc0 = maskNanWithInf<Traits>(rewriter, loc, src0, paddingInfValue);
    Value newSrc1 = maskNanWithInf<Traits>(rewriter, loc, src1, paddingInfValue);

    Value result = Traits::createBinaryOp(
        rewriter, loc, newSrc0, newSrc1,
        utils::createEmptyOp(rewriter, loc, src0), targetKind);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Normalizes reduction ops with maxnumf/minnumf body to maximumf/minimumf by
/// masking NaN in the source operand and replacing the body combiner op.
///
/// eg.
///   dst = reduce { maxnumf } src
///    is normalized to
///    new_src = select(isnan(src), -inf, src)
///    dst = reduce { maximumf } new_src
///
/// For minnumf reductions, NaN is masked with +inf and the body uses minimumf.
template <typename OpType, typename Traits>
struct NormalizeReduceMinMaxNumFOpTemplate : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeReduceMinMaxNumF(op))
      return failure();

    bool isMax = Traits::isReduceMaxNumF(op);
    double paddingInfValue =
        isMax ? -std::numeric_limits<double>::infinity()
              : std::numeric_limits<double>::infinity();

    Location loc = op->getLoc();
    Value src = Traits::getReduceMinMaxNumFInput(op);
    Value newSrc = maskNanWithInf<Traits>(rewriter, loc, src, paddingInfValue);

    Traits::updateReduceMinMaxNumFOp(op, rewriter, newSrc, isMax);
    return success();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZEMATHTEMPLATE_H
