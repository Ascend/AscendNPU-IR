//===- NormalizeComparisonTemplate.h ---------------------------*- C++ -*-===//
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
//
// This file defines templates for normalizing comparison operations.
// All compare op templates should be placed here.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZECOMPARISONTEMPLATE_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZECOMPARISONTEMPLATE_H

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Normalize/Utils/Kinds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir {

/// Template for normalizing cmp vne to not(cmp veq).
///
/// This template handles the common pattern where a "not equal" comparison
/// is transformed into a "not(equal)" comparison to handle NAN values correctly.
///
/// Example transformation:
///   y = cmp x, z {vne} -> i1
/// is normalized to:
///   tmp = cmp x, z {veq} -> i1
///   y = not tmp -> i1
///
/// The Traits class must provide:
///   - shouldNormalize(op): returns true if the op should be normalized
///   - createCmpOp(rewriter, loc, lhs, rhs, kind): creates comparison op
///   - createUnaryOp(rewriter, loc, input, output, kind): creates unary op
///
/// @tparam SourceOp The source operation type (e.g., hfusion::CompareOp or
/// hivm::VCmpOp)
/// @tparam Traits The traits class providing Dialect-specific implementations
template <typename SourceOp, typename Traits>
struct NormalizeCmpVneOpTemplate : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalize(op))
      return failure();

    auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dpsOp) {
      op->emitError("NormalizeCmpVneOpTemplate: operation does not implement "
                    "DestinationStyleOpInterface");
      return failure();
    }

    auto inputs = dpsOp.getDpsInputs();
    Value output = dpsOp.getDpsInits()[0];
    Location loc = op->getLoc();

    Value veqResult = Traits::createCmpOp(rewriter, loc, inputs[0], inputs[1],
                                          CompareKind::EQ);
    Value vnotResult = Traits::createUnaryOp(rewriter, loc, veqResult, output,
                                             UnaryKind::Not);

    rewriter.replaceOp(op, vnotResult);
    return success();
  }
};

/// Removes the redundant `overflow_mode` annotation attached to compare
/// results.
template <typename SourceOp, typename Traits>
struct NormalizeCmpOpTemplate : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeCmp(op))
      return failure();

    auto overflowModeAttr =
        utils::getAnnotateOpWithAttr(op->getResult(0), "overflow_mode");
    if (!overflowModeAttr.has_value())
      return failure();

    rewriter.eraseOp(overflowModeAttr.value());
    return success();
  }
};

/// Normalizes the specific compare-to-zero pattern to a cast-to-bool op.
///
/// Example:
///   scalar = const 0
///   zero = fill(scalar, dst) -> tensor<...xi8>
///   y = cmp x, zero {ne} -> tensor<...xi1>
/// is normalized to:
///   y = cast x -> tensor<...xi1>
///
/// This pattern accepts either `cmp(x, zero)` or `cmp(zero, x)`.
template <typename SourceOp, typename Traits>
struct NormalizeCmpToCastOpTemplate : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeCmpToCast(op))
      return failure();

    auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dpsOp) {
      op->emitError("NormalizeCmpToCastOpTemplate: operation does not "
                    "implement DestinationStyleOpInterface");
      return failure();
    }

    auto inputs = dpsOp.getDpsInputs();
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    bool lhsIsZero = Traits::isZeroTensorProducer(lhs);
    bool rhsIsZero = Traits::isZeroTensorProducer(rhs);
    if (!lhsIsZero && !rhsIsZero)
      return failure();

    Value inputToCast = lhsIsZero && !rhsIsZero ? rhs : lhs;
    Value output = dpsOp.getDpsInits()[0];
    if (!Traits::isSupportedCastToBool(getElementTypeOrSelf(inputToCast)))
      return failure();

    Value cast = Traits::createCastOp(rewriter, op->getLoc(), inputToCast,
                                      getElementTypeOrSelf(output.getType()),
                                      CastRoundKind::RInt, output);
    rewriter.replaceOp(op, cast);
    return success();
  }
};

/// Normalizes the compare op by casting inputs to a wider compare type first.
///
/// Branch 1: normalize `i8` compare to `f16` compare.
/// Example:
///   x0 = [1, -3, 7, 0] : tensor<4xi8>
///   x1 = [1,  2, 7, 9] : tensor<4xi8>
///   y = cmp x0, x1 {veq}
/// is normalized to:
///   x0_cast = cast x0 -> tensor<...xf16>
///   x1_cast = cast x1 -> tensor<...xf16>
///   y = cmp x0_cast, x1_cast {veq}
///
/// Branch 2: normalize ordered `i32` compare to `i64` compare.
/// Example:
///   x0 = [10, -5,  3, 8] : tensor<4xi32>
///   x1 = [12, -5, -1, 9] : tensor<4xi32>
///   y = cmp x0, x1 {vlt}
/// is normalized to:
///   x0_cast = cast x0 -> tensor<...xi64>
///   x1_cast = cast x1 -> tensor<...xi64>
///   y = cmp x0_cast, x1_cast {vlt}
template <typename SourceOp, typename Traits>
struct NormalizeI8I32CmpOpTemplate : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalize(op))
      return failure();

    auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dpsOp) {
      op->emitError("NormalizeI8I32CmpOpTemplate: operation does not "
                    "implement DestinationStyleOpInterface");
      return failure();
    }

    auto inputs = dpsOp.getDpsInputs();
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    Type lhsElemType = getElementTypeOrSelf(lhs.getType());
    Type rhsElemType = getElementTypeOrSelf(rhs.getType());
    if (lhsElemType != rhsElemType) {
      op->emitError("NormalizeI8I32CmpOpTemplate: lhs and rhs elemType mismatch");
      return failure();
    }

    Type targetType;
    if (lhsElemType.isInteger(8)) {
      targetType = rewriter.getF16Type();
    } else if (lhsElemType.isInteger(32) && !Traits::isEqOrNeCompare(op)) {
      targetType = rewriter.getI64Type();
    } else {
      return failure();
    }

    Value lhsInit =
        utils::createEmptyOpWithTargetElemType(rewriter, op->getLoc(), lhs,
                                               targetType);
    Value castLhs = Traits::createCastOp(rewriter, op->getLoc(), lhs,
                                         targetType, CastRoundKind::Default,
                                         lhsInit);
    Value rhsInit =
        utils::createEmptyOpWithTargetElemType(rewriter, op->getLoc(), rhs,
                                               targetType);
    Value castRhs = Traits::createCastOp(rewriter, op->getLoc(), rhs,
                                         targetType, CastRoundKind::Default,
                                         rhsInit);
    Value cmp =
        Traits::createCmpOp(rewriter, op->getLoc(), castLhs, castRhs, op);
    rewriter.replaceOp(op, cmp);
    return success();
  }
};

/// Normalizes xor to the equivalent and/or/not form.
///
/// Example:
///   y = xor(x0, x1)
/// is normalized to:
///   t0 = and(x0, x1)
///   t1 = not(t0)
///   t2 = or(x0, x1)
///   y = and(t1, t2)
template <typename SourceOp, typename Traits>
struct NormalizeXorOpTemplate : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeXor(op))
      return failure();

    auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dpsOp) {
      op->emitError("NormalizeXorOpTemplate: operation does not implement "
                    "DestinationStyleOpInterface");
      return failure();
    }

    Value lhs = dpsOp.getDpsInputs()[0];
    Value rhs = dpsOp.getDpsInputs()[1];
    Value output = dpsOp.getDpsInits()[0];
    Location loc = op->getLoc();
    Value orInit = utils::createEmptyOp(rewriter, loc, output);
    Value orValue =
        Traits::createBinaryOp(rewriter, loc, lhs, rhs, orInit, BinaryKind::Or);
    Value andInit = utils::createEmptyOp(rewriter, loc, output);
    Value andValue = Traits::createBinaryOp(rewriter, loc, lhs, rhs, andInit,
                                            BinaryKind::And);
    Value notValue = Traits::createUnaryOp(rewriter, loc, andValue, andValue,
                                           UnaryKind::Not);
    Value resultInit = utils::createEmptyOp(rewriter, loc, output);
    Value result = Traits::createBinaryOp(rewriter, loc, notValue, orValue,
                                          resultInit, BinaryKind::And);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Normalizes i8 shift to i16 shift + cast-back.
///
/// Example:
///   y = shift(x0, x1) with tensor<...xi8>
/// is normalized to:
///   x0_cast = cast x0 -> tensor<...xi16>
///   x1_cast = cast x1 -> tensor<...xi16>
///   t0 = shift(x0_cast, x1_cast) -> tensor<...xi16>
///   y = cast t0 -> tensor<...xi8>
template <typename SourceOp, typename Traits>
struct NormalizeShiftI8ToI16OpTemplate : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeShift(op))
      return failure();

    auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dpsOp) {
      op->emitError("NormalizeShiftI8ToI16OpTemplate: operation does not "
                    "implement DestinationStyleOpInterface");
      return failure();
    }

    auto inputs = dpsOp.getDpsInputs();
    Value input = inputs[0];
    Type inputElemType = getElementTypeOrSelf(input.getType());
    if (!inputElemType.isInteger(8))
      return failure();

    Location loc = op->getLoc();
    Type targetElemType = rewriter.getI16Type();
    CastSignKind castSignKind = Traits::getShiftCastSignKind(op);
    Value castInput = Traits::createCastOp(rewriter, loc, input, targetElemType,
                                           CastRoundKind::Default, Value(),
                                           castSignKind);
    Value castShift = Traits::createCastOp(rewriter, loc, inputs[1],
                                           targetElemType,
                                           CastRoundKind::Default, Value(),
                                           castSignKind);
    Value shiftedInit =
        utils::createEmptyOpWithTargetElemType(rewriter, loc, input,
                                               targetElemType);
    Value shifted = Traits::createShiftOp(rewriter, loc, castInput, castShift,
                                          shiftedInit, ShiftKind::RightSigned,
                                          op.getOperation());
    Value result = Traits::createCastOp(
        rewriter, loc, shifted, rewriter.getI8Type(),
        CastRoundKind::TruncWithOverflow, Value(), castSignKind);
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename Traits>
/// Returns the integer mask used to clear the sign bit of a floating-point
/// bit pattern. For example, the 16-bit mask is `0b0111111111111111`.
Value getSignMaskConstValue(PatternRewriter &rewriter, Location loc,
                            int bitwidth) {
  Type intType = rewriter.getIntegerType(bitwidth);
  int64_t maskValue = bitwidth == 32 ? 0x7FFFFFFF : 0x7FFF;
  return rewriter.create<arith::ConstantOp>(
      loc, intType, rewriter.getIntegerAttr(intType, maskValue));
}

template <typename Traits>
/// Returns the additive inverse of the IEEE `+inf` bit pattern interpreted as
/// an integer. Adding this value is equivalent to subtracting the `+inf`
/// encoding in the integer domain.
Value getComplementOfInfConstValue(PatternRewriter &rewriter, Location loc,
                                   int bitwidth) {
  Type intType = rewriter.getIntegerType(bitwidth);
  int64_t infValue = bitwidth == 32 ? -0x7F800000 : -0x7C00;
  return rewriter.create<arith::ConstantOp>(
      loc, intType, rewriter.getIntegerAttr(intType, infValue));
}

template <typename Traits>
/// Broadcasts a scalar constant to the same shape as `source`, but with the
/// requested destination element type.
Value materializeScalarToShape(PatternRewriter &rewriter, Location loc,
                               Value source, Type elemType, Value scalar) {
  Value init =
      utils::createEmptyOpWithTargetElemType(rewriter, loc, source, elemType);
  return Traits::createFillOp(rewriter, loc, scalar, init);
}

template <typename Traits>
/// Reinterprets `source` with a new element type while keeping the same shape.
Value createBitcastWithElemType(PatternRewriter &rewriter, Location loc,
                                Value source, Type targetElemType) {
  auto shapedType = cast<ShapedType>(source.getType());
  return Traits::createBitcastOp(rewriter, loc, shapedType.clone(targetElemType),
                                 source);
}

template <typename Traits>
/// Clears the sign bit of a 16-bit or 32-bit floating-point bit pattern by
/// bitcasting to the matching integer element type and applying an `and` with
/// the sign mask.
Value maskSignBit(PatternRewriter &rewriter, Location loc, Value input,
                  Value signMask = Value()) {
  Type elemType = getElementTypeOrSelf(input.getType());
  if (!signMask) {
    signMask = getSignMaskConstValue<Traits>(
        rewriter, loc, elemType.getIntOrFloatBitWidth());
  }
  Type castType = rewriter.getIntegerType(elemType.getIntOrFloatBitWidth());
  Value signMaskValue =
      materializeScalarToShape<Traits>(rewriter, loc, input, castType, signMask);
  Value intInput = createBitcastWithElemType<Traits>(rewriter, loc, input, castType);
  Value andInit = utils::createEmptyOp(rewriter, loc, intInput);
  return Traits::createBinaryOp(rewriter, loc, intInput, signMaskValue,
                                andInit, BinaryKind::And);
}

template <typename Traits>
/// Subtracts the `+inf` encoding in the integer domain by adding its negated
/// integer bit pattern, e.g. `add(input, -f16_inf_bits)`.
Value minusInfConstValue(PatternRewriter &rewriter, Location loc, Value input,
                         Value negInf = Value()) {
  if (!negInf) {
    Type elemType = getElementTypeOrSelf(input.getType());
    negInf = getComplementOfInfConstValue<Traits>(
        rewriter, loc, elemType.getIntOrFloatBitWidth());
  }
  Value addInit = utils::createEmptyOp(rewriter, loc, input);
  return Traits::createBinaryOp(rewriter, loc, input, negInf, addInit,
                                BinaryKind::Add);
}

template <typename SourceOp, typename Traits>
struct NormalizeIsInfOpTemplate : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalize(op))
      return failure();

    Value input = Traits::getInput(op);
    Value output = Traits::getOutput(op);
    Type elemType = getElementTypeOrSelf(input.getType());
    if (!Traits::isSupportedElementType(elemType))
      return failure();

    auto loc = op->getLoc();
    Type castType = rewriter.getIntegerType(elemType.getIntOrFloatBitWidth());
    // step 1: clear the sign bit so +inf and -inf share one representation.
    Value signMask = getSignMaskConstValue<Traits>(
        rewriter, loc, elemType.getIntOrFloatBitWidth());
    Value maskedSignValue = maskSignBit<Traits>(rewriter, loc, input, signMask);

    // step 2: shift the integer bit pattern by the encoding of `-inf`.
    Value negInf = getComplementOfInfConstValue<Traits>(
        rewriter, loc, elemType.getIntOrFloatBitWidth());
    Value minusInfValue =
        minusInfConstValue<Traits>(rewriter, loc, maskedSignValue, negInf);

    // step 3: map the exact-inf case to an integer mask in {0, 1}.
    // Reinterpret the shifted integer value as float and take abs. Exact
    // infinity becomes zero here, while finite inputs and NaNs stay non-zero.
    Value floatMinusInf =
        createBitcastWithElemType<Traits>(rewriter, loc, minusInfValue,
                                          elemType);
    Value absInit = utils::createEmptyOp(rewriter, loc, floatMinusInf);
    Value absValue = Traits::createUnaryOp(rewriter, loc, floatMinusInf, absInit,
                                           UnaryKind::Abs);

    Value intAbsValue =
        createBitcastWithElemType<Traits>(rewriter, loc, absValue, castType);
    Value posOne = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, 1));
    Value minInit = utils::createEmptyOp(rewriter, loc, intAbsValue);
    Value minValue = Traits::createBinaryOp(rewriter, loc, intAbsValue, posOne,
                                            minInit, BinaryKind::MinSigned);
    Value negOne = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, -1));
    Value mulValue = Traits::createBinaryOp(rewriter, loc, minValue, negOne,
                                            minValue, BinaryKind::Mul);
    Value addValue = Traits::createBinaryOp(rewriter, loc, mulValue, posOne,
                                            mulValue, BinaryKind::Add);

    // step 4: lower the integer mask to the final bool result.
    Value result =
        Traits::lowerIntMaskToBoolResult(rewriter, loc, addValue, output);
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename SourceOp, typename Traits>
struct NormalizeIsNanOpTemplate : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalize(op))
      return failure();

    Value input = Traits::getInput(op);
    Value output = Traits::getOutput(op);
    Type elemType = getElementTypeOrSelf(input.getType());
    if (!Traits::isSupportedElementType(elemType))
      return failure();

    auto loc = op->getLoc();
    Type castType = rewriter.getIntegerType(elemType.getIntOrFloatBitWidth());
    // step 1: clear the sign bit so the exponent test ignores +/-.
    Value signMask = getSignMaskConstValue<Traits>(
        rewriter, loc, elemType.getIntOrFloatBitWidth());
    Value maskedSignValue = maskSignBit<Traits>(rewriter, loc, input, signMask);

    // step 2: shift the integer bit pattern by the encoding of `-inf`.
    Value negInf = getComplementOfInfConstValue<Traits>(
        rewriter, loc, elemType.getIntOrFloatBitWidth());
    Value minusInfValue =
        minusInfConstValue<Traits>(rewriter, loc, maskedSignValue, negInf);

    // step 3: clamp the shifted value to [0, 1]; only NaN keeps a non-zero
    // mask after the min/max pair.
    Value posOne = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, 1));
    Value minValue = Traits::createBinaryOp(rewriter, loc, minusInfValue, posOne,
                                            minusInfValue, BinaryKind::MinSigned);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, 0));
    Value maxValue = Traits::createBinaryOp(rewriter, loc, minValue, zero,
                                            minValue, BinaryKind::MaxSigned);

    // step 4: lower the integer mask to the final bool result.
    Value result =
        Traits::lowerIntMaskToBoolResult(rewriter, loc, maxValue, output);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZECOMPARISONTEMPLATE_H
