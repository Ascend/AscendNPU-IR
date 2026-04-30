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
#include "bishengir/Transforms/Normalize/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/IR/TypeUtilities.h"

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
