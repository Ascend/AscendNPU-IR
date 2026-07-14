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

#ifndef BISHENGIR_TRANSFORMS_REGBASE_NORMALIZE_NORMALIZEARITHMETICTEMPLATE_H
#define BISHENGIR_TRANSFORMS_REGBASE_NORMALIZE_NORMALIZEARITHMETICTEMPLATE_H

#include <optional>

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/RegBaseArchUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/regbase/Normalize/Utils/Kinds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {
template <typename Traits>
Value castToF32(PatternRewriter &rewriter, Location loc, Value input) {
  Type inputElemType = getElementTypeOrSelf(input.getType());
  CastRoundKind roundKind =
      inputElemType.isF16() || inputElemType.isBF16() ? CastRoundKind::RInt
                                                      : CastRoundKind::Trunc;
  return Traits::createCastOp(rewriter, loc, input, rewriter.getF32Type(),
                              roundKind);
}

template <typename Traits>
Value castThroughI32(PatternRewriter &rewriter, Location loc,
                            Value input, Type targetElemType,
                            CastRoundKind firstRoundKind) {
  Value resultI32 = Traits::createCastOp(rewriter, loc, input,
                                         rewriter.getI32Type(), firstRoundKind);
  return Traits::createCastOp(rewriter, loc, resultI32, targetElemType,
                              CastRoundKind::TruncWithOverflow);
}

template <typename Traits>
Value createConstantShift(PatternRewriter &rewriter, Location loc, Value input,
                          int64_t shiftValue, ShiftKind kind) {
  Type shiftType = getElementTypeOrSelf(input.getType());
  Value shift = rewriter.create<arith::ConstantOp>(
      loc, shiftType, rewriter.getIntegerAttr(shiftType, shiftValue));
  Value shiftInit = utils::createEmptyOp(rewriter, loc, input);
  return Traits::createShiftOp(rewriter, loc, input, shift, shiftInit, kind);
}

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

/// Normalizes integer `powi(x, y)` through `powf` on small integer types:
///
///   powi(x, y) => cast_back(powf(to_f32(x), to_f32(y)))
///
/// The floating-point `powf` is used as the implementation vehicle, while the
/// traits define the dialect-specific casts around it.
///
/// Example for `i8`:
///   powi(2_i8, 3_i8)
///     => powf(2.0_f32, 3.0_f32)
///     => 8.0_f32
///     => 8_i8
template <typename PowOpType, typename Traits>
struct NormalizeVPowiToPowfTemplate : public OpRewritePattern<PowOpType> {
public:
  using OpRewritePattern<PowOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeVPowi(op))
      return failure();

    auto inputs = op.getDpsInputs();
    auto inits = op.getDpsInits();
    if (inputs.size() != 2 || inits.size() != 1)
      return failure();

    Type lhsElemType = getElementTypeOrSelf(inputs[0].getType());
    Type rhsElemType = getElementTypeOrSelf(inputs[1].getType());
    Type resultElemType = getElementTypeOrSelf(inits[0].getType());
    if (lhsElemType != rhsElemType || lhsElemType != resultElemType)
      return failure();

    if (!lhsElemType.isInteger(8) && !lhsElemType.isInteger(16))
      return failure();

    Location loc = op.getLoc();
    Value lhsF32 = castSmallIntegerToF32(rewriter, loc, inputs[0]);
    Value rhsF32 = castSmallIntegerToF32(rewriter, loc, inputs[1]);
    Value powInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, inits[0], rewriter.getF32Type());
    Value powResult =
        Traits::createBinaryOp(rewriter, loc, lhsF32, rhsF32, powInit,
                               BinaryKind::Powf);
    Value castBack =
        castPowResultToType(rewriter, loc, powResult, resultElemType);
    rewriter.replaceOp(op, castBack);
    return success();
  }

private:
  static Value castSmallIntegerToF32(PatternRewriter &rewriter, Location loc,
                                     Value input) {
    Type inputElemType = getElementTypeOrSelf(input.getType());
    if (inputElemType.isInteger(8)) {
      Value inputF16 = Traits::createCastOp(rewriter, loc, input,
                                            rewriter.getF16Type(),
                                            CastRoundKind::RInt);
      return castToF32<Traits>(rewriter, loc, inputF16);
    }

    return castToF32<Traits>(rewriter, loc, input);
  }

  static Value castPowResultToType(PatternRewriter &rewriter, Location loc,
                                   Value input, Type targetElemType) {
    return castThroughI32<Traits>(rewriter, loc, input, targetElemType,
                                  CastRoundKind::Trunc);
  }
};

/// Normalizes `mulext(x, y)` by widening the operands, materializing the full
/// product, and then extracting the low and high halves.
///
/// For an input element type with bit width `N`:
///
///   x' = cast_{2N}(x)
///   y' = cast_{2N}(y)
///   p  = x' * y'
///   low  = trunc_N((p << N) >>s N)
///   high = trunc_N(p >>s N)
///
/// `>>s` is an arithmetic right shift. After the widening multiply, `high`
/// keeps bits `[2N-1:N]` and `low` keeps bits `[N-1:0]`.
///
/// Example for `i8`:
///   x = 20, y = 13
///   p = sext_i16(20) * sext_i16(13) = 260 = 0x0104
///   low  = trunc_i8(0x0104) = 0x04
///   high = trunc_i8(0x0104 >> 8) = 0x01
template <typename MulExtOpType, typename Traits>
struct NormalizeMulExtOpTemplate : public OpRewritePattern<MulExtOpType> {
public:
  using OpRewritePattern<MulExtOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulExtOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeMulExt(op))
      return failure();

    Value lhs = Traits::getMulExtLhs(op);
    Value rhs = Traits::getMulExtRhs(op);
    auto lhsType = getElementTypeOrSelf(lhs.getType());
    auto rhsType = getElementTypeOrSelf(rhs.getType());
    Type extendedType = Traits::getMulExtExtendedType(rewriter, lhsType);
    if (!extendedType || lhsType != rhsType)
      return failure();

    Location loc = op.getLoc();
    // Widen both operands before multiplying so the full 2N-bit product is
    // available for the low/high extraction below.
    Value lhsExt = Traits::createCastOp(rewriter, loc, lhs, extendedType,
                                        CastRoundKind::RInt);
    Value rhsExt = Traits::createCastOp(rewriter, loc, rhs, extendedType,
                                        CastRoundKind::RInt);

    Value mulInit = utils::createEmptyOp(rewriter, loc, lhsExt);
    Value mulRes = Traits::createBinaryOp(rewriter, loc, lhsExt, rhsExt, mulInit,
                                          BinaryKind::Mul);

    int64_t bitWidth = lhsType.getIntOrFloatBitWidth();

    // Signed right shift by N keeps the upper N bits of the 2N-bit product.
    Value high = createConstantShift<Traits>(rewriter, loc, mulRes, bitWidth,
                                             ShiftKind::RightSigned);

    // `(p << N) >> N` clears the upper half and leaves only the lower N bits in
    // the widened lane, ready for truncation back to the original type.
    Value lowShift = createConstantShift<Traits>(
        rewriter, loc, mulRes, bitWidth, ShiftKind::Left);
    Value low = createConstantShift<Traits>(rewriter, loc, lowShift, bitWidth,
                                            ShiftKind::RightSigned);

    Type resultType = lhsType;
    // Truncate each extracted half back to the original element type.
    Value lowTrunc = Traits::createCastOp(
        rewriter, loc, low, resultType, CastRoundKind::TruncWithOverflow);
    Value highTrunc = Traits::createCastOp(
        rewriter, loc, high, resultType, CastRoundKind::TruncWithOverflow);
    rewriter.replaceOp(op, {lowTrunc, highTrunc});
    return success();
  }
};

/// Normalizes `sub(s, v)` to `add(s, mul(v, -1))`.
/// e.g.
///   y = sub(s, x)
///   is normalized to
///   %mul = mul(x, -1)
///   y = add(%mul, s)
///
/// When the scalar constant is zero on Ascend950, the final add is skipped.
/// The operand order of the final add is controlled by Traits::kPlaceScalarFirstInFinalAdd.
template <typename SubOpType, typename Traits>
struct NormalizeSubVSToVMulAndVAddTemplate
    : public OpRewritePattern<SubOpType> {
public:
  using OpRewritePattern<SubOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeSubScalarVector(op))
      return failure();

    Value scalar = Traits::getSubScalar(op);
    Value vec = Traits::getSubVector(op);
    Type scalarType = scalar.getType();
    Location loc = op.getLoc();

    TypedAttr negOneAttr;
    if (auto intTy = dyn_cast<IntegerType>(scalarType)) {
      negOneAttr = rewriter.getIntegerAttr(intTy, -1);
    } else if (auto floatTy = dyn_cast<FloatType>(scalarType)) {
      negOneAttr = rewriter.getFloatAttr(floatTy, -1.0);
    } else {
      llvm_unreachable("unsupported scalar type");
    }
    Value negOne = rewriter.create<arith::ConstantOp>(loc, negOneAttr);

    Value mulInit = utils::createEmptyOp(rewriter, loc, vec);
    Value mulResult = Traits::createBinaryOp(rewriter, loc, vec, negOne,
                                             mulInit, BinaryKind::Mul);

    Value result = mulResult;
    if (!shouldSkipFinalAddForZeroScalarOnAscend950(op, scalar)) {
      Value addLhs = mulResult;
      Value addRhs = scalar;
      if constexpr (Traits::kPlaceScalarFirstInFinalAdd) {
        addLhs = scalar;
        addRhs = mulResult;
      }
      result = Traits::createBinaryOp(rewriter, loc, addLhs, addRhs,
                                      op.getDpsInits()[0], BinaryKind::Add);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static bool shouldSkipFinalAddForZeroScalarOnAscend950(SubOpType op,
                                                         Value scalar) {
    auto moduleOp = op->template getParentOfType<ModuleOp>();
    if (!moduleOp || !hacc::utils::isAscend950(moduleOp))
      return false;

    auto constOp = scalar.template getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return false;

    Attribute attr = constOp.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isZero();
    if (auto floatAttr = dyn_cast<FloatAttr>(attr))
      return floatAttr.getValue().isZero();
    return false;
  }
};

/// Normalizes integer `div(a, b)` / `div_unsigned(a, b)`.
/// Two rewrite paths are tried in order, controlled by the traits:
///   membase: casts integer operands to f32, divides in f32, truncates back.
///   regbase: widens i8 to i16, divides in i16, then casts back (with
///            overflow correction for -128/-1 on Ascend950).
template <typename DivOpType, typename Traits>
struct NormalizeDivSIandDivUIOpTemplate : public OpRewritePattern<DivOpType> {
public:
  using OpRewritePattern<DivOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(DivOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeDivSIandDivUIOp(op))
      return failure();

    if (succeeded(Traits::rewriteDivSIandDivUIOp(op, rewriter)))
      return success();

    if (succeeded(Traits::rewriteI8IntegerDivThroughI16(op, rewriter)))
      return success();

    return failure();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_REGBASE_NORMALIZE_NORMALIZEARITHMETICTEMPLATE_H
