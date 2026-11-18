//===- NormalizeMath.cpp -----------------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeUtils.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Transforms/regbase/Normalize/NormalizeMathTemplate.h"
#include "bishengir/Transforms/regbase/Normalize/Utils/ScalarTemplateHelpers.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::hfusion {

/// normalize logb(x) to ln(x) / ln(b) when log base b is not e
/// eg.
/// y = hfusion elemwise unary {log2} (x)
///  is normalized to
///  y = linalg.elemwise_unary {log}(x) / linalg.elemwise_unary {log}(2)
struct HFusionNormalizeLogLikeTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeLogLike(hfusion::ElemwiseUnaryOp op) {
    if (!op.hasPureTensorSemantics())
      return false;
    return op.getFun() == hfusion::UnaryFn::log2 ||
           op.getFun() == hfusion::UnaryFn::log10;
  }

  static float getLogBase(hfusion::ElemwiseUnaryOp op) {
    if (op.getFun() == hfusion::UnaryFn::log2)
      return 2.0f;
    if (op.getFun() == hfusion::UnaryFn::log10)
      return 10.0f;
    llvm::report_fatal_error("unsupport log op");
  }

  static Value castBackLogLikeF16Result(PatternRewriter &rewriter, Location loc,
                                        Value result, Value dst) {
    auto roundingAttr =
        rewriter.getAttr<hfusion::RoundModeAttr>(hfusion::RoundMode::RINT);
    auto modeAttr = rewriter.getNamedAttr(
        hfusion::RoundModeAttr::getMnemonic(), roundingAttr);
    return rewriter
        .create<hfusion::CastOp>(loc, TypeRange(dst.getType()), ValueRange(result),
                                 ValueRange(dst), modeAttr)
        .getResult(0);
  }
};

/// normalize log1p(x) to ln(x + 1)
/// eg.
/// y = hfusion elemwise unary {log1p} (x)
///  is normalized to
///  y = linalg.elemwise_unary {log}(x + 1)
struct HFusionNormalizeLog1pTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeLog1p(hfusion::ElemwiseUnaryOp op) {
    return op.hasPureTensorSemantics() &&
           op.getFun() == hfusion::UnaryFn::log1p;
  }
};

using NormalizeLogLikeOpRegBase =
    mlir::NormalizeLogLikeOpTemplate<hfusion::ElemwiseUnaryOp,
                                     HFusionNormalizeLogLikeTraits>;
using NormalizeLog1pOpRegBase =
    mlir::NormalizeLog1pOpTemplate<hfusion::ElemwiseUnaryOp,
                                   HFusionNormalizeLog1pTraits>;
///  normalize mod op to rec op
///   z = x % y
///  is normalized to
///   rem = x - truncate_div(x, y) * y
///  e.g.
///   41 % 20 = 1; 41 % (-20) = -19; (-72) % 8 = 0
///  fp16/bf16 type needs to convert to fp32 to calculate for higher
///  accuracy
struct HFusionModTraits : public NormalizeTraitsBase {
  static bool isSupportedType(Type elemType) {
    if (elemType.isInteger(1) || elemType.isInteger(8))
      return true;

    if (elemType.isInteger())
      return false;

    return true;
  }

  static bool shouldNormalize(ElemwiseBinaryOp op) {
    if (!op.hasPureTensorSemantics())
      return false;
    auto fun = op.getFun();
    return fun == hfusion::BinaryFn::mod || fun == hfusion::BinaryFn::modui;
  }

  static CastSignKind getCastSignKind(ElemwiseBinaryOp op) {
    return op.getFun() == hfusion::BinaryFn::modui ? CastSignKind::Unsigned
                                                   : CastSignKind::Signed;
  }

  static BinaryKind getModKind(ElemwiseBinaryOp op) {
    return op.getFun() == hfusion::BinaryFn::modui ? BinaryKind::ModUnsigned
                                                   : BinaryKind::Mod;
  }

  static Value createDivOpForMod(PatternRewriter &rewriter, Location loc,
                                 Value x, Value y, Type elemType) {
    if (elemType.isInteger())
      return createBinaryOp(rewriter, loc, x, y,
                            utils::createEmptyOp(rewriter, loc, x),
                            BinaryKind::Div);
    auto divOp = hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp,
                                         hfusion::BinaryFn,
                                         hfusion::BinaryFnAttr>(
                     rewriter, loc, hfusion::BinaryFn::divfhp,
                     mlir::ValueRange{x, y},
                     utils::createEmptyOp(rewriter, loc, x))
                     ->getResults()[0];
    return hfusion::castTo(rewriter, divOp, elemType, hfusion::RoundMode::TRUNC);
  }

  /// Returns true when `value` is consumed as a 16-bit (f16/bf16) result.
  ///
  /// Triton lowers f16/bf16 `%` to `extf -> arith.remf (f32) -> truncf`, so the
  /// mod looks like an f32 op even though the user-visible result is 16-bit.
  /// The 16-bit test/reference computes `x - trunc(x/y) * y` in f32 and expects
  /// the rounded-product result, while the FMA path removes that rounding (up
  /// to half an ulp of the product). Keep the legacy `mul + sub` path whenever
  /// the result is consumed as f16/bf16.
  ///
  /// The check walks through shape-only tensor ops so that a cast followed by
  /// e.g. `tensor.extract_slice` cannot hide the 16-bit consumer.
  static bool isConsumedAs16Bit(Value value) {
    SmallVector<Value, 4> worklist{value};
    SmallPtrSet<Operation *, 8> visited;
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      for (Operation *user : current.getUsers()) {
        if (!visited.insert(user).second)
          continue;
        if (auto castOp = dyn_cast<hfusion::CastOp>(user)) {
          Type outElemType =
              getElementTypeOrSelf(castOp->getResult(0).getType());
          if (outElemType.isF16() || outElemType.isBF16())
            return true;
          continue;
        }
        if (isa<tensor::ExtractSliceOp, tensor::ExpandShapeOp,
                tensor::CollapseShapeOp, tensor::CastOp>(user)) {
          for (Value result : user->getResults())
            worklist.push_back(result);
        }
      }
    }
    return false;
  }

  static bool shouldPreserveLowPrecision(ElemwiseBinaryOp op) {
    return op.getNumResults() > 0 && isConsumedAs16Bit(op.getResult(0));
  }

  /// Compute the remainder residual `x - truncDiv * y`.
  ///
  /// On Ascend950 f32, use a fused multiply-add so that the intermediate
  /// product is not rounded separately. The naive `x - truncDiv * y`
  /// decomposition loses up to half an ulp of the product; when the exact
  /// remainder is close to zero that error dominates the result and breaks
  /// the `atol=1e-5` contract of ops such as aten.remainder.
  ///
  /// The FMA path is only used when the truncated quotient is within FP32's
  /// exact-integer range (plus one binade, where the rounded quotient can be
  /// off by at most one). Beyond that, `truncDiv` can be many units away from
  /// the true integer quotient and FMA would expose that pre-existing error;
  /// keep the historical `mul + sub` result instead. Two range/sign selects
  /// repair the at-most-one-ulp quotient error at integer boundaries.
  static Value createModResidual(PatternRewriter &rewriter, Location loc,
                                 Value x, Value y, Value truncDiv,
                                 Type elemType, bool preserveLowPrecision) {
    Value mul =
        createBinaryOp(rewriter, loc, truncDiv, y,
                       utils::createEmptyOp(rewriter, loc, x), BinaryKind::Mul);
    Value legacy =
        createBinaryOp(rewriter, loc, x, mul,
                       utils::createEmptyOp(rewriter, loc, x), BinaryKind::Sub);

    if (!archIsAscend950() || !archisModFma || !elemType.isF32() ||
        preserveLowPrecision)
      return legacy;

    Value negOne =
        utils::createConstantOp<float>(rewriter, loc, elemType, -1.0f);

    // `y` may be a scalar (e.g. `abs(exponent) % 2.0` inside powf); the range
    // check below needs a shaped value to compare against.
    auto ensureTensor = [&](Value value) -> Value {
      if (isa<RankedTensorType>(value.getType()))
        return value;
      auto refTy = cast<RankedTensorType>(x.getType());
      auto resultTy = RankedTensorType::get(refTy.getShape(), value.getType());
      return rewriter.create<tensor::GenerateOp>(
          loc, resultTy, ValueRange{},
          [&](OpBuilder &b, Location genLoc, ValueRange) {
            b.create<tensor::YieldOp>(genLoc, value);
          });
    };
    Value yTensor = ensureTensor(y);

    Value negTruncDiv =
        createBinaryOp(rewriter, loc, truncDiv, negOne,
                       utils::createEmptyOp(rewriter, loc, x), BinaryKind::Mul);
    auto fmaInit = utils::createEmptyOp(rewriter, loc, x);
    Value rem =
        hfusion::createTernaryOp<hfusion::ElemwiseTernaryOp, hfusion::TernaryFn,
                                 hfusion::TernaryFnAttr>(
            rewriter, loc, hfusion::TernaryFn::fma,
            ValueRange({negTruncDiv, y, x}), ValueRange(fmaInit))
            ->getResult(0);

    Value zero = utils::createConstantOp<float>(rewriter, loc, elemType, 0.0f);

    // `trunc(fl(x/y))` can be one unit away from the true integer quotient at
    // an integer boundary. If the quotient rounded down, the FMA residual is
    // in `(y, 2y)`; reduce it once. If it rounded up, the residual is in
    // `(-y, 0)` and has the wrong sign; add one `|y|` afterwards.
    Value absY = createUnaryOp(rewriter, loc, yTensor,
                               utils::createEmptyOp(rewriter, loc, yTensor),
                               UnaryKind::Abs);
    Value negAbsY = createBinaryOp(rewriter, loc, absY, negOne,
                                   utils::createEmptyOp(rewriter, loc, absY),
                                   BinaryKind::Mul);

    Value absRem =
        createUnaryOp(rewriter, loc, rem,
                      utils::createEmptyOp(rewriter, loc, rem), UnaryKind::Abs);
    Value tooLarge = hfusion::createCmpOp(rewriter, loc, absRem, absY,
                                          hfusion::CompareFn::vge)
                         ->getResult(0);
    Value remNegForRange =
        hfusion::createCmpOp(rewriter, loc, rem, zero, hfusion::CompareFn::vlt)
            ->getResult(0);
    Value rangeOffset =
        createSelectOp(rewriter, loc, remNegForRange, negAbsY, absY,
                       utils::createEmptyOp(rewriter, loc, absY));
    Value reduced = createBinaryOp(rewriter, loc, rem, rangeOffset,
                                   utils::createEmptyOp(rewriter, loc, rem),
                                   BinaryKind::Sub);
    rem = createSelectOp(rewriter, loc, tooLarge, reduced, rem,
                         utils::createEmptyOp(rewriter, loc, rem));

    Value remNonZero =
        hfusion::createCmpOp(rewriter, loc, rem, zero, hfusion::CompareFn::vne)
            ->getResult(0);
    Value remNeg =
        hfusion::createCmpOp(rewriter, loc, rem, zero, hfusion::CompareFn::vlt)
            ->getResult(0);
    Value xNeg =
        hfusion::createCmpOp(rewriter, loc, x, zero, hfusion::CompareFn::vlt)
            ->getResult(0);
    Value signDiff = createBinaryOp(rewriter, loc, remNeg, xNeg,
                                    utils::createEmptyOp(rewriter, loc, remNeg),
                                    BinaryKind::Xor);
    Value needSignFix = createBinaryOp(
        rewriter, loc, remNonZero, signDiff,
        utils::createEmptyOp(rewriter, loc, remNonZero), BinaryKind::And);
    Value signOffset =
        createSelectOp(rewriter, loc, xNeg, negAbsY, absY,
                       utils::createEmptyOp(rewriter, loc, absY));
    Value signFixed = createBinaryOp(rewriter, loc, rem, signOffset,
                                     utils::createEmptyOp(rewriter, loc, rem),
                                     BinaryKind::Add);
    rem = createSelectOp(rewriter, loc, needSignFix, signFixed, rem,
                         utils::createEmptyOp(rewriter, loc, rem));

    // For larger quotients `truncDiv` is not an exact integer and FMA would
    // preserve the pre-existing quotient error. Keep the legacy residual there
    // instead of trying to correct an unbounded error with fixed-size steps.
    Value absTruncDiv = createUnaryOp(
        rewriter, loc, truncDiv, utils::createEmptyOp(rewriter, loc, truncDiv),
        UnaryKind::Abs);
    Value fmaLimit =
        utils::createConstantOp<float>(rewriter, loc, elemType, 33554432.0f);
    Value useFma = hfusion::createCmpOp(rewriter, loc, absTruncDiv, fmaLimit,
                                        hfusion::CompareFn::vlt)
                       ->getResult(0);
    return createSelectOp(rewriter, loc, useFma, rem, legacy,
                          utils::createEmptyOp(rewriter, loc, rem));
  }
};

using NormalizeModOpRegBase = mlir::NormalizeModOpTemplate<hfusion::ElemwiseBinaryOp, HFusionModTraits>;

///  TODO: hfusion::binaryfn::floormod unsupport right now
///  normalize mod op to rec op
///   z = x % y
///  is normalized to
///   z = x - x // y * y
struct NormalizeFloorModOpRegBase
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::BinaryFn::mod) {
      return failure();
    }

    Type elemType = getElementTypeOrSelf(op.getInputs()[0].getType());
    if (!elemType.isIntOrIndexOrFloat()) {
      return failure();
    }
    if (elemType.isInteger(8)) {
      // i8 mod must be converted to f16 mod before
      return failure();
    }

    /// step 1: div = x / y
    auto emptyDivTensor =
        utils::createEmptyOp(rewriter, op->getLoc(), op.getInputs()[0]);
    auto divOP =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::div,
            ValueRange(op.getInputs()), ValueRange(emptyDivTensor));

    Operation *tempOp = divOP;

    /// step 2: floor = floor(res)
    if (isa<FloatType>(elemType)) {
      // insert extra floor for float mod
      auto emptyFloorTensor =
          utils::createEmptyOp(rewriter, op->getLoc(), op.getInputs()[0]);
      auto floorOp =
          hfusion::createUnaryOp<linalg::ElemwiseUnaryOp, linalg::UnaryFn,
                                 linalg::UnaryFnAttr>(
              rewriter, op->getLoc(), linalg::UnaryFn::floor,
              ValueRange{divOP->getResults()[0]}, ValueRange(emptyFloorTensor));
      tempOp = floorOp;
    }

    /// step 3:
    /// for int mod: mul = div * y
    /// for float mod: mul = floor * y
    auto emptyMulTensor =
        utils::createEmptyOp(rewriter, op->getLoc(), op.getInputs()[0]);
    auto mulOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::mul,
            ValueRange({tempOp->getResults()[0], op.getInputs()[1]}),
            ValueRange(emptyMulTensor));
    /// step 4: mod = x - mul
    auto emptySubTensor =
        utils::createEmptyOp(rewriter, op->getLoc(), op.getInputs()[0]);
    auto subOP =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::sub,
            ValueRange({op.getInputs()[0], mulOp->getResults()[0]}),
            ValueRange(emptySubTensor));

    rewriter.replaceOp(op, subOP);
    return success();
  }
};

struct HFusionNormalizeCeilandFloorTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeCeilandFloor(linalg::ElemwiseUnaryOp op) {
    return op.hasPureTensorSemantics() &&
           (op.getFun() == linalg::UnaryFn::ceil ||
            op.getFun() == linalg::UnaryFn::floor);
  }

  static CastRoundKind getCeilOrFloorRoundKind(linalg::ElemwiseUnaryOp op) {
    return op.getFun() == linalg::UnaryFn::ceil ? CastRoundKind::Ceil
                                                 : CastRoundKind::Floor;
  }

  static Value getCeilandFloorInput(linalg::ElemwiseUnaryOp op) {
    return op.getInputs()[0];
  }

  static Value getCeilandFloorOutput(linalg::ElemwiseUnaryOp op) {
    return op.getOutputs()[0];
  }
};

using NormalizeCeilandFloorOpRegBase =
    mlir::NormalizeCeilandFloorOpTemplate<linalg::ElemwiseUnaryOp,
                                          HFusionNormalizeCeilandFloorTraits>;

/// normalize nearbyint(x) to cast with RINT round mode.
struct NormalizeNearbyintOpRegBase
    : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics() ||
        op.getFun() != hfusion::UnaryFn::nearbyint)
      return failure();

    Location loc = op.getLoc();
    Value originalSrc = op.getInputs()[0];
    Value src = originalSrc;
    Value dst = op.getOutputs()[0];
    Type inType = getElementTypeOrSelf(src.getType());
    Type outType = getElementTypeOrSelf(dst.getType());
    if (!(inType.isF16() || inType.isBF16() || inType.isF32()) ||
        !(outType.isF16() || outType.isBF16() || outType.isF32()))
      llvm::report_fatal_error(
          "nearbyint normalize only supports f16, bf16 or f32");

    if (!NormalizeTraitsBase::archIsAscend950()) {
      if ((inType.isF16() || inType.isBF16()) && inType == outType) {
        src = NormalizeTraitsBase::createCastOp(
            rewriter, loc, src, rewriter.getF32Type(), CastRoundKind::RInt);
        src = NormalizeTraitsBase::createCastOp(
            rewriter, loc, src, rewriter.getF32Type(), CastRoundKind::RInt);
      }
    }

    Value result = NormalizeTraitsBase::createCastOp(
        rewriter, loc, src, outType, CastRoundKind::RInt, dst);
    Value signSrc = originalSrc;
    if (inType != outType)
      signSrc = NormalizeTraitsBase::createCastOp(
          rewriter, loc, originalSrc, outType, CastRoundKind::Round);
    result = buildCopysign(rewriter, loc, result, signSrc);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  Value buildCopysign(PatternRewriter &rewriter, Location loc, Value magnitude,
                      Value sign) const {
    auto empty = utils::createEmptyOp(rewriter, loc, magnitude);
    return rewriter
        .create<hfusion::ElemwiseBinaryOp>(
            loc, TypeRange{empty.getType()}, ValueRange{magnitude, sign},
            ValueRange{empty},
            ArrayRef<NamedAttribute>{rewriter.getNamedAttr(
                "fun", hfusion::BinaryFnAttr::get(
                           rewriter.getContext(), hfusion::BinaryFn::copysign))})
        ->getResult(0);
  }
};

/// normalize copysign(x, y) by preserving x magnitude bits and y sign bit.
struct NormalizeCopysignOpRegBase
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics() ||
        op.getFun() != hfusion::BinaryFn::copysign)
      return failure();

    Value magnitude = op.getInputs()[0];
    Value sign = op.getInputs()[1];
    auto inputTensorType = dyn_cast<RankedTensorType>(magnitude.getType());
    if (!inputTensorType || !isa<RankedTensorType>(sign.getType()))
      return failure();

    Type elementType = getElementTypeOrSelf(magnitude.getType());
    Type intType;
    uint64_t magnitudeMaskVal;
    uint64_t signMaskVal;
    if (elementType.isF32()) {
      intType = rewriter.getI32Type();
      magnitudeMaskVal = 0x7FFFFFFFU;
      signMaskVal = 0x80000000U;
    } else if (elementType.isF16() || elementType.isBF16()) {
      intType = rewriter.getI16Type();
      magnitudeMaskVal = 0x7FFFU;
      signMaskVal = 0x8000U;
    } else {
      llvm::report_fatal_error(
          "only support input Type is f16, bf16 or f32");
    }

    Location loc = op.getLoc();
    auto intTensorType = inputTensorType.clone(intType);
    Value magnitudeBits = NormalizeTraitsBase::createBitcastOp(
        rewriter, loc, intTensorType, magnitude);
    Value signBits =
        NormalizeTraitsBase::createBitcastOp(rewriter, loc, intTensorType, sign);
    Value magnitudeMask = rewriter.create<arith::ConstantOp>(
        loc, intType, rewriter.getIntegerAttr(intType, magnitudeMaskVal));
    Value signMask = rewriter.create<arith::ConstantOp>(
        loc, intType, rewriter.getIntegerAttr(intType, signMaskVal));

    Value maskedMagnitude = NormalizeTraitsBase::createBinaryOp(
        rewriter, loc, magnitudeBits, magnitudeMask,
        utils::createEmptyOp(rewriter, loc, magnitudeBits), BinaryKind::And);
    Value maskedSign = NormalizeTraitsBase::createBinaryOp(
        rewriter, loc, signBits, signMask,
        utils::createEmptyOp(rewriter, loc, signBits), BinaryKind::And);
    Value resultBits = NormalizeTraitsBase::createBinaryOp(
        rewriter, loc, maskedMagnitude, maskedSign,
        utils::createEmptyOp(rewriter, loc, maskedMagnitude), BinaryKind::Or);
    Value result = NormalizeTraitsBase::createBitcastOp(
        rewriter, loc, magnitude.getType(), resultBits);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// normalize 2^x to exp{ln(2)*x}
/// eg.
/// y = hfusion elemwise unary {exp2} (x)
/// is normalized to
///  y = linalg.elemwise_unary{vexp}(ln2 * x)
struct HFusionNormalizeExp2Traits : public NormalizeTraitsBase {
  static bool shouldNormalizeExp2(hfusion::ElemwiseUnaryOp op) {
    return op.hasPureTensorSemantics() && op.getFun() == hfusion::UnaryFn::exp2;
  }
};

using NormalizeExp2OpRegBase =
    mlir::NormalizeExp2OpTemplate<hfusion::ElemwiseUnaryOp,
                                  HFusionNormalizeExp2Traits>;

struct HFusionNormalizeMaxNumFTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeElemwiseMinMaxNumF(
      hfusion::ElemwiseBinaryOp op) {
    return op.hasPureTensorSemantics() &&
           op.getFun() == hfusion::BinaryFn::maxnumf;
  }
  static bool isMaxNumF(hfusion::ElemwiseBinaryOp op) { return true; }
  static Value getMinMaxNumFInput0(hfusion::ElemwiseBinaryOp op) {
    return op.getInputs()[0];
  }
  static Value getMinMaxNumFInput1(hfusion::ElemwiseBinaryOp op) {
    return op.getInputs()[1];
  }
};

struct HFusionNormalizeMinNumFTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeElemwiseMinMaxNumF(
      hfusion::ElemwiseBinaryOp op) {
    return op.hasPureTensorSemantics() &&
           op.getFun() == hfusion::BinaryFn::minnumf;
  }
  static bool isMaxNumF(hfusion::ElemwiseBinaryOp op) { return false; }
  static Value getMinMaxNumFInput0(hfusion::ElemwiseBinaryOp op) {
    return op.getInputs()[0];
  }
  static Value getMinMaxNumFInput1(hfusion::ElemwiseBinaryOp op) {
    return op.getInputs()[1];
  }
};

using NormalizeElemwiseMaxNumFOpRegBase =
    mlir::NormalizeElemwiseMinMaxNumFOpTemplate<
        hfusion::ElemwiseBinaryOp, HFusionNormalizeMaxNumFTraits>;
using NormalizeElemwiseMinNumFOpRegBase =
    mlir::NormalizeElemwiseMinMaxNumFOpTemplate<
        hfusion::ElemwiseBinaryOp, HFusionNormalizeMinNumFTraits>;

struct HFusionNormalizeReduceMinMaxNumFTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeReduceMinMaxNumF(linalg::ReduceOp op) {
    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    auto *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    return isa<arith::MaxNumFOp>(bodyOp) || isa<arith::MinNumFOp>(bodyOp);
  }

  static bool isReduceMaxNumF(linalg::ReduceOp op) {
    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    return isa<arith::MaxNumFOp>(yieldOp.getValues()[0].getDefiningOp());
  }

  static Value getReduceMinMaxNumFInput(linalg::ReduceOp op) {
    return op->getOperands()[0];
  }

  static void updateReduceMinMaxNumFOp(linalg::ReduceOp op,
                                       PatternRewriter &rewriter,
                                       Value newSrc, bool isMax) {
    op->setOperand(0, newSrc);
    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    auto *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(bodyOp);
    Operation *newOp;
    if (isMax) {
      newOp = rewriter.create<arith::MaximumFOp>(
          bodyOp->getLoc(), bodyOp->getOperand(0), bodyOp->getOperand(1));
    } else {
      newOp = rewriter.create<arith::MinimumFOp>(
          bodyOp->getLoc(), bodyOp->getOperand(0), bodyOp->getOperand(1));
    }
    rewriter.replaceAllUsesWith(bodyOp->getResult(0), newOp->getResult(0));
    rewriter.eraseOp(bodyOp);
  }
};

using NormalizeReduceMinMaxNumFOpRegBase =
    mlir::NormalizeReduceMinMaxNumFOpTemplate<
        linalg::ReduceOp, HFusionNormalizeReduceMinMaxNumFTraits>;

/// normalize expm1(x) to exp(x) - 1
/// eg.
/// y = hfusion elemwise unary {expm1} (x)
/// is normalized to
///  y = linalg.elemwise_unary{exp}(x) - 1
struct HFusionNormalizeExpM1Traits : public NormalizeTraitsBase {
  static bool shouldNormalizeExpM1(hfusion::ElemwiseUnaryOp op) {
    return op.hasPureTensorSemantics() &&
           op.getFun() == hfusion::UnaryFn::expm1;
  }
};

using NormalizeExpM1OpRegBase =
    mlir::NormalizeExpM1OpTemplate<hfusion::ElemwiseUnaryOp,
                                   HFusionNormalizeExpM1Traits>;
/// step 1. clip x into [-3.92,3.92]
/// step 2. numer=((((((CST0*y)+T1)*y+T2)*y+T3)*y+T4)*y+T5)*x, y=x^2
/// step 3. demon=((((y+P1)*y+P2)*y+P3)*y+P4)*y+P5, y=x^2
/// step 4: erf(x) = numer / denom
struct HFusionNormalizeErfTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeErf(hfusion::ElemwiseUnaryOp op) {
    return op.hasPureTensorSemantics() && op.getFun() == hfusion::UnaryFn::erf;
  }
};

using NormalizeErfOpRegBase =
    mlir::NormalizeErfOpTemplate<hfusion::ElemwiseUnaryOp,
                                 HFusionNormalizeErfTraits>;

/// normalize ilogb(x), which is exponent of frexp(x), to floor(log2(abs(x)))
struct HFusionNormalizeIlogbTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeIlogb(hfusion::ElemwiseUnaryOp op) {
    return op.hasPureTensorSemantics() &&
           op.getFun() == hfusion::UnaryFn::ilogb;
  }

  static Value createIlogbResult(PatternRewriter &rewriter, Location loc,
                                 Value log2) {
    Value floorInit = utils::createEmptyOp(rewriter, loc, log2);
    return createUnaryOp(rewriter, loc, log2, floorInit, UnaryKind::Floor);
  }
};

using NormalizeIlogbOpRegBase =
    mlir::NormalizeIlogbOpTemplate<hfusion::ElemwiseUnaryOp,
                                   HFusionNormalizeIlogbTraits>;
/// normalize `ldexp(x, y)` to `x * y`
struct HFusionNormalizeLdexpTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeLdexp(hfusion::ElemwiseBinaryOp op) {
    return op.hasPureTensorSemantics() &&
           op.getFun() == hfusion::BinaryFn::ldexp;
  }
};

using NormalizeLdexpOpRegBase =
    mlir::NormalizeLdexpOpTemplate<hfusion::ElemwiseBinaryOp,
                                   HFusionNormalizeLdexpTraits>;

/// normalize powf(baseNum, exponent) as below
/// powf(x, y) = 1, when abs(x) = 1 and abs(y) = inf
///            = nan, when x = -inf and y is not integer value or y is finite
///            = nan, when x < 0 and x is finite. and y is finite and y is not
///            integer
///            = x ^ y = exp(y * ln(|x|)), when x > 0
///            = x ^ y = ((-1) ^ y) * exp(y * ln|x|), when x <  0
///            = 1, when y == 0
/// so
/// partialRes0 = x ^ y = exp(y * ln(|x|)), when x > 0
///             = x ^ y = ((-1) ^ y) * exp(y * ln|x|), when x <  0
/// partialRes1 = select(abs(x)==1 && abs(y)==inf, 1, partialRes0)
/// partialRes2 = select((abs(x) != inf and x < 0 and abs(y) != inf
///               and floor(y) != y), nan, partialRes1), namely when x is
///               negative finite and y is finite and not integer, result is nan
/// pow(x, y) = select(y == 0, 1, partialRes2)
/// TODO : support nan boundary case
/// note: hardware vln will output -inf when x == 0
struct HFusionNormalizePowfTraits : public NormalizeTraitsBase {
  static bool shouldNormalizePowf(hfusion::ElemwiseBinaryOp op) {
    return op.hasPureTensorSemantics() &&
           op.getFun() == hfusion::BinaryFn::powf;
  }

  /// Recovers and returns a scalar `arith::ConstantOp` for the exponent from
  /// the wrapper forms that are common on the HFusion path.
  /// Example:
  ///   1. `linalg.fill(0.5)` feeding `powf`
  ///   2. `hfusion.cast(linalg.fill(0.5))` feeding `powf`
  ///   3. a shaped `arith.constant dense<0.5>`
  static arith::ConstantOp getPowfExponentScalarConstant(
      Value exponent, PatternRewriter &rewriter) {
    if (auto castOp = exponent.getDefiningOp<hfusion::CastOp>()) {
      if (auto fillOp = castOp.getDpsInputs()[0].getDefiningOp<linalg::FillOp>()) {
        return dyn_cast_or_null<arith::ConstantOp>(
            fillOp.getInputs()[0].getDefiningOp());
      }
    }

    if (auto fillOp = exponent.getDefiningOp<linalg::FillOp>())
      return dyn_cast_or_null<arith::ConstantOp>(
          fillOp.getInputs()[0].getDefiningOp());

    if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(
            exponent.getDefiningOp())) {
      if (!isa<ShapedType>(constOp.getType()))
        return constOp;
      auto scalarElem =
          getScalarFromConstantOp(rewriter, exponent.getLoc(), constOp);
      if (scalarElem.has_value())
        return dyn_cast_or_null<arith::ConstantOp>(
            scalarElem->getDefiningOp());
    }

    return nullptr;
  }

  /// Computes the parity coefficient `(-2 * mod) + 1` that maps the integer
  /// exponent parity (`abs(exponent) % 2`) to `-1` (odd) or `1` (even).
  /// On Ascend950 with f32, fuses `mul + add` into a single HFusion FMA.
  static Value computeParityCoefficient(PatternRewriter &rewriter,
                                        Location loc, Value mod,
                                        Value negativeTwo,
                                        Value positiveOne) {
    Type elemType = getElementTypeOrSelf(mod.getType());
    if (archisAscend950 && elemType.isF32()) {
      auto fmaInit = utils::createEmptyOp(rewriter, loc, mod);
      return hfusion::createTernaryOp<hfusion::ElemwiseTernaryOp,
                                      hfusion::TernaryFn,
                                      hfusion::TernaryFnAttr>(
                 rewriter, loc, hfusion::TernaryFn::fma,
                 ValueRange({mod, negativeTwo, positiveOne}),
                 ValueRange(fmaInit))
          ->getResult(0);
    }
    Value mul = createBinaryOp(rewriter, loc, mod, negativeTwo,
                               utils::createEmptyOp(rewriter, loc, mod),
                               BinaryKind::Mul);
    return createBinaryOp(rewriter, loc, mul, positiveOne,
                          utils::createEmptyOp(rewriter, loc, mod),
                          BinaryKind::Add);
  }
};

using NormalizePowfOpRegBase =
    mlir::NormalizePowfOpTemplate<hfusion::ElemwiseBinaryOp,
                                  HFusionNormalizePowfTraits>;

void populateNormalizeModPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeModOpRegBase>(patterns.getContext());
}

void populateNormalizePrimaryMathPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeCeilandFloorOpRegBase>(ctx);
  patterns.add<NormalizeLogLikeOpRegBase>(ctx);
  patterns.add<NormalizeLog1pOpRegBase>(ctx);
  patterns.add<NormalizeReduceMinMaxNumFOpRegBase>(ctx);
  patterns.add<NormalizeElemwiseMaxNumFOpRegBase>(ctx);
  patterns.add<NormalizeElemwiseMinNumFOpRegBase>(ctx);
  patterns.add<NormalizeNearbyintOpRegBase>(ctx);
  patterns.add<NormalizeCopysignOpRegBase>(ctx);
  patterns.add<NormalizeExp2OpRegBase>(ctx);
  patterns.add<NormalizeExpM1OpRegBase>(ctx);
  patterns.add<NormalizeErfOpRegBase>(ctx);
}

void populateNormalizeLateMathPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeIlogbOpRegBase>(ctx);
  patterns.add<NormalizeLdexpOpRegBase>(ctx);
  patterns.add<NormalizePowfOpRegBase>(ctx);
}
} // namespace mlir::hfusion
