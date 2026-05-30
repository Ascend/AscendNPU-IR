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

#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"
#include "bishengir/Transforms/Normalize/Utils/ScalarTemplateHelpers.h"
#include "bishengir/Transforms/Normalize/NormalizeMathTemplate.h"

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
    llvm_unreachable("unsupport log op");
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

using NormalizeLogLikeOp =
    mlir::NormalizeLogLikeOpTemplate<hfusion::ElemwiseUnaryOp,
                                     HFusionNormalizeLogLikeTraits>;
using NormalizeLog1pOp =
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
};

using NormalizeModOp = mlir::NormalizeModOpTemplate<hfusion::ElemwiseBinaryOp, HFusionModTraits>;

///  TODO: hfusion::binaryfn::floormod unsupport right now
///  normalize mod op to rec op
///   z = x % y
///  is normalized to
///   z = x - x // y * y
struct NormalizeFloorModOp
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

struct NormalizeCeilandFloorOp
    : public OpRewritePattern<linalg::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != linalg::UnaryFn::ceil &&
        op.getFun() != linalg::UnaryFn::floor) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());

    assert(!inType.isInteger() && "Cast in floor/ceil mode doesn't support "
                                  "integer type input");
    OpBuilder builder(op);
    Value src = op.getInputs()[0];
    hfusion::RoundMode roundMode = op.getFun() == linalg::UnaryFn::ceil
                                       ? hfusion::RoundMode::CEIL
                                       : hfusion::RoundMode::FLOOR;
    if (!archisAscend950) {
      if ((inType.isF16() || inType.isBF16()) && inType == outType) {
        // 910B only support fp32 ceil and floor, so change to fp16->fp32,
        // fp32 ceil/floor and fp32->fp16
        // TODO: add platform info to isHWSupportCeilFLoor(Type)

        // Step1: cast to fp32 to do ceil or floor
        auto intermediate = hfusion::castTo(builder, src, rewriter.getF32Type(),
                                            hfusion::RoundMode::RINT);
        // Step2: enable fp32 cast ability with ceil or floor mode
        // Otherwise, cast fp32 to fp16 type in ceil or floor mode just changes
        // precision loss part.
        src = hfusion::castTo(builder, intermediate, rewriter.getF32Type(),
                              roundMode);
      }
    }

    auto castOp =
        hfusion::castTo(builder, src, outType, roundMode, op.getOutputs()[0]);
    rewriter.replaceOp(op, castOp);
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

using NormalizeExp2Op =
    mlir::NormalizeExp2OpTemplate<hfusion::ElemwiseUnaryOp,
                                  HFusionNormalizeExp2Traits>;

struct NormalizeMinMax {
  /// Returns a new operand for BinaryFn::maxf (BinaryFn::minf)
  /// that is used when normalizing maxnumf (minnumf) to maxf (minf).
  Value createNewSrcForMinMaxNumFOp(PatternRewriter &rewriter, Location loc,
                                    Value src, double paddingInfValue) const {
    auto elementType = getElementTypeOrSelf(src.getType());
    auto constInfinity = utils::createConstantOp<double>(
        rewriter, loc, elementType, paddingInfValue);

    auto isNanResultTensorType =
        utils::getTensorTypeWithSameShape(src.getType(), rewriter.getI1Type());
    auto isNanOp = rewriter.create<IsNanOp>(loc, isNanResultTensorType, src);

    auto selectOpOut = utils::createEmptyOp(rewriter, loc, src);
    auto selectOp = rewriter.create<SelectOp>(
        loc, TypeRange(selectOpOut),
        ValueRange({isNanOp->getResult(0), constInfinity, src}),
        ValueRange(selectOpOut));

    return selectOp->getResult(0);
  }
};

/// Normalize maxnumf (minnumf) to maxf (minf).
/// eg.
/// dst = hfusion.elemwise_binary {maxnumf} (src0, src1)
/// is normalized to
/// src0_nan_mask = hfusion.isnan(src0)
/// new_src0 = hfusion.select(src0_nan_mask, -inf, src0)
/// src1_nan_mask = hfusion.isnan(src1)
/// new_src1 = hfusion.select(src1_nan_mask, -inf, src1)
/// dst = hfusion.elemwise_binary {maxf} (new_src0, new_src1)
template <BinaryFn funFrom>
struct NormalizeElemwiseMinMaxNumFOp
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp>,
      public NormalizeMinMax {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    static_assert(funFrom == BinaryFn::maxnumf || funFrom == BinaryFn::minnumf,
                  "Argument mismatch. NormaliseMinMaxNumFOp expects "
                  "hfusion::BinaryFn::maxnumf or hfusion::BinaryFn::minnumf");

    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != funFrom) {
      return failure();
    }

    constexpr auto funTo =
        (funFrom == BinaryFn::maxnumf) ? BinaryFn::maxf : BinaryFn::minf;
    constexpr auto paddingInfSign = (funFrom == BinaryFn::maxnumf) ? -1 : 1;

    auto res = rewriteMinMaxNumFOp<funTo>(
        op, rewriter, paddingInfSign * std::numeric_limits<double>::infinity());

    rewriter.replaceOp(op, res);
    return success();
  }

private:
  /// Normalize maxnumf (minnumf) to maxf (minf)
  /// Check comment before struct definition
  template <hfusion::BinaryFn hfusionFn>
  Value rewriteMinMaxNumFOp(hfusion::ElemwiseBinaryOp op,
                            PatternRewriter &rewriter,
                            double paddingInfValue) const {
    static_assert(
        hfusionFn == BinaryFn::maxf || hfusionFn == BinaryFn::minf,
        "Normalization hfusion::BinaryFn::maxnumf (minnumf) allows "
        "only hfusion::BinaryFn::maxf (minf) to be used for replacement");

    Value src0 = op.getInputs()[0];
    Value src1 = op.getInputs()[1];

    auto newSrc0 = createNewSrcForMinMaxNumFOp(rewriter, op->getLoc(), src0,
                                               paddingInfValue);
    auto newSrc1 = createNewSrcForMinMaxNumFOp(rewriter, op->getLoc(), src1,
                                               paddingInfValue);
    auto minMaxFOpOut = utils::createEmptyOp(rewriter, op->getLoc(), src0);
    auto minMaxFOp = createBinaryOp<ElemwiseBinaryOp, BinaryFn, BinaryFnAttr>(
        rewriter, op->getLoc(), hfusionFn, ValueRange({newSrc0, newSrc1}),
        ValueRange(minMaxFOpOut));

    return minMaxFOp->getResult(0);
  }
};

// Normalize reduction op of maxnumf and minnumf
// dst = linalg.reduce { arith.maxnumf } src
// is normalized to
// src_nan_mask = hfusion.isnan(src)
// new_src = hfusion.select(src_nan_mask, -inf, src)
// dst = linalg.reduce { arith.maximumf } src
struct NormalizeReduceMinMaxNumFOp : public OpRewritePattern<linalg::ReduceOp>,
                                     public NormalizeMinMax {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;
  virtual ~NormalizeReduceMinMaxNumFOp() = default;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    auto *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    if (!isa<arith::MaxNumFOp>(bodyOp) && !isa<arith::MinNumFOp>(bodyOp)) {
      return failure();
    }
    auto paddingInfSign = (isa<arith::MaxNumFOp>(bodyOp)) ? -1 : 1;
    Value src0 = op->getOperands()[0];
    auto newSrc0 = createNewSrcForMinMaxNumFOp(
        rewriter, op->getLoc(), src0,
        paddingInfSign * std::numeric_limits<double>::infinity());
    op->setOperand(0, newSrc0);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(bodyOp);
    Operation *newOp;
    if (isa<arith::MaxNumFOp>(bodyOp)) {
      newOp = rewriter.create<arith::MaximumFOp>(
          bodyOp->getLoc(), bodyOp->getOperand(0), bodyOp->getOperand(1));
    } else {
      newOp = rewriter.create<arith::MinimumFOp>(
          bodyOp->getLoc(), bodyOp->getOperand(0), bodyOp->getOperand(1));
    }
    rewriter.replaceAllUsesWith(bodyOp->getResult(0), newOp->getResult(0));
    rewriter.eraseOp(bodyOp);
    return success();
  }
};

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

using NormalizeExpM1Op =
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

using NormalizeErfOp =
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

using NormalizeIlogbOp =
    mlir::NormalizeIlogbOpTemplate<hfusion::ElemwiseUnaryOp,
                                   HFusionNormalizeIlogbTraits>;
/// normalize `ldexp(x, y)` to `x * y`
struct HFusionNormalizeLdexpTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeLdexp(hfusion::ElemwiseBinaryOp op) {
    return op.hasPureTensorSemantics() &&
           op.getFun() == hfusion::BinaryFn::ldexp;
  }
};

using NormalizeLdexpOp =
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

using NormalizePowfOp =
    mlir::NormalizePowfOpTemplate<hfusion::ElemwiseBinaryOp,
                                  HFusionNormalizePowfTraits>;

void populateNormalizeModPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeModOp>(patterns.getContext());
}

void populateNormalizePrimaryMathPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeCeilandFloorOp>(ctx);
  patterns.add<NormalizeLogLikeOp>(ctx);
  patterns.add<NormalizeLog1pOp>(ctx);
  patterns.add<NormalizeReduceMinMaxNumFOp>(ctx);
  patterns.add<NormalizeElemwiseMinMaxNumFOp<BinaryFn::maxnumf>>(ctx);
  patterns.add<NormalizeElemwiseMinMaxNumFOp<BinaryFn::minnumf>>(ctx);
  patterns.add<NormalizeExp2Op>(ctx);
  patterns.add<NormalizeExpM1Op>(ctx);
  patterns.add<NormalizeErfOp>(ctx);
}

void populateNormalizeLateMathPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeIlogbOp>(ctx);
  patterns.add<NormalizeLdexpOp>(ctx);
  patterns.add<NormalizePowfOp>(ctx);
}
} // namespace mlir::hfusion
