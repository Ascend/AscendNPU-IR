//===- NormalizeArithmetic.cpp ----------------------------------*- C++ -*-===//
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

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Transforms/Normalize/NormalizeArithmeticTemplate.h"
#include "llvm/ADT/APFloat.h"

namespace mlir {
/// Normalizes `rsqrt(x)` to `rec(sqrt(x))`
struct HFusionNormalizeRSqrtTraits
    : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeRSqrt(hfusion::ElemwiseUnaryOp op) {
    return op.hasPureTensorSemantics() && op.getFun() == hfusion::UnaryFn::rsqrt;
  }
};

/// Normalizes `mul(rec_like(x), y)` to `div(y, x)`
/// (1/b) * a -> a/b
/// a * (1/b) -> a/b
struct HFusionNormalizeMulRecTraits
    : public hfusion::NormalizeTraitsBase {
  using RecOpType = hfusion::ElemwiseUnaryOp;
  using DivOpType = linalg::ElemwiseBinaryOp;

  static bool shouldNormalizeMulRec(linalg::ElemwiseBinaryOp op) {
    return op.hasPureTensorSemantics() && op.getFun() == linalg::BinaryFn::mul;
  }
};

/// Normalizes `div(1, x)` to `rec(x)`.
struct HFusionNormalizeDivVSToRecTraits
    : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeDiv(linalg::ElemwiseBinaryOp op) {
    return op.hasPureTensorSemantics() && op.getFun() == linalg::BinaryFn::div;
  }
};

struct HFusionNormalizeVPowiToPowfTraits
    : public hfusion::NormalizeTraitsBase {
  static bool shouldNormalizeVPowi(hfusion::ElemwiseBinaryOp op) {
    return op.getFun() == hfusion::BinaryFn::powi;
  }
};

struct HFusionNormalizeMulExtTraits : public hfusion::NormalizeTraitsBase {
  static bool shouldNormalizeMulExt(hfusion::MulExtOp) { return true; }

  static Value getMulExtLhs(hfusion::MulExtOp op) { return op.getLhs(); }

  static Value getMulExtRhs(hfusion::MulExtOp op) { return op.getRhs(); }

  static Type getMulExtExtendedType(PatternRewriter &rewriter, Type inputType) {
    return inputType.isInteger(8) ? rewriter.getI16Type() : Type();
  }
};

/// normalize VSUB(s, v) to VADD(s,VMULS(v, -1)).
struct HFusionNormalizeSubVSToVMulAndVAddTraits
    : public hfusion::NormalizeTraitsBase {
  // Keep the scalar in the first operand slot of the final add.
  static constexpr bool kPlaceScalarFirstInFinalAdd = true;

  static bool shouldNormalizeSubScalarVector(linalg::ElemwiseBinaryOp op) {
    return op.hasPureTensorSemantics() && op.getFun() == linalg::BinaryFn::sub &&
           hfusion::isSVOp(op);
  }

  static Value getSubScalar(linalg::ElemwiseBinaryOp op) {
    return op.getDpsInputs()[0];
  }

  static Value getSubVector(linalg::ElemwiseBinaryOp op) {
    return op.getDpsInputs()[1];
  }
};

} // namespace mlir

namespace mlir::hfusion {

/// normalize negf op to mul op
/// eg.
///  y = linalg.elemwise_unary {negf} (x)
///  is normalized to
///  y = linalg.elemwise_binary {mul} (x, -1)
struct NormalizeNegToMul : public OpRewritePattern<linalg::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != linalg::UnaryFn::negf) {
      return failure();
    }

    auto input = op.getDpsInputs()[0];
    auto elementType = getElementTypeOrSelf(input.getType());
    Value one = rewriter.create<arith::ConstantOp>(
        op->getLoc(), elementType, rewriter.getFloatAttr(elementType, -1.0));
    auto mulOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::mul,
            ValueRange{input, one}, ValueRange(op.getDpsInits()[0]));
    rewriter.replaceOp(op, mulOp);
    return success();
  }
};

/// Normalize divsi with i8 type for regbase:
/// c = a / b
/// is normalized to
/// a16 = castTo<i16>(a)
/// b16 = castTo<i16>(b)
/// c16 = a16 / b16
/// c = castTo<i8>(c16, mode = TRUNC, sat=true)
/// Normalize divsi and divui for membase:
/// supports i8/i16/i32/i64 type
/// c = a / b
/// is normalized to
/// fa = castTo<f32>(a)
/// fb = castTo<f32>(b)
/// fc = fa / fb
/// c = castTo<integer>(fc, mode = TRUNC)
struct HFusionNormalizeDivSIandDivUIOpTraits
    : public hfusion::NormalizeTraitsBase {
  static bool shouldNormalizeDivSIandDivUIOp(linalg::ElemwiseBinaryOp op) {
    if (!op.hasPureTensorSemantics())
      return false;
    if ((op.getFun() != linalg::BinaryFn::div) &&
        (op.getFun() != linalg::BinaryFn::div_unsigned))
      return false;

    auto resTensor = op.getResultTensors()[0];
    auto resTy = dyn_cast<TensorType>(resTensor.getType());
    if (!resTy)
      return false;
    auto elemTySrc = getElementTypeOrSelf(resTy);
    if (archisMembased)
      return elemTySrc.isInteger();
    if (archIsRegbased())
      return elemTySrc.isInteger(8);
    return true;
  }

  static LogicalResult rewriteDivSIandDivUIOp(linalg::ElemwiseBinaryOp op,
                                              PatternRewriter &rewriter) {
    if (!archisMembased)
      return failure();

    rewriter.setInsertionPoint(op);
    auto resTensor = op.getResultTensors()[0];
    auto elemTySrc = getElementTypeOrSelf(resTensor.getType());
    auto inputs = op.getDpsInputs();
    auto res = hfusion::divWithRoundMode(rewriter, op.getLoc(), elemTySrc,
                                         inputs[0], inputs[1], resTensor,
                                         hfusion::RoundMode::TRUNC);
    rewriter.replaceOp(op, res);
    return success();
  }

  static LogicalResult rewriteI8IntegerDivThroughI16(
      linalg::ElemwiseBinaryOp op, PatternRewriter &rewriter) {
    if (archisMembased)
      return failure();

    rewriter.setInsertionPoint(op);
    auto loc = op.getLoc();
    auto resTensor = op.getResultTensors()[0];
    auto elemTySrc = getElementTypeOrSelf(resTensor.getType());
    auto inputs = op.getDpsInputs();
    hfusion::TypeFn castIntegerType =
        op.getFun() == linalg::BinaryFn::div ? hfusion::TypeFn::cast_signed
                                             : hfusion::TypeFn::cast_unsigned;

    Value castI16X = hfusion::castTo(rewriter, inputs[0], rewriter.getI16Type(),
                                     castIntegerType);
    Value castI16Y = hfusion::castTo(rewriter, inputs[1], rewriter.getI16Type(),
                                     castIntegerType);

    auto divInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, resTensor, rewriter.getI16Type());
    auto divI16 =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, op.getFun(), ValueRange{castI16X, castI16Y},
            ValueRange(divInit))
            ->getResults()[0];

    if (!archisAscend950) {
      Value res = hfusion::castTo(rewriter, divI16, elemTySrc,
                                  hfusion::RoundMode::TRUNC);
      rewriter.replaceOp(op, res);
      return success();
    }
    if (op.getFun() == linalg::BinaryFn::div_unsigned) {
      Value res = hfusion::castTo(rewriter, divI16, elemTySrc,
                                  hfusion::RoundMode::TRUNC);
      rewriter.replaceOp(op, res);
      return success();
    }

    // avoid -128/-1 overflow error in i8 with div.i16
    auto i8ExcdNum =
        utils::createConstantOp<int>(rewriter, loc, rewriter.getI16Type(), 128);
    auto i8MinNum = utils::createConstantOp<int>(rewriter, loc,
                                                 rewriter.getI16Type(), -128);
    Value exceedMask =
        hfusion::createCmpOp(rewriter, loc, divI16, i8ExcdNum, CompareFn::veq)
            ->getResult(0);
    auto finalResult =
        rewriter
            .create<hfusion::SelectOp>(loc, TypeRange(divInit),
                                       ValueRange{exceedMask, i8MinNum, divI16},
                                       ValueRange(divI16))
            .getResults()[0];

    Value res = hfusion::castTo(rewriter, finalResult, rewriter.getF16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow*/ false, true);
    res = hfusion::castTo(rewriter, res, elemTySrc, hfusion::RoundMode::TRUNC,
                          std::nullopt,
                          /*enableOverflow*/ false, true, castIntegerType);
    rewriter.replaceOp(op, res);
    return success();
  }
};

/// normalize ceildivsi or floordivsi i8/i16/i32/i64 as bellow
/// eg.
///   %res = ceildivsi/floordivsi %lhs, %rhs : i8
/// is normalized to
///   %lhsF32 = cast %src i8 to f32
///   %rhsF32 = cast %rhs i8 to f32
///   %divF32 = div %lhsF32, %rhsF32 : f32
///   %castF32 = ceilop/floorop %divF32
///   %res = cast %castF32 f32 to i8
struct NormalizeCDivandFloorDivIntOp
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto fun = op.getFun();
    if (!(fun == hfusion::BinaryFn::ceildivsi ||
          fun == hfusion::BinaryFn::ceildivui ||
          fun == hfusion::BinaryFn::floordivsi)) {
      return failure();
    }

    auto resTensor = op.getResultTensors()[0];
    auto resTy = dyn_cast<TensorType>(resTensor.getType());
    auto elemTySrc = getElementTypeOrSelf(resTy);
    if (!elemTySrc.isInteger()) {
      return failure();
    }

    // step1. res = divWithRoundMode(x, y, FLOOR/CEIL)
    rewriter.setInsertionPoint(op);
    auto inputs = op.getDpsInputs();

    auto loc = op->getLoc();
    // TODO: fix to use uint type after support uint op
    hfusion::RoundMode roundMode = (fun == hfusion::BinaryFn::ceildivsi ||
                                    fun == hfusion::BinaryFn::ceildivui)
                                       ? hfusion::RoundMode::CEIL
                                       : hfusion::RoundMode::FLOOR;
    auto res = hfusion::divWithRoundMode(rewriter, loc, elemTySrc, inputs[0],
                                         inputs[1], resTensor, roundMode);
    rewriter.replaceOp(op, res);
    return success();
  }
};

/// normalize div with constant denominator
/// a/cst -> rec = 1/cst; a * rec
struct NormalizeDivWithCstDenominator : public OpRewritePattern<linalg::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;

  bool checkTypeConsistencyFails(double reciprocal, FloatType targetT) const {
    llvm::APFloat castedRecip(reciprocal);
    bool losesInfo;
    castedRecip.convert(targetT.getFloatSemantics(),
                        llvm::RoundingMode::NearestTiesToEven,
                        &losesInfo);
    // to preserve precision exclude fp number overflow
    // and fp number denormal value cases
    bool fail = castedRecip.isDenormal() || castedRecip.isInfinity();
    return fail;
  }

  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics() ||
        op.getFun() != linalg::BinaryFn::div) {
      return failure();
    }

    auto inputs = op.getDpsInputs();
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    auto rhsOp = rhs.getDefiningOp<arith::ConstantOp>();

    if (!rhsOp) {
      return failure();
    }

    // skip the integer division case
    if (auto constFloatAttr = dyn_cast<FloatAttr>(rhsOp.getValue())) {
      double reciprocal = 1 / constFloatAttr.getValueAsDouble();

      FloatType rhsDType = dyn_cast<FloatType>(rhs.getType());
      if (!rhsDType) {
        return failure();
      }

      if (checkTypeConsistencyFails(reciprocal, rhsDType)) {
        return failure();
      }

      {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(rhsOp);
        rhs = rewriter.create<arith::ConstantOp>(rhsOp.getLoc(), rhsDType,
                                          rewriter.getFloatAttr(rhsDType, reciprocal));
      }

      auto out = utils::createEmptyOp(rewriter, op.getLoc(), op.getResult(0));
      auto attr = rewriter.getAttr<linalg::BinaryFnAttr>(linalg::BinaryFn::mul);
      auto fnAttr = rewriter.getNamedAttr("fun", attr);
      Value mul = rewriter.create<linalg::ElemwiseBinaryOp>(
                        op.getLoc(), ValueRange{lhs, rhs}, ValueRange{out}, fnAttr)->getResults()[0];

      rewriter.replaceOp(op, mul);
      return success();
    }
    return failure();
  }
};

using NormalizeRSqrtOp =
    NormalizeRSqrtOpTemplate<hfusion::ElemwiseUnaryOp, HFusionNormalizeRSqrtTraits>;
using NormalizeMulRecOp =
    NormalizeMulRecOpTemplate<linalg::ElemwiseBinaryOp, HFusionNormalizeMulRecTraits>;
using NormalizeDivVSToRec =
    NormalizeDivVSToRecTemplate<linalg::ElemwiseBinaryOp, HFusionNormalizeDivVSToRecTraits>;
using NormalizeVPowiToPowf =
    NormalizeVPowiToPowfTemplate<hfusion::ElemwiseBinaryOp,
                                 HFusionNormalizeVPowiToPowfTraits>;
using NormalizeMulExtOp =
    NormalizeMulExtOpTemplate<hfusion::MulExtOp, HFusionNormalizeMulExtTraits>;
using NormalizeSubVSToVMulAndVAdd =
    NormalizeSubVSToVMulAndVAddTemplate<linalg::ElemwiseBinaryOp,
                                        HFusionNormalizeSubVSToVMulAndVAddTraits>;
using NormalizeDivSIandDivUIOp =
    NormalizeDivSIandDivUIOpTemplate<linalg::ElemwiseBinaryOp,
                                     HFusionNormalizeDivSIandDivUIOpTraits>;

void populateNormalizeMulRecPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeMulRecOp>(patterns.getContext());
}

void populateNormalizeArithmeticPatterns(RewritePatternSet &patterns, bool enableFastDiv) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeNegToMul>(ctx);
  patterns.add<NormalizeDivVSToRec>(ctx);
  patterns.add<NormalizeVPowiToPowf>(ctx);
  patterns.add<NormalizeSubVSToVMulAndVAdd>(ctx);
  patterns.add<NormalizeRSqrtOp>(ctx);
  if (enableFastDiv) {
    patterns.add<NormalizeDivWithCstDenominator>(ctx);
  }
}

void populateNormalizePreFinalArithmeticPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeCDivandFloorDivIntOp>(ctx);
  patterns.add<NormalizeMulExtOp>(ctx);
}

void populateNormalizeFinalArithmeticPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeDivSIandDivUIOp>(patterns.getContext());
}
} // namespace mlir::hfusion
