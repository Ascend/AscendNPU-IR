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
#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Transforms/Normalize/NormalizeArithmeticTemplate.h"

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
struct NormalizeDivSIandDivUIOp
    : public OpRewritePattern<linalg::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if ((op.getFun() != linalg::BinaryFn::div) &&
        (op.getFun() != linalg::BinaryFn::div_unsigned)) {
      return failure();
    }

    auto loc = op->getLoc();
    auto resTensor = op.getResultTensors()[0];
    auto resTy = dyn_cast<TensorType>(resTensor.getType());
    auto elemTySrc = getElementTypeOrSelf(resTy);
    if (archisMembased && (!elemTySrc.isInteger())) {
      return failure();
    }
    if (archIsRegbased && (!elemTySrc.isInteger(8))) {
      return failure();
    }
    rewriter.setInsertionPoint(op);
    auto inputs = op.getDpsInputs();
    if (archisMembased) {
      auto res = hfusion::divWithRoundMode(rewriter, loc, elemTySrc, inputs[0],
                                           inputs[1], resTensor,
                                           hfusion::RoundMode::TRUNC);
      rewriter.replaceOp(op, res);
      return success();
    }
    // cast lhs and rhs from u8/i8 to u16/i16
    hfusion::TypeFn castIntegerType =
        (op.getFun() == linalg::BinaryFn::div_unsigned)
            ? hfusion::TypeFn::cast_unsigned
            : hfusion::TypeFn::cast_signed;
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
        createCmpOp(rewriter, loc, divI16, i8ExcdNum, CompareFn::veq)
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

/// normalize VSUB(s, v) to VADD(s,VMULS(v, -1)).
struct NormalizeSubVSToVMulAndVAdd
    : public OpRewritePattern<linalg::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return failure();

    if (op.getFun() != linalg::BinaryFn::sub)
      return failure();

    if (!isSVOp(op))
      return failure();

    auto inputs = op.getDpsInputs();
    Value vec = inputs[1];
    Type scalarType = inputs[0].getType();
    Location loc = op.getLoc();

    auto negOne = utils::createConstantOp<float>(rewriter, loc, scalarType, -1);
    Value empty = utils::createEmptyOp(rewriter, loc, vec);
    auto *resOp = createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                 linalg::BinaryFnAttr>(
        rewriter, loc, linalg::BinaryFn::mul, ValueRange{vec, negOne}, empty);

    // Because vsubs is not a supported hardware instruction,
    // Then vsubs(s, v) = vadds(s, vmuls(-1, v)). This computation can be
    // simplified into vmuls(-1, v), when scalar s equals to zero. Usually we
    // can add SimplifyOps Pass after Normalize at the end of `preProcess`
    // function in HFusionPipelines pass, however this may cause some errors on
    // A2/A3 platform. So this simplify process is taken here to solve this
    // reduntant instruction in simd-vf of A5 platform only.
    // TODO: Pls check errors from pipeline on A2/A3, and checkout detail info
    // by issue #74, on gitcode of A2/A3.
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!(hacc::utils::isAscend950(moduleOp) && isConstantZero(inputs[0]))) {
      resOp = createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                             linalg::BinaryFnAttr>(
          rewriter, loc, linalg::BinaryFn::add,
          ValueRange{inputs[0], resOp->getResult(0)}, op.getDpsInits()[0]);
    }

    rewriter.replaceAllUsesWith(op->getResults(), resOp->getResults());
    rewriter.eraseOp(op);
    return success();
  }
  bool isConstantZero(Value val) const {
    if (auto constOp =
            dyn_cast_or_null<arith::ConstantOp>(val.getDefiningOp())) {
      Attribute attr = constOp.getValue();
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr))
        return intAttr.getValue().isZero();
      if (auto floatAttr = mlir::dyn_cast<FloatAttr>(attr))
        return floatAttr.getValue().isZero();
    }
    return false;
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

/// normalize mulext(x, y) as bellow
/// inputs: N-bit number x, y
/// step1: perform extension to generate 2N-bit operands from x and y
/// step2: multiply 2N-bit x and y to get mul_res
/// step3: get the high half of the operand by N-bit-right-shifting mul_res
/// step4: get the low half of the operand by N-bit-left-shifting
/// and later N-bit-right-shifting mul_res
/// step5: cast result back to origin type
/// outputs: the N-bit low and the N-bit high halves of the product.
class NormalizeMulExtOp : public OpRewritePattern<hfusion::MulExtOp> {
public:
  using OpRewritePattern<hfusion::MulExtOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::MulExtOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto lhsType = getElementTypeOrSelf(lhs.getType());
    auto rhsType = getElementTypeOrSelf(rhs.getType());
    if (!lhsType.isInteger(8) || !rhsType.isInteger(8)) {
      return failure();
    }

    // step1: perform extension.
    Value lhsI16 = hfusion::castTo(rewriter, lhs, rewriter.getI16Type());
    Value rhsI16 = hfusion::castTo(rewriter, rhs, rewriter.getI16Type());

    // step2: multiply
    auto loc = op.getLoc();
    auto mulInit = utils::createEmptyOp(rewriter, loc, lhsI16);
    auto mulRes =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul, ValueRange({lhsI16, rhsI16}),
            ValueRange(mulInit))
            ->getResult(0);

    // step3: get the high half of the operand
    auto bitWidth = lhsType.getIntOrFloatBitWidth();
    arith::ConstantOp shiftValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI16Type(),
        rewriter.getIntegerAttr(rewriter.getI16Type(), bitWidth));
    auto shrHighBitInit = utils::createEmptyOp(rewriter, loc, lhsI16);
    auto shrHighBit =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, hfusion::BinaryFn::shrsi,
            ValueRange{mulRes, shiftValue}, ValueRange(shrHighBitInit))
            ->getResult(0);

    // step4: get the low half of the operand
    auto shlInit = utils::createEmptyOp(rewriter, loc, lhsI16);
    auto shlRes =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, hfusion::BinaryFn::shli,
            ValueRange{mulRes, shiftValue}, ValueRange(shlInit))
            ->getResult(0);
    auto shrLowBitInit = utils::createEmptyOp(rewriter, loc, lhsI16);
    auto shrLowBit =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, hfusion::BinaryFn::shrsi,
            ValueRange{shlRes, shiftValue}, ValueRange(shrLowBitInit))
            ->getResult(0);

    // step5: cast result back to origin type i8
    auto roundMode = hfusion::RoundMode::TRUNCWITHOVERFLOW;
    auto highBitI8 =
        hfusion::castTo(rewriter, shrHighBit, rewriter.getI8Type(), roundMode);
    auto lowBitI8 =
        hfusion::castTo(rewriter, shrLowBit, rewriter.getI8Type(), roundMode);
    rewriter.replaceOp(op, {lowBitI8, highBitI8});
    return success();
  }
};

/// Normalize Powi from I8/I16 to Powf F32
/// Compute with F32, then cast back to I8/I16
/// For example:
/// result = hfusion.powi(i8 x, i8y)
/// is legalized to
/// x_1 = cast x from i8 to f32
/// y_1 = cast y from i8 to f32
/// z_1 = hfusion.powf(f32 x_1, f32 y_1)
/// result = cast z_1 from f32 to i8
struct NormalizeVPowiToPowf
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getFun() != hfusion::BinaryFn::powi) {
      return rewriter.notifyMatchFailure(op, "Doesn't match powi");
    }

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> outputs = op.getOutputs();
    SmallVector<Value> newInputs;
    SmallVector<Value> newOutputs;
    if (allI8ElemType(inputs) && allI8ElemType(outputs)) {
      newInputs =
          normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, inputs);
      newOutputs =
          normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, outputs);
    } else if (allI16ElemType(inputs) && allI16ElemType(outputs)) {
      newInputs =
          normalizeSrcToTargetType<int16_t, Float32Type>(rewriter, inputs);
      newOutputs =
          normalizeSrcToTargetType<int16_t, Float32Type>(rewriter, outputs);
    } else {
      return rewriter.notifyMatchFailure(op, "powi type is not i8 nor i16");
    }
    Operation *newOp =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(rewriter, op->getLoc(),
                                                       hfusion::BinaryFn::powf,
                                                       newInputs, newOutputs);
    if (allI8ElemType(outputs)) {
      replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                     rewriter);
    } else if (allI16ElemType(outputs)) {
      replaceI16ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                      rewriter);
    }
    return success();
  }
};

using NormalizeRSqrtOp =
    NormalizeRSqrtOpTemplate<hfusion::ElemwiseUnaryOp, HFusionNormalizeRSqrtTraits>;
using NormalizeMulRecOp =
    NormalizeMulRecOpTemplate<linalg::ElemwiseBinaryOp, HFusionNormalizeMulRecTraits>;
using NormalizeDivVSToRec =
    NormalizeDivVSToRecTemplate<linalg::ElemwiseBinaryOp, HFusionNormalizeDivVSToRecTraits>;

void populateNormalizeMulRecPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeMulRecOp>(patterns.getContext());
}

void populateNormalizeArithmeticPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeNegToMul>(ctx);
  patterns.add<NormalizeDivVSToRec>(ctx);
  patterns.add<NormalizeVPowiToPowf>(ctx);
  patterns.add<NormalizeSubVSToVMulAndVAdd>(ctx);
  patterns.add<NormalizeRSqrtOp>(ctx);
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
