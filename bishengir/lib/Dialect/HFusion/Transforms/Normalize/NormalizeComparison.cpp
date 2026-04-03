//===- NormalizeComparison.cpp ----------------------------------*- C++ -*-===//
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
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Transforms/Normalize/NormalizeComparison.h"

namespace mlir::hfusion {

/// normalize the specific cmp pattern to cast op
/// eg.
///  scalar = const 0
///  src0 = fill(scalar, dst) -> i8
///  y = hfusion.cmpi x, src0 {vne} ->  i1
/// is normalized to
///  y = hfusion.cast x -> i1
struct NormalizeCmpToCastOp : public OpRewritePattern<CompareOp> {
public:
  using OpRewritePattern<CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    llvm::SmallVector<Value> inputs = op.getInputs();
    auto isZeroFill = [&](Value src) {
      if (auto fillOp = src.getDefiningOp<linalg::FillOp>()) {
        if (auto cstOp =
                fillOp.getInputs()[0].getDefiningOp<arith::ConstantIntOp>()) {
          return op.getCompareFn() == CompareFn::vne && cstOp.value() == 0;
        }
      }
      return false;
    };
    bool lhsIsZero = isZeroFill(inputs[0]);
    bool rhsIsZero = isZeroFill(inputs[1]);
    bool isValidPattern = lhsIsZero || rhsIsZero;
    if (!isValidPattern) {
      return failure();
    }

    Value inputToCast = lhsIsZero && !rhsIsZero ? inputs[1] : inputs[0];
    hfusion::RoundMode rounding = hfusion::RoundMode::RINT;
    auto roundingAttr = rewriter.getAttr<hfusion::RoundModeAttr>(rounding);
    auto modeAttr = rewriter.getNamedAttr(hfusion::RoundModeAttr::getMnemonic(),
                                          roundingAttr);
    auto castOp = rewriter.create<hfusion::CastOp>(
        op->getLoc(), TypeRange(op.getResults()), ValueRange{inputToCast},
        ValueRange{op.getOutputs()[0]}, ArrayRef{modeAttr});
    rewriter.replaceOp(op, castOp);

    return success();
  }
};

struct HFusionCmpVneTraits : public NormalizeTraitsBase {
  static bool shouldNormalize(CompareOp op) {
    return op.hasPureTensorSemantics() && op.getCompareFn() == CompareFn::vne;
  }
};

using NormalizeCmpVneOp = mlir::NormalizeCmpVneOpTemplate<CompareOp, HFusionCmpVneTraits>;

struct NormalizeCmpOp : public OpRewritePattern<hfusion::CompareOp> {
public:
  using OpRewritePattern<CompareOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::CompareOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    // hfusion::CompareOp is used in cast op when casting to bool(int1).
    // Overflowmode annotation mark is useless in this case,
    // and would cause redundant vf in PreVectorizationFusion Pass.
    // So here we pair and delete overflow mark on CompareOp.
    auto overflowMode = getAnnotateOverflowMode(op);
    if (overflowMode.has_value()) {
      auto overflowModeAttr =
          utils::getAnnotateOpWithAttr(op->getResult(0), "overflow_mode");
      assert(overflowModeAttr.has_value());
      annotation::MarkOp markOp =
          dyn_cast<annotation::MarkOp>(overflowModeAttr.value());
      rewriter.eraseOp(markOp);
      return success();
    }
    return failure();
  }
};

/// get the constant integer value which is used mask sign bit
/// e.g. 8 bit mask value is 0b01111111
Value getSignMaskConstValue(PatternRewriter &rewriter, Location loc,
                            int bitwidth) {
  if (bitwidth == 32) {
    arith::ConstantOp maskCstOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(0x7FFFFFFF));
    return maskCstOp->getResults()[0];
  }
  if (bitwidth == 16) {
    arith::ConstantOp maskCstOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI16IntegerAttr(0x7FFF));
    return maskCstOp->getResults()[0];
  }
  llvm_unreachable("unsupported bitwidth");
}

/// get the complement of constant integer value of inf
/// e.g. 16 bit float inf is 0b0111110000000000
///      32 bit float inf is 0b01111111100000000000000000000000
Value getComplementOfInfConstValue(PatternRewriter &rewriter, Location loc,
                                   int bitwidth) {
  if (bitwidth == 32) {
    arith::ConstantOp maskCstOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(-1 * (0x7F800000)));
    return maskCstOp->getResults()[0];
  }
  if (bitwidth == 16) {
    arith::ConstantOp maskCstOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI16IntegerAttr(-1 * (0x7C00)));
    return maskCstOp->getResults()[0];
  }
  llvm_unreachable("unsupported bitwidth");
}

/// mask the sign bit of f32/f16 type input
Value maskSignBit(PatternRewriter &rewriter, Location loc, Value input) {
  Type elemType = getElementTypeOrSelf(input.getType());
  Type castType = rewriter.getIntegerType(elemType.getIntOrFloatBitWidth());
  // 1. init mask constant
  // 2. vdup(7FFF) : (I32/I16)
  auto fillInit =
      utils::createEmptyOpWithTargetElemType(rewriter, loc, input, castType);
  auto fillOp = rewriter.create<linalg::FillOp>(
      loc,
      ValueRange{getSignMaskConstValue(rewriter, loc,
                                       elemType.getIntOrFloatBitWidth())},
      ValueRange{fillInit});
  auto bitcastEmptyOp =
      utils::createEmptyOpWithTargetElemType(rewriter, loc, fillInit, castType);
  auto shapedType = dyn_cast_if_present<ShapedType>(input.getType());
  auto bitcastOp = rewriter.create<hfusion::BitcastOp>(
      loc, TypeRange{shapedType.clone(castType)}, ValueRange{input},
      ValueRange{bitcastEmptyOp});
  auto bitcastInit = bitcastOp->getResults()[0];
  auto vandInit = utils::createEmptyOp(rewriter, loc, bitcastInit);

  // 3. vand(input, input, vdup) : (I32/I16)
  auto vandOP =
      hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                              hfusion::BinaryFnAttr>(
          rewriter, loc, hfusion::BinaryFn::vand,
          ValueRange{bitcastInit, fillOp->getResults()[0]},
          ValueRange{vandInit});
  return vandOP->getResults()[0];
}

/// minus the input with integer value of inf
Value minusInfConstValue(PatternRewriter &rewriter, Location loc, Value input) {
  // namely add complement of integer value of inf
  // e.g. vadd(input, input, -1 * f16_inf).
  auto addInit = utils::createEmptyOp(rewriter, loc, input);
  Type elemType = getElementTypeOrSelf(input.getType());
  auto addOp = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
      rewriter, loc, linalg::BinaryFn::add,
      ValueRange{input, getComplementOfInfConstValue(
                            rewriter, loc, elemType.getIntOrFloatBitWidth())},
      ValueRange{addInit});
  return addOp->getResults()[0];
}

struct NormalizeIsInfOp : public OpRewritePattern<hfusion::IsInfOp> {
public:
  using OpRewritePattern<hfusion::IsInfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::IsInfOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Type elemType = getElementTypeOrSelf(input.getType());
    if (!elemType.isF16() && !elemType.isBF16() && !elemType.isF32()) {
      return failure();
    }

    // step 1: mask sign bit.
    // 1. vdup(7FFF) : (I32/I16)
    auto loc = op->getLoc();
    auto maskedSignValue = maskSignBit(rewriter, loc, input);

    // step 2: compared with negtive Infinity
    // 3.vadd(input, input, neg_inf_bitcast_as_int).
    auto minusInfValue = minusInfConstValue(rewriter, loc, maskedSignValue);
    // 4.vabs(input, input) : (F16/F32)
    auto rebitcastEmptyOp = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, minusInfValue, elemType);
    auto shapedType = dyn_cast_if_present<ShapedType>(input.getType());
    auto rebitcastOp = rewriter.create<hfusion::BitcastOp>(
        loc, TypeRange{shapedType.clone(elemType)}, ValueRange{minusInfValue},
        ValueRange{rebitcastEmptyOp});
    Value rebitcastInit = rebitcastOp->getResults()[0];
    auto absInit = utils::createEmptyOp(rewriter, loc, rebitcastInit);
    auto absOP = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                        linalg::UnaryFn, linalg::UnaryFnAttr>(
        rewriter, loc, linalg::UnaryFn::abs, ValueRange{rebitcastInit},
        ValueRange{absInit});

    // 5.vmin(input, input, 1) : (I32/I16)
    Type castType = rewriter.getIntegerType(elemType.getIntOrFloatBitWidth());
    auto bitcastOpForMinEmptyOp = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, absOP->getResults()[0], castType);
    auto bitcastOpForMin = rewriter.create<hfusion::BitcastOp>(
        loc, TypeRange{shapedType.clone(castType)},
        ValueRange{absOP->getResults()[0]}, ValueRange{bitcastOpForMinEmptyOp});
    Value bitcastOpForMinInit = bitcastOpForMin.getResults()[0];
    auto minInit = utils::createEmptyOp(rewriter, loc, bitcastOpForMinInit);
    arith::ConstantOp posOneOp = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, 1));
    auto minOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::min_signed,
            ValueRange{bitcastOpForMinInit, posOneOp->getResults()[0]},
            ValueRange{minInit});

    // 6.vmuls(input, input, -1) : (I32/I16)
    arith::ConstantOp negOneOp = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, -1));
    auto mulOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange({minOp->getResults()[0], negOneOp->getResults()[0]}),
            minOp->getResults()[0]);

    // 7.vadds(input, input, 1) : (I32/I16)
    auto addsOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange({mulOp->getResults()[0], posOneOp->getResults()[0]}),
            mulOp->getResults()[0]);

    // 8.cast(input, int->i1)
    auto roundingAttr =
        rewriter.getAttr<hfusion::RoundModeAttr>(hfusion::RoundMode::RINT);
    auto modeAttr = rewriter.getNamedAttr(hfusion::RoundModeAttr::getMnemonic(),
                                          roundingAttr);
    hfusion::CastOp castToDst = rewriter.create<hfusion::CastOp>(
        loc, TypeRange(op.getOutput()), addsOp->getResults()[0], op.getOutput(),
        modeAttr);
    rewriter.replaceOp(op, castToDst);
    return success();
  }
};

struct NormalizeIsNanOp : public OpRewritePattern<hfusion::IsNanOp> {
public:
  using OpRewritePattern<hfusion::IsNanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::IsNanOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Type elemType = getElementTypeOrSelf(input.getType());
    if (!elemType.isF16() && !elemType.isBF16() && !elemType.isF32()) {
      return failure();
    }

    // step 1: mask sign bit.
    // 1. vdup(7FFF) : (I32/I16)
    auto loc = op->getLoc();
    auto maskedSignValue = maskSignBit(rewriter, loc, input);

    // step 2: compared with negtive Infinity
    // 3.vadd(input, input, neg_inf_bitcast_as_int).
    auto minusInfValue = minusInfConstValue(rewriter, loc, maskedSignValue);

    // step3: change temp result to 1 which is > 1
    // vmin(input, input, 1) : (I32/I16)
    Type castType = rewriter.getIntegerType(elemType.getIntOrFloatBitWidth());
    arith::ConstantOp posOneOp = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, 1));
    auto minOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::min_signed,
            ValueRange{minusInfValue, posOneOp->getResults()[0]},
            ValueRange{minusInfValue});

    // step4. change temp result to 0 which is < 0
    // vmax(input, input, 0) : (I32/I16)
    arith::ConstantOp zeroOp = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, 0));
    auto maxOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::max_signed,
            ValueRange({minOp->getResults()[0], zeroOp->getResults()[0]}),
            minOp->getResults()[0]);

    // step5. cast int32 to int1
    // cast(input, i32 -> i1)
    auto roundingAttr =
        rewriter.getAttr<hfusion::RoundModeAttr>(hfusion::RoundMode::RINT);
    auto modeAttr = rewriter.getNamedAttr(hfusion::RoundModeAttr::getMnemonic(),
                                          roundingAttr);
    hfusion::CastOp castToDst = rewriter.create<hfusion::CastOp>(
        loc, TypeRange(op.getOutput()), maxOp->getResults()[0], op.getOutput(),
        modeAttr);
    rewriter.replaceOp(op, castToDst);
    return success();
  }
};

/// normalize i8/i32 CompareOp
///   i8 -> f16
///   i32 -> i64 (except vne and veq)
/// e.g.
///   hfusion.compare ins(%src1, %src2 : tensor<6x6xi32>, tensor<6x6xi32>)
/// is normalized to
///   %cast1 = hfusion.cast %src1 : tensor<6x6xi32> to tensor<6x6xi64>
///   %cast2 = hfusion.cast %src2 : tensor<6x6xi32> to tensor<6x6xi64>
///   hfusion.compare ins(%cast1, %cast2 : tensor<6x6xi64>, tensor<6x6xi64>)
struct NormalizeI8I32CmpOp : public OpRewritePattern<CompareOp> {
public:
  using OpRewritePattern<CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];
    Type lhsElemType = getElementTypeOrSelf(lhs.getType());
#ifndef NDEBUG
    Type rhsElemType = getElementTypeOrSelf(rhs.getType());
    assert(lhsElemType == rhsElemType && "lhs and rhs elemType mismatch");
#endif

    Type targetType = rewriter.getI64Type();
    hfusion::CompareFn cmpFn = op.getCompareFn();
    if (lhsElemType.isInteger(8)) {
      targetType = rewriter.getF16Type();
    } else if (lhsElemType.isInteger(32) && cmpFn != hfusion::CompareFn::vne &&
               cmpFn != hfusion::CompareFn::veq) {
      targetType = rewriter.getI64Type();
    } else {
      return failure();
    }

    hfusion::RoundMode rounding =
        utils::selectRoundMode<hfusion::RoundMode>(lhsElemType, targetType);
    Value castLhs = hfusion::castTo(rewriter, lhs, targetType, rounding);
    Value castRhs = hfusion::castTo(rewriter, rhs, targetType, rounding);
    auto newCmpOp =
        createCmpOp(rewriter, op->getLoc(), castLhs, castRhs, cmpFn);
    rewriter.replaceOp(op, newCmpOp);
    return success();
  }
};

/// normalize x xor y into (!(x&y)) & (x|y)
struct NormalizeXorOp : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::BinaryFn::vxor) {
      return failure();
    }

    auto inputs = op.getDpsInputs();
    auto outs = op.getDpsInits();
    assert(!outs.empty() && isa<ShapedType>(outs[0].getType()));

    // x|y
    auto emptyVorOp = utils::createEmptyOp(rewriter, op->getLoc(), outs[0]);
    auto orOp =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, op->getLoc(), hfusion::BinaryFn::vor, inputs,
            ValueRange(emptyVorOp));
    // x&y
    auto emptyVandOp = utils::createEmptyOp(rewriter, op->getLoc(), outs[0]);
    auto vandOp =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, op->getLoc(), hfusion::BinaryFn::vand, inputs,
            ValueRange(emptyVandOp));

    // !(x&y)
    auto vnotOp =
        hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp, hfusion::UnaryFn,
                               hfusion::UnaryFnAttr>(
            rewriter, op->getLoc(), hfusion::UnaryFn::vnot,
            ValueRange{vandOp->getResults()}, ValueRange(vandOp->getResults()));

    // xorop
    auto emptyVxorOp = utils::createEmptyOp(rewriter, op->getLoc(), outs[0]);
    auto vxorOp =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, op->getLoc(), hfusion::BinaryFn::vand,
            ValueRange{vnotOp->getResults()[0], orOp->getResults()[0]},
            ValueRange(emptyVxorOp));
    rewriter.replaceOp(op, vxorOp);
    return success();
  }
};

/// normalize shift i8 as bellow
/// eg.
///   %res = shift %src : i8
/// is normalized to
///   %tmp0 = cast %src i8 to i16
///   %tmp1 = shift %tmp0 : i16
///   %res = cast %tmp1 i16 to i8
struct NormalizeShiftI8ToI16
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto fun = op.getFun();
    if (!(fun == hfusion::BinaryFn::shli || fun == hfusion::BinaryFn::shrsi ||
          fun == hfusion::BinaryFn::shrui)) {
      return failure();
    }

    Value input = op.getDpsInputs()[0];
    Type inputElemType = getElementTypeOrSelf(input.getType());
    if (!inputElemType.isInteger(8)) {
      return failure();
    }

    auto loc = op->getLoc();
    auto targetElemType = rewriter.getI16Type();
    auto shift = op.getDpsInputs()[1];
    hfusion::TypeFn cast_integer_type =
        (fun == hfusion::BinaryFn::shrui || fun == hfusion::BinaryFn::shli)
            ? hfusion::TypeFn::cast_unsigned
            : hfusion::TypeFn::cast_signed;
    Value inputOfI16 =
        hfusion::castTo(rewriter, input, targetElemType, cast_integer_type);
    Value shiftOfI16 =
        hfusion::castTo(rewriter, shift, targetElemType, cast_integer_type);

    auto shiftInit = utils::createEmptyOp(rewriter, loc, inputOfI16);
    Value resOfI16 =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, fun, ValueRange{inputOfI16, shiftOfI16},
            ValueRange(shiftInit))
            ->getResults()[0];

    auto srcElemType = rewriter.getI8Type();
    auto selectMode =
        utils::selectRoundMode<hfusion::RoundMode>(targetElemType, srcElemType);
    auto roundMode = (fun == hfusion::BinaryFn::shli)
                         ? hfusion::RoundMode::TRUNCWITHOVERFLOW
                         : selectMode;
    auto resOfI8 =
        hfusion::castTo(rewriter, resOfI16, srcElemType, roundMode,
                        std::nullopt, true, false, cast_integer_type);

    rewriter.replaceOp(op, resOfI8);
    return success();
  }
};

void populateNormalizeI8I32CmpPatterns(RewritePatternSet &patterns) {
  if (!archIsRegbased)
    patterns.add<NormalizeI8I32CmpOp>(patterns.getContext());
}

void populateNormalizeCmpToCastPatterns(RewritePatternSet &patterns) {
  if (!archIsRegbased)
    patterns.add<NormalizeCmpToCastOp>(patterns.getContext());
}

void populateNormalizeComparisonCleanupPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeCmpOp>(ctx);
  patterns.add<NormalizeIsInfOp>(ctx);
  patterns.add<NormalizeIsNanOp>(ctx);
}

void populateNormalizeBitwiseComparisonPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  if (!archIsRegbased) {
    patterns.add<NormalizeXorOp>(ctx);
    patterns.add<NormalizeShiftI8ToI16>(ctx);
  }
}

void populateNormalizeShiftI8ToI16(RewritePatternSet &patterns) {
  if (archIsRegbased)
    patterns.add<NormalizeShiftI8ToI16>(patterns.getContext());
}

void populateNormalizeCmpVnePatterns(RewritePatternSet &patterns) {
  if (!archIsRegbased)
    patterns.add<NormalizeCmpVneOp>(patterns.getContext());
}
} // namespace mlir::hfusion
