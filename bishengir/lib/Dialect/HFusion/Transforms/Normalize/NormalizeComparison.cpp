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
#include "bishengir/Transforms/Normalize/NormalizeComparisonTemplate.h"

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

struct HFusionIsInfNanNormalizeTraitsBase : public NormalizeTraitsBase {
  static bool isSupportedElementType(Type elemType) {
    return elemType.isF16() || elemType.isBF16() || elemType.isF32();
  }

  // HFusion isinf/isnan are not DestinationStyleOpInterface ops today.
  // They expose one input / one result through getInput()/getOutput(), so the
  // shared template cannot use the same DPS accessor shape as HIVM here.
  template <typename OpTy>
  static Value getInput(OpTy op) {
    return op.getInput();
  }

  template <typename OpTy>
  static Value getOutput(OpTy op) {
    return op.getOutput();
  }

  /// Converts the intermediate integer mask in `{0, 1}` to the final bool
  /// result tensor by reusing the op's destination and issuing an explicit
  /// `hfusion.cast(... -> i1)`.
  static Value lowerIntMaskToBoolResult(PatternRewriter &rewriter, Location loc,
                                        Value input, Value output) {
    auto roundingAttr =
        rewriter.getAttr<hfusion::RoundModeAttr>(hfusion::RoundMode::RINT);
    auto modeAttr = rewriter.getNamedAttr(hfusion::RoundModeAttr::getMnemonic(),
                                          roundingAttr);
    return rewriter
        .create<hfusion::CastOp>(loc, TypeRange(output), input, output, modeAttr)
        .getResult(0);
  }
};

/// Normalizes `hfusion.is_inf` to existing HFusion primitive ops:
///   bitcast -> vand(sign_mask) -> add(-inf_bits) -> bitcast -> abs
///   -> bitcast -> min_signed(1) -> mul(-1) -> add(1) -> cast(i1)
struct HFusionIsInfTraits : public HFusionIsInfNanNormalizeTraitsBase {
  static bool shouldNormalize(hfusion::IsInfOp) { return true; }
};

/// Normalizes `hfusion.is_nan` to existing HFusion primitive ops:
///   bitcast -> vand(sign_mask) -> add(-inf_bits) -> min_signed(1)
///   -> max_signed(0) -> cast(i1)
struct HFusionIsNanTraits : public HFusionIsInfNanNormalizeTraitsBase {
  static bool shouldNormalize(hfusion::IsNanOp) { return true; }
};

using NormalizeIsInfOp =
    mlir::NormalizeIsInfOpTemplate<hfusion::IsInfOp, HFusionIsInfTraits>;
using NormalizeIsNanOp =
    mlir::NormalizeIsNanOpTemplate<hfusion::IsNanOp, HFusionIsNanTraits>;

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
