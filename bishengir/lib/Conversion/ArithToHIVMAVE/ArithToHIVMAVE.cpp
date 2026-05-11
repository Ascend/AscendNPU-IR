//===- ArithToHIVMAVE.cpp - convert arith op to ave op --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/ArithToHIVMAVE/ArithToHIVMAVE.h"
#include "bishengir/Conversion/HFusionToHIVM/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-arith-to-hivmave"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOHIVMAVE
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;
using namespace hivmave;
using namespace mlir::hivm::util;

static bool isEligibleVectorType(Type type) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    unsigned factor = vecType.getElementType().isInteger(64) ? 2 : 1;
    return vecType.getShape().size() == 1 &&
           vecType.getShape()[0] * vecType.getElementTypeBitWidth() <=
               factor * util::VL_BITS;
  }
  return false;
}
namespace {
template <typename ArithUnaryOp, typename HivmVFUnaryOp>
struct UnaryOpPattern : public OpConversionPattern<ArithUnaryOp> {
  using OpConversionPattern<ArithUnaryOp>::OpConversionPattern;

  using ArithUnaryOpAdaptor =
      typename OpConversionPattern<ArithUnaryOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(ArithUnaryOp op, ArithUnaryOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = adaptor.getOperands()[0];
    auto resType = op.getResult().getType();

    // TODO: need to check whether the operation is in the vector function.
    VectorType resVecType = mlir::dyn_cast<VectorType>(resType);
    if (!resVecType || resVecType.getShape().size() != 1) {
      LLVM_DEBUG(DBGS() << "legalize" << ArithUnaryOp::getOperationName()
                        << "failed."
                        << "\n";);
      return failure();
    }

    Value mask = hivmave::findReuseableMaskOrCreateOne(op, resVecType, rewriter);

    auto res = rewriter.create<HivmVFUnaryOp>(loc, resType, src, mask, nullptr);
    rewriter.replaceOp(op, res);
    return success();
  }
};

template <typename ArithBinaryOp, typename HivmVFBinaryOp>
struct BinaryOpPattern : public OpConversionPattern<ArithBinaryOp> {
  using OpConversionPattern<ArithBinaryOp>::OpConversionPattern;

  using ArithBinaryOpAdaptor =
      typename OpConversionPattern<ArithBinaryOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(ArithBinaryOp op, ArithBinaryOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto resType = op.getResult().getType();

    // TODO: need to check whether the operation is in the vector function.
    VectorType resVecType = mlir::dyn_cast<VectorType>(resType);
    if (!resVecType || resVecType.getShape().size() != 1) {
      LLVM_DEBUG(DBGS() << "legalize" << ArithBinaryOp::getOperationName()
                        << "failed."
                        << "\n";);
      return failure();
    }

    Value mask = hivmave::findReuseableMaskOrCreateOne(op, resVecType, rewriter);

    HivmVFBinaryOp res;

    if constexpr (std::is_same_v<ArithBinaryOp, arith::DivUIOp>) {
      res =
        rewriter.create<HivmVFBinaryOp>(loc, resType, lhs, rhs, mask, TypeFnAttr::get(op->getContext(), TypeFn::cast_unsigned), nullptr);
    }
    else if constexpr (std::is_same_v<ArithBinaryOp, arith::DivSIOp> || std::is_same_v<ArithBinaryOp, arith::DivFOp>) {
      res =
        rewriter.create<HivmVFBinaryOp>(loc, resType, lhs, rhs, mask, TypeFnAttr::get(op->getContext(), TypeFn::cast_signed), nullptr);
    }
    else{
      res =
        rewriter.create<HivmVFBinaryOp>(loc, resType, lhs, rhs, mask, nullptr);
    }

    // Set this attribute for later use in reduceInitElimination.
    if (op->hasAttr("reductionOp")) {
      res->setAttr("reductionOp", rewriter.getUnitAttr());
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

template <typename ArithMulExtendOp, typename HivmVFVMULLOp>
struct ArithMulExtendOpPattern : public OpConversionPattern<ArithMulExtendOp> {
  using OpConversionPattern<ArithMulExtendOp>::OpConversionPattern;

  using ArithBinaryOpAdaptor =
      typename OpConversionPattern<ArithMulExtendOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(ArithMulExtendOp op, ArithBinaryOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lowResType = op.getResult(0).getType();
    auto highResType = op.getResult(1).getType();
    auto resType = op.getResultTypes();
    VectorType lowResVecType = mlir::dyn_cast<VectorType>(lowResType);
    VectorType highResVecType = mlir::dyn_cast<VectorType>(highResType);

    if (!lowResVecType || !highResVecType || resType.size() !=2) {
      LLVM_DEBUG(DBGS() << "legalize" << ArithMulExtendOp::getOperationName()
                        << "failed."
                        << "\n";);
      return failure();
    }

    Value mask = hivmave::findReuseableMaskOrCreateOne(op, lowResVecType, rewriter);

    HivmVFVMULLOp res =
        rewriter.create<HivmVFVMULLOp>(loc, resType, lhs, rhs, mask, nullptr);
    rewriter.replaceOp(op, {res->getResult(0), res->getResult(1)});
    return success();
  }
};

struct RoundOpPattern : public OpConversionPattern<math::RoundOp> {
  using OpConversionPattern<math::RoundOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::RoundOp op, math::RoundOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resType = op.getResult().getType();
    auto src = adaptor.getOperands()[0];

    VectorType resVecType = mlir::dyn_cast<VectorType>(resType);
    if (!resVecType) {
      return failure();
    }

    Value mask = hivmave::findReuseableMaskOrCreateOne(op, resVecType, rewriter);

    hfusion::RoundModeAttr roundingAttr =
        op->template getAttrOfType<hfusion::RoundModeAttr>("round_mode");
    if (!roundingAttr)
      roundingAttr = hfusion::RoundModeAttr::get(op->getContext(),
                                                 hfusion::RoundMode::ROUND);
    hivm::RoundMode hvRndMode =
        hfusion_conversion_utils::mapRoundModeHFusionToHiVM(
            roundingAttr.getValue());
    hivm::RoundModeAttr rnd =
        hivm::RoundModeAttr::get(op->getContext(), hvRndMode);

    if (roundingAttr.getValue() == hfusion::RoundMode::TRUNCWITHOVERFLOW ||
        roundingAttr.getValue() == hfusion::RoundMode::ODD) {
      llvm_unreachable("Cannot support rounding mode");
    }

    hivmave::VFVtrcOp res =
        rewriter.create<hivmave::VFVtrcOp>(loc, resVecType, src, rnd, mask);
    rewriter.replaceOp(op, res);
    return success();
  }
};

template <typename OpToBeConverted>
struct VFTypeConvertionPattern : public OpConversionPattern<OpToBeConverted> {
  using OpConversionPattern<OpToBeConverted>::OpConversionPattern;

  using VFTypeConversionOpAdaptor =
      typename OpConversionPattern<OpToBeConverted>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpToBeConverted op, VFTypeConversionOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value in = op.getIn();
    Value out = op.getOut();
    Type inType = in.getType();
    Type outType = out.getType();
    if (!(isa<VectorType>(inType) && isa<VectorType>(outType)))
      return failure();
    VectorType inVecType = cast<VectorType>(inType);
    VectorType outVecType = cast<VectorType>(outType);
    Type inElemType = inVecType.getElementType();
    Type outElemType = outVecType.getElementType();
    // When convert data type, the bit width of mask should be consistent with
    // the data type of the larger. For example, if convert i32 to i8, the bit
    // width of mask should be 4, if convert i8 to i32, the bit width of mask
    // should also be 4.
    Value adjustedMask = hivmave::findReuseableMaskOrCreateOne(
        op, outVecType, rewriter);

    // TODO: set these dynamically
    VCVT_PartTypeAttr part =
        VCVT_PartTypeAttr::get(rewriter.getContext(), VCVT_PartType::PART_EVEN);
    VCVT_PPTypeAttr pp =
        VCVT_PPTypeAttr::get(rewriter.getContext(), VCVT_PPType::PP0);

    // When SPR.CTRL[60]=1'b0 (previously init in InitEntryKernel), set `sat` to
    // false for non-saturation mode for `VCVTFI`, `VCVTII` and `VCVTFF` from
    // wider dynamic range to narrower dynamic range.
    // For `VCVTFF` with fdst == f32, e.g. VCVTFF.f162f32, and VTRC.f32 of
    // `VTRC`, only non-saturation mode is supported, with all these attributes
    // neglected.
    BoolAttr sat = rewriter.getBoolAttr(false);

    // Check enable_saturate
    if (auto enableSaturate = op->template getAttrOfType<mlir::BoolAttr>("enable_saturate")) {
      sat = enableSaturate;
    }

    hfusion::RoundMode roundingMode;
    hfusion::RoundModeAttr roundingAttr =
        op->template getAttrOfType<hfusion::RoundModeAttr>("round_mode");
    if (!roundingAttr)
      roundingMode = hfusion::RoundMode::ROUND;
    else
      roundingMode = roundingAttr.getValue();

    hivm::RoundMode hvRndMode =
        hfusion_conversion_utils::mapRoundModeHFusionToHiVM(roundingMode);
    hivm::RoundModeAttr rnd =
        hivm::RoundModeAttr::get(op->getContext(), hvRndMode);

    hfusion::UnsignedMode unsignedMode;
    hfusion::UnsignedModeAttr unsignedAttr =
        op->template getAttrOfType<hfusion::UnsignedModeAttr>("unsigned_mode");
    if (!unsignedAttr)
      unsignedMode = hfusion::UnsignedMode::SI2SI;
    else
      unsignedMode = unsignedAttr.getValue();
 
    hivm::UnsignedMode hvUniMode =
        hfusion_conversion_utils::mapUnsignedModeHFusionToHiVM(unsignedMode);
    hivm::UnsignedModeAttr uni =
        hivm::UnsignedModeAttr::get(op->getContext(), hvUniMode);

    Operation *newOp;
    if constexpr (std::is_same_v<OpToBeConverted, arith::ExtUIOp>) {
      if (inElemType.isSignlessInteger(8) && outElemType.isSignlessInteger(32))
        newOp = rewriter.create<VFExtUIOp>(loc, outVecType, in, adjustedMask,
                                           nullptr, pp);
      else if (outElemType.isSignlessInteger(64) &&
              (inElemType.isSignlessInteger(1) ||
               inElemType.isSignlessInteger(8) ||
               inElemType.isSignlessInteger(16))) {
        auto newElemType = rewriter.getIntegerType(32);
        auto newOutVecType = VectorType::get(outVecType.getShape(), newElemType);
        // i8/i16/bool -> i64
        Value maskForI32 = hivmave::findReuseableMaskOrCreateOne(op, newOutVecType,
                                                                     rewriter);
        auto i32Op = inElemType.isSignlessInteger(8)
            ? rewriter.create<VFExtUIOp>(loc, newOutVecType, in, maskForI32,
                                         nullptr, pp)
            : rewriter.create<VFExtUIOp>(loc, newOutVecType, in, maskForI32,
                                         part, nullptr);
        // i32 -> i64
        Value maskForI64 = hivmave::findReuseableMaskOrCreateOne(i32Op, outVecType,
                                                                 rewriter);
        newOp = rewriter.create<VFExtUIOp>(loc, outVecType, i32Op.getResult(),
                                           maskForI64, part, nullptr);
      } else
        newOp = rewriter.create<VFExtUIOp>(loc, outVecType, in, adjustedMask,
                                           part, nullptr);
    } else if constexpr (std::is_same_v<OpToBeConverted, arith::ExtSIOp>) {
      if (inElemType.isSignlessInteger(8) && outElemType.isSignlessInteger(32))
        newOp = rewriter.create<VFExtSIOp>(loc, outVecType, in, adjustedMask,
                                           nullptr, pp);

      // In the initial pass, the transition to i32 is added, but both the
      // PreVectorizationFusion and Canonicalizer passes optimize this process.
      // Therefore, the final choice is to add the i32 transition in the HIVM-level
      // pass ConvertArithToHIVMAVE.
      else if ((inElemType.isSignlessInteger(1) &&
                outElemType.isSignlessInteger(64)) ||
               (inElemType.isSignlessInteger(8) &&
                outElemType.isSignlessInteger(64)) ||
               (inElemType.isSignlessInteger(16) &&
                outElemType.isSignlessInteger(64))) {
        auto newElemType = rewriter.getIntegerType(32);
        auto newOutVecType = VectorType::get(outVecType.getShape(), newElemType);
        // i8/i16/bool -> i64
        Value maskForI32 = hivmave::findReuseableMaskOrCreateOne(op, newOutVecType,
                                                                     rewriter);
        auto i32Op = inElemType.isSignlessInteger(8)
            ? rewriter.create<VFExtSIOp>(loc, newOutVecType, in, maskForI32,
                                         nullptr, pp)
            : rewriter.create<VFExtSIOp>(loc, newOutVecType, in, maskForI32,
                                         part, nullptr);
        // i32 -> i64
        Value maskForI64 = hivmave::findReuseableMaskOrCreateOne(i32Op, outVecType,
                                                                 rewriter);
        newOp = rewriter.create<VFExtSIOp>(loc, outVecType, i32Op.getResult(), 
                                           maskForI64, part, nullptr);
      } else
        newOp = rewriter.create<VFExtSIOp>(loc, outVecType, in, adjustedMask,
                                           part, nullptr);
    } else if constexpr (std::is_same_v<OpToBeConverted, arith::ExtFOp>) {
      newOp =
          rewriter.create<VFExtFOp>(loc, outVecType, in, adjustedMask, part);
    } else if constexpr (std::is_same_v<OpToBeConverted, arith::TruncIOp>) {
      const bool isI32ToI8 = inElemType.isSignlessInteger(32)
                            && outElemType.isSignlessInteger(8);
      const bool isI64ToI16 = inElemType.isSignlessInteger(64)
                            && outElemType.isSignlessInteger(16);
      const bool isI64ToI8 = inElemType.isSignlessInteger(64)
                            && outElemType.isSignlessInteger(8);
      if (isI32ToI8) {
        if (sat) {
          hivm::UnsignedModeAttr u2uAttr = hivm::UnsignedModeAttr::get(
                                          op->getContext(), hivm::UnsignedMode::UI2UI);
          hivm::UnsignedModeAttr s2uAttr = hivm::UnsignedModeAttr::get(
                                            op->getContext(), hivm::UnsignedMode::SI2UI);
          if (hvUniMode == hivm::UnsignedMode::SI2UI) {
            newOp = rewriter.create<VFTruncIOp>(loc, outVecType, in, adjustedMask,
                                            sat, nullptr, pp, s2uAttr);
          } else if (hvUniMode == hivm::UnsignedMode::UI2UI) {
            newOp = rewriter.create<VFTruncIOp>(loc, outVecType, in, adjustedMask,
                                            sat, nullptr, pp, u2uAttr);
          } else {
            newOp = rewriter.create<VFTruncIOp>(loc, outVecType, in, adjustedMask,
                                            sat, nullptr, pp, nullptr);
          }
        } else {
          newOp = rewriter.create<VFTruncIOp>(loc, outVecType, in, adjustedMask,
                                            sat, nullptr, pp, nullptr);
        }
      } else if (isI64ToI8 || isI64ToI16) {
        if (sat) {
          hivm::UnsignedModeAttr u2uAttr = hivm::UnsignedModeAttr::get(
                                          op->getContext(), hivm::UnsignedMode::UI2UI);
          hivm::UnsignedModeAttr u2sAttr = hivm::UnsignedModeAttr::get(
                                          op->getContext(), hivm::UnsignedMode::UI2SI);
          hivm::UnsignedModeAttr s2uAttr = hivm::UnsignedModeAttr::get(
                                          op->getContext(), hivm::UnsignedMode::SI2UI);
          auto newElemType = rewriter.getIntegerType(32);
          auto newOutVecType = VectorType::get(outVecType.getShape(), newElemType);
          Value maskForU32 = hivmave::findReuseableMaskOrCreateOne(op, newOutVecType,
                                                                  rewriter);
          if (isI64ToI16) {
            if (hvUniMode == hivm::UnsignedMode::SI2UI) {
              auto u32Op = rewriter.create<VFTruncIOp>(loc, newOutVecType, in, maskForU32,
                                                      sat, part, nullptr, s2uAttr);
              newOp = rewriter.create<VFTruncIOp>(loc, outVecType, u32Op.getResult(),
                                            adjustedMask, sat, part, nullptr, u2uAttr);
            } else if (hvUniMode == hivm::UnsignedMode::UI2SI) {
              auto u32Op = rewriter.create<VFTruncIOp>(loc, newOutVecType, in, maskForU32,
                                                      sat, part, nullptr, u2uAttr);
              newOp = rewriter.create<VFTruncIOp>(loc, outVecType, u32Op.getResult(),
                                            adjustedMask, sat, part, nullptr, u2sAttr);
            } else if (hvUniMode == hivm::UnsignedMode::UI2UI) {
              auto u32Op = rewriter.create<VFTruncIOp>(loc, newOutVecType, in, maskForU32,
                                                      sat, part, nullptr, u2uAttr);
              newOp = rewriter.create<VFTruncIOp>(loc, outVecType, u32Op.getResult(),
                                            adjustedMask, sat, part, nullptr, u2uAttr);
            } else {
              auto u32Op = rewriter.create<VFTruncIOp>(loc, newOutVecType, in, maskForU32,
                                                      sat, part, nullptr, s2uAttr);
              newOp = rewriter.create<VFTruncIOp>(loc, outVecType, u32Op.getResult(),
                                            adjustedMask, sat, part, nullptr, u2sAttr);
            }
          } else {
            if (hvUniMode == hivm::UnsignedMode::SI2UI) {
              auto u32Op = rewriter.create<VFTruncIOp>(loc, newOutVecType, in, maskForU32,
                                                      sat, part, nullptr, s2uAttr);
              newOp = rewriter.create<VFTruncIOp>(loc, outVecType, u32Op.getResult(),
                                            adjustedMask, sat, nullptr, pp, u2uAttr);
            } else if (hvUniMode == hivm::UnsignedMode::UI2SI) {
              // vcvt instr donot support int to int8,
              // use u64->u32->u8->f16->s8 to transfer in saturation mode.
              auto u32Op = rewriter.create<VFTruncIOp>(loc, newOutVecType, in, maskForU32,
                                                      sat, part, nullptr, u2uAttr);
              auto u8Op = rewriter.create<VFTruncIOp>(loc, outVecType, u32Op.getResult(),
                                            adjustedMask, sat, nullptr, pp, u2uAttr);
              auto f16ElemType = rewriter.getF16Type();
              auto f16OutVecType = VectorType::get(outVecType.getShape(), f16ElemType);
              Value maskForF16 = hivmave::findReuseableMaskOrCreateOne(
                                op, f16OutVecType, rewriter);
              auto f16Op = rewriter.create<VFUIntToFpOp>(loc, f16OutVecType, u8Op.getResult(),
                                                        adjustedMask, nullptr, part);
              newOp = rewriter.create<VFFpToSIntOp>(loc, outVecType, f16Op.getResult(),
                                                    adjustedMask, rnd, sat, part);
            } else if (hvUniMode == hivm::UnsignedMode::UI2UI) {
              auto u32Op = rewriter.create<VFTruncIOp>(loc, newOutVecType, in, maskForU32,
                                                      sat, part, nullptr, u2uAttr);
              newOp = rewriter.create<VFTruncIOp>(loc, outVecType, u32Op.getResult(),
                                            adjustedMask, sat, nullptr, pp, u2uAttr);
            } else {
              auto u32Op = rewriter.create<VFTruncIOp>(loc, newOutVecType, in, maskForU32,
                                                      sat, part, nullptr, s2uAttr);
              auto u8Op = rewriter.create<VFTruncIOp>(loc, outVecType, u32Op.getResult(),
                                            adjustedMask, sat, nullptr, pp, u2uAttr);
              auto f16ElemType = rewriter.getF16Type();
              auto f16OutVecType = VectorType::get(outVecType.getShape(), f16ElemType);
              Value maskForF16 = hivmave::findReuseableMaskOrCreateOne(
                                op, f16OutVecType, rewriter);
              auto f16Op = rewriter.create<VFUIntToFpOp>(loc, f16OutVecType, u8Op.getResult(),
                                                        adjustedMask, nullptr, part);
              newOp = rewriter.create<VFFpToSIntOp>(loc, outVecType, f16Op.getResult(),
                                                    adjustedMask, rnd, sat, part);
            }
          }
        } else {
          auto newElemType = rewriter.getIntegerType(32);
          auto newOutVecType = VectorType::get(outVecType.getShape(), newElemType);
          // i64 -> i32
          Value maskForI32 = hivmave::findReuseableMaskOrCreateOne(op, newOutVecType,
                                                                  rewriter);
          auto i32Op = rewriter.create<VFTruncIOp>(loc, newOutVecType, in, maskForI32,
                                                  sat, part, nullptr, nullptr);
          // i32 -> i8/i16
          newOp = outElemType.isSignlessInteger(8)
              ? rewriter.create<VFTruncIOp>(loc, outVecType, i32Op.getResult(),
                                            adjustedMask, sat, nullptr, pp, nullptr)
              : rewriter.create<VFTruncIOp>(loc, outVecType, i32Op.getResult(),
                                            adjustedMask, sat, part, nullptr, nullptr);
        }
      } else
        newOp = rewriter.create<VFTruncIOp>(loc, outVecType, in, adjustedMask,
                                            sat, part, nullptr, uni);
    } else if constexpr (std::is_same_v<OpToBeConverted, arith::TruncFOp>) {
      newOp = rewriter.create<VFTruncFOp>(loc, outVecType, in, adjustedMask,
                                          rnd, sat, part);
    } else if constexpr (std::is_same_v<OpToBeConverted, arith::SIToFPOp>) {
      if (inElemType.isSignlessInteger(64))
        newOp = rewriter.create<VFSIntToFpOp>(loc, outVecType, in, adjustedMask,
                                              rnd, part);
      else if (inElemType.getIntOrFloatBitWidth() ==
               outElemType.getIntOrFloatBitWidth())
        newOp = rewriter.create<VFSIntToFpOp>(loc, outVecType, in, adjustedMask,
                                              rnd, nullptr);
      else
        newOp = rewriter.create<VFSIntToFpOp>(loc, outVecType, in, adjustedMask,
                                              nullptr, part);
    } else if constexpr (std::is_same_v<OpToBeConverted, arith::UIToFPOp>) {
      if (inElemType.isSignlessInteger(8) && outElemType.isF16())
        newOp = rewriter.create<VFUIntToFpOp>(loc, outVecType, in, adjustedMask,
                                              nullptr, part);
      else if (inElemType.isSignlessInteger(16)){
        auto newElemType = rewriter.getIntegerType(32);
        auto newOutVecType = VectorType::get(outVecType.getShape(), newElemType);
        // u16 -> i32          
        Value maskForI32 = hivmave::findReuseableMaskOrCreateOne(op, newOutVecType,
                                                                    rewriter);
        auto i32Op = rewriter.create<VFExtUIOp>(loc, newOutVecType, in, maskForI32,
                                        part, nullptr);
        // i32 -> f32
        Value maskForf32 = hivmave::findReuseableMaskOrCreateOne(i32Op, outVecType,
                                                                rewriter);
        newOp = rewriter.create<VFSIntToFpOp>(loc, outVecType, i32Op.getResult(), maskForf32,
                                              rnd, nullptr);
      } else
        newOp = rewriter.create<VFUIntToFpOp>(loc, outVecType, in, adjustedMask,
                                              rnd, part);
    } else if constexpr (std::is_same_v<OpToBeConverted, arith::FPToSIOp>) {
      if (inElemType.isF16() && outElemType.isSignlessInteger(32))
        newOp = rewriter.create<VFFpToSIntOp>(loc, outVecType, in, adjustedMask,
                                              rnd, nullptr, part);
      else if (inElemType.getIntOrFloatBitWidth() ==
               outElemType.getIntOrFloatBitWidth())
        newOp = rewriter.create<VFFpToSIntOp>(loc, outVecType, in, adjustedMask,
                                              rnd, rewriter.getBoolAttr(true),
                                              nullptr);
      else
        newOp = rewriter.create<VFFpToSIntOp>(loc, outVecType, in, adjustedMask,
                                              rnd, sat, part);
    } else if constexpr (std::is_same_v<OpToBeConverted, arith::FPToUIOp>) {
      newOp = rewriter.create<VFFpToUIntOp>(loc, outVecType, in, adjustedMask,
                                            rnd, sat, part);
    } else
      return failure();

    if (auto elementAlignmentAttr = op->getParentOp()->getAttr(
            mlir::utils::elementAlignmentBitWidth))
      newOp->setAttr(mlir::utils::elementAlignmentBitWidth,
                                  elementAlignmentAttr);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Conversion pattern for dense arith.constant.
///
/// Broadcast pattern handles splat dense arith.constant
///
///  BEFORE:
///  ```mlir
///  %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
///  %7 = hivm.hir.ave.pge <ALL> :
///  %8 = hivm.hir.ave.add %6, %cst, %7 : vector<64xf32>, vector<64xi1>
///  ```
///
///  AFTER:
///  ```mlir
///  %cst = arith.constant 1.000000e+00 : f32
///  %8 = hivm.hir.ave.pge <ALL> : vector<64xi1>
///  %9 = hivm.hir.ave.broadcast %cst, %8
///  %10 = hivm.hir.ave.add %6, %9, %7 : vector<64xf32>, vector<256xi1>
///
///  Another example of two dim arith.constant:
///  BEFORE:
///  ```mlir
///  %cst = arith.constant dense<1.000000e+00> : vector<4x16xf32>
///  ```
///
///  AFTER:
///  ```mlir
///  %cst = arith.constant 1.000000e+00 : f32
///  %8 = hivm.hir.ave.pge <ALL> : vector<64xi1>
///  %9 = hivm.hir.ave.broadcast %cst, %8
///
struct ConstantOpToHivmBroadcastLowering
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp constantOp,
                  arith::ConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType = dyn_cast<VectorType>(constantOp.getResult().getType());
    auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValueAttr());

    if (!tileType || !denseAttr || !denseAttr.isSplat())
      return failure();
    if (!(mlir::utils::isValidHIVMTileVectorType(tileType) ||
          mlir::utils::isValidTwoDimVectorType(tileType)))
      return failure();
    auto tileElementType = tileType.getElementType();

    auto loc = constantOp.getLoc(); 
    Value scalarValue;
    if (llvm::isa<FloatType>(tileElementType)) {
      auto initVal = denseAttr.getSplatValue<APFloat>();
      scalarValue = rewriter.create<arith::ConstantOp>(
          loc, tileElementType,
          rewriter.getFloatAttr(tileElementType, initVal));
    }
    if (llvm::isa<IntegerType>(tileElementType)) {
      auto initVal = denseAttr.getSplatValue<APInt>();
      if (tileElementType.isInteger(1)) {
        bool allTrue = denseAttr.getValues<bool>()[0];
        scalarValue = createMaskByPGE(tileType, rewriter, loc, allTrue);
        rewriter.replaceOp(constantOp, scalarValue);
        return success();
      } else if (tileElementType.isInteger(8)) {
        scalarValue = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerType(16),
            rewriter.getIntegerAttr(rewriter.getIntegerType(16),
                                    initVal.sext(16)));
      } else
        scalarValue = rewriter.create<arith::ConstantOp>(
            loc, tileElementType,
            rewriter.getIntegerAttr(tileElementType, initVal));
    }
    Operation *brcOp = nullptr;
    // Check if the tile element type is FP8 (f8e4m3fn or f8e5m2)
    if ((tileElementType.isFloat8E4M3FN() || tileElementType.isFloat8E5M2()) &&
        tileType.getRank() <= 1) {
      auto resType = constantOp.getResult().getType();
      VectorType resVecType = mlir::dyn_cast<VectorType>(resType);
      if (!resVecType || resVecType.getShape().size() != 1)
        return failure();
      auto vecSize = resVecType.getShape();
      auto i8VecTy = VectorType::get({vecSize}, rewriter.getI8Type());
      // Check if the constant attribute is zero (splat value)
      if (denseAttr.getSplatValue<APFloat>().isZero()) {
        // Keep the same process logic for scalar value as i8 did above
        scalarValue = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerType(16),
            rewriter.getIntegerAttr(rewriter.getIntegerType(16), 0));
      }
      // Broadcast the scalar value to i8 vector type and cast back
      brcOp = getBroadcastOp(scalarValue, i8VecTy, rewriter, loc);
      rewriter.replaceOpWithNewOp<vector::BitCastOp>(
          constantOp, TypeRange{tileType}, brcOp->getResult(0));
      return success();
    }

    if (tileType.getRank() > 1) {
      auto trimmedType = trimNonScalableUnitDims(tileType);
      trimmedType = VectorType::get({trimmedType.getNumElements()},
                                    trimmedType.getElementType());
      brcOp = getBroadcastOp(scalarValue, trimmedType, rewriter, loc);
      if (tileType != trimmedType) {
        auto ucc = rewriter.create<UnrealizedConversionCastOp>(
            loc, tileType, brcOp->getResult(0));
        rewriter.replaceOp(constantOp, ucc.getResult(0));
      } else
        rewriter.replaceOp(constantOp, brcOp);
    } else {
      brcOp = getBroadcastOp(scalarValue, tileType, rewriter, loc);

      rewriter.replaceOp(constantOp, brcOp);
    }
    return success();
  }
};

/// Conversion pattern for dense arith.constant.
///
/// Plt pattern handles binary dense arith.constant
///
///  BEFORE:
///  ```mlir
///  %cst = arith.constant dense<[1, 1, 1, 1, 1, 0, 0,..., 0]>: vector<64xi8>
///  ```
///
///  AFTER:
///  ```mlir
///  %c5 = arith.constant 5 : index
///  %res, %new_true_shape = ave.hir.plt %c5 : vector<64xi1>, index
///  %0 = ave.hir.pge <ALL> : vector<64xi1>
///  %c1_i8 = arith.constant 1 : i8
///  %1 = ave.hir.broadcast %c1_i8, %0 : i8, vector<64xi1> -> vector<64xi8>
///  %c0_i8 = arith.constant 0 : i8
///  %2 = ave.hir.broadcast %c0_i8, %0 : i8, vector<64xi1> -> vector<64xi8>
///  %3 = ave.hir.vsel %res, %1, %2 : vector<64xi1>, vector<64xi8>
///  ```
struct ConstantOpToHivmPLTLowering
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  template <typename T>
  struct APTypeHelper;

  template <>
  struct APTypeHelper<APInt> {

    static APInt createDefault(Type elementType) {
      return APInt(elementType.getIntOrFloatBitWidth(), 0);
    }

    static Value createScalar(ConversionPatternRewriter &rewriter, const Location &loc, VectorType tileType,
                                                  Type elementType, const APInt& value, Value mask) {
      // To avoid known issues: https://codehub-y.huawei.com/CompilerKernel/BiShengCompiler/AscendNPU-IR/issues/81
      if (elementType.isInteger(8)) {
        auto constValue = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerType(16), rewriter.getIntegerAttr(rewriter.getIntegerType(16), value.zext(16)));
        return rewriter.create<hivmave::VFBroadcastScalarMaskOp>(
            loc, tileType, constValue, mask);
      }
        
      auto constValue = rewriter.create<arith::ConstantOp>(
            loc, elementType, rewriter.getIntegerAttr(elementType, value));
      return rewriter.create<hivmave::VFBroadcastScalarMaskOp>(
            loc, tileType, constValue, mask);
    }
  };

  template <>
  struct APTypeHelper<llvm::APFloat> {

    static APFloat createDefault(Type elementType) {
      auto floatType = mlir::cast<FloatType>(elementType);
      return APFloat(floatType.getFloatSemantics());
    }

    static Value createScalar(ConversionPatternRewriter &rewriter, const Location &loc, VectorType tileType,
                                                  Type elementType, const APFloat& value, Value mask) {
      auto constValue = rewriter.create<arith::ConstantOp>(
            loc, elementType, rewriter.getFloatAttr(elementType, value));
      return rewriter.create<hivmave::VFBroadcastScalarMaskOp>(
            loc, tileType, constValue, mask);
    }
  };

  template<typename T>
  LogicalResult extractTwoValuesAndCreateBoolArray(DenseElementsAttr denseAttr,
                                                   int64_t vectorLength,
                                                   T& value1, T& value2,
                                                   SmallVector<bool>& boolValues) const {

    SmallVector<T> distinctValues;
    for (int i = 0; i < vectorLength; ++i) {
      T currentValue = denseAttr.getValues<T>()[i];
      if (llvm::find(distinctValues, currentValue) == distinctValues.end()) {
        distinctValues.push_back(currentValue);
        if (distinctValues.size() > 2) {
          return failure();
        }
      }
    }

    if (distinctValues.size() != 2) {
      return failure();
    }

    value1 = distinctValues[0];
    value2 = distinctValues[1];

    boolValues.clear();
    boolValues.reserve(vectorLength);
    for (int i = 0; i < vectorLength; ++i) {
      T currentValue = denseAttr.getValues<T>()[i];
      boolValues.push_back(currentValue == value1);
    }

    return success();
  }

  struct SegmentChange {
    size_t position;
    bool value;
  };

  SmallVector<SegmentChange> collectSegmentChanges(ArrayRef<bool> boolValues) const {
    SmallVector<SegmentChange> changes;
    
    if (boolValues.empty()) return changes;
    
    bool currentValue = boolValues[0];
    
    for (size_t i = 1; i < boolValues.size(); ++i) {
      if (boolValues[i] != currentValue) {
        changes.push_back({i - 1, currentValue});
        currentValue = boolValues[i];
      }
    }
    return changes;
  }

  template <typename TileElementType, typename APType>
  LogicalResult processConstantOp(arith::ConstantOp constantOp,
                                DenseElementsAttr denseAttr,
                                TileElementType tileElementType,
                                VectorType tileType,
                                ConversionPatternRewriter &rewriter) const {
    
    // To handle float binary case like [5.0, 5.0, 5.0, 6.0, 6.0, 6.0,..., 6.0]
    // Handwritten cases can be handled, but real cases have not been encountered.
    // Possesses the corresponding ability but has not undergone the corresponding test.
    if (!mlir::isa<IntegerType>(tileElementType) || !(tileElementType.isInteger(8) || 
                  tileElementType.isInteger(16) || tileElementType.isInteger(32))) {
      return rewriter.notifyMatchFailure(constantOp,
            "Currently, only support integer type i8/i16/i32.");
    }

    auto constUsers = llvm::to_vector<6>(constantOp->getUsers());
    sort(constUsers.begin(), constUsers.end());
    Operation *firstUser = *constUsers.begin();
    auto loc = firstUser->getLoc();

    int64_t vectorLength = tileType.getNumElements();
    APType value1 = APTypeHelper<APType>::createDefault(tileElementType);
    APType value2 = APTypeHelper<APType>::createDefault(tileElementType);
    SmallVector<bool> boolValues;

    if (failed(extractTwoValuesAndCreateBoolArray(denseAttr, vectorLength, 
                                                 value1, value2, boolValues))) {
      return failure();
    }

    Value resOp;
    auto segmentChanges = collectSegmentChanges(boolValues);
    auto init = segmentChanges.front();

    // Could not handle complex binary case like [5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 5, 5,...]
    if (segmentChanges.size() > 1) {
      return rewriter.notifyMatchFailure(constantOp,
            "Currently, do not handle complex case.");
    }

    auto i1VectorType = VectorType::get(vectorLength, rewriter.getI1Type());
    resOp = rewriter.create<hivmave::VFPltOp>(
                loc, i1VectorType, rewriter.getIndexType(), 
                rewriter.create<arith::ConstantIndexOp>(loc, init.position + 1))->getResult(0);
    
    Value mask = hivmave::findReuseableMaskOrCreateOne(constantOp, i1VectorType, rewriter);
    auto scalarValue1 = APTypeHelper<APType>::createScalar(rewriter, loc, tileType, tileElementType, value1, mask);
    auto scalarValue2 = APTypeHelper<APType>::createScalar(rewriter, loc, tileType, tileElementType, value2, mask);

    resOp = rewriter.create<VFSelectOp>(loc, tileType, resOp, scalarValue1, scalarValue2);

    rewriter.replaceOp(constantOp, resOp);
    
    return success();
  }

  LogicalResult
  matchAndRewrite(arith::ConstantOp constantOp,
                  arith::ConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType = dyn_cast<VectorType>(constantOp.getResult().getType());
    auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValueAttr());

    if (!tileType || tileType.getRank() != 1)
      return failure();

    if (!denseAttr || denseAttr.isSplat())
      return failure();
    
    auto elementType = tileType.getElementType();
    // 1.Convert binary dense vector into boolean array.
    // 2.Combine instructions based on boolean array.
    // 3.Use vsel to convert vector from boolean to real type. 
    if (mlir::isa<IntegerType>(elementType)) {
      IntegerType tileElementType = dyn_cast<IntegerType>(elementType);
      return processConstantOp<IntegerType, APInt>(constantOp,
        denseAttr, tileElementType, tileType, rewriter);
    } else if (mlir::isa<FloatType>(elementType)){
      FloatType tileElementType = dyn_cast<FloatType>(elementType);
      return processConstantOp<FloatType, APFloat>(constantOp,
        denseAttr, tileElementType, tileType, rewriter);
    } else {
      return rewriter.notifyMatchFailure(constantOp,
          "we currently handle integer-type or float16/32");
    }
  }
};

///
/// VCI/VCP pattern handles integer constant vectors that can be generated
/// from existing vector instructions (VCI/VCP)
///
///  BEFORE:
///  ```mlir
///  %cst = arith.constant dense<0, 1, ..., 63> : vector<64xi32>
///  ```
///
///  AFTER:
///  ```mlir
///  %cst = hivm.hir.ave.vci 0 : vector<64xi32>
///  ```
struct ConstantOpToHivmVCIVCPLowering
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  template <typename T>
  static arith::ConstantOp ConstantCreateHelper(OpBuilder &rewriter,
      Location loc, Type elementType, T value) {
    if (mlir::isa<IndexType>(elementType) && std::is_integral_v<T>) {
      return rewriter.create<arith::ConstantOp>(loc, elementType,
          IntegerAttr::get(elementType, value));
    } else if (mlir::isa<IntegerType>(elementType) && std::is_integral_v<T>) {
      return rewriter.create<arith::ConstantOp>(loc,
          IntegerAttr::get(elementType, value));
    } else if (mlir::isa<FloatType>(elementType)) {
      return rewriter.create<arith::ConstantOp>(loc,
          FloatAttr::get(elementType, value));
    } else {
      return arith::ConstantOp();
    }
  }

  template <typename T>
  struct APTypeHelper;

  template <>
  struct APTypeHelper<APInt> {
    using ElementType = int64_t;

    static ElementType extract(const APInt &x) {
      return x.getSExtValue();
    }

    static ElementType extractZ(const APInt &x) {
      return x.getZExtValue();
    }

    static void divrem(const ElementType &dividedNum,
                       const ElementType &divisorNum, 
                       APInt &newValue, ElementType &remainder) {
      APInt divided(64, dividedNum);
      APInt::sdivrem(divided, divisorNum, newValue, remainder);
    }

    static bool isInteger(const ElementType &x) {
      (void)x;
      return true;
    }

    static std::tuple<bool, ElementType, ElementType, std::vector<APInt>> 
    extractArithmeticSequenceParams(DenseElementsAttr denseAttr, unsigned valueCheckRange) {
      auto values = llvm::to_vector<0>(denseAttr.getValues<APInt>());
      std::vector<APInt> quotients;
      auto extractFunc = APTypeHelper<APInt>::extract;
      if (denseAttr.getElementType().isSignlessInteger())
        extractFunc = APTypeHelper<APInt>::extractZ;
      long addValue = extractFunc(values[0]);
      long mulValue = extractFunc(values[1]) - addValue;
      
      if (mulValue == 0)
        return std::make_tuple(false, 0, 0, quotients);
      
      for (unsigned i = 0; i < valueCheckRange; ++i) {
        APInt newValue;
        long remainder = 0;

        long scoreDiff = extractFunc(values[i]) - addValue;
        divrem(scoreDiff, mulValue, newValue, remainder);
        
        if (remainder != 0 || newValue.slt(0))
          return std::make_tuple(false, 0, 0, quotients);

        quotients.push_back(newValue);
      }
      
      return std::make_tuple(true, addValue, mulValue, quotients);
    }
  };
  template <>
  struct APTypeHelper<APFloat> {
    using ElementType = float;

    static ElementType extract(const APFloat &v) {
      return v.convertToFloat();   // RianF 用double吗
    }

    static void divrem(const ElementType &dividendNum,
                       const ElementType &divisorNum,
                       APInt &quotientAPInt, ElementType &remainderFloat) {
      APFloat dividend(dividendNum);
      APFloat divisor(divisorNum);
      
      // Integer quotient
      APFloat quotient = dividend;
      quotient.divide(divisor, APFloat::rmNearestTiesToEven);
      bool ignored = false;
      APSInt quotientAPSInt(64, true);
      quotient.convertToInteger(quotientAPSInt, APFloat::rmTowardZero,
                                &ignored);
      quotientAPInt = static_cast<llvm::APInt>(quotientAPSInt);

      // remainder
      dividend.remainder(divisor);        
      remainderFloat = dividend.convertToFloat();
    }

    static bool isInteger(const ElementType &x) {
      APFloat tmp(x);
      return tmp.isInteger();
    }

    static std::tuple<bool, ElementType, ElementType, std::vector<APInt>> 
    extractArithmeticSequenceParams(DenseElementsAttr denseAttr, unsigned valueCheckRange) {
      auto values = llvm::to_vector<0>(denseAttr.getValues<APFloat>());
      std::vector<APInt> quotients;
      
      double epsilon;
      if (denseAttr.getElementType().isF16() || denseAttr.getElementType().isBF16())
        epsilon = 1e-3;
      else if (denseAttr.getElementType().isF32())
        epsilon = 1e-4;
      else {
        // currently, only support f16 or f32
        return std::make_tuple(false, 0.0f, 0.0f, quotients);
      }
      
      std::vector<ElementType> diffs;
      diffs.reserve(valueCheckRange - 1);
      std::transform(values.begin() + 1, values.begin() + valueCheckRange, values.begin(), std::back_inserter(diffs),
                     [](const APFloat &curr, const APFloat &prev) { return extract(curr) - extract(prev); });
      
      auto [min_iter, max_iter] = std::minmax_element(diffs.begin(), diffs.end());
      ElementType dmin = *min_iter;
      ElementType dmax = *max_iter;

      if (std::abs(dmax - dmin) > 2 * epsilon) {
        return std::make_tuple(false, 0.0f, 0.0f, quotients);
      }

      ElementType addValue = extract(values[0]);
      ElementType mulValue = std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size();

      // Avoid divide 0
      if (std::abs(mulValue) < 1e-7) {
        return std::make_tuple(false, 0.0f, 0.0f, quotients);
      }
      
      for (unsigned i = 0; i < valueCheckRange; ++i) {
        double x = static_cast<double>(i);
        double y_actual = extract(values[i]);
        double y_expected = addValue + mulValue * x;
        double error = std::abs(y_actual - y_expected);
        
        if (error > epsilon) {
          return std::make_tuple(false, 0.0f, 0.0f, quotients);
        }
        
        quotients.emplace_back(64, static_cast<int64_t>(std::round(x)));
      }
      
      return std::make_tuple(true, addValue, mulValue, quotients);
    }
  };


  /// helper class to wrap the logic of integer pattern checking
  template <typename T>
  struct PatternValueCheckerBase {
    virtual ~PatternValueCheckerBase() = default;
    virtual bool checkValue(unsigned index, int64_t value) = 0;
    virtual bool isSupportI8()  const = 0;
    virtual bool isSupportI16() const = 0;
    virtual bool isSupportI32() const = 0;
    virtual bool isSupportI64() const = 0;
    virtual bool isSupportF8() const = 0;
    virtual bool isSupportF16() const = 0;
    virtual bool isSupportF32() const = 0;
    // common base data and functions
    bool isInvalid = false;
    bool processValue(unsigned index, int64_t value) {
      if (!isInvalid && !static_cast<T *>(this)->checkValue(index, value)) {
        isInvalid = true;
      }
      return !isInvalid;
    }
    operator bool() const { return !isInvalid; }
    bool checkDataType(Type elementType) {
      if (isInvalid)
        return !isInvalid;
      auto bitWidth = elementType.getIntOrFloatBitWidth();
      if (elementType.isInteger()) {
        switch (bitWidth) {
        case 8:
          isInvalid = !isSupportI8();
          break;
        case 16:
          isInvalid = !isSupportI16();
          break;
        case 32:
          isInvalid = !isSupportI32();
          break;
        case 64:
          isInvalid = !isSupportI64();
          break;
        default:
          isInvalid = true;
          break;
        }
      } else if (llvm::isa<FloatType>(elementType)) {
        switch (bitWidth) {
        case 8:   
          isInvalid = !isSupportF8();
          break;
        case 16:
          isInvalid = !isSupportF16();
          break;
        case 32:
          isInvalid = !isSupportF32();
          break;
        default:
          isInvalid = true;
          break;
        }
      }
      return !isInvalid;
    }
  };
  template<typename ElementType>
  struct VCIPattern : public PatternValueCheckerBase<VCIPattern<ElementType>> {
    // 0, 1, 2, 3, 4, 5, 6, 7, ...
    bool checkValue(unsigned index, int64_t value) override {
      return value == index;
    }
    bool isSupportI8()  const override { return true; }
    bool isSupportI16() const override { return true; }
    bool isSupportI32() const override { return true; }
    bool isSupportI64() const override { return true; }
    bool isSupportF8()  const override { return true; } 
    bool isSupportF16() const override { return true; }
    bool isSupportF32() const override { return true; }
    Operation *createInstr(ConversionPatternRewriter &rewriter, Location loc,
                           Type destType, Type elementType,
                           ElementType &mulValue, ElementType &addValue) {
      /** Pattern1 */
      ElementType baseValue = 0;
      if (mulValue == 1 || mulValue == -1) {
        baseValue = addValue;
        addValue = 0;
      }
      auto baseValueOp = ConstantCreateHelper<ElementType>(
          rewriter, loc, elementType, baseValue);
      if (!baseValueOp) {
        return nullptr;
      }
      auto vciType = hivmave::VCIType::INCREASE;
      /** Pattern1 */
      if (mulValue == -1) {
        mulValue = 1;
        vciType = hivmave::VCIType::DECREASE;
      }
      auto vciOp = rewriter.create<hivmave::VFVCIOp>(
          loc, destType, baseValueOp.getResult(), vciType);
      return vciOp;
    }
  };
  template <unsigned pat, unsigned chnX, unsigned toY>
  struct VCPPattern
      : public PatternValueCheckerBase<VCPPattern<pat, chnX, toY>> {
    bool checkValue(unsigned index, int64_t value) override {
      return value == (index / chnX) * toY + index % chnX;
    }
    bool isSupportI8()  const override { return false; }
    bool isSupportI16() const override { return true; }
    bool isSupportI32() const override { return true; }
    bool isSupportI64() const override { return false; }
    bool isSupportF8()  const override { return false; }
    bool isSupportF16() const override { return false; }
    bool isSupportF32() const override { return false; }
    unsigned getPatID() const { return pat; }
  };

  // the base version (without typename... Args) handles leaf case (1 pattern)
  // the composite version (with the variadic Args) handles >1 pattern case
  template <typename PatternT>
  static unsigned checkDataType(Type elementType, PatternT &p) {
    if (p.checkDataType(elementType))
      return 1;
    return 0;
  }
  template <typename PatternT, typename... Args>
  static unsigned checkDataType(Type elementType, PatternT &p, Args &&...args) {
    return checkDataType(elementType, p) + checkDataType(elementType, args...);
  }
  template <typename PatternT>
  static unsigned processValue(unsigned index, int64_t value, PatternT &p) {
    if (p.processValue(index, value))
      return 1;
    return 0;
  }
  template <typename PatternT, typename... Args>
  static unsigned processValue(unsigned index, int64_t value, PatternT &p,
                               Args &&...args) {
    return processValue(index, value, p) + processValue(index, value, args...);
  }

  template<typename ElementType>
  Operation *createMulAddIfNeeded(ConversionPatternRewriter &rewriter,
                                  Operation *defineOp,
                                  arith::ConstantOp constantOp,
                                  Type elementType, ElementType mulValue,
                                  ElementType addValue) const {
    /** Pattern1 */
    if (mulValue == 1 && addValue == 0) {
      return defineOp;
    }
    /** Pattern2 */
    VectorType resultType = cast<VectorType>(constantOp.getType());
    auto loc = constantOp.getLoc();
    auto mask = createMaskByPGE(resultType, rewriter, loc);
    if (mulValue != 1) {
      auto mulValueOp = ConstantCreateHelper<ElementType>(
          rewriter, loc, elementType, mulValue);
      if (!mulValueOp) 
        return nullptr;
      auto vmulOp = rewriter.create<hivmave::VFMulsOp>(
          loc, resultType, defineOp->getResult(0), mulValueOp.getResult(), mask,
          nullptr);
      defineOp = vmulOp;
    }
    if (addValue != 0) {
      auto addValueOp = ConstantCreateHelper<ElementType>(
          rewriter, loc, elementType, addValue);
      if (!addValueOp) 
        return nullptr;
      auto vaddOp = rewriter.create<hivmave::VFAddsOp>(
          loc, resultType, defineOp->getResult(0), addValueOp.getResult(), mask,
          nullptr);
      defineOp = vaddOp;
    }
    return defineOp;
  }

  /** Pattern3 */
  template<typename APType>
  unsigned adjustForTailPadding(const SmallVector<APType, 0> &values) const {
    // we may have vectors with a constant (e.g., 0) at the end
    // in case the result is feed into a masked operation
    // (e.g., vector.gather using only the first 16 value of i16 vector)
    // we still want to be able to match these instructions
    using ElementType = typename APTypeHelper<APType>::ElementType;
    unsigned range = values.size();
    ElementType tailPaddingValue = APTypeHelper<APType>::extract(values.back());
    if (APTypeHelper<APType>::extract(values[range - 2]) == tailPaddingValue) {
      // we indeed has tail padding
      // the last two has already been checked, so we skip them
      range = values.size() - 2;
      for (unsigned i = values.size() - 3; i >= 1; --i) {
        if (APTypeHelper<APType>::extract(values[i]) != tailPaddingValue)
          break;
        range = i;
      }
    } else if (tailPaddingValue == 0 && range > 0) {
      // we have only one tail padding
      --range;
    }
    return range;
  }

  /** Pattern3 */
  template <typename TileElementType, typename APType>
  Operation *handleTailPadding(ConversionPatternRewriter &rewriter,
                               arith::ConstantOp constantOp,
                               VectorType tileType,
                               const SmallVector<APType, 0> &values,
                               unsigned valueCheckRange, 
                               TileElementType tileElemType,
                               Operation *defineOp) const {
    using ElementType = typename APTypeHelper<APType>::ElementType;
                                
    auto loc = constantOp.getLoc();
    ElementType tailPaddingValue = APTypeHelper<APType>::extract(values.back());
    auto scalarValue = ConstantCreateHelper<ElementType>(
        rewriter, loc, tileElemType, tailPaddingValue);
    if (!scalarValue)
      return nullptr;
    
    auto brcOp = getBroadcastOp(scalarValue, tileType, rewriter, loc);
    auto maskType =
        VectorType::get({tileType.getNumElements()}, rewriter.getI1Type());
    auto pgePatternAttr =
        getPgePatternAttr(rewriter, valueCheckRange, values.size());

    Operation *p = nullptr;
    if (failed(pgePatternAttr)) {
      auto resType = cast<VectorType>(constantOp.getResult().getType());
      VectorType resVecType = mlir::dyn_cast<VectorType>(resType);
      if (!resVecType || !isOneDimLikeVecType(resVecType))
        llvm_unreachable("Not a 1D-like vector");

      int boundCst = resVecType.getShape().back();
      if (boundCst < 256) {
        auto trueShape = rewriter.create<arith::ConstantIndexOp>(loc, boundCst);
        p = rewriter.create<hivmave::VFPltOp>(
            loc, maskType, rewriter.getIndexType(), trueShape);

      } else
        llvm_unreachable(
            "TODO support irregular length for arith.constant dense values");

    } else
      p = rewriter.create<hivmave::VFPgeOp>(loc, maskType,
                                            pgePatternAttr.value());

    auto sel = rewriter.create<hivmave::VFSelectOp>(
        loc, tileType, p->getResult(0), defineOp->getResult(0),
        brcOp->getResult(0));
    return sel;
  }

  template <typename TileElementType, typename APType>
  LogicalResult processConstantOp(arith::ConstantOp constantOp,
                                DenseElementsAttr denseAttr,
                                TileElementType tileElementType,
                                VectorType tileType,
                                ConversionPatternRewriter &rewriter) const {
      using ElementType = typename APTypeHelper<APType>::ElementType;

      VCIPattern<ElementType> p_vci;
      VCPPattern<0, 4, 8> p_vcp0;
      VCPPattern<1, 4, 16> p_vcp1;
      VCPPattern<2, 4, 32> p_vcp2;
      VCPPattern<3, 8, 16> p_vcp3;
      VCPPattern<4, 8, 32> p_vcp4;
#define PATTERN_LIST p_vci, p_vcp0, p_vcp1, p_vcp2, p_vcp3, p_vcp4
      if (checkDataType(tileElementType, PATTERN_LIST) == 0)
        return failure();

      // try to decompose the integer vector as <0, 1, 2, 3, ...> * A + B
      auto values = llvm::to_vector<0>(denseAttr.getValues<APType>());
      
      unsigned valueCheckRange = adjustForTailPadding<APType>(values);
      if (valueCheckRange == 1)
        // patterns like [x, 0, 0, ...] are not supported here
        // (we should use a single vsel for them)
        return failure();

      auto [isArithmeticSeq, addValue, mulValue, quotients] = 
          APTypeHelper<APType>::extractArithmeticSequenceParams(denseAttr, valueCheckRange);

      if (!isArithmeticSeq || mulValue == 0)
        return failure();

      for (unsigned i = 0; i < valueCheckRange; ++i) {
        if (processValue(i, quotients[i].getSExtValue(), PATTERN_LIST) == 0)
            return failure();
      }
      Operation *defineOp = nullptr;
      if (p_vci)
        defineOp = p_vci.createInstr(rewriter, constantOp.getLoc(),
            denseAttr.getType(), tileElementType, mulValue, addValue);
      else
        llvm_unreachable("TODO implement arith.constant dense -> VCP lowering");
      if (defineOp == nullptr)
        return failure();

      defineOp = createMulAddIfNeeded<ElementType>(rewriter, defineOp,
          constantOp, tileElementType, mulValue, addValue);
      if (defineOp == nullptr)
        return failure();

      if (valueCheckRange != values.size())
        defineOp = handleTailPadding<TileElementType, APType>(
            rewriter, constantOp, tileType, values,
            valueCheckRange, tileElementType, defineOp);
      if (defineOp == nullptr)
        return failure();
      rewriter.replaceOp(constantOp, defineOp);
      return success();
    }
  
  /** Support for converting arithmetic sequences to VCI
    Pattern1: [2, 3, 4, ...] -> vci
    Pattern2: [2, 4, 6, ...] -> vci + vmul + vadd
    Pattern3: [2, 4, 6, ..., 6] -> vci + vmul + vadd + vsel
   */
  LogicalResult
  matchAndRewrite(arith::ConstantOp constantOp,
                  arith::ConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType = dyn_cast<VectorType>(constantOp.getResult().getType());
    if (!tileType || !mlir::utils::isValidHIVMTileVectorType(tileType))
      return failure();

    // only consider dense arith.constant that has enough elements
    auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValueAttr());
    if (!denseAttr || denseAttr.getNumElements() < 4)
      return failure();
    // we currently handle integer-type indices or float16/32 
    Type eltType = denseAttr.getElementType();
    if (mlir::isa<IntegerType>(eltType)) {
      IntegerType tileElementType = dyn_cast<IntegerType>(eltType);
      return processConstantOp<IntegerType, APInt>(constantOp,
        denseAttr, tileElementType, tileType, rewriter);
    } else if (mlir::isa<FloatType>(eltType)) {
      FloatType tileElementType = dyn_cast<FloatType>(eltType);
      return processConstantOp<FloatType, APFloat>(constantOp,
        denseAttr, tileElementType, tileType, rewriter);
    } else {
      return rewriter.notifyMatchFailure(constantOp,
          "we currently handle integer-type indices or float16/32");
    }
  }
};

template <typename ArithCmpOp>
static hivmave::CmpType getHIVMCmpType(ArithCmpOp op) {
  hivmave::CmpType cmpType;
  if constexpr (std::is_same_v<ArithCmpOp, arith::CmpFOp>) {
    arith::CmpFPredicate pred = op.getPredicate();
    // ignore orderness for now
    if (pred == arith::CmpFPredicate::OEQ ||
        pred == arith::CmpFPredicate::UEQ) {
      cmpType = CmpType::EQ;
    } else if (pred == arith::CmpFPredicate::OGT ||
               pred == arith::CmpFPredicate::UGT) {
      cmpType = CmpType::GT;
    } else if (pred == arith::CmpFPredicate::OGE ||
               pred == arith::CmpFPredicate::UGE) {
      cmpType = CmpType::GE;
    } else if (pred == arith::CmpFPredicate::OLT ||
               pred == arith::CmpFPredicate::ULT) {
      cmpType = CmpType::LT;
    } else if (pred == arith::CmpFPredicate::OLE ||
               pred == arith::CmpFPredicate::ULE) {
      cmpType = CmpType::LE;
    } else if (pred == arith::CmpFPredicate::ONE ||
               pred == arith::CmpFPredicate::UNE) {
      cmpType = CmpType::NE;
    } else {
      llvm_unreachable("unsupported CmpFPredicate!");
    }
  } else if constexpr (std::is_same_v<ArithCmpOp, arith::CmpIOp>) {
    arith::CmpIPredicate pred = op.getPredicate();
    if (pred == arith::CmpIPredicate::eq) {
      cmpType = CmpType::EQ;
    } else if (pred == arith::CmpIPredicate::sgt) {
      cmpType = CmpType::GT;
    } else if (pred == arith::CmpIPredicate::sge) {
      cmpType = CmpType::GE;
    } else if (pred == arith::CmpIPredicate::slt) {
      cmpType = CmpType::LT;
    } else if (pred == arith::CmpIPredicate::sle) {
      cmpType = CmpType::LE;
    } else if (pred == arith::CmpIPredicate::ule) {
      cmpType = CmpType::ULE;
    } else if (pred == arith::CmpIPredicate::uge) {
      cmpType = CmpType::UGE;
    } else if (pred == arith::CmpIPredicate::ult) {
      cmpType = CmpType::ULT;
    } else if (pred == arith::CmpIPredicate::ugt) {
      cmpType = CmpType::UGT;
    } else if (pred == arith::CmpIPredicate::ne) {
      cmpType = CmpType::NE;
    } else {
      llvm_unreachable("unsupported CmpIPredicate!");
    }
  }
  return cmpType;
}

template <typename ArithCmpOp, typename HivmVFCmpOp>
struct CmpOpPattern : public OpConversionPattern<ArithCmpOp> {
  using OpConversionPattern<ArithCmpOp>::OpConversionPattern;

  using ArithCmpOpAdaptor = typename OpConversionPattern<ArithCmpOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(ArithCmpOp op, ArithCmpOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto resType = op.getResult().getType();

    VectorType resVecType = mlir::dyn_cast<VectorType>(resType);
    VectorType srcVecType = mlir::dyn_cast<VectorType>(lhs.getType());

    if (!resVecType || !srcVecType || resVecType.getShape().size() != 1)
      return failure();

    Value mask = hivmave::findReuseableMaskOrCreateOne(op, resVecType, rewriter);

    if (srcVecType.getElementType().isInteger(1) &&
        resVecType.getElementType().isInteger(1)) {
      auto elemBitWidthAttr = MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8);

      if constexpr (std::is_same_v<ArithCmpOp, arith::CmpIOp>) {
        auto pred = op.getPredicate();
        if (pred == arith::CmpIPredicate::uge) {
          auto notRhs = rewriter.create<hivmave::PregNotOp>(
              loc, resType, elemBitWidthAttr, rhs, mask);
          auto res = rewriter.create<hivmave::PregOrOp>(
              loc, resType, elemBitWidthAttr, lhs, notRhs, mask);
          rewriter.replaceOp(op, res);
          return success();
        } else if (pred == arith::CmpIPredicate::ugt) {
          auto notRhs = rewriter.create<hivmave::PregNotOp>(
              loc, resType, elemBitWidthAttr, rhs, mask);
          auto res = rewriter.create<hivmave::PregAndOp>(
              loc, resType, elemBitWidthAttr, lhs, notRhs, mask);
          rewriter.replaceOp(op, res);
          return success();
        } else if (pred == arith::CmpIPredicate::ule) {
          auto notLhs = rewriter.create<hivmave::PregNotOp>(
              loc, resType, elemBitWidthAttr, lhs, mask);
          auto res = rewriter.create<hivmave::PregOrOp>(
              loc, resType, elemBitWidthAttr, notLhs, rhs, mask);
          rewriter.replaceOp(op, res);
          return success();
        } else if (pred == arith::CmpIPredicate::ult) {
          auto notLhs = rewriter.create<hivmave::PregNotOp>(
              loc, resType, elemBitWidthAttr, lhs, mask);
          auto res = rewriter.create<hivmave::PregAndOp>(
              loc, resType, elemBitWidthAttr, notLhs, rhs, mask);
          rewriter.replaceOp(op, res);
          return success();
        }
      }
    }

    hivmave::CmpType cmpType = getHIVMCmpType<ArithCmpOp>(op);

    auto res =
        rewriter.create<HivmVFCmpOp>(loc, resType, cmpType, lhs, rhs, mask);

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct SelectOpPattern : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  using SelectOpAdaptor =
      typename OpConversionPattern<arith::SelectOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, SelectOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto trueVal = op.getTrueValue();
    auto falseVal = op.getFalseValue();
    auto condition = op.getCondition();
    auto resType = op.getResult().getType();

    VectorType resVecType = mlir::dyn_cast<VectorType>(resType);

    if (!resVecType || resVecType.getShape().size() != 1)
      return failure();

    // Check if the result type is FP8: f8e5m2 or f8e4m3fn
    auto resDtype = resVecType.getElementType();
    if (resDtype.isFloat8E5M2() || resDtype.isFloat8E4M3FN()) {
      auto vecSize = resVecType.getShape();
      // Create i8 vector type
      auto i8VecTy = VectorType::get({vecSize}, rewriter.getI8Type());

      // Bitcast the result of VFSelectOp from f8e5m2 to i8
      trueVal = rewriter.create<vector::BitCastOp>(loc, i8VecTy, trueVal);
      falseVal = rewriter.create<vector::BitCastOp>(loc, i8VecTy, falseVal);
      // Create VFSelectOp
      auto res = rewriter.create<hivmave::VFSelectOp>(loc, i8VecTy, condition,
                                                      trueVal, falseVal);

      rewriter.replaceOpWithNewOp<vector::BitCastOp>(op, TypeRange{resType},
                                                     res);

      return success();
    }

    auto res = rewriter.create<hivmave::VFSelectOp>(loc, resType, condition,
                                                    trueVal, falseVal);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct BitcastOpPattern : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern<arith::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = op.getResult().getType();
    auto resVecType = mlir::dyn_cast<VectorType>(resType);

    if (!resVecType)
      return failure();

    rewriter.replaceOpWithNewOp<vector::BitCastOp>(op, resType,
                                                   adaptor.getIn());
    return success();
  }
};

struct ArithToHIVMAVEPass
    : public impl::ConvertArithToHIVMAVEBase<ArithToHIVMAVEPass> {
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<hivmave::AVEDialect, BuiltinDialect,
                           vector::VectorDialect, hivm::HIVMDialect>();
    target.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
      return !isa<VectorType>(op->getResult(0).getType());
    });
    RewritePatternSet patterns(&getContext());
    hivmave::populateArithToHIVMAVEConversionPatterns(patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

static bool isXorWithAllTruePge(Value src) {
  if (isa<BlockArgument>(src))
    return false;
  // since applyPartialConversion is always top‑down, so it is garanteed
  // arith.constant is converted to pge/brc
  auto defOp = src.getDefiningOp<hivmave::VFPgeOp>();
  if (!defOp)
    return false;
  return defOp.getPattern() == PgePattern::ALL;
}

struct XOrIOpLowering : public OpConversionPattern<arith::XOrIOp> {
  using OpConversionPattern<arith::XOrIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::XOrIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Type resType = op.getType();
    Value notOperand = nullptr;
    VectorType resVecType = dyn_cast<VectorType>(resType);
    if (!resVecType || resVecType.getShape().size() != 1) {
      return failure();
    }
    Type elemType = resVecType.getElementType();
    Value mask = hivmave::findReuseableMaskOrCreateOne(op, resVecType, rewriter);
    Value res;
    if (elemType.isInteger(1)) {
      auto elemBitWidthAttr =
          MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8);
      notOperand = isXorWithAllTruePge(lhs) ? rhs : notOperand;
      notOperand = isXorWithAllTruePge(rhs) ? lhs : notOperand;
      if (notOperand) {
        Value notOp = (notOperand == rhs) ? rhs : lhs;
        res = rewriter.create<hivmave::PregNotOp>(
            loc, resVecType, elemBitWidthAttr, notOp, mask);
      } else {
        res = rewriter.create<hivmave::PregXorOp>(
            loc, resVecType, elemBitWidthAttr, lhs, rhs, mask);
      }
    } else {
      if (!isEligibleVectorType(resType))
        return failure();
      notOperand = isAllNegOnesVector(lhs) ? rhs : notOperand;
      notOperand = isAllNegOnesVector(rhs) ? lhs : notOperand;
      if (notOperand)
        res = rewriter.create<hivmave::VFNotOp>(loc, resType, notOperand, mask,
                                                nullptr);
      else
        res = rewriter.create<hivmave::VFXorOp>(loc, resType, lhs, rhs, mask,
                                                nullptr);
    }
    rewriter.replaceOp(op, res);
    return success();
  }

private:
  // check if a value is a constant vector of all ones (-1)
  bool isAllNegOnesVector(Value value) const {
    if (auto broadcastOp =
            value.getDefiningOp<hivmave::VFBroadcastScalarMaskOp>()) {
      Value scalar = broadcastOp->getOperand(0);
      Type vecType = broadcastOp.getType();
      VectorType resVecType = dyn_cast<VectorType>(vecType);
      if (!resVecType)
        return false;
      Type elemType = resVecType.getElementType();
      if (!elemType.isIntOrIndex())
        return false;
      unsigned elemWidth = elemType.getIntOrFloatBitWidth();
      if (auto constOp = scalar.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          APInt scalarValue = intAttr.getValue();
          if (scalarValue.getBitWidth() < elemWidth)
            return false;
          APInt elemBits = scalarValue.extractBits(elemWidth, 0);
          return elemBits.isAllOnes();
        }
      }
    }
    return false;
  }
};

template <typename SourceOp>
struct ShiftOpLowering : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Type resType = op.getType();

    VectorType resVecType = dyn_cast<VectorType>(resType);
    if (!isEligibleVectorType(resType)) {
      return failure();
    }

    Value mask = hivmave::findReuseableMaskOrCreateOne(op, resVecType, rewriter);

    if constexpr (std::is_same_v<SourceOp, arith::ShLIOp>) {
      Value boolConst =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
      auto shlOp = rewriter.create<hivmave::VFShlOp>(loc, resType, lhs, rhs,
                                                     mask, boolConst, nullptr);
      rewriter.replaceOp(op, shlOp.getResult());
    } else if constexpr (std::is_same_v<SourceOp, arith::ShRSIOp>) {
      Value boolConst =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
      auto shrOp = rewriter.create<hivmave::VFShrOp>(loc, resType, lhs, rhs,
                                                     mask, boolConst, nullptr);
      rewriter.replaceOp(op, shrOp.getResult());
    } else if constexpr (std::is_same_v<SourceOp, arith::ShRUIOp>) {
      Value boolConst =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
      auto shrOp = rewriter.create<hivmave::VFShrOp>(loc, resType, lhs, rhs,
                                                     mask, boolConst, nullptr);
      rewriter.replaceOp(op, shrOp.getResult());
    }

    return success();
  }
};

template <typename SourceOp>
struct LogicOpLowering : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto resType = op.getResult().getType();
    VectorType resVecType = mlir::dyn_cast<VectorType>(resType);
    if (!resVecType || !isOneDimLikeVecType(resVecType)) {
      LLVM_DEBUG(DBGS() << "legalize " << SourceOp::getOperationName()
                        << " failed: Not a 1D-like vector\n";);
      return failure();
    }
    Type elemType = resVecType.getElementType();
    Value mask = hivmave::findReuseableMaskOrCreateOne(op, resVecType, rewriter);
    if (elemType.isInteger(1)) {
      auto elemBitWidthAttr =
          MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8);
      Value res;
      if (std::is_same<SourceOp, arith::AndIOp>::value) {
        res = rewriter.create<hivmave::PregAndOp>(
            loc, resType, elemBitWidthAttr, lhs, rhs, mask);
      } else if (std::is_same<SourceOp, arith::OrIOp>::value) {
        res = rewriter.create<hivmave::PregOrOp>(loc, resType, elemBitWidthAttr,
                                                 lhs, rhs, mask);
      } else {
        LLVM_DEBUG(DBGS() << "Unsupported operation: "
                          << SourceOp::getOperationName() << "\n";);
        return failure();
      }
      rewriter.replaceOp(op, res);
      return success();
    } else {
      if (std::is_same<SourceOp, arith::AndIOp>::value) {
        rewriter.replaceOpWithNewOp<hivmave::VFAndOp>(op, resType, lhs, rhs,
                                                      mask, nullptr);
        return success();
      } else if (std::is_same<SourceOp, arith::OrIOp>::value) {
        rewriter.replaceOpWithNewOp<hivmave::VFOrOp>(op, resType, lhs, rhs,
                                                     mask, nullptr);
        return success();
      }
    }
    return failure();
  }
};

void mlir::hivmave::populateArithToHIVMAVEConversionPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<BinaryOpPattern<arith::AddFOp, hivmave::VFAddOp>,
           BinaryOpPattern<arith::AddIOp, hivmave::VFAddOp>,
           BinaryOpPattern<arith::SubFOp, hivmave::VFSubOp>,
           BinaryOpPattern<arith::SubIOp, hivmave::VFSubOp>,
           BinaryOpPattern<arith::MulFOp, hivmave::VFMulOp>,
           BinaryOpPattern<arith::MulIOp, hivmave::VFMulOp>,
           BinaryOpPattern<arith::DivFOp, hivmave::VFDivOp>,
           BinaryOpPattern<arith::DivSIOp, hivmave::VFDivOp>,
           BinaryOpPattern<arith::DivUIOp, hivmave::VFDivOp>,
           BinaryOpPattern<arith::MaxSIOp, hivmave::VMaxSIOp>,
           BinaryOpPattern<arith::MaxUIOp, hivmave::VMaxUIOp>,
           BinaryOpPattern<arith::MinSIOp, hivmave::VMinSIOp>,
           BinaryOpPattern<arith::MinUIOp, hivmave::VMinUIOp>,
           BinaryOpPattern<arith::RemSIOp, hivmave::VFModOp>,
           BinaryOpPattern<arith::RemUIOp, hivmave::VFModUIOp>,
           BinaryOpPattern<arith::MaximumFOp, hivmave::VFMaxOp>,
           BinaryOpPattern<arith::MinimumFOp, hivmave::VFMinOp>,
           BinaryOpPattern<mathExt::DivFHPOp, hivmave::VFDivFHPOp>,
           ArithMulExtendOpPattern<arith::MulSIExtendedOp, hivmave::VFVMULLOp>,
           ArithMulExtendOpPattern<arith::MulUIExtendedOp, hivmave::VFVMULLOp>,
           LogicOpLowering<arith::AndIOp>, LogicOpLowering<arith::OrIOp>,
           XOrIOpLowering, ShiftOpLowering<arith::ShLIOp>,
           ShiftOpLowering<arith::ShRSIOp>, ShiftOpLowering<arith::ShRUIOp>,
           CmpOpPattern<arith::CmpFOp, hivmave::VFCmpOp>,
           CmpOpPattern<arith::CmpIOp, hivmave::VFCmpOp>,
           UnaryOpPattern<math::SqrtOp, hivmave::VFSqrtOp>,
           UnaryOpPattern<math::RsqrtOp, hivmave::VFRsqrtOp>,
           UnaryOpPattern<arith::NegFOp, hivmave::VFNegOp>,
           UnaryOpPattern<math::ExpOp, hivmave::VFExpOp>,
           UnaryOpPattern<math::AbsFOp, hivmave::VFAbsOp>,
           UnaryOpPattern<math::AbsIOp, hivmave::VFAbsOp>, RoundOpPattern,
           UnaryOpPattern<math::LogOp, hivmave::VFLnOp>,
           VFTypeConvertionPattern<arith::ExtFOp>,
           VFTypeConvertionPattern<arith::TruncFOp>,
           VFTypeConvertionPattern<arith::ExtSIOp>,
           VFTypeConvertionPattern<arith::ExtUIOp>,
           VFTypeConvertionPattern<arith::TruncIOp>,
           VFTypeConvertionPattern<arith::SIToFPOp>,
           VFTypeConvertionPattern<arith::UIToFPOp>,
           VFTypeConvertionPattern<arith::FPToSIOp>,
           VFTypeConvertionPattern<arith::FPToUIOp>,
           ConstantOpToHivmPLTLowering, ConstantOpToHivmBroadcastLowering,
           ConstantOpToHivmVCIVCPLowering, SelectOpPattern, BitcastOpPattern>(
          patterns.getContext());
}

std::unique_ptr<Pass> mlir::createArithToHIVMAVEConversionPass() {
  return std::make_unique<ArithToHIVMAVEPass>();
}
