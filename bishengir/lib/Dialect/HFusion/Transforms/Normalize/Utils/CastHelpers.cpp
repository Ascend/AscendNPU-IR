//===- CastHelpers.cpp --------------------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"

namespace mlir::hfusion {
Value castInToF32ToOut(hfusion::CastOp &op, PatternRewriter &rewriter) {
  auto dstTy = getElementTypeOrSelf(op.getDpsInitOperand(0)->get());
  auto castSrcToF32 = castTo(rewriter, op.getDpsInputOperand(0)->get(),
                             rewriter.getF32Type(), op.getCast());
  auto castF32ToOut =
      hfusion::castTo(rewriter, castSrcToF32, dstTy, TypeFn::cast_signed);
  return castF32ToOut;
}

Value castU32ToI64ToF32(hfusion::CastOp &op, PatternRewriter &rewriter) {
  auto castU32ToI64 = castTo(rewriter, op.getDpsInputOperand(0)->get(),
                             rewriter.getI64Type(), op.getCast());
  auto castI64ToFp32 = hfusion::castTo(
      rewriter, castU32ToI64, rewriter.getF32Type(), TypeFn::cast_signed);
  return castI64ToFp32;
}

Value castU32ToI64ToF32ToOut(hfusion::CastOp &op, Type targetType,
                                    PatternRewriter &rewriter) {
  // u32 -> i64 -> fp32
  Value u32ToF32Result = castU32ToI64ToF32(op, rewriter);
  // fp32 -> fp16/bf16
  auto castF32ToOut = hfusion::castTo(rewriter, u32ToF32Result, targetType,
                                      hfusion::TypeFn::cast_signed);
  return castF32ToOut;
}

// i1/i8/i16/u8/u16 -> f16 -> targetType
Value castSrcToFp16ToTargetType(hfusion::CastOp &op, Type targetType,
                                       PatternRewriter &rewriter) {
  Type f16Type = rewriter.getF16Type();
  Value dpsInput = op.getDpsInputOperand(0)->get();
  auto castSrcToF16 = castTo(rewriter, dpsInput, f16Type, op.getCast());
  return castTo(rewriter, castSrcToF16, targetType, TypeFn::cast_signed);
}

// i64/i8 -> i1
Value castSrcTypeToI1ByVCmp(hfusion::CastOp &op, Type srcType,
                                   PatternRewriter &rewriter) {
  // 1. cast src to f16/f32
  Value inValue = op.getInputs()[0];
  Value castF16OrF32Value;
  if (srcType.isInteger(8)) {
    castF16OrF32Value =
        hfusion::castTo(rewriter, inValue, rewriter.getF16Type());
  } else if (srcType.isInteger(16)) {
    castF16OrF32Value = hfusion::castTo(
        rewriter, inValue, rewriter.getF16Type(), hfusion::RoundMode::RINT);
  } else if (srcType.isInteger(32)) {
    castF16OrF32Value = hfusion::castTo(
        rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);
  } else if (srcType.isInteger(64)) {
    castF16OrF32Value = hfusion::castTo(
        rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);
  } else if (srcType.isBF16()) {
    castF16OrF32Value = hfusion::castTo(
        rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);
  } else if (srcType.isF32() || srcType.isF16()) {
    castF16OrF32Value = inValue;
  } else {
    llvm_unreachable("unsupport srcType to i1.");
  }

  // 2. cast: f16/f32 -> i1, dst = vcmpvs_ne(src, 0)
  auto elementType = getElementTypeOrSelf(castF16OrF32Value);
  arith::ConstantOp constZero = rewriter.create<arith::ConstantOp>(
      op->getLoc(), elementType, rewriter.getFloatAttr(elementType, 0.0));

  Value castI1Value = createCmpOp(rewriter, op.getLoc(), castF16OrF32Value,
                                  constZero, CompareFn::vne)
                          ->getResult(0);
  return castI1Value;
}

// membase: i8 -> f16 -> f32 -> i64
// regbase: i8 -> i32 -> i64
Value castI8ToI64(hfusion::CastOp &op, PatternRewriter &rewriter) {
  if (!archIsRegbased) {
    // i8 -> f16 -> f32
    Value i8ToF32Result =
        castSrcToFp16ToTargetType(op, rewriter.getF32Type(), rewriter);
    // f32->i64
    Type i64Type = rewriter.getIntegerType(64);
    auto castF32ToDst =
        castTo(rewriter, i8ToF32Result, i64Type, TypeFn::cast_signed);
    return castF32ToDst;
  } else {
    Value dpsInput = op.getDpsInputOperand(0)->get();
    auto castIntegerType = op.getCast();
    auto castValue =
        castTo(rewriter, dpsInput, rewriter.getI32Type(), castIntegerType);
    return castTo(rewriter, castValue, rewriter.getI64Type(), castIntegerType);
  }
}

hfusion::CastMode getCastMode(hfusion::CastOp op) {
  auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
  auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());

  const bool isF32ToI16 = inType.isF32() && outType.isInteger(16);
  const bool isF32ToI8 = inType.isF32() && outType.isInteger(8);
  const bool isF16ToI8 = inType.isF16() && outType.isInteger(8);
  const bool isI64ToI32 = inType.isInteger(64) && outType.isInteger(32);
  const bool isI64ToI16 = inType.isInteger(64) && outType.isInteger(16);
  const bool isI64ToI8 = inType.isInteger(64) && outType.isInteger(8);
  const bool isI32ToI16 = inType.isInteger(32) && outType.isInteger(16);
  const bool isI32ToI8 = inType.isInteger(32) && outType.isInteger(8);
  const bool isI16ToI8 = inType.isInteger(16) && outType.isInteger(8);

  if (isF32ToI16)
    return hfusion::CastMode::F32TOI16;
  if (isF32ToI8)
    return hfusion::CastMode::F32TOI8;
  if (isF16ToI8)
    return hfusion::CastMode::F16TOI8;
  if (isI64ToI32)
    return hfusion::CastMode::I64TOI32;
  if (isI64ToI16)
    return hfusion::CastMode::I64TOI16;
  if (isI64ToI8)
    return hfusion::CastMode::I64TOI8;
  if (isI32ToI16)
    return hfusion::CastMode::I32TOI16;
  if (isI32ToI8)
    return hfusion::CastMode::I32TOI8;
  if (isI16ToI8)
    return hfusion::CastMode::I16TOI8;

  llvm_unreachable("unsupported cast mode");
}

template <typename OpType>
std::optional<bool> getAnnotateAttrBool(OpType op, StringRef attr) {
  std::optional<Operation *> attrOp =
    utils::getAnnotateOpWithAttr(op.getResult(0), attr);
  if (!attrOp.has_value())
    return std::nullopt;

  if (auto boolAttr =
      attrOp.value()->getAttrOfType<BoolAttr>(attr)) {
    return boolAttr.getValue();
  }

  return std::nullopt;
}

/// normalize cast from large bit width to small bit width, and dst's data type
/// is integer, when overflow mode is saturate.
/// if data is overflow, it will be saturated to the extreme in this scenario.
/// e.g. Input (float32): tensor([ 128.7000,  127.5000,  100.3000, -129.2000,
/// -128.4000]), Output(int8): tensor([ 127,  127,  100, -128, -128],
/// dtype=torch.int8)
LogicalResult handleSaturateOverFlowMode(hfusion::CastOp op,
                                         PatternRewriter &rewriter) {
  hfusion::CastMode castMode = getCastMode(op);
  Value castValue = op.getInputs()[0];
  auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());
  hfusion::TypeFn castIntegerType = op.getCast();

  switch (castMode) {
  case hfusion::CastMode::F32TOI16:
    castValue = hfusion::castTo(
        rewriter, castValue, outType, hfusion::RoundMode::TRUNC, std::nullopt,
        /*enableOverflow=*/false, false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::F32TOI8:
    // step 1: cast f32 to f16 in TRUNC mode
    castValue =
        hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                        hfusion::RoundMode::TRUNC, std::nullopt,
                        /*enableOverflow=*/false, false, castIntegerType);
    // step 2: cast f16 to i8 in TRUNC mode
    castValue = hfusion::castTo(
        rewriter, castValue, outType, hfusion::RoundMode::TRUNC, std::nullopt,
        /*enableOverflow=*/false, false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::F16TOI8:
    castValue = hfusion::castTo(
        rewriter, castValue, outType, hfusion::RoundMode::TRUNC, std::nullopt,
        /*enableOverflow=*/false, false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I64TOI32:
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::RINT,
                        std::nullopt, /*enableOverflow=*/false);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I64TOI16:
    // step 1: cast i32 to f32 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 2: cast f32 to i16 in TRUNC mode
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I64TOI8:
    // step 1: cast i32 to f32 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 2: cast f32 to f16 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 3: cast f16 to i8 in TRUNC mode
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow=*/false);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I32TOI16:
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::RINT,
                        std::nullopt, /*enableOverflow=*/false);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I32TOI8:
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 2: cast f32 to f16 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 3: cast f16 to i8 in TRUNC mode
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow=*/false);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I16TOI8:
    // step 1: cast i16 to f16 in TRUNC mode
    castValue =
        hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                        hfusion::RoundMode::TRUNC, std::nullopt,
                        /*enableOverflow=*/false, false, hfusion::TypeFn{});
    // step 2: cast f16 to i8 in TRUNC mode
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow=*/false);
    rewriter.replaceOp(op, castValue);
    return success();
  }
}

LogicalResult handleTruncOverFlowMode(hfusion::CastOp op,
                                      PatternRewriter &rewriter) {
  assert(!archIsRegbased);
  auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
  auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());
  auto castIntegerType = op.getCast();

  const bool isF32ToI16 = inType.isF32() && outType.isInteger(16);
  const bool isF32ToI8 = inType.isF32() && outType.isInteger(8);
  const bool isF16ToI8 = inType.isF16() && outType.isInteger(8);
  const bool isI64ToI16 = inType.isInteger(64) && outType.isInteger(16);
  const bool isI64ToI8 = inType.isInteger(64) && outType.isInteger(8);
  const bool isI32ToI8 = inType.isInteger(32) && outType.isInteger(8);
  const bool isI16ToI8 = inType.isInteger(16) && outType.isInteger(8);
  Value castValue = op.getInputs()[0];
  // TODO: The round_mode will be flushed and will be fixed during
  // reconstruction.
  if (isF32ToI16 && op.getEnableOverflow()) {
    // step1: cast f32 to i32 in TRUNC mode
    Value castI32Value = hfusion::castTo(
        rewriter, castValue, rewriter.getI32Type(), hfusion::RoundMode::TRUNC,
        std::nullopt, true, false, castIntegerType);
    // step2: cast i32 to i16
    castValue = hfusion::castTo(rewriter, castI32Value, rewriter.getI16Type(),
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF32ToI8) {
    // step 1: cast f32 to i32 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getI32Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt, true,
                                false, castIntegerType);
    // step 2: cast i32 to i8 in TRUNCWITHOVERFLOW mode
    castValue = hfusion::castTo(rewriter, castValue, outType,
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF16ToI8 && op.getEnableOverflow()) {
    Value overflowResult = hfusion::OverflowProcess(
        rewriter, castValue, getElementTypeOrSelf(outType));
    castValue =
        hfusion::castTo(rewriter, overflowResult, outType,
                        hfusion::RoundMode::TRUNC, std::nullopt,
                        /*enableOverflow=*/false, false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI64ToI16 || isI64ToI8) {
    // step 1: cast i64 to i32 in TRUNCWITHOVERFLOW mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getI32Type(),
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    // step 2: cast i32 to i16/i8 in TRUNCWITHOVERFLOW mode
    castValue = hfusion::castTo(rewriter, castValue, outType,
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if ((isI32ToI8 || isI16ToI8) &&
             op.getRoundMode() != hfusion::RoundMode::TRUNCWITHOVERFLOW) {
    castValue = hfusion::castTo(rewriter, castValue, outType,
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    rewriter.replaceOp(op, castValue);
    return success();
  }
  return failure();
}

// Handle overflow_mode = saturate
LogicalResult handleOverflowModeForSaturate(hfusion::CastOp op,
                                            PatternRewriter &rewriter,
                                            bool enableSaturate) {
  auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
  auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());

  const bool isF32ToI8 = inType.isF32() && outType.isInteger(8);
  const bool isI64ToI16 = inType.isInteger(64) && outType.isInteger(16);
  const bool isI64ToI8 = inType.isInteger(64) && outType.isInteger(8);
  const bool isI32ToI8 = inType.isInteger(32) && outType.isInteger(8);
  const bool isF16ToI8 = inType.isF16() && outType.isInteger(8);
  const bool isF32ToI16 = inType.isF32() && outType.isInteger(16);

  const bool isI64ToI32 = inType.isInteger(64) && outType.isInteger(32);
  const bool isI32ToI16 = inType.isInteger(32) && outType.isInteger(16);
  const bool isI16ToI8 = inType.isInteger(16) && outType.isInteger(8);
  hfusion::TypeFn castIntegerType = op.getCast();
  hfusion::UnsignedMode unsignedMode = hfusion::UnsignedMode::SI2SI;

  auto srcUnsigned = getAnnotateAttrBool(op, util::saturateSrcUnsigned);
  const bool srcIsUnsigned = srcUnsigned.value_or(false);
  auto dstUnsigned = getAnnotateAttrBool(op, util::saturateDstUnsigned);
  const bool dstIsUnsigned = dstUnsigned.value_or(false);

  auto srcAttr =
      utils::getAnnotateOpWithAttr(op->getResult(0), util::saturateSrcUnsigned);
  if (srcAttr.has_value()) {
    annotation::MarkOp srcMarkOp =
      dyn_cast<annotation::MarkOp>(srcAttr.value());
    rewriter.eraseOp(srcMarkOp);
  }
  auto dstAttr =
      utils::getAnnotateOpWithAttr(op->getResult(0), util::saturateDstUnsigned);
  if (dstAttr.has_value()) {
    annotation::MarkOp dstMarkOp =
      dyn_cast<annotation::MarkOp>(dstAttr.value());
    rewriter.eraseOp(dstMarkOp);
  }

  const bool isSIToSI = !srcIsUnsigned && !dstIsUnsigned;
  const bool isSIToUI = !srcIsUnsigned && dstIsUnsigned;
  const bool isUIToSI = srcIsUnsigned && !dstIsUnsigned;
  const bool isUIToUI = srcIsUnsigned && dstIsUnsigned;

  if (isSIToUI) {
    unsignedMode = hfusion::UnsignedMode::SI2UI;
  } else if (isUIToSI) {
    unsignedMode = hfusion::UnsignedMode::UI2SI;
  } else if (isUIToUI) {
    unsignedMode = hfusion::UnsignedMode::UI2UI;
  }

  Value castValue = op.getInputs()[0];
  if (isF32ToI16) {
    castValue =
        hfusion::castTo(rewriter, castValue, rewriter.getI32Type(),
                        hfusion::RoundMode::TRUNCWITHOVERFLOW, std::nullopt,
                        /*enableOverflow*/ false, enableSaturate);
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF32ToI8) {
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow*/ false, enableSaturate);
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF16ToI8) {
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI64ToI32) {
    castValue =
        hfusion::castTo(rewriter, castValue, outType,
                        hfusion::RoundMode::TRUNC, std::nullopt,
                        /*enableOverflow*/ false, enableSaturate,
                        castIntegerType, unsignedMode);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI64ToI16) {
    if (isSIToSI) {
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue = hfusion::castTo(rewriter, castValue, outType,
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  castIntegerType);
    } else {
      castValue = hfusion::castTo(rewriter, castValue, outType,
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  castIntegerType, unsignedMode);
    }

    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI64ToI8) {
    if (isSIToSI) {
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue = hfusion::castTo(rewriter, castValue, outType,
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  castIntegerType);
    } else {
      castValue = hfusion::castTo(rewriter, castValue, outType,
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  castIntegerType, unsignedMode);
    }

    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI32ToI16) {
    if (isSIToSI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    } else if (isSIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    } else if (isUIToSI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    } else if (isUIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    }

    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI32ToI8) {
    if (isSIToSI) {
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue =
          hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                          std::nullopt, /*enableOverflow*/ false,
                          enableSaturate, castIntegerType);
    } else if (isSIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    } else if (isUIToSI) { // u32-s16-f16-s8
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getI16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  castIntegerType, hfusion::UnsignedMode::UI2SI);
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue =
          hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                          std::nullopt, /*enableOverflow*/ false,
                          enableSaturate, castIntegerType);
    } else if (isUIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    }

    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI16ToI8) {
    if (isSIToSI) {
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue =
          hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                          std::nullopt, /*enableOverflow*/ false,
                          enableSaturate, castIntegerType, unsignedMode);
    } else if (isSIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    } else if (isUIToSI) {
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getI8Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  hfusion::TypeFn::cast_unsigned,
                                  hfusion::UnsignedMode::UI2UI);
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  hfusion::TypeFn::cast_unsigned);
      castValue =
          hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                          std::nullopt, /*enableOverflow*/ false,
                          enableSaturate, castIntegerType);
    } else if (isUIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    }

    rewriter.replaceOp(op, castValue);
    return success();
  }
  return failure();
}

// Handle overflow_mode = trunc
// For IntToInt, we retain the original hfusion.cast statement
LogicalResult handleOverflowModeForTrunc(hfusion::CastOp op,
                                         PatternRewriter &rewriter) {
  auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
  auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());

  const bool isF32ToI16 = inType.isF32() && outType.isInteger(16);
  const bool isF32ToI8 = inType.isF32() && outType.isInteger(8);
  hfusion::TypeFn castIntegerType = op.getCast();

  Value castValue = op.getInputs()[0];
  if (isF32ToI16) {
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getI32Type(),
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getI16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt, true,
                                false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF32ToI8) {
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getI32Type(),
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getIntegerType(8),
                                hfusion::RoundMode::TRUNC, std::nullopt, true,
                                false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  }
  return failure();
}

} // namespace mlir::hfusion
