//===-------- CastingTemplateHelpers.h ------------------------------------===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_CASTINGTEMPLATEHELPERS_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_CASTINGTEMPLATEHELPERS_H

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Normalize/Utils/Kinds.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>
#include <optional>

namespace mlir {

constexpr StringLiteral kOverflowModeAttr = "overflow_mode";
constexpr StringLiteral kEnableOverflowAttr = "enable_overflow";
constexpr StringLiteral kSaturateSrcUnsignedAttr = "saturate_src_unsigned";
constexpr StringLiteral kSaturateDstUnsignedAttr = "saturate_dst_unsigned";

template <typename CastOpType>
bool requiresOverflowNormalization(CastOpType op) {
  auto inType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
  auto outType = getElementTypeOrSelf(op.getDpsInits()[0].getType());
  int64_t srcBitWidth = inType.getIntOrFloatBitWidth();
  int64_t dstBitWidth = outType.getIntOrFloatBitWidth();
  return srcBitWidth > dstBitWidth && outType.isInteger() &&
         !outType.isInteger(1);
}

template <typename CastOpType>
OverflowCastKind getOverflowCastKind(CastOpType op) {
  auto inType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
  auto outType = getElementTypeOrSelf(op.getDpsInits()[0].getType());

  if (inType.isF32() && outType.isInteger(16))
    return OverflowCastKind::F32ToI16;
  if (inType.isF32() && outType.isInteger(8))
    return OverflowCastKind::F32ToI8;
  if (inType.isF16() && outType.isInteger(8))
    return OverflowCastKind::F16ToI8;
  if (inType.isInteger(64) && outType.isInteger(32))
    return OverflowCastKind::I64ToI32;
  if (inType.isInteger(64) && outType.isInteger(16))
    return OverflowCastKind::I64ToI16;
  if (inType.isInteger(64) && outType.isInteger(8))
    return OverflowCastKind::I64ToI8;
  if (inType.isInteger(32) && outType.isInteger(16))
    return OverflowCastKind::I32ToI16;
  if (inType.isInteger(32) && outType.isInteger(8))
    return OverflowCastKind::I32ToI8;
  if (inType.isInteger(16) && outType.isInteger(8))
    return OverflowCastKind::I16ToI8;

  return OverflowCastKind::Unsupported;
}

template <typename Traits, typename CastOpType>
bool isTerminalNativeSaturateCast(CastOpType op) {
  if (!Traits::archIsRegbased() || !Traits::hasSaturateEnabled(op))
    return false;

  // These patterns already match the backend's final native saturating narrow.
  OverflowCastKind kind = getOverflowCastKind(op);
  if (kind == OverflowCastKind::F16ToI8)
    return Traits::matchCastRoundMode(op, CastExecutionKind::Trunc);

  if (kind == OverflowCastKind::I64ToI32 || kind == OverflowCastKind::I32ToI16)
    return Traits::matchCastRoundMode(op, CastExecutionKind::Trunc) ||
           Traits::matchCastRoundMode(op, CastExecutionKind::TruncWithOverflow);

  if (kind == OverflowCastKind::I64ToI16 || kind == OverflowCastKind::I64ToI8)
    return Traits::matchCastRoundMode(op, CastExecutionKind::Trunc) &&
           !Traits::matchCastUnsignedMode(op, CastUnsignedModeKind::SignedToSigned);

  if (kind == OverflowCastKind::I32ToI8 || kind == OverflowCastKind::I16ToI8)
    return Traits::matchCastRoundMode(op, CastExecutionKind::Trunc) &&
           (Traits::matchCastUnsignedMode(op, CastUnsignedModeKind::SignedToUnsigned) ||
            Traits::matchCastUnsignedMode(op, CastUnsignedModeKind::UnsignedToUnsigned));

  return false;
}

template <typename Traits>
bool supportsDirectTruncWithOverflow(OverflowCastKind kind) {
  return !Traits::archIsRegbased() &&
         (kind == OverflowCastKind::I32ToI8 ||
          kind == OverflowCastKind::I16ToI8);
}

template <typename Traits, typename CastOpType>
bool canNormalizeTruncOverflow(CastOpType op, OverflowCastKind kind) {
  if (Traits::archIsRegbased())
    return kind == OverflowCastKind::F32ToI16 ||
           kind == OverflowCastKind::F32ToI8;

  // Mem-based backends only accept the cases we can rebuild from legal pieces.
  switch (kind) {
  case OverflowCastKind::F32ToI16:
    return Traits::hasOverflowEnabled(op);
  case OverflowCastKind::F32ToI8:
    return true;
  case OverflowCastKind::F16ToI8:
    return Traits::hasOverflowEnabled(op);
  case OverflowCastKind::I64ToI16:
  case OverflowCastKind::I64ToI8:
  case OverflowCastKind::I32ToI8:
  case OverflowCastKind::I16ToI8:
    return true;
  case OverflowCastKind::I64ToI32:
  case OverflowCastKind::I32ToI16:
  case OverflowCastKind::Unsupported:
    return false;
  }

  llvm_unreachable("unhandled overflow cast kind");
}

template <typename Traits>
bool useLegacyFloatTruncOverflowSequence(OverflowCastKind kind) {
  return Traits::archIsRegbased() &&
         (kind == OverflowCastKind::F32ToI16 ||
          kind == OverflowCastKind::F32ToI8);
}

template <typename Traits, typename CastOpType>
bool skipDirectTruncWithOverflowRewrite(CastOpType op, OverflowCastKind kind) {
  if (!Traits::archIsRegbased() &&
      (kind == OverflowCastKind::I32ToI8 || kind == OverflowCastKind::I16ToI8))
    return Traits::matchCastRoundMode(op,
                                      CastExecutionKind::TruncWithOverflow);
  return false;
}

template <typename Traits>
bool supportsTwoStepTruncOverflow(OverflowCastKind kind) {
  return !Traits::archIsRegbased() &&
         (kind == OverflowCastKind::F32ToI16 ||
          kind == OverflowCastKind::F32ToI8 ||
          kind == OverflowCastKind::F16ToI8 ||
          kind == OverflowCastKind::I64ToI16 ||
          kind == OverflowCastKind::I64ToI8);
}

template <typename Traits, typename CastOpType>
bool shouldPreprocessTruncOverflowInput(CastOpType op, OverflowCastKind kind) {
  return !Traits::archIsRegbased() && kind == OverflowCastKind::F16ToI8 &&
         Traits::supportsF16ToI8TruncOverflowPreprocess &&
         Traits::hasOverflowEnabled(op);
}

template <typename Traits, typename CastOpType>
Value preprocessTruncOverflowInput(PatternRewriter &rewriter, Location loc,
                                   CastOpType op, Value input) {
  Type targetElemType = getElementTypeOrSelf(op.getDpsInits()[0].getType());
  Type srcElemType = getElementTypeOrSelf(input.getType());
  Value targetTensor =
      utils::createEmptyOpWithTargetElemType(rewriter, loc, input, srcElemType);

  auto createOffset = [&rewriter, loc, srcElemType](double offset) {
    return rewriter
        .create<arith::ConstantOp>(
            loc, srcElemType, rewriter.getFloatAttr(srcElemType, offset))
        .getResult();
  };

  uint32_t bits = targetElemType.getIntOrFloatBitWidth();
  double overflowRange = static_cast<double>(1ULL << bits);
  double signedOffset = static_cast<double>(1ULL << (bits - 1));

  Value value = input;
  if (targetElemType.isSignlessInteger()) {
    // Shift to unsigned space so floating-point mod implements signed wrap.
    value = Traits::createBinaryOp(rewriter, loc, value,
                                   createOffset(signedOffset), targetTensor,
                                   BinaryKind::Add);
  }

  value = Traits::createBinaryOp(rewriter, loc, value,
                                 createOffset(overflowRange), targetTensor,
                                 BinaryKind::Mod);

  if (targetElemType.isSignlessInteger()) {
    value = Traits::createBinaryOp(rewriter, loc, value,
                                   createOffset(-signedOffset), targetTensor,
                                   BinaryKind::Add);
  }

  return value;
}

template <typename CastOpType>
Value getCastPrimaryValue(CastOpType op) {
  return op->getNumResults() ? op->getResult(0) : op.getDpsInits()[0];
}

template <typename CastOpType>
std::optional<StringRef> getOverflowModeAnnotation(CastOpType op) {
  std::optional<Operation *> overflowMode =
      utils::getAnnotateOpWithAttr(getCastPrimaryValue(op), kOverflowModeAttr);
  if (!overflowMode.has_value())
    return std::nullopt;

  if (auto overflowAttr =
          overflowMode.value()->getAttrOfType<StringAttr>(kOverflowModeAttr))
    return overflowAttr.getValue();

  return std::nullopt;
}

template <typename CastOpType>
std::optional<bool> getCastAnnotationBool(CastOpType op, StringRef attr) {
  std::optional<Operation *> attrOp =
      utils::getAnnotateOpWithAttr(getCastPrimaryValue(op), attr);
  if (!attrOp.has_value())
    return std::nullopt;

  if (auto boolAttr = attrOp.value()->getAttrOfType<BoolAttr>(attr))
    return boolAttr.getValue();
  if (attrOp.value()->hasAttrOfType<UnitAttr>(attr))
    return true;

  return std::nullopt;
}

template <typename CastOpType>
void eraseCastAnnotationIfPresent(PatternRewriter &rewriter, CastOpType op,
                                  StringRef attr) {
  std::optional<Operation *> attrOp =
      utils::getAnnotateOpWithAttr(getCastPrimaryValue(op), attr);
  if (!attrOp.has_value())
    return;

  attrOp.value()->removeAttr(attr);
  if (attrOp.value()->getAttrs().empty())
    rewriter.eraseOp(attrOp.value());
}

template <typename CastOpType>
bool hasSaturateOverflowModeAnnotation(CastOpType op) {
  auto overflowMode = getOverflowModeAnnotation(op);
  return overflowMode.has_value() && overflowMode->ends_with("saturate");
}

template <typename CastOpType>
CastUnsignedModeKind getSaturateUnsignedMode(CastOpType op) {
  const bool srcIsUnsigned =
      getCastAnnotationBool(op, kSaturateSrcUnsignedAttr).value_or(false);
  const bool dstIsUnsigned =
      getCastAnnotationBool(op, kSaturateDstUnsignedAttr).value_or(false);

  if (!srcIsUnsigned && !dstIsUnsigned)
    return CastUnsignedModeKind::SignedToSigned;
  if (!srcIsUnsigned && dstIsUnsigned)
    return CastUnsignedModeKind::SignedToUnsigned;
  if (srcIsUnsigned && !dstIsUnsigned)
    return CastUnsignedModeKind::UnsignedToSigned;
  return CastUnsignedModeKind::UnsignedToUnsigned;
}

template <typename Traits, typename CastOpType>
LogicalResult lowerTruncOverflowMode(CastOpType op,
                                     PatternRewriter &rewriter) {
  auto kind = getOverflowCastKind(op);
  if (!canNormalizeTruncOverflow<Traits>(op, kind))
    return failure();

  eraseCastAnnotationIfPresent(rewriter, op, kOverflowModeAttr);
  eraseCastAnnotationIfPresent(rewriter, op, kEnableOverflowAttr);

  Location loc = op.getLoc();
  Type outType = getElementTypeOrSelf(op.getDpsInits()[0].getType());
  Value input = op.getDpsInputs()[0];

  auto replaceWith = [&rewriter, op](Value value) {
    rewriter.replaceOp(op, value);
    return success();
  };

  switch (kind) {
  case OverflowCastKind::F32ToI16:
  case OverflowCastKind::F32ToI8: {
    // Preserve overflow on the stage where the backend can legally express it.
    if (useLegacyFloatTruncOverflowSequence<Traits>(kind)) {
      Value castI32 = Traits::castValue(
          rewriter, loc, op, input, rewriter.getI32Type(),
          CastExecutionKind::TruncWithOverflow);
      return replaceWith(Traits::castValue(rewriter, loc, op, castI32, outType,
                                           CastExecutionKind::TruncEnableOverflow));
    }
    Value castI32 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getI32Type(),
                                      CastExecutionKind::TruncEnableOverflow);
    return replaceWith(Traits::castValue(rewriter, loc, op, castI32, outType,
                                         CastExecutionKind::TruncWithOverflow));
  }
  case OverflowCastKind::F16ToI8: {
    if constexpr (Traits::supportsF16ToI8TruncOverflowPreprocess) {
      if (shouldPreprocessTruncOverflowInput<Traits>(op, kind)) {
        // Pre-wrap in FP, then finish with a plain trunc cast.
        Value overflowInput =
            preprocessTruncOverflowInput<Traits>(rewriter, loc, op, input);
        return replaceWith(Traits::castValue(
            rewriter, loc, op, overflowInput, outType,
            CastExecutionKind::Trunc));
      }
    }
    if (!supportsTwoStepTruncOverflow<Traits>(kind))
      return failure();
    Value castI32 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getI32Type(),
                                      CastExecutionKind::Trunc);
    return replaceWith(Traits::castValue(rewriter, loc, op, castI32, outType,
                                         CastExecutionKind::TruncWithOverflow));
  }
  case OverflowCastKind::I64ToI32:
  case OverflowCastKind::I32ToI16:
  case OverflowCastKind::I32ToI8:
  case OverflowCastKind::I16ToI8:
    if (!supportsDirectTruncWithOverflow<Traits>(kind) ||
        skipDirectTruncWithOverflowRewrite<Traits>(op, kind))
      return failure();
    return replaceWith(Traits::castValue(rewriter, loc, op, input, outType,
                                         CastExecutionKind::TruncWithOverflow));
  case OverflowCastKind::I64ToI16:
  case OverflowCastKind::I64ToI8: {
    // Split wide integer narrowing through I32 to keep overflow semantics.
    Value castI32 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getI32Type(),
                                      CastExecutionKind::TruncWithOverflow);
    return replaceWith(Traits::castValue(rewriter, loc, op, castI32, outType,
                                         CastExecutionKind::TruncWithOverflow));
  }
  case OverflowCastKind::Unsupported:
    return failure();
  }

  llvm_unreachable("unhandled overflow cast kind");
}

template <typename Traits, typename CastOpType>
LogicalResult lowerMembaseSaturateOverflowMode(CastOpType op,
                                               PatternRewriter &rewriter) {
  auto kind = getOverflowCastKind(op);
  Location loc = op.getLoc();
  Type outType = getElementTypeOrSelf(op.getDpsInits()[0].getType());
  Value input = op.getDpsInputs()[0];

  auto replaceWith = [&rewriter, op](Value value) {
    rewriter.replaceOp(op, value);
    return success();
  };

  // Membase saturation is synthesized via legal intermediate float/int casts.
  switch (kind) {
  case OverflowCastKind::F32ToI16:
    return replaceWith(Traits::castValue(rewriter, loc, op, input, outType,
                                         CastExecutionKind::Trunc));
  case OverflowCastKind::F32ToI8: {
    Value castF16 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getF16Type(),
                                      CastExecutionKind::Trunc);
    return replaceWith(Traits::castValue(rewriter, loc, op, castF16, outType,
                                         CastExecutionKind::Trunc));
  }
  case OverflowCastKind::F16ToI8:
    return replaceWith(Traits::castValue(rewriter, loc, op, input, outType,
                                         CastExecutionKind::Trunc));
  case OverflowCastKind::I64ToI32:
    return replaceWith(Traits::castValue(rewriter, loc, op, input, outType,
                                         CastExecutionKind::RInt));
  case OverflowCastKind::I64ToI16: {
    Value castF32 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getF32Type(),
                                      CastExecutionKind::Trunc);
    return replaceWith(Traits::castValue(rewriter, loc, op, castF32, outType,
                                         CastExecutionKind::Trunc));
  }
  case OverflowCastKind::I64ToI8: {
    Value castF32 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getF32Type(),
                                      CastExecutionKind::Trunc);
    Value castF16 = Traits::castValue(rewriter, loc, op, castF32,
                                      rewriter.getF16Type(),
                                      CastExecutionKind::Trunc);
    return replaceWith(Traits::castValue(rewriter, loc, op, castF16, outType,
                                         CastExecutionKind::Trunc));
  }
  case OverflowCastKind::I32ToI16:
    return replaceWith(Traits::castValue(rewriter, loc, op, input, outType,
                                         CastExecutionKind::RInt));
  case OverflowCastKind::I32ToI8: {
    Value castF32 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getF32Type(),
                                      CastExecutionKind::Trunc);
    Value castF16 = Traits::castValue(rewriter, loc, op, castF32,
                                      rewriter.getF16Type(),
                                      CastExecutionKind::Trunc);
    return replaceWith(Traits::castValue(rewriter, loc, op, castF16, outType,
                                         CastExecutionKind::Trunc));
  }
  case OverflowCastKind::I16ToI8: {
    Value castF16 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getF16Type(),
                                      CastExecutionKind::Trunc);
    return replaceWith(Traits::castValue(rewriter, loc, op, castF16, outType,
                                         CastExecutionKind::Trunc));
  }
  case OverflowCastKind::Unsupported:
    return failure();
  }

  llvm_unreachable("unhandled overflow cast kind");
}

template <typename Traits, typename CastOpType>
LogicalResult lowerRegbaseSaturateOverflowMode(
    CastOpType op, PatternRewriter &rewriter,
    CastUnsignedModeKind unsignedModeKind) {
  auto kind = getOverflowCastKind(op);
  Location loc = op.getLoc();
  Type outType = getElementTypeOrSelf(op.getDpsInits()[0].getType());
  Value input = op.getDpsInputs()[0];
  const bool isSignedToSigned =
      unsignedModeKind == CastUnsignedModeKind::SignedToSigned;

  auto replaceWith = [&rewriter, op](Value value) {
    rewriter.replaceOp(op, value);
    return success();
  };

  // Reg-based backends prefer native saturating narrows when unsigned modes fit.
  switch (kind) {
  case OverflowCastKind::F32ToI16: {
    Value castI32 = Traits::castValue(
        rewriter, loc, op, input, rewriter.getI32Type(),
        CastExecutionKind::TruncWithOverflow);
    return replaceWith(Traits::castValue(
        rewriter, loc, op, castI32, outType, CastExecutionKind::Trunc,
        CastSignKind::Preserve, /*enableSaturate=*/true));
  }
  case OverflowCastKind::F32ToI8: {
    Value castF16 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getF16Type(),
                                      CastExecutionKind::Trunc);
    return replaceWith(Traits::castValue(
        rewriter, loc, op, castF16, outType, CastExecutionKind::Trunc,
        CastSignKind::Preserve, /*enableSaturate=*/true));
  }
  case OverflowCastKind::F16ToI8:
    return replaceWith(Traits::castValue(
        rewriter, loc, op, input, outType, CastExecutionKind::Trunc,
        CastSignKind::Preserve, /*enableSaturate=*/true));
  case OverflowCastKind::I64ToI32:
    return replaceWith(Traits::castValue(
        rewriter, loc, op, input, outType, CastExecutionKind::Trunc,
        CastSignKind::Preserve, /*enableSaturate=*/true, unsignedModeKind));
  case OverflowCastKind::I64ToI16: {
    if (!isSignedToSigned) {
      return replaceWith(Traits::castValue(
          rewriter, loc, op, input, outType, CastExecutionKind::Trunc,
          CastSignKind::Preserve, /*enableSaturate=*/true, unsignedModeKind));
    }
    Value castF32 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getF32Type(),
                                      CastExecutionKind::Trunc);
    Value castF16 = Traits::castValue(rewriter, loc, op, castF32,
                                      rewriter.getF16Type(),
                                      CastExecutionKind::Trunc);
    return replaceWith(Traits::castValue(
        rewriter, loc, op, castF16, outType, CastExecutionKind::Trunc,
        CastSignKind::Preserve, /*enableSaturate=*/true));
  }
  case OverflowCastKind::I64ToI8: {
    if (!isSignedToSigned) {
      return replaceWith(Traits::castValue(
          rewriter, loc, op, input, outType, CastExecutionKind::Trunc,
          CastSignKind::Preserve, /*enableSaturate=*/true, unsignedModeKind));
    }
    Value castF32 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getF32Type(),
                                      CastExecutionKind::Trunc);
    Value castF16 = Traits::castValue(rewriter, loc, op, castF32,
                                      rewriter.getF16Type(),
                                      CastExecutionKind::Trunc);
    return replaceWith(Traits::castValue(
        rewriter, loc, op, castF16, outType, CastExecutionKind::Trunc,
        CastSignKind::Preserve, /*enableSaturate=*/true));
  }
  case OverflowCastKind::I32ToI16:
    return replaceWith(Traits::castValue(
        rewriter, loc, op, input, outType, CastExecutionKind::Trunc,
        CastSignKind::Preserve, /*enableSaturate=*/true, unsignedModeKind));
  case OverflowCastKind::I32ToI8: {
    if (unsignedModeKind == CastUnsignedModeKind::UnsignedToSigned) {
      // Route through an unsigned-preserving narrow before reinterpreting sign.
      Value castI16 = Traits::castValue(
          rewriter, loc, op, input, rewriter.getI16Type(),
          CastExecutionKind::Trunc, CastSignKind::Preserve,
          /*enableSaturate=*/true,
          CastUnsignedModeKind::UnsignedToSigned);
      Value castF16 = Traits::castValue(rewriter, loc, op, castI16,
                                        rewriter.getF16Type(),
                                        CastExecutionKind::Trunc);
      return replaceWith(Traits::castValue(
          rewriter, loc, op, castF16, outType, CastExecutionKind::Trunc,
          CastSignKind::Preserve, /*enableSaturate=*/true));
    }
    if (!isSignedToSigned) {
      return replaceWith(Traits::castValue(
          rewriter, loc, op, input, outType, CastExecutionKind::Trunc,
          CastSignKind::Preserve, /*enableSaturate=*/true, unsignedModeKind));
    }
    Value castF32 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getF32Type(),
                                      CastExecutionKind::Trunc);
    Value castF16 = Traits::castValue(rewriter, loc, op, castF32,
                                      rewriter.getF16Type(),
                                      CastExecutionKind::Trunc);
    return replaceWith(Traits::castValue(
        rewriter, loc, op, castF16, outType, CastExecutionKind::Trunc,
        CastSignKind::Preserve, /*enableSaturate=*/true));
  }
  case OverflowCastKind::I16ToI8: {
    if (unsignedModeKind == CastUnsignedModeKind::UnsignedToSigned) {
      // Avoid signed saturation too early; keep the narrow unsigned first.
      Value castI8 = Traits::castValue(
          rewriter, loc, op, input, rewriter.getI8Type(),
          CastExecutionKind::Trunc, CastSignKind::Unsigned,
          /*enableSaturate=*/true,
          CastUnsignedModeKind::UnsignedToUnsigned);
      Value castF16 =
          Traits::castValue(rewriter, loc, op, castI8, rewriter.getF16Type(),
                            CastExecutionKind::Trunc, CastSignKind::Unsigned);
      return replaceWith(Traits::castValue(
          rewriter, loc, op, castF16, outType, CastExecutionKind::Trunc,
          CastSignKind::Preserve, /*enableSaturate=*/true));
    }
    if (!isSignedToSigned) {
      return replaceWith(Traits::castValue(
          rewriter, loc, op, input, outType, CastExecutionKind::Trunc,
          CastSignKind::Preserve, /*enableSaturate=*/true, unsignedModeKind));
    }
    Value castF16 = Traits::castValue(rewriter, loc, op, input,
                                      rewriter.getF16Type(),
                                      CastExecutionKind::Trunc);
    return replaceWith(Traits::castValue(
        rewriter, loc, op, castF16, outType, CastExecutionKind::Trunc,
        CastSignKind::Preserve, /*enableSaturate=*/true, unsignedModeKind));
  }
  case OverflowCastKind::Unsupported:
    return failure();
  }

  llvm_unreachable("unhandled overflow cast kind");
}

template <typename Traits, typename CastOpType>
LogicalResult lowerSaturateOverflowMode(CastOpType op,
                                        PatternRewriter &rewriter) {
  CastUnsignedModeKind unsignedModeKind = getSaturateUnsignedMode(op);
  eraseCastAnnotationIfPresent(rewriter, op, kOverflowModeAttr);

  if (Traits::archIsRegbased()) {
    eraseCastAnnotationIfPresent(rewriter, op, kSaturateSrcUnsignedAttr);
    eraseCastAnnotationIfPresent(rewriter, op, kSaturateDstUnsignedAttr);
    return lowerRegbaseSaturateOverflowMode<Traits>(op, rewriter,
                                                   unsignedModeKind);
  }
  return lowerMembaseSaturateOverflowMode<Traits>(op, rewriter);
}

template <typename Traits, typename CastOpType>
LogicalResult lowerOverflowMode(CastOpType op, PatternRewriter &rewriter) {
  if (hasSaturateOverflowModeAnnotation(op)) {
    return lowerSaturateOverflowMode<Traits>(op, rewriter);
  }

  return lowerTruncOverflowMode<Traits>(op, rewriter);
}

template <typename Traits, typename CastOpType>
Value castInToF32ToOut(CastOpType op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Type dstType = getElementTypeOrSelf(op.getDpsInits()[0].getType());
  Value castSrcToF32 = Traits::castValue(
      rewriter, loc, op, op.getDpsInputs()[0], rewriter.getF32Type(),
      CastExecutionKind::Default, CastSignKind::Preserve);
  if (dstType.isF32())
    return castSrcToF32;

  return Traits::castValue(rewriter, loc, op, castSrcToF32, dstType,
                           CastExecutionKind::Default,
                           dstType.isInteger() ? CastSignKind::Signed
                                               : CastSignKind::Preserve);
}

template <typename Traits, typename CastOpType>
Value castU32ToI64ToF32(CastOpType op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value castU32ToI64 = Traits::castValue(
      rewriter, loc, op, op.getDpsInputs()[0], rewriter.getI64Type(),
      CastExecutionKind::Default, CastSignKind::Preserve);
  return Traits::castValue(rewriter, loc, op, castU32ToI64,
                           rewriter.getF32Type(), CastExecutionKind::Default,
                           CastSignKind::Signed);
}

template <typename Traits, typename CastOpType>
Value castU32ToI64ToF32ToOut(CastOpType op, Type targetType,
                             PatternRewriter &rewriter) {
  Value u32ToF32Result = castU32ToI64ToF32<Traits>(op, rewriter);
  return Traits::castValue(rewriter, op.getLoc(), op, u32ToF32Result,
                           targetType, CastExecutionKind::Default,
                           CastSignKind::Signed);
}

template <typename Traits, typename CastOpType>
Value castSrcToFp16ToTargetType(CastOpType op, Type targetType,
                                PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value castSrcToF16 = Traits::castValue(
      rewriter, loc, op, op.getDpsInputs()[0], rewriter.getF16Type(),
      CastExecutionKind::Default, CastSignKind::Preserve);
  if (targetType.isF16())
    return castSrcToF16;

  if (targetType.isBF16()) {
    Value castF16ToF32 =
        Traits::castValue(rewriter, loc, op, castSrcToF16, rewriter.getF32Type(),
                          CastExecutionKind::Default,
                          CastSignKind::Preserve);
    return Traits::castValue(rewriter, loc, op, castF16ToF32, targetType,
                             CastExecutionKind::Default,
                             CastSignKind::Preserve);
  }

  return Traits::castValue(rewriter, loc, op, castSrcToF16, targetType,
                           CastExecutionKind::Default,
                           CastSignKind::Signed);
}

template <typename Traits, typename CastOpType>
Value castSrcTypeToI1ByCmp(CastOpType op, Type srcType,
                           PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value input = op.getDpsInputs()[0];
  Value castedValue;
  if (srcType.isInteger(8)) {
    castedValue = Traits::castValue(rewriter, loc, op, input,
                                    rewriter.getF16Type());
  } else if (srcType.isInteger(16)) {
    castedValue = Traits::castValue(rewriter, loc, op, input,
                                    rewriter.getF16Type(),
                                    CastExecutionKind::RInt);
  } else if (srcType.isInteger(32) || srcType.isInteger(64) ||
             srcType.isBF16()) {
    castedValue = Traits::castValue(rewriter, loc, op, input,
                                    rewriter.getF32Type(),
                                    CastExecutionKind::RInt);
  } else if (srcType.isF16() || srcType.isF32()) {
    castedValue = input;
  } else {
    llvm_unreachable("unsupported source type for cast-to-i1 lowering");
  }

  Value zero = Traits::buildZeroForCompare(rewriter, loc, op, castedValue);
  return Traits::createCmpOp(rewriter, loc, castedValue, zero, CompareKind::NE);
}

template <typename Traits, typename CastOpType>
Value castI8ToI64(CastOpType op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  if (!Traits::archIsRegbased()) {
    Value i8ToF32Result =
        castSrcToFp16ToTargetType<Traits>(op, rewriter.getF32Type(), rewriter);
    return Traits::castValue(rewriter, loc, op, i8ToF32Result,
                             rewriter.getI64Type(),
                             CastExecutionKind::Default,
                             CastSignKind::Signed);
  }

  Value castValue = Traits::castValue(rewriter, loc, op, op.getDpsInputs()[0],
                                      rewriter.getI32Type(),
                                      CastExecutionKind::Default,
                                      CastSignKind::Preserve);
  return Traits::castValue(rewriter, loc, op, castValue, rewriter.getI64Type(),
                           CastExecutionKind::Default,
                           CastSignKind::Preserve);
}

template <typename Traits, typename CastOpType>
Value castI1ToI32ViaF16(CastOpType op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value castF16Value = Traits::castValue(rewriter, loc, op, op.getDpsInputs()[0],
                                         rewriter.getF16Type(),
                                         CastExecutionKind::RInt);
  return Traits::castValue(rewriter, loc, op, castF16Value,
                           rewriter.getI32Type(), CastExecutionKind::RInt);
}

template <typename Traits, typename CastOpType>
Value castI1ToI64ViaF32(CastOpType op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value castF32Value = Traits::castValue(rewriter, loc, op, op.getDpsInputs()[0],
                                         rewriter.getF32Type(),
                                         CastExecutionKind::RInt);
  return Traits::castValue(rewriter, loc, op, castF32Value,
                           rewriter.getI64Type(),
                           CastExecutionKind::Default,
                           CastSignKind::Signed);
}

template <typename Traits, typename CastOpType>
Value castI32ToF16ViaF32(CastOpType op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value castF32Value = Traits::castValue(rewriter, loc, op, op.getDpsInputs()[0],
                                         rewriter.getF32Type());
  return Traits::castValue(rewriter, loc, op, castF32Value,
                           rewriter.getF16Type());
}

template <typename Traits, typename CastOpType>
Value castI16ToI64(CastOpType op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value castValue;
  if (Traits::archIsRegbased()) {
    castValue = Traits::castValue(rewriter, loc, op, op.getDpsInputs()[0],
                                  rewriter.getI32Type(),
                                  CastExecutionKind::Default,
                                  CastSignKind::Preserve);
  } else {
    castValue = Traits::castValue(rewriter, loc, op, op.getDpsInputs()[0],
                                  rewriter.getF32Type(),
                                  CastExecutionKind::RInt);
  }

  return Traits::castValue(rewriter, loc, op, castValue, rewriter.getI64Type(),
                           CastExecutionKind::Default,
                           CastSignKind::Preserve);
}

template <typename Traits, typename CastOpType>
Value castI16ToI32ViaF32(CastOpType op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value castF32Value = Traits::castValue(rewriter, loc, op, op.getDpsInputs()[0],
                                         rewriter.getF32Type(),
                                         CastExecutionKind::RInt);
  return Traits::castValue(rewriter, loc, op, castF32Value,
                           rewriter.getI32Type());
}

template <typename Traits, typename CastOpType>
Value castF32ToU32ViaI64(CastOpType op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value castF32ToI64 = Traits::castValue(rewriter, loc, op, op.getDpsInputs()[0],
                                         rewriter.getI64Type(),
                                         CastExecutionKind::Default,
                                         CastSignKind::Signed);
  return Traits::castValue(rewriter, loc, op, castF32ToI64,
                           rewriter.getI32Type(), CastExecutionKind::Default,
                           CastSignKind::Signed);
}

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_CASTINGTEMPLATEHELPERS_H
