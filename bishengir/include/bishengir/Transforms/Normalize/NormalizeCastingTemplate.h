//===-------- NormalizeCastingTemplate.h ----------------------------------===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZECASTINGTEMPLATE_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZECASTINGTEMPLATE_H

#include "bishengir/Transforms/Normalize/Utils/CastingTemplateHelpers.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

template <typename CastOpType, typename Traits>
struct NormalizeCastLoweringOpTemplate : public OpRewritePattern<CastOpType> {
public:
  using OpRewritePattern<CastOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeCast(op))
      return failure();

    auto inType = getElementTypeOrSelf(op.getDpsInputs()[0].getType());
    auto outType = getElementTypeOrSelf(op.getDpsInits()[0].getType());
    const bool isRegBased = Traits::archIsRegbased();

    auto replaceWith = [&rewriter, op](Value value) {
      rewriter.replaceOp(op, value);
      return success();
    };

    if (requiresOverflowNormalization(op)) {
      if (succeeded(lowerOverflowMode<Traits>(op, rewriter)))
        return success();
      return failure();
    }

    const bool isI64ToF16 = inType.isInteger(64) && outType.isF16();
    const bool isWideIntegerToBF16 =
        (inType.isInteger(64) || inType.isInteger(32) || inType.isInteger(16)) &&
        outType.isBF16();
    const bool isU16ToF16 = inType.isInteger(16) && outType.isF16() &&
                            Traits::isUnsignedCast(op);
    if (isI64ToF16 || isWideIntegerToBF16 || isU16ToF16) {
      return replaceWith(castInToF32ToOut<Traits>(op, rewriter));
    }

    const bool isU32ToF32 = inType.isInteger(32) && outType.isF32() &&
                            Traits::isUnsignedCast(op);
    if (isU32ToF32) {
      return replaceWith(castU32ToI64ToF32<Traits>(op, rewriter));
    }

    const bool isU32ToF16 = inType.isInteger(32) && outType.isF16() &&
                            Traits::isUnsignedCast(op);
    const bool isU32ToBF16 = inType.isInteger(32) && outType.isBF16() &&
                             Traits::isUnsignedCast(op);
    if (isU32ToF16 || isU32ToBF16) {
      return replaceWith(
          castU32ToI64ToF32ToOut<Traits>(op, outType, rewriter));
    }

    const bool isI8ToI64 = inType.isInteger(8) && outType.isInteger(64);
    if (isI8ToI64) {
      return replaceWith(castI8ToI64<Traits>(op, rewriter));
    }

    const bool isI8ToF32 = inType.isInteger(8) && outType.isF32();
    const bool isI8ToI32 = inType.isInteger(8) && outType.isInteger(32);
    const bool isI8ToI16 = inType.isInteger(8) && outType.isInteger(16);
    const bool isI8ToBF16 = inType.isInteger(8) && outType.isBF16();
    const bool isI1ToI16 = inType.isInteger(1) && outType.isInteger(16);
    const bool isI1ToF32 = inType.isInteger(1) && outType.isF32();
    if (isI8ToF32 || isI8ToBF16 || isI1ToI16 ||
        (!isRegBased && (isI8ToI32 || isI8ToI16 || isI1ToF32))) {
      return replaceWith(
          castSrcToFp16ToTargetType<Traits>(op, outType, rewriter));
    }

    const bool isI1ToI32 = inType.isInteger(1) && outType.isInteger(32);
    if (!isRegBased && isI1ToI32) {
      return replaceWith(castI1ToI32ViaF16<Traits>(op, rewriter));
    }

    const bool isI1ToI64 = inType.isInteger(1) && outType.isInteger(64);
    if (isI1ToI64) {
      return replaceWith(castI1ToI64ViaF32<Traits>(op, rewriter));
    }

    const bool isI32ToF16 = inType.isInteger(32) && outType.isF16();
    if (isI32ToF16) {
      return replaceWith(castI32ToF16ViaF32<Traits>(op, rewriter));
    }

    const bool isI64ToI1 = inType.isInteger(64) && outType.isInteger(1);
    const bool isI32ToI1 = inType.isInteger(32) && outType.isInteger(1);
    const bool isI16ToI1 = inType.isInteger(16) && outType.isInteger(1);
    const bool isI8ToI1 = inType.isInteger(8) && outType.isInteger(1);
    const bool isBf16ToI1 = inType.isBF16() && outType.isInteger(1);
    const bool isF32ToI1 = inType.isF32() && outType.isInteger(1);
    const bool isF16ToI1 = inType.isF16() && outType.isInteger(1);
    if (isI64ToI1 || isI32ToI1 || isI16ToI1 || isI8ToI1 || isBf16ToI1 ||
        isF32ToI1 || isF16ToI1) {
      return replaceWith(castSrcTypeToI1ByCmp<Traits>(op, inType, rewriter));
    }

    const bool isI16ToI64 = inType.isInteger(16) && outType.isInteger(64);
    if (isI16ToI64) {
      return replaceWith(castI16ToI64<Traits>(op, rewriter));
    }

    const bool isI16ToI32 = inType.isInteger(16) && outType.isInteger(32);
    if (!isRegBased && isI16ToI32) {
      return replaceWith(castI16ToI32ViaF32<Traits>(op, rewriter));
    }

    const bool isAnyToF8 = (!inType.isF32()) &&
                           (outType.isFloat8E4M3FN() || outType.isFloat8E5M2());
    if (isAnyToF8) {
      return replaceWith(castInToF32ToOut<Traits>(op, rewriter));
    }

    const bool isF8ToAny = (inType.isFloat8E4M3FN() || inType.isFloat8E5M2()) &&
                           (!outType.isF32());
    if (isF8ToAny) {
      return replaceWith(castInToF32ToOut<Traits>(op, rewriter));
    }

    const bool isF32ToU32 = inType.isF32() && outType.isInteger(32) &&
                            Traits::isUnsignedCast(op);
    if (isF32ToU32) {
      return replaceWith(castF32ToU32ViaI64<Traits>(op, rewriter));
    }

    return failure();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZECASTINGTEMPLATE_H
