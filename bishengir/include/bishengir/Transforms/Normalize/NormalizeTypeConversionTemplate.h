//===-------- NormalizeTypeConversionTemplate.h ---------------------------===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZETYPECONVERSIONTEMPLATE_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZETYPECONVERSIONTEMPLATE_H

#include "bishengir/Transforms/Normalize/Utils/Kinds.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include <cstdint>

namespace mlir {

/// Normalizes bool/i8 ops that share the generic
/// cast-up / rebuild-same-op / cast-back skeleton.
///
/// The exact widened element type is decided by `Traits::getTargetType`.
/// Typical cases are:
/// - `i1 -> f16`, rebuild, then cast/compare back to `i1`
/// - `i8 -> f16/f32`, rebuild, then cast back to `i8`
template <typename ElemType, typename OpType, typename Traits>
struct NormalizeToTargetTemplate : public OpRewritePattern<OpType>,
                                   private Traits {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalize(op))
      return failure();

    SmallVector<Value> inputs = Traits::getInputs(op);
    SmallVector<Value> outputs = Traits::getOutputs(op);
    if (!hasAnyNormalizedElemType(inputs) &&
        !hasAnyNormalizedElemType(outputs)) {
      return failure();
    }

    Type targetType = Traits::getTargetType(rewriter, op);
    if (!targetType)
      return failure();

    CastSignKind intKind = Traits::getCastSignKind(op);
    Location loc = op.getLoc();
    SmallVector<Value> newInputs =
        normalizeValuesToTargetType(rewriter, loc, inputs, targetType, intKind);
    SmallVector<Value> newOutputs = normalizeValuesToTargetType(
        rewriter, loc, outputs, targetType, intKind);
    Operation *newOp = Traits::rebuildOpInTargetType(rewriter, loc, op,
                                                     newInputs, newOutputs);

    Traits::replaceResults(rewriter, op, newOp->getResults(), intKind);
    return success();
  }

private:
  static bool isNormalizedElemType(Type elemType) {
    if constexpr (std::is_same_v<ElemType, bool>)
      return elemType.isInteger(1);
    if constexpr (std::is_same_v<ElemType, int8_t>)
      return elemType.isInteger(8);
    return false;
  }

  static bool hasAnyNormalizedElemType(ValueRange values) {
    return llvm::any_of(values, [](Value value) {
      return isNormalizedElemType(getElementTypeOrSelf(value.getType()));
    });
  }

  static SmallVector<Value> normalizeValuesToTargetType(
      PatternRewriter &rewriter, Location loc, ValueRange values,
      Type targetType, CastSignKind intKind = CastSignKind::Signed) {
    SmallVector<Value> result;
    result.reserve(values.size());
    for (Value value : values) {
      Type elemType = getElementTypeOrSelf(value.getType());
      if (!isNormalizedElemType(elemType)) {
        result.push_back(value);
        continue;
      }

      result.push_back(Traits::createCastOp(rewriter, loc, value, targetType,
                                             CastRoundKind::Default, Value(),
                                             intKind));
    }
    return result;
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZETYPECONVERSIONTEMPLATE_H