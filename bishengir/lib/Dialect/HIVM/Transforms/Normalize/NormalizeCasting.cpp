//===-------- NormalizeCasting.cpp --------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Normalize/NormalizeCastingTemplate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
struct HIVMNormalizeCastLoweringTraits : public hivm::NormalizeTraitsBase {
  static constexpr bool supportsF16ToI8TruncOverflowPreprocess = false;

  static bool shouldNormalizeCast(hivm::VCastOp op) {
    const bool hasSaturateOverflowMode =
        mlir::hasSaturateOverflowModeAnnotation(op);
    return op.hasPureTensorSemantics() && op.getBroadcast().empty() &&
           op.getTranspose().empty() &&
           (hasSaturateOverflowMode ||
            !mlir::isTerminalNativeSaturateCast<HIVMNormalizeCastLoweringTraits>(op));
  }

  static bool hasOverflowEnabled(hivm::VCastOp op) {
    if (auto enableOverflow = getCastAnnotationBool(op, kEnableOverflowAttr))
      return *enableOverflow;

    auto overflowMode = getOverflowModeAnnotation(op);
    return overflowMode.has_value() && *overflowMode == "trunc";
  }

  static bool hasSaturateEnabled(hivm::VCastOp op) {
    return hasSaturateOverflowModeAnnotation(op);
  }

  static bool isUnsignedCast(hivm::VCastOp op) {
    return op.getCast() == hivm::TypeFn::cast_unsigned;
  }

  static Value buildZeroForCompare(PatternRewriter &rewriter, Location loc,
                                   hivm::VCastOp, Value input) {
    Type elementType = getElementTypeOrSelf(input.getType());
    Value zeroScalar = rewriter
                           .create<arith::ConstantOp>(
                               loc, elementType, rewriter.getFloatAttr(elementType, 0.0))
                           .getResult();
    Value zeroDst = utils::createEmptyOp(rewriter, loc, input);
    auto brcOp = rewriter.create<hivm::VBrcOp>(
        loc, TypeRange(zeroDst.getType()), zeroScalar, zeroDst);
    return brcOp->getResult(0);
  }
};

using NormalizeCastLoweringOp =
    NormalizeCastLoweringOpTemplate<hivm::VCastOp,
                                    HIVMNormalizeCastLoweringTraits>;
} // namespace mlir

void mlir::hivm::populateNormalizeCastingPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeCastLoweringOp>(ctx);
}
