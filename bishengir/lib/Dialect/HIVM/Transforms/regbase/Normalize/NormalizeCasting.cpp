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
#include "bishengir/Transforms/regbase/Normalize/NormalizeCastingTemplate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include <optional>

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

struct HIVMNormalizeBrcCastTraits : public hivm::NormalizeTraitsBase {
  static bool shouldNormalizeBrcCast(hivm::VCastOp op) {
    return op.hasPureTensorSemantics() && op.getBroadcast().empty() &&
           op.getTranspose().empty();
  }

  static bool shouldSkipBrcCast(hivm::VCastOp, Operation *defOp, Type srcTy,
                                TensorType dstTy) {
    return isa<hivm::VBrcOp>(defOp) &&
           getElementTypeOrSelf(srcTy).isF16() &&
           (dstTy.getElementType().isInteger(1) ||
            dstTy.getElementType().isInteger(8));
  }
};

struct HIVMNormalizeFillCastToTensorBrcTraits
    : public hivm::NormalizeTraitsBase {
  static bool shouldNormalizeFillCastToTensorBrc(hivm::VCastOp op) {
    return op.hasPureTensorSemantics() && op.getBroadcast().empty() &&
           op.getTranspose().empty();
  }
};

struct HIVMNormalizeScalarExtensionTraits : public hivm::NormalizeTraitsBase {
  template <typename OpTy> static bool shouldSkipScalarExtension(OpTy op) {
    return isa<hivm::VCastOp>(op->getParentOp()) ||
           isa<linalg::MatmulOp>(op->getParentOp()) ||
           isa<linalg::BatchMatmulOp>(op->getParentOp());
  }
};

struct HIVMNormalizeAnyToF32UnaryRecOpTraits
    : public hivm::NormalizeTraitsBase {
  static bool shouldNormalizeAnyToF32UnaryRec(hivm::VRecOp op) {
    return op.hasPureTensorSemantics() && op.getBroadcast().empty() &&
           op.getTranspose().empty();
  }
};

struct HIVMNormalizeScalarCastTraits : public hivm::NormalizeTraitsBase {
  static bool shouldNormalizeScalarCast(hivm::VCastOp op) {
    return op.hasPureTensorSemantics() && op.getBroadcast().empty() &&
           op.getTranspose().empty();
  }

  static Value castScalarFromRankZeroTensor(PatternRewriter &rewriter,
                                            Location loc, hivm::VCastOp op,
                                            Value scalar, Type dstType) {
    auto tensorType = RankedTensorType::get({1}, scalar.getType());
    Value fromElementsOp =
        rewriter.create<tensor::FromElementsOp>(loc, tensorType, scalar);
    auto castOp = hivm::castTo(
        rewriter, loc, fromElementsOp, op.getRoundModeAttr(), dstType);
    castOp.setCastAttr(op.getCastAttr());
    if (hasSaturateOverflowModeAnnotation(op)) {
      castOp->setAttr(kOverflowModeAttr, rewriter.getStringAttr("saturate"));
      if (getCastAnnotationBool(op, kSaturateSrcUnsignedAttr).value_or(false))
        castOp->setAttr(kSaturateSrcUnsignedAttr, rewriter.getUnitAttr());
      if (getCastAnnotationBool(op, kSaturateDstUnsignedAttr).value_or(false))
        castOp->setAttr(kSaturateDstUnsignedAttr, rewriter.getUnitAttr());
    }
    if (op->getAttrOfType<BoolAttr>(kEnableOverflowAttr))
      castOp->setAttr(kEnableOverflowAttr, rewriter.getBoolAttr(true));
    Value castedTensor = castOp->getResults().empty() ? castOp.getSingleDst()
                                                      : castOp->getResults()[0];
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    return rewriter.create<tensor::ExtractOp>(loc, castedTensor, ValueRange{c0});
  }
};

struct HIVMNormalizeSortTraits : public hivm::NormalizeTraitsBase {
  static bool isSupportedSortElementType(Type elemType) {
    auto floatType = dyn_cast<FloatType>(elemType);
    if (floatType && (floatType.isF16() || floatType.isF32()))
      return true;

    auto intType = dyn_cast<IntegerType>(elemType);
    return intType && (intType.isInteger(32) || intType.isInteger(64));
  }

  static Value createCastOp(PatternRewriter &rewriter, Location loc,
                            Value input, Type targetElemType) {
    return NormalizeTraitsBase::createCastOp(rewriter, loc, input,
                                             targetElemType,
                                             CastRoundKind::Round);
  }

  static Value createSortOp(PatternRewriter &rewriter, hivm::VSortOp op,
                            Value input) {
    Value dst = utils::createEmptyOpWithTargetElemType(
        rewriter, op.getLoc(), input, getElementTypeOrSelf(input.getType()));
    return rewriter
        .create<hivm::VSortOp>(op.getLoc(), TypeRange{input.getType()}, input,
                               ValueRange{dst}, op.getDescending(),
                               op.getSortAxis())
        .getResult()
        .front();
  }
};

using NormalizeBrcCast =
    NormalizeBrcCastTemplate<hivm::VCastOp, HIVMNormalizeBrcCastTraits>;
using NormalizefillCastToTensorBrc =
    NormalizeFillCastToTensorBrcTemplate<hivm::VCastOp,
                                         HIVMNormalizeFillCastToTensorBrcTraits>;
template <typename CastOp>
using NormalizeScalarExtension =
    NormalizeScalarExtensionTemplate<CastOp,
                                     HIVMNormalizeScalarExtensionTraits>;
using NormalizeAnyToF32UnaryRecOp =
    NormalizeAnyToF32UnaryRecOpTemplate<hivm::VRecOp,
                                        HIVMNormalizeAnyToF32UnaryRecOpTraits>;
using NormalizeCastLoweringOp =
    NormalizeCastLoweringOpTemplate<hivm::VCastOp,
                                    HIVMNormalizeCastLoweringTraits>;
using NormalizeScalarCastOp =
    NormalizeScalarCastOpTemplate<hivm::VCastOp,
                                  HIVMNormalizeScalarCastTraits>;
using NormalizeSortOp =
    NormalizeSortOpTemplate<hivm::VSortOp, HIVMNormalizeSortTraits>;
} // namespace mlir

void mlir::hivm::populateNormalizeCastingPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeBrcCast>(ctx);
  patterns.add<NormalizefillCastToTensorBrc>(ctx);
  patterns.add<NormalizeAnyToF32UnaryRecOp>(ctx);
  patterns.add<NormalizeCastLoweringOp>(ctx);
}

void mlir::hivm::populateNormalizeFinalCastingPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeScalarExtension<arith::ExtFOp>>(ctx);
  if (archIsRegbased)
    patterns.add<NormalizeScalarCastOp>(ctx);
}

void mlir::hivm::populateNormalizeSortPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeSortOp>(patterns.getContext());
}
