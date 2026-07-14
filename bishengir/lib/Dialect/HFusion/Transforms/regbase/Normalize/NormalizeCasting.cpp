//===- NormalizeCasting.cpp ----------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeUtils.h"
#include "bishengir/Transforms/regbase/Normalize/NormalizeCastingTemplate.h"

namespace mlir::hfusion {
struct HFusionNormalizeCastLoweringTraits : public hfusion::NormalizeTraitsBase {
  static constexpr bool supportsF16ToI8TruncOverflowPreprocess = true;

  static bool shouldNormalizeCast(hfusion::CastOp op) {
    const bool hasSaturateOverflowMode =
        mlir::hasSaturateOverflowModeAnnotation(op);
    return op.hasPureTensorSemantics() &&
           (hasSaturateOverflowMode ||
            !mlir::isTerminalNativeSaturateCast<HFusionNormalizeCastLoweringTraits>(op));
  }

  static bool hasOverflowEnabled(hfusion::CastOp op) {
    return op.getEnableOverflow();
  }

  static bool hasSaturateEnabled(hfusion::CastOp op) {
    return op.getEnableSaturate();
  }

  static bool isUnsignedCast(hfusion::CastOp op) {
    return op.getCast() == hfusion::TypeFn::cast_unsigned;
  }


  static Value buildZeroForCompare(PatternRewriter &rewriter, Location loc,
                                   hfusion::CastOp, Value input) {
    Type elementType = getElementTypeOrSelf(input.getType());
    return rewriter
        .create<arith::ConstantOp>(loc, elementType,
                                   rewriter.getFloatAttr(elementType, 0.0))
        .getResult();
  }
};

struct HFusionNormalizeBrcCastTraits : public hfusion::NormalizeTraitsBase {
  static bool shouldNormalizeBrcCast(hfusion::CastOp op) {
    return op.hasPureTensorSemantics();
  }

  static bool shouldSkipBrcCast(hfusion::CastOp, Operation *defOp, Type srcTy,
                                TensorType dstTy) {
    // Disable conversion from brc f16 + cast i8/bool as combined with
    // NormalizeToTargetType pass causes infinite loop.
    return isa<linalg::BroadcastOp>(defOp) && isF16ElemType(srcTy) &&
           (isI1ElemType(dstTy) || isI8ElemType(dstTy));
  }
};

struct HFusionNormalizeFillCastToTensorBrcTraits
    : public hfusion::NormalizeTraitsBase {
  static bool shouldNormalizeFillCastToTensorBrc(hfusion::CastOp op) {
    return op.hasPureTensorSemantics();
  }
};

using HFusionNormalizeTruncfExtfTraits = hfusion::NormalizeTraitsBase;

struct HFusionNormalizeTruncfBf16Traits : public hfusion::NormalizeTraitsBase {
  static bool isInsideDialectCast(Operation &op) {
    return isa<hfusion::CastOp>(op.getParentOp());
  }
};

struct HFusionNormalizeScalarExtensionTraits
    : public hfusion::NormalizeTraitsBase {
  template <typename OpTy> static bool shouldSkipScalarExtension(OpTy op) {
    return isa<hfusion::CastOp>(op->getParentOp()) ||
           isa<linalg::MatmulOp>(op->getParentOp()) ||
           isa<linalg::BatchMatmulOp>(op->getParentOp());
  }
};

struct HFusionNormalizeAnyToF32UnaryRecOpTraits
    : public hfusion::NormalizeTraitsBase {
  static bool shouldNormalizeAnyToF32UnaryRec(hfusion::ElemwiseUnaryOp op) {
    return op.hasPureTensorSemantics() && op.getFun() == hfusion::UnaryFn::rec;
  }
};

struct HFusionNormalizeScalarCastTraits : public hfusion::NormalizeTraitsBase {
  static bool shouldNormalizeScalarCast(hfusion::CastOp op) {
    return op.hasPureTensorSemantics();
  }

  static Value castScalarFromRankZeroTensor(PatternRewriter &rewriter,
                                            Location loc, hfusion::CastOp op,
                                            Value scalar, Type dstType) {
    auto tensorType = RankedTensorType::get({1}, scalar.getType());
    Value fromElementsOp =
        rewriter.create<tensor::FromElementsOp>(loc, tensorType, scalar);
    Value castOp = hfusion::castTo(rewriter, fromElementsOp, dstType,
                                   op.getRoundMode(), std::nullopt,
                                   op.getEnableOverflow());
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    return rewriter.create<tensor::ExtractOp>(loc, castOp, ValueRange{c0});
  }
};

struct HFusionNormalizeSortTraits : public hfusion::NormalizeTraitsBase {
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

  static Value createSortOp(PatternRewriter &rewriter, hfusion::SortOp op,
                            Value input) {
    return rewriter
        .create<hfusion::SortOp>(op.getLoc(), input.getType(), input,
                                 op.getDescending(), op.getSortAxis())
        ->getResult(0);
  }
};

using NormalizeBrcCastRegBase =
    NormalizeBrcCastTemplate<hfusion::CastOp, HFusionNormalizeBrcCastTraits>;
using NormalizefillCastToTensorBrcRegBase =
    NormalizeFillCastToTensorBrcTemplate<hfusion::CastOp,
                                         HFusionNormalizeFillCastToTensorBrcTraits>;
using NormalizetruncfExtfRegBase =
    NormalizeTruncfExtfTemplate<arith::ExtFOp,
                                HFusionNormalizeTruncfExtfTraits>;
using NormalizetruncfBf16RegBase =
    NormalizeTruncfBf16Template<arith::TruncFOp,
                                HFusionNormalizeTruncfBf16Traits>;
template <typename CastOp>
using NormalizeScalarExtensionRegBase =
    NormalizeScalarExtensionTemplate<CastOp,
                                     HFusionNormalizeScalarExtensionTraits>;
using NormalizeAnyToF32UnaryRecOpRegBase =
    NormalizeAnyToF32UnaryRecOpTemplate<hfusion::ElemwiseUnaryOp,
                                        HFusionNormalizeAnyToF32UnaryRecOpTraits>;
using NormalizeCastLoweringOpRegBase =
    NormalizeCastLoweringOpTemplate<hfusion::CastOp,
                                    HFusionNormalizeCastLoweringTraits>;
using NormalizeScalarCastOpRegBase =
    NormalizeScalarCastOpTemplate<hfusion::CastOp,
                                  HFusionNormalizeScalarCastTraits>;
using NormalizeSortOpRegBase =
    NormalizeSortOpTemplate<hfusion::SortOp, HFusionNormalizeSortTraits>;

void populateNormalizeCastingPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeBrcCastRegBase>(ctx);
  patterns.add<NormalizefillCastToTensorBrcRegBase>(ctx);
  patterns.add<NormalizeAnyToF32UnaryRecOpRegBase>(ctx);
  patterns.add<NormalizeCastLoweringOpRegBase>(ctx);
}

void populateNormalizePreScalarCastingPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizetruncfExtfRegBase>(patterns.getContext());
}

void populateNormalizeFinalCastingPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizetruncfBf16RegBase>(ctx);
  patterns.add<NormalizeScalarExtensionRegBase<arith::ExtFOp>>(ctx);
  if (archIsRegbased)
    patterns.add<NormalizeScalarCastOpRegBase>(ctx);
}

void populateNormalizeSortPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeSortOpRegBase>(patterns.getContext());
}
} // namespace mlir::hfusion
