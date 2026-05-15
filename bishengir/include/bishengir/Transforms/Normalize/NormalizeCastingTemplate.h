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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

/// Rewrites `fill(x)` or `broadcast(x)` followed by `cast` as:
///   cast(fill(x)) = fill(cast(x))
///   cast(broadcast(x)) = broadcast(cast(x))
///
/// This moves the cast onto the scalar or lower-rank source so the tensor-wide
/// cast is replaced by one smaller cast plus the original fill/broadcast.
/// Example:
///   `cast(fill(%cst)) -> fill(cast(%cst))`.
template <typename CastOpType, typename Traits>
struct NormalizeBrcCastTemplate : public OpRewritePattern<CastOpType> {
public:
  using OpRewritePattern<CastOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOpType castOp,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeBrcCast(castOp))
      return failure();

    Value src = castOp.getDpsInputs()[0];
    if (isa<BlockArgument>(src))
      return failure();

    Operation *defOp = src.getDefiningOp();
    if (!Traits::matchFillOp(defOp) && !Traits::matchBroadcastOp(defOp))
      return failure();

    auto srcTy = src.getType();
    auto dstTy = dyn_cast<TensorType>(castOp.getDpsInits()[0].getType());
    if (!dstTy)
      return failure();

    if (Traits::shouldSkipBrcCast(castOp, defOp, srcTy, dstTy))
      return failure();

    Value input = Traits::matchFillOp(defOp) ? Traits::getFillInput(defOp)
                                             : Traits::getBroadcastInput(defOp);
    auto defaultRoundMode =
        utils::selectRoundMode<decltype(castOp.getRoundMode())>(
            getElementTypeOrSelf(srcTy), getElementTypeOrSelf(dstTy));
    if (defaultRoundMode != castOp.getRoundMode() &&
        !isa<ShapedType>(input.getType()))
      return rewriter.notifyMatchFailure(
          castOp, "either round mode or datatype is not supported!");
    Value castedVal = Traits::createCastOp(rewriter, castOp.getLoc(), input,
                                           getElementTypeOrSelf(dstTy),
                                           castOp.getRoundMode());

    Value emptyTensor =
        utils::createEmptyOp(rewriter, castOp.getLoc(), castOp.getDpsInits()[0]);
    Value result = Traits::matchFillOp(defOp)
                       ? Traits::createFillOp(rewriter, castOp.getLoc(),
                                              castedVal, emptyTensor)
                       : Traits::createBroadcastOp(
                             rewriter, castOp.getLoc(), castedVal, emptyTensor,
                             Traits::getBroadcastDims(defOp));
    rewriter.replaceOp(castOp, result);
    return success();
  }
};

/// Rewrites `cast(fill(x))` on unsupported scalar round modes as:
///   cast(fill(x)) = broadcast(cast([x]))
/// where `[x]` is a point tensor created from the scalar `x`.
///
/// This preserves the requested round mode by materializing a supported tensor
/// cast first and then broadcasting the one-element result. Example:
///   `cast(fill(%cst)) -> broadcast(cast(fill(%cst : tensor<()>)))`.
template <typename CastOpType, typename Traits>
struct NormalizeFillCastToTensorBrcTemplate
    : public OpRewritePattern<CastOpType> {
public:
  using OpRewritePattern<CastOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOpType castOp,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeFillCastToTensorBrc(castOp))
      return failure();

    Value src = castOp.getDpsInputs()[0];
    if (isa<BlockArgument>(src))
      return failure();

    Operation *defOp = src.getDefiningOp();
    if (!Traits::matchFillOp(defOp))
      return failure();

    auto dstTy = dyn_cast<TensorType>(castOp.getDpsInits()[0].getType());
    if (!dstTy || dstTy.getRank() == 0)
      return failure();

    Value input = Traits::getFillInput(defOp);
    auto defaultRoundMode =
        utils::selectRoundMode<decltype(castOp.getRoundMode())>(
            getElementTypeOrSelf(src.getType()), getElementTypeOrSelf(dstTy));
    if (defaultRoundMode == castOp.getRoundMode() ||
        isa<ShapedType>(input.getType()))
      return rewriter.notifyMatchFailure(
          castOp, "either round mode or datatype is not supported!");
    auto pointSrcTensorType = RankedTensorType::get({}, input.getType());
    Value pointSrcTensor =
        utils::createStaticShapeEmptyOp(rewriter, castOp.getLoc(),
                                        pointSrcTensorType);
    Value pointTensor =
        Traits::createFillOp(rewriter, castOp.getLoc(), input, pointSrcTensor);
    Value castedVal = Traits::createCastOp(rewriter, castOp.getLoc(),
                                           pointTensor,
                                           getElementTypeOrSelf(dstTy),
                                           castOp.getRoundMode());

    Value emptyTensor =
        utils::createEmptyOp(rewriter, castOp.getLoc(), castOp.getDpsInits()[0]);
    SmallVector<int64_t> dims;
    for (int64_t i = 0; i < dstTy.getRank(); ++i)
      dims.push_back(i);

    Value result = Traits::createBroadcastOp(rewriter, castOp.getLoc(),
                                             castedVal, emptyTensor, dims);
    rewriter.replaceOp(castOp, result);
    return success();
  }
};

/// Eliminates a canceling truncate/extend pair:
///   extf(truncf(x)) = x
///
/// Example:
///   `arith.extf (arith.truncf %x : f32 to f16) : f16 to f32`
/// becomes `%x` when the outer result type matches the original input type.
template <typename ExtFOpType, typename Traits>
struct NormalizeTruncfExtfTemplate : public OpRewritePattern<ExtFOpType> {
public:
  using OpRewritePattern<ExtFOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtFOpType extOp,
                                PatternRewriter &rewriter) const override {
    auto src = extOp.getIn();
    if (isa<BlockArgument>(src))
      return failure();
    auto defOp = src.template getDefiningOp<arith::TruncFOp>();
    if (!defOp)
      return failure();
    if (defOp.getIn().getType() != extOp.getOut().getType())
      return failure();
    rewriter.replaceAllUsesWith(extOp.getOut(), defOp.getIn());
    return success();
  }
};

/// Rewrites scalar `truncf` to bf16 through a length-1 tensor cast:
///   truncf(x : f32 -> bf16) = extract(cast(from_elements(x)))
///
/// This is used when the dialect supports the tensor cast path but not the
/// direct scalar `f32 -> bf16` truncation. Example:
///   `arith.truncf %x : f32 to bf16`
/// becomes `tensor.extract(cast(tensor.from_elements %x))`.
template <typename TruncFOpType, typename Traits>
struct NormalizeTruncfBf16Template : public OpRewritePattern<TruncFOpType> {
  using OpRewritePattern<TruncFOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(TruncFOpType op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getIn();
    Value dst = op.getOut();
    Type srcType = src.getType();
    Type dstType = dst.getType();

    if (!srcType.isF32() || !dstType.isBF16())
      return failure();
    if (Traits::isInsideDialectCast(*op))
      return failure();

    Value result = Traits::castScalarThroughTensor(rewriter, op.getLoc(), src,
                                                   dstType);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrites scalar extension-like ops through a length-1 tensor cast:
///   ext(x : Ts -> Td) = extract(cast(from_elements(x)))
///
/// This preserves scalar semantics by reusing the dialect tensor cast path.
/// Example:
///   `arith.extf %x : bf16 to f32`
/// becomes `tensor.extract(cast(tensor.from_elements %x))`.
template <typename ScalarCastOpType, typename Traits>
struct NormalizeScalarExtensionTemplate
    : public OpRewritePattern<ScalarCastOpType> {
  using OpRewritePattern<ScalarCastOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScalarCastOpType op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getIn();
    Value dst = op.getOut();
    Type srcType = src.getType();
    Type dstType = dst.getType();
    if (isa<ShapedType>(srcType) || isa<ShapedType>(dstType))
      return failure();
    if (Traits::shouldSkipScalarExtension(op))
      return failure();

    Value result = Traits::castScalarThroughTensor(rewriter, op.getLoc(), src,
                                                   dstType);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrites reciprocal on non-f32 tensors as:
///   rec(x : T) = cast(rec(cast(x -> f32)) -> T)
///
/// This computes the reciprocal in f32 and casts back so one implementation
/// can cover lower-precision element types. Example:
///   `rec(%x : tensor<...xf16>)`
/// becomes `cast(rec(cast(%x to tensor<...xf32>)) to tensor<...xf16>)`.
template <typename UnaryOpType, typename Traits>
struct NormalizeAnyToF32UnaryRecOpTemplate
    : public OpRewritePattern<UnaryOpType> {
public:
  using OpRewritePattern<UnaryOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnaryOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeAnyToF32UnaryRec(op))
      return failure();

    Value inValue = op.getDpsInputs()[0];
    Value outValue = op.getDpsInits()[0];
    Type inType = getElementTypeOrSelf(inValue.getType());
    Type outType = getElementTypeOrSelf(outValue.getType());
    if (inType != outType || inType.isF32())
      return failure();

    Location loc = op->getLoc();
    Value castedInValue = Traits::createCastOp(
        rewriter, loc, inValue, rewriter.getF32Type(), std::nullopt);
    Value resEmptyOp = utils::createEmptyOp(rewriter, loc, castedInValue);
    Value newValue = Traits::createUnaryOp(rewriter, loc, castedInValue,
                                           resEmptyOp, UnaryKind::Rec);
    Value result =
        Traits::createCastOp(rewriter, loc, newValue, outType, std::nullopt);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrites a rank-0 tensor cast through scalar extract and insert:
///   cast(tensor<x>) = insert(cast(extract(tensor<x>)))
///
/// This reduces a scalar tensor cast to an ordinary scalar cast wrapped by
/// rank-0 tensor glue. Example:
///   `cast(%arg0 : tensor<f32> -> tensor<bf16>)`
/// becomes `tensor.insert(cast(tensor.extract %arg0))`.
template <typename CastOpType, typename Traits>
struct NormalizeScalarCastOpTemplate : public OpRewritePattern<CastOpType> {
public:
  using OpRewritePattern<CastOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOpType castOp,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeScalarCast(castOp))
      return failure();

    Value originalInput = castOp.getDpsInputs()[0];
    Value originalOutput = castOp.getDpsInits()[0];
    auto inputTensorType = dyn_cast<RankedTensorType>(originalInput.getType());
    auto outputTensorType = dyn_cast<RankedTensorType>(originalOutput.getType());
    if (!inputTensorType || !outputTensorType ||
        inputTensorType.getRank() != 0 || outputTensorType.getRank() != 0)
      return failure();

    Location loc = castOp.getLoc();
    Value scalarInput =
        rewriter.create<tensor::ExtractOp>(loc, originalInput, ValueRange{});
    Value castedScalar = Traits::castScalarFromRankZeroTensor(
        rewriter, loc, castOp, scalarInput, outputTensorType.getElementType());
    Value result = rewriter.create<tensor::InsertOp>(
        loc, castedScalar, originalOutput, ValueRange{});

    rewriter.replaceOp(castOp, result);
    return success();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZECASTINGTEMPLATE_H
