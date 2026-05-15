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

#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"
#include "bishengir/Transforms/Normalize/NormalizeCastingTemplate.h"

namespace mlir::hfusion {

/// linalg.(fill/brc) + hfusion.cast
/// is normalized to
/// (arith/hfusion).cast + linalg.(fill/brc)
/// in order to cast quickly
struct NormalizeBrcCast : public OpRewritePattern<hfusion::CastOp> {
  std::optional<Value> getCastedValue(PatternRewriter &rewriter, Location loc,
                                      Value cst, Type srcType, Type dstType,
                                      hfusion::RoundMode roundMode) const {
    auto srcElmTy = getElementTypeOrSelf(srcType);
    auto dstElmTy = getElementTypeOrSelf(dstType);

    hfusion::RoundMode defaultRounding =
        utils::selectRoundMode<hfusion::RoundMode>(srcElmTy, dstElmTy);
    bool scalarSrc = !isa<ShapedType>(cst.getType());
    // only scalar cast has default round mode (e.g arith.sitofp -> <trunc>)
    // do not use scalar castTo when round modes mismatch
    if (!(defaultRounding == roundMode) && scalarSrc)
      return std::nullopt;

    return hfusion::castTo(rewriter, cst, dstElmTy, roundMode);
  }

public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    if (!castOp.hasPureTensorSemantics()) {
      return failure();
    }

    Value src = castOp.getDpsInputs()[0];
    if (isa<BlockArgument>(src))
      return failure();

    Operation *defOp = src.getDefiningOp();
    if (!isa<linalg::FillOp>(defOp) && !isa<linalg::BroadcastOp>(defOp))
      return failure();

    auto srcTy = src.getType();
    auto dstTy = dyn_cast<TensorType>(castOp.getOutputs()[0].getType());
    // Disable conversion from brc f16 + cast i8/bool as
    // combined with NormalizeToTargetType pass causes infinite loop
    if (isa<linalg::BroadcastOp>(defOp) && isF16ElemType(srcTy) &&
        (isI1ElemType(dstTy) || isI8ElemType(dstTy))) {
      return failure();
    }

    auto roundMode = castOp.getRoundMode();
    Location loc = castOp.getLoc();

    Value cst = isa<linalg::FillOp>(defOp)
                    ? dyn_cast<linalg::FillOp>(defOp).getInputs()[0]
                    : dyn_cast<linalg::BroadcastOp>(defOp).getInput();

    auto castedVal =
        getCastedValue(rewriter, loc, cst, srcTy, dstTy, roundMode);
    if (!castedVal.has_value())
      return rewriter.notifyMatchFailure(
          castOp, "either round mode or datatype is not supported!");
    Value emptyTensor =
        utils::createEmptyOp(rewriter, loc, castOp.getOutputs()[0]);
    auto *newFillOrBrcOp =
        isa<linalg::FillOp>(defOp)
            ? rewriter.create<linalg::FillOp>(loc, *castedVal, emptyTensor)
            : rewriter.create<linalg::BroadcastOp>(
                  loc, *castedVal, emptyTensor,
                  dyn_cast<linalg::BroadcastOp>(defOp).getDimensionsAttr());

    rewriter.replaceAllUsesWith(castOp.getResults(),
                                newFillOrBrcOp->getResults());
    rewriter.eraseOp(castOp);

    return success();
  }
};

/// convert scalar to point tensor + hfusion.cast + linalg.broadcast
/// on unsupported round modes to optimize linalg.fill + hfusion.cast
struct NormalizefillCastToTensorBrc : public OpRewritePattern<hfusion::CastOp> {
  std::optional<Value>
  getPointTensorCastedValue(PatternRewriter &rewriter, Location loc, Value cst,
                            Type srcType, Type dstType,
                            hfusion::RoundMode roundMode) const {
    auto srcElmTy = getElementTypeOrSelf(srcType);
    auto dstElmTy = getElementTypeOrSelf(dstType);

    hfusion::RoundMode defaultRounding =
        utils::selectRoundMode<hfusion::RoundMode>(srcElmTy, dstElmTy);
    bool scalarSrc = !isa<ShapedType>(cst.getType());
    if ((defaultRounding == roundMode) || !scalarSrc)
      return std::nullopt;

    auto pointSrcTensorType = RankedTensorType::get({}, cst.getType());
    Value pointSrcTensor =
        utils::createStaticShapeEmptyOp(rewriter, loc, pointSrcTensorType);
    auto newFillOp = rewriter.create<linalg::FillOp>(loc, cst, pointSrcTensor);

    return hfusion::castTo(rewriter, newFillOp->getResult(0), dstElmTy,
                           roundMode);
  }

public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    if (!castOp.hasPureTensorSemantics()) {
      return failure();
    }

    Value src = castOp.getDpsInputs()[0];
    if (isa<BlockArgument>(src))
      return failure();

    Operation *defOp = src.getDefiningOp();
    if (!isa<linalg::FillOp>(defOp))
      return failure();
    auto fillOp = dyn_cast<linalg::FillOp>(defOp);
    auto srcTy = src.getType();
    auto dstTy = dyn_cast<TensorType>(castOp.getOutputs()[0].getType());
    if (dstTy.getRank() == 0)
      return failure();

    auto roundMode = castOp.getRoundMode();
    Location loc = castOp.getLoc();

    Value cst = fillOp.getInputs()[0];

    auto castedVal =
        getPointTensorCastedValue(rewriter, loc, cst, srcTy, dstTy, roundMode);
    if (!castedVal.has_value())
      return rewriter.notifyMatchFailure(
          castOp, "either round mode or datatype is not supported!");
    Value emptyTensor =
        utils::createEmptyOp(rewriter, loc, castOp.getOutputs()[0]);
    SmallVector<int64_t> dim;
    for (int64_t i = 0; i < dstTy.getRank(); ++i)
      dim.push_back(i);

    auto brcOp =
        rewriter.create<linalg::BroadcastOp>(loc, *castedVal, emptyTensor, dim);

    rewriter.replaceAllUsesWith(castOp.getResults(), brcOp->getResults());
    rewriter.eraseOp(castOp);

    return success();
  }
};

struct NormalizetruncfExtf : public OpRewritePattern<arith::ExtFOp> {
public:
  using OpRewritePattern<arith::ExtFOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtFOp extOp,
                                PatternRewriter &rewriter) const override {
    auto src = extOp.getIn();
    if (isa<BlockArgument>(src))
      return failure();
    auto defOp = src.getDefiningOp<arith::TruncFOp>();
    if (!defOp)
      return failure();
    if (defOp.getIn().getType() != extOp.getOut().getType())
      return failure();
    rewriter.replaceAllUsesWith(extOp.getOut(), defOp.getIn());
    return success();
  }
};

// example:
// arith.truncf %arg0 : f32 to bf16
// is normalized to
// %c0 = arith.constant 0 : index
// %from_elements = tensor.from_elements %arg0 : tensor<1xf32>
// %0 = tensor.empty() : tensor<1xbf16>
// %1 = hfusion.cast ins(%from_elements : tensor<1xf32>) outs(%0 :
// tensor<1xbf16>) -> tensor<1xbf16> %extracted = tensor.extract %1[%c0] :
// tensor<1xbf16> for there is no implementation for f32 to bf16 scalar truncf
struct NormalizetruncfBf16 : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const final {
    auto src = op.getIn();
    auto dst = op.getOut();
    auto srcType = src.getType();
    auto dstType = dst.getType();

    if (!srcType.isF32() || !dstType.isBF16()) {
      return failure();
    }

    if (isa<hfusion::CastOp>(op->getParentOp())) {
      return failure();
    }

    auto loc = op.getLoc();
    SmallVector<Value> extentOperands{src};
    auto tensorType = RankedTensorType::get({1}, srcType);
    Value fromElementsOp = rewriter.create<tensor::FromElementsOp>(
        loc, tensorType, extentOperands);
    assert(fromElementsOp.getDefiningOp() != nullptr);
    auto castOp = hfusion::castTo(
        rewriter, fromElementsOp.getDefiningOp()->getResult(0), dstType);

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices = {c0};
    auto extractOp = rewriter.create<tensor::ExtractOp>(
        loc, castOp.getDefiningOp()->getResult(0), indices);
    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

// example:
// arith.extf %arg0 : bf16 to f32
// is normalized to
// %c0 = arith.constant 0 : index
// %from_elements = tensor.from_elements %arg0 : tensor<1xbf16>
// %0 = tensor.empty() : tensor<1xf32>
// %1 = hfusion.cast ins(%from_elements : tensor<1xbf16>) outs(%0 :
// tensor<1xf32>) -> tensor<1xf32> %extracted = tensor.extract %1[%c0] :
// tensor<1xf32> for there is no implementation for bf16 to f32 scalar extf
template <typename CastOp>
struct NormalizeScalarExtension : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const final {
    auto src = op.getIn();
    auto dst = op.getOut();
    auto srcType = src.getType();
    auto dstType = dst.getType();
    if (mlir::isa<ShapedType>(srcType) || mlir::isa<ShapedType>(dstType)) {
      return failure();
    }
    if (isa<hfusion::CastOp>(op->getParentOp()) ||
        isa<linalg::MatmulOp>(op->getParentOp()) ||
        isa<linalg::BatchMatmulOp>(op->getParentOp())) {
      return failure();
    }

    auto loc = op.getLoc();
    SmallVector<Value> extentOperands{src};
    auto tensorType = RankedTensorType::get({1}, srcType);
    Value fromElementsOp = rewriter.create<tensor::FromElementsOp>(
        loc, tensorType, extentOperands);
    auto castOp = hfusion::castTo(
        rewriter, fromElementsOp.getDefiningOp()->getResult(0), dstType);

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices = {c0};
    auto extractOp = rewriter.create<tensor::ExtractOp>(
        loc, castOp.getDefiningOp()->getResult(0), indices);
    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

struct NormalizeAnyToF32UnaryRecOp
    : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    // currently, only applied to rec unary function
    if (op.getFun() != hfusion::UnaryFn::rec)
      return failure();

    Value inValue = op.getInputs()[0];
    Value outValue = op.getOutputs()[0];

    Type inType = getElementTypeOrSelf(inValue.getType());
    Type outType = getElementTypeOrSelf(outValue.getType());
    // currently, only need handle case where the input type is equal to output
    // type
    if (inType != outType)
      return failure();

    if (inType.isF32())
      return failure();

    Location loc = op->getLoc();

    // TODO: cast to more efficient data type
    auto castedInValue =
        hfusion::castTo(rewriter, inValue, rewriter.getF32Type());

    // create new elemwise_unary op
    auto resEmptyOp = utils::createEmptyOp(rewriter, loc, castedInValue);
    Operation *newOp =
        hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp, hfusion::UnaryFn,
                               hfusion::UnaryFnAttr>(
            rewriter, loc, hfusion::UnaryFn::rec, castedInValue, resEmptyOp);

    // TODO: cast to more efficient data type
    auto castedOutValue =
        hfusion::castTo(rewriter, newOp->getResult(0), outType);
    rewriter.replaceOp(op, castedOutValue);
    return success();
  }
};

struct HFusionNormalizeCastLoweringTraits : public hfusion::NormalizeTraitsBase {
  static constexpr bool supportsF16ToI8TruncOverflowPreprocess = true;

  static bool shouldNormalizeCast(hfusion::CastOp op) {
    const bool hasSaturateOverflowMode = mlir::hasSaturateOverflowModeAnnotation(op);
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

using NormalizeCastLoweringOp =
    NormalizeCastLoweringOpTemplate<hfusion::CastOp,
                                    HFusionNormalizeCastLoweringTraits>;

struct NormalizeScalarCastOp : public OpRewritePattern<hfusion::CastOp> {
public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    Value originalInput = castOp.getInputs()[0];
    Value originalOutput = castOp.getOutputs()[0];

    auto inputTensorType = dyn_cast<RankedTensorType>(originalInput.getType());

    auto outputTensorType =
        dyn_cast<RankedTensorType>(originalOutput.getType());
    if (!inputTensorType || !outputTensorType ||
        inputTensorType.getRank() != 0 || outputTensorType.getRank() != 0) {
      return failure();
    }

    Type elemType = inputTensorType.getElementType();
    Type outElemType = outputTensorType.getElementType();

    Location loc = castOp.getLoc();
    auto extractInput =
        rewriter.create<tensor::ExtractOp>(loc, originalInput, ValueRange{});
    SmallVector<Value> extentOperands{extractInput};
    RankedTensorType inputDimType = RankedTensorType::get({1}, elemType);
    Value fromElementsOp = rewriter.create<tensor::FromElementsOp>(
        loc, inputDimType, extentOperands);

    auto newCastOp = hfusion::castTo(rewriter, fromElementsOp, outElemType,
                                     castOp.getRoundMode(), std::nullopt,
                                     castOp.getEnableOverflow());
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices = {c0};
    auto extractOp =
        rewriter.create<tensor::ExtractOp>(loc, newCastOp, indices);
    Value newInsert = rewriter.create<tensor::InsertOp>(
        loc, extractOp.getResult(), castOp.getOutputs()[0], ValueRange{});

    rewriter.replaceOp(castOp, newInsert);
    return success();
  }
};

void populateNormalizeCastingPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeBrcCast>(ctx);
  patterns.add<NormalizefillCastToTensorBrc>(ctx);
  patterns.add<NormalizeAnyToF32UnaryRecOp>(ctx);
  patterns.add<NormalizeCastLoweringOp>(ctx);
}

void populateNormalizePreScalarCastingPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizetruncfExtf>(patterns.getContext());
}

void populateNormalizeFinalCastingPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizetruncfBf16>(ctx);
  patterns.add<NormalizeScalarExtension<arith::ExtFOp>>(ctx);
  if (archIsRegbased)
    patterns.add<NormalizeScalarCastOp>(ctx);
}
} // namespace mlir::hfusion
