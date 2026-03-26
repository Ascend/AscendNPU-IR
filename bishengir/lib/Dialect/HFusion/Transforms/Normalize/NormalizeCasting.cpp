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
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"

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

struct NormalizeCastLoweringOp : public OpRewritePattern<hfusion::CastOp> {
public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::CastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());
    int64_t srcBitWidth = inType.getIntOrFloatBitWidth();
    int64_t dstBitWidth = outType.getIntOrFloatBitWidth();
    auto castIntegerType = op.getCast();

    // Deal with trunc with overflow for wider-to-narrower-integer-of-non-i1.
    if (srcBitWidth > dstBitWidth && outType.isInteger() &&
        !outType.isInteger(1)) {
      auto overflowMode = getAnnotateOverflowMode(op);
      bool enableSaturate =
          overflowMode.has_value() && overflowMode->ends_with("saturate");
      if (!archIsRegbased) {
        if (enableSaturate) {
          auto overflowModeAttr =
              utils::getAnnotateOpWithAttr(op->getResult(0), "overflow_mode");
          if (!overflowModeAttr.has_value())
            return failure();
          annotation::MarkOp markOp =
              dyn_cast<annotation::MarkOp>(overflowModeAttr.value());
          rewriter.eraseOp(markOp);
          return handleSaturateOverFlowMode(op, rewriter);
        }
        return handleTruncOverFlowMode(op, rewriter);
      } else {
        if (enableSaturate) {
          auto overflowModeAttr =
              utils::getAnnotateOpWithAttr(op->getResult(0), "overflow_mode");
          if (!overflowModeAttr.has_value()) {
            return failure();
          }
          annotation::MarkOp markOp =
              dyn_cast<annotation::MarkOp>(overflowModeAttr.value());
          rewriter.eraseOp(markOp);
          return handleOverflowModeForSaturate(op, rewriter, enableSaturate);
        }
        return handleOverflowModeForTrunc(op, rewriter);
      }
    }

    const bool isI64ToF16 = inType.isInteger(64) && outType.isF16();
    const bool isIntegerToBF16 =
        (inType.isInteger(64) || inType.isInteger(32) || inType.isInteger(16) ||
         inType.isInteger(8)) &&
        outType.isBF16();
    const bool isU16ToF16 = inType.isInteger(16) && outType.isF16() &&
                            hfusion::TypeFn::cast_unsigned == castIntegerType;
    if (isI64ToF16 || isIntegerToBF16 || isU16ToF16) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to f16) " << "\n ");
      Value castResult;
      // I8ToBF16: I8ToF16 -> F16ToBF16 (regbase)
      // I8ToBF16: I8ToF16 -> F16ToF32 -> F32ToBF16 (membase)
      if (false && archIsRegbased && isIntegerToBF16 && inType.isInteger(8)) {
        // fixme: arith dialect has no fp-to-bf16 conversion. Need to extend
        // hfusion op first.
        castResult =
            castSrcToFp16ToTargetType(op, rewriter.getBF16Type(), rewriter);
      } else {
        castResult = castInToF32ToOut(op, rewriter);
      }
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isU32ToF32 = inType.isInteger(32) && outType.isF32() &&
                            hfusion::TypeFn::cast_unsigned == castIntegerType;
    if (isU32ToF32) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to I64 to " << outType << ")\n");
      Value castResult = castU32ToI64ToF32(op, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isU32ToF16 = inType.isInteger(32) && outType.isF16() &&
                            hfusion::TypeFn::cast_unsigned == castIntegerType;
    const bool isU32ToBF16 = inType.isInteger(32) && outType.isBF16() &&
                             hfusion::TypeFn::cast_unsigned == castIntegerType;
    if (isU32ToF16 || isU32ToBF16) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to I64 to F32 to " << outType << ")\n");
      Type targetType = getElementTypeOrSelf(outType);
      Value castResult = castU32ToI64ToF32ToOut(op, targetType, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isI8ToI64 = inType.isInteger(8) && outType.isInteger(64);
    if (isI8ToI64) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to f16 to f32 to " << outType << ")\n");
      Value castResult = castI8ToI64(op, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isI8ToF32 = inType.isInteger(8) && outType.isF32();
    const bool isI8ToI32 = inType.isInteger(8) && outType.isInteger(32);
    const bool isI8ToI16 = inType.isInteger(8) && outType.isInteger(16);
    const bool isI1ToI16 = inType.isInteger(1) && outType.isInteger(16);
    const bool isI1ToF32 = inType.isInteger(1) && outType.isF32();

    if (isI8ToF32 || isI1ToI16 ||
        (!archIsRegbased && (isI8ToI32 || isI8ToI16 || isI1ToF32))) {
      Type targetType = getElementTypeOrSelf(outType);
      Value castResult = castSrcToFp16ToTargetType(op, targetType, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isI1ToI32 = inType.isInteger(1) && outType.isInteger(32);
    if (!archIsRegbased && isI1ToI32) {
      Value inValue = op.getInputs()[0];
      Value castF16Value = hfusion::castTo(
          rewriter, inValue, rewriter.getF16Type(), hfusion::RoundMode::RINT);
      Value castI32Value =
          hfusion::castTo(rewriter, castF16Value, rewriter.getI32Type(),
                          hfusion::RoundMode::RINT);
      rewriter.replaceOp(op, castI32Value);
      return success();
    }

    const bool isI1ToI64 = inType.isInteger(1) && outType.isInteger(64);
    if (isI1ToI64) {
      Value inValue = op.getInputs()[0];
      Value castF32Value = hfusion::castTo(
          rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);

      Value castI64Value =
          hfusion::castTo(rewriter, castF32Value, rewriter.getI64Type());
      rewriter.replaceOp(op, castI64Value);
      return success();
    }

    const bool isI32ToF16 = inType.isInteger(32) && outType.isF16();
    if (isI32ToF16) {
      Value inValue = op.getInputs()[0];
      Value castF32Value =
          hfusion::castTo(rewriter, inValue, rewriter.getF32Type());

      Value castF16Value =
          hfusion::castTo(rewriter, castF32Value, rewriter.getF16Type());
      rewriter.replaceOp(op, castF16Value);
      return success();
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
      Value castResult = castSrcTypeToI1ByVCmp(op, inType, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    // I16ToI64: I16ToF32 -> F32ToI64 (membase)
    // I16ToI64: I16ToI32 -> I32ToI64 (regbase)
    const bool isI16ToI64 = inType.isInteger(16) && outType.isInteger(64);
    if (isI16ToI64) {
      Value inValue = op.getInputs()[0];
      Value castValue;
      if (archIsRegbased) {
        castValue = hfusion::castTo(rewriter, inValue, rewriter.getI32Type(),
                                    castIntegerType);
      } else {
        castValue = hfusion::castTo(rewriter, inValue, rewriter.getF32Type(),
                                    hfusion::RoundMode::RINT);
      }

      Value castI64Value = hfusion::castTo(
          rewriter, castValue, rewriter.getI64Type(), castIntegerType);
      rewriter.replaceOp(op, castI64Value);
      return success();
    }

    // I16ToI32: I16ToF32 -> F32ToI32 (membase)
    // I16ToI32: OK (regbase)
    const bool isI16ToI32 = inType.isInteger(16) && outType.isInteger(32);
    if (!archIsRegbased && isI16ToI32) {
      Value inValue = op.getInputs()[0];
      Value castF32Value = hfusion::castTo(
          rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);

      Value castI32Value =
          hfusion::castTo(rewriter, castF32Value, rewriter.getI32Type());
      rewriter.replaceOp(op, castI32Value);
      return success();
    }

    // AnyToF8: AnyToF32 -> F32ToF8
    const bool isAnyToF8 = (!inType.isF32()) &&
                           (outType.isFloat8E4M3FN() || outType.isFloat8E5M2());
    if (isAnyToF8) {
      Value castResult = castInToF32ToOut(op, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    // F8ToAny: F8ToF32 -> F32ToAny
    const bool isF8ToAny = (inType.isFloat8E4M3FN() || inType.isFloat8E5M2()) &&
                           (!outType.isF32());
    if (isF8ToAny) {
      Value castResult = castInToF32ToOut(op, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isF32ToU32 = inType.isF32() && outType.isInteger(32) &&
                            TypeFn::cast_unsigned == castIntegerType;
    if (isF32ToU32) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to I64 to " << outType << ")\n");
      // F32 -> I64
      Value castF32ToI64 =
          hfusion::castTo(rewriter, op.getDpsInputOperand(0)->get(),
                          rewriter.getI64Type(), TypeFn::cast_signed);
      // I64 -> U32
      Value castI64ToU32 = hfusion::castTo(
          rewriter, castF32ToI64, rewriter.getI32Type(), TypeFn::cast_signed);
      rewriter.replaceOp(op, castI64ToU32);
      return success();
    }

    return failure();
  }
};

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
