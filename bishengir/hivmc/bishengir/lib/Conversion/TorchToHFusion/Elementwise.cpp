//===- Elementwise.cpp - Conversion impl. for Element-wise Ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TorchToHFusion/PopulatePatterns.h"
#include "bishengir/Conversion/TorchToHFusion/Rewrite.h"
#include "bishengir/Conversion/TorchToHFusion/TorchToHFusion.h"
#include "bishengir/Conversion/TorchToHFusion/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/DebugStringHelper.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/APSInt.h"
#include <iostream>
#include <numeric>
#include <set>
#include <type_traits>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
static LogicalResult
normalizeElewiseOperands(ConversionPatternRewriter &b,
                         const TypeConverter *converter, Operation *op,
                         const SmallVector<Value> &operands,
                         SmallVector<Value> &res, Value &emptyOp) {
  auto loc = op->getLoc();

  auto tensorOperands = llvm::to_vector<6>(llvm::make_filter_range(
      operands, [](Value v) { return isa<RankedTensorType>(v.getType()); }));

  // Get result shape.
  SmallVector<Value> resultShape;
  getElementwiseResultShape(b, loc, tensorOperands, resultShape);

  auto resultType = cast<RankedTensorType>(
      converter->convertType(op->getResult(0).getType()));

  for (auto operand : operands) {
    // If operand is scalar, convert to tensor.
    auto scalarTensor = convertScalarToTensor(b, loc, operand, resultShape,
                                              resultType.getElementType());

    auto maybeCastTensor =
        createHFusionCastOp(b, loc, resultType.getElementType(), scalarTensor);
    if (failed(maybeCastTensor))
      return op->emitError("cast type faield.");
    Value castTensor = *maybeCastTensor;
    // If shape is not match, try broad cast.
    auto broadcast = broadcastTensorToShape(b, loc, castTensor, resultType);
    if (failed(broadcast)) {
      op->emitWarning() << "Try broad cast type " << castTensor.getType()
                        << " to " << castTensor << "failed.";
      return failure();
    }

    res.emplace_back(*broadcast);
  }

  emptyOp = b.create<tensor::EmptyOp>(loc, getAsOpFoldResult(resultShape),
                                      resultType.getElementType());

  return success();
}

// TODO : Remove this wrapper when we support lowering float to f32
Type f32Wrapper(Type srcType) {
  if (isa<mlir::FloatType>(srcType) && srcType.isF64()) {
    return mlir::FloatType::getF32(srcType.getContext());
  }
  return srcType;
}

LogicalResult normalizeElewiseOperandsHFusionCompare(
    ConversionPatternRewriter &b, const TypeConverter *converter, Operation *op,
    const SmallVector<Value> &operands, SmallVector<Value> &res,
    Value &emptyOp) {
  auto loc = op->getLoc();

  auto tensorOperands = llvm::to_vector<6>(llvm::make_filter_range(
      operands, [](Value v) { return isa<RankedTensorType>(v.getType()); }));
  // Get result shape.
  SmallVector<Value> resultShape;
  getElementwiseResultShape(b, loc, tensorOperands, resultShape);
  auto resultType = cast<RankedTensorType>(
      converter->convertType(op->getResult(0).getType()));
  auto lType =
      cast<RankedTensorType>(converter->convertType(operands[0].getType()));
  Type rType;
  if (isa<TensorType>(operands[1].getType())) {
    rType =
        cast<RankedTensorType>(converter->convertType(operands[1].getType()))
            .getElementType();
  } else {
    rType = operands[1].getType();
  }
  auto maybeCastType = getPromotionType(op, lType.getElementType(), rType);
  if (failed(maybeCastType)) {
    return failure();
  }
  auto castType = f32Wrapper(*maybeCastType);
  for (auto operand : operands) {
    // If operand is scalar, convert to tensor.
    auto scalarTensor =
        convertScalarToTensor(b, loc, operand, resultShape, castType);
    auto convertTypeTensor =
        createHFusionCastOp(b, loc, castType, scalarTensor);
    if (failed(convertTypeTensor)) {
      return failure();
    }
    res.emplace_back(*convertTypeTensor);
  }
  emptyOp = b.create<tensor::EmptyOp>(loc, getAsOpFoldResult(resultShape),
                                      resultType.getElementType());

  return success();
}

template <typename AtenOpT, linalg::BinaryFn linalgFn>
class ConvertAtenToAddSub : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify CompatibleTypes.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto converter = OpConversionPattern<AtenOpT>::getTypeConverter();
    auto loc = op->getLoc();

    SmallVector<Value> normalizedOperands;
    Value emptyOp;
    if (failed(normalizeElewiseOperands(rewriter, converter, op,
                                        adaptor.getOperands(),
                                        normalizedOperands, emptyOp)))
      return failure();

    assert((normalizedOperands.size() == 3) &&
           "For add/sub, operands must be 3.");

    // Add/Sub LHS, RHS, ALPHA = LHS Add/Sub RHS * ALPHA
    auto mulOp = createLinalgBinary<linalg::BinaryFn::mul>(
        rewriter, loc, normalizedOperands[1] /*RHS */,
        normalizedOperands[2] /*ALPHA*/, emptyOp);

    auto newOp = createLinalgBinary<linalgFn>(
        rewriter, loc, normalizedOperands[0], mulOp, emptyOp);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

template <typename AtenOpT, linalg::BinaryFn linalgSignedFn,
          linalg::BinaryFn linalgUnSignedFn = linalgSignedFn>
class ConvertAtenToLinalgBinary : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify CompatibleTypes.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto converter = OpConversionPattern<AtenOpT>::getTypeConverter();
    auto loc = op->getLoc();

    SmallVector<Value> normalizedOperands;
    Value emptyOp;
    if (failed(normalizeElewiseOperands(rewriter, converter, op,
                                        adaptor.getOperands(),
                                        normalizedOperands, emptyOp)))
      return failure();

    assert((normalizedOperands.size() == 2) &&
           "For binary, operands must be 2.");

    auto resultElementType = getElementTypeOrSelf(emptyOp);
    if (resultElementType.isUnsignedInteger()) {
      auto newOp = createLinalgBinary<linalgUnSignedFn>(
          rewriter, loc, normalizedOperands[0], normalizedOperands[1], emptyOp);
      rewriter.replaceOp(op, newOp);
    } else {
      auto newOp = createLinalgBinary<linalgSignedFn>(
          rewriter, loc, normalizedOperands[0], normalizedOperands[1], emptyOp);
      rewriter.replaceOp(op, newOp);
    }

    return success();
  }
};

template <typename AtenOpT, hfusion::BinaryFn hfusionFn>
class ConvertAtenToHfusionBinary : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify CompatibleTypes.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto converter = OpConversionPattern<AtenOpT>::getTypeConverter();
    auto loc = op->getLoc();

    SmallVector<Value> normalizedOperands;
    Value emptyOp;
    if (failed(normalizeElewiseOperands(rewriter, converter, op,
                                        adaptor.getOperands(),
                                        normalizedOperands, emptyOp)))
      return failure();

    assert((normalizedOperands.size() == 2) &&
           "For binary, operands must be 2.");
    auto newOp = createHFusionBinary<hfusionFn>(
        rewriter, loc, normalizedOperands[0], normalizedOperands[1], emptyOp);
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

template <typename AtenOpT, linalg::UnaryFn linalgFn>
class ConvertAtenToLinalgUnary : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify CompatibleTypes.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto converter = OpConversionPattern<AtenOpT>::getTypeConverter();
    auto loc = op->getLoc();

    SmallVector<Value> normalizedOperands;
    Value emptyOp;
    if (failed(normalizeElewiseOperands(rewriter, converter, op,
                                        adaptor.getOperands(),
                                        normalizedOperands, emptyOp)))
      return failure();

    assert((normalizedOperands.size() == 1) &&
           "For unary, operands must be 1.");
    auto newOp = createLinalgUnary<linalgFn>(rewriter, loc,
                                             normalizedOperands[0], emptyOp);
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

template <typename AtenOpT, hfusion::UnaryFn hfusionFn>
class ConvertAtenToHfusionUnary : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify CompatibleTypes.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto converter = OpConversionPattern<AtenOpT>::getTypeConverter();
    auto loc = op->getLoc();

    SmallVector<Value> normalizedOperands;
    Value emptyOp;
    if (failed(normalizeElewiseOperands(rewriter, converter, op,
                                        adaptor.getOperands(),
                                        normalizedOperands, emptyOp)))
      return failure();

    assert((normalizedOperands.size() == 1) &&
           "For unary, operands must be 1.");
    auto newOp = createHFusionUnary<hfusionFn>(rewriter, loc,
                                               normalizedOperands[0], emptyOp);
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

template <typename AtenOpT, hfusion::CompareFn CompareFnTy>
class ConvertAtenToHFusionCompare : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify CompatibleTypes.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    auto converter = OpConversionPattern<AtenOpT>::getTypeConverter();
    auto loc = op->getLoc();
    SmallVector<Value> normalizedOperands;
    Value emptyOp;
    if (failed(normalizeElewiseOperandsHFusionCompare(
            rewriter, converter, op, adaptor.getOperands(), normalizedOperands,
            emptyOp)))
      return failure();
    assert((normalizedOperands.size() == 2) &&
           "For binary, operands must be 2.");
    auto newOp = createHFusionCompare<CompareFnTy>(
        rewriter, loc, normalizedOperands[0], normalizedOperands[1], emptyOp);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

class ConvertAtenToSigmoid : public OpConversionPattern<AtenSigmoidOp> {
public:
  using OpConversionPattern<AtenSigmoidOp>::OpConversionPattern;
  using OpAdaptor = typename AtenSigmoidOp::Adaptor;
  LogicalResult
  matchAndRewrite(AtenSigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify CompatibleTypes.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto converter = OpConversionPattern<AtenSigmoidOp>::getTypeConverter();
    auto loc = op->getLoc();

    SmallVector<Value> normalizedOperands;
    Value emptyOp;
    if (failed(normalizeElewiseOperands(rewriter, converter, op,
                                        adaptor.getOperands(),
                                        normalizedOperands, emptyOp)))
      return failure();

    assert((normalizedOperands.size() == 1) &&
           "For sigmoid, operands must be 1.");

    Type outTy = cast<RankedTensorType>(
                     converter->convertType(op->getResult(0).getType()))
                     .getElementType();
    // FIXME: If input's tensor type is int, should convert it to float.

    // sigmoid(x) = 1 / (e^-x + 1)
    Value arg = normalizedOperands[0];
    auto negate =
        createLinalgUnary<linalg::UnaryFn::negf>(rewriter, loc, arg, emptyOp);
    auto exp =
        createLinalgUnary<linalg::UnaryFn::exp>(rewriter, loc, negate, emptyOp);
    auto one =
        rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(outTy, 1.0));
    auto added = createLinalgBinary<linalg::BinaryFn::add>(rewriter, loc, exp,
                                                           one, emptyOp);
    auto result = createLinalgBinary<linalg::BinaryFn::div>(rewriter, loc, one,
                                                            added, emptyOp);

    rewriter.replaceOp(op, result);
    return success();
  }
};

class ConvertAtenToWhere : public OpConversionPattern<AtenWhereSelfOp> {
public:
  using OpConversionPattern<AtenWhereSelfOp>::OpConversionPattern;
  using OpAdaptor = typename AtenWhereSelfOp::Adaptor;
  LogicalResult
  matchAndRewrite(AtenWhereSelfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify CompatibleTypes.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto converter = OpConversionPattern<AtenWhereSelfOp>::getTypeConverter();

    auto allOperands = adaptor.getOperands();
    SmallVector<Value> oprandsToBroadcast = {allOperands[1], allOperands[2]};

    SmallVector<Value> normalizedOperands;

    Value emptyOp;
    if (failed(normalizeElewiseOperands(rewriter, converter, op,
                                        oprandsToBroadcast, normalizedOperands,
                                        emptyOp)))
      return failure();

    Value args1 = normalizedOperands[0];
    Value args2 = normalizedOperands[1];

    rewriter.replaceOpWithNewOp<hfusion::SelectOp>(
        op, ValueRange{allOperands[0], args1, args2}, ValueRange{emptyOp});
    return success();
  }
};

static LogicalResult checkClampParamsValid(AtenClampOp op,
                                           const TypeConverter *converter,
                                           Value min, Value max) {
  if (isa<Torch::OptionalType>(min.getType()) ||
      isa<Torch::OptionalType>(max.getType())) {
    return op.emitError("unimplemented: runtime optional type in clamp");
  }
  Type dtype = cast<RankedTensorType>(converter->convertType(op.getType()))
                   .getElementType();
  if (!dtype.isIntOrFloat()) {
    return op.emitError("unimplement type for clamp");
  }
  if (isa<Torch::NoneType>(min.getType()) &&
      isa<Torch::NoneType>(max.getType())) {
    return op.emitError("In clamp, one of min and max must be not none");
  }
  return success();
}

class ConvertAtenToClamp : public OpConversionPattern<AtenClampOp> {
public:
  using OpConversionPattern<AtenClampOp>::OpConversionPattern;
  using OpAdaptor = typename AtenClampOp::Adaptor;
  LogicalResult
  matchAndRewrite(AtenClampOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify CompatibleTypes.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto converter = OpConversionPattern<AtenClampOp>::getTypeConverter();
    auto loc = op->getLoc();

    auto self = adaptor.getSelf();
    auto min = adaptor.getMin();
    auto max = adaptor.getMax();
    if (failed(checkClampParamsValid(op, converter, min, max)))
      return failure();

    // Process inputs because func normalizeElewiseOperands does not support
    // none.
    SmallVector<Value> operands = {self};
    int minIdx = 0, maxIdx = 0;
    if (!isa<Torch::NoneType>(min.getType())) {
      operands.emplace_back(min);
      minIdx = 1;
    }
    if (!isa<Torch::NoneType>(max.getType())) {
      operands.emplace_back(max);
      maxIdx = minIdx + 1;
    }
    SmallVector<Value> normalizedOperands;
    Value emptyOp;
    if (failed(normalizeElewiseOperands(rewriter, converter, op, operands,
                                        normalizedOperands, emptyOp)))
      return failure();

    auto resultElementType = getElementTypeOrSelf(emptyOp);
    bool isUnsigned = resultElementType.isUnsignedInteger();

    auto result = normalizedOperands[0];
    if (minIdx > 0)
      result =
          isUnsigned
              ? createLinalgBinary<linalg::BinaryFn::max_unsigned>(
                    rewriter, loc, result, normalizedOperands[minIdx], emptyOp)
              : createLinalgBinary<linalg::BinaryFn::max_signed>(
                    rewriter, loc, result, normalizedOperands[minIdx], emptyOp);

    if (maxIdx > 0)
      result =
          isUnsigned
              ? createLinalgBinary<linalg::BinaryFn::min_unsigned>(
                    rewriter, loc, result, normalizedOperands[maxIdx], emptyOp)
              : createLinalgBinary<linalg::BinaryFn::min_signed>(
                    rewriter, loc, result, normalizedOperands[maxIdx], emptyOp);

    rewriter.replaceOp(op, result);
    return success();
  }
};

class ConvertAtenToCast : public OpConversionPattern<AtenToDtypeOp> {
public:
  using OpConversionPattern<AtenToDtypeOp>::OpConversionPattern;
  using OpAdaptor = typename AtenToDtypeOp::Adaptor;
  LogicalResult
  matchAndRewrite(AtenToDtypeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify CompatibleTypes.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto converter = OpConversionPattern<AtenToDtypeOp>::getTypeConverter();
    auto loc = op->getLoc();
    auto input = adaptor.getSelf();
    auto resultTy =
        cast<RankedTensorType>(converter->convertType(op.getType()));
    if (!resultTy)
      return op->emitError("type conversion failed");
    auto dtype = resultTy.getElementType();

    auto result = createHFusionCastOp(rewriter, loc, dtype, input);
    if (failed(result))
      return op->emitError("hfusion dtype cast failed.");

    rewriter.replaceOp(op, *result);
    return success();
  }
};

} // namespace

void mlir::populateElementWisePatternsAndLegality(TypeConverter &typeConverter,
                                                  RewritePatternSet &patterns,
                                                  ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // clang-format off
#define INSERT_LINALG_BINARY_PATTERN(AtenOp, SignFn, UnsignFn)                        \
 do {                                                                                 \
  target.addIllegalOp<AtenOp>();                                                      \
  patterns.add<ConvertAtenToLinalgBinary<AtenOp, SignFn, UnsignFn>>(typeConverter,    \
                                                                            context); \
                                                              } while (0)

  INSERT_LINALG_BINARY_PATTERN(AtenMulTensorOp, linalg::BinaryFn::mul,
                                                linalg::BinaryFn::mul);
  INSERT_LINALG_BINARY_PATTERN(AtenMulScalarOp, linalg::BinaryFn::mul,
                                                linalg::BinaryFn::mul);
  INSERT_LINALG_BINARY_PATTERN(AtenDivTensorOp, linalg::BinaryFn::div,
                                                linalg::BinaryFn::div);
  INSERT_LINALG_BINARY_PATTERN(AtenDivScalarOp, linalg::BinaryFn::div,
                                                linalg::BinaryFn::div);
  INSERT_LINALG_BINARY_PATTERN(AtenClampMinTensorOp, linalg::BinaryFn::max_signed,
                                                     linalg::BinaryFn::max_unsigned);
  INSERT_LINALG_BINARY_PATTERN(AtenClampMinOp, linalg::BinaryFn::max_signed,
                                               linalg::BinaryFn::max_unsigned);
  INSERT_LINALG_BINARY_PATTERN(AtenClampMaxOp, linalg::BinaryFn::min_signed,
                                               linalg::BinaryFn::min_unsigned);
  INSERT_LINALG_BINARY_PATTERN(AtenClampMaxTensorOp, linalg::BinaryFn::min_signed,
                                                     linalg::BinaryFn::min_unsigned);
  INSERT_LINALG_BINARY_PATTERN(AtenMaximumOp, linalg::BinaryFn::max_signed,
                                              linalg::BinaryFn::max_unsigned);
  INSERT_LINALG_BINARY_PATTERN(AtenMinimumOp, linalg::BinaryFn::min_signed,
                                              linalg::BinaryFn::min_unsigned);
#undef INSERT_LINALG_BINARY_PATTERN

#define INSERT_LINALG_UNARY_PATTERN(AtenOp, FnType)                         \
 do {                                                                       \
  target.addIllegalOp<AtenOp>();                                            \
  patterns.add<ConvertAtenToLinalgUnary<AtenOp, FnType>>(typeConverter,     \
                                                            context);       \
                                                               } while (0)

  INSERT_LINALG_UNARY_PATTERN(AtenAbsOp, linalg::UnaryFn::abs);
  INSERT_LINALG_UNARY_PATTERN(AtenCeilOp, linalg::UnaryFn::ceil);
  INSERT_LINALG_UNARY_PATTERN(AtenFloorOp, linalg::UnaryFn::floor);
  INSERT_LINALG_UNARY_PATTERN(AtenNegOp, linalg::UnaryFn::negf);
  INSERT_LINALG_UNARY_PATTERN(AtenLogOp, linalg::UnaryFn::log);
  INSERT_LINALG_UNARY_PATTERN(AtenExpOp, linalg::UnaryFn::exp);
#undef INSERT_LINALG_UNARY_PATTERN

#define INSERT_HFUSION_UNARY_PATTERN(AtenOp, FnType)                        \
 do {                                                                       \
  target.addIllegalOp<AtenOp>();                                            \
  patterns.add<ConvertAtenToHfusionUnary<AtenOp, FnType>>(typeConverter,    \
                                                            context);       \
                                                               } while (0)

  INSERT_HFUSION_UNARY_PATTERN(AtenReciprocalOp, hfusion::UnaryFn::rec);
  INSERT_HFUSION_UNARY_PATTERN(AtenReluOp, hfusion::UnaryFn::relu);
  INSERT_HFUSION_UNARY_PATTERN(AtenRsqrtOp, hfusion::UnaryFn::rsqrt);
  INSERT_HFUSION_UNARY_PATTERN(AtenSqrtOp, hfusion::UnaryFn::sqrt);
  INSERT_HFUSION_UNARY_PATTERN(AtenErfOp, hfusion::UnaryFn::erf);
  INSERT_HFUSION_UNARY_PATTERN(AtenTanhOp, hfusion::UnaryFn::tanh);
  INSERT_HFUSION_UNARY_PATTERN(AtenSinOp, hfusion::UnaryFn::sin);
  INSERT_HFUSION_UNARY_PATTERN(AtenCosOp, hfusion::UnaryFn::cos);
  INSERT_HFUSION_UNARY_PATTERN(AtenBitwiseNotOp, hfusion::UnaryFn::vnot);
#undef INSERT_HFUSION_UNARY_PATTERN

#define INSERT_HFUSION_BINARY_PATTERN(AtenOp, FnType)                        \
 do {                                                                        \
  target.addIllegalOp<AtenOp>();                                             \
  patterns.add<ConvertAtenToHfusionBinary<AtenOp, FnType>>(typeConverter,    \
                                                            context);        \
                                                               } while (0)

  INSERT_HFUSION_BINARY_PATTERN(AtenPowTensorTensorOp, hfusion::BinaryFn::powf);
  INSERT_HFUSION_BINARY_PATTERN(AtenPowTensorScalarOp, hfusion::BinaryFn::powf);
  INSERT_HFUSION_BINARY_PATTERN(AtenPowScalarOp, hfusion::BinaryFn::powf);
  INSERT_HFUSION_BINARY_PATTERN(AtenLogicalAndOp, hfusion::BinaryFn::vand);
  INSERT_HFUSION_BINARY_PATTERN(AtenLogicalOrOp, hfusion::BinaryFn::vor);
#undef INSERT_HFUSION_BINARY_PATTERN

#define INSERT_HFUSION_COMPARE_PATTERN(AtenOp, FnType)                        \
 do {                                                                         \
  target.addIllegalOp<AtenOp>();                                              \
  patterns.add<ConvertAtenToHFusionCompare<AtenOp, FnType>>(typeConverter,    \
                                                            context);         \
                                                               } while (0)

  INSERT_HFUSION_COMPARE_PATTERN(AtenGtScalarOp, hfusion::CompareFn::vgt);
  INSERT_HFUSION_COMPARE_PATTERN(AtenGtTensorOp, hfusion::CompareFn::vgt);
  INSERT_HFUSION_COMPARE_PATTERN(AtenLtScalarOp, hfusion::CompareFn::vlt);
  INSERT_HFUSION_COMPARE_PATTERN(AtenLtTensorOp, hfusion::CompareFn::vlt);
  INSERT_HFUSION_COMPARE_PATTERN(AtenNeScalarOp, hfusion::CompareFn::vne);
  INSERT_HFUSION_COMPARE_PATTERN(AtenNeTensorOp, hfusion::CompareFn::vne);
  INSERT_HFUSION_COMPARE_PATTERN(AtenEqScalarOp, hfusion::CompareFn::veq);
  INSERT_HFUSION_COMPARE_PATTERN(AtenEqTensorOp, hfusion::CompareFn::veq);
  INSERT_HFUSION_COMPARE_PATTERN(AtenGeScalarOp, hfusion::CompareFn::vge);
  INSERT_HFUSION_COMPARE_PATTERN(AtenGeTensorOp, hfusion::CompareFn::vge);
  INSERT_HFUSION_COMPARE_PATTERN(AtenLeScalarOp, hfusion::CompareFn::vle);
  INSERT_HFUSION_COMPARE_PATTERN(AtenLeTensorOp, hfusion::CompareFn::vle);
#undef INSERT_HFUSION_COMPARE_PATTERN

#define INSERT_ADDSUB_PATTERN(AtenOp, FnType)                                \
 do {                                                                        \
  target.addIllegalOp<AtenOp>();                                             \
  patterns.add<ConvertAtenToAddSub<AtenOp, FnType>>(typeConverter, context); \
                                                               } while (0)

  INSERT_ADDSUB_PATTERN(AtenAddTensorOp, linalg::BinaryFn::add);
  INSERT_ADDSUB_PATTERN(AtenAddScalarOp, linalg::BinaryFn::add);
  INSERT_ADDSUB_PATTERN(AtenSubTensorOp, linalg::BinaryFn::sub);
  INSERT_ADDSUB_PATTERN(AtenSubScalarOp, linalg::BinaryFn::sub);
#undef INSERT_ADDSUB_PATTERN

  // clang-format on
  target.addIllegalOp<AtenSigmoidOp>();
  patterns.add<ConvertAtenToSigmoid>(typeConverter, context);
  target.addIllegalOp<AtenClampOp>();
  patterns.add<ConvertAtenToClamp>(typeConverter, context);
  target.addIllegalOp<AtenToDtypeOp>();
  patterns.add<ConvertAtenToCast>(typeConverter, context);

  // Uncategorized Pattern
  target.addIllegalOp<AtenWhereSelfOp>();
  patterns.add<ConvertAtenToWhere>(typeConverter, context);
}
