//===- Datamovement.cpp - Conversion impl. for DataMovement Ops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===-----------------------------------------------------------------------===//

#include "bishengir/Conversion/TorchToHFusion/PopulatePatterns.h"
#include "bishengir/Conversion/TorchToHFusion/TorchToHFusion.h"
#include "bishengir/Conversion/TorchToHFusion/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/APInt.h"

#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertAtenPermuteOp : public OpConversionPattern<AtenPermuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenPermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    SmallVector<int64_t> dimensions;
    if (!matchPattern(op.getDims(), m_TorchListOfConstantInts(dimensions)))
      return op.emitError("all dimensions must be constant");

    Value self = adaptor.getSelf();
    auto result =
        mlir::permuteTensor(op, rewriter, op->getLoc(), dimensions, self);
    if (failed(result))
      return op.emitError("failed to perform permutation of tensor");

    auto outType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outType, *result);
    return success();
  }
};
} // namespace

namespace {

class ConvertAtenBroadcastToOp : public OpConversionPattern<AtenBroadcastToOp> {
public:
  ConvertAtenBroadcastToOp(const TypeConverter &typeConverter,
                           MLIRContext *context, bool ensureNoImplicitBroadcast)
      : OpConversionPattern<AtenBroadcastToOp>(typeConverter, context),
        ensureNoImplicitBroadcast(ensureNoImplicitBroadcast) {}
  LogicalResult
  matchAndRewrite(AtenBroadcastToOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Value self = adaptor.getSelf();

    SmallVector<Value> inShape;
    if (!getListConstructElements(adaptor.getSize(), inShape)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the size list is not from list construct");
    }

    // For dynamic input dimension we need to use the `broadcastToShape`
    // which in this case is `inShapeConverted` because this shape will yield
    // us the dimension size of the output.
    SmallVector<bool> useBroadcastToShape;
    int64_t inputRank = cast<RankedTensorType>(self.getType()).getRank();
    for (size_t i = inShape.size() - inputRank, e = inShape.size(); i < e;
         ++i) {
      int64_t dim;
      if (matchPattern(inShape[i], m_TorchConstantInt(&dim))) {
        if (dim < 0) {
          useBroadcastToShape.push_back(false);
        } else {
          useBroadcastToShape.push_back(true);
        }
      } else {
        // Note: Dynamic -1 (inferred) broadcast shapes are unimplemented.
        useBroadcastToShape.push_back(true);
      }
    }

    SmallVector<Value> inShapeConverted = getTypeConvertedValues(
        rewriter, op.getLoc(), getTypeConverter(), inShape);
    auto newResultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    Value result;
    if (failed(broadcastToGivenShape(op, rewriter, self, inShapeConverted,
                                     newResultType, ensureNoImplicitBroadcast,
                                     result, useBroadcastToShape))) {
      return op.emitError("unable to perform broadcast operation");
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  bool ensureNoImplicitBroadcast;
};
} // namespace

void mlir::populateDataMovementPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const ConvertTorchToHFusionOptions &options) {
  // Add some legal ops for torch-torch lowering.
  target.addLegalOp<ConstantIntOp>();
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenPermuteOp>();
  patterns.add<ConvertAtenPermuteOp>(typeConverter, context);

  bool ensureNoImplicitBroadcast = options.ensureNoImplicitBroadcast;
  target.addIllegalOp<AtenBroadcastToOp>();
  patterns.add<ConvertAtenBroadcastToOp>(typeConverter, context,
                                         ensureNoImplicitBroadcast);
}