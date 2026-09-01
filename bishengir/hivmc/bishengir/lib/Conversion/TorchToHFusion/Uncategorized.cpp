//===- Uncategorized .cpp - Conversion impl. for uncategorized Ops --------===//
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
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
// Approximation:
//
//  Gelu(x) ≈ 0.5 * x * (1 + tanh[sqrt(2/pi) * (x + 0.044715 * x^3)])
//  Ref: https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
//
//  tanh(t) = 2 / [1 + exp(-2t)] - 1
//  and
//  Gelu(x) ≈ x / [1 + exp(-sqrt(8/pi) * (x + 0.044715 * x^3))]
class ConvertAtenGeluOp : public OpConversionPattern<AtenGeluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenGeluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto selfType = cast<TensorType>(adaptor.getSelf().getType());
    Type elementType = selfType.getElementType();
    if (!isa<mlir::FloatType>(elementType)) {
      return rewriter.notifyMatchFailure(
          op, "Only floating-point datatype legalization supported");
    }

    std::string approximate;
    if (!matchPattern(op.getApproximate(), m_TorchConstantStr(approximate)) ||
        (approximate != "none" && approximate != "tanh")) {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported value of approximate");
    }

    Location loc = op->getLoc();
    Value x = adaptor.getSelf();
    Type resultDTy = elementType;
    Type accumulatorDType = getDefaultAccType(rewriter, resultDTy);
    if (accumulatorDType != resultDTy) {
      elementType = accumulatorDType;
      auto maybeCastX = createHFusionCastOp(rewriter, loc, elementType, x);
      if (failed(maybeCastX))
        return op->emitError("hfusion dtype cast failed.");
      x = *maybeCastX;
    }

    Value one =
        rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 1));
    Value coefficient = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, 0.044715));
    Value negSqrt8pi = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, -1.5957691216));
    Value emptyTensor = utils::createEmptyOp(rewriter, loc, x);

    // 1. get x^3
    auto squareX = createLinalgBinary<linalg::BinaryFn::mul>(rewriter, loc, x,
                                                             x, emptyTensor);
    auto cubeX = createLinalgBinary<linalg::BinaryFn::mul>(
        rewriter, loc, x, squareX, emptyTensor);
    // 2. get 0.044715 * x^3
    auto mulCo = createLinalgBinary<linalg::BinaryFn::mul>(
        rewriter, loc, cubeX, coefficient, emptyTensor);
    // 3. get x + 0.044715 * x^3
    auto addX = createLinalgBinary<linalg::BinaryFn::add>(rewriter, loc, mulCo,
                                                          x, emptyTensor);
    // 4. get -sqrt(8/pi) * (x + 0.044715 * x^3)
    auto varT = createLinalgBinary<linalg::BinaryFn::mul>(
        rewriter, loc, addX, negSqrt8pi, emptyTensor);
    // 5. get exp(-sqrt(8/pi) * (x + 0.044715 * x^3))
    auto exp = createLinalgUnary<linalg::UnaryFn::exp>(rewriter, loc, varT,
                                                       emptyTensor);
    // 6. get 1 + exp(-sqrt(8/pi) * (x + 0.044715 * x^3))
    auto addOne = createLinalgBinary<linalg::BinaryFn::add>(rewriter, loc, exp,
                                                            one, emptyTensor);
    // 7. get x / [1 + exp(-sqrt(8/pi) * (x + 0.044715 * x^3))]
    auto gelu = createLinalgBinary<linalg::BinaryFn::div>(rewriter, loc, x,
                                                          addOne, emptyTensor);

    if (accumulatorDType != resultDTy) {
      auto result = createHFusionCastOp(rewriter, loc, resultDTy, gelu);
      if (failed(result))
        return op->emitError("hfusion dtype cast failed.");
      gelu = *result;
    }
    rewriter.replaceOp(op, gelu);
    return success();
  }
};
} // namespace

void mlir::populateUncategorizedPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenGeluOp>();
  patterns.add<ConvertAtenGeluOp>(typeConverter, context);
}
