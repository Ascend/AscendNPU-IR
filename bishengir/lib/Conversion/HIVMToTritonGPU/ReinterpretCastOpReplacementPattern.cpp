//===-------------  Conversion from ReinterpretCastOp to UnrealizedConversionCastOp dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::hivm;

namespace {
// Process all of the ReinterpretCastOp before dialect conversion
class ReinterpretCastOpReplacementPattern
    : public OpConversionPattern<memref::ReinterpretCastOp> {

public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value adaptedInput = op.getSource();
    Type outputType = op.getResult().getType();
    if (!outputType) {
      return failure();
    }
    auto unrealizedCast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), TypeRange(outputType), ValueRange(adaptedInput));
    rewriter.replaceOp(op, unrealizedCast.getResults());
    return success();
  }
};
} // namespace

void mlir::hivm::populateReinterpretCastToUnrealizedCastPatterns(
    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ReinterpretCastOpReplacementPattern>(ctx);
}
