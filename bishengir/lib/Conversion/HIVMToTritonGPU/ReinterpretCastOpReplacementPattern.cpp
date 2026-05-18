//===-------------  Conversion from ReinterpretCastOp to Triton dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {
// Reinterpret-cast view semantics are consumed by the memory-access lowering
// patterns.  Keep the cast legal during dialect conversion by preserving its
// base pointer and dynamic layout operands in an unrealized cast.
class ReinterpretCastOpReplacementPattern
    : public OpConversionPattern<memref::ReinterpretCastOp> {

public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    SmallVector<Value> inputs{op.getSource()};
    llvm::append_range(inputs, op.getOffsets());
    llvm::append_range(inputs, op.getStrides());
    auto unrealizedCast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getResult().getType(), inputs);
    rewriter.replaceOp(op, unrealizedCast.getResult(0));
    return success();
  }
};
} // namespace

void mlir::hivm::populateReinterpretCastToUnrealizedCastPatterns(
    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ReinterpretCastOpReplacementPattern>(ctx);
}
