//===-------------  Conversion from Bufferization to Triton dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonUtils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::hivm;

namespace {
// Process all of the ToTensorOp before dialect conversion
class ToTensorOpReplacementPattern
    : public OpConversionPattern<bufferization::ToTensorOp> {

public:
  using OpConversionPattern::OpConversionPattern;
  using OpConversionPattern<bufferization::ToTensorOp>::OneToNOpAdaptor;
  LogicalResult
  matchAndRewrite(bufferization::ToTensorOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefVal = op.getMemref();
    auto memrefTy = dyn_cast<MemRefType>(memrefVal.getType());
    if (!memrefTy) {
      return rewriter.notifyMatchFailure(op, "Invalid memref type");
    }
    auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!tensorTy || !tensorTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "Unsupport unranked or dynamic shape tensor");
    }

    FailureOr<MemRefDescriptor> desc =
        getMemRefDescriptor(rewriter, op.getLoc(), memrefTy,
                            adaptor.getMemref());
    if (failed(desc) || desc->getRank() != tensorTy.getRank())
      return rewriter.notifyMatchFailure(op, "no descriptor for memref");
    FailureOr<Value> maybePtrs = buildMemRefDescriptorPointers(
        rewriter, op.getLoc(), *desc, HIVMToTritonTypeConvert(memrefTy),
        tensorTy.getShape());
    if (failed(maybePtrs)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize to_tensor source pointers");
    }

    // Generate tt.load operator to get value from pointer tensor
    auto valTensor = rewriter.create<triton::LoadOp>(
        op.getLoc(), tensorTy, *maybePtrs, Value{}, Value{},
        llvm::ArrayRef<int32_t>{}, triton::PaddingOptionAttr{});

    rewriter.replaceOp(op, valTensor);
    return success();
  }
};
} // namespace

void mlir::hivm::populateBufferizationToTritonPatterns(
    TritonTypeConverter &converter, RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ToTensorOpReplacementPattern>(converter, ctx);
}
