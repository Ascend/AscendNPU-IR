//===-------------  Conversion from Bufferization to Triton dialect -------===//
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
// Process all of the ToTensorOp before dialect conversion
class ToTensorOpReplacementPattern
    : public OpConversionPattern<bufferization::ToTensorOp> {

public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(bufferization::ToTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefVal = op.getMemref();
    auto memrefTy = dyn_cast<MemRefType>(memrefVal.getType());
    if (!memrefTy) {
      return rewriter.notifyMatchFailure(op, "Invalid memref type");
    }
    if (!memref::isStaticShapeAndContiguousRowMajor(memrefTy)) {
      return rewriter.notifyMatchFailure(op, "Unsupport non-contigous memref");
    }

    // Convert memref type to triton pointer type
    auto ptrTy = HIVMToTritonTypeConvert(memrefTy);

    auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!tensorTy || !tensorTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "Unsupport unranked or dynamic shape tensor");
    }

    // Generate tt.make_range operator to get continuous sequence
    auto num = tensorTy.getNumElements();
    auto rangeTensorTy = RankedTensorType::get({num}, rewriter.getI32Type());
    auto mkrng = rewriter.create<triton::MakeRangeOp>(op.getLoc(),
                                                      rangeTensorTy, 0, num);

    mlir::Value offset = mkrng;
    // Generate tt.reshape operator to get multi-dim continuous sequence tensor
    // of target shape if needed
    if (tensorTy.getRank() > 1) {
      auto reshapeTensorTy = RankedTensorType::get(
          tensorTy.getShape(), rangeTensorTy.getElementType());
      auto reshape = rewriter.create<triton::ReshapeOp>(op.getLoc(),
                                                        reshapeTensorTy, mkrng);
      offset = reshape;
    }

    auto ttPtr = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), ptrTy,
                                                             memrefVal);

    // Generate tt.splat operator to get pointer tensor of target shape
    auto ptrTensor = RankedTensorType::get(tensorTy.getShape(), ptrTy);
    auto splat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTensor,
                                                  ttPtr.getResult(0));

    // Generate tt.addptr operator to get pointer tensor with offset
    auto addptr = rewriter.create<triton::AddPtrOp>(op.getLoc(), ptrTensor,
                                                    splat, offset);

    // Generate tt.load operator to get value from pointer tensor
    auto valTensor = rewriter.create<triton::LoadOp>(
        op.getLoc(), tensorTy, addptr, Value{}, Value{},
        llvm::ArrayRef<int32_t>{}, triton::PaddingOptionAttr{});

    rewriter.replaceOp(op, valTensor);
    return success();
  }
};
} // namespace

void mlir::hivm::populateBufferizationToTritonPatterns(
    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ToTensorOpReplacementPattern>(ctx);
}
