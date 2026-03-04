//===---- HIVMToTensor.cpp - conversion from HIVM to Tensor dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

struct HIVMToTensorConcatOp : public OpRewritePattern<hivm::VConcatOp> {
  using OpRewritePattern<hivm::VConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VConcatOp concatOp,
                                PatternRewriter &rewriter) const final{
    Location loc = concatOp.getLoc();
    ValueRange srcs = concatOp.getSrc();
    Value dst = concatOp.getDst();

    SmallVector<Value> tensorSrcs;
    for (Value src : srcs) {
      if (mlir::isa<MemRefType>(src.getType())) {
        // TODO: need to fix bufferization::ToTensorOp
        Value tensor = rewriter.create<bufferization::ToTensorOp>(
            loc, cast<MemRefType>(src.getType()), src);
        tensorSrcs.push_back(tensor);
      } else {
        tensorSrcs.push_back(src);
      }
    }

    int64_t concatDim = static_cast<int64_t>(concatOp.getDim());

    auto tensorConcatOp =
        rewriter.create<tensor::ConcatOp>(loc, concatDim, tensorSrcs);
    dst.replaceAllUsesWith(tensorConcatOp.getResult());
    rewriter.replaceOp(concatOp, tensorConcatOp);
    return success();
  }
};

struct HIVMToTensorPadOp : public OpRewritePattern<hivm::VPadOp> {
  using OpRewritePattern<hivm::VPadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VPadOp padOp,
                                PatternRewriter &rewriter) const final {
    Location loc = padOp.getLoc();
    auto src = padOp.getSrc();
    auto dst = padOp.getDst();
    auto padValue = padOp.getPadValue();
    auto lowPads = padOp.getMixedLowPad();
    auto highPads = padOp.getMixedHighPad();

    //  if src is memrefType, need convert to TensorType first
    Value tensorSrc = src;
    if (mlir::isa<MemRefType>(tensorSrc.getType())) {
      // TODO: need to fix bufferization::ToTensorOp
      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          loc, cast<MemRefType>(tensorSrc.getType()), src);
      tensorSrc = tensor;
    }

    // if outputType is memrefType, need convert to TensorType first
    Type outputType = dst.getType();
    if (mlir::isa<MemRefType>(outputType)) {
      outputType =
          RankedTensorType::get(cast<ShapedType>(outputType).getShape(),
                                cast<MemRefType>(outputType).getElementType());
    }
    TensorType outputTensorType = cast<RankedTensorType>(outputType);

    // convert to tensor::PadOp
    auto tensorPadOp =
        rewriter.create<tensor::PadOp>(padOp->getLoc(), outputTensorType,
                                       tensorSrc, lowPads, highPads, padValue);

    dst.replaceAllUsesWith(tensorPadOp.getResult());
    rewriter.replaceOp(padOp, tensorPadOp);

    return success();
  }
};

void mlir::hivm::populateHIVMToTensorPatterns(RewritePatternSet &patterns) {
  patterns.add<HIVMToTensorConcatOp, HIVMToTensorPadOp>(patterns.getContext());
}
