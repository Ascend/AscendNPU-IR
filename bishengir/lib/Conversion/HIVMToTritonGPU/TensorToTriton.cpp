//===---- TensorToTriton.cpp - conversion from Tensor to Triton dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace {
template <typename ReshapeOpTy>
struct TensorReshapeToTritonReshape;

template <>
struct TensorReshapeToTritonReshape<tensor::ExpandShapeOp>
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp op,
                                PatternRewriter &rewriter) const final {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrcType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getResultType());
    if (!srcTy || !dstTy)
      return failure();
    // The SIMT path only needs a layout-preserving rank change here; keep it
    // in Triton IR so later vbrc lowering can consume it directly.
    rewriter.replaceOpWithNewOp<triton::ReshapeOp>(op, dstTy, op.getSrc(),
                                                   /*allowReorder=*/false);
    return success();
  }
};

template <>
struct TensorReshapeToTritonReshape<tensor::CollapseShapeOp>
    : public OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern<tensor::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp op,
                                PatternRewriter &rewriter) const final {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrcType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getResultType());
    if (!srcTy || !dstTy)
      return failure();
    // Mirror expand_shape lowering so shape-only canonicalizations stay inside
    // the Triton dialect after Stage1 conversion.
    rewriter.replaceOpWithNewOp<triton::ReshapeOp>(op, dstTy, op.getSrc(),
                                                   /*allowReorder=*/false);
    return success();
  }
};
} // namespace

void mlir::hivm::populateTensorToTritonPatterns(RewritePatternSet &patterns) {
  patterns.add<TensorReshapeToTritonReshape<tensor::ExpandShapeOp>,
               TensorReshapeToTritonReshape<tensor::CollapseShapeOp>>(
      patterns.getContext());
}
