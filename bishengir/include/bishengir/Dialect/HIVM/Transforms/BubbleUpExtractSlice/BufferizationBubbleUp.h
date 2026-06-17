//===- BufferizationBubbleUp.h - Bufferization bubble-up patterns ---------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_BUBBLEUPEXTRACTSLICE_BUFFERIZATIONBUBBLEUP_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_BUBBLEUPEXTRACTSLICE_BUFFERIZATIONBUBBLEUP_H

#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/BubbleUpUtils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::hivm::detail {

/// Insert UCC propagators only at the to_tensor memref (not the full chain).
struct InsertBufferizationPropagationPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override;
};

/// Propagate sliced shape requirements up the memref chain (toward alloc).
struct BufferizationPropagateUpPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  explicit BufferizationPropagateUpPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/3) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp propagateOp,
                                PatternRewriter &rewriter) const override;

private:
  LogicalResult
  propagateUpMemorySpaceCast(memref::MemorySpaceCastOp castOp,
                             UnrealizedConversionCastOp propagateOp,
                             PatternRewriter &rewriter) const;

  LogicalResult propagateUpSubView(memref::SubViewOp subViewOp,
                                   UnrealizedConversionCastOp propagateOp,
                                   PatternRewriter &rewriter) const;

  LogicalResult propagateUpAlloc(memref::AllocOp allocOp,
                                 UnrealizedConversionCastOp propagateOp,
                                 PatternRewriter &rewriter) const;

  LogicalResult
  propagateUpReinterpretCast(memref::ReinterpretCastOp castOp,
                             UnrealizedConversionCastOp propagateOp,
                             PatternRewriter &rewriter) const;
};

/// Propagate sliced shape requirements down to memref users.
struct BufferizationPropagateDownPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  explicit BufferizationPropagateDownPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/4) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp propagateOp,
                                PatternRewriter &rewriter) const override;

private:
  LogicalResult propagateDownMarkOp(annotation::MarkOp markOp,
                                    UnrealizedConversionCastOp propagateOp,
                                    OpOperand &use,
                                    PatternRewriter &rewriter) const;

  LogicalResult propagateDownLoadOp(hivm::LoadOp loadOp,
                                    UnrealizedConversionCastOp propagateOp,
                                    PatternRewriter &rewriter) const;

  LogicalResult propagateDownSubView(memref::SubViewOp subViewOp,
                                     UnrealizedConversionCastOp propagateOp,
                                     PatternRewriter &rewriter) const;

  LogicalResult
  propagateDownMemorySpaceCast(memref::MemorySpaceCastOp castOp,
                               UnrealizedConversionCastOp propagateOp,
                               PatternRewriter &rewriter) const;
};

struct BufferizationPropagatePostProcessPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  explicit BufferizationPropagatePostProcessPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp propagateOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::hivm::detail

#endif
