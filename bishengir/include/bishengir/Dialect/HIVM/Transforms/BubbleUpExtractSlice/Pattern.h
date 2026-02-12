//===- Pattern.h ----------------------------------------------------------===//
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
//============================================================================//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_BUBBLEUPEXTRACTSLICE_PATTERN_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_BUBBLEUPEXTRACTSLICE_PATTERN_H

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <memory>

namespace mlir::hivm::detail {

// Strategy pattern for different bubble up operations
class BubbleUpStrategy {
public:
  virtual ~BubbleUpStrategy() = default;
  virtual LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const = 0;
  /// should return true or false based on whether the defining op of this
  /// source of this sliceOp is actually supported.
  virtual bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const = 0;
};

class BroadcastBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;

  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class ReduceBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;

  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class ExpandBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;

  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class CollapseBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;

  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class LoopBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;

  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class LoopArgsBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;

  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class ElementwiseBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;

  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class BitcastBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;

  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class ExtractSliceBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class InsertSliceBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

/// Base class for bubble up patterns using Template Method pattern
class BubbleUpPattern : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  // Use shared_ptr to manage strategy lifetime
  BubbleUpPattern(MLIRContext *context,
                  SmallVector<std::shared_ptr<BubbleUpStrategy>> strategies)
      : OpRewritePattern<tensor::ExtractSliceOp>(context),
        bubbleUpStrategies(std::move(strategies)) {}

protected:
  /// Template method pattern
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const final;

  /// Common utility methods
  Operation *getDefOpForInsertionPoint(OpOperand &opr) const;

private:
  SmallVector<std::shared_ptr<BubbleUpStrategy>> bubbleUpStrategies;
};

class BubbleUpSubviewFromTiling : public OpRewritePattern<memref::SubViewOp> {
public:
  explicit BubbleUpSubviewFromTiling(MLIRContext *context)
      : OpRewritePattern<memref::SubViewOp>(context, /*benefit=*/0) {}

protected:
  /// Template method pattern
  LogicalResult matchAndRewrite(memref::SubViewOp subviewOp,
                                PatternRewriter &rewriter) const final;

private:
  FailureOr<memref::SubViewOp>
  createNewSrcAfterBubbledUp(RewriterBase &rewriter, size_t tilingDim,
                             memref::SubViewOp subviewOp,
                             memref::SubViewOp parentViewOp) const;

  FailureOr<memref::SubViewOp> createNewSubviewOpAfterBubbleUp(
      RewriterBase &rewriter, size_t tilingDim, memref::SubViewOp subviewOp,
      memref::SubViewOp parentViewOp, memref::SubViewOp newSrc) const;
};

} // namespace mlir::hivm::detail
#endif