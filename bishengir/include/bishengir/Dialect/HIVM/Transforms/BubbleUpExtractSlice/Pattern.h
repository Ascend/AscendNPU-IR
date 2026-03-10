//===- Pattern.h ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

class VTransposeBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;

  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class FixpipeBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class BufferizationBubbleUpStrategy : public BubbleUpStrategy {
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

class IfBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;

  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class SelectBubbleUpStrategy : public BubbleUpStrategy {
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