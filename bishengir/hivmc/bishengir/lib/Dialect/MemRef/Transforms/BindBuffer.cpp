//===- BindBuffer.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to bind buffer according to annotation.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_BINDBUFFER
#include "bishengir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct BindBuffer : public impl::BindBufferBase<BindBuffer> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

struct BindBufferPattern : public OpRewritePattern<annotation::MarkOp> {
public:
  using OpRewritePattern<annotation::MarkOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(annotation::MarkOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.isAnnotatedByDynamicAttr(memref::kBindBufferAttrName))
      return rewriter.notifyMatchFailure(op,
                                         "Not annotated by bind buffer info");

    auto targetAlloc = op.getDynamicAttrValue(memref::kBindBufferAttrName);
    if (!isa<memref::AllocOp>(targetAlloc.getDefiningOp()))
      return rewriter.notifyMatchFailure(op,
                                         "The target buffer is not an alloc");

    SmallVector<Value> allocs = utils::tracebackMemRefVec(op.getSrc());
    if (allocs.empty())
      return rewriter.notifyMatchFailure(op, "Cannot find source alloc");

    for (auto sourceAlloc : allocs) {
      if (failed(bindBuffer(op, rewriter, targetAlloc, sourceAlloc))) {
        return failure();
      }
    }

    hivm::removeMarkOpDynamicAttr(op, memref::kBindBufferAttrName, rewriter);
    return success();
  }

  LogicalResult bindBuffer(annotation::MarkOp op,
                           mlir::PatternRewriter &rewriter, Value targetAlloc,
                           Value sourceAlloc) const {
    OpBuilder::InsertionGuard g(rewriter);
    Type targetType = targetAlloc.getType();
    Type sourceType = sourceAlloc.getType();
    // Try casting if the memref types are different.
    if (targetType != sourceType) {
      rewriter.setInsertionPointAfterValue(targetAlloc);
      if (memref::CastOp::areCastCompatible(targetType, sourceType)) {
        targetAlloc = rewriter.create<memref::CastOp>(op.getLoc(), sourceType,
                                                      targetAlloc);
      } else if (memref::MemorySpaceCastOp::areCastCompatible(targetType,
                                                              sourceType)) {
        targetAlloc = rewriter.create<memref::MemorySpaceCastOp>(
            op.getLoc(), sourceType, targetAlloc);
      } else {
        return rewriter.notifyMatchFailure(
            op, "Source and Target Type are not cast-compatible");
      }
    }

    // Replace source alloc by the target alloc.
    // Note: This requires the target alloc to dominate the chain of users from
    // the alloc to the annotated value.
    // For example:
    // ```
    // %source = ...
    // %cast = memref.memspace_cast %source
    // ..
    // %target = ...
    // annotation.mark %cast [bind_buffer = %target]
    // ```
    // This is invalid right now.
    rewriter.replaceAllUsesWith(sourceAlloc, targetAlloc);
    return success();
  }
};

void BindBuffer::runOnOperation() {
  MLIRContext *context = &getContext();
  auto operation = getOperation();
  RewritePatternSet patterns(context);
  patterns.insert<BindBufferPattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(operation, std::move(patterns)))) {
    operation->emitError("Invalid IR after bind buffer.");
    signalPassFailure();
    return;
  }

  auto walkResult = operation.walk([](annotation::MarkOp op) {
    if (op.isAnnotatedByDynamicAttr(memref::kBindBufferAttrName))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    operation->emitError("Failed to bind all allocs");
    signalPassFailure();
  }
}

std::unique_ptr<Pass> memref::createBindBufferPass() {
  return std::make_unique<BindBuffer>();
}
