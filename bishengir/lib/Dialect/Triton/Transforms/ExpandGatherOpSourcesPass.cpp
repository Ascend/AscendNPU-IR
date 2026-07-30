//===- ExpandTritonGatherSourcesPass.cpp ------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
// In the case that a tt.gather op's indices tensor is larger than its source
// tensor, and num elements of the result tensor > 32, expands the source tensor
// to be the same size of the indices tensor using tt.expand_dims, tt.broadcast,
// and tt.reshape ops to expand the source tensor
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace bishengir::triton {
#define GEN_PASS_DEF_EXPANDGATHEROPSOURCES
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct ExpandGatherSourcePattern : public OpRewritePattern<triton::GatherOp> {
  using OpRewritePattern<triton::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::GatherOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value indices = op.getIndices();
    Location loc = op->getLoc();

    RankedTensorType srcType = op.getSrc().getType();
    RankedTensorType indType = op.getIndices().getType();
    Type elementType = srcType.getElementType();

    ArrayRef<int64_t> srcShape = srcType.getShape();
    ArrayRef<int64_t> indicesShape = indType.getShape();
    uint32_t axis = op.getAxis();

    // If indices tensor is smaller than the source tensor, or the number of
    // elements in the indices tensor fits within a warp, do not expand the
    // source tensor
    if (indicesShape[axis] <= srcShape[axis] ||
        indType.getNumElements() <= 32) {
      return failure();
    }

    Value newSrc;

    if (srcShape[axis] == 1) {
      newSrc = rewriter.create<triton::BroadcastOp>(
          loc, op.getResult().getType(), src);
    } else {
      SmallVector<int64_t> curShape(srcShape);
      curShape.insert(curShape.begin() + axis, 1);

      RankedTensorType curType = RankedTensorType::get(curShape, elementType);
      Value expanded =
          rewriter.create<triton::ExpandDimsOp>(loc, curType, src, axis);

      // Divisibility not checked as both tensors are powers of two, and since
      // indicesShape[axis] > srcShape[axis], we know indicesShape[axis] is
      // divisible by srcShape[axis]
      curShape[axis] = indicesShape[axis] / srcShape[axis];
      curType = RankedTensorType::get(curShape, elementType);
      Value broadcasted =
          rewriter.create<triton::BroadcastOp>(loc, curType, expanded);

      curType = RankedTensorType::get(indicesShape, elementType);
      newSrc = rewriter.create<triton::ReshapeOp>(loc, curType, broadcasted);
    }

    auto newOp = rewriter.create<triton::GatherOp>(loc, newSrc, indices, axis);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

class ExpandGatherOpSourcesPass
    : public impl::ExpandGatherOpSourcesBase<ExpandGatherOpSourcesPass> {
public:
  using ExpandGatherOpSourcesBase::ExpandGatherOpSourcesBase;
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ExpandGatherSourcePattern>(context);

    if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
      mod.emitError("Unsupported gather operations found in the "
                    "SIMT kernel");
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createExpandGatherOpSourcesPass() {
  return std::make_unique<ExpandGatherOpSourcesPass>();
}

} // namespace bishengir::triton
