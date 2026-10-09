//===- NormalizeSIMTVF.cpp - Normalize SIMT VF operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZESIMTVF
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Rewrite a rank-0 slice insertion as a scalar insertion. Rank reduction can
/// produce this form after folding unit dimensions, but the Triton slice
/// lowering requires the source and destination tensors to have matching,
/// non-zero ranks.
struct NormalizeRankZeroInsertSlice
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSourceType().getRank() != 0)
      return failure();

    Value scalar = rewriter.create<tensor::ExtractOp>(
        op.getLoc(), op.getSource(), ValueRange{});
    SmallVector<Value> indices;
    indices.reserve(op.getMixedOffsets().size());
    for (OpFoldResult offset : op.getMixedOffsets())
      indices.push_back(
          getValueOrCreateConstantIndexOp(rewriter, op.getLoc(), offset));

    rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, scalar, op.getDest(),
                                                  indices);
    return success();
  }
};

struct NormalizeSIMTVFPass
    : public impl::NormalizeSIMTVFBase<NormalizeSIMTVFPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasUnsupportedRankReduction = false;
    module.walk([&](tensor::InsertSliceOp op) {
      int64_t sourceRank = op.getSourceType().getRank();
      int64_t destRank = op.getDestType().getRank();
      if (sourceRank == 0 || sourceRank == destRank)
        return;

      op.emitOpError()
          << "cannot normalize rank-reduced insertion with source rank "
          << sourceRank << " and destination rank " << destRank
          << "; only rank-0 sources are currently supported";
      hasUnsupportedRankReduction = true;
    });
    if (hasUnsupportedRankReduction) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<NormalizeRankZeroInsertSlice>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createNormalizeSIMTVFPass() {
  return std::make_unique<NormalizeSIMTVFPass>();
}
