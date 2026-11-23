//===-- DecomposeAllocTensorCopy.cpp - Decompose alloc_tensor(copy) -------===//
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
//
// This pass rewrites `bufferization.alloc_tensor (copy...)` into a
// `bufferization.alloc_tensor` + `hivm.hir.copy` pair: `alloc_tensor
// (copy...)` carries no in-place write, so a subsequent One-Shot Bufferization
// analysis cannot see/deal with the write it represents, which blocks the
// conflict analysis. In contrast, `hivm.hir.copy` carries an in-place write,
// so the decomposed form keeps the write visible to the following conflict
// analysis.
//
// On AIC kernels, copies of broadcast results are instead rewritten as a
// fresh allocation plus a clone of the broadcast writing into it, avoiding
// a read of the original result.
//
// All discardable attributes of the `alloc_tensor` are carried over to the
// replacement write operation.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_DECOMPOSEALLOCTENSORCOPY
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Recompute a broadcast into fresh storage instead of copying its result.
struct DecomposeAllocTensorVBrcPattern
    : public OpRewritePattern<bufferization::AllocTensorOp> {
  explicit DecomposeAllocTensorVBrcPattern(MLIRContext *ctx, int benefit = 2)
      : OpRewritePattern(ctx, benefit) {}

  LogicalResult matchAndRewrite(bufferization::AllocTensorOp allocTensorOp,
                                PatternRewriter &rewriter) const override {
    // Leave marked allocations to the generic copy pattern, which carries
    // their attributes over.
    if (!allocTensorOp->getDiscardableAttrs().empty())
      return failure();

    Value copySrc = allocTensorOp.getCopy();
    if (!copySrc)
      return failure();

    auto brcOp = copySrc.getDefiningOp<hivm::VBrcOp>();
    if (!brcOp)
      return failure();
    // A memref source may have changed since the original broadcast executed.
    if (isa<BaseMemRefType>(brcOp.getSrc().getType()))
      return failure();

    rewriter.setInsertionPoint(allocTensorOp);
    Location loc = allocTensorOp.getLoc();
    Value dst = utils::createAllocTensorOp(rewriter, loc, copySrc);
    auto newBrc = cast<hivm::VBrcOp>(rewriter.clone(*brcOp));
    rewriter.modifyOpInPlace(newBrc,
                             [&]() { newBrc.getDstMutable().assign(dst); });
    rewriter.replaceOp(allocTensorOp, newBrc.getResult());
    return success();
  }
};

/// Rewrite `bufferization.alloc_tensor (copy...)` into a
/// `bufferization.alloc_tensor` + `hivm.hir.copy` pair.
struct DecomposeAllocTensorCopyPattern
    : public OpRewritePattern<bufferization::AllocTensorOp> {
  using OpRewritePattern<bufferization::AllocTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::AllocTensorOp allocTensorOp,
                                PatternRewriter &rewriter) const override {
    Value copySrc = allocTensorOp.getCopy();
    if (!copySrc)
      return failure();
    rewriter.setInsertionPoint(allocTensorOp);
    Location loc = allocTensorOp.getLoc();
    auto value = utils::createAllocTensorOp(rewriter, loc, copySrc);
    auto copyOp = rewriter.create<hivm::CopyOp>(loc, allocTensorOp.getType(),
                                                copySrc, value);
    // Carry over all discardable attributes of the alloc_tensor to the copy,
    // skipping inherent properties such as `operandSegmentSizes`.
    for (auto attr : allocTensorOp->getDiscardableAttrs())
      copyOp->setAttr(attr.getName(), attr.getValue());
    rewriter.replaceOp(allocTensorOp, copyOp.getResult(0));
    return success();
  }
};

struct DecomposeAllocTensorCopyPass
    : public impl::DecomposeAllocTensorCopyBase<DecomposeAllocTensorCopyPass> {
  void runOnOperation() override;
};
} // namespace

void DecomposeAllocTensorCopyPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  auto funcCoreType = hivm::queryFuncCoreType(funcOp);
  if (funcCoreType && *funcCoreType == hivm::TFuncCoreType::AIC)
    patterns.add<DecomposeAllocTensorVBrcPattern>(ctx, /*benefit=*/2);
  patterns.add<DecomposeAllocTensorCopyPattern>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createDecomposeAllocTensorCopyPass() {
  return std::make_unique<DecomposeAllocTensorCopyPass>();
}
