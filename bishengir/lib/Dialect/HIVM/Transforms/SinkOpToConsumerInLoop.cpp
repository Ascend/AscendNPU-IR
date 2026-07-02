//===- SinkOpToConsumerInLoop.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SINKOPTOCONSUMERINLOOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
#define DEBUG_TYPE "hivm-sink-op-to-consumer-in-loop"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

template <typename OpTy>
class SinkOpToConsumerInLoop : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (op.getOperation()->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "doesn't only have one result");

    auto opResult = op.getOperation()->getResult(0);
    SmallVector<OpOperand *> uses;
    for (auto &use : opResult.getUses()) {
      uses.push_back(&use);
      auto user = use.getOwner();
      if (op->getBlock() == user->getBlock())
        return rewriter.notifyMatchFailure(
            op, "and its user are in the same block, so that we can't sink it");

      auto loopParent = user->template getParentOfType<scf::ForOp>();
      if (!loopParent)
        return rewriter.notifyMatchFailure(op, "the user is not in a loop");

      // For manual VF case, scopeOp will be outlined into VF before ops(fillOp,
      // vbrcOp) are vectorized, so we can't sink these ops into scopeOp. See
      // more details in issue !1199.
      auto scopeParent = user->template getParentOfType<scope::ScopeOp>();
      if (scopeParent && scopeParent->hasAttr("outline"))
        return rewriter.notifyMatchFailure(op, "can't sink op into manual VF");
    }
    for_each(uses, [&](OpOperand *use) {
      auto user = use->getOwner();
      rewriter.setInsertionPoint(user);
      IRMapping map;
      Operation *newOp = rewriter.clone(*op.getOperation(), map);
      rewriter.modifyOpInPlace(user, [&]() { use->set(newOp->getResult(0)); });
    });
    rewriter.eraseOp(op);
    return success();
  }
};

struct SinkOpToConsumerInLoopPass
    : public impl::SinkOpToConsumerInLoopBase<SinkOpToConsumerInLoopPass> {
  void runOnOperation() override;
};
} // namespace

void SinkOpToConsumerInLoopPass::runOnOperation() {
  auto funcOp = getOperation();
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<SinkOpToConsumerInLoop<hivm::VBrcOp>,
               SinkOpToConsumerInLoop<linalg::BroadcastOp>,
               SinkOpToConsumerInLoop<linalg::FillOp>>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createSinkOpToConsumerInLoopPass() {
  return std::make_unique<SinkOpToConsumerInLoopPass>();
}
