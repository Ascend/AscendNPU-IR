//===- SinkOpToConsumerInLoop.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
#define GEN_PASS_DEF_SINKOPTOCONSUMERINLOOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-sink-op-to-consumer-in-loop"

using namespace mlir;
using namespace mlir::hivm;

namespace {

template <typename OpTy>
class SinkOpToConsumerInLoop : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (op.getOperation()->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "doesn't only have one result");

    auto users = op.getOperation()->getResult(0).getUsers();
    if (!llvm::hasSingleElement(users))
      return rewriter.notifyMatchFailure(op, "has more than one user");

    Operation *singleUser = *users.begin();
    auto scfParent = dyn_cast_if_present<scf::ForOp>(singleUser->getParentOp());
    if (!scfParent)
      return rewriter.notifyMatchFailure(op, "the user is not in a loop");

    if (op->getBlock() == singleUser->getBlock())
      return rewriter.notifyMatchFailure(op, "already in the same block");

    rewriter.setInsertionPoint(singleUser);
    IRMapping map;
    Operation *newBrcOp = rewriter.clone(*op.getOperation(), map);
    rewriter.replaceOp(op, newBrcOp);
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
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createSinkOpToConsumerInLoopPass() {
  return std::make_unique<SinkOpToConsumerInLoopPass>();
}
