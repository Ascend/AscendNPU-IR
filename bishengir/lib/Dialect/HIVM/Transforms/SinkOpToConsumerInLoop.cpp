//===- SinkOpToConsumerInLoop.cpp -------------------------------*- C++ -*-===//
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
//===---------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
    if (hacc::utils::isRegBasedArch(utils::getTopLevelModuleOp(op)) &&
        !llvm::hasSingleElement(users))
      return rewriter.notifyMatchFailure(op, "has more than one user");

    auto opResult = op.getOperation()->getResult(0);
    SmallVector<OpOperand *> uses;
    uint64_t notFusedCounter = 0;
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

      // If op is sinked into loop but not fused into vf, we will have a
      // performance regression. So we will not sink it in this case.
      if (!isa<linalg::LinalgOp, hfusion::ElemwiseUnaryOp>(user)) {
        notFusedCounter++;
      }
    }

    if (hacc::utils::isAscend910_95(utils::getTopLevelModuleOp(op)) &&
        notFusedCounter > 1) {
      return rewriter.notifyMatchFailure(
          op, "if op is sinked into loop and it's user can't be fused into vf, "
              "this will cause performance regression");
    }

    for_each(uses, [&](OpOperand *use) {
      auto *user = use->getOwner();
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
