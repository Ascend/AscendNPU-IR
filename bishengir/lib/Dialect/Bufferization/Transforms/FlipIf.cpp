//===- FlipIf.cpp - Temporarily flip state update branches ----------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

// TopDown bufferization depends on branch traversal order. Consider:
//
//   for (..., old = initial) {
//     next = if (cond) {
//       new = produce_new(old);
//       consume_old(old);
//       yield new;
//     } else {
//       yield old;
//     }
//     yield next;
//   }
//
// Without flipping, analysis can insert two copies:
//   - else:        yield copy(old);
//   - for backedge: yield copy(next);
//
// Temporarily flip the condition and swap branches so analysis visits the
// forwarding branch first. In this example, only one copy remains:
//   - update branch: yield copy(new);  // After consume_old finishes.
//   - forwarding branch and for backedge: no copy.
//
// Restore the original condition and branch order after bufferization. Normal
// conflict analysis still determines copy placement; loop-entry copies are
// omitted from this example.

#include "bishengir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace bishengir::bufferization {
#define GEN_PASS_DEF_FLIPIF
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bishengir::bufferization

using namespace mlir;

namespace {
constexpr StringLiteral kState = "flip_if.restore_state";

bool matchesStateUpdate(scf::IfOp ifOp) {
  if (!ifOp.elseBlock())
    return false;
  for (Value ifElseYieldValue : ifOp.elseYield().getOperands()) {
    if (!isa<TensorType>(ifElseYieldValue.getType()))
      continue;
    auto iterArg = dyn_cast<BlockArgument>(ifElseYieldValue);
    if (!iterArg)
      continue;
    Operation *loop = iterArg.getOwner()->getParentOp();
    auto whileOp = dyn_cast<scf::WhileOp>(loop);
    const bool isLoopState =
        isa<scf::ForOp>(loop) ||
        (whileOp && iterArg.getOwner()->getParent() == &whileOp.getAfter());
    if (isLoopState && loop->isProperAncestor(ifOp))
      return true;
  }
  return false;
}

void swapBranches(scf::IfOp branch) {
  Region temporary;
  temporary.takeBody(branch.getThenRegion());
  branch.getThenRegion().takeBody(branch.getElseRegion());
  branch.getElseRegion().takeBody(temporary);
}

struct FlipIfPattern : OpRewritePattern<scf::IfOp> {
  bool restoreOnly;

  FlipIfPattern(MLIRContext *context, bool restoreOnly)
      : OpRewritePattern(context), restoreOnly(restoreOnly) {}

  LogicalResult matchAndRewrite(scf::IfOp branch,
                                PatternRewriter &rewriter) const override {
    auto condition = branch.getCondition().getDefiningOp<arith::CmpIOp>();
    bool marked = condition && condition->hasAttr(kState);
    if (!restoreOnly) {
      if (marked || !matchesStateUpdate(branch))
        return failure();
      rewriter.setInsertionPoint(branch);
      auto zero = rewriter.create<arith::ConstantIntOp>(branch.getLoc(), 0, 1);
      // SCF folds xor-true conditions by swapping their branches. An eq-false
      // cmpi survives the intervening DecomposeAllocTensorCopy folding.
      auto inverted = rewriter.create<arith::CmpIOp>(
          branch.getLoc(), arith::CmpIPredicate::eq, branch.getCondition(),
          zero);
      // The scalar condition survives when bufferization rebuilds the if.
      inverted->setAttr(kState, rewriter.getUnitAttr());
      rewriter.modifyOpInPlace(branch, [&] {
        branch.getConditionMutable().assign(inverted.getResult());
        swapBranches(branch);
      });
      return success();
    }
    if (!marked)
      return failure();
    rewriter.modifyOpInPlace(branch, [&] {
      branch.getConditionMutable().assign(condition.getLhs());
      swapBranches(branch);
    });
    if (condition->use_empty())
      rewriter.eraseOp(condition);
    return success();
  }
};

struct FlipIfPass : bishengir::bufferization::impl::FlipIfBase<FlipIfPass> {
  explicit FlipIfPass(const bishengir::bufferization::FlipIfOptions &options)
      : FlipIfBase(options) {}

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FlipIfPattern>(&getContext(), restoreOnly);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
bishengir::bufferization::createFlipIfPass(const FlipIfOptions &options) {
  return std::make_unique<FlipIfPass>(options);
}
