//===--- CanonicalizeIterArg.cpp - Eliminate unused iter args -------------===//
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

#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/SCF/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_CANONICALIZEITERARG
#include "bishengir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "scf-canonicalize-iter-arg"

using namespace mlir;

static void handleIfElse(scf::IfOp ifOp, OpResult ifResult,
                         SetVector<Value> &equivalenceSet,
                         SmallVector<Value> &dfsStack) {
  // Add defined trace value to tentative list
  equivalenceSet.insert(ifResult);
  unsigned pos = ifResult.getResultNumber();
  dfsStack.push_back(ifOp.thenYield().getOperand(pos));
  dfsStack.push_back(ifOp.elseYield().getOperand(pos));
}

static void handleLoops(LoopLikeOpInterface loop, BlockArgument iterArg,
                        SetVector<Value> &equivalenceSet,
                        SmallVector<Value> &dfsStack) {
  // Since loop ops may or may not have induction var as block argument, we try
  // to offset the arg number
  unsigned argNo = iterArg.getArgNumber();
  unsigned resultNo = argNo - (iterArg.getParentBlock()->getNumArguments() -
                               loop.getInits().size());

  equivalenceSet.insert(loop->getResult(resultNo));

  for (Region *region : loop.getLoopRegions()) {
    assert(region->getBlocks().size() == 1 &&
           "Expecting loop regions to only have one block");
    Block &body = region->getBlocks().front();
    equivalenceSet.insert(body.getArgument(argNo));
    Operation *terminator = body.getTerminator();
    if (auto condOp = dyn_cast<scf::ConditionOp>(terminator))
      dfsStack.push_back(condOp.getArgs()[resultNo]);
    else
      dfsStack.push_back(terminator->getOperand(resultNo));
  }

  dfsStack.push_back(loop.getInits()[resultNo]);
}

/// Try to use proof by contradiction to prove whether or not the block arg
/// remains unchanged throughout all iterations
static bool isIterArgUnchanged(LoopLikeOpInterface loop, BlockArgument arg,
                               SetVector<Value> &equivalenceSet) {
  Value initVal = loop.getTiedLoopInit(arg)->get();
  // Build tentative equivalence set
  equivalenceSet.insert(arg);
  equivalenceSet.insert(initVal);
  Value resultVal = loop.getTiedLoopResult(arg);
  equivalenceSet.insert(resultVal);

  // Used to trace within nested scf structures
  SmallVector<Value> dfsStack;
  Value yieldVal = loop.getTiedLoopYieldedValue(arg)->get();
  dfsStack.push_back(yieldVal);
  while (!dfsStack.empty()) {
    Value traceUp = dfsStack.pop_back_val();

    // If we've already traced this value (init or iter arg), then this branch
    // holds equivalence
    LLVM_DEBUG(llvm::dbgs() << "\tTracing " << traceUp << "\n");
    if (equivalenceSet.contains(traceUp))
      continue;

    // Value could be block arg or result, get the defining operation either way
    Operation *defining = nullptr;
    BlockArgument innerArg = nullptr;
    auto opResult = dyn_cast<OpResult>(traceUp);
    if (opResult) {
      defining = opResult.getOwner();
    } else {
      assert(isa<BlockArgument>(traceUp) &&
             "Expecting non-OpResult value to be block argument");
      innerArg = cast<BlockArgument>(traceUp);
      defining = innerArg.getParentBlock()->getParentOp();
    }

    // If the current value's defining op is not within the scope of the current
    // loop being checked, we assume its not equivalent
    if (!defining || !loop->isAncestor(defining)) {
      LLVM_DEBUG(llvm::dbgs() << "\tNot ancestor\n");
      return false;
    }

    // Trace both branches of the if op while adding the corresponding result to
    // the equivalence set
    if (auto ifOp = dyn_cast<scf::IfOp>(defining)) {
      handleIfElse(ifOp, opResult, equivalenceSet, dfsStack);
      continue;
    }

    auto innerLoop = dyn_cast<LoopLikeOpInterface>(defining);
    // If the defining operation is not a loop/if op, then we say its unsafe to
    // assume equivalence
    if (!innerLoop) {
      LLVM_DEBUG(llvm::dbgs() << "\tNot scf\n");
      return false;
    }

    // Add values defined by the loop to tentative list, and trace values used
    // by the loop
    if (opResult)
      innerArg = innerLoop.getTiedLoopRegionIterArg(opResult);

    handleLoops(innerLoop, innerArg, equivalenceSet, dfsStack);
  }
  return true;
}

namespace {

template <typename LoopT>
struct CanonicalizeIterArgPattern : public OpRewritePattern<LoopT> {
public:
  using OpRewritePattern<LoopT>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(LoopT op, mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "\n\n========================================================"
                  "========================For loop\n"
               << op << "\n\n");
    bool changed = false;
    Operation *parentOp = op->getParentOp();
    DominanceInfo domInfo(parentOp);
    eliminateCommonSubExpressions(rewriter, domInfo, parentOp, &changed);

    SetVector<Value> equivalenceSet;
    for (BlockArgument arg : op.getRegionIterArgs()) {
      Value yieldVal = op.getTiedLoopYieldedValue(arg)->get();
      Value initVal = op.getTiedLoopInit(arg)->get();
      Value resultVal = op.getTiedLoopResult(arg);
      // Additional check to make sure we didn't clean this already
      if (yieldVal == initVal) {
        if (resultVal.use_empty())
          continue;
        resultVal.replaceAllUsesWith(initVal);
        changed = true;
      }
      if (!isIterArgUnchanged(op, arg, equivalenceSet)) {
        equivalenceSet.clear();
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "Matched " << yieldVal << "\n\tas unchanged\n\n");
      while (!equivalenceSet.empty()) {
        Value alias = equivalenceSet.pop_back_val();
        if (alias != initVal)
          alias.replaceAllUsesWith(initVal);
      }
      changed = true;
    }
    return success(changed);
  }
};

struct CanonicalizeIterArgPass
    : public impl::CanonicalizeIterArgBase<CanonicalizeIterArgPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    patterns.insert<CanonicalizeIterArgPattern<scf::ForOp>,
                    CanonicalizeIterArgPattern<scf::WhileOp>>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> scf::createCanonicalizeIterArgPass() {
  return std::make_unique<CanonicalizeIterArgPass>();
}
