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
#include "mlir/IR/Value.h"
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
    if (opResult) {
      // getTiedLoopRegionIterArg doesn't work on while loops, so we use a more
      // general way to get the tied iter arg
      Block *innerBlk =
          &innerLoop.getLoopRegions().front()->getBlocks().front();
      unsigned offset =
          innerBlk->getNumArguments() - innerLoop.getRegionIterArgs().size();
      unsigned argNum = opResult.getResultNumber() + offset;
      innerArg = innerBlk->getArgument(argNum);
    }

    handleLoops(innerLoop, innerArg, equivalenceSet, dfsStack);
  }
  return true;
}

static bool isNoEffect(Operation *op) {
  if (auto mei = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    mei.getEffects(effects);
    return effects.empty();
  }
  // If op has no interface, Assume no Effects.
  return true;
}

static bool isInLoopBody(Value x, Block *body) {
  if (!body)
    return false;
  if (auto barg = dyn_cast<BlockArgument>(x))
    return barg.getOwner() == body;
  if (auto *defOp = x.getDefiningOp())
    return defOp->getBlock() == body;
  return false;
}

static LogicalResult
isIterationIndependent(Value yield, Block *body, BlockArgument allowedIterArg,
                  SmallPtrSetImpl<Operation *> &tobeDeletedOps) {
  SmallVector<Value, 8> worklist;
  SmallPtrSet<Value, 16> visitedVals;
  worklist.push_back(yield);
  visitedVals.insert(yield);

  SmallPtrSet<Operation *, 16> visitedOps;

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (cur == allowedIterArg)
      continue; // reach only accepted leaf

    Operation *def = cur.getDefiningOp();
    if (!def)
      return failure();

    if (def->getBlock() != body)
      return failure();

    // No memory effects.
    if (!isNoEffect(def))
      return failure();

    visitedOps.insert(def);
    for (Value operand : def->getOperands()) {
      if (!isInLoopBody(operand, body)) {
        continue; // ignore outside loop constants
      }

      if (visitedVals.insert(operand).second) // avoid double visit
        worklist.push_back(operand);
    }
  }
  // cannot delete iter arg if used elsewhere
  for (OpOperand &use : allowedIterArg.getUses()) {
    Operation *user = use.getOwner();
    if (!visitedOps.contains(user)) {
      if (!(isa<scf::YieldOp>(user) &&
            body->getArgument(use.getOperandNumber() + 1) == allowedIterArg))
        return failure();
    }
  }
  // visited ops to be deleted should only be used in current trace chain
  for (Operation *op : visitedOps) {
    for (Value res : op->getResults()) {
      for (OpOperand &use : res.getUses()) {
        Operation *user = use.getOwner();
        if (!visitedOps.contains(user)) {
          if (!(isa<scf::YieldOp>(user) &&
                body->getArgument(use.getOperandNumber() + 1) ==
                    allowedIterArg))
            return failure();
        }
      }
    }
  }

  tobeDeletedOps.insert(visitedOps.begin(), visitedOps.end());
  return success();
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

template <typename LoopT>
struct RemoveDeadIterArgPattern : public OpRewritePattern<LoopT> {
public:
  using OpRewritePattern<LoopT>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(LoopT forOp, mlir::PatternRewriter &rewriter) const override {
    unsigned numResults = forOp.getNumResults();
    if (numResults == 0)
      return failure();
    Block *body = forOp.getBody();
    auto yield = cast<scf::YieldOp>(body->getTerminator());

    SmallVector<unsigned, 4> removableIdxs;
    SmallVector<SmallPtrSet<Operation *, 8>, 4> opsToErasePerIdx;

    for (unsigned i = 0, e = numResults; i < e; ++i) {
      Value res = forOp.getResult(i);
      if (!res.use_empty())
        continue;
      BlockArgument iterArg = body->getArgument(i + 1); // iterarg[i]
      Value yielded = yield.getOperand(i);

      SmallPtrSet<Operation *, 8> tobeDeletedOps;
      if (failed(isIterationIndependent(yielded, body, iterArg, tobeDeletedOps)))
        continue;

      removableIdxs.push_back(i);
      opsToErasePerIdx.emplace_back(std::move(tobeDeletedOps));
    }

    if (removableIdxs.empty())
      return failure();

    // have to rewrite for op with new types
    llvm::SmallBitVector keep(numResults, true);
    for (unsigned idx : removableIdxs)
      keep.reset(idx);
    SmallVector<Type, 4> newResultTypes;
    for (unsigned i = 0; i < numResults; ++i)
      if (keep.test(i))
        newResultTypes.push_back(forOp.getResult(i).getType());
    SmallVector<Value, 4> newInitArgs;
    newInitArgs.reserve(forOp.getInitArgs().size());
    for (auto [i, init] : llvm::enumerate(forOp.getInitArgs()))
      if (keep.test(i))
        newInitArgs.push_back(init);
    rewriter.setInsertionPoint(forOp);
    auto newFor = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange newIterArgs) {
          // Map and Clone block
          IRMapping mapping;
          mapping.map(forOp.getInductionVar(), iv);

          unsigned newIdx = 0;
          for (unsigned i = 0; i < numResults; ++i) {
            if (keep.test(i)) {
              mapping.map(forOp.getRegionIterArgs()[i], newIterArgs[newIdx++]);
            }
          }

          SmallPtrSet<Operation *, 16> deletableAll;
          for (auto &set : opsToErasePerIdx)
            deletableAll.insert(set.begin(), set.end());

          for (Operation &op : forOp.getBody()->without_terminator()) {
            if (deletableAll.contains(&op))
              continue;
            b.clone(op, mapping);
          }

          auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          SmallVector<Value, 4> newYieldOperands;
          newYieldOperands.reserve(newResultTypes.size());
          for (unsigned i = 0; i < numResults; ++i) {
            if (!keep.test(i))
              continue;
            Value mapped = mapping.lookupOrNull(oldYield.getOperand(i));
            if (!mapped)
              mapped = oldYield.getOperand(i); // safe fallback
            newYieldOperands.push_back(mapped);
          }
          b.create<scf::YieldOp>(loc, newYieldOperands);
        });
    SmallVector<Value, 4> newResults(newFor.getResults().begin(),
                                     newFor.getResults().end());
    unsigned newIdx = 0;
    for (unsigned i = 0; i < numResults; ++i) {
      if (keep.test(i)) {
        forOp.getResult(i).replaceAllUsesWith(newResults[newIdx++]);
      }
    }

    rewriter.eraseOp(forOp);
    return success();
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
                    CanonicalizeIterArgPattern<scf::WhileOp>,
                    RemoveDeadIterArgPattern<scf::ForOp>>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> scf::createCanonicalizeIterArgPass() {
  return std::make_unique<CanonicalizeIterArgPass>();
}
