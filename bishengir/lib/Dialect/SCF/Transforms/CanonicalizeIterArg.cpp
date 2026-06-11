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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <queue>

#include "bishengir/Dialect/SCF/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_CANONICALIZEITERARG
#include "bishengir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "scf-canonicalize-iter-arg"

using namespace mlir;

static void printKeepMask(raw_ostream &os, const llvm::SmallBitVector &keep) {
  os << '[';
  for (unsigned i = 0, e = keep.size(); i != e; ++i)
    os << (keep.test(i) ? '1' : '0');
  os << ']';
}

static void
debugUnexpectedBuilderFailure(StringRef where, Operation *op,
                              const llvm::SmallBitVector *keep = nullptr,
                              StringRef reason = {}, Value failedValue = {},
                              Operation *failedOp = nullptr) {
  LLVM_DEBUG({
    llvm::dbgs() << "Unexpected builder failure in " << where << "\n";
    if (!reason.empty())
      llvm::dbgs() << "  reason: " << reason << "\n";
    if (keep) {
      llvm::dbgs() << "  keep mask: ";
      printKeepMask(llvm::dbgs(), *keep);
      llvm::dbgs() << "\n";
    }
    if (failedValue)
      llvm::dbgs() << "  failed value: " << failedValue << "\n";
    if (failedOp) {
      llvm::dbgs() << "  failed nested op: ";
      failedOp->print(
          llvm::dbgs(),
          OpPrintingFlags().printGenericOpForm().elideLargeElementsAttrs());
      llvm::dbgs() << "\n";
    }
    if (op) {
      llvm::dbgs() << "  parent op: ";
      op->print(
          llvm::dbgs(),
          OpPrintingFlags().printGenericOpForm().elideLargeElementsAttrs());
      llvm::dbgs() << "\n";
    }
  });
  // Do NOT hard-assert here: the rebuild logic still legitimately fails on
  // some IR patterns (e.g. when a kept yielded value's defining ops cannot be
  // remapped because some dependency wasn't added to neededOps -- this is
  // especially common for mix-mode (cv-pipeline) kernels whose backward
  // traces include hivm.hir.anchor / cross-core anchor chains that the new
  // rebuild path from commit 3361019ae does not fully model). The callers
  // already handle the failure() return path gracefully by leaving the
  // original op in place -- aborting compilation in release builds is far
  // worse than skipping the canonicalization. Note: this project's build
  // currently passes both -DNDEBUG and -UNDEBUG (the latter wins), so a
  // bare `#ifndef NDEBUG assert(false)` would still abort. Until the root
  // cause is fixed, keep this strictly diagnostic.
}

static void handleIfElse(scf::IfOp ifOp, OpResult ifResult,
                         SetVector<Value> &equivalenceSet,
                         SmallVector<Value> &dfsStack) {
  // Add defined trace value to tentative list
  equivalenceSet.insert(ifResult);
  unsigned pos = ifResult.getResultNumber();
  dfsStack.push_back(ifOp.thenYield().getOperand(pos));
  dfsStack.push_back(ifOp.elseYield().getOperand(pos));
}

static bool handleLoops(LoopLikeOpInterface loop, BlockArgument iterArg,
                        SetVector<Value> &equivalenceSet,
                        SmallVector<Value> &dfsStack) {
  // Since loop ops may or may not have induction var as block argument, we try
  // to offset the arg number
  unsigned argNo = iterArg.getArgNumber();

  OpResult res = loop.getTiedLoopResult(iterArg);
  if (!res)
    return false;

  unsigned resultNo = res.getResultNumber();
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
  return true;
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
  if (resultVal)
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

    if (!handleLoops(innerLoop, innerArg, equivalenceSet, dfsStack))
      return false;
  }
  return true;
}

static bool isNoEffect(Operation *op) {
  if (auto mei = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    mei.getEffects(effects);
    return effects.empty();
  }
  // If op has no interface, assume has effects.
  return false;
}

static bool hasRecursiveSideEffects(Operation *op) {
  bool hasEffects = false;
  op->walk([&](Operation *nestedOp) {
    if (!isNoEffect(nestedOp)) {
      hasEffects = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasEffects;
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
      continue; // accepted leaf

    // Other block args of the loop body (induction var, other iter args)
    // are valid leaves — they don't depend on allowedIterArg.
    if (auto ba = dyn_cast<BlockArgument>(cur)) {
      if (ba.getOwner() == body)
        continue;
      return failure();
    }

    Operation *def = cur.getDefiningOp();
    if (!def)
      return failure();

    if (def->getBlock() != body)
      return failure();

    if (!isNoEffect(def))
      return failure();

    visitedOps.insert(def);
    for (Value operand : def->getOperands()) {
      if (!isInLoopBody(operand, body))
        continue; // loop-invariant constant
      if (visitedVals.insert(operand).second)
        worklist.push_back(operand);
    }
  }

  // Collect ops that directly use allowedIterArg — only these MUST be deleted,
  // since the iter arg will no longer exist in the rebuilt loop.
  SmallPtrSet<Operation *, 8> mustDelete;
  for (OpOperand &use : allowedIterArg.getUses()) {
    Operation *user = use.getOwner();
    if (isa<scf::YieldOp>(user)) {
      // OK if yielded back at its own position; fail if used at another
      // position
      if (body->getArgument(use.getOperandNumber() + 1) == allowedIterArg)
        continue;
      return failure();
    }
    if (!visitedOps.contains(user))
      return failure();
    mustDelete.insert(user);
  }

  // Verify must-delete ops can actually be deleted: all their result uses
  // must be within mustDelete or the yield-for-allowedIterArg.
  for (Operation *op : mustDelete) {
    for (Value res : op->getResults()) {
      for (OpOperand &use : res.getUses()) {
        Operation *user = use.getOwner();
        if (mustDelete.contains(user))
          continue;
        if (isa<scf::YieldOp>(user) &&
            body->getArgument(use.getOperandNumber() + 1) == allowedIterArg)
          continue;
        // Result is used outside the delete set — can't safely delete this op
        return failure();
      }
    }
  }

  tobeDeletedOps.insert(mustDelete.begin(), mustDelete.end());
  return success();
}

namespace {

template <typename LoopT>
struct CanonicalizeYieldValPattern : public OpRewritePattern<LoopT> {
public:
  using OpRewritePattern<LoopT>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(LoopT op, mlir::PatternRewriter &rewriter) const override {
    bool changed = false;
    for (BlockArgument arg : op.getRegionIterArgs()) {
      Value yieldVal = op.getTiedLoopYieldedValue(arg)->get();
      Value resultVal = op.getTiedLoopResult(arg);

      if (resultVal.use_empty())
        continue;

      if (!isa<TensorType>(yieldVal.getType()))
        continue;

      if (auto *defOp = yieldVal.getDefiningOp()) {
        if (op.getBodyRegion().isAncestor(defOp->getParentRegion()))
          continue;
      } else {
        auto blockArg = cast<BlockArgument>(yieldVal);
        if (blockArg.getOwner() == op.getBody())
          continue;
      }
      changed = true;
      resultVal.replaceAllUsesWith(yieldVal);
    }
    return success(changed);
  }
};

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
        if (!resultVal || resultVal.use_empty())
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
      if (failed(
              isIterationIndependent(yielded, body, iterArg, tobeDeletedOps)))
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
    // Copy attributes from oldFor to newFor.
    newFor->setAttrs(forOp->getAttrs());
    rewriter.eraseOp(forOp);
    return success();
  }
};

// Operands that are structurally required by an SCF op even though they are
// not part of its iter-arg channels (predicate of an scf.if, control operands
// of an scf.for). These must be marked live whenever the op itself is live so
// the rebuild stage finds the producers it needs.
static llvm::SmallVector<Value> getSCFNeededVals(Operation *defOp) {
  llvm::SmallVector<Value> collectedVals;
  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    // condition
    collectedVals.push_back(ifOp.getCondition());
  } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    // induction vars
    auto numControlOperands = forOp.getNumControlOperands();
    for (unsigned i = 0; i < numControlOperands; ++i) {
      collectedVals.push_back(forOp->getOperand(i));
    }
  }
  return collectedVals;
}

static llvm::SmallVector<Value> tracebackMemValsStep(Value val) {
  llvm::SmallVector<Value> collectedVals;
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    if (auto forOp = dyn_cast_if_present<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (auto *initOperand = forOp.getTiedLoopInit(blockArg)) {
        collectedVals.push_back(initOperand->get());
      }
      if (auto *yieldedValue = forOp.getTiedLoopYieldedValue(blockArg)) {
        collectedVals.push_back(yieldedValue->get());
      }
    } else if (auto whileOp =
                   dyn_cast<scf::WhileOp>(blockArg.getOwner()->getParentOp())) {
      if (blockArg.getOwner()->getParent() == &whileOp.getBefore()) {
        if (auto *initOperand = whileOp.getTiedLoopInit(blockArg)) {
          collectedVals.push_back(initOperand->get());
        }
        if (auto *yieldedValueAfter =
                whileOp.getTiedLoopYieldedValue(blockArg)) {
          collectedVals.push_back(yieldedValueAfter->get());
        }
      } else {
        assert(blockArg.getOwner()->getParent() == &whileOp.getAfter());
        auto argNum = blockArg.getArgNumber();
        assert(whileOp.getConditionOp().getArgs().size() > argNum);
        auto yieldedValueBefore = whileOp.getConditionOp().getArgs()[argNum];
        collectedVals.push_back(yieldedValueBefore);
      }
    }
    return collectedVals;
  }

  auto resultVal = dyn_cast<OpResult>(val);
  assert(resultVal != nullptr);
  if (!resultVal) {
    return collectedVals;
  }

  auto *defOp = resultVal.getDefiningOp();
  auto resultNum = resultVal.getResultNumber();
  assert(defOp != nullptr);

  auto scfNeededVals = getSCFNeededVals(defOp);
  llvm::append_range(collectedVals, scfNeededVals);

  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    // then
    auto thenYield = ifOp.thenYield();
    auto yieldedValueThen = thenYield->getOperand(resultNum);
    collectedVals.push_back(yieldedValueThen);
    // else
    if (ifOp.elseBlock()) {
      auto elseYield = ifOp.elseYield();
      auto yieldedValueElse = elseYield->getOperand(resultNum);
      collectedVals.push_back(yieldedValueElse);
    }
  } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    // yield
    assert(forOp.getYieldedValues().size() > resultNum);
    auto yieldedValue = forOp.getYieldedValues()[resultNum];
    collectedVals.push_back(yieldedValue);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
    // yield before
    assert(whileOp.getConditionOp().getArgs().size() > resultNum);
    auto yieldedValueBefore = whileOp.getConditionOp().getArgs()[resultNum];
    collectedVals.push_back(yieldedValueBefore);
    // yield after
    assert(whileOp.getYieldedValues().size() > resultNum);
    auto yieldedValueAfter = whileOp.getYieldedValues()[resultNum];
    collectedVals.push_back(yieldedValueAfter);
  } else if (auto scopeOp = dyn_cast<scope::ScopeOp>(defOp)) {
    Region &scopeRegion = scopeOp.getRegion();
    Block &scopeBlock = scopeRegion.front();
    auto returnOp = dyn_cast<scope::ReturnOp>(scopeBlock.getTerminator());
    assert(returnOp != nullptr);
    assert(returnOp->getOperands().size() > resultNum);
    auto returnedValue = returnOp->getOperand(resultNum);
    collectedVals.push_back(returnedValue);
  } else {
    defOp->walk([&](Operation *nested) {
      llvm::append_range(collectedVals, nested->getOperands());
    });
  }

  return collectedVals;
}

// BFS over values: from each rootVal walk backward via tracebackMemValsStep
// until shouldStop accepts. Visited stop-values are returned in insertion
// order. `onVisitStep` is called for every value popped (used to record the
// defining ops as "needed" for the rebuild).
template <typename StopPredicateT, typename VisitStepT>
static llvm::SmallVector<Value>
tracebackMemVals(const SmallVectorImpl<Value> &rootVals,
                 StopPredicateT &&shouldStop, VisitStepT &&onVisitStep) {
  std::queue<Value> que;
  llvm::DenseSet<Value> visitedVals;
  llvm::SetVector<Value> collectedValsSet;
  for (Value rootVal : rootVals) {
    DEBUG_WITH_TYPE("scf-canonicalize-iter-arg-traceback",
                    { llvm::dbgs() << "traceback-root: " << rootVal << '\n'; });
    if (visitedVals.insert(rootVal).second) {
      que.push(rootVal);
    }
  }

  while (!que.empty()) {
    auto curVal = que.front();
    DEBUG_WITH_TYPE("scf-canonicalize-iter-arg-traceback",
                    { llvm::dbgs() << "traceback-step: " << curVal << '\n'; });
    que.pop();

    onVisitStep(curVal);

    if (shouldStop(curVal)) {
      collectedValsSet.insert(curVal);
      continue;
    }

    auto nextVals = tracebackMemValsStep(curVal);
    if (!nextVals.empty()) {
      for (auto nextVal : nextVals) {
        if (visitedVals.insert(nextVal).second) {
          que.push(nextVal);
        }
      }
    }
  }

  return collectedValsSet.takeVector();
}

// Resolve `oldValue` against the rebuild mapping in two ways:
//   1. If the rewrite has already produced a replacement, use it.
//   2. If `oldValue` is defined in a region that strictly contains the block
//      we are inserting into, the value is in scope and reusable as-is.
// A return of nullptr means the caller cannot satisfy the operand and should
// abort the rebuild for this op.
static Value mapForRebuild(IRMapping &mapping, Value oldValue,
                           Block *builderBlock) {
  // Prefer an already-remapped value when available.
  if (Value mapped = mapping.lookupOrNull(oldValue))
    return mapped;
  // If defBlock's region is an ancestor of (or equal to) builderBlock's region,
  // the value is defined in an outer or same scope and is safe to reuse as-is.
  // Otherwise it is defined inside the region being rebuilt and must be mapped.
  Block *defBlock = nullptr;
  if (auto barg = dyn_cast<BlockArgument>(oldValue))
    defBlock = barg.getOwner();
  else if (Operation *defOp = oldValue.getDefiningOp())
    defBlock = defOp->getBlock();
  if (defBlock) {
    Region *defRegion = defBlock->getParent();
    Region *builderRegion = builderBlock->getParent();
    if (defRegion && defRegion->isAncestor(builderRegion))
      return oldValue;
  }
  return nullptr;
}

static void copyIRMapping(const IRMapping &from, IRMapping &to) {
  // Clone all mapping tables so nested rebuild scopes can branch independently.
  for (auto [k, v] : from.getValueMap())
    to.map(k, v);
  for (auto [k, v] : from.getBlockMap())
    to.map(k, v);
  for (auto [k, v] : from.getOperationMap())
    to.map(k, v);
}

// Filter a range by a SmallBitVector kept-mask. `kept=true` keeps elements
// whose bit is set; `kept=false` returns the dropped elements (used to verify
// nothing live still references them).
template <typename RangeT>
static auto filterByKeep(RangeT &&range, const llvm::SmallBitVector &keep,
                         bool kept = true) {
  using ElementType = std::decay_t<decltype(*std::begin(range))>;
  llvm::SmallVector<ElementType> filtered;
  for (auto [i, element] : llvm::enumerate(range)) {
    if (keep.test(i) == kept) {
      filtered.push_back(element);
    }
  }
  return filtered;
}

static LogicalResult
cloneOpRecursivelyDroppingIterArgs(mlir::PatternRewriter &rewriter,
                                   Operation &oldOp, IRMapping &mapping,
                                   SmallPtrSetImpl<Operation *> &neededOps);

// Rebuild an scf.if while dropping iter-arg-style results that turned out to
// be dead.
//
// Two-pass strategy:
//   1. First create a result-less scf.if and clone live ops into both branches
//      so mapForRebuild has a real block ancestry chain when checking which
//      yield operands are reconstructable.
//   2. Inspect the cloned bodies, decide which results survive based on
//      yield-operand mappability, then create the final scf.if with the
//      compacted result types and splice the cloned ops over.
//
// The first-pass scf.if is then erased empty. Doing it in two passes avoids
// having to predict yield mappability without a real IR neighborhood, which
// is unreliable for nested control flow.
static LogicalResult
rebuildIfOpDroppingIterArgs(mlir::PatternRewriter &rewriter, scf::IfOp oldIf,
                            IRMapping &outerMapping,
                            SmallPtrSetImpl<Operation *> &neededOps) {
  OpBuilder::InsertionGuard guard(rewriter);
  Value mappedCond = mapForRebuild(outerMapping, oldIf.getCondition(),
                                   rewriter.getInsertionBlock());
  if (!mappedCond)
    return failure();

  const unsigned numResults = oldIf.getNumResults();
  auto oldThenYield = oldIf.thenYield();
  auto oldElseYield = oldIf.elseBlock() ? oldIf.elseYield() : scf::YieldOp();

  bool rebuildFailed = false;
  StringRef rebuildFailureReason;
  Value rebuildFailureValue;
  Operation *rebuildFailureOp = nullptr;

  // First pass: build a no-result ifOp so the then/else blocks are attached
  // to the IR. This gives mapForRebuild a valid block ancestry chain when
  // cloning ops and checking yield operand mappability.
  auto firstIf = rewriter.create<scf::IfOp>(
      oldIf.getLoc(), TypeRange{}, mappedCond,
      /*addThenBlock=*/true, /*addElseBlock=*/oldIf.elseBlock() != nullptr);

  IRMapping thenMapping;
  copyIRMapping(outerMapping, thenMapping);
  {
    OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToStart(firstIf.thenBlock());
    for (Operation &regionOp : oldIf.thenBlock()->without_terminator()) {
      if (!neededOps.contains(&regionOp))
        continue;
      if (failed(cloneOpRecursivelyDroppingIterArgs(rewriter, regionOp,
                                                    thenMapping, neededOps))) {
        rebuildFailed = true;
        rebuildFailureReason =
            "failed to clone live op in then while rebuilding scf.if";
        rebuildFailureOp = &regionOp;
        break;
      }
    }
    // Placeholder yield to keep the block structurally valid.
    rewriter.create<scf::YieldOp>(oldThenYield.getLoc());
  }

  IRMapping elseMapping;
  if (!rebuildFailed && oldIf.elseBlock()) {
    copyIRMapping(outerMapping, elseMapping);
    OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToStart(firstIf.elseBlock());
    for (Operation &regionOp : oldIf.elseBlock()->without_terminator()) {
      if (!neededOps.contains(&regionOp))
        continue;
      if (failed(cloneOpRecursivelyDroppingIterArgs(rewriter, regionOp,
                                                    elseMapping, neededOps))) {
        rebuildFailed = true;
        rebuildFailureReason =
            "failed to clone live op in else while rebuilding scf.if";
        rebuildFailureOp = &regionOp;
        break;
      }
    }
    rewriter.create<scf::YieldOp>(oldElseYield.getLoc());
  }

  if (rebuildFailed) {
    rewriter.eraseOp(firstIf);
    debugUnexpectedBuilderFailure("rebuildIfOpDroppingIterArgs", oldIf,
                                  /*keep=*/nullptr, rebuildFailureReason,
                                  rebuildFailureValue, rebuildFailureOp);
    return failure();
  }

  // Determine which results survive by checking yield operand mappability
  // against the attached first-pass blocks.
  llvm::SmallBitVector keep(numResults, false);
  SmallVector<Value, 4> thenYieldOperands;
  SmallVector<Value, 4> elseYieldOperands;
  for (unsigned i = 0; i < numResults; ++i) {
    Value mappedThen = mapForRebuild(thenMapping, oldThenYield.getOperand(i),
                                     firstIf.thenBlock());
    Value mappedElse =
        oldIf.elseBlock()
            ? mapForRebuild(elseMapping, oldElseYield.getOperand(i),
                            firstIf.elseBlock())
            : nullptr;
    if (!mappedThen || (oldIf.elseBlock() && !mappedElse)) {
      continue;
    }
    keep.set(i);
    thenYieldOperands.push_back(mappedThen);
    if (oldIf.elseBlock())
      elseYieldOperands.push_back(mappedElse);
  }

  SmallVector<Type, 4> newResultTypes;
  for (Type resultType : filterByKeep(oldIf.getResultTypes(), keep)) {
    newResultTypes.push_back(resultType);
  }

  // Second pass: build the final ifOp with correct result types, splicing
  // cloned ops from the first-pass blocks (excluding placeholder yields).
  auto newIf = rewriter.create<scf::IfOp>(
      oldIf.getLoc(), newResultTypes, mappedCond,
      /*addThenBlock=*/true, /*addElseBlock=*/oldIf.elseBlock() != nullptr);
  // Copy attributes from oldIf to newIf.
  newIf->setAttrs(oldIf->getAttrs());

  {
    Block *src = firstIf.thenBlock();
    newIf.thenBlock()->getOperations().splice(
        newIf.thenBlock()->begin(), src->getOperations(), src->begin(),
        std::prev(src->end()));
    {
      OpBuilder::InsertionGuard innerGuard(rewriter);
      rewriter.setInsertionPointToEnd(newIf.thenBlock());
      rewriter.create<scf::YieldOp>(oldThenYield.getLoc(), thenYieldOperands);
    }
  }
  if (oldIf.elseBlock() && newIf.elseBlock()) {
    Block *src = firstIf.elseBlock();
    newIf.elseBlock()->getOperations().splice(
        newIf.elseBlock()->begin(), src->getOperations(), src->begin(),
        std::prev(src->end()));
    {
      OpBuilder::InsertionGuard innerGuard(rewriter);
      rewriter.setInsertionPointToEnd(newIf.elseBlock());
      rewriter.create<scf::YieldOp>(oldElseYield.getLoc(), elseYieldOperands);
    }
  }
  // firstIf now has empty blocks (only placeholder yields remain); safe to
  // erase since no ops were double-owned.
  rewriter.eraseOp(firstIf);

  // Expose mapping from old compacted channels to new results.
  unsigned newIdx = 0;
  for (Value oldResult : filterByKeep(oldIf.getResults(), keep)) {
    outerMapping.map(oldResult, newIf.getResult(newIdx++));
  }
  return success();
}

// Rebuild an scf.for with a compacted iter-arg channel set.
//
// Single-pass: the new for-op is created without a body builder so its body
// block is attached to the IR before we populate it. That ordering matters
// because mapForRebuild walks parent regions to validate value scoping, and
// an unattached block would make every operand check fail.
//
// Returns failure() and erases the new for-op when any operand cannot be
// remapped, preserving the original loop for the caller.
static FailureOr<scf::ForOp>
rebuildForOpDroppingIterArgs(mlir::PatternRewriter &rewriter, scf::ForOp oldFor,
                             const llvm::SmallBitVector &keep,
                             SmallPtrSetImpl<Operation *> &neededOps,
                             IRMapping &outerMapping) {
  OpBuilder::InsertionGuard guard(rewriter);
  // Keep-mask must match the for-op channel count.
  const unsigned numResults = oldFor.getNumResults();
  if (keep.size() != numResults)
    return failure();

  Block *oldBody = oldFor.getBody();
  // lb/ub/step/initArgs are defined in the parent scope of oldFor.
  Block *builderBlock = rewriter.getInsertionBlock();
  Value lb = mapForRebuild(outerMapping, oldFor.getLowerBound(), builderBlock);
  Value ub = mapForRebuild(outerMapping, oldFor.getUpperBound(), builderBlock);
  Value step = mapForRebuild(outerMapping, oldFor.getStep(), builderBlock);
  if (!lb || !ub || !step)
    return failure();

  // Materialize filtered init args in parent scope.
  SmallVector<Value, 4> newInitArgs;
  for (Value initArg : filterByKeep(oldFor.getInitArgs(), keep)) {
    Value mappedInit = mapForRebuild(outerMapping, initArg, builderBlock);
    if (!mappedInit) {
      return failure();
    }
    newInitArgs.push_back(mappedInit);
  }

  // Create the for op with no body builder so the op is inserted first and
  // its body block is properly linked into the IR before we populate it.
  // This ensures builder.getInsertionBlock() has a valid parent chain when
  // mapForRebuild walks up to check value ancestry.
  auto newFor =
      rewriter.create<scf::ForOp>(oldFor.getLoc(), lb, ub, step, newInitArgs);
  // Copy attributes from oldFor to newFor.
  newFor->setAttrs(oldFor->getAttrs());

  Block *newBody = newFor.getBody();
  IRMapping bodyMapping;
  copyIRMapping(outerMapping, bodyMapping);
  bodyMapping.map(oldFor.getInductionVar(), newFor.getInductionVar());

  // Map surviving old iter-args to the rebuilt for body arguments.
  unsigned iterArgIdx = 0;
  for (Value iterArg : filterByKeep(oldFor.getRegionIterArgs(), keep)) {
    bodyMapping.map(iterArg, newFor.getRegionIterArgs()[iterArgIdx++]);
  }

  bool rebuildFailed = false;
  StringRef rebuildFailureReason;
  Value rebuildFailureValue;
  Operation *rebuildFailureOp = nullptr;

  {
    OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToStart(newBody);

    // Rebuild only live ops; nested control flow is handled recursively.
    for (Operation &op : oldBody->without_terminator()) {
      if (!neededOps.contains(&op))
        continue;
      if (failed(cloneOpRecursivelyDroppingIterArgs(rewriter, op, bodyMapping,
                                                    neededOps))) {
        rebuildFailed = true;
        rebuildFailureReason =
            "failed to clone live op while rebuilding scf.for";
        rebuildFailureOp = &op;
        break;
      }
    }

    SmallVector<Value> newYieldOperands;
    if (!rebuildFailed) {
      auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
      for (Value oldYieldOperand : filterByKeep(oldYield.getOperands(), keep)) {
        Value mapped = mapForRebuild(bodyMapping, oldYieldOperand,
                                     rewriter.getInsertionBlock());
        if (!mapped) {
          rebuildFailed = true;
          rebuildFailureReason =
              "failed to remap kept yielded value while rebuilding scf.for";
          rebuildFailureValue = oldYieldOperand;
          break;
        }
        newYieldOperands.push_back(mapped);
      }
    }

    if (!rebuildFailed) {
      if (newBody->mightHaveTerminator()) {
        if (auto yieldOp = dyn_cast<scf::YieldOp>(newBody->getTerminator())) {
          rewriter.eraseOp(yieldOp);
        }
      }
      OpBuilder::InsertionGuard innerGuard(rewriter);
      rewriter.setInsertionPointToEnd(newBody);
      rewriter.create<scf::YieldOp>(oldFor.getLoc(), newYieldOperands);
    }
  }

  if (rebuildFailed) {
    rewriter.eraseOp(newFor);
    debugUnexpectedBuilderFailure("rebuildForOpDroppingIterArgs", oldFor, &keep,
                                  rebuildFailureReason, rebuildFailureValue,
                                  rebuildFailureOp);
    return failure();
  }

  // Expose mapping from old compacted channels to new results.
  unsigned newIdx = 0;
  for (Value oldResult : filterByKeep(oldFor.getResults(), keep)) {
    outerMapping.map(oldResult, newFor.getResult(newIdx++));
  }
  return newFor;
}

// Rebuild an scf.while with compacted before/after channels.
//
// scf.while keeps the same channel count across inits, before-block args,
// after-block args, and yields, so the keep-mask is shared. The before
// region is populated first because its scf.condition feeds the after
// region's block arguments; populating after first would break operand
// remapping inside the before region.
static FailureOr<scf::WhileOp> rebuildWhileOpDroppingIterArgs(
    mlir::PatternRewriter &rewriter, scf::WhileOp oldWhile,
    const llvm::SmallBitVector &keep, SmallPtrSetImpl<Operation *> &neededOps,
    IRMapping &outerMapping) {
  OpBuilder::InsertionGuard guard(rewriter);
  // while carries the same channel count across inits/results/before/after.
  const unsigned numInits = oldWhile.getInits().size();
  const unsigned numResults = oldWhile.getNumResults();
  if (numInits == 0 || numInits != numResults || keep.size() != numResults) {
    return failure();
  }

  Block *beforeBody = oldWhile.getBeforeBody();
  Block *afterBody = oldWhile.getAfterBody();
  auto oldCond = oldWhile.getConditionOp();
  auto oldYield = oldWhile.getYieldOp();
  if (!beforeBody || !afterBody || !oldCond || !oldYield) {
    return failure();
  }
  if (beforeBody->getNumArguments() != numInits ||
      afterBody->getNumArguments() != numResults ||
      oldCond.getArgs().size() != numResults ||
      oldYield.getNumOperands() != numInits) {
    return failure();
  }

  // Build compacted init operands for channels that survive.
  // Init args are defined in the parent scope of oldWhile.
  SmallVector<Value> newInits;
  for (Value initArg : filterByKeep(oldWhile.getInits(), keep)) {
    Value mappedInit =
        mapForRebuild(outerMapping, initArg, rewriter.getInsertionBlock());
    if (!mappedInit) {
      return failure();
    }
    newInits.push_back(mappedInit);
  }

  SmallVector<Type> newResultTypes;
  for (Value oldResult : filterByKeep(oldWhile.getResults(), keep)) {
    newResultTypes.push_back(oldResult.getType());
  }

  bool rebuildFailed = false;
  StringRef rebuildFailureReason;
  Value rebuildFailureValue;
  Operation *rebuildFailureOp = nullptr;

  // Create the while op with no body builders so it is inserted first and
  // both regions are properly linked into the IR before we populate them.
  auto newWhile = rewriter.create<scf::WhileOp>(
      oldWhile.getLoc(), newResultTypes, newInits,
      /*beforeBuilder=*/nullptr, /*afterBuilder=*/nullptr);
  // Copy attributes from oldWhile to newWhile.
  newWhile->setAttrs(oldWhile->getAttrs());

  // Populate the before region.
  {
    OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToStart(newWhile.getBeforeBody());

    IRMapping beforeMapping;
    copyIRMapping(outerMapping, beforeMapping);
    unsigned newIdx = 0;
    for (Value oldArg : filterByKeep(beforeBody->getArguments(), keep)) {
      beforeMapping.map(oldArg,
                        newWhile.getBeforeBody()->getArgument(newIdx++));
    }

    for (Operation &op : beforeBody->without_terminator()) {
      if (!neededOps.contains(&op)) {
        continue;
      }
      if (failed(cloneOpRecursivelyDroppingIterArgs(rewriter, op, beforeMapping,
                                                    neededOps))) {
        rebuildFailed = true;
        rebuildFailureReason = "failed to clone live op in before region while "
                               "rebuilding scf.while";
        rebuildFailureOp = &op;
        break;
      }
    }

    Value mappedCond;
    if (!rebuildFailed) {
      mappedCond = mapForRebuild(beforeMapping, oldCond.getCondition(),
                                 rewriter.getInsertionBlock());
      if (!mappedCond) {
        rebuildFailed = true;
        rebuildFailureReason = "failed to remap scf.condition predicate while "
                               "rebuilding scf.while";
        rebuildFailureValue = oldCond.getCondition();
      }
    }

    SmallVector<Value, 4> newCondArgs;
    if (!rebuildFailed) {
      for (Value oldCondArg : filterByKeep(oldCond.getArgs(), keep)) {
        Value mapped = mapForRebuild(beforeMapping, oldCondArg,
                                     rewriter.getInsertionBlock());
        if (!mapped) {
          rebuildFailed = true;
          rebuildFailureReason =
              "failed to remap scf.condition arg while rebuilding scf.while";
          rebuildFailureValue = oldCondArg;
          break;
        }
        newCondArgs.push_back(mapped);
      }
    }

    if (!rebuildFailed) {
      OpBuilder::InsertionGuard innerGuard(rewriter);
      rewriter.setInsertionPointToEnd(newWhile.getBeforeBody());
      rewriter.create<scf::ConditionOp>(oldCond.getLoc(), mappedCond,
                                        newCondArgs);
    }
  }

  if (rebuildFailed) {
    rewriter.eraseOp(newWhile);
    debugUnexpectedBuilderFailure("rebuildWhileOpDroppingIterArgs", oldWhile,
                                  &keep, rebuildFailureReason,
                                  rebuildFailureValue, rebuildFailureOp);
    return failure();
  }

  // Populate the after region.
  {
    OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToStart(newWhile.getAfterBody());

    IRMapping afterMapping;
    copyIRMapping(outerMapping, afterMapping);
    unsigned newIdx = 0;
    for (Value oldArg : filterByKeep(afterBody->getArguments(), keep)) {
      afterMapping.map(oldArg, newWhile.getAfterBody()->getArgument(newIdx++));
    }

    for (Operation &op : afterBody->without_terminator()) {
      if (!neededOps.contains(&op))
        continue;
      if (failed(cloneOpRecursivelyDroppingIterArgs(rewriter, op, afterMapping,
                                                    neededOps))) {
        rebuildFailed = true;
        rebuildFailureReason = "failed to clone live op in after region while "
                               "rebuilding scf.while";
        rebuildFailureOp = &op;
        break;
      }
    }

    SmallVector<Value, 4> newYieldOperands;
    if (!rebuildFailed) {
      for (Value oldYieldOperand : filterByKeep(oldYield.getOperands(), keep)) {
        Value mapped = mapForRebuild(afterMapping, oldYieldOperand,
                                     rewriter.getInsertionBlock());
        if (!mapped) {
          rebuildFailed = true;
          rebuildFailureReason =
              "failed to remap scf.yield operand while rebuilding scf.while";
          rebuildFailureValue = oldYieldOperand;
          break;
        }
        newYieldOperands.push_back(mapped);
      }
    }

    if (!rebuildFailed) {
      OpBuilder::InsertionGuard innerGuard(rewriter);
      rewriter.setInsertionPointToEnd(newWhile.getAfterBody());
      rewriter.create<scf::YieldOp>(oldYield.getLoc(), newYieldOperands);
    }
  }

  if (rebuildFailed) {
    rewriter.eraseOp(newWhile);
    debugUnexpectedBuilderFailure("rebuildWhileOpDroppingIterArgs", oldWhile,
                                  &keep, rebuildFailureReason,
                                  rebuildFailureValue, rebuildFailureOp);
    return failure();
  }

  // Expose mapping from old compacted channels to new results.
  unsigned newIdx = 0;
  for (Value oldResult : filterByKeep(oldWhile.getResults(), keep)) {
    outerMapping.map(oldResult, newWhile.getResult(newIdx++));
  }
  return newWhile;
}

// Recursive driver: clone an op into the rebuilt region, recursing into nested
// SCF ops with their own keep-masks derived from operand mappability.
//
// For nested scf.for/scf.while, channels whose init args cannot be remapped
// are dropped from the nested op as well. The assert that no needed op still
// uses a dropped result enforces an invariant the backward-trace stage was
// supposed to establish: by the time we reach this point, every value the
// rebuild relies on must already be live on its tied channel.
static LogicalResult
cloneOpRecursivelyDroppingIterArgs(mlir::PatternRewriter &rewriter,
                                   Operation &oldOp, IRMapping &outerMapping,
                                   SmallPtrSetImpl<Operation *> &neededOps) {
  OpBuilder::InsertionGuard guard(rewriter);
  // Handle structured control-flow ops with dedicated rebuild paths.
  if (auto oldIf = dyn_cast<scf::IfOp>(&oldOp)) {
    return rebuildIfOpDroppingIterArgs(rewriter, oldIf, outerMapping,
                                       neededOps);
  }

  if (auto nestedFor = dyn_cast<scf::ForOp>(&oldOp)) {
    // Derive which nested channels can survive based on mappable init args.
    const unsigned childResults = nestedFor.getNumResults();
    llvm::SmallBitVector childKeep(childResults, true);
    for (auto [i, initArg] : llvm::enumerate(nestedFor.getInitArgs())) {
      Value mappedInit =
          mapForRebuild(outerMapping, initArg, rewriter.getInsertionBlock());
      if (!mappedInit) {
        childKeep.reset(i);
      }
    }

    // Guard against dropping a nested channel still needed by a live op.
    //
    // The outer rebuild is currently iterating through `oldOp.getParentOp()`'s
    // body and will rebuild that parent's terminator(s) via `filterByKeep`
    // on the *outer* keep mask. If the immediate-parent terminator (e.g. the
    // outer `scf.yield`) is the user of a dropped nested result, the
    // corresponding operand on the new terminator will be filtered out by
    // that mechanism — the use never makes it into the rebuilt IR, so it is
    // not an obstacle here. Real obstacles are uses by ops that the outer
    // rebuild will clone as-is (any non-terminator user). Limit the assert
    // to those.
    auto isHandledByOuterTerminator = [&](Operation *user) {
      Operation *parent = oldOp.getParentOp();
      if (!parent)
        return false;
      if (auto outerFor = dyn_cast<scf::ForOp>(parent))
        return user == outerFor.getBody()->getTerminator();
      if (auto outerIf = dyn_cast<scf::IfOp>(parent)) {
        if (user == outerIf.thenYield().getOperation())
          return true;
        return outerIf.elseBlock() &&
               user == outerIf.elseYield().getOperation();
      }
      if (auto outerWhile = dyn_cast<scf::WhileOp>(parent)) {
        return user == outerWhile.getConditionOp().getOperation() ||
               user == outerWhile.getYieldOp().getOperation();
      }
      return false;
    };

    for (Value result :
         filterByKeep(nestedFor.getResults(), childKeep, false)) {
      for (OpOperand &use : result.getUses()) {
        if (isHandledByOuterTerminator(use.getOwner()))
          continue;
        if (neededOps.contains(use.getOwner())) {
          assert(false &&
                 "dropping nested iter-arg/result still used by needed op");
          return failure();
        }
      }
    }

    auto rebuiltOr = rebuildForOpDroppingIterArgs(
        rewriter, nestedFor, childKeep, neededOps, outerMapping);
    if (failed(rebuiltOr)) {
      debugUnexpectedBuilderFailure(
          "cloneOpRecursivelyDroppingIterArgs(nested scf.for)", nestedFor,
          &childKeep,
          "nested scf.for rebuild failed while cloning live control flow",
          Value(), &oldOp);
      return failure();
    }
    return success();
  }

  if (auto nestedWhile = dyn_cast<scf::WhileOp>(&oldOp)) {
    // Derive which nested while channels are still reconstructable.
    const unsigned childResults = nestedWhile.getNumResults();
    llvm::SmallBitVector childKeep(childResults, true);
    for (auto [i, initArg] : llvm::enumerate(nestedWhile.getInits())) {
      Value mappedInit =
          mapForRebuild(outerMapping, initArg, rewriter.getInsertionBlock());
      if (!mappedInit) {
        childKeep.reset(i);
      }
    }

    // Guard against dropping a nested channel still needed by a live op.
    for (Value result :
         filterByKeep(nestedWhile.getResults(), childKeep, false)) {
      for (OpOperand &use : result.getUses()) {
        if (neededOps.contains(use.getOwner())) {
          assert(false &&
                 "dropping nested while result still used by needed op");
          return failure();
        }
      }
    }

    auto rebuiltOr = rebuildWhileOpDroppingIterArgs(
        rewriter, nestedWhile, childKeep, neededOps, outerMapping);
    if (failed(rebuiltOr)) {
      debugUnexpectedBuilderFailure(
          "cloneOpRecursivelyDroppingIterArgs(nested scf.while)", nestedWhile,
          &childKeep,
          "nested scf.while rebuild failed while cloning live control flow",
          Value(), &oldOp);
      return failure();
    }
    return success();
  }

  // Plain ops are cloned directly through the current mapping.
  LLVM_DEBUG({
    llvm::dbgs() << "Cloning non-SCF op directly: ";
    oldOp.print(
        llvm::dbgs(),
        OpPrintingFlags().printGenericOpForm().elideLargeElementsAttrs());
    llvm::dbgs() << "\n";
  });
  for (Value oldOperand : oldOp.getOperands()) {
    Value mapped = mapForRebuild(outerMapping, oldOperand, oldOp.getBlock());
    if (!mapped) {
      debugUnexpectedBuilderFailure(
          "cloneOpRecursivelyDroppingIterArgs(non-SCF op)", &oldOp,
          /*keep=*/nullptr, "missing mapped operand while cloning non-SCF op",
          oldOperand);
      return failure();
    }
  }
  rewriter.clone(oldOp, outerMapping);
  return success();
}

static void seedLiveRootsFromResults(ValueRange results,
                                     llvm::SmallBitVector &keep,
                                     SmallVectorImpl<Value> &rootVals) {
  // Any externally used loop result is immediately live.
  for (auto [i, resultVal] : llvm::enumerate(results)) {
    if (resultVal.use_empty())
      continue;
    rootVals.push_back(resultVal);
    keep.set(i);
  }
}

static void collectRootsFromSideEffects(Block *body,
                                        SmallPtrSetImpl<Operation *> &neededOps,
                                        SmallVectorImpl<Value> &rootVals) {
  // Side-effecting ops are treated as live roots; trace through their inputs.
  body->walk([&](Operation *nestedOp) {
    if (!hasRecursiveSideEffects(nestedOp)) {
      return;
    }
    neededOps.insert(nestedOp);
    if (isa<scf::IfOp, scf::ForOp>(nestedOp)) {
      auto scfNeededVals = getSCFNeededVals(nestedOp);
      llvm::append_range(rootVals, scfNeededVals);
    } else {
      for (Value operand : nestedOp->getOperands()) {
        rootVals.push_back(operand);
      }
    }
  });
}

// Backward-trace dead-iter-arg removal for scf.for.
//
// Replaces the use-driven RemoveDeadIterArgPattern when the nested IR contains
// SCF/scope ops the simple use-walk cannot reason about. Algorithm:
//
//   1. Seed live roots from externally-used loop results and from every
//      side-effecting op inside the body. Side-effecting ops anchor liveness
//      regardless of whether their results escape the loop.
//   2. Trace backward from the roots through SCF channels until we find every
//      loop-carried block argument that contributes to a live value, recording
//      the producing ops as `neededOps`.
//   3. Each newly-discovered live channel makes its tied yield operand a new
//      root; iterate until the keep mask is stable.
//   4. Validate that every kept yield operand is something the rebuild can
//      reconstruct (a kept block arg or a needed op in the body).
//   5. Rebuild the loop with only the surviving channels and redirect uses.
//
// If every channel turns out to be live, the pattern fails so the rewrite
// driver can move on.
struct RemoveDeadIterArgBackwardForPattern
    : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter) const override {
    // Empty carried state means nothing to prune.
    const unsigned numResults = forOp.getNumResults();
    if (numResults == 0)
      return failure();

    Block *body = forOp.getBody();
    auto oldYield = cast<scf::YieldOp>(body->getTerminator());
    llvm::SmallBitVector keep(numResults, false);
    SmallPtrSet<Operation *, 32> neededOps;
    SmallVector<Value, 32> rootVals;
    seedLiveRootsFromResults(forOp.getResults(), keep, rootVals);
    collectRootsFromSideEffects(body, neededOps, rootVals);

    auto traceForRoots = [&]() {
      return tracebackMemVals(
          rootVals,
          [&](Value v) {
            auto barg = dyn_cast<BlockArgument>(v);
            return barg && barg.getOwner() == body && barg.getArgNumber() > 0;
          },
          [&](Value tracedVal) {
            if (Operation *def = tracedVal.getDefiningOp()) {
              LLVM_DEBUG({
                llvm::dbgs() << "Adding op to neededOps during for traceback: ";
                def->print(llvm::dbgs(), OpPrintingFlags()
                                             .printGenericOpForm()
                                             .elideLargeElementsAttrs());
                llvm::dbgs() << "\n";
              });
              neededOps.insert(def);
            }
          });
    };

    // Trace backwards to discover live loop-carried channels and defining ops.
    // Newly-kept channels make their tied yield values live, so keep tracing
    // until all kept yielded values have contributed to neededOps.
    llvm::SmallBitVector rootedChannels(numResults, false);
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Value> tracedIterArgs = traceForRoots();
      for (Value iterArgVal : tracedIterArgs) {
        auto barg = cast<BlockArgument>(iterArgVal);
        unsigned idx = barg.getArgNumber() - 1;
        if (idx >= numResults) {
          assert(false && "iter arg index out of range");
          continue;
        }
        if (!keep.test(idx)) {
          keep.set(idx);
          changed = true;
        }
      }
      for (unsigned i = 0; i < numResults; ++i) {
        if (!keep.test(i) || rootedChannels.test(i))
          continue;
        rootVals.push_back(oldYield.getOperand(i));
        rootedChannels.set(i);
        changed = true;
      }
    }

    // If every channel is live, this rewrite does not apply.
    SmallVector<unsigned, 4> removableIdxs;
    for (unsigned i = 0; i < numResults; ++i)
      if (!keep.test(i))
        removableIdxs.push_back(i);
    if (removableIdxs.empty())
      return failure();

    // Validate kept yielded values are reconstructable before mutation.
    for (unsigned i = 0; i < numResults; ++i) {
      if (!keep.test(i))
        continue;
      Value oldOperand = oldYield.getOperand(i);
      if (!isInLoopBody(oldOperand, body))
        continue;
      if (auto barg = dyn_cast<BlockArgument>(oldOperand)) {
        if (barg.getArgNumber() == 0)
          continue;
        unsigned idx = barg.getArgNumber() - 1;
        if (idx < numResults && keep.test(idx))
          continue;
        assert(false && "unmapped kept yield block argument");
        return failure();
      }
      Operation *def = oldOperand.getDefiningOp();
      if (!def || def->getBlock() != body || !neededOps.contains(def)) {
        assert(false && "unmapped kept yield operand from old loop body");
        return failure();
      }
    }

    // Rebuild loop with compacted channels, recursively updating nested SCF.
    IRMapping topMapping;
    auto rebuiltOr = rebuildForOpDroppingIterArgs(rewriter, forOp, keep,
                                                  neededOps, topMapping);
    if (failed(rebuiltOr)) {
      debugUnexpectedBuilderFailure(
          "RemoveDeadIterArgBackwardForPattern", forOp, &keep,
          "top-level scf.for rebuild failed unexpectedly");
      return failure();
    }
    // Redirect uses from old kept results to rebuilt results.
    for (unsigned i = 0; i < numResults; ++i) {
      if (!keep.test(i))
        continue;
      Value mapped = topMapping.lookupOrNull(forOp.getResult(i));
      if (!mapped) {
        assert(false && "missing mapped result in rebuilt for loop");
        return failure();
      }
      forOp.getResult(i).replaceAllUsesWith(mapped);
    }
    rewriter.eraseOp(forOp);
    return success();
  }
};

// Backward-trace dead-iter-arg removal for scf.while.
//
// Same algorithm shape as the for-loop variant, with two extra wrinkles:
//   - The scf.condition predicate controls termination, so values feeding it
//     are roots even if the corresponding while result is unused.
//   - Both the before and after regions can host side-effecting ops, so the
//     side-effect collection runs on each region.
struct RemoveDeadIterArgBackwardWhilePattern
    : public OpRewritePattern<scf::WhileOp> {
public:
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(scf::WhileOp whileOp,
                  mlir::PatternRewriter &rewriter) const override {
    // Empty carried state means nothing to prune.
    const unsigned numResults = whileOp.getNumResults();
    if (numResults == 0)
      return failure();

    Block *beforeBody = whileOp.getBeforeBody();
    Block *afterBody = whileOp.getAfterBody();
    auto oldCond = whileOp.getConditionOp();
    auto oldYield = whileOp.getYieldOp();
    if (!beforeBody || !afterBody || !oldCond || !oldYield)
      return failure();

    llvm::SmallBitVector keep(numResults, false);
    SmallPtrSet<Operation *, 32> neededOps;
    SmallVector<Value, 32> rootVals;
    seedLiveRootsFromResults(whileOp.getResults(), keep, rootVals);
    collectRootsFromSideEffects(beforeBody, neededOps, rootVals);
    collectRootsFromSideEffects(afterBody, neededOps, rootVals);
    // The scf.condition predicate controls loop execution, so values used to
    // compute it are live even when the corresponding while result is unused.
    rootVals.push_back(oldCond.getCondition());

    auto traceWhileRoots = [&]() {
      return tracebackMemVals(
          rootVals,
          [&](Value v) {
            auto barg = dyn_cast<BlockArgument>(v);
            if (!barg)
              return false;
            return (barg.getOwner() == beforeBody ||
                    barg.getOwner() == afterBody) &&
                   barg.getArgNumber() < numResults;
          },
          [&](Value tracedVal) {
            if (Operation *def = tracedVal.getDefiningOp())
              neededOps.insert(def);
          });
    };

    // Trace backward through both regions to recover live carried channels.
    // Newly-kept channels make their tied condition/yield values live, so keep
    // tracing until the keep mask stops changing.
    llvm::SmallBitVector rootedChannels(numResults, false);
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Value> tracedArgs = traceWhileRoots();
      for (Value argVal : tracedArgs) {
        auto barg = cast<BlockArgument>(argVal);
        unsigned idx = barg.getArgNumber();
        if (!keep.test(idx)) {
          keep.set(idx);
          changed = true;
        }
      }
      for (unsigned i = 0; i < numResults; ++i) {
        if (!keep.test(i) || rootedChannels.test(i))
          continue;
        rootVals.push_back(oldCond.getArgs()[i]);
        rootVals.push_back(oldYield.getOperand(i));
        rootedChannels.set(i);
        changed = true;
      }
    }

    // If every channel is live, this rewrite does not apply.
    SmallVector<unsigned, 4> removableIdxs;
    for (unsigned i = 0; i < numResults; ++i)
      if (!keep.test(i))
        removableIdxs.push_back(i);
    if (removableIdxs.empty())
      return failure();

    // Helper to check if an old value can be materialized in rebuilt regions.
    auto isMappable = [&](Value oldVal) -> bool {
      if (!isInLoopBody(oldVal, beforeBody) && !isInLoopBody(oldVal, afterBody))
        return true;
      if (auto barg = dyn_cast<BlockArgument>(oldVal))
        return barg.getArgNumber() < numResults &&
               keep.test(barg.getArgNumber());
      if (Operation *def = oldVal.getDefiningOp())
        return neededOps.contains(def);
      return false;
    };
    // Validate both condition args and yielded args for kept channels.
    for (unsigned i = 0; i < numResults; ++i) {
      if (!keep.test(i))
        continue;
      if (!isMappable(oldCond.getArgs()[i]) ||
          !isMappable(oldYield.getOperand(i))) {
        assert(false && "unmapped kept while channel value");
        return failure();
      }
    }

    // Rebuild while with compacted channels, recursively updating nested SCF.
    IRMapping topMapping;
    auto rebuiltOr = rebuildWhileOpDroppingIterArgs(rewriter, whileOp, keep,
                                                    neededOps, topMapping);
    if (failed(rebuiltOr)) {
      debugUnexpectedBuilderFailure(
          "RemoveDeadIterArgBackwardWhilePattern", whileOp, &keep,
          "top-level scf.while rebuild failed unexpectedly");
      return failure();
    }
    // Redirect uses from old kept results to rebuilt results.
    for (unsigned i = 0; i < numResults; ++i) {
      if (!keep.test(i))
        continue;
      Value mapped = topMapping.lookupOrNull(whileOp.getResult(i));
      if (!mapped) {
        assert(false && "missing mapped result in rebuilt while");
        return failure();
      }
      whileOp.getResult(i).replaceAllUsesWith(mapped);
    }
    rewriter.eraseOp(whileOp);
    return success();
  }
};

struct CanonicalizeIterArgPass
    : public impl::CanonicalizeIterArgBase<CanonicalizeIterArgPass> {
  using Base::Base;
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    tensor::populateFoldTensorEmptyPatterns(patterns);

    patterns.insert<CanonicalizeYieldValPattern<scf::ForOp>,
                    CanonicalizeIterArgPattern<scf::ForOp>,
                    CanonicalizeIterArgPattern<scf::WhileOp>,
                    RemoveDeadIterArgPattern<scf::ForOp>>(
        patterns.getContext());

    if (enableRemoveDeadIterArgBackwardPatterns) {
      // Vector-side functions are excluded: their later passes assume the
      // simpler use-driven removal semantics, and the backward rebuild has
      // been observed to interact badly with vector-only rewrites.
      if (!funcOp->hasAttr(hivm::VectorFunctionAttr::name)) {
        patterns.insert<RemoveDeadIterArgBackwardForPattern,
                        RemoveDeadIterArgBackwardWhilePattern>(
            patterns.getContext());
      }
    }

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> scf::createCanonicalizeIterArgPass() {
  return std::make_unique<CanonicalizeIterArgPass>();
}
