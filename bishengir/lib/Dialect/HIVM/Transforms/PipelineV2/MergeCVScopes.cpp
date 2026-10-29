//===- MergeCVScopes.cpp - Merge CV Scopes for CV Pipelining V2 -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Util.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "merge-cv-scopes"

namespace mlir {
namespace hivm {

#define GEN_PASS_DEF_MERGECVSCOPES
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

namespace {

class LoopScopeMerger {
public:
  using MergeRuleFn = void (LoopScopeMerger::*)();

  explicit LoopScopeMerger(scf::ForOp loopOp) : loopOp(loopOp) {}

  void run(ArrayRef<MergeRuleFn> rules);

  // Rule 1: Starting from cross-core transfers (fixpipe -> UB or copy -> CBUF),
  // traverse backward along intra-iteration scope dependencies and merge
  // compatible scopes, stopping traversal at any scope that has an incompatible
  // dependency.
  void applyCrossCoreTransferBackwardRule();

private:
  static bool isCrossCoreTransferOp(Operation *op);
  static bool hasCrossCoreTransfer(ScopeOp scope);

  SetVector<ScopeOp> getIntraIterScopeDeps(ScopeOp scope);
  ScopeOp mergeScopeGroup(SmallPtrSetImpl<Operation *> &group);

  scf::ForOp loopOp;
};

struct MergeCVScopesPass : public impl::MergeCVScopesBase<MergeCVScopesPass> {
protected:
  void runOnOperation() override;
};

} // namespace

//===----------------------------------------------------------------------===//
// LoopScopeMerger Implementation
//===----------------------------------------------------------------------===//

bool LoopScopeMerger::isCrossCoreTransferOp(Operation *op) {
  if (auto fixpipe = dyn_cast_or_null<FixpipeOp>(op)) {
    auto space = GetBufferSpaceAttr(fixpipe.getDst());
    return space && space->getAddressSpace() == AddressSpace::UB;
  }
  if (auto copyOp = dyn_cast_or_null<CopyOp>(op)) {
    auto space = GetBufferSpaceAttr(copyOp.getDst());
    return space && space->getAddressSpace() == AddressSpace::L1;
  }
  return false;
}

bool LoopScopeMerger::hasCrossCoreTransfer(ScopeOp scope) {
  return scope
      ->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op != scope.getOperation() && isa<ScopeOp>(op))
          return WalkResult::skip();
        return isCrossCoreTransferOp(op) ? WalkResult::interrupt()
                                         : WalkResult::advance();
      })
      .wasInterrupted();
}

SetVector<ScopeOp> LoopScopeMerger::getIntraIterScopeDeps(ScopeOp scope) {
  SetVector<ScopeOp> deps;
  Block *parentBlock = scope->getBlock();
  if (!parentBlock)
    return deps;
  scope.walk([&](Operation *op) {
    if (op == scope.getOperation())
      return;
    for (Value opnd : op->getOperands()) {
      Operation *defOp = opnd.getDefiningOp();
      while (defOp && defOp->getBlock() != parentBlock)
        defOp = defOp->getParentOp();
      if (auto depScope = dyn_cast_or_null<ScopeOp>(defOp);
          depScope && depScope != scope && depScope->getBlock() == parentBlock)
        deps.insert(depScope);
    }
  });
  return deps;
}

ScopeOp LoopScopeMerger::mergeScopeGroup(SmallPtrSetImpl<Operation *> &group) {
  if (group.size() < 2)
    return group.empty() ? ScopeOp() : dyn_cast<ScopeOp>(*group.begin());

  Block *parentBlock = (*group.begin())->getBlock();
  if (!parentBlock || llvm::any_of(group, [&](Operation *op) {
        return op->getBlock() != parentBlock;
      }))
    return ScopeOp();

  Operation *terminator = parentBlock->getTerminator();

  DenseMap<Operation *, SmallVector<Operation *>> opDeps;
  DenseMap<Operation *, SmallVector<Operation *>> opUsers;
  for (Operation &topOp : *parentBlock) {
    if (&topOp == terminator)
      continue;
    SmallPtrSet<Operation *, 8> seenDeps;
    topOp.walk([&](Operation *nested) {
      for (Value opnd : nested->getOperands()) {
        Operation *def = opnd.getDefiningOp();
        while (def && def->getBlock() != parentBlock)
          def = def->getParentOp();
        if (def && def != &topOp && seenDeps.insert(def).second) {
          opDeps[&topOp].push_back(def);
          opUsers[def].push_back(&topOp);
        }
      }
    });
  }

  auto collectExternalReachable =
      [&](const DenseMap<Operation *, SmallVector<Operation *>> &adj) {
        SmallPtrSet<Operation *, 16> visited;
        SmallVector<Operation *> stack;
        for (Operation *mOp : group) {
          for (Operation *next : adj.lookup(mOp)) {
            if (!group.contains(next) && visited.insert(next).second)
              stack.push_back(next);
          }
        }
        while (!stack.empty()) {
          Operation *curr = stack.pop_back_val();
          for (Operation *next : adj.lookup(curr)) {
            if (!group.contains(next) && visited.insert(next).second)
              stack.push_back(next);
          }
        }
        return visited;
      };

  SmallPtrSet<Operation *, 16> extAncestors;
  SmallPtrSet<Operation *, 16> extDescendants;
  while (group.size() >= 2) {
    extAncestors = collectExternalReachable(opDeps);
    extDescendants = collectExternalReachable(opUsers);

    SmallPtrSet<Operation *, 16> cycleExternal;
    for (Operation *u : extAncestors) {
      if (extDescendants.contains(u))
        cycleExternal.insert(u);
    }
    if (cycleExternal.empty())
      break;

    // Prune any non-tail scope in `group` that reaches `cycleExternal` so
    // merging cannot introduce a cycle in the parent block DAG.
    Operation *lastInGroup = nullptr;
    for (Operation &op : *parentBlock) {
      if (group.contains(&op))
        lastInGroup = &op;
    }
    SmallPtrSet<Operation *, 16> canReachCycle = cycleExternal;
    SmallVector<Operation *> revStack(cycleExternal.begin(),
                                      cycleExternal.end());
    while (!revStack.empty()) {
      Operation *curr = revStack.pop_back_val();
      for (Operation *pred : opDeps.lookup(curr)) {
        if (canReachCycle.insert(pred).second)
          revStack.push_back(pred);
      }
    }
    bool removed = false;
    for (Operation &op : *parentBlock) {
      if (&op != lastInGroup && group.contains(&op) &&
          canReachCycle.contains(&op)) {
        group.erase(&op);
        removed = true;
      }
    }
    if (!removed)
      return ScopeOp();
  }

  SmallVector<ScopeOp> orderedScopes;
  for (ScopeOp s : parentBlock->getOps<ScopeOp>()) {
    if (group.contains(s.getOperation()))
      orderedScopes.push_back(s);
  }
  if (orderedScopes.size() < 2)
    return orderedScopes.empty() ? ScopeOp() : orderedScopes.front();

  Operation *earliestDescendant = nullptr;
  for (Operation &op : *parentBlock) {
    if (extDescendants.contains(&op)) {
      earliestDescendant = &op;
      break;
    }
  }

  if (earliestDescendant) {
    SmallVector<Operation *> ancestorsToHoist;
    for (Operation &op : *parentBlock) {
      if (extAncestors.contains(&op) &&
          earliestDescendant->isBeforeInBlock(&op))
        ancestorsToHoist.push_back(&op);
    }
    for (Operation *anc : ancestorsToHoist)
      anc->moveBefore(earliestDescendant);
  }

  Operation *afterLast = orderedScopes.back()->getNextNode();
  Operation *insertPt =
      (earliestDescendant && earliestDescendant->isBeforeInBlock(afterLast))
          ? earliestDescendant
          : afterLast;

  for (ScopeOp s : orderedScopes) {
    if (s.getOperation() != insertPt)
      s->moveBefore(insertPt);
  }

  ScopeOp merged = orderedScopes.front();
  for (ScopeOp next : llvm::drop_begin(orderedScopes)) {
    ScopeOp nextMerged = mergeScopes(merged, next, next);
    if (!nextMerged)
      return ScopeOp();
    merged = nextMerged;
  }
  return merged;
}

void LoopScopeMerger::applyCrossCoreTransferBackwardRule() {
  DenseSet<Operation *> processedSeeds;
  while (true) {
    ScopeOp seed;
    loopOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op != loopOp.getOperation() && isa<scf::ForOp>(op))
        return WalkResult::skip();
      if (auto scope = dyn_cast<ScopeOp>(op)) {
        if (!processedSeeds.contains(scope.getOperation()) &&
            hasCrossCoreTransfer(scope)) {
          seed = scope;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (!seed)
      break;

    processedSeeds.insert(seed.getOperation());
    Block *seedBlock = seed->getBlock();
    auto seedCore = getScopeCoreType(seed);
    if (!seedBlock || !seedCore)
      continue;
    TCoreType targetCore = *seedCore;
    SmallPtrSet<Operation *, 16> mergeSet = {seed.getOperation()};
    SmallVector<ScopeOp> worklist = {seed};

    while (!worklist.empty()) {
      ScopeOp curr = worklist.pop_back_val();
      SetVector<ScopeOp> deps = getIntraIterScopeDeps(curr);

      bool hasIncompatibleDep = llvm::any_of(deps, [&](ScopeOp dep) {
        if (dep->getBlock() != seedBlock)
          return false;
        auto depCore = getScopeCoreType(dep);
        return depCore && !areCompatibleCoreTypes(targetCore, *depCore);
      });
      if (hasIncompatibleDep)
        continue;

      for (ScopeOp dep : deps) {
        if (dep->getBlock() != seedBlock)
          continue;
        auto depCore = getScopeCoreType(dep);
        if (!depCore)
          continue;
        FailureOr<TCoreType> newTarget =
            resolveMergedCoreType(targetCore, *depCore);
        if (failed(newTarget))
          continue;
        targetCore = *newTarget;
        if (mergeSet.insert(dep.getOperation()).second)
          worklist.push_back(dep);
      }
    }

    if (mergeSet.size() > 1) {
      if (ScopeOp merged = mergeScopeGroup(mergeSet))
        processedSeeds.insert(merged.getOperation());
    }
  }
}

void LoopScopeMerger::run(ArrayRef<MergeRuleFn> rules) {
  for (MergeRuleFn rule : rules)
    (this->*rule)();
}

//===----------------------------------------------------------------------===//
// MergeCVScopesPass Implementation
//===----------------------------------------------------------------------===//

void MergeCVScopesPass::runOnOperation() {
  static constexpr LoopScopeMerger::MergeRuleFn kDefaultRuleOrder[] = {
      &LoopScopeMerger::applyCrossCoreTransferBackwardRule,
  };

  SmallVector<scf::ForOp> targetLoops;
  getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
    bool hasScopeInLoop =
        forOp
            ->walk<WalkOrder::PreOrder>([&](Operation *op) {
              if (op != forOp.getOperation() && isa<scf::ForOp>(op))
                return WalkResult::skip();
              return isa<ScopeOp>(op) ? WalkResult::interrupt()
                                      : WalkResult::advance();
            })
            .wasInterrupted();
    if (forOp->hasAttr(CVPipelinedLoopAttr::name) || hasScopeInLoop)
      targetLoops.push_back(forOp);
  });

  for (scf::ForOp forOp : targetLoops) {
    LoopScopeMerger merger(forOp);
    merger.run(kDefaultRuleOrder);
  }

  LLVM_DEBUG({
    for (scf::ForOp forOp : targetLoops)
      writeScopeDependenceGraphToDotFile(forOp, "merged");
  });
}

std::unique_ptr<Pass> createMergeCVScopesPass() {
  return std::make_unique<MergeCVScopesPass>();
}

} // namespace hivm
} // namespace mlir
