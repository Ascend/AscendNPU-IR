//===- IntraStageScheduling.cpp - Schedule inside CV scopes -------------===//
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
//
// Scheduling inside one CV pipeline stage, after MergeCVScopes. The pass is a
// driver over a list of independent rules: it walks the scopes of every
// pipelined loop and offers each scope to every rule in turn.
//
// To add a rule:
//
//  1. Derive from `SchedulingRule` in the rules section below. `matches`
//     selects the scopes the rule cares about, `apply` rewrites one of them
//     and reports whether the IR changed.
//  2. Register it in `populateSchedulingRules`. Rules run in registration
//     order, so place a rule after the ones whose output it builds on.
//  3. Reuse the helpers section rather than re-deriving dependences:
//     `findNextScopeOnCore` locates the neighbouring core scope,
//     `addBackwardSlice` and `addCrossCoreCopyWriters` build a
//     dependence-closed operation set, `canHoistGroup` checks that the set may
//     move ahead of the rest of the body, and `createLeadingGroupScope`
//     performs the move and rewires the results.
//
// A rule sees each scope that existed when the walk started, once. Scopes a
// rule creates are not offered to any rule, and no rule is retried after
// another one changes the IR, so a rule must not depend on being run to a
// fixed point.
//
//===----------------------------------------------------------------------===//

#include "Util.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "intra-stage-scheduling"

namespace mlir {
namespace hivm {
#define GEN_PASS_DEF_INTRASTAGESCHEDULING
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace hivm
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

//===----------------------------------------------------------------------===//
// Helpers shared by the scheduling rules
//===----------------------------------------------------------------------===//

/// Return the first scope after `scope` in the same block that runs on
/// `coreType`, or null when the stage has none.
ScopeOp findNextScopeOnCore(ScopeOp scope, TCoreType coreType) {
  for (Operation *next = scope->getNextNode(); next; next = next->getNextNode())
    if (auto candidate = dyn_cast<ScopeOp>(next))
      if (getScopeCoreType(candidate) == coreType)
        return candidate;
  return nullptr;
}

/// Return the ancestor of `op` that is an immediate child of `block`, or null
/// when `op` lies outside `block`.
Operation *getAncestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

/// Return the indices of the results of `scope` that `consumer` reads.
llvm::SmallVector<unsigned> getResultsReadBy(ScopeOp scope, ScopeOp consumer) {
  llvm::SmallVector<unsigned> indices;
  for (auto [idx, result] : llvm::enumerate(scope.getResults()))
    if (llvm::any_of(result.getUsers(), [&](Operation *user) {
          return consumer->isAncestor(user);
        }))
      indices.push_back(static_cast<unsigned>(idx));
  return indices;
}

/// Return true when every use of every result of `scope` lies inside
/// `consumer`.
bool feedsOnly(ScopeOp scope, ScopeOp consumer) {
  return llvm::all_of(scope.getResults(), [&](Value result) {
    return llvm::all_of(result.getUsers(), [&](Operation *user) {
      return consumer->isAncestor(user);
    });
  });
}

/// Add `op` and the operations of `body` it transitively depends on to
/// `slice`. Operands captured by nested regions are followed as well, and
/// definitions outside `body` are ignored: they stay valid wherever inside
/// `body` the slice is placed.
void addBackwardSlice(Operation *op, Block *body,
                      llvm::SetVector<Operation *> &slice) {
  SmallVector<Operation *> worklist;
  auto push = [&](Operation *candidate) {
    Operation *inBody = getAncestorInBlock(candidate, body);
    if (inBody && slice.insert(inBody))
      worklist.push_back(inBody);
  };

  push(op);
  while (!worklist.empty()) {
    worklist.pop_back_val()->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        push(operand.getDefiningOp());
    });
  }
}

/// Add the cross-core copies that fill the L1 buffers already in `slice`,
/// along with their own dependencies.
///
/// A cube scope reads the tensor view of an L1 buffer, not the copy that
/// writes it. That copy is therefore not a data-flow predecessor of the value
/// the cube consumes, yet the cube depends on it, so it has to be pulled in
/// through the buffer it writes.
void addCrossCoreCopyWriters(Block *body, llvm::SetVector<Operation *> &slice) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : *body) {
      if (slice.contains(&op) || !isCrossCoreCopy(&op))
        continue;
      memref::AllocOp alloc = traceAlloc(cast<CopyOp>(&op).getDst());
      if (!alloc || !slice.contains(alloc.getOperation()))
        continue;
      addBackwardSlice(&op, body, slice);
      changed = true;
    }
  }
}

/// Classify the buffers `root` and its nested operations touch: set `foreign`
/// when one of them is allocated outside `body`, and record the ones allocated
/// inside `body` in `locals`.
void collectTouchedBuffers(Operation *root, Block *body, bool &foreign,
                           llvm::DenseSet<Operation *> &locals) {
  root->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!isa<BaseMemRefType>(operand.getType()))
        continue;
      memref::AllocOp alloc = traceAlloc(operand);
      if (alloc && getAncestorInBlock(alloc.getOperation(), body))
        locals.insert(alloc.getOperation());
      else
        foreign = true;
    }
  });
}

/// Return true when `slice` can run ahead of the operations of `body` that
/// stay behind.
///
/// Values are in SSA form, so the move preserves data flow, but memory is not:
/// a buffer the group reads could be written by an operation it now runs
/// before. Two buffers are known not to overlap when both are allocated in
/// `body` and the allocations differ, which is the case for the cross-core
/// buffer a group allocates for itself. Anything reaching the scope from the
/// outside is assumed to overlap.
///
/// TODO: Replace this with a real alias query once one is available here, so
/// that groups loading from global memory can be hoisted as well.
bool canHoistGroup(Block *body, const llvm::SetVector<Operation *> &slice) {
  bool groupTouchesForeign = false;
  llvm::DenseSet<Operation *> groupBuffers;
  for (Operation *op : slice)
    collectTouchedBuffers(op, body, groupTouchesForeign, groupBuffers);

  for (Operation &op : *body) {
    if (slice.contains(&op))
      continue;
    bool touchesForeign = false;
    llvm::DenseSet<Operation *> buffers;
    collectTouchedBuffers(&op, body, touchesForeign, buffers);
    if (groupTouchesForeign && touchesForeign)
      return false;
    if (llvm::any_of(buffers, [&](Operation *alloc) {
          return groupBuffers.contains(alloc);
        }))
      return false;
  }
  return true;
}

/// Move `ops` into one new `scope.scope` at the beginning of `parent`'s body
/// and return that scope.
///
/// `ops` must be closed under its operands within the body, which
/// `addBackwardSlice` guarantees, so that every operand it does not define
/// itself comes from outside the body and dominates the new position. Callers
/// decide whether the move is legal with respect to memory; `canHoistGroup`
/// answers that. The values the rest of the body still reads become the
/// group's results and their uses are redirected, so the body keeps working on
/// the same values.
///
/// The group inherits `parent`'s core type and carries `vf-scope`, which marks
/// it as a scheduling group rather than one of the scopes OutlineCVScopes and
/// MergeCVScopes produce.
ScopeOp createLeadingGroupScope(ScopeOp parent,
                                const llvm::SetVector<Operation *> &ops) {
  Block *body = &parent.getRegion().front();

  // Keep the original relative order inside the group.
  SmallVector<Operation *> grouped;
  for (Operation &op : *body)
    if (ops.contains(&op))
      grouped.push_back(&op);

  SmallVector<Value> returns;
  SmallVector<SmallVector<OpOperand *>> externalUses;
  for (Operation *op : grouped) {
    for (Value result : op->getResults()) {
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : result.getUses()) {
        Operation *owner = getAncestorInBlock(use.getOwner(), body);
        if (!owner || !ops.contains(owner))
          uses.push_back(&use);
      }
      if (!uses.empty()) {
        returns.push_back(result);
        externalUses.push_back(std::move(uses));
      }
    }
  }

  OpBuilder builder(body, body->begin());
  auto group =
      builder.create<ScopeOp>(parent.getLoc(), TypeRange(ValueRange(returns)));
  if (Attribute coreType = parent->getAttr(kPipelinedLoopCoreTypeAttrName))
    group->setAttr(kPipelinedLoopCoreTypeAttrName, coreType);
  group->setAttr(kVFScopeAttrName, builder.getUnitAttr());
  Block *groupBody = builder.createBlock(&group.getRegion());
  for (Operation *op : grouped)
    op->moveBefore(groupBody, groupBody->end());
  builder.setInsertionPointToEnd(groupBody);
  builder.create<scope::ReturnOp>(group.getLoc(), returns);
  for (auto [result, uses] : llvm::zip(group.getResults(), externalUses))
    for (OpOperand *use : uses)
      use->set(result);
  return group;
}

//===----------------------------------------------------------------------===//
// Scheduling rule interface
//===----------------------------------------------------------------------===//

/// One intra-stage scheduling transformation.
class SchedulingRule {
public:
  virtual ~SchedulingRule() = default;

  /// Name of the rule, used in debug output.
  virtual StringRef getName() const = 0;

  /// Return true when `scope` is the kind of scope this rule rewrites. Keep it
  /// cheap: it runs for every scope of every pipelined loop.
  virtual bool matches(ScopeOp scope) const = 0;

  /// Rewrite `scope`, which `matches` accepted. Return true when the IR
  /// changed. Returning false must leave `scope` untouched.
  ///
  /// `scope` must still be alive when this returns, because the driver offers
  /// it to the remaining rules. A rule that would replace it has to rewrite it
  /// in place instead.
  virtual bool apply(ScopeOp scope) const = 0;
};

using SchedulingRuleList = SmallVector<std::unique_ptr<SchedulingRule>>;

//===----------------------------------------------------------------------===//
// Rule: hoist the dependencies of the next cube scope
//===----------------------------------------------------------------------===//

/// Group the operations a vector scope prepares for the next cube scope into a
/// leading sub-scope, so the cube waits for that group instead of the whole
/// vector scope.
class HoistCubeDependency : public SchedulingRule {
public:
  StringRef getName() const override { return "hoist-cube-dependency"; }

  bool matches(ScopeOp scope) const override {
    return getScopeCoreType(scope) == TCoreType::VECTOR;
  }

  bool apply(ScopeOp vf) const override {
    // A scope that returns nothing hands no dependency to a cube scope.
    if (vf.getNumResults() == 0)
      return false;

    ScopeOp cube = findNextScopeOnCore(vf, TCoreType::CUBE);
    if (!cube)
      return false;

    llvm::SmallVector<unsigned> readResults = getResultsReadBy(vf, cube);
    if (readResults.empty())
      return false;

    // The whole scope already serves the cube: there is nothing to run after
    // the group, so hoisting it buys no overlap.
    if (feedsOnly(vf, cube))
      return false;

    Block *body = &vf.getRegion().front();
    auto returnOp = cast<scope::ReturnOp>(body->getTerminator());

    llvm::SetVector<Operation *> slice;
    for (unsigned idx : readResults)
      addBackwardSlice(returnOp->getOperand(idx).getDefiningOp(), body, slice);
    addCrossCoreCopyWriters(body, slice);
    slice.remove(returnOp.getOperation());
    if (slice.empty() || slice.size() == body->getOperations().size() - 1)
      return false;
    if (!canHoistGroup(body, slice))
      return false;

    createLeadingGroupScope(vf, slice);
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Rule registry
//===----------------------------------------------------------------------===//

/// Build the rules the pass runs, in the order they run.
void populateSchedulingRules(SchedulingRuleList &rules) {
  rules.push_back(std::make_unique<HoistCubeDependency>());
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct IntraStageSchedulingPass
    : mlir::hivm::impl::IntraStageSchedulingBase<IntraStageSchedulingPass> {
  void runOnOperation() override {
    SchedulingRuleList rules;
    populateSchedulingRules(rules);
    if (rules.empty())
      return;

    for (LoopLikeOpInterface loop : collectPipelinedLoops()) {
      // Collect before rewriting: a rule inserts scopes into the bodies being
      // walked.
      SmallVector<ScopeOp> scopes;
      loop->walk([&](ScopeOp scope) { scopes.push_back(scope); });

      for (ScopeOp scope : scopes) {
        for (const std::unique_ptr<SchedulingRule> &rule : rules) {
          if (!rule->matches(scope))
            continue;
          if (rule->apply(scope))
            LLVM_DEBUG(llvm::dbgs()
                       << "[" DEBUG_TYPE "] applied " << rule->getName()
                       << " to scope at " << scope.getLoc() << '\n');
        }
      }
    }
  }

private:
  /// Return the loops PipelineV2 has staged, outermost first, without
  /// descending into a loop that is already collected.
  SmallVector<LoopLikeOpInterface> collectPipelinedLoops() {
    SmallVector<LoopLikeOpInterface> loops;
    getOperation().walk<WalkOrder::PreOrder>([&](LoopLikeOpInterface loop) {
      // TODO: Temporary transition measure: skip loops already pipelined by
      // legacy CVPipelining.
      if (loop->hasAttr(LegacyCVPipelineAttr::name))
        return WalkResult::skip();
      if (!loop->hasAttr(CVPipelinedLoopAttr::name))
        return WalkResult::advance();
      loops.push_back(loop);
      return WalkResult::skip();
    });
    return loops;
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createIntraStageSchedulingPass() {
  return std::make_unique<IntraStageSchedulingPass>();
}
