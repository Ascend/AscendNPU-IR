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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <functional>
#include <utility>

#define DEBUG_TYPE "merge-cv-scopes"

namespace mlir {
namespace hivm {

#define GEN_PASS_DEF_MERGECVSCOPES
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

namespace {

class ScopeDependenceGraph {
public:
  explicit ScopeDependenceGraph(scf::ForOp loop, bool withCarriedEdges = true);
  explicit ScopeDependenceGraph(Block *body);
  unsigned getSize() const { return nodes.size(); }
  Operation *getNode(unsigned i) const { return nodes[i]; }
  std::optional<unsigned> getIndex(Operation *op) const;
  ArrayRef<unsigned> getSuccessors(unsigned i) const { return intra[i]; }
  ArrayRef<unsigned> getPredecessors(unsigned i) const {
    return predecessors[i];
  }
  SmallVector<SmallVector<Operation *>> formGroups() const;
  void printSummary(llvm::raw_ostream &os,
                    ArrayRef<SmallVector<Operation *>> groups) const;

  struct MemAccess {
    bool read = false;
    bool write = false;
  };

  struct NodeMemInfo {
    llvm::MapVector<Value, MemAccess> buffers;
    bool globalRead = false;
    bool globalWrite = false;
  };

  static void collectMemAccesses(Operation *top, NodeMemInfo &info);
  static bool hasMemConflict(const NodeMemInfo &a, const NodeMemInfo &b);

private:
  struct CarriedEdge {
    unsigned from;
    unsigned to;
    unsigned argIndex;
  };

  static Value getMemBase(Value v);
  static void recordMemref(NodeMemInfo &info, Value v, bool isWrite);
  static SmallVector<SmallVector<unsigned>>
  computeSCCs(ArrayRef<SmallVector<unsigned>> succ);

  void buildIntraEdges();
  void buildCarriedEdges();

  scf::ForOp loop;
  Block *body;
  SmallVector<Operation *> nodes;
  DenseMap<Operation *, unsigned> index;
  SmallVector<SmallVector<unsigned>> intra;
  SmallVector<SmallVector<unsigned>> predecessors;
  SmallVector<CarriedEdge> carried;
};

class LoopScopeMerger {
public:
  using MergeRuleFn = bool (LoopScopeMerger::*)();

  struct MergeRule {
    MergeRuleFn fn;
    StringLiteral dotPrefix;
  };

  explicit LoopScopeMerger(scf::ForOp loopOp) : loopOp(loopOp) {}

  void run(ArrayRef<MergeRule> rules);
  scf::ForOp getLoopOp() const { return loopOp; }

  // Rule 0: Clone/merge scalar-only `CUBE_AND_VECTOR` scopes that generate
  // loop-carried-dependency results used by other `CUBE` or `VECTOR` scopes
  // into every user scope, giving each merged scope its own loop `iter_arg`
  // and erasing the original scalar-only scope.
  bool applyScalarLoopCarriedDepRule();

  // Rule 1: Group scopes connected by loop-carried dependencies into an outer
  // `{cv_merged_scope}` wrapper scope, keeping member scopes nested inside so
  // cross-core boundaries remain visible.
  bool applyLoopCarriedDepGroupingRule();

  // Rule 2: Within each `{cv_merged_scope}`, merge compatible scopes along
  // intra-iteration dependencies, and if the `{cv_merged_scope}` reduces to a
  // single inner scope, collapse it into a single work item.
  bool applyMergedScopeSameCoreRule();

  // Rule 3: Starting from cross-core transfers (fixpipe -> UB or copy -> CBUF),
  // traverse backward along intra-iteration scope dependencies within the same
  // parent block and merge compatible scopes, stopping traversal at any scope
  // that has an incompatible dependency.
  bool applyCrossCoreTransferBackwardRule();

  // Rule 4: Merge small (< 3 CoreOps) `VECTOR` scopes that have only
  // (1 input, 0 output) or (0 input, 1 output) into their connected `VECTOR`
  // scope.
  bool applySmallVectorScopeMergeRule();

  // Rule 5: Merge same-core scopes connected by an intra-iteration edge or
  // sharing a predecessor or successor in the loop body or a nested scope body.
  bool applyGraphSameCoreRule();

  // Rule 6: Wrap a consecutive run of scopes that resolve to the same single
  // core into one `{cv_merged_scope}`. A `{cv_merged_scope}` whose nested
  // scopes all have that one core type counts as a scope of that core.
  // Same-core scopes inside that wrapper are then fused in program order.
  // A different core, a mixed wrapper, or a non-scope op ends the run.
  bool applySameCoreRunGroupingRule();

  // Rule 7: Starting from a Cube-and-Vector scope with an alloca inside,
  // trace downward to find its users and sink the alloca, initialization, and
  // loads into the user cube and/or vector scopes.
  bool applySinkAllocaInCAVScopeRule();

  // Run Rule 6 early for normalized matmul counters.
  bool applySinkNormalizedCounterInCAVScopeRule();

private:
  static bool isScalarOnlyCubeAndVectorScope(ScopeOp scope);
  struct AllocaAccesses {
    std::pair<ScopeOp, Value> initialLoad;
    std::pair<memref::StoreOp, ScopeOp> initialStore;
  };

  static bool isCrossCoreTransferOp(Operation *op);
  static bool hasCrossCoreTransfer(ScopeOp scope);
  static bool isEligibleForCoreMerge(ScopeOp scope);
  static bool collapseSingleChildMergedScope(ScopeOp outerScope);
  // Fuse direct children that have the same Cube or Vector core, in order.
  // A single-core `{cv_merged_scope}` is fused internally and collapsed first
  // so it can join that sequence.
  static void fuseOrderedSameCoreScopes(Block *body);
  static void collectAllocaScopes(scf::ForOp &loopOp,
                                  SmallVector<ScopeOp> &allocaScopes);
  static void collectAllocaAccesses(
      scf::ForOp loopOp, ArrayRef<ScopeOp> allocaScopes,
      DenseMap<ScopeOp, AllocaAccesses> &accessesByAllocaScope);
  static bool sinkLoadsToUserScopes(
      ArrayRef<ScopeOp> allocaScopes,
      DenseMap<ScopeOp, AllocaAccesses> &accessesByAllocaScope);
  static bool sinkStoresToUserScopes(
      ArrayRef<ScopeOp> allocaScopes,
      DenseMap<ScopeOp, AllocaAccesses> &accessesByAllocaScope);
  static bool sinkAllocasToUserScopes(ArrayRef<ScopeOp> allocaScopes);
  bool sinkAllocasInCAVScopes(bool onlyNormalizedCounters);

  SetVector<ScopeOp> getIntraIterScopeDeps(ScopeOp scope);
  ScopeOp mergeScopeGroup(SmallPtrSetImpl<Operation *> &group);
  ScopeOp tryMergeBackwardFromSeed(ScopeOp seed);
  bool tryMergeSmallScopeIntoConnected(ScopeOp scopeA, ScopeOp scopeB,
                                       bool aIsConsumer);

  scf::ForOp loopOp;
};

struct MergeCVScopesPass : public impl::MergeCVScopesBase<MergeCVScopesPass> {
protected:
  void runOnOperation() override;
};

} // namespace

//===----------------------------------------------------------------------===//
// ScopeDependenceGraph Implementation
//===----------------------------------------------------------------------===//

ScopeDependenceGraph::ScopeDependenceGraph(scf::ForOp loop,
                                           bool withCarriedEdges)
    : loop(loop), body(loop.getBody()) {
  buildIntraEdges();
  if (withCarriedEdges)
    buildCarriedEdges();
}

ScopeDependenceGraph::ScopeDependenceGraph(Block *body) : body(body) {
  buildIntraEdges();
}

std::optional<unsigned> ScopeDependenceGraph::getIndex(Operation *op) const {
  auto it = index.find(op);
  if (it == index.end())
    return std::nullopt;
  return it->second;
}

void ScopeDependenceGraph::printSummary(
    llvm::raw_ostream &os, ArrayRef<SmallVector<Operation *>> groups) const {
  os << "[MergeCVScopes] " << loop->getName() << " @" << loop->getLoc() << "\n";
  for (const CarriedEdge &e : carried) {
    os << "[MergeCVScopes]   carried: "
       << "#" << e.from << " " << nodes[e.from]->getName() << " -> "
       << "#" << e.to << " " << nodes[e.to]->getName() << " via arg "
       << e.argIndex << (e.from == e.to ? " (self)" : "") << "\n";
  }
  if (groups.empty()) {
    os << "[MergeCVScopes]   no loop-carried group; nothing to merge\n";
  } else {
    for (auto [gi, group] : llvm::enumerate(groups)) {
      os << "[MergeCVScopes]   group " << gi << ": {";
      llvm::interleaveComma(
          group, os, [&](Operation *op) { os << "#" << index.lookup(op); });
      os << "}\n";
    }
  }
}

/// Follow view-like ops back to the buffer that is actually accessed.
Value ScopeDependenceGraph::getMemBase(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(def)) {
      v = view.getViewSource();
      continue;
    }
    if (isa<memref::CastOp, memref::MemorySpaceCastOp, memref::CollapseShapeOp,
            memref::ExpandShapeOp, memref::ReinterpretCastOp>(def) &&
        def->getNumOperands() > 0) {
      v = def->getOperand(0);
      continue;
    }
    break;
  }
  return v;
}

void ScopeDependenceGraph::recordMemref(NodeMemInfo &info, Value v,
                                        bool isWrite) {
  if (!v || !isa<BaseMemRefType>(v.getType()))
    return;
  MemAccess &a = info.buffers[getMemBase(v)];
  if (isWrite)
    a.write = true;
  else
    a.read = true;
}

void ScopeDependenceGraph::collectMemAccesses(Operation *top,
                                              NodeMemInfo &info) {
  top->walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>() || isa<annotation::MarkOp>(op))
      return;
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return;
    if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      iface.getEffects(effects);
      for (MemoryEffects::EffectInstance &e : effects) {
        bool isWrite =
            isa<MemoryEffects::Write, MemoryEffects::Free>(e.getEffect());
        bool isRead = isa<MemoryEffects::Read>(e.getEffect());
        if (!isWrite && !isRead)
          continue; // Allocate
        Value v = e.getValue();
        if (!v) {
          if (isWrite)
            info.globalWrite = true;
          else
            info.globalRead = true;
          continue;
        }
        recordMemref(info, v, isWrite);
      }
      return;
    }
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
      for (OpOperand &opnd : op->getOpOperands())
        recordMemref(info, opnd.get(), dps.isDpsInit(&opnd));
      return;
    }
    // Treat an op with no memref operand as an unknown global read-write.
    info.globalRead = true;
    info.globalWrite = true;
    for (Value v : op->getOperands())
      recordMemref(info, v, true);
  });
}

bool ScopeDependenceGraph::hasMemConflict(const NodeMemInfo &a,
                                          const NodeMemInfo &b) {
  if ((a.globalWrite && (b.globalRead || b.globalWrite)) ||
      (a.globalRead && b.globalWrite))
    return true;
  for (const auto &entry : a.buffers) {
    auto it = b.buffers.find(entry.first);
    if (it == b.buffers.end())
      continue;
    const MemAccess &x = entry.second;
    const MemAccess &y = it->second;
    if ((x.write && (y.read || y.write)) || (x.read && y.write))
      return true;
  }
  return false;
}

void ScopeDependenceGraph::buildIntraEdges() {
  for (Operation &op : body->without_terminator()) {
    index[&op] = nodes.size();
    nodes.push_back(&op);
  }
  unsigned n = nodes.size();
  intra.resize(n);

  for (unsigned j = 0; j < n; ++j) {
    Operation *user = nodes[j];
    llvm::SetVector<unsigned> preds;
    user->walk([&](Operation *nested) {
      for (Value v : nested->getOperands()) {
        Operation *def = v.getDefiningOp();
        if (!def)
          continue;
        Operation *anc = body->findAncestorOpInBlock(*def);
        if (!anc || anc == user)
          continue;
        auto it = index.find(anc);
        if (it != index.end())
          preds.insert(it->second);
      }
    });
    for (unsigned i : preds)
      intra[i].push_back(j);
  }

  SmallVector<NodeMemInfo> mem(n);
  for (unsigned i = 0; i < n; ++i)
    collectMemAccesses(nodes[i], mem[i]);
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = i + 1; j < n; ++j)
      if (hasMemConflict(mem[i], mem[j]))
        intra[i].push_back(j);

  predecessors.resize(n);
  for (unsigned i = 0; i < n; ++i) {
    llvm::sort(intra[i]);
    intra[i].erase(std::unique(intra[i].begin(), intra[i].end()),
                   intra[i].end());
    for (unsigned successor : intra[i])
      predecessors[successor].push_back(i);
  }
}

void ScopeDependenceGraph::buildCarriedEdges() {
  auto yield = cast<scf::YieldOp>(body->getTerminator());
  DenseMap<Value, Value> yieldedInto;
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs()))
    yieldedInto[arg] = yield.getOperand(i);

  for (auto [argIndex, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    // Producers: top-level ops defining the yielded value.
    SmallVector<unsigned> producers;
    SmallVector<Value> wl{yieldedInto.lookup(arg)};
    SmallPtrSet<Value, 8> seen;
    while (!wl.empty()) {
      Value v = wl.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      if (Operation *def = v.getDefiningOp()) {
        if (Operation *anc = body->findAncestorOpInBlock(*def)) {
          auto it = index.find(anc);
          if (it != index.end())
            producers.push_back(it->second);
        }
        continue;
      }
      if (Value next = yieldedInto.lookup(v))
        wl.push_back(next);
    }

    // Users: top-level ops using the block argument.
    SmallVector<unsigned> users;
    for (OpOperand &use : arg.getUses()) {
      Operation *anc = body->findAncestorOpInBlock(*use.getOwner());
      if (!anc || anc == yield.getOperation())
        continue;
      auto it = index.find(anc);
      if (it != index.end() && !llvm::is_contained(users, it->second))
        users.push_back(it->second);
    }

    for (unsigned p : producers)
      for (unsigned u : users)
        carried.push_back({p, u, static_cast<unsigned>(argIndex)});
  }
}

SmallVector<SmallVector<unsigned>>
ScopeDependenceGraph::computeSCCs(ArrayRef<SmallVector<unsigned>> succ) {
  unsigned n = succ.size();
  SmallVector<int> dfsIndex(n, -1), low(n, 0);
  SmallVector<bool> onStack(n, false);
  SmallVector<unsigned> stack;
  SmallVector<SmallVector<unsigned>> sccs;
  int counter = 0;

  std::function<void(unsigned)> visit = [&](unsigned v) {
    dfsIndex[v] = low[v] = counter++;
    stack.push_back(v);
    onStack[v] = true;
    for (unsigned w : succ[v]) {
      if (dfsIndex[w] < 0) {
        visit(w);
        low[v] = std::min(low[v], low[w]);
      } else if (onStack[w]) {
        low[v] = std::min(low[v], dfsIndex[w]);
      }
    }
    if (low[v] != dfsIndex[v])
      return;
    SmallVector<unsigned> comp;
    unsigned w;
    do {
      w = stack.pop_back_val();
      onStack[w] = false;
      comp.push_back(w);
    } while (w != v);
    llvm::sort(comp);
    sccs.push_back(std::move(comp));
  };

  for (unsigned v = 0; v < n; ++v)
    if (dfsIndex[v] < 0)
      visit(v);
  return sccs;
}

/// Group the scopes based on SCC and sort them by program order.
SmallVector<SmallVector<Operation *>> ScopeDependenceGraph::formGroups() const {
  // Loop-carried edges, plus the reverse of each edge between two different
  // ops. That ties producer and consumer together, and the SCC then also
  // picks up every op on intra-iteration paths between them.
  SmallVector<SmallVector<unsigned>> combined(intra.begin(), intra.end());
  for (const CarriedEdge &e : carried) {
    combined[e.from].push_back(e.to);
    if (e.from != e.to)
      combined[e.to].push_back(e.from);
  }

  SmallVector<SmallVector<Operation *>> groups;
  for (SmallVector<unsigned> &comp : computeSCCs(combined)) {
    unsigned numScopes = llvm::count_if(
        comp, [&](unsigned i) { return isa<scope::ScopeOp>(nodes[i]); });
    if (numScopes < 2)
      continue;
    SmallVector<Operation *> ops;
    for (unsigned i : comp)
      ops.push_back(nodes[i]);
    groups.push_back(std::move(ops));
  }
  llvm::sort(groups, [&](const SmallVector<Operation *> &a,
                         const SmallVector<Operation *> &b) {
    return index.lookup(a.front()) < index.lookup(b.front());
  });
  return groups;
}

/// Wraps top-level ops of the loop body into a new scope placed at the position
/// of the last member. The members keep their own scopes, nested inside it.
/// Ops between the members that depend on a member are moved right after the
/// new scope, all other ops keep their position.
static FailureOr<scope::ScopeOp>
wrapGroupInScope(Block *body, ArrayRef<Operation *> members) {
  if (members.empty())
    return failure();

  ScopeDependenceGraph g(body);

  SmallVector<unsigned> idx;
  for (Operation *m : members) {
    std::optional<unsigned> i = g.getIndex(m);
    if (!i)
      return failure();
    idx.push_back(*i);
  }
  llvm::sort(idx);
  idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
  unsigned first = idx.front();
  unsigned last = idx.back();

  llvm::BitVector isMember(g.getSize());
  for (unsigned i : idx)
    isMember.set(i);

  llvm::BitVector outside(g.getSize());
  for (unsigned i = first; i <= last; ++i) {
    if (!isMember.test(i) && !outside.test(i))
      continue;
    for (unsigned s : g.getSuccessors(i)) {
      if (!isMember.test(s)) {
        outside.set(s);
        continue;
      }
      if (!isMember.test(i))
        return failure();
    }
  }

  SmallVector<Operation *> toMoveAfter;
  for (unsigned i = first + 1; i < last; ++i)
    if (outside.test(i))
      toMoveAfter.push_back(g.getNode(i));

  SmallVector<Operation *> memberOps;
  for (unsigned i : idx)
    memberOps.push_back(g.getNode(i));
  SmallPtrSet<Operation *, 16> memberSet(memberOps.begin(), memberOps.end());

  // Member results used outside the set become results of the new scope.
  SmallVector<Value> escaping;
  SmallVector<Type> resultTypes;
  SmallVector<Location> locs;
  for (Operation *m : memberOps) {
    locs.push_back(m->getLoc());
    for (Value r : m->getResults()) {
      bool usedOutside = llvm::any_of(r.getUses(), [&](OpOperand &u) {
        Operation *anc = body->findAncestorOpInBlock(*u.getOwner());
        return !anc || !memberSet.contains(anc);
      });
      if (usedOutside) {
        escaping.push_back(r);
        resultTypes.push_back(r.getType());
      }
    }
  }

  Operation *lastOp = memberOps.back();
  OpBuilder builder(lastOp->getContext());
  builder.setInsertionPointAfter(lastOp);
  Location loc = builder.getFusedLoc(locs);

  auto mergedOp = builder.create<scope::ScopeOp>(loc, resultTypes);
  mergedOp->setAttr(kCVMergedScopeAttrName, builder.getUnitAttr());

  Block *mergedBody = builder.createBlock(&mergedOp.getRegion());
  builder.setInsertionPointToEnd(mergedBody);
  auto retOp = builder.create<scope::ReturnOp>(loc, escaping);

  for (Operation *m : memberOps)
    m->moveBefore(retOp);

  Operation *mergedPtr = mergedOp.getOperation();
  for (auto [i, v] : llvm::enumerate(escaping)) {
    Value newRes = mergedOp->getResult(i);
    v.replaceUsesWithIf(newRes, [&](OpOperand &u) {
      return !mergedPtr->isAncestor(u.getOwner());
    });
  }

  Operation *prev = mergedPtr;
  for (Operation *op : toMoveAfter) {
    op->moveAfter(prev);
    prev = op;
  }

  return mergedOp;
}

//===----------------------------------------------------------------------===//
// LoopScopeMerger Implementation
//===----------------------------------------------------------------------===//

bool LoopScopeMerger::isScalarOnlyCubeAndVectorScope(ScopeOp scope) {
  if (!scope || scope->hasAttr(kCVMergedScopeAttrName) ||
      getScopeCoreType(scope) != TCoreType::CUBE_AND_VECTOR)
    return false;
  return !scope
              ->walk([](Operation *op) {
                return isCoreOp(op) ? WalkResult::interrupt()
                                    : WalkResult::advance();
              })
              .wasInterrupted();
}

bool LoopScopeMerger::applyScalarLoopCarriedDepRule() {
  bool changed = false;
  while (true) {
    Block *body = loopOp.getBody();
    auto yieldOp = cast<scf::YieldOp>(body->getTerminator());

    ScopeOp targetScalarScope;
    unsigned targetArgIdx = 0;
    SmallVector<ScopeOp> targetUserScopes;

    for (ScopeOp scalarScope : body->getOps<ScopeOp>()) {
      if (!isScalarOnlyCubeAndVectorScope(scalarScope) ||
          scalarScope.getNumResults() != 1)
        continue;

      Value res = scalarScope.getResult(0);
      std::optional<unsigned> foundArgIdx;
      for (auto [idx, opnd] : llvm::enumerate(yieldOp.getOperands())) {
        if (opnd == res) {
          if (foundArgIdx) {
            foundArgIdx.reset();
            break;
          }
          foundArgIdx = static_cast<unsigned>(idx);
        }
      }
      if (!foundArgIdx || *foundArgIdx >= loopOp.getNumRegionIterArgs())
        continue;

      unsigned argIdx = *foundArgIdx;
      BlockArgument origIterArg = loopOp.getRegionIterArg(argIdx);

      SmallPtrSet<Operation *, 8> seenUserScopes;
      bool hasUnsupportedIterArgUse = false;
      for (OpOperand &use : origIterArg.getUses()) {
        Operation *anc = body->findAncestorOpInBlock(*use.getOwner());
        if (!anc || anc == scalarScope.getOperation())
          continue;
        auto userScope = dyn_cast<ScopeOp>(anc);
        auto userCore = userScope ? getScopeCoreType(userScope) : std::nullopt;
        if (!userScope || userScope->hasAttr(kCVMergedScopeAttrName) ||
            (userCore != TCoreType::CUBE && userCore != TCoreType::VECTOR)) {
          hasUnsupportedIterArgUse = true;
          break;
        }
        seenUserScopes.insert(userScope.getOperation());
      }
      if (hasUnsupportedIterArgUse || seenUserScopes.empty())
        continue;

      bool hasUnsupportedResUse =
          llvm::any_of(res.getUses(), [&](OpOperand &use) {
            if (use.getOwner() == yieldOp.getOperation() &&
                use.getOperandNumber() == argIdx)
              return false;
            Operation *anc = body->findAncestorOpInBlock(*use.getOwner());
            return !anc || !seenUserScopes.contains(anc);
          });
      if (hasUnsupportedResUse)
        continue;

      SmallVector<ScopeOp> userScopes;
      for (ScopeOp s : body->getOps<ScopeOp>()) {
        if (seenUserScopes.contains(s.getOperation()))
          userScopes.push_back(s);
      }

      bool canCloneToAllUsers = true;
      scalarScope.walk([&](Operation *op) {
        for (Value opnd : op->getOperands()) {
          Operation *def = opnd.getDefiningOp();
          if (!def || scalarScope->isProperAncestor(def))
            continue;
          Operation *ancDef = body->findAncestorOpInBlock(*def);
          if (!ancDef)
            continue;
          for (ScopeOp userScope : userScopes) {
            if (ancDef != userScope.getOperation() &&
                !ancDef->isBeforeInBlock(userScope)) {
              canCloneToAllUsers = false;
              return;
            }
          }
        }
      });
      if (!canCloneToAllUsers)
        continue;

      targetScalarScope = scalarScope;
      targetArgIdx = argIdx;
      targetUserScopes = std::move(userScopes);
      break;
    }

    if (!targetScalarScope)
      return changed;

    unsigned privateBase = 0;
    if (targetUserScopes.size() > 1) {
      SmallVector<Value> newInits(targetUserScopes.size() - 1,
                                  loopOp.getInitArgs()[targetArgIdx]);
      FailureOr<unsigned> baseOr =
          hivm::appendPassThroughIterArgs(loopOp, newInits);
      if (failed(baseOr))
        return changed;
      privateBase = *baseOr;
    }

    body = loopOp.getBody();
    yieldOp = cast<scf::YieldOp>(body->getTerminator());
    BlockArgument origIterArg = loopOp.getRegionIterArg(targetArgIdx);
    Value res = targetScalarScope.getResult(0);

    for (size_t idx = 0; idx < targetUserScopes.size(); ++idx) {
      ScopeOp userScope = targetUserScopes[idx];
      unsigned userArgIdx =
          (idx == 0) ? targetArgIdx
                     : (privateBase + static_cast<unsigned>(idx) - 1);
      BlockArgument userIterArg = loopOp.getRegionIterArg(userArgIdx);

      if (userIterArg != origIterArg) {
        origIterArg.replaceUsesWithIf(userIterArg, [&](OpOperand &use) {
          return userScope->isProperAncestor(use.getOwner());
        });
      }

      OpBuilder builder(targetScalarScope->isBeforeInBlock(userScope)
                            ? userScope.getOperation()
                            : userScope->getNextNode());
      IRMapping mapping;
      mapping.map(origIterArg, userIterArg);
      auto clonedScalarScope = cast<ScopeOp>(
          builder.clone(*targetScalarScope.getOperation(), mapping));
      Value clonedRes = clonedScalarScope.getResult(0);

      res.replaceUsesWithIf(clonedRes, [&](OpOperand &use) {
        return userScope->isProperAncestor(use.getOwner());
      });
      yieldOp.setOperand(userArgIdx, clonedRes);

      ScopeOp merged = mergeScopes(clonedScalarScope, userScope, userScope);
      assert(merged && "expected scalar-only CUBE_AND_VECTOR scope to merge");
    }

    assert(targetScalarScope.use_empty() &&
           "expected scalar-only scope to have no remaining uses");
    targetScalarScope.erase();
    changed = true;
  }
}

bool LoopScopeMerger::applyLoopCarriedDepGroupingRule() {
  ScopeDependenceGraph graph(loopOp);
  SmallVector<SmallVector<Operation *>> groups = graph.formGroups();
  LLVM_DEBUG(graph.printSummary(llvm::dbgs(), groups));

  bool changed = false;
  for (SmallVector<Operation *> &group : groups) {
    if (failed(wrapGroupInScope(loopOp.getBody(), group)))
      loopOp->emitWarning("[MergeCVScopes] loop-carried group is not convex; "
                          "left unmerged");
    else
      changed = true;
  }
  return changed;
}

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

bool LoopScopeMerger::isEligibleForCoreMerge(ScopeOp scope) {
  return scope && !scope->hasAttr(kCVMergedScopeAttrName) &&
         getScopeCoreType(scope).has_value();
}

bool LoopScopeMerger::collapseSingleChildMergedScope(ScopeOp outerScope) {
  if (!outerScope || !outerScope->hasAttr(kCVMergedScopeAttrName))
    return false;
  Block &outerBody = outerScope.getRegion().front();
  if (!llvm::hasSingleElement(outerBody.without_terminator()))
    return false;
  auto innerScope = dyn_cast<ScopeOp>(&outerBody.front());
  if (!innerScope || !isEligibleForCoreMerge(innerScope))
    return false;

  auto outerReturn = cast<scope::ReturnOp>(outerBody.getTerminator());
  Block &innerBody = innerScope.getRegion().front();
  auto innerReturn = cast<scope::ReturnOp>(innerBody.getTerminator());

  if (auto coreType = getScopeCoreType(innerScope)) {
    outerScope->setAttr(kPipelinedLoopCoreTypeAttrName,
                        TCoreTypeAttr::get(outerScope.getContext(), *coreType));
  }
  outerScope->removeAttr(kCVMergedScopeAttrName);

  for (auto [res, retVal] :
       llvm::zip(innerScope.getResults(), innerReturn.getOperands()))
    res.replaceAllUsesWith(retVal);

  outerBody.getOperations().splice(outerReturn->getIterator(),
                                   innerBody.getOperations(), innerBody.begin(),
                                   innerReturn->getIterator());
  innerScope.erase();
  return true;
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
        auto s = dyn_cast<ScopeOp>(op);
        return !s || op->getBlock() != parentBlock ||
               !isEligibleForCoreMerge(s);
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

ScopeOp LoopScopeMerger::tryMergeBackwardFromSeed(ScopeOp seed) {
  Block *seedBlock = seed->getBlock();
  auto seedCore = getScopeCoreType(seed);
  if (!seedBlock || !seedCore)
    return ScopeOp();
  TCoreType targetCore = *seedCore;
  SmallPtrSet<Operation *, 16> mergeSet = {seed.getOperation()};
  SmallVector<ScopeOp> worklist = {seed};

  while (!worklist.empty()) {
    ScopeOp curr = worklist.pop_back_val();
    SetVector<ScopeOp> deps = getIntraIterScopeDeps(curr);

    bool hasIncompatibleDep = llvm::any_of(deps, [&](ScopeOp dep) {
      if (dep->getBlock() != seedBlock || !isEligibleForCoreMerge(dep))
        return false;
      auto depCore = getScopeCoreType(dep);
      return depCore && !areCompatibleCoreTypes(targetCore, *depCore);
    });
    if (hasIncompatibleDep)
      continue;

    for (ScopeOp dep : deps) {
      if (dep->getBlock() != seedBlock || !isEligibleForCoreMerge(dep))
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

  if (mergeSet.size() <= 1)
    return ScopeOp();
  return mergeScopeGroup(mergeSet);
}

bool LoopScopeMerger::applyMergedScopeSameCoreRule() {
  bool changed = false;
  SmallVector<ScopeOp> mergedScopes;
  for (ScopeOp scope : loopOp.getBody()->getOps<ScopeOp>()) {
    if (scope->hasAttr(kCVMergedScopeAttrName))
      mergedScopes.push_back(scope);
  }

  for (ScopeOp outerScope : mergedScopes) {
    DenseSet<Operation *> processedSeeds;
    while (true) {
      ScopeOp seed;
      outerScope->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op != outerScope.getOperation() && isa<scf::ForOp>(op))
          return WalkResult::skip();
        if (auto scope = dyn_cast<ScopeOp>(op)) {
          if (scope != outerScope &&
              !processedSeeds.contains(scope.getOperation()) &&
              isEligibleForCoreMerge(scope)) {
            seed = scope;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (!seed)
        break;

      processedSeeds.insert(seed.getOperation());
      if (ScopeOp merged = tryMergeBackwardFromSeed(seed)) {
        processedSeeds.insert(merged.getOperation());
        if (merged != seed)
          changed = true;
      }
    }

    changed |= collapseSingleChildMergedScope(outerScope);
  }
  return changed;
}

bool LoopScopeMerger::applyCrossCoreTransferBackwardRule() {
  bool changed = false;
  DenseSet<Operation *> processedSeeds;
  while (true) {
    ScopeOp seed;
    loopOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op != loopOp.getOperation() && isa<scf::ForOp>(op))
        return WalkResult::skip();
      if (auto scope = dyn_cast<ScopeOp>(op)) {
        if (!processedSeeds.contains(scope.getOperation()) &&
            isEligibleForCoreMerge(scope) && hasCrossCoreTransfer(scope)) {
          seed = scope;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (!seed)
      break;

    processedSeeds.insert(seed.getOperation());
    if (ScopeOp merged = tryMergeBackwardFromSeed(seed)) {
      processedSeeds.insert(merged.getOperation());
      if (merged != seed)
        changed = true;
    }
  }
  return changed;
}

static unsigned countCoreOps(ScopeOp scope) {
  unsigned count = 0;
  scope->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op != scope.getOperation() && isa<ScopeOp>(op))
      return WalkResult::skip();
    if (isCoreOp(op))
      ++count;
    return WalkResult::advance();
  });
  return count;
}

static ScopeOp pruneUnusedWrapperResults(ScopeOp wrapper) {
  if (!wrapper || !wrapper->hasAttr(kCVMergedScopeAttrName))
    return wrapper;
  if (llvm::none_of(wrapper.getResults(),
                    [](Value r) { return r.use_empty(); }))
    return wrapper;

  auto retOp =
      cast<scope::ReturnOp>(wrapper.getRegion().front().getTerminator());
  SmallVector<Value> keptOperands;
  SmallVector<Value> keptResults;
  for (auto [opnd, res] :
       llvm::zip(retOp.getOperands(), wrapper.getResults())) {
    if (!res.use_empty()) {
      keptOperands.push_back(opnd);
      keptResults.push_back(res);
    }
  }
  ScopeOp newWrapper = recreateScope(wrapper, keptOperands);
  for (auto [oldRes, newRes] : llvm::zip(keptResults, newWrapper.getResults()))
    oldRes.replaceAllUsesWith(newRes);
  wrapper.erase();
  return newWrapper;
}

bool LoopScopeMerger::tryMergeSmallScopeIntoConnected(ScopeOp scopeA,
                                                      ScopeOp scopeB,
                                                      bool aIsConsumer) {
  Block *blockA = scopeA->getBlock();
  Block *blockB = scopeB->getBlock();
  if (!blockA || !blockB)
    return false;

  ScopeDependenceGraph::NodeMemInfo memA;
  ScopeDependenceGraph::collectMemAccesses(scopeA, memA);
  auto conflictsWith = [&](Operation *op) {
    ScopeDependenceGraph::NodeMemInfo memOp;
    ScopeDependenceGraph::collectMemAccesses(op, memOp);
    return ScopeDependenceGraph::hasMemConflict(memA, memOp);
  };

  if (blockA == blockB) {
    if (!aIsConsumer) {
      if (!scopeA->isBeforeInBlock(scopeB))
        return false;
      for (Operation *op = scopeA->getNextNode(); op != scopeB.getOperation();
           op = op->getNextNode()) {
        if (conflictsWith(op))
          return false;
      }
      if (auto w = dyn_cast<ScopeOp>(blockA->getParentOp());
          w && w->hasAttr(kCVMergedScopeAttrName))
        w = pruneUnusedWrapperResults(w);
      if (!llvm::all_of(scopeA->getUsers(), [&](Operation *u) {
            return scopeB->isProperAncestor(u);
          }))
        return false;
      if (scopeA->getNextNode() != scopeB.getOperation())
        scopeA->moveBefore(scopeB);
      ScopeOp merged = mergeScopes(scopeA, scopeB, scopeB);
      if (!merged)
        return false;
      if (auto w = dyn_cast<ScopeOp>(merged->getParentOp());
          w && w->hasAttr(kCVMergedScopeAttrName))
        collapseSingleChildMergedScope(w);
      return true;
    }

    if (!scopeB->isBeforeInBlock(scopeA) || !scopeA->use_empty())
      return false;
    for (Operation *op = scopeB->getNextNode(); op != scopeA.getOperation();
         op = op->getNextNode()) {
      if (conflictsWith(op))
        return false;
    }
    if (scopeB->getNextNode() != scopeA.getOperation())
      scopeA->moveAfter(scopeB);
    ScopeOp merged = mergeScopes(scopeB, scopeA, scopeA);
    if (!merged)
      return false;
    if (auto w = dyn_cast<ScopeOp>(merged->getParentOp());
        w && w->hasAttr(kCVMergedScopeAttrName))
      collapseSingleChildMergedScope(w);
    return true;
  }

  // `scopeA` in `loopOp.getBody()` and `scopeB` inside a `{cv_merged_scope}`
  // wrapper directly in `loopOp.getBody()`.
  if (blockA == loopOp.getBody()) {
    auto wrapper = dyn_cast<ScopeOp>(blockB->getParentOp());
    if (!wrapper || !wrapper->hasAttr(kCVMergedScopeAttrName) ||
        wrapper->getBlock() != loopOp.getBody())
      return false;

    wrapper = pruneUnusedWrapperResults(wrapper);
    blockB = &wrapper.getRegion().front();

    if (!aIsConsumer) {
      if (!scopeA->isBeforeInBlock(wrapper))
        return false;
      if (!llvm::all_of(scopeA->getUsers(), [&](Operation *u) {
            return scopeB->isProperAncestor(u);
          }))
        return false;
      for (Operation *op = scopeA->getNextNode(); op != wrapper.getOperation();
           op = op->getNextNode()) {
        if (conflictsWith(op))
          return false;
      }
      for (Operation &op : *blockB) {
        if (&op == scopeB.getOperation())
          break;
        if (conflictsWith(&op))
          return false;
      }

      scopeA->moveBefore(scopeB);
      ScopeOp merged = mergeScopes(scopeA, scopeB, scopeB);
      assert(merged && "validated small vector scope must merge");
      collapseSingleChildMergedScope(wrapper);
      return true;
    }

    if (!wrapper->isBeforeInBlock(scopeA) || !scopeA->use_empty())
      return false;
    for (Operation *op = wrapper->getNextNode(); op != scopeA.getOperation();
         op = op->getNextNode()) {
      if (conflictsWith(op))
        return false;
    }
    for (Operation *op = scopeB->getNextNode();
         op && !op->hasTrait<OpTrait::IsTerminator>(); op = op->getNextNode()) {
      if (conflictsWith(op))
        return false;
    }

    auto wRet = cast<scope::ReturnOp>(blockB->getTerminator());
    bool validWrapperInputs = true;
    scopeA->walk([&](Operation *nested) {
      for (Value opnd : nested->getOperands()) {
        if (auto res = dyn_cast<OpResult>(opnd);
            res && res.getOwner() == wrapper.getOperation()) {
          if (wRet.getOperand(res.getResultNumber()).getDefiningOp() !=
              scopeB.getOperation())
            validWrapperInputs = false;
        }
      }
    });
    if (!validWrapperInputs)
      return false;

    scopeA->walk([&](Operation *nested) {
      for (OpOperand &opnd : nested->getOpOperands()) {
        if (auto res = dyn_cast<OpResult>(opnd.get());
            res && res.getOwner() == wrapper.getOperation()) {
          opnd.set(wRet.getOperand(res.getResultNumber()));
        }
      }
    });
    scopeA->moveAfter(scopeB);
    wrapper = pruneUnusedWrapperResults(wrapper);
    ScopeOp merged = mergeScopes(scopeB, scopeA, scopeA);
    assert(merged && "validated small vector scope must merge");
    collapseSingleChildMergedScope(wrapper);
    return true;
  }

  // `scopeA` inside a `{cv_merged_scope}` wrapper directly in
  // `loopOp.getBody()` and `scopeB` in `loopOp.getBody()`.
  if (blockB == loopOp.getBody()) {
    auto wrapper = dyn_cast<ScopeOp>(blockA->getParentOp());
    if (!wrapper || !wrapper->hasAttr(kCVMergedScopeAttrName) ||
        wrapper->getBlock() != loopOp.getBody())
      return false;

    wrapper = pruneUnusedWrapperResults(wrapper);
    blockA = &wrapper.getRegion().front();
    auto finalizeWrapper = [](ScopeOp w) {
      if (!w)
        return;
      if (w.getRegion().front().without_terminator().empty())
        w.erase();
      else
        collapseSingleChildMergedScope(w);
    };

    if (!aIsConsumer) {
      if (!wrapper->isBeforeInBlock(scopeB))
        return false;
      for (Operation *op = scopeA->getNextNode();
           op && !op->hasTrait<OpTrait::IsTerminator>();
           op = op->getNextNode()) {
        if (conflictsWith(op))
          return false;
      }
      for (Operation *op = wrapper->getNextNode(); op != scopeB.getOperation();
           op = op->getNextNode()) {
        if (conflictsWith(op))
          return false;
      }

      auto wRet = cast<scope::ReturnOp>(blockA->getTerminator());
      for (OpOperand &use : llvm::make_early_inc_range(scopeA->getUses())) {
        if (scopeB->isProperAncestor(use.getOwner()))
          continue;
        if (use.getOwner() != wRet.getOperation())
          return false;
        Value wRes = wrapper.getResult(use.getOperandNumber());
        if (!llvm::all_of(wRes.getUsers(), [&](Operation *u) {
              return scopeB->isProperAncestor(u);
            }))
          return false;
      }

      for (auto [idx, retVal] : llvm::enumerate(wRet.getOperands())) {
        if (retVal.getDefiningOp() == scopeA.getOperation())
          wrapper.getResult(idx).replaceAllUsesWith(retVal);
      }
      wrapper = pruneUnusedWrapperResults(wrapper);
      scopeA->moveBefore(scopeB);
      finalizeWrapper(wrapper);
      ScopeOp merged = mergeScopes(scopeA, scopeB, scopeB);
      assert(merged && "validated small vector scope must merge");
      return true;
    }

    if (!scopeB->isBeforeInBlock(wrapper) || !scopeA->use_empty())
      return false;
    for (Operation &op : *blockA) {
      if (&op == scopeA.getOperation())
        break;
      if (conflictsWith(&op))
        return false;
    }
    for (Operation *op = scopeB->getNextNode(); op != wrapper.getOperation();
         op = op->getNextNode()) {
      if (conflictsWith(op))
        return false;
    }

    scopeA->moveAfter(scopeB);
    finalizeWrapper(wrapper);
    ScopeOp merged = mergeScopes(scopeB, scopeA, scopeA);
    assert(merged && "validated small vector scope must merge");
    return true;
  }

  return false;
}

bool LoopScopeMerger::applySmallVectorScopeMergeRule() {
  bool anyChanged = false;
  auto findEnclosingLeafScope = [&](Operation *user) -> ScopeOp {
    for (Operation *p = user; p && p != loopOp.getOperation();
         p = p->getParentOp()) {
      if (auto s = dyn_cast<ScopeOp>(p);
          s && !s->hasAttr(kCVMergedScopeAttrName))
        return s;
    }
    return ScopeOp();
  };

  while (true) {
    SmallVector<ScopeOp> candidates;
    loopOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op != loopOp.getOperation() && isa<scf::ForOp>(op))
        return WalkResult::skip();
      auto scope = dyn_cast<ScopeOp>(op);
      if (!scope || !isEligibleForCoreMerge(scope))
        return WalkResult::advance();
      if (getScopeCoreType(scope) == TCoreType::VECTOR &&
          countCoreOps(scope) < 3)
        candidates.push_back(scope);
      return WalkResult::skip();
    });

    bool changed = false;
    for (ScopeOp scopeA : candidates) {
      SmallPtrSet<Operation *, 4> inputScopes;
      bool hasUnsupportedInput = false;
      scopeA->walk([&](Operation *op) {
        if (hasUnsupportedInput || op == scopeA.getOperation())
          return;
        for (Value opnd : op->getOperands()) {
          if (scopeA.getRegion().isAncestor(opnd.getParentRegion()))
            continue;
          if (!loopOp.getRegion().isAncestor(opnd.getParentRegion()))
            continue;
          if (opnd == loopOp.getInductionVar())
            continue;

          Value unwrapped = opnd;
          while (auto res = dyn_cast<OpResult>(unwrapped)) {
            auto wrapper = dyn_cast<ScopeOp>(res.getOwner());
            if (!wrapper || !wrapper->hasAttr(kCVMergedScopeAttrName) ||
                !loopOp->isProperAncestor(wrapper))
              break;
            auto wRet = cast<scope::ReturnOp>(
                wrapper.getRegion().front().getTerminator());
            unwrapped = wRet.getOperand(res.getResultNumber());
          }

          auto srcScope = unwrapped.getDefiningOp<ScopeOp>();
          if (!srcScope || srcScope == scopeA ||
              !isEligibleForCoreMerge(srcScope)) {
            hasUnsupportedInput = true;
            return;
          }
          inputScopes.insert(srcScope.getOperation());
        }
      });
      if (hasUnsupportedInput)
        continue;

      SmallPtrSet<Operation *, 4> outputScopes;
      bool hasUnsupportedOutput = false;
      SmallVector<Value, 4> outWorklist(scopeA.result_begin(),
                                        scopeA.result_end());
      SmallPtrSet<Value, 4> visitedOut(outWorklist.begin(), outWorklist.end());
      while (!outWorklist.empty() && !hasUnsupportedOutput) {
        Value cur = outWorklist.pop_back_val();
        for (OpOperand &use : cur.getUses()) {
          Operation *user = use.getOwner();
          if (isa<scope::ReturnOp>(user)) {
            if (auto wrapper = dyn_cast<ScopeOp>(user->getParentOp());
                wrapper && wrapper->hasAttr(kCVMergedScopeAttrName) &&
                loopOp->isProperAncestor(wrapper)) {
              Value wRes = wrapper.getResult(use.getOperandNumber());
              if (visitedOut.insert(wRes).second)
                outWorklist.push_back(wRes);
              continue;
            }
          }
          if (ScopeOp dstScope = findEnclosingLeafScope(user)) {
            if (dstScope == scopeA || !isEligibleForCoreMerge(dstScope)) {
              hasUnsupportedOutput = true;
              break;
            }
            outputScopes.insert(dstScope.getOperation());
            continue;
          }
          hasUnsupportedOutput = true;
          break;
        }
      }
      if (hasUnsupportedOutput)
        continue;

      bool isOneInputZeroOutput =
          inputScopes.size() == 1 && outputScopes.empty();
      bool isZeroInputOneOutput =
          inputScopes.empty() && outputScopes.size() == 1;
      if (!isOneInputZeroOutput && !isZeroInputOneOutput)
        continue;

      ScopeOp scopeB = cast<ScopeOp>(
          isOneInputZeroOutput ? *inputScopes.begin() : *outputScopes.begin());
      if (!isEligibleForCoreMerge(scopeB) ||
          getScopeCoreType(scopeB) != TCoreType::VECTOR)
        continue;

      if (tryMergeSmallScopeIntoConnected(scopeA, scopeB,
                                          isOneInputZeroOutput)) {
        changed = true;
        anyChanged = true;
        break;
      }
    }

    if (!changed)
      break;
  }
  return anyChanged;
}

static bool isNormalizedCounterLoad(Operation *op) {
  auto load = dyn_cast<memref::LoadOp>(op);
  if (!load)
    return false;
  Value counter = load.getMemRef();
  while (auto scope = counter.getDefiningOp<ScopeOp>()) {
    auto returnOp = cast<scope::ReturnOp>(scope.getBody()->getTerminator());
    counter = returnOp.getOperand(cast<OpResult>(counter).getResultNumber());
  }
  auto alloca = counter.getDefiningOp<memref::AllocaOp>();
  return alloca && alloca->hasAttr(kNormalizeMatmulCounterAttr);
}

bool LoopScopeMerger::applyGraphSameCoreRule() {
  auto mergeInBlock = [&](Block *body) {
    ScopeDependenceGraph graph(body);
    auto tryMerge = [&](unsigned first, unsigned second) {
      auto firstScope = dyn_cast<ScopeOp>(graph.getNode(first));
      auto secondScope = dyn_cast<ScopeOp>(graph.getNode(second));
      if (!isEligibleForCoreMerge(firstScope) ||
          !isEligibleForCoreMerge(secondScope))
        return false;
      auto firstCore = getScopeCoreType(firstScope);
      auto secondCore = getScopeCoreType(secondScope);
      bool sameCore =
          (*firstCore == TCoreType::CUBE || *firstCore == TCoreType::VECTOR) &&
          firstCore == secondCore;
      bool counterSelection =
          *firstCore == TCoreType::CUBE_AND_VECTOR &&
          (*secondCore == TCoreType::CUBE ||
           *secondCore == TCoreType::VECTOR) &&
          firstScope.getNumResults() == 1 &&
          llvm::any_of(
              firstScope.getBody()->without_terminator(),
              [](Operation &op) { return isNormalizedCounterLoad(&op); }) &&
          llvm::all_of(firstScope->getUsers(),
                       [&](Operation *user) {
                         return secondScope->isProperAncestor(user);
                       }) &&
          llvm::all_of(
              firstScope.getBody()->without_terminator(), [](Operation &op) {
                return isMemoryEffectFree(&op) || isNormalizedCounterLoad(&op);
              });
      if (!sameCore && !counterSelection)
        return false;

      SmallVector<Operation *> members{firstScope, secondScope};
      auto wrapper = wrapGroupInScope(body, members);
      if (failed(wrapper))
        return false;
      ScopeOp merged = mergeScopes(firstScope, secondScope, secondScope);
      assert(merged && "validated same-core scopes must merge");
      for (NamedAttribute attr : merged->getAttrs())
        (*wrapper)->setAttr(attr.getName(), attr.getValue());
      collapseSingleChildMergedScope(*wrapper);
      collapseSingleChildMergedScope(dyn_cast<ScopeOp>(body->getParentOp()));
      return true;
    };

    auto tryMergeNeighbors = [&](ArrayRef<unsigned> neighbors) {
      for (unsigned i = 0; i < neighbors.size(); ++i)
        for (unsigned j = i + 1; j < neighbors.size(); ++j)
          if (tryMerge(neighbors[i], neighbors[j]))
            return true;
      return false;
    };

    bool changed = false;
    for (unsigned i = 0; i < graph.getSize(); ++i) {
      // Wrappers and scopes with other core types can anchor a fan-in or
      // fan-out without becoming members of the merge themselves.
      if (!isa<ScopeOp>(graph.getNode(i)))
        continue;
      for (unsigned successor : graph.getSuccessors(i)) {
        if (tryMerge(i, successor)) {
          changed = true;
          break;
        }
      }
      if (changed || tryMergeNeighbors(graph.getPredecessors(i)) ||
          tryMergeNeighbors(graph.getSuccessors(i))) {
        changed = true;
        break;
      }
    }
    return changed;
  };

  bool anyChanged = false;
  while (true) {
    SmallVector<Block *> bodies{loopOp.getBody()};
    loopOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op != loopOp.getOperation() && isa<scf::ForOp>(op))
        return WalkResult::skip();
      if (auto scope = dyn_cast<ScopeOp>(op))
        bodies.push_back(scope.getBody());
      return WalkResult::advance();
    });
    if (!llvm::any_of(bodies, mergeInBlock))
      return anyChanged;
    anyChanged = true;
    // A contraction or wrapper collapse can erase scopes and their blocks.
    // Rediscover bodies and rebuild graphs before considering another pair.
  }
}

/// Core type of `scope` when it is a single Cube or Vector scope, or a
/// `{cv_merged_scope}` whose nested core scopes are all that one type.
static std::optional<TCoreType> getSingleCoreKind(ScopeOp scope) {
  if (!scope)
    return std::nullopt;
  auto asSingleCore = [](std::optional<TCoreType> core) {
    if (core == TCoreType::CUBE || core == TCoreType::VECTOR)
      return core;
    return std::optional<TCoreType>();
  };
  if (!scope->hasAttr(kCVMergedScopeAttrName))
    return asSingleCore(getScopeCoreType(scope));

  std::optional<TCoreType> only;
  WalkResult walk = scope->walk([&](ScopeOp inner) {
    if (inner == scope || inner->hasAttr(kCVMergedScopeAttrName))
      return WalkResult::advance();
    std::optional<TCoreType> raw = getScopeCoreType(inner);
    if (!raw)
      return WalkResult::advance();
    std::optional<TCoreType> core = asSingleCore(raw);
    if (!core || (only && *only != *core))
      return WalkResult::interrupt();
    only = core;
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return std::nullopt;
  return only;
}

void LoopScopeMerger::fuseOrderedSameCoreScopes(Block *body) {
  if (!body)
    return;

  SmallVector<ScopeOp> nested;
  for (Operation &op : body->without_terminator()) {
    auto scope = dyn_cast<ScopeOp>(&op);
    if (scope && scope->hasAttr(kCVMergedScopeAttrName) &&
        getSingleCoreKind(scope))
      nested.push_back(scope);
  }
  for (ScopeOp scope : nested) {
    fuseOrderedSameCoreScopes(scope.getBody());
    collapseSingleChildMergedScope(scope);
  }

  ScopeOp previous;
  for (Operation &op : llvm::make_early_inc_range(*body)) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      break;
    auto current = dyn_cast<ScopeOp>(&op);
    std::optional<TCoreType> currCore =
        current ? getScopeCoreType(current) : std::nullopt;
    if (!current || !isEligibleForCoreMerge(current) ||
        (currCore != TCoreType::CUBE && currCore != TCoreType::VECTOR)) {
      previous = ScopeOp();
      continue;
    }
    if (previous) {
      std::optional<TCoreType> prevCore = getScopeCoreType(previous);
      if (prevCore && *prevCore == *currCore) {
        if (ScopeOp merged = mergeScopes(previous, current, current)) {
          previous = merged;
          continue;
        }
      }
    }
    previous = current;
  }
}

bool LoopScopeMerger::applySameCoreRunGroupingRule() {
  SmallVector<SmallVector<Operation *>> runs;
  SmallVector<Operation *> run;
  std::optional<TCoreType> runCore;
  auto flush = [&]() {
    if (run.size() >= 2)
      runs.push_back(std::move(run));
    run.clear();
    runCore.reset();
  };
  for (Operation &op : loopOp.getBody()->without_terminator()) {
    auto scope = dyn_cast<ScopeOp>(&op);
    std::optional<TCoreType> core =
        scope ? getSingleCoreKind(scope) : std::nullopt;
    if (!core || (runCore && *runCore != *core))
      flush();
    if (!core)
      continue;
    runCore = core;
    run.push_back(scope.getOperation());
  }
  flush();

  bool changed = false;
  // Later runs sit further down the block. Wrapping them first leaves the
  // earlier runs' operations in place.
  for (auto it = runs.rbegin(); it != runs.rend(); ++it) {
    FailureOr<scope::ScopeOp> wrapped = wrapGroupInScope(loopOp.getBody(), *it);
    if (succeeded(wrapped)) {
      fuseOrderedSameCoreScopes(wrapped->getBody());
      // One remaining child is the fused core scope. Collapse the wrapper so
      // a second run of this pass does not do it again.
      collapseSingleChildMergedScope(*wrapped);
      changed = true;
      continue;
    }
    loopOp->emitWarning("[MergeCVScopes] same-core run is not convex; "
                        "left unmerged");
  }
  return changed;
}

void LoopScopeMerger::collectAllocaScopes(scf::ForOp &loopOp,
                                          SmallVector<ScopeOp> &allocaScopes) {
  for (ScopeOp scope : loopOp.getOps<ScopeOp>()) {
    if (getScopeCoreType(scope) != TCoreType::CUBE_AND_VECTOR)
      continue;

    Block &body = scope.getRegion().front();
    if (body.getOperations().size() != 2)
      continue;

    auto alloca = dyn_cast<memref::AllocaOp>(&body.front());
    auto returnOp = dyn_cast<scope::ReturnOp>(body.getTerminator());
    if (alloca && returnOp && returnOp.getNumOperands() == 1 &&
        returnOp.getOperand(0) == alloca.getResult())
      allocaScopes.push_back(scope);
  }
}

void LoopScopeMerger::collectAllocaAccesses(
    scf::ForOp loopOp, ArrayRef<ScopeOp> allocaScopes,
    DenseMap<ScopeOp, AllocaAccesses> &accessesByAllocaScope) {
  DenseMap<Value, ScopeOp> allocaScopeByMemref;
  for (ScopeOp scope : allocaScopes) {
    accessesByAllocaScope.try_emplace(scope);
    allocaScopeByMemref[scope.getResult(0)] = scope;
  }

  for (ScopeOp ownerScope : loopOp.getOps<ScopeOp>()) {
    if (getScopeCoreType(ownerScope) != TCoreType::CUBE_AND_VECTOR)
      continue;

    DenseSet<Value> seenLoads;
    for (Operation &op : ownerScope.getOps()) {
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        auto it = allocaScopeByMemref.find(store.getMemRef());
        if (it == allocaScopeByMemref.end())
          continue;

        auto &initialStore = accessesByAllocaScope[it->second].initialStore;
        if (!initialStore.first)
          initialStore = {store, ownerScope};
      } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
        auto it = allocaScopeByMemref.find(load.getMemRef());
        if (it == allocaScopeByMemref.end() ||
            !seenLoads.insert(load.getMemRef()).second)
          continue;

        auto &initialLoad = accessesByAllocaScope[it->second].initialLoad;
        auto returnOp = cast<scope::ReturnOp>(
            ownerScope.getRegion().front().getTerminator());
        if (load->hasOneUse() && *(load->getUsers().begin()) == returnOp) {
          for (auto [index, returnedValue] :
               llvm::enumerate(returnOp.getOperands())) {
            if (returnedValue == load.getResult()) {
              initialLoad = {ownerScope, ownerScope.getResult(index)};
              break;
            }
          }
        }
      }
    }
  }
}

static LogicalResult
collectScopeLocalDependencies(Operation *op, ScopeOp scope,
                              SetVector<Operation *> &dependencies) {
  std::function<LogicalResult(Value)> collectDependency = [&](Value value) {
    if (!scope.getRegion().isAncestor(value.getParentRegion()))
      return success();
    Operation *def = value.getDefiningOp();
    if (!def || def->getBlock() != op->getBlock() || !isMemoryEffectFree(def))
      return failure();
    if (dependencies.contains(def))
      return success();

    SetVector<Value> operands(def->operand_begin(), def->operand_end());
    getUsedValuesDefinedAbove(def->getRegions(), operands);
    for (Value operand : operands)
      if (failed(collectDependency(operand)))
        return failure();
    dependencies.insert(def);
    return success();
  };
  return success(llvm::all_of(op->getOperands(), [&](Value operand) {
    return succeeded(collectDependency(operand));
  }));
}

bool LoopScopeMerger::sinkLoadsToUserScopes(
    ArrayRef<ScopeOp> allocaScopes,
    DenseMap<ScopeOp, AllocaAccesses> &accessesByAllocaScope) {
  bool changed = false;
  for (ScopeOp scope : allocaScopes) {
    auto it = accessesByAllocaScope.find(scope);
    ScopeOp loadScope = it->second.initialLoad.first;
    if (!loadScope)
      continue;

    Value retVal = it->second.initialLoad.second;
    auto returnOp =
        cast<scope::ReturnOp>(loadScope.getRegion().front().getTerminator());
    auto load = returnOp.getOperand(cast<OpResult>(retVal).getResultNumber())
                    .getDefiningOp<memref::LoadOp>();
    SetVector<Operation *> dependencies;
    if (failed(collectScopeLocalDependencies(load, loadScope, dependencies)))
      continue;

    SetVector<ScopeOp> userScopes;
    bool isNormalizedCounter =
        cast<memref::AllocaOp>(&scope.getRegion().front().front())
            ->hasAttr(kNormalizeMatmulCounterAttr);
    for (Operation *user : retVal.getUsers()) {
      auto userScope = user->getParentOfType<ScopeOp>();
      if (userScope && userScope->getBlock() == loadScope->getBlock() &&
          loadScope->isBeforeInBlock(userScope) &&
          (getScopeCoreType(userScope) == TCoreType::CUBE ||
           getScopeCoreType(userScope) == TCoreType::VECTOR ||
           (isNormalizedCounter &&
            getScopeCoreType(userScope) == TCoreType::CUBE_AND_VECTOR)))
        userScopes.insert(userScope);
    }

    for (ScopeOp userScope : userScopes) {
      OpBuilder builder =
          OpBuilder::atBlockBegin(&userScope.getRegion().front());
      IRMapping mapping;
      for (Operation *dependency : dependencies)
        builder.clone(*dependency, mapping);
      auto clonedLoad = cast<memref::LoadOp>(builder.clone(*load, mapping));
      retVal.replaceUsesWithIf(clonedLoad.getResult(), [&](OpOperand &use) {
        return userScope->isProperAncestor(use.getOwner());
      });
      changed = true;
    }

    if (!retVal.use_empty())
      continue;

    changed = true;
    if (loadScope.getNumResults() == 1 &&
        loadScope.getRegion().front().getOperations().size() ==
            dependencies.size() + 1) {
      loadScope.erase();
      continue;
    }

    // Drop the sunk result while preserving the source scope's other results
    // and operations, including any remaining internal uses of the load.
    SmallVector<Value> returnOperands;
    SmallVector<Value> keptResults;
    for (auto [result, returnedValue] :
         llvm::zip(loadScope.getResults(), returnOp.getOperands())) {
      if (result == retVal)
        continue;
      returnOperands.push_back(returnedValue);
      keptResults.push_back(result);
    }
    ScopeOp newScope = recreateScope(loadScope, returnOperands);
    for (auto [oldResult, newResult] :
         llvm::zip(keptResults, newScope.getResults())) {
      oldResult.replaceAllUsesWith(newResult);
      for (auto &entry : accessesByAllocaScope) {
        if (entry.second.initialLoad.second == oldResult)
          entry.second.initialLoad.second = newResult;
      }
    }
    if (load->use_empty())
      load.erase();
    for (Operation *dependency : llvm::reverse(dependencies))
      if (isOpTriviallyDead(dependency))
        dependency->erase();
    bool eraseScope = newScope.getNumResults() == 0 &&
                      newScope.getRegion().front().getOperations().size() == 1;
    for (auto &entry : accessesByAllocaScope) {
      AllocaAccesses &accesses = entry.second;
      if (accesses.initialLoad.first == loadScope) {
        if (accesses.initialLoad.second == retVal)
          accesses.initialLoad = {};
        else
          accesses.initialLoad.first = newScope;
      }
      if (accesses.initialStore.second == loadScope)
        accesses.initialStore.second = newScope;
    }
    loadScope.erase();
    if (eraseScope)
      newScope.erase();
  }
  return changed;
}

bool LoopScopeMerger::sinkStoresToUserScopes(
    ArrayRef<ScopeOp> allocaScopes,
    DenseMap<ScopeOp, AllocaAccesses> &accessesByAllocaScope) {
  bool changed = false;
  for (ScopeOp scope : allocaScopes) {
    auto it = accessesByAllocaScope.find(scope);
    memref::StoreOp store = it->second.initialStore.first;
    ScopeOp storeScope = it->second.initialStore.second;
    if (!store)
      continue;

    SetVector<Operation *> dependencies;
    if (failed(collectScopeLocalDependencies(store, storeScope, dependencies)))
      continue;

    Value retVal = scope.getResult(0);
    ScopeOp firstCubeScope;
    for (Operation *user : retVal.getUsers()) {
      ScopeOp parentScope = user->getParentOfType<ScopeOp>();
      if (getScopeCoreType(parentScope) != TCoreType::CUBE ||
          parentScope->getBlock() != storeScope->getBlock() ||
          !storeScope->isBeforeInBlock(parentScope))
        continue;

      // Use lists are not ordered by program position.
      if (!firstCubeScope || parentScope->isBeforeInBlock(firstCubeScope))
        firstCubeScope = parentScope;
    }
    if (!firstCubeScope)
      continue;

    OpBuilder builder =
        OpBuilder::atBlockBegin(&firstCubeScope.getRegion().front());
    IRMapping mapping;
    for (Operation *dependency : dependencies)
      builder.clone(*dependency, mapping);
    for (OpOperand &operand : store->getOpOperands())
      operand.set(mapping.lookupOrDefault(operand.get()));
    store->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
    it->second.initialStore.second = firstCubeScope;
    // Keep dependencies still used by other operations in the source scope.
    for (Operation *dependency : llvm::reverse(dependencies))
      if (isOpTriviallyDead(dependency))
        dependency->erase();
    if (storeScope.getNumResults() == 0 &&
        storeScope.getRegion().front().getOperations().size() == 1)
      storeScope.erase();
    changed = true;
  }
  return changed;
}

bool LoopScopeMerger::sinkAllocasToUserScopes(ArrayRef<ScopeOp> allocaScopes) {
  bool changed = false;
  for (ScopeOp scope : allocaScopes) {
    Value retVal = scope.getResult(0);
    ScopeOp firstScope;

    for (Operation *user : retVal.getUsers()) {
      Operation *ancestor = scope->getBlock()->findAncestorOpInBlock(*user);
      auto parentScope = dyn_cast_if_present<ScopeOp>(ancestor);
      if (!parentScope)
        continue;

      // Use lists are not ordered by program position.
      if (!firstScope || parentScope->isBeforeInBlock(firstScope))
        firstScope = parentScope;
    }
    if (!firstScope)
      continue;

    auto alloca = cast<memref::AllocaOp>(&scope.getRegion().front().front());

    OpBuilder builder =
        OpBuilder::atBlockBegin(&firstScope.getRegion().front());
    auto clonedAlloca = cast<memref::AllocaOp>(builder.clone(*alloca));
    retVal.replaceUsesWithIf(clonedAlloca.getResult(), [&](OpOperand &use) {
      return firstScope->isProperAncestor(use.getOwner());
    });
    changed = true;

    // Later scopes must share this allocation through a
    // scope result; the local alloca value cannot escape its region directly.
    if (llvm::none_of(retVal.getUsers(), [&](Operation *user) {
          return user->getParentOfType<ScopeOp>() != firstScope;
        }))
      continue;

    auto returnOp =
        cast<scope::ReturnOp>(firstScope.getRegion().front().getTerminator());
    SmallVector<Value> returnOperands(returnOp.getOperands());
    returnOperands.push_back(clonedAlloca.getResult());
    ScopeOp newScope = recreateScope(firstScope, returnOperands);
    for (auto [oldResult, newResult] : llvm::zip(
             firstScope.getResults(),
             newScope.getResults().take_front(firstScope.getNumResults())))
      oldResult.replaceAllUsesWith(newResult);
    retVal.replaceUsesWithIf(newScope.getResults().back(), [&](OpOperand &use) {
      return use.getOwner()->getParentOfType<ScopeOp>() != firstScope;
    });
    firstScope.erase();
  }

  for (ScopeOp scope : allocaScopes) {
    if (scope->use_empty()) {
      scope.erase();
      changed = true;
    }
  }
  return changed;
}

bool LoopScopeMerger::sinkAllocasInCAVScopes(bool onlyNormalizedCounters) {
  SmallVector<ScopeOp> allocaScopes;
  collectAllocaScopes(loopOp, allocaScopes);
  if (onlyNormalizedCounters)
    llvm::erase_if(allocaScopes, [](ScopeOp scope) {
      auto alloca = cast<memref::AllocaOp>(&scope.getRegion().front().front());
      return !alloca->hasAttr(kNormalizeMatmulCounterAttr);
    });
  if (allocaScopes.empty())
    return false;

  DenseMap<ScopeOp, AllocaAccesses> accessesByAllocaScope;
  collectAllocaAccesses(loopOp, allocaScopes, accessesByAllocaScope);
  bool changed = false;
  changed |= sinkLoadsToUserScopes(allocaScopes, accessesByAllocaScope);
  changed |= sinkStoresToUserScopes(allocaScopes, accessesByAllocaScope);
  changed |= sinkAllocasToUserScopes(allocaScopes);
  return changed;
}

bool LoopScopeMerger::applySinkAllocaInCAVScopeRule() {
  return sinkAllocasInCAVScopes(/*onlyNormalizedCounters=*/false);
}

bool LoopScopeMerger::applySinkNormalizedCounterInCAVScopeRule() {
  return sinkAllocasInCAVScopes(/*onlyNormalizedCounters=*/true);
}

void LoopScopeMerger::run(ArrayRef<MergeRule> rules) {
  for (const MergeRule &rule : rules) {
    if ((this->*rule.fn)())
      LLVM_DEBUG(writeScopeDependenceGraphToDotFile(loopOp, rule.dotPrefix));
  }
}

//===----------------------------------------------------------------------===//
// MergeCVScopesPass Implementation
//===----------------------------------------------------------------------===//

void MergeCVScopesPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (isCVPipelineSnapshot(funcOp) || hasLegacyCVPipelinedLoop(funcOp))
    return;
  static constexpr LoopScopeMerger::MergeRule kDefaultRuleOrder[] = {
      {&LoopScopeMerger::applySinkNormalizedCounterInCAVScopeRule,
       "merged.0-sink-normalized-counter"},
      {&LoopScopeMerger::applyScalarLoopCarriedDepRule, "merged.1-scalar-lcd"},
      {&LoopScopeMerger::applyLoopCarriedDepGroupingRule,
       "merged.2-lcd-grouping"},
      {&LoopScopeMerger::applyMergedScopeSameCoreRule,
       "merged.3-merged-scope-same-core"},
      {&LoopScopeMerger::applyCrossCoreTransferBackwardRule,
       "merged.4-cross-core-backward"},
      {&LoopScopeMerger::applySmallVectorScopeMergeRule,
       "merged.5-small-vector-merge"},
      {&LoopScopeMerger::applyGraphSameCoreRule, "merged.6-graph-same-core"},
      {&LoopScopeMerger::applySameCoreRunGroupingRule,
       "merged.7-same-core-run"},
      {&LoopScopeMerger::applySinkAllocaInCAVScopeRule,
       "merged.8-sink-alloca"}};

  SmallVector<scf::ForOp> targetLoops;
  funcOp.walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
    if (forOp->hasAttr(CVPipelinedLoopAttr::name))
      targetLoops.push_back(forOp);
  });

  for (scf::ForOp &forOp : targetLoops) {
    LoopScopeMerger merger(forOp);
    merger.run(kDefaultRuleOrder);
    forOp = merger.getLoopOp();
  }
}

std::unique_ptr<Pass> createMergeCVScopesPass() {
  return std::make_unique<MergeCVScopesPass>();
}

} // namespace hivm
} // namespace mlir
