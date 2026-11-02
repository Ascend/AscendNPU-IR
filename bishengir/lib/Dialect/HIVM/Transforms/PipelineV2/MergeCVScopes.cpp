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
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
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

#include <functional>

#define DEBUG_TYPE "merge-cv-scopes"

namespace mlir {
namespace hivm {

#define GEN_PASS_DEF_MERGECVSCOPES
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

namespace {

class ScopeDependenceGraph {
public:
  explicit ScopeDependenceGraph(scf::ForOp loop, bool withCarriedEdges = true);
  unsigned getSize() const { return nodes.size(); }
  Operation *getNode(unsigned i) const { return nodes[i]; }
  std::optional<unsigned> getIndex(Operation *op) const;
  ArrayRef<unsigned> getSuccessors(unsigned i) const { return intra[i]; }
  SmallVector<SmallVector<Operation *>> formGroups() const;
  void printSummary(llvm::raw_ostream &os,
                    ArrayRef<SmallVector<Operation *>> groups) const;

private:
  struct MemAccess {
    bool read = false;
    bool write = false;
  };

  struct NodeMemInfo {
    llvm::MapVector<Value, MemAccess> buffers;
    bool globalRead = false;
    bool globalWrite = false;
  };

  struct CarriedEdge {
    unsigned from;
    unsigned to;
    unsigned argIndex;
  };

  static Value getMemBase(Value v);
  static void recordMemref(NodeMemInfo &info, Value v, bool isWrite);
  static void collectMemAccesses(Operation *top, NodeMemInfo &info);
  static bool hasMemConflict(const NodeMemInfo &a, const NodeMemInfo &b);
  static SmallVector<SmallVector<unsigned>>
  computeSCCs(ArrayRef<SmallVector<unsigned>> succ);

  void buildIntraEdges();
  void buildCarriedEdges();

  scf::ForOp loop;
  Block *body;
  SmallVector<Operation *> nodes;
  DenseMap<Operation *, unsigned> index;
  SmallVector<SmallVector<unsigned>> intra;
  SmallVector<CarriedEdge> carried;
};

class LoopScopeMerger {
public:
  using MergeRuleFn = void (LoopScopeMerger::*)();

  explicit LoopScopeMerger(scf::ForOp loopOp) : loopOp(loopOp) {}

  void run(ArrayRef<MergeRuleFn> rules);

  // Rule 1: Group scopes connected by loop-carried dependencies into an outer
  // `{cv_merged_scope}` wrapper scope, keeping member scopes nested inside so
  // cross-core boundaries remain visible.
  void applyLoopCarriedDepGroupingRule();

  // Rule 2: Within each `{cv_merged_scope}`, merge compatible scopes along
  // intra-iteration dependencies, and if the `{cv_merged_scope}` reduces to a
  // single inner scope, collapse it into a single work item.
  void applyMergedScopeSameCoreRule();

  // Rule 3: Starting from cross-core transfers (fixpipe -> UB or copy -> CBUF),
  // traverse backward along intra-iteration scope dependencies within the same
  // parent block and merge compatible scopes, stopping traversal at any scope
  // that has an incompatible dependency.
  void applyCrossCoreTransferBackwardRule();

private:
  static bool isCrossCoreTransferOp(Operation *op);
  static bool hasCrossCoreTransfer(ScopeOp scope);
  static bool isEligibleForCoreMerge(ScopeOp scope);
  static void collapseSingleChildMergedScope(ScopeOp outerScope);

  SetVector<ScopeOp> getIntraIterScopeDeps(ScopeOp scope);
  ScopeOp mergeScopeGroup(SmallPtrSetImpl<Operation *> &group);
  ScopeOp tryMergeBackwardFromSeed(ScopeOp seed);

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
wrapGroupInScope(scf::ForOp loop, ArrayRef<Operation *> members) {
  if (members.empty())
    return failure();

  Block *body = loop.getBody();
  ScopeDependenceGraph g(loop, /*withCarriedEdges=*/false);

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

void LoopScopeMerger::applyLoopCarriedDepGroupingRule() {
  ScopeDependenceGraph graph(loopOp);
  SmallVector<SmallVector<Operation *>> groups = graph.formGroups();
  LLVM_DEBUG(graph.printSummary(llvm::dbgs(), groups));

  for (SmallVector<Operation *> &group : groups) {
    if (failed(wrapGroupInScope(loopOp, group)))
      loopOp->emitWarning("[MergeCVScopes] loop-carried group is not convex; "
                          "left unmerged");
  }
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

void LoopScopeMerger::collapseSingleChildMergedScope(ScopeOp outerScope) {
  if (!outerScope || !outerScope->hasAttr(kCVMergedScopeAttrName))
    return;
  Block &outerBody = outerScope.getRegion().front();
  if (!llvm::hasSingleElement(outerBody.without_terminator()))
    return;
  auto innerScope = dyn_cast<ScopeOp>(&outerBody.front());
  if (!innerScope || !isEligibleForCoreMerge(innerScope))
    return;

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

void LoopScopeMerger::applyMergedScopeSameCoreRule() {
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
      if (ScopeOp merged = tryMergeBackwardFromSeed(seed))
        processedSeeds.insert(merged.getOperation());
    }
    collapseSingleChildMergedScope(outerScope);
  }
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
    if (ScopeOp merged = tryMergeBackwardFromSeed(seed))
      processedSeeds.insert(merged.getOperation());
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
      &LoopScopeMerger::applyLoopCarriedDepGroupingRule,
      &LoopScopeMerger::applyMergedScopeSameCoreRule,
      &LoopScopeMerger::applyCrossCoreTransferBackwardRule,
  };

  SmallVector<scf::ForOp> targetLoops;
  getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
    if (forOp->hasAttr(CVPipelinedLoopAttr::name))
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
