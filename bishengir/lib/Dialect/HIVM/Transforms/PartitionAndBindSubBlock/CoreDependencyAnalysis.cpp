//-------------------------CoreDependencyAnalysis.cpp-------------------------//
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
#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/CoreDependencyAnalysis.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/PartitionTypes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "hivm-partition-and-bind-sub-block"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::partition_and_bind;

namespace {

/// Raise `valueCore[v]` in `out` toward `c`; returns true if the core changed.
bool raiseValue(CoreAssignment &out, Value v, Core c) {
  Core &slot = out.valueCore[v];
  Core next = join(slot, c);
  if (next == slot)
    return false;
  slot = next;
  return true;
}

/// A sub-block core lives only on buffers (tensors/memrefs); scalars are
/// replicated on every core and carry none.
bool carriesCore(Value v) { return isa<ShapedType>(v.getType()); }

/// The buffer operands that define an op's core and receive it by
/// back-propagation: all shaped operands for a result-less writer, only the DPS
/// inputs for a DPS op, else all shaped operands.
void forEachCoreOperand(Operation *op, llvm::function_ref<void(Value)> fn) {
  auto emitShaped = [&fn](Value v) {
    if (carriesCore(v))
      fn(v);
  };
  // An atomic simt scope has no SSA operands; its core comes from the buffers
  // it captures from the enclosing region.
  if (isAtomicSimtScope(op)) {
    llvm::SetVector<Value> captured;
    getUsedValuesDefinedAbove(op->getRegions(), captured);
    for (Value v : captured)
      emitShaped(v);
    return;
  }
  if (op->getResults().empty()) {
    for (Value operand : op->getOperands())
      emitShaped(operand);
    return;
  }
  if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
    for (Value in : dps.getDpsInputs())
      emitShaped(in);
    return;
  }
  for (Value operand : op->getOperands())
    emitShaped(operand);
}

/// Forward, guard-determining core: the join of the single cores of the buffer
/// values an op touches
Core forwardCore(const CoreAssignment &out, Operation *op) {
  Core c = Core::Bottom;
  auto addSingle = [&out, &c](Value v) {
    Core vc = out.valueCoreOf(v);
    if (isSingleCore(vc))
      c = join(c, vc);
  };
  forEachCoreOperand(op, addSingle);
  for (Value res : op->getResults())
    if (carriesCore(res))
      addSingle(res);
  return c;
}

/// The core an op's output resides on
Core residenceCore(const CoreAssignment &out, Operation *op) {
  Core c = Core::Bottom;
  bool hasShapedResult = false;
  for (Value res : op->getResults())
    if (carriesCore(res)) {
      hasShapedResult = true;
      c = join(c, out.valueCoreOf(res));
    }
  if (hasShapedResult)
    return c;
  return forwardCore(out, op);
}

bool bridgeWriterCoreToReaders(CoreAssignment &out, Operation *writer) {
  // Source core: the single-core join of the writer's tensor (non-memref)
  // inputs. The written memref is the carrier, not a core source.
  Core src = Core::Bottom;
  for (Value operand : writer->getOperands()) {
    if (!carriesCore(operand) || isa<BaseMemRefType>(operand.getType()))
      continue;
    if (Core c = out.valueCoreOf(operand); isSingleCore(c))
      src = join(src, c);
  }
  if (!isSingleCore(src))
    return false;

  // Collect the buffers the writer actually writes.
  auto effOp = dyn_cast<MemoryEffectOpInterface>(writer);
  if (!effOp)
    return false;
  llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
  effOp.getEffects(effects);

  bool changed = false;
  for (const MemoryEffects::EffectInstance &eff : effects) {
    if (!isa<MemoryEffects::Write>(eff.getEffect()))
      continue;
    Value written = eff.getValue();
    if (!written || !isa<BaseMemRefType>(written.getType()))
      continue;
    llvm::SmallVector<Value, 4> aliasWork{written};
    llvm::SmallPtrSet<Value, 4> aliasSeen;
    while (!aliasWork.empty()) {
      Value buf = aliasWork.pop_back_val();
      if (!aliasSeen.insert(buf).second)
        continue;
      for (Operation *user : buf.getUsers()) {
        if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(user))
          changed |= raiseValue(out, toTensor.getResult(), src);
        if (isa<bufferization::ToTensorOp, memref::MemorySpaceCastOp>(user) ||
            isa<ViewLikeOpInterface>(user))
          for (Value res : user->getResults())
            aliasWork.push_back(res);
      }
    }
  }
  return changed;
}

/// Derives a fixpipe's sub-block destination: the join of the cores of the ops
/// that read the buffer it writes
struct FixpipeReachability {
  FixpipeReachability(const CoreAssignment &out) : out(out) {}

  const CoreAssignment &out;
  llvm::SmallVector<Operation *, 8> worklist;
  llvm::SmallPtrSet<Operation *, 16> visited;
  Core joined = Core::Bottom;

  Core readerCore(Operation *op) const {
    for (Operation *p = op; p; p = p->getParentOp()) {
      if (Core sc = getSubBlockCoreOf(p); isSingleCore(sc))
        return sc;
      if (isAtomicSimtScope(p)) // a read inside a free simt scope takes its core.
        return out.coreOf(p);
    }
    return out.coreOf(op);
  }

  void enqueue(Operation *op) {
    if (!op || !visited.insert(op).second)
      return;
    worklist.push_back(op);
    if (isCubeOrSharedOp(op) ||
        isa<bufferization::ToTensorOp, memref::MemorySpaceCastOp>(op) ||
        isa<ViewLikeOpInterface>(op))
      return;
    joined = join(joined, readerCore(op));
  }

  void bridge(Operation *cube) {
    for (Value operand : cube->getOperands()) {
      if (!isa<BaseMemRefType>(operand.getType()))
        continue;
      llvm::SmallVector<Value, 4> aliasWork{traceBufferBase(operand)};
      llvm::SmallPtrSet<Value, 4> aliasSeen;
      while (!aliasWork.empty()) {
        Value buf = aliasWork.pop_back_val();
        if (!aliasSeen.insert(buf).second)
          continue;
        for (Operation *user : buf.getUsers()) {
          if (user == cube)
            continue;
          enqueue(user);
          if (isa<bufferization::ToTensorOp, memref::MemorySpaceCastOp>(user) ||
              isa<ViewLikeOpInterface>(user))
            for (Value res : user->getResults())
              aliasWork.push_back(res);
        }
      }
    }
  }

  Core run(Operation *fixpipe) {
    worklist.push_back(fixpipe);
    visited.insert(fixpipe);
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (isCubeOrSharedOp(op))
        bridge(op);
    }
    return joined;
  }
};

/// A scope/for/if op whose region-boundary cores are joined by `connect`. An
/// atomic simt scope is a leaf, so it is excluded.
bool isStructural(Operation *op) {
  return isa<scope::ScopeOp, scf::ForOp, scf::IfOp>(op) &&
         !isAtomicSimtScope(op);
}

/// Join the cores of the values connected across a scope/for/if region boundary
/// (result <-> yield, loop iter-arg/init). Returns true if any core changed.
bool connect(CoreAssignment &out, Operation *op) {
  bool changed = false;
  if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
    Operation *term = scopeOp.getRegion().front().getTerminator();
    for (unsigned i = 0, e = scopeOp.getNumResults(); i < e; ++i) {
      Value res = scopeOp.getResult(i);
      Value ret = term->getOperand(i);
      Core c = join(out.valueCoreOf(res), out.valueCoreOf(ret));
      if (c == Core::Bottom)
        continue;
      changed |= raiseValue(out, res, c);
      changed |= raiseValue(out, ret, c);
    }
    return changed;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    Operation *term = forOp.getBody()->getTerminator();
    for (unsigned i = 0, e = forOp.getNumResults(); i < e; ++i) {
      Value res = forOp.getResult(i);
      if (!carriesCore(res))
        continue;
      Value arg = forOp.getRegionIterArg(i);
      Value init = forOp.getInitArgs()[i];
      Value yld = term->getOperand(i);
      Core c = join(join(out.valueCoreOf(res), out.valueCoreOf(arg)),
                    join(out.valueCoreOf(init), out.valueCoreOf(yld)));
      if (c == Core::Bottom)
        continue;
      changed |= raiseValue(out, res, c);
      changed |= raiseValue(out, arg, c);
      changed |= raiseValue(out, init, c);
      changed |= raiseValue(out, yld, c);
    }
    return changed;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    for (unsigned i = 0, e = ifOp.getNumResults(); i < e; ++i) {
      Value res = ifOp.getResult(i);
      if (!carriesCore(res))
        continue;
      Value thenY = ifOp.thenYield().getOperand(i);
      Value elseY = ifOp.elseBlock() ? ifOp.elseYield().getOperand(i) : Value();
      Core c = join(out.valueCoreOf(res),
                    join(out.valueCoreOf(thenY),
                         elseY ? out.valueCoreOf(elseY) : Core::Bottom));
      if (c == Core::Bottom)
        continue;
      changed |= raiseValue(out, res, c);
      changed |= raiseValue(out, thenY, c);
      if (elseY)
        changed |= raiseValue(out, elseY, c);
    }
    return changed;
  }
  return false;
}

/// Forward: push a core along the data flow (operands -> results).
bool forwardStep(func::FuncOp func, CoreAssignment &out) {
  bool changed = false;
  func->walk([&](Operation *op) {
    if (isInsideAtomicSimtScope(op)) // leaf scope: interior is not partitioned.
      return;
    if (isStructural(op)) {
      changed |= connect(out, op);
      return;
    }
    if (isCubeOrSharedOp(op)) // cube carries no vector core.
      return;
    Core c = forwardCore(out, op);
    if (c == Core::Bottom)
      return;
    for (Value res : op->getResults())
      changed |= raiseValue(out, res, c);
  });
  return changed;
}

/// Backward: pull a core against the data flow (results -> operands).
bool backwardStep(func::FuncOp func, CoreAssignment &out) {
  bool changed = false;
  func->walk([&](Operation *op) {
    if (isInsideAtomicSimtScope(op)) // leaf scope: interior is not partitioned.
      return;
    if (isStructural(op)) {
      changed |= connect(out, op);
      return;
    }
    if (isCubeOrSharedOp(op))
      return;
    Core c = residenceCore(out, op);
    if (c != Core::Bottom)
      forEachCoreOperand(
          op, [&](Value operand) { changed |= raiseValue(out, operand, c); });
    if (op->getNumResults() == 0 && writesAnyBuffer(op))
      changed |= bridgeWriterCoreToReaders(out, op);
  });
  return changed;
}

/// The block's running (V0, V1) load, seeded on first use from its already
/// single-core ops (before any free node is placed there).
std::pair<unsigned, unsigned> &
loadFor(llvm::DenseMap<Block *, std::pair<unsigned, unsigned>> &blockLoad,
        const CoreAssignment &out, Block *b) {
  auto [it, inserted] = blockLoad.try_emplace(b);
  if (inserted) {
    unsigned v0 = 0, v1 = 0;
    for (Operation &child : *b) {
      if (out.coreOf(&child) == Core::V0)
        ++v0;
      else if (out.coreOf(&child) == Core::V1)
        ++v1;
    }
    it->second = {v0, v1};
  }
  return it->second;
}

} // namespace

//===----------------------------------------------------------------------===//
// DefaultFreeNodePlacementPolicy
//===----------------------------------------------------------------------===//

Core DefaultFreeNodePlacementPolicy::placeFreeNode(Operation * /*op*/,
                                                   unsigned /*currentV0Load*/,
                                                   unsigned /*currentV1Load*/) {
  return Core::V0;
}

//===----------------------------------------------------------------------===//
// LoadBalancedFreeNodePlacementPolicy
//===----------------------------------------------------------------------===//

/// Place a free node on the currently lighter sub-core
Core LoadBalancedFreeNodePlacementPolicy::placeFreeNode(
    Operation * /*op*/, unsigned currentV0Load, unsigned currentV1Load) {
  return currentV1Load < currentV0Load ? Core::V1 : Core::V0;
}

Core CoreAssignment::coreOf(Operation *op) const {
  auto it = opCore.find(op);
  return it == opCore.end() ? Core::Bottom : it->second;
}

CoreAssignment CoreDependencyAnalysis::run() {
  CoreAssignment out;

  // (1) Discover the `{sub_block}` scopes.
  discoverSupernodes(out);

  // (2) Propagate the core to a fixpoint over SSA def-use edges.
  propagateValues(out);

  // (3) Derive each guardable op's core from the values it touches.
  deriveOpCores(out);

  // (4) Place any guardable op still at Bottom onto a concrete core.
  placeFreeNodes(out);

  // (5) set each fixpipe's destination from where its users landed.
  deriveFixpipeDestinations(out);

  return out;
}

//===----------------------------------------------------------------------===//
// (1) discoverSupernodes
//===----------------------------------------------------------------------===//

void CoreDependencyAnalysis::discoverSupernodes(CoreAssignment &out) {
  func->walk([&](scope::ScopeOp scopeOp) {
    Core core = getSubBlockCoreOf(scopeOp.getOperation());
    if (!isSingleCore(core))
      return;
    out.supernodes.push_back(Supernode(scopeOp, core));
    out.opCore[scopeOp.getOperation()] = core;
    for (Value res : scopeOp.getResults())
      out.valueCore[res] = core;
  });
  LDBG("discovered " << out.supernodes.size() << " supernode(s)");
}

//===----------------------------------------------------------------------===//
// (2) propagateValues
//===----------------------------------------------------------------------===//

void CoreDependencyAnalysis::propagateValues(CoreAssignment &out) {

  bool outerChanged = true;
  while (outerChanged) {
    outerChanged = false;
    while (forwardStep(func, out))
      outerChanged = true;
    while (backwardStep(func, out))
      outerChanged = true;
  }
}

//===----------------------------------------------------------------------===//
// (3) deriveOpCores
//===----------------------------------------------------------------------===//

void CoreDependencyAnalysis::deriveOpCores(CoreAssignment &out) {
  func->walk([&](Operation *op) {
    if (isInsideAtomicSimtScope(op)) // leaf scope: interior is not partitioned.
      return;
    if (isCubeOrSharedOp(op) || op->hasTrait<OpTrait::IsTerminator>())
      return;
    if (isa<scope::ScopeOp>(op) && !isAtomicSimtScope(op)) // scopes carry a core.
      return;
    if (isa<scf::ForOp, scf::IfOp>(op)) // shared control-flow: never guarded.
      return;
    out.opCore[op] = residenceCore(out, op);
  });
}

//===----------------------------------------------------------------------===//
// (5) deriveFixpipeDestinations
//===----------------------------------------------------------------------===//

Core CoreDependencyAnalysis::fixpipeDestination(
    Operation *fixpipe, const CoreAssignment &out) const {
  FixpipeReachability reach{out};
  return reach.run(fixpipe);
}

void CoreDependencyAnalysis::deriveFixpipeDestinations(CoreAssignment &out) {
  func->walk([&](hivm::FixpipeOp fixpipe) {
    // Only an L0C -> UB fixpipe is the target
    if (auto mt = dyn_cast<MemRefType>(fixpipe.getDst().getType())) {
      std::optional<hivm::AddressSpace> as =
          hivm::getOptionalHIVMAddressSpace(mt);
      if (!as || *as != hivm::AddressSpace::UB)
        return;
    }
    switch (fixpipeDestination(fixpipe.getOperation(), out)) {
    case Core::Both:
      out.doubleWriteFixpipes.insert(fixpipe.getOperation());
      break;
    case Core::V0:
      out.fixpipeSubBlock[fixpipe.getOperation()] = 0;
      break;
    case Core::V1:
      out.fixpipeSubBlock[fixpipe.getOperation()] = 1;
      break;
    case Core::Bottom: // no sub-core reader -- leave default.
      break;
    }
  });

  LDBG(out.fixpipeSubBlock.size()
       << " sub-block, " << out.doubleWriteFixpipes.size()
       << " double-write fixpipe(s)");
}

//===----------------------------------------------------------------------===//
// (4) placeFreeNodes
//===----------------------------------------------------------------------===//

void CoreDependencyAnalysis::placeFreeNodes(CoreAssignment &out) {
  // Collect the free (Bottom, guardable) ops in program order.
  llvm::SmallVector<Operation *, 32> freeOps;
  llvm::SmallPtrSet<Operation *, 32> freeSet;
  func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isInsideAtomicSimtScope(op))
      return;
    if (isCubeOrSharedOp(op) || op->hasTrait<OpTrait::IsTerminator>())
      return;
    if (isStructural(op)) // non-simt scopes + loops: never pinned.
      return;
    if (!isGuardable(op))
      return;
    if (out.opCore.lookup(op) != Core::Bottom)
      return; // already assigned by value propagation.
    freeOps.push_back(op);
    freeSet.insert(op);
  });

  llvm::EquivalenceClasses<Operation *> components;
  for (Operation *op : freeOps)
    components.insert(op);
  for (Operation *op : freeOps)
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp();
          def && freeSet.contains(def))
        components.unionSets(op, def);

  llvm::DenseMap<Block *, std::pair<unsigned, unsigned>> blockLoad;
  llvm::DenseMap<Operation *, Core> componentCore;

  for (Operation *op : freeOps) {
    std::pair<unsigned, unsigned> &load =
        loadFor(blockLoad, out, op->getBlock());
    Operation *leader = components.getLeaderValue(op);
    auto [it, isNew] = componentCore.try_emplace(leader, Core::Bottom);
    if (isNew)
      it->second = policy.placeFreeNode(op, load.first, load.second);
    Core core = it->second;
    out.opCore[op] = core;
    if (core == Core::V0)
      ++load.first;
    else if (core == Core::V1)
      ++load.second;
  }
}
