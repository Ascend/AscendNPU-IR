//-------------------------CoreDependencyAnalysis.cpp-------------------------//
//
// Assigns each op a sub-block core by propagating V0/V1 through the operand DAG.
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

#define DEBUG_TYPE "hivm-partition-and-bind-sub-block"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::partition_and_bind;

namespace {

/// Raise `op`'s core in `out` toward `l`; returns true if the core changed.
bool raiseCore(CoreAssignment &out, const Operation *op, Core l) {
  Core &slot = out.opCore[op];
  Core next = join(slot, l);
  if (next == slot)
    return false;
  slot = next;
  return true;
}

/// A cube op that starts a new program-order cube stage.
bool opensNewStage(Operation *op) {
  return isa<hivm::MatmulOp, hivm::MmadL1Op, hivm::BatchMmadL1Op,
             hivm::MixMatmulOp>(op);
}

/// Only HIVM ops count as sub-core work.
bool isSubCoreWork(Operation &op) {
  return isa<scope::ScopeOp>(&op) ||
         op.getName().getDialectNamespace() == "hivm";
}

/// Recompute the per-stage V0/V1 work counts (flat op count for now).
void countStageLoads(Block &entry,
                     const llvm::DenseMap<Operation *, unsigned> &opStage,
                     unsigned numStages, const CoreAssignment &out,
                     llvm::SmallVectorImpl<unsigned> &v0PerStage,
                     llvm::SmallVectorImpl<unsigned> &v1PerStage) {
  v0PerStage.assign(numStages, 0);
  v1PerStage.assign(numStages, 0);
  for (Operation &opRef : entry) {
    if (!isSubCoreWork(opRef))
      continue;
    unsigned s = opStage.lookup(&opRef);
    Core c = out.coreOf(&opRef);
    if (c == Core::V0)
      ++v0PerStage[s];
    else if (c == Core::V1)
      ++v1PerStage[s];
  }
}

struct FixpipeReachability {
  using ConsumerMap =
      llvm::DenseMap<Operation *, llvm::SmallVector<Operation *, 4>>;

  FixpipeReachability(Block &entry, const CoreAssignment &out,
                      const ConsumerMap &consumers)
      : entry(entry), out(out), consumers(consumers) {}

  Block &entry;
  const CoreAssignment &out;
  const ConsumerMap &consumers;
  llvm::SmallVector<Operation *, 8> worklist;
  llvm::SmallPtrSet<Operation *, 16> visited;
  Core joined = Core::Bottom;

  Operation *topLevelOf(Operation *op) const {
    Operation *cur = op;
    while (cur && cur->getBlock() != &entry)
      cur = cur->getParentOp();
    return cur;
  }

  void enqueue(Operation *op) {
    Operation *top = topLevelOf(op);
    if (!top || !visited.insert(top).second)
      return;
    worklist.push_back(top);
    if (!isCubeOrSharedOp(top))
      joined = join(joined, out.coreOf(top));
  }

  void bridge(Operation *cube) {
    for (Value operand : cube->getOperands()) {
      if (!isa<BaseMemRefType>(operand.getType()))
        continue;
      llvm::SmallVector<Value, 4> aliasWork{operand};
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
      if (auto it = consumers.find(op); it != consumers.end())
        for (Operation *c : it->second)
          enqueue(c);
      if (isCubeOrSharedOp(op))
        bridge(op);
    }
    return joined;
  }
};

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

  // (2) Build the producer/consumer DAG over entry-block direct children.
  buildDependencyDag();

  // (3) spread the cores. Everything left is a free node for placeFreeNodes.
  propagateCores(out);

  // (4) put any node still at Bottom onto a concrete core.
  placeFreeNodes(out);

  // (5) set each fixpipe's destination from where its users landed.
  deriveFixpipeDestinations(out);

  return out;
}

//===----------------------------------------------------------------------===//
// (1) discoverSupernodes
//===----------------------------------------------------------------------===//

void CoreDependencyAnalysis::discoverSupernodes(CoreAssignment &out) {
  Block &entry = func::FuncOp(func).getFunctionBody().front();

  for (Operation &op : entry) {
    // Only `scope.scope` ops carrying `{sub_block = n}` anchor a supernode.
    auto scopeOp = dyn_cast<scope::ScopeOp>(&op);
    if (!scopeOp)
      continue;

    Core core = getSubBlockCoreOf(scopeOp.getOperation());
    if (!isSingleCore(core))
      continue;

    out.supernodes.push_back(Supernode(scopeOp, core));
    out.opCore[scopeOp.getOperation()] = core;
  }

  LDBG("discovered " << out.supernodes.size() << " supernode(s)");
}

//===----------------------------------------------------------------------===//
// (2) buildDependencyDag
//===----------------------------------------------------------------------===//

Operation *CoreDependencyAnalysis::definingTopLevelOp(Value v) const {
  Operation *def = v.getDefiningOp();
  if (!def)
    return nullptr;

  Block &entry = func::FuncOp(func).getFunctionBody().front();
  Operation *cur = def;
  while (cur && cur->getBlock() != &entry)
    cur = cur->getParentOp();

  // `cur` is now the entry-block ancestor of `def`, or null if `def` lives
  // outside this func's entry block entirely.
  return cur;
}

void CoreDependencyAnalysis::buildDependencyDag() {
  producers.clear();
  consumers.clear();

  Block &entry = func::FuncOp(func).getFunctionBody().front();

  for (Operation &op : entry) {
    Operation *consumer = &op;
    // Ensure every entry-block op appears as a key even with no producers, so
    // later passes can iterate the map uniformly.
    producers.try_emplace(consumer);
    consumers.try_emplace(consumer);

    // Walk all operands the op uses, including operands referenced from nested
    // regions
    llvm::SmallPtrSet<Operation *, 8> seenProducers;

    for (Value operand : op.getOperands())
      recordUse(consumer, operand, seenProducers);

    op.walk([&](Operation *nested) {
      if (nested == consumer)
        return;
      for (Value operand : nested->getOperands())
        recordUse(consumer, operand, seenProducers);
    });
  }
}

void CoreDependencyAnalysis::recordUse(
    Operation *consumer, Value operand,
    llvm::SmallPtrSet<Operation *, 8> &seenProducers) {
  Operation *producer = definingTopLevelOp(operand);
  if (!producer || producer == consumer)
    return;
  if (!seenProducers.insert(producer).second)
    return;
  producers[consumer].push_back(producer);
  consumers[producer].push_back(consumer);
}

//===----------------------------------------------------------------------===//
// (3) propagateCores
//===----------------------------------------------------------------------===//

void CoreDependencyAnalysis::propagateCores(CoreAssignment &out) {
  Block &entry = func::FuncOp(func).getFunctionBody().front();

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &opRef : entry) {
      Operation *op = &opRef;
      if (isCubeOrSharedOp(op))
        continue;
      Core c = out.coreOf(op);
      if (c == Core::Bottom)
        continue;
      if (auto it = producers.find(op); it != producers.end())
        for (Operation *p : it->second)
          if (!isCubeOrSharedOp(p))
            changed |= raiseCore(out, p, c);
      if (isSingleCore(c))
        if (auto it = consumers.find(op); it != consumers.end())
          for (Operation *u : it->second)
            if (!isCubeOrSharedOp(u))
              changed |= raiseCore(out, u, c);
    }
  }
}

//===----------------------------------------------------------------------===//
// (5) deriveFixpipeDestinations
//===----------------------------------------------------------------------===//

Core CoreDependencyAnalysis::fixpipeDestination(
    Operation *fixpipe, const CoreAssignment &out) const {
  Block &entry = func::FuncOp(func).getFunctionBody().front();
  FixpipeReachability reach{entry, out, consumers};
  return reach.run(fixpipe);
}

void CoreDependencyAnalysis::deriveFixpipeDestinations(CoreAssignment &out) {
  Block &entry = func::FuncOp(func).getFunctionBody().front();
  for (Operation &opRef : entry) {
    auto fixpipe = dyn_cast<hivm::FixpipeOp>(&opRef);
    if (!fixpipe)
      continue;
    // Only an L0C -> UB fixpipe is sub-block-targetable: sub_block_idx selects
    if (auto mt = dyn_cast<MemRefType>(fixpipe.getDst().getType())) {
      std::optional<hivm::AddressSpace> as =
          hivm::getOptionalHIVMAddressSpace(mt);
      if (!as || *as != hivm::AddressSpace::UB)
        continue;
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
  }

  LDBG(out.fixpipeSubBlock.size()
       << " sub-block, " << out.doubleWriteFixpipes.size()
       << " double-write fixpipe(s)");
}

//===----------------------------------------------------------------------===//
// (4) placeFreeNodes
//===----------------------------------------------------------------------===//

void CoreDependencyAnalysis::placeFreeNodes(CoreAssignment &out) {
  Block &entry = func::FuncOp(func).getFunctionBody().front();

  // Assign each entry-block op its cube-stage index in program order.
  llvm::DenseMap<Operation *, unsigned> opStage;
  unsigned stage = 0;
  for (Operation &opRef : entry) {
    if (opensNewStage(&opRef))
      ++stage;
    opStage[&opRef] = stage;
  }
  unsigned numStages = stage + 1;

  // Per-stage V0/V1 work counts (flat op count for now).
  llvm::SmallVector<unsigned, 8> v0PerStage, v1PerStage;
  countStageLoads(entry, opStage, numStages, out, v0PerStage, v1PerStage);

  for (Operation &opRef : entry) {
    Operation *op = &opRef;
    if (out.coreOf(op) != Core::Bottom || isCubeOrSharedOp(op))
      continue;
    unsigned s = opStage[op];
    out.opCore[op] = policy.placeFreeNode(op, v0PerStage[s], v1PerStage[s]);
    propagateCores(out);
    countStageLoads(entry, opStage, numStages, out, v0PerStage, v1PerStage);
  }
}
