//===- BufferPoolSync.cpp - Buffer pool synchronization -----------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/BufferPoolSync.h"
#include "bishengir/Dialect/HIVM/Transforms/BufferPoolUtils.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::syncsolver;
using namespace mlir::hivm::buffer_pool;

LogicalResult BufferPoolSync::collect(func::FuncOp function, OperationBase *ir,
                                      const SyncSolverOptions &options) {
  if (!function || !options.isIntraCoreMode())
    return success();
  auto result = function.walk([&](PoolSlotOp slot) -> WalkResult {
    auto root = slot.getSource().getDefiningOp<PointerCastOp>();
    auto loop = slot->getParentOfType<LoopLikeOpInterface>();
    Block *body = getLoopBodyBlock(loop, slot);
    if (!options.isRegBasedArch || !body || !root ||
        getOptionalHIVMAddressSpace(root.getType()) != AddressSpace::L1) {
      slot.emitError("explicit buffer slots require L1 storage in a RegBase "
                     "scheduling loop");
      return WalkResult::interrupt();
    }
    annotation::MarkOp end;
    bool producer = false;
    for (Operation *user : slot->getUsers()) {
      if (!user->hasAttr(ExperimentalPoolReadyAttr::name) &&
          !user->hasAttr(ExperimentalPoolReleaseAttr::name))
        continue;
      auto mark = dyn_cast<annotation::MarkOp>(user);
      if (!mark || end ||
          user->hasAttr(ExperimentalPoolReadyAttr::name) ==
              user->hasAttr(ExperimentalPoolReleaseAttr::name) ||
          user->getBlock() != slot->getBlock() ||
          !slot->isBeforeInBlock(user)) {
        slot.emitError("invalid or duplicate buffer protocol boundary");
        return WalkResult::interrupt();
      }
      end = mark;
      producer = user->hasAttr(ExperimentalPoolReadyAttr::name);
    }
    if (!end) {
      slot.emitError("missing ready/release boundary from EnableBufferPool");
      return WalkResult::interrupt();
    }
    auto mark = utils::getAnnotateOpWithAttr(root, MemPoolIdAttr::name);
    if (!mark)
      mark = utils::getAnnotateOpWithAttr(root, MemPoolSizeAttr::name);
    if (!root->hasAttr(ExperimentalBufferSlotsAttr::name)) {
      slot.emitError("missing explicit buffer table annotation");
      return WalkResult::interrupt();
    }
    std::optional<int64_t> id;
    auto poolId = mark
                      ? (*mark)->getAttrOfType<IntegerAttr>(MemPoolIdAttr::name)
                      : IntegerAttr{};
    auto size = mark
                    ? (*mark)->getAttrOfType<IntegerAttr>(MemPoolSizeAttr::name)
                    : IntegerAttr{};
    int64_t count = root.getAddrs().size();
    if (count < 1 || count > 8 || bool(poolId) != bool(size) ||
        (size && (size.getInt() != count || poolId.getInt() < 0))) {
      slot.emitError("inconsistent buffer pool identity or table size");
      return WalkResult::interrupt();
    }
    if (poolId)
      id = poolId.getInt();
    size_t group = 0;
    for (; group < pools.size(); ++group) {
      Pool &pool = pools[group];
      if (pool.body != body || pool.id != id || (!id && pool.root != root))
        continue;
      break;
    }
    if (group == pools.size())
      pools.push_back({loop, body, root, id, count});
    bindingIndex[slot.getResult()] = bindings.size();
    Binding binding;
    binding.slot = slot;
    binding.end = end;
    binding.group = group;
    binding.producer = producer;
    bindings.push_back(std::move(binding));
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  if (bindings.empty())
    return success();

  // Reuse GSS's MTE2 writes and virtual MTE1 MMAD readers. Derived views
  // retain their pool binding through the ordinary memory analysis.
  bool valid = true;
  DenseMap<Operation *, OperationBase *> operationIR;
  auto visit = [&](OperationBase *op, auto &recurse) -> void {
    if (op->op)
      operationIR.try_emplace(op->op, op);
    if (isa<Loop>(op))
      for (Pool &pool : pools)
        if (pool.loop == op->op)
          pool.loopIR = cast<Loop>(op);
    if (auto *scope = dyn_cast<syncsolver::Scope>(op)) {
      for (auto &child : scope->body)
        recurse(child.get(), recurse);
      return;
    }
    auto *rw = dyn_cast<RWOperation>(op);
    if (!rw)
      return;
    auto collect = [&](ArrayRef<MemInfo> accesses, bool write, PIPE pipe) {
      for (const MemInfo &info : accesses) {
        auto it = bindingIndex.find(info.value);
        if (it == bindingIndex.end())
          continue;
        Binding &binding = bindings[it->second];
        Operation *access = rw->op;
        // MmadL1's virtual MTE1 readers have no MLIR operation of their own.
        for (OperationBase *parent = rw->parentOp; !access && parent;
             parent = parent->parentOp)
          access = parent->op;
        while (access && access->getBlock() != binding.slot->getBlock())
          access = access->getParentOp();
        if (write != binding.producer || !access ||
            !binding.slot->isBeforeInBlock(access) ||
            !access->isBeforeInBlock(binding.end)) {
          binding.slot.emitError(
              "access is outside its buffer protocol lifetime");
          valid = false;
          continue;
        }
        PIPE actualPipe = info.pipe.value_or(pipe);
        if (actualPipe != (write ? PIPE::PIPE_MTE2 : PIPE::PIPE_MTE1)) {
          binding.slot.emitError("L1 buffer pools require MTE2 writes and MTE1 "
                                 "reads");
          valid = false;
          continue;
        }
        auto &pipeAccesses = binding.accesses;
        if (pipeAccesses.empty() || pipeAccesses.back() != rw)
          pipeAccesses.push_back(rw);
        auto *top = operationIR.lookup(access);
        assert(top && "translated access must have a scope or operation");
        binding.scope = cast<syncsolver::Scope>(top->parentOp);
        Pool &pool = pools[binding.group];
        (write ? pool.hasProducer : pool.hasConsumer) = true;
      }
    };
    collect(rw->readMemInfo, false, rw->pipeRead);
    collect(rw->writeMemInfo, true, rw->pipeWrite);
  };
  visit(ir, visit);
  for (Binding &binding : bindings) {
    if (!binding.scope) {
      binding.slot.emitError("buffer binding has no modelled memory accesses");
      valid = false;
    }
  }
  if (!valid)
    return failure();
  for (Pool &pool : pools) {
    if (!pool.loopIR)
      return pool.loop->emitError("missing translated buffer pool loop");
    if (!pool.hasProducer || !pool.hasConsumer)
      return pool.loop->emitError(
          "buffer protocol requires producers and consumers");
  }

  // Keep pre-existing flags disjoint from the protocol. Macro-pinned IDs are
  // reserved by CustomMacroSyncState when the ordinary solver resets.
  result = function.walk([&](Operation *op) -> WalkResult {
    if (!isa<hivm::SetFlagOp, hivm::WaitFlagOp>(op))
      return WalkResult::advance();
    auto src = op->getAttrOfType<PipeAttr>("set_pipe").getPipe();
    auto dst = op->getAttrOfType<PipeAttr>("wait_pipe").getPipe();
    if (!((src == PIPE::PIPE_MTE2 && dst == PIPE::PIPE_MTE1) ||
          (src == PIPE::PIPE_MTE1 && dst == PIPE::PIPE_MTE2)))
      return WalkResult::advance();
    auto event = op->getAttrOfType<EventAttr>("static_event_id");
    if (!event) {
      op->emitError(
          "cannot reserve a pre-existing dynamic event for buffer pools");
      return WalkResult::interrupt();
    }
    auto &ids = fixedIds[{src, dst}];
    int64_t id = static_cast<int64_t>(event.getEvent());
    if (!llvm::is_contained(ids, id))
      ids.push_back(id);
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void BufferPoolSync::reset(GetSolver getSolver) {
  clearEventIdNodes();
  for (const auto &[pair, ids] : fixedIds)
    for (int64_t id : ids)
      getSolver(pair.first, pair.second)->reserveEventId(id);
}

bool BufferPoolSync::managesEventIds(PIPE src, PIPE dst) const {
  return !pools.empty() &&
         ((src == PIPE::PIPE_MTE2 && dst == PIPE::PIPE_MTE1) ||
          (src == PIPE::PIPE_MTE1 && dst == PIPE::PIPE_MTE2));
}

EventIdNode *BufferPoolSync::getEventIdNode(const BufferPoolInfo &pool) const {
  return sharedEventIdNodes.lookup(
      std::make_tuple(pool.loop, pool.resource, unsigned(pool.ready)));
}

void BufferPoolSync::setEventIdNode(const BufferPoolInfo &pool,
                                    EventIdNode *node) {
  auto &stored = sharedEventIdNodes[std::make_tuple(pool.loop, pool.resource,
                                                    unsigned(pool.ready))];
  assert(!stored || stored == node);
  stored = node;
}

LogicalResult BufferPoolSync::forEachDependency(
    const DenseMap<OperationBase *, std::vector<Occurrence *>> &occurrences,
    llvm::function_ref<LogicalResult(Occurrence *, Occurrence *, PIPE, PIPE,
                                     EventIdInfo)>
        visit) {
  auto visitPair = [&](RWOperation *set, RWOperation *wait, PIPE src, PIPE dst,
                       EventIdInfo info) -> LogicalResult {
    const auto &pool = *info.bufferPoolInfo;
    for (Occurrence *setOcc : occurrences.lookup(set)) {
      auto *loopOcc = setOcc->getParentWithOp(pool.loop);
      if (loopOcc->loopSplitIndex < 0 ||
          setOcc->syncIrIndex >= loopOcc->loopSplitIndex)
        continue;
      for (Occurrence *waitOcc : occurrences.lookup(wait)) {
        if (waitOcc->getParentWithOp(pool.loop) != loopOcc ||
            setOcc->endIndex >= waitOcc->startIndex ||
            (waitOcc->syncIrIndex < loopOcc->loopSplitIndex) != pool.ready)
          continue;
        return visit(setOcc, waitOcc, src, dst, info);
      }
    }
    llvm_unreachable("missing pool dependency occurrences");
  };
  // Pair each stream's producer and consumer. Members of the same pool share
  // their event node, rather than creating cross-member memory dependencies.
  for (Binding &producer : bindings) {
    if (!producer.producer)
      continue;
    for (Binding &consumer : bindings) {
      if (consumer.producer || producer.group != consumer.group ||
          producer.slot.getSource() != consumer.slot.getSource())
        continue;
      const Pool &pool = pools[producer.group];
      for (bool ready : {true, false}) {
        PIPE src = ready ? PIPE::PIPE_MTE2 : PIPE::PIPE_MTE1;
        PIPE dst = ready ? PIPE::PIPE_MTE1 : PIPE::PIPE_MTE2;
        Binding &set = ready ? producer : consumer;
        Binding &wait = ready ? consumer : producer;
        RWOperation *first = wait.accesses.front();
        // Padding retains ordinary GSS protection. Slot reuse waits at the DMA.
        if (!ready) {
          auto dma = llvm::find_if(wait.accesses, [](RWOperation *op) {
            return !dyn_cast_or_null<VBrcOp>(op->op);
          });
          if (dma != wait.accesses.end())
            first = *dma;
        }
        EventIdInfo info(pool.size);
        info.cannotRepeatFlagId = true;
        info.bufferPoolInfo =
            BufferPoolInfo{pool.root, set.slot.getSlot(), wait.slot.getSlot(),
                           set.scope, wait.scope,         pool.loopIR,
                           ready};
        if (failed(visitPair(set.accesses.back(), first, src, dst, info)))
          return failure();
      }
    }
  }
  return success();
}

bool BufferPoolSync::hasAccess(RWOperation *op) const {
  if (bindings.empty())
    return false;
  auto isBinding = [&](const MemInfo &info) {
    return info.poolSlotInfo &&
           bindingIndex.contains(info.poolSlotInfo->binding->getResult(0));
  };
  return llvm::any_of(op->readMemInfo, isBinding) ||
         llvm::any_of(op->writeMemInfo, isBinding);
}

bool BufferPoolSync::covers(RWOperation *a, RWOperation *b,
                            const MemInfo &infoA, const MemInfo &infoB) const {
  if (!infoA.poolSlotInfo || !infoB.poolSlotInfo)
    return false;
  auto lhs = bindingIndex.find(infoA.poolSlotInfo->binding->getResult(0));
  auto rhs = bindingIndex.find(infoB.poolSlotInfo->binding->getResult(0));
  if (lhs == bindingIndex.end() || rhs == bindingIndex.end() ||
      bindings[lhs->second].group != bindings[rhs->second].group)
    return false;
  if (lhs == rhs && a->preOrderIndex < b->preOrderIndex)
    return false; // Preserve ordering within one actual submission.

  auto containsAccess = [&](size_t index, RWOperation *op,
                            const MemInfo &info) {
    const Binding &binding = bindings[index];
    PIPE pipe =
        info.pipe.value_or(binding.producer ? op->pipeWrite : op->pipeRead);
    return pipe == (binding.producer ? PIPE::PIPE_MTE2 : PIPE::PIPE_MTE1) &&
           llvm::is_contained(binding.accesses, op);
  };
  if (!containsAccess(lhs->second, a, infoA) ||
      !containsAccess(rhs->second, b, infoB))
    return false;

  // L1 padding is protected by ordinary GSS before the pool's DMA wait.
  auto padding = dyn_cast_or_null<VBrcOp>(b->op);
  if (padding && padding.getPipe() == PIPE::PIPE_MTE2 &&
      a->pipeRead == PIPE::PIPE_MTE1)
    return false;

  const Binding &bindingA = bindings[lhs->second];
  const Binding &bindingB = bindings[rhs->second];
  if (bindingA.producer != bindingB.producer)
    return true; // The fixed ready/free edge covers this pair.
  // Slot reuse orders two producers through the intervening consumer:
  // old write -> ready -> read -> free -> new write. Two readers need no edge.
  return bindingA.producer;
}

void BufferPoolSync::eraseBoundaries() {
  for (Binding &binding : bindings)
    binding.end.erase();
}
