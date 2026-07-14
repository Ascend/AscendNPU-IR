//===-------- CustomMacroSync.cpp ---- Graph Sync Solver -----------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/CustomMacroSync.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <deque>

namespace mlir::hivm::syncsolver {
namespace {

struct OrderedFuncIr {
  llvm::SmallVector<OperationBase *> ops;
  llvm::DenseMap<Operation *, RWOperation *> mlirToRw;
};

OrderedFuncIr collectOrderedFuncIr(OperationBase *root) {
  OrderedFuncIr result;
  if (!root)
    return result;

  auto collectOpsInOrder = [&](OperationBase *op, auto &recurse) -> void {
    if (auto *scope = llvm::dyn_cast<Scope>(op)) {
      for (auto &child : scope->body)
        recurse(child.get(), recurse);
      return;
    }
    if (auto *loop = llvm::dyn_cast<Loop>(op)) {
      for (auto &child : loop->body)
        recurse(child.get(), recurse);
      return;
    }
    result.ops.push_back(op);
  };
  collectOpsInOrder(root, collectOpsInOrder);

  for (auto *op : result.ops) {
    if (auto *rw = llvm::dyn_cast<RWOperation>(op))
      if (rw->op)
        result.mlirToRw[rw->op] = rw;
  }
  return result;
}

std::optional<llvm::SmallVector<int64_t>>
matchSetFlagEventIds(const std::deque<std::unique_ptr<SyncOp>> &syncOps,
                     hivm::PIPE src, hivm::PIPE dst) {
  for (auto &syncOp : syncOps) {
    auto *setOp = llvm::dyn_cast<SetFlagOp>(syncOp.get());
    if (setOp && setOp->pipeSrc == src && setOp->pipeDst == dst &&
        !setOp->eventIds.empty())
      return setOp->eventIds;
  }
  return std::nullopt;
}

std::optional<llvm::SmallVector<int64_t>> findSetFlagEventIdsBeforeOp(
    llvm::ArrayRef<OperationBase *> orderedOps, OperationBase *beforeOp,
    const SyncMap &syncMapAfter, hivm::PIPE src, hivm::PIPE dst) {
  for (auto *op : orderedOps) {
    if (op == beforeOp)
      break;
    auto it = syncMapAfter.find(op);
    if (it == syncMapAfter.end())
      continue;
    if (auto ids = matchSetFlagEventIds(it->second, src, dst))
      return ids;
  }
  return std::nullopt;
}

std::optional<llvm::SmallVector<int64_t>>
findSetFlagEventIdsInMaps(const SyncMap &syncMapBefore,
                          const SyncMap &syncMapAfter, hivm::PIPE src,
                          hivm::PIPE dst) {
  for (const auto *map : {&syncMapAfter, &syncMapBefore}) {
    for (auto &[op, syncOps] : *map) {
      (void)op;
      if (auto ids = matchSetFlagEventIds(syncOps, src, dst))
        return ids;
    }
  }
  return std::nullopt;
}

bool hasExternalSetFlag(llvm::ArrayRef<OperationBase *> orderedOps,
                        OperationBase *beforeOp, const SyncMap &syncMapBefore,
                        const SyncMap &syncMapAfter, hivm::PIPE setPipe,
                        hivm::PIPE waitPipe) {
  return findSetFlagEventIdsBeforeOp(orderedOps, beforeOp, syncMapAfter,
                                     setPipe, waitPipe)
             .has_value() ||
         findSetFlagEventIdsInMaps(syncMapBefore, syncMapAfter, setPipe,
                                   waitPipe)
             .has_value();
}

llvm::SmallVector<int64_t>
resolveSyncEventSlotEventIds(hivm::SyncEventSlotAttr slot, OperationBase *rwOp,
                             llvm::ArrayRef<OperationBase *> orderedOps,
                             const SyncMap &syncMapBefore,
                             const SyncMap &syncMapAfter) {
  if (auto pinned = getSlotPinnedEventId(slot))
    return {*pinned};

  if (!slotHasExplicitPipes(slot))
    return {};

  auto setPipe = getSlotSetPipe(slot);
  auto waitPipe = getSlotWaitPipe(slot);
  if (auto ids = findSetFlagEventIdsBeforeOp(orderedOps, rwOp, syncMapAfter,
                                             setPipe, waitPipe))
    return *ids;
  if (auto ids = findSetFlagEventIdsInMaps(syncMapBefore, syncMapAfter, setPipe,
                                           waitPipe))
    return *ids;
  return {0};
}

std::optional<int64_t> resolveInternalSlotEventId(
    hivm::SyncEventSlotAttr slot, hivm::CustomMacroOp macro,
    const SyncSolverOptions &options,
    llvm::function_ref<std::unique_ptr<EventIdSolver> &(hivm::PIPE, hivm::PIPE)>
        getEventIdSolver) {
  if (auto pinned = getSlotPinnedEventId(slot))
    return pinned;

  auto [setPipe, waitPipe] = resolveSlotPipePair(slot, macro);
  const int64_t eventIdMax =
      getHWAvailableEventIdNum(options.syncMode, setPipe, waitPipe);
  auto allocated =
      getEventIdSolver(setPipe, waitPipe)->allocateUnusedEventId(eventIdMax);
  if (!allocated)
    return std::nullopt;
  return *allocated;
}

bool isComplementaryBoundaryOp(hivm::SyncEventSlotAttr slot, SyncOp *syncOp) {
  if (slotMacroInternal(slot))
    return false;
  return slotMacroWaits(slot) ? llvm::isa<SetFlagOp>(syncOp)
                              : llvm::isa<WaitFlagOp>(syncOp);
}

Value materializeSyncRelatedArg(
    IRRewriter &rewriter, hivm::CustomMacroOp macroOp, size_t slotIdx,
    hivm::SyncEventSlotAttr slot, llvm::ArrayRef<Value> collectedArgs,
    Value defaultValue,
    const llvm::DenseMap<hivm::CustomMacroOp, llvm::SmallVector<int64_t>>
        &resolvedIds) {
  if (slotIdx < collectedArgs.size() && collectedArgs[slotIdx])
    return collectedArgs[slotIdx];

  if (auto it = resolvedIds.find(macroOp); it != resolvedIds.end()) {
    if (slotIdx < it->second.size() && it->second[slotIdx] >= 0) {
      return rewriter.create<arith::ConstantOp>(
          macroOp->getLoc(),
          rewriter.getIntegerAttr(rewriter.getI64Type(), it->second[slotIdx]));
    }
  }

  if (auto pinned = getSlotPinnedEventId(slot)) {
    return rewriter.create<arith::ConstantOp>(
        macroOp->getLoc(),
        rewriter.getIntegerAttr(rewriter.getI64Type(), *pinned));
  }

  return defaultValue;
}

void injectBoundarySyncForSlot(
    hivm::SyncEventSlotAttr slot, OperationBase *rwOp,
    llvm::ArrayRef<OperationBase *> orderedOps, const SyncMap &syncMapBefore,
    const SyncMap &syncMapAfter, llvm::SmallVector<int64_t> eventIds,
    SyncMap &syncMapBeforeOut, SyncMap &syncMapAfterOut) {
  if (slotMacroInternal(slot) || !slotHasExplicitPipes(slot))
    return;

  auto setPipe = getSlotSetPipe(slot);
  auto waitPipe = getSlotWaitPipe(slot);
  if (slotMacroWaits(slot)) {
    if (hasExternalSetFlag(orderedOps, rwOp, syncMapBefore, syncMapAfter,
                           setPipe, waitPipe))
      return;
    auto setOp = std::make_unique<SetFlagOp>(rwOp->op, rwOp->parentOp, eventIds,
                                             setPipe, waitPipe);
    syncMapBeforeOut[rwOp].push_front(std::move(setOp));
    return;
  }

  assert(slotMacroSets(slot));
  auto waitOp = std::make_unique<WaitFlagOp>(rwOp->op, rwOp->parentOp, eventIds,
                                             setPipe, waitPipe);
  syncMapAfterOut[rwOp].push_back(std::move(waitOp));
}

} // namespace

void applyCustomMacroPinnedEventId(ConflictPair &conflictPair,
                                   RWOperation *rwOp1, RWOperation *rwOp2,
                                   hivm::PIPE setPipe, hivm::PIPE waitPipe) {
  for (auto *rwOp : {rwOp1, rwOp2}) {
    auto customMacroOp =
        llvm::dyn_cast_if_present<hivm::CustomMacroOp>(rwOp->op);
    if (!customMacroOp)
      continue;
    if (auto pinned = customMacroOp.getUserPinnedEventId(setPipe, waitPipe)) {
      conflictPair.pinnedEventId = pinned;
      break;
    }
  }
}

void CustomMacroSyncState::markConflict(StringRef message) {
  conflict = true;
  conflictMessage_ = message.str();
}

void CustomMacroSyncState::collectReservedEventIds(
    func::FuncOp funcOp, const SyncSolverOptions &options) {
  if (!funcOp)
    return;

  reservedEventIds.clear();
  llvm::DenseMap<std::tuple<hivm::PIPE, hivm::PIPE, int64_t>, Operation *>
      seenReservations;
  funcOp.walk([&](hivm::CustomMacroOp customMacroOp) {
    for (auto slot : customMacroOp.getUserSyncEventSlots()) {
      if (auto pinned = getSlotPinnedEventId(slot)) {
        auto [setPipe, waitPipe] = resolveSlotPipePair(slot, customMacroOp);
        auto reservationKey = std::make_tuple(setPipe, waitPipe, *pinned);
        if (seenReservations.contains(reservationKey)) {
          markConflict("duplicate user-specified sync event id on the same "
                       "pipe pair across custom_macro ops");
          return;
        }
        const int64_t eventIdMax =
            getHWAvailableEventIdNum(options.syncMode, setPipe, waitPipe);
        if (*pinned >= eventIdMax) {
          markConflict("user-specified sync event id cannot be satisfied by "
                       "GraphSyncSolver for this pipe pair");
          return;
        }
        seenReservations[reservationKey] = customMacroOp;
        reservedEventIds[{setPipe, waitPipe}].push_back(*pinned);
      }
    }
  });
}

void CustomMacroSyncState::applyReservedEventIds(
    llvm::function_ref<std::unique_ptr<EventIdSolver> &(hivm::PIPE, hivm::PIPE)>
        getEventIdSolver) {
  for (auto &[pipePair, eventIds] : reservedEventIds) {
    auto [pipeSrc, pipeDst] = pipePair;
    for (int64_t eventId : eventIds)
      getEventIdSolver(pipeSrc, pipeDst)->reserveEventId(eventId);
  }
}

void CustomMacroSyncState::validatePinnedAssignments(
    llvm::ArrayRef<std::unique_ptr<ConflictPair>> conflictPairs) {
  if (conflict)
    return;

  for (auto &conflictPair : conflictPairs) {
    if (!conflictPair->pinnedEventId || conflictPair->isUseless ||
        conflictPair->replacedWithUnitFlag || !conflictPair->eventIdNode) {
      continue;
    }
    auto assignedIds = conflictPair->eventIdNode->getEventIds();
    if (assignedIds.size() != 1 ||
        assignedIds.front() != *conflictPair->pinnedEventId) {
      markConflict("GraphSyncSolver assigned a different event id than the one "
                   "pinned by custom_macro");
      return;
    }
  }
}

void CustomMacroSyncState::injectBoundarySync(
    SyncMap &syncMapBefore, SyncMap &syncMapAfter, func::FuncOp funcOp,
    OperationBase *funcIr, const SyncSolverOptions &options,
    llvm::function_ref<std::unique_ptr<EventIdSolver> &(hivm::PIPE, hivm::PIPE)>
        getEventIdSolver) {
  if (!funcIr || !funcOp)
    return;

  slotEventIds.clear();
  auto orderedFuncIr = collectOrderedFuncIr(funcIr);

  funcOp.walk([&](hivm::CustomMacroOp macroOp) {
    auto *rwOp = orderedFuncIr.mlirToRw.lookup(macroOp.getOperation());
    if (!rwOp)
      return;

    llvm::SmallVector<int64_t> resolvedIds;
    for (auto slot : macroOp.getSyncEventSlots()) {
      if (slot.getMacroSync() != hivm::SyncEventSlotMacroSync::internal) {
        auto eventIds = resolveSyncEventSlotEventIds(
            slot, rwOp, orderedFuncIr.ops, syncMapBefore, syncMapAfter);
        resolvedIds.push_back(eventIds.empty() ? -1 : eventIds.front());
        injectBoundarySyncForSlot(slot, rwOp, orderedFuncIr.ops, syncMapBefore,
                                  syncMapAfter, eventIds, syncMapBefore,
                                  syncMapAfter);
        continue;
      }

      auto internalId =
          resolveInternalSlotEventId(slot, macroOp, options, getEventIdSolver);
      if (!internalId) {
        markConflict("GraphSyncSolver could not allocate an event id for "
                     "custom_macro internal sync_event_slot");
        return;
      }
      resolvedIds.push_back(*internalId);
    }
    if (!resolvedIds.empty())
      slotEventIds[macroOp] = std::move(resolvedIds);
  });
}

void CustomMacroSyncCodegenState::recordBoundaryArgIfNeeded(
    OperationBase *opBase, SyncOp *syncOp, IRRewriter &rewriter,
    llvm::function_ref<Value(SetWaitOp *, Location)> getEventIdValue) {
  auto *rwOp = llvm::dyn_cast<RWOperation>(opBase);
  if (!rwOp)
    return;

  auto customMacroOp = llvm::dyn_cast<hivm::CustomMacroOp>(rwOp->op);
  if (!customMacroOp)
    return;

  auto slots = customMacroOp.getSyncEventSlots();
  if (slots.empty())
    return;

  auto *setWaitOp = llvm::dyn_cast<SetWaitOp>(syncOp);
  if (!setWaitOp)
    return;

  auto &slotValues = collectedArgs[customMacroOp];
  slotValues.resize(slots.size());
  for (auto [idx, slot] : llvm::enumerate(slots)) {
    if (!slotMatchesPipePair(slot, setWaitOp->pipeSrc, setWaitOp->pipeDst))
      continue;
    if (!isComplementaryBoundaryOp(slot, syncOp))
      continue;
    slotValues[idx] = getEventIdValue(setWaitOp, customMacroOp->getLoc());
  }
}

void CustomMacroSyncCodegenState::populateSyncRelatedArgs(
    func::FuncOp funcOp, IRRewriter &rewriter) {
  funcOp.walk([&](hivm::CustomMacroOp customMacroOp) {
    const int numSlots = customMacroOp.getNumSyncRelatedArgs();
    auto slotValues = collectedArgs.lookup(customMacroOp);

    rewriter.setInsertionPoint(customMacroOp);
    auto defaultValue = rewriter.create<arith::ConstantOp>(
        customMacroOp->getLoc(),
        rewriter.getIntegerAttr(rewriter.getI64Type(), -1));
    llvm::SmallVector<Value> newArgs(numSlots, defaultValue);

    for (auto [idx, slot] :
         llvm::enumerate(customMacroOp.getSyncEventSlots())) {
      if (idx >= static_cast<size_t>(numSlots))
        break;
      newArgs[idx] =
          materializeSyncRelatedArg(rewriter, customMacroOp, idx, slot,
                                    slotValues, defaultValue, resolvedIds);
    }

    customMacroOp.getSyncRelatedArgsMutable().assign(newArgs);
  });
}

} // namespace mlir::hivm::syncsolver
