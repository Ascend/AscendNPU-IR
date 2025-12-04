//===------------ SyncEventIdAllocation.cpp ----Event id allocate ---------===//
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

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncEventIdAllocation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"

#define DEBUG_TYPE "hivm-inject-sync"

using namespace mlir;
using namespace mlir::hivm;

void SyncEventIdAllocation::Allocate(uint32_t runNum) {
  //  allocate event id
  for (auto &element : syncIR) {
    AllocateEventId(element.get());
  }
  // widen event id
  for (auto &e : syncIR) {
    WidenEventId(e->pipeAfter);
  }
  // Ignore partial special synchronization without event_id.
  IgnoreBackHeadAndTailSync();
  // Reallocated unallocated pipes with event IDs for synchronization.
  if (!reallocatedPipePair.empty()) {
    ReallocatedEventId();
    // again widen event id.
    for (auto &e : syncIR) {
      WidenEventId(e->pipeAfter);
    }
  }
  // Change no event id sync to pipe all if possible.
  auto status = ChangeNoEventIdSyncToPipeAll();
  if (status.failed() && runNum < kMaxWidenTryNum) {
    if (tryWidenOnFirstFound()) {
      // clear and reallocate event ids.
      reallocatedPipePair.clear();
      eventCyclePool.clear();
      clearAllocatedEventId();
      Allocate(runNum + 1);
    }
  }
}

bool SyncEventIdAllocation::tryWidenOnFirstFound() {
  for (auto pipePair : reallocatedPipePair) {
    for (auto &e : syncIR) {
      for (auto &sync : e->pipeAfter) {
        if (sync->isSyncSetType() && !sync->uselessSync &&
            (ScopePair(sync) == pipePair)) {
          if (TryWidenByOtherSync(sync)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void SyncEventIdAllocation::reserveBlockAllEventIds() {
  bool blockSyncAllExists = false;
  for (auto &element : syncIR) {
    for (auto &syncOp : element->pipeBefore) {
      if (syncOp->GetType() == SyncOperation::TYPE::SYNC_BLOCK_ALL) {
        blockSyncAllExists = true;
        break;
      }
    }
    for (auto &syncOp : element->pipeAfter) {
      if (syncOp->GetType() == SyncOperation::TYPE::SYNC_BLOCK_ALL) {
        blockSyncAllExists = true;
        break;
      }
    }
    if (blockSyncAllExists) {
      break;
    }
  }
  if (blockSyncAllExists) {
    reservedBlockSyncEventIdNum = 2;
  }
}

void SyncEventIdAllocation::SetBlockSyncAllEventID(SyncOperation *sync) {
  if (sync->syncCoreType == TCoreType::CUBE) {
    sync->eventIds.push_back(kBlockSyncAllCubeEventId);
  } else if (sync->syncCoreType == TCoreType::VECTOR) {
    sync->eventIds.push_back(kBlockSyncAllVectorEventId);
  } else {
    llvm_unreachable("auto-inserted sync all operation must be all cube "
                     "or all vector");
  }
}

void SyncEventIdAllocation::AllocateEventId(InstanceElement *e) {
  // When allocating an event ID,
  // forward and reverse synchronization are allocated together.
  for (auto &sync : e->pipeBefore) {
    if (sync->uselessSync) {
      continue;
    }
    if (!sync->eventIds.empty()) {
      // sync must has not allocated event id.
      continue;
    }
    if (sync->isBarrierType()) {
      // PIPE_BARRIER no need event id.
      continue;
    }
    if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_ALL) {
      // Block sync all_cube and all_vector event id.
      SetBlockSyncAllEventID(sync);
    } else if (sync->GetType() == SyncOperation::TYPE::SET_EVENT ||
               sync->GetType() == SyncOperation::TYPE::WAIT_EVENT ||
               sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET ||
               sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
      // Set or Wait sync event id.
      SetEventId(sync);
    } else {
      llvm_unreachable("InjectSync does not support sync types!");
    }
  }
}

size_t SyncEventIdAllocation::GetCompilerAvailableEventIdNum(
    const SyncOperation *sync) {
  if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET ||
      sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    // inter-block synchronization.
    return kBlockSyncSetWaitEventIdNum - reservedBlockSyncEventIdNum;
  }
  auto it = reservedEventIdNum.find({sync->GetSrcPipe(), sync->GetDstPipe()});
  if (it != reservedEventIdNum.end()) {
    // Sync related to pipe_s, considering conflicts with the
    // instruction library, currently auto sync only 6 can be used.
    return kTotalEventIdNum - it->second;
  }
  return kTotalEventIdNum;
}

void SyncEventIdAllocation::SetEventId(SyncOperation *sync) {
  size_t eventIdNum = GetCompilerAvailableEventIdNum(sync);
  // if it is true, it means the event id can be allocated for other ops and no
  // lifetime conflict but may bring redundant pipe conflict.
  SmallVector<bool> eventIdLifetimeAvailableStatus =
      GetEventPool(sync, eventIdNum);
  // if it is true, it means the event id is idle and no one has used it
  SmallVector<bool> eventIdIdleStatus = GetEventIdIdleStatus(sync, eventIdNum);
  assert(eventIdLifetimeAvailableStatus.size() == eventIdNum);
  // Continuous allocation based on the required number of IDs.
  size_t idSize = static_cast<size_t>(sync->eventIdNum);
  SmallVector<int> canAllocaEventId = GetAvailableEventId(
      sync, eventIdLifetimeAvailableStatus, eventIdIdleStatus, eventIdNum);
  if (canAllocaEventId.empty()) {
    // There is no event_id available for use.
    return;
  } else if (canAllocaEventId.size() >= idSize) {
    assert(canAllocaEventId.size() == idSize &&
           "GetAvailableEventId guarantee only return <= idSize results");
    // Obtained all the required event_id count.
    for (auto &id : canAllocaEventId) {
      SetEventPool(sync, id);
    }
  } else if (reallocatedPipePair.count(ScopePair(sync)) &&
             (canAllocaEventId.size() < idSize)) {
    // Reallocate, if multiple event_ids cannot be assigned but there are
    // event_ids that can be assigned, then only one event_id should be
    // allocated.
    assert(canAllocaEventId.size() > 0);
    SetEventPool(sync, canAllocaEventId[0]);
    sync->eventIdNum = 1;
  }
}

SmallVector<int> SyncEventIdAllocation::UpdateBlockAvailableEventId(
    SyncOperation *sync, SmallVector<bool> eventIdLifetimeAvailableStatus,
    size_t eventIdNum) {
  SmallVector<int> canAllocaEventId;
  size_t idSize = static_cast<size_t>(sync->eventIdNum);
  for (unsigned id = 0; id < eventIdNum; id++) {
    if (canAllocaEventId.size() == idSize) {
      break;
    }
    if (!canAllocaEventId.empty() && !eventIdLifetimeAvailableStatus[id]) {
      canAllocaEventId.clear();
      continue;
    }
    if (eventIdLifetimeAvailableStatus[id]) {
      canAllocaEventId.push_back(id);
    }
  }
  return canAllocaEventId;
}

SmallVector<int> SyncEventIdAllocation::GetAvailableEventId(
    SyncOperation *sync, SmallVector<bool> eventIdLifetimeAvailableStatus,
    SmallVector<bool> eventIdIdleStatus, size_t eventIdNum) {
  SmallVector<int> canAllocaEventId;
  size_t idSize = static_cast<size_t>(sync->eventIdNum);
  if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET ||
      sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    return UpdateBlockAvailableEventId(sync, eventIdLifetimeAvailableStatus,
                                       eventIdNum);
  }
  for (unsigned id = 0; id < eventIdNum; id++) {
    if (canAllocaEventId.size() == idSize) {
      break;
    }
    // Prioritize using no use event ids.
    if (eventIdLifetimeAvailableStatus[id] && eventIdIdleStatus[id]) {
      eventIdLifetimeAvailableStatus[id] = false;
      canAllocaEventId.push_back(id);
    }
  }

  for (unsigned id = 0; id < eventIdNum; id++) {
    if (canAllocaEventId.size() == idSize) {
      break;
    }
    // Next, use the assigned ids.
    if (eventIdLifetimeAvailableStatus[id]) {
      eventIdLifetimeAvailableStatus[id] = false;
      canAllocaEventId.push_back(id);
    }
  }
  return canAllocaEventId;
}

SmallVector<bool>
SyncEventIdAllocation::GetEventIdIdleStatus(SyncOperation *sync,
                                            size_t eventIdNum) {
  SmallVector<bool> eventIdIdleStatus;
  int scopePair = ScopePair(sync);
  EventCyclePool &seqPool = eventCyclePool[scopePair];
  for (size_t i = 0; i < eventIdNum; i++) {
    auto &syncLifeCycle = seqPool.slot[i];
    if (syncLifeCycle.empty()) {
      eventIdIdleStatus.push_back(true);
    } else {
      eventIdIdleStatus.push_back(false);
    }
  }
  return eventIdIdleStatus;
}

SmallVector<bool> SyncEventIdAllocation::GetEventPool(const SyncOperation *sync,
                                                      size_t eventIdNum) {
  SmallVector<bool> eventIdPool(eventIdNum, true);
  assert(sync->GetSyncIndex() < syncOperations.size());
  auto &syncPair = syncOperations[sync->GetSyncIndex()];
  auto *setFlag = syncPair[0].get();
  auto *waitFlag = syncPair[1].get();

  if (setFlag->GetForEndIndex().has_value()) {
    if (reallocatedPipePair.count(ScopePair(sync))) {
      auto *ptr = dyn_cast<LoopInstanceElement>(
          syncIR[setFlag->GetForEndIndex().value()].get());
      assert(ptr != nullptr);
      FindUseEventID(ptr->beginId, ptr->endId, setFlag, eventIdPool);
    } else {
      FindUseEventID(0, syncIR.size() - 1, setFlag, eventIdPool);
    }
  } else {
    FindUseEventID(setFlag->GetSyncIRIndex(), waitFlag->GetSyncIRIndex(),
                   setFlag, eventIdPool);
  }
  return eventIdPool;
}

int SyncEventIdAllocation::ScopePair(const SyncOperation *s) {
  if (s->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET ||
      s->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    // For inter block synchronization, event id is global shared and then the
    // scope pair is always same.
    return 0;
  }
  // For intra block synchronization, each pipe pair has fixed number event ids
  // and then scope pair make a difference between each pipe pair.
  auto srcT = static_cast<unsigned int>(s->GetActualSrcPipe()); // [8:15]
  auto dstT = static_cast<unsigned int>(s->GetActualDstPipe()); // [0:7]
  return static_cast<int>((dstT << 8) | srcT);
}

void SyncEventIdAllocation::FindUseEventID(unsigned int begin, unsigned int end,
                                           const SyncOperation *s,
                                           SmallVector<bool> &eventId) {
  auto eventIdSize = GetCompilerAvailableEventIdNum(s);
  assert(eventId.size() == eventIdSize);
  assert(begin < end);
  int scopePair = ScopePair(s);
  eventCyclePool.try_emplace(scopePair, EventCyclePool(eventIdSize));
  EventCyclePool &seqPool = eventCyclePool[scopePair];
  for (size_t i = 0; i < eventIdSize; i++) {
    auto &syncLifeCycle = seqPool.slot[i];
    if (syncLifeCycle.empty()) {
      continue;
    } else if (CheckSyncLifeCycleConflict(syncLifeCycle, begin, end, eventId,
                                          i)) {
      continue;
    }
  }
}

bool SyncEventIdAllocation::CheckSyncLifeCycleConflict(
    SmallVector<unsigned int> &syncLifeCycle, unsigned int begin,
    unsigned int end, SmallVector<bool> &eventId, unsigned i) const {
  assert((syncLifeCycle.size() & 0x1) == 0 && "sync_life_cycle error.");
  if (syncLifeCycle[0] <= begin) {
    return true;
  }
  UpdateEventId(syncLifeCycle, begin, end, eventId, i);
  return false;
}

void SyncEventIdAllocation::UpdateEventId(
    SmallVector<unsigned int> &syncLifeCycle, const unsigned int begin,
    const unsigned int end, SmallVector<bool> &eventId,
    const unsigned index) const {
  for (size_t j = 0; j < syncLifeCycle.size(); j++) {
    if (syncLifeCycle[j] <= begin) {
      if (syncLifeCycle[j - 1] >= end && (j & 0x1) == 0) {
        break;
      } else {
        eventId[index] = false;
      }
    } else if (j == syncLifeCycle.size() - 1) {
      assert((j & 0x1) == 1);
      if (syncLifeCycle[j] >= end) {
        break;
      } else {
        eventId[index] = false;
      }
    }
  }
}

void SyncEventIdAllocation::SetEventPool(const SyncOperation *sync,
                                         unsigned eventId) {
  assert(sync->GetSyncIndex() < syncOperations.size());
  auto &syncPair = syncOperations[sync->GetSyncIndex()];
  auto &setFlag = syncPair[0];
  auto &waitFlag = syncPair[1];
  if (setFlag->GetForEndIndex().has_value()) {
    if (reallocatedPipePair.count(ScopePair(sync))) {
      auto *ptr = dyn_cast<LoopInstanceElement>(
          syncIR[setFlag->GetForEndIndex().value()].get());
      assert(ptr != nullptr);
      SetUseEventID(ptr->beginId, ptr->endId, setFlag.get(), eventId);
    } else {
      SetUseEventID(0, syncIR.size(), setFlag.get(), eventId);
    }
  } else {
    SetUseEventID(setFlag->GetSyncIRIndex(), waitFlag->GetSyncIRIndex(),
                  setFlag.get(), eventId);
  }
  setFlag->eventIds.push_back(eventId);
  waitFlag->eventIds.push_back(eventId);
  if (setFlag->GetForEndIndex().has_value()) {
    // Process the following scenarios:
    // for():
    //   for():
    //    waitFlag
    //    compound1
    //    compound2
    //    setFlag
    // change to :
    // setFlag
    // for():
    //   for():
    //    waitFlag
    //    compound1
    //    compound2
    //    setFlag
    // waitFlag
    UpdateBackwardMatchSync(setFlag.get(), waitFlag.get(), eventId);
  }
}

void SyncEventIdAllocation::UpdateBackwardMatchSync(
    const SyncOperation *setFlag, const SyncOperation *waitFlag,
    unsigned eventId) {
  auto syncFront = std::make_unique<SyncOperation>(
      setFlag->GetType(), setFlag->GetSrcPipe(), setFlag->GetDstPipe(),
      static_cast<unsigned>(syncOperations.size()), setFlag->GetSyncIRIndex(),
      setFlag->GetForEndIndex());
  auto syncEnd = syncFront->GetMatchSync(waitFlag->GetSyncIRIndex());
  syncFront->syncCoreType = setFlag->syncCoreType;
  syncEnd->syncCoreType = waitFlag->syncCoreType;
  syncFront->eventIds.push_back(eventId);
  syncEnd->eventIds.push_back(eventId);
  assert(!syncIR.empty());
  if (reallocatedPipePair.count(ScopePair(setFlag))) {
    auto *ptr = dyn_cast<LoopInstanceElement>(
        syncIR[setFlag->GetForEndIndex().value()].get());
    assert(ptr != nullptr);
    syncFront->SetSyncIRIndex(ptr->beginId);
    syncEnd->SetSyncIRIndex(ptr->endId);
    syncFront->reallocatedLoopHeadTailSync = true;
    syncEnd->reallocatedLoopHeadTailSync = true;
    syncIR[ptr->beginId]->pipeBefore.push_back(syncFront.get());
    syncIR[ptr->endId]->pipeAfter.push_back(syncEnd.get());
  } else {
    syncFront->SetSyncIRIndex(0);
    syncEnd->SetSyncIRIndex(syncIR.size() - 1);
    syncIR[0]->pipeBefore.push_back(syncFront.get());
    syncIR[syncIR.size() - 1]->pipeAfter.push_back(syncEnd.get());
  }
  insertedBackwardSync.insert(syncFront.get());
  insertedBackwardSync.insert(syncEnd.get());
  SmallVector<std::unique_ptr<SyncOperation>> newSync;
  newSync.emplace_back(std::move(syncFront));
  newSync.emplace_back(std::move(syncEnd));
  syncOperations.emplace_back(std::move(newSync));
}

void SyncEventIdAllocation::SetUseEventID(unsigned int begin, unsigned int end,
                                          const SyncOperation *setFlag,
                                          unsigned int eventId) {
  assert(begin < end);
  int scopePair = ScopePair(setFlag);
  eventCyclePool.try_emplace(
      scopePair, EventCyclePool(GetCompilerAvailableEventIdNum(setFlag)));
  EventCyclePool &seqPool = eventCyclePool[scopePair];
  auto &syncLifeCycle = seqPool.slot[eventId];
  bool isInsert = false;
  if (syncLifeCycle.empty()) {
    syncLifeCycle.push_back(end);
    syncLifeCycle.push_back(begin);
    isInsert = true;
  } else {
    assert((syncLifeCycle.size() & 0x1) == 0 && "syncLifeCycle error.");
    /** When we insert a new declaration cycle,
     * we insert a pair of open intervals directly at
     * both ends of syncLifeCycle.
     * Insert [20, 16] to [15, 10], [8, 5].
     * =>[20, 16], [15, 10], [8, 5].;
     */
    if (syncLifeCycle[0] <= begin) {
      syncLifeCycle.insert(syncLifeCycle.begin(), begin);
      syncLifeCycle.insert(syncLifeCycle.begin(), end);
      return;
    } else if (syncLifeCycle.back() >= end) {
      syncLifeCycle.insert(syncLifeCycle.end(), end);
      syncLifeCycle.insert(syncLifeCycle.end(), begin);
      return;
    } else if (ExtendLifecycle(syncLifeCycle, begin, end)) {
      return;
    }
  }
  if (!isInsert)
    llvm_unreachable("Can't insert this sync cycle!");
}

bool SyncEventIdAllocation::ExtendLifecycle(
    SmallVector<unsigned int> &syncLifeCycle, unsigned int beginNew,
    unsigned int endNew) const {
  for (size_t j = 0; j < syncLifeCycle.size() / 2U; j++) {
    assert(j * 2U + 1 < syncLifeCycle.size());
    // When inserting an existing lifecycle, choose to extend the lifecycle.
    // The original syncLifeCycle distribution is as follows:
    // [endOld0, beginOld0], [endOld1, beginOld1], ...
    uint &endOld = syncLifeCycle[j * 2U];
    uint &beginOld = syncLifeCycle[j * 2U + 1];

    // When we insert a new declaration cycle,
    // When max_range is the default value true:
    // Insert [18, 10] to [20, 16], [8, 5].
    // => [20, 10], [8, 5].
    bool widenLifeCycleBegin = endOld >= endNew && endNew > beginOld;

    // When we insert a new declaration cycle,
    // If max_range is the default value true:
    // Insert [23, 18] to [20, 16], [8, 5].
    // => [23, 16], [8, 5].
    bool widenLifeCycleEnd = endOld > beginNew && beginNew >= beginOld;

    // When we insert a new declaration cycle,
    // we insert a pair of open intervals directly
    // in the middle of syncLifeCycle
    // (j + 1) * 2U idx means next_endOld
    // if j == ((syncLifeCycle.size() / 2U) - 1),
    // next_endOld does not exist.
    // Insert [15, 10] to [20, 16], [8, 5].
    // => [20, 16], [15, 10], [8, 5];
    //
    bool insertMiddleLifecycle = j < ((syncLifeCycle.size() / 2U) - 1) &&
                                 beginOld >= endNew &&
                                 beginNew >= syncLifeCycle[(j + 1) * 2U];
    if (widenLifeCycleBegin) {
      beginOld = std::min(beginOld, beginNew);
#ifndef NDEBUG
      if (j < ((syncLifeCycle.size() / 2U) - 1)) {
        uint &next_endOld = syncLifeCycle[(j + 1) * 2U];
        assert(beginOld >= next_endOld && "Set event id failed;");
      }
#endif
      return true;
    } else if (widenLifeCycleEnd) {
      endOld = std::max(endOld, endNew);
#ifndef NDEBUG
      if (j > 0) {
        uint &last_beginOld = syncLifeCycle[(j * 2U) - 1];
        assert(endOld <= last_beginOld && "Set event id failed");
      }
#endif
      return true;
    } else if (insertMiddleLifecycle) {
      syncLifeCycle.insert(syncLifeCycle.begin() + (j + 1) * 2U, beginNew);
      syncLifeCycle.insert(syncLifeCycle.begin() + (j + 1) * 2U, endNew);
      return true;
    }
  }
  return false;
}

void SyncEventIdAllocation::WidenEventId(SyncOps syncVector) {
  for (auto &sync : syncVector) {
    if (sync->isSyncSetType() && sync->eventIds.empty() && !sync->uselessSync) {
      // Replace sync of the same type and perform widen processing.
      bool canWiden = TryWidenByOtherSync(sync);
      if (!canWiden) {
        int scopePair = ScopePair(sync);
        reallocatedPipePair.insert(scopePair);
      }
    }
  }
}

void SyncEventIdAllocation::clearAllocatedEventId() {
  for (auto &e : syncIR) {
    SyncOps newPipeBefore;
    for (auto *sync : e->pipeBefore) {
      if (!insertedBackwardSync.contains(sync)) {
        newPipeBefore.push_back(sync);
      }
    }
    e->pipeBefore = newPipeBefore;

    SyncOps newPipeAfter;
    for (auto *sync : e->pipeAfter) {
      if (!insertedBackwardSync.contains(sync)) {
        newPipeAfter.push_back(sync);
      }
    }
    e->pipeAfter = newPipeAfter;
  }

  for (auto &e : syncIR) {
    for (auto &sync : e->pipeBefore) {
      if (!sync->isBarrierType()) {
        ClearEventId(sync);
      }
    }
    for (auto &sync : e->pipeAfter) {
      if (!sync->isBarrierType()) {
        ClearEventId(sync);
      }
    }
  }
}

void SyncEventIdAllocation::ReallocatedEventId() {
  // reallocate event id with new policy: backward sync will insert head-tail
  // match sync just before and after the nearest parent loop not the whole ir.
  for (auto pipePair : reallocatedPipePair) {
    // Clear pipePair event id cycle pool.
    eventCyclePool.erase(pipePair);
  }

  ClearReallocatedBackwardMatchSync();
  for (auto &e : syncIR) {
    for (auto &sync : e->pipeBefore) {
      if (!sync->isBarrierType() &&
          reallocatedPipePair.count(ScopePair(sync))) {
        ClearEventId(sync);
        SetEventId(sync);
      }
    }
  }
}

void SyncEventIdAllocation::ClearEventId(const SyncOperation *sync) {
  // Clearly identify all the eventIDs of the sync for pipes.
  if (sync->eventIds.empty()) {
    return;
  }
  // Clearly identify all the eventIDs of the sync for pipes.
  auto &syncPair = syncOperations[sync->GetSyncIndex()];
  assert(syncPair.size() > 1);
  SyncOperation *setSync = syncPair[0].get();
  SyncOperation *waitSync = syncPair[1].get();
  setSync->eventIds.clear();
  waitSync->eventIds.clear();
}

void SyncEventIdAllocation::ClearReallocatedBackwardMatchSync() {
  SyncOps newPipeBefore;
  for (auto &sync : syncIR[0]->pipeBefore) {
    if (!(sync->isSyncSetType() &&
          reallocatedPipePair.count(ScopePair(sync)))) {
      newPipeBefore.push_back(sync);
    }
  }
  // update sync pipeBefore.
  syncIR[0]->pipeBefore = newPipeBefore;

  SyncOps newPipeAfter;
  for (auto &sync : syncIR[syncIR.size() - 1]->pipeAfter) {
    if (!(sync->isSyncWaitType() &&
          reallocatedPipePair.count(ScopePair(sync)))) {
      newPipeAfter.push_back(sync);
    }
  }
  // update sync pipeAfter.
  syncIR[syncIR.size() - 1]->pipeAfter = newPipeAfter;
}

llvm::LogicalResult SyncEventIdAllocation::ChangeNoEventIdSyncToPipeAll() {
  for (auto &e : syncIR) {
    for (auto &sync : e->pipeAfter) {
      if (sync->GetType() == SyncOperation::TYPE::WAIT_EVENT &&
          sync->reallocatedLoopHeadTailSync) {
        MoveOutBackwardMatchSync(sync);
      }
      if (sync->GetType() == SyncOperation::TYPE::SET_EVENT &&
          sync->eventIds.empty() && !sync->uselessSync) {
        // Convert to pipe_all synchronization.
        assert(sync->GetSyncIndex() < syncOperations.size());
        auto &syncPair = syncOperations[sync->GetSyncIndex()];
        syncPair[0]->uselessSync = true;
        syncPair[1]->SetPipeAll();
      }
      if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET &&
          sync->eventIds.empty() && !sync->uselessSync) {
        return failure();
      }
    }
  }
  return success();
}

void SyncEventIdAllocation::MoveOutBackwardMatchSync(
    const SyncOperation *reallocatedSync) {
  auto &syncPair = syncOperations[reallocatedSync->GetSyncIndex()];
  assert(syncPair.size() > 1);
  assert(!reallocatedSync->eventIds.empty());
  SyncOperation *setSync = syncPair[0].get();
  SyncOperation *waitSync = syncPair[1].get();
  bool isConflictEventId = false;

  for (unsigned int i = 0; i <= syncIR.size() - 1; i++) {
    if (isConflictEventId) {
      break;
    }
    if ((i > setSync->GetSyncIRIndex()) && (i < waitSync->GetSyncIRIndex())) {
      continue;
    }
    for (auto &sync : syncIR[i]->pipeBefore) {
      if (!sync->uselessSync &&
          reallocatedSync->GetSyncIndex() != sync->GetSyncIndex() &&
          sync->GetActualSrcPipe() == reallocatedSync->GetActualSrcPipe() &&
          sync->GetActualDstPipe() == reallocatedSync->GetActualDstPipe() &&
          sync->eventIds == reallocatedSync->eventIds) {
        isConflictEventId = true;
        break;
      }
    }
  }

  if (!isConflictEventId) {
    setSync->uselessSync = true;
    waitSync->uselessSync = true;
    assert(setSync->eventIds.size() == 1);
    assert(setSync->eventIds[0] == waitSync->eventIds[0]);
    UpdateBackwardMatchSync(setSync, waitSync, setSync->eventIds[0]);
  }
}

void SyncEventIdAllocation::IgnoreBackHeadAndTailSync() {
  SyncOps newPipeBefore;
  for (auto &sync : syncIR[0]->pipeBefore) {
    bool isPipeMTE1ToPipeMSync = sync->GetSrcPipe() == hivm::PIPE::PIPE_M &&
                                 sync->GetDstPipe() == hivm::PIPE::PIPE_MTE1;
    if (!isPipeMTE1ToPipeMSync) {
      continue;
    }
    assert(sync->GetSyncIndex() < syncOperations.size());
    auto &syncPair = syncOperations[sync->GetSyncIndex()];
    assert(syncPair.size() > 1);
    if (sync->eventIds.empty()) {
      SyncOperation *setSync = syncPair[0].get();
      SyncOperation *waitSync = syncPair[1].get();
      setSync->uselessSync = true;
      waitSync->uselessSync = true;
    }
  }
}

bool SyncEventIdAllocation::TryWidenByOtherSync(const SyncOperation *sync) {
  assert(!sync->isBarrierType());
  assert(sync->GetSyncIndex() < syncOperations.size());
  auto &syncPair = syncOperations[sync->GetSyncIndex()];
  assert(syncPair.size() > 1);
  SyncOperation *setSync = syncPair[0].get();
  SyncOperation *waitSync = syncPair[1].get();

  SyncOperation *widenSync = FindWidenSync(setSync, waitSync);
  if (widenSync == nullptr) {
    return false;
  }
  assert(!widenSync->GetEventIDs().empty());
  setSync->uselessSync = true;
  waitSync->uselessSync = true;
  assert(!widenSync->isBarrierType());
  assert(widenSync->GetSyncIndex() < syncOperations.size());
  auto &widenSyncPair = syncOperations[widenSync->GetSyncIndex()];
  assert(widenSyncPair.size() > 1);
  SyncOperation *widenSet = widenSyncPair[0].get();

  assert(setSync->GetSyncIRIndex() >= widenSet->GetSyncIRIndex());
  auto *widenSetSyncIR = syncIR[widenSet->GetSyncIRIndex()].get();

  SyncOps newPipeAfter;
  if (setSync->GetSyncIRIndex() != widenSet->GetSyncIRIndex()) {
    bool removeSync = false;
    for (auto &s : widenSetSyncIR->pipeAfter) {
      if (s == widenSet) {
        syncIR[setSync->GetSyncIRIndex()]->pipeAfter.push_back(widenSet);
        widenSet->SetSyncIRIndex(setSync->GetSyncIRIndex());
        widenSet->reuseCntForWiden++;
        removeSync = true;
      } else {
        newPipeAfter.push_back(s);
      }
    }
    widenSetSyncIR->pipeAfter = newPipeAfter;
    if (!removeSync)
      llvm_unreachable("in widen fun, remove sync failed");
  }
  return true;
}

SyncOperation *
SyncEventIdAllocation::FindWidenSync(const SyncOperation *setSync,
                                     const SyncOperation *waitSync) {
  assert(setSync->GetSyncIRIndex() < syncIR.size());
  SyncOperation *widenSetSync = nullptr;
  int endIndex;
  if (setSync->GetForEndIndex().has_value()) {
    auto *forElement = syncIR[setSync->GetForEndIndex().value()].get();
    auto *forCompound = dyn_cast<LoopInstanceElement>(forElement);
    assert(forCompound != nullptr);
    endIndex = static_cast<int>(forCompound->beginId);
  } else {
    endIndex = 0;
  }
  for (int loopId = static_cast<int>(setSync->GetSyncIRIndex()); loopId >= endIndex;
       loopId--) {
    auto *tmpIr = syncIR[loopId].get();
    assert(tmpIr != nullptr);
    if (auto *loopInst = dyn_cast<LoopInstanceElement>(syncIR[loopId].get())) {
      if (loopInst->getLoopKind() == KindOfLoop::LOOP_END) {
        loopId = static_cast<int>(loopInst->beginId);
      } else if (loopInst->getLoopKind() == KindOfLoop::LOOP_BEGIN) {
        break;
      }
    }
    if (auto *branchInst = dyn_cast<BranchInstanceElement>(syncIR[loopId].get())) {
      if (branchInst->getBranchKind() == KindOfBranch::IF_END) {
        loopId = static_cast<int>(branchInst->beginId);
      } else if (branchInst->getBranchKind() == KindOfBranch::IF_BEGIN ||
                 branchInst->getBranchKind() == KindOfBranch::ELSE_BEGIN) {
        break;
      }
    }
    for (auto &setSame : tmpIr->pipeAfter) {
      // Both are forward sync.
      bool isBothForward = !setSame->GetForEndIndex().has_value() &&
                           !setSync->GetForEndIndex().has_value();
      // Both are same back sync.
      bool isBothBackward = setSame->GetForEndIndex().has_value() &&
                            setSync->GetForEndIndex().has_value() &&
                            (setSame->GetForEndIndex().value() ==
                             setSync->GetForEndIndex().value());
      // Match sync types.
      bool isSameTypeSync = setSame != setSync &&
                            setSame->GetDstPipe() == setSync->GetDstPipe() &&
                            setSame->GetSrcPipe() == setSync->GetSrcPipe() &&
                            (isBothBackward || isBothForward);
      if (!isSameTypeSync || setSame->uselessSync ||
          setSame->eventIds.empty()) {
        continue;
      }
      auto &syncPair = syncOperations[setSame->GetSyncIndex()];

      assert(syncPair.size() > 1);
      SyncOperation *waitSame = syncPair[1].get();
      // Forward sync widen as :
      //   set_flag(dstPipe, srcPipe, event_id_0)    ---> setSame
      //   set_flag(dstPipe, srcPipe, -1)            ---> setSync
      //   wait_flag(dstPipe, srcPipe, event_id_0)   ---> waitSame
      //   wait_flag(dstPipe, srcPipe, -1)           ---> waitSync
      bool canForwardReuse =
          !setSame->GetForEndIndex().has_value() &&
          (setSync->GetSyncIRIndex() > setSame->GetSyncIRIndex() &&
           setSync->GetSyncIRIndex() <= waitSame->GetSyncIRIndex());

      // Back sync widen as :
      //   for :
      //     wait_flag(dstPipe, srcPipe, event_id_0)    ---> waitSame
      //     wait_flag(dstPipe, srcPipe, -1)            ---> waitSync
      //     set_flag(dstPipe, srcPipe, event_id_0)     ---> setSame
      //     set_flag(dstPipe, srcPipe, -1)             ---> setSync
      // or :
      // Back sync widen as :
      //   for :
      //     wait_flag(dstPipe, srcPipe, event_id_0)    ---> waitSame
      //     set_flag(dstPipe, srcPipe, event_id_0)     ---> setSame

      //     wait_flag(dstPipe, srcPipe, -1)            ---> waitSync
      //     set_flag(dstPipe, srcPipe, -1)             ---> setSync
      bool canBackwardReuse =
          setSame->GetForEndIndex().has_value() &&
          ((waitSync->GetSyncIRIndex() < setSame->GetSyncIRIndex() &&
            waitSync->GetSyncIRIndex() >= waitSame->GetSyncIRIndex()) ||
           (setSame->GetSyncIRIndex() < waitSync->GetSyncIRIndex() &&
            waitSame->GetSyncIRIndex() < waitSync->GetSyncIRIndex())) &&
          (waitSame->eventIdNum == waitSync->eventIdNum);
      if (canForwardReuse || canBackwardReuse) {
        if (!widenSetSync) {
          widenSetSync = setSame;
        } else {
          // Choose the sync of which the reuse times is smallest.
          widenSetSync =
              widenSetSync->reuseCntForWiden > setSame->reuseCntForWiden
                  ? setSame
                  : widenSetSync;
        }
      }
    }
  }
  return widenSetSync;
}

const llvm::DenseMap<std::pair<hivm::PIPE, hivm::PIPE>, uint64_t>
    SyncEventIdAllocation::reservedEventIdNum = {
        {{hivm::PIPE::PIPE_V, hivm::PIPE::PIPE_S}, 1},
        {{hivm::PIPE::PIPE_S, hivm::PIPE::PIPE_V}, 1},
        {{hivm::PIPE::PIPE_MTE2, hivm::PIPE::PIPE_V}, 1},
};
