//===-------------- SyncEventIdAllocation.h ----Event id allocate ---------===//
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
#ifndef BISHENGIR_SYNCEVENTIDALLOCATION_H
#define BISHENGIR_SYNCEVENTIDALLOCATION_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace hivm {

constexpr const uint kTotalEventIdNum = 8;

constexpr const uint kBlockSyncAllCubeEventId = 14;

constexpr const uint kBlockSyncAllVectorEventId = 15;

constexpr const uint kBlockSyncSetWaitEventIdNum = 14;

/// Sync event id cycle pool
struct EventCyclePool {
  SmallVector<SmallVector<unsigned>> slot;
  explicit EventCyclePool(size_t size = 0) : slot(size) {}
};

using SyncCycle = DenseMap<int, EventCyclePool>;

class SyncEventIdAllocation {
public:
  SyncEventIdAllocation(SyncIRs &syncIR, SyncOperations &syncOperations)
      : syncIR(syncIR), syncOperations(syncOperations){};

  ~SyncEventIdAllocation() = default;

  /// Allocate entrance, allocate sync event id.
  void Allocate();

private:
  /// Allocate sync event id.
  void AllocateEventId(InstanceElement *e);

  /// Obtain the number of IDs used for synchronization.
  size_t GetCompilerAvailableEventIdNum(const SyncOperation *sync);

  /// Set event id to sync.
  void SetEventId(SyncOperation *sync);

  /// Get the current allocation status of the EventPool.
  SmallVector<bool> GetEventPool(const SyncOperation *sync, size_t eventIdNum);

  /// SrcPipe and dstPipe for int conversion.
  int ScopePair(const SyncOperation *s);

  /// Find event id in EventPool that are already in use and have conflicts.
  void FindUseEventID(unsigned int begin, unsigned int end,
                      const SyncOperation *s, SmallVector<bool> &eventId);

  /// Check event if conflicts based on sync life cycle.
  bool CheckSyncLifeCycleConflict(SmallVector<unsigned int> &syncLifeCycle,
                                  unsigned int begin, unsigned int end,
                                  SmallVector<bool> &eventId, unsigned i) const;

  /// Update the status of the allocated event table.
  void UpdateEventId(SmallVector<unsigned int> &syncLifeCycle,
                     const unsigned int begin, const unsigned int end,
                     SmallVector<bool> &eventId, const unsigned index) const;

  /// Set event id and update EventPool.
  void SetEventPool(const SyncOperation *sync, unsigned eventId);

  /// Reverse sync requires completion of setFlag or waitFlag in the IR header
  /// or tail.
  void UpdateBackwardMatchSync(const SyncOperation *setFlag,
                               const SyncOperation *waitFlag, unsigned eventId);

  /// Update EventPool.
  void SetUseEventID(unsigned int begin, unsigned int end,
                     const SyncOperation *setFlag, unsigned int eventId);

  /// Expand the current sync life cycle.
  bool ExtendLifecycle(SmallVector<unsigned int> &syncLifeCycle,
                       unsigned int beginNew, unsigned int endNew) const;

  /// Event id not assigned, widen even id.
  void WidenEventId(SyncOps syncVector);

  /// Reallocate even id pairs of unallocated pipes.
  void ReallocatedEventId();

  /// clear already insert reverse head tail sync.
  void ClearReallocatedBackwardMatchSync();

  /// Get unused event id.
  SmallVector<bool> GetEventIdIdleStatus(SyncOperation *sync,
                                         size_t eventIdNum);

  /// Change unassigned event_id sync to pipe_all.
  void ChangeNoEventIdSyncToPipeAll();

  /// Move the reassigned reallocated sync to head and tail.
  void MoveOutBackwardMatchSync(const SyncOperation *reallocatedSync);

  /// Check if there are sync of the same type that can be replaced and perform
  /// widen processing.
  bool CanWidenByOtherSync(const SyncOperation *sync);

  /// Find sync of the same type that can be widen.
  SyncOperation *FindWidenSync(const SyncOperation *setSync,
                               const SyncOperation *waitSync);

  /// Clearly identify all the eventIDs of the sync.
  void ClearEventId(const SyncOperation *sync);

  /// Get the currently available Ids.
  SmallVector<int>
  GetAvailableEventId(SyncOperation *sync,
                      SmallVector<bool> eventIdLifetimeAvailableStatus,
                      SmallVector<bool> eventIdIdleStatus, size_t eventIdNum);

  SmallVector<int>
  UpdateBlockAvailableEventId(SyncOperation *sync,
                              SmallVector<bool> eventIdLifetimeAvailableStatus,
                              size_t eventIdNum);

  /// Set block sync all_cube and all_vector event id.
  void SetBlockSyncAllEventID(SyncOperation *sync);

  /// Ignore the insertion synchronization because of reverse.
  void IgnoreBackHeadAndTailSync();

private:
  /// Save the Global syncIR.
  SyncIRs &syncIR;

  /// Save the Global Sync Memory.
  SyncOperations &syncOperations;

  /// map from scope pair to EventPool.
  SyncCycle eventCyclePool;

  llvm::SmallSet<int, 16> reallocatedPipePair;

  static const llvm::DenseMap<std::pair<hivm::PIPE, hivm::PIPE>, uint64_t>
      reservedEventIdNum;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_SYNCEVENTIDALLOCATION_H
