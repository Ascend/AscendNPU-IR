//===-------------- SyncEventIdAllocation.h ----Event id allocate ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_SYNCEVENTIDALLOCATION_H
#define BISHENGIR_SYNCEVENTIDALLOCATION_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include <cstdint>

namespace mlir {
namespace hivm {

constexpr const uint kTotalEventIdNum = 8;

constexpr const uint kBlockSyncAllCubeEventId = 14;

constexpr const uint kBlockSyncAllVectorEventId = 15;

constexpr const uint kBlockSyncSetWaitEventIdNum = 16;

// The maximum number of attempts to widen event IDs.
constexpr const uint kMaxWidenTryNum = 99;

/// Sync event id cycle pool
struct EventCyclePool {
  SmallVector<SmallVector<unsigned>> slot;
  explicit EventCyclePool(size_t size = 0) : slot(size) {}
};

using SyncCycle = DenseMap<int, EventCyclePool>;

class SyncEventIdAllocation {
public:
  SyncEventIdAllocation(SyncIRs &syncIR, SyncOperations &syncOperations)
      : syncIR(syncIR), syncOperations(syncOperations) {
    // Reserved eventIds for block-all operations if needed.
    reserveBlockAllEventIds();
  };

  ~SyncEventIdAllocation() = default;

  /// Allocate entrance, allocate sync event id.
  void Allocate(uint32_t runNum = 0);

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

  /// clear all allocated event ids.
  void clearAllocatedEventId();

  /// Get unused event id.
  SmallVector<bool> GetEventIdIdleStatus(SyncOperation *sync,
                                         size_t eventIdNum);

  /// Change unassigned event_id sync to pipe_all.
  llvm::LogicalResult ChangeNoEventIdSyncToPipeAll();

  /// Move the reassigned reallocated sync to head and tail.
  void MoveOutBackwardMatchSync(const SyncOperation *reallocatedSync);

  /// Check if there are sync of the same type that can be replaced and perform
  /// widen processing.
  bool TryWidenByOtherSync(const SyncOperation *sync);

  /// Try to widen on the first found relocatable sync pair.
  bool tryWidenOnFirstFound();

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

  /// Check wether block-all operations is used.
  void reserveBlockAllEventIds();

  /// Check if any sync operations failed to allocate event IDs and if there are
  /// multibuffer syncs, fall back multibuffer syncs to 2-buffer and or 1-buffer
  /// depending on the multibuffer num is odd or even.
  bool TryFallbackMultiBuffer();

private:
  /// Save the Global syncIR.
  SyncIRs &syncIR;

  /// Save the Global Sync Memory.
  SyncOperations &syncOperations;

  /// map from scope pair to EventPool.
  SyncCycle eventCyclePool;

  /// Record reallocated pipe pairs.
  llvm::SmallSet<int, 16> reallocatedPipePair;

  // Record inserted backward sync pairs.
  llvm::DenseSet<SyncOperation *> insertedBackwardSync;

  /// Reserved event IDs used in the template library.
  static const llvm::DenseMap<std::pair<hivm::PIPE, hivm::PIPE>, uint64_t>
      reservedEventIdNum;

  /// Number of reserved event IDs for block synchronization.
  uint64_t reservedBlockSyncEventIdNum{0};
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_SYNCEVENTIDALLOCATION_H
