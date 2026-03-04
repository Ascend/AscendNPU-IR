//===------------ RemoveRedundantSync.h ----Remove redundant sync ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_REMOVEREDUNDANTSYNC_H
#define BISHENGIR_REMOVEREDUNDANTSYNC_H

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"

namespace mlir {
namespace hivm {

class RemoveRedundantSync {
public:
  RemoveRedundantSync(
      SyncIRs &syncIR, SyncOperations &syncOperations,
      SyncAnalysisMode syncAnalysisMode = SyncAnalysisMode::NORMALSYNC)
      : syncIR(syncIR), syncOperations(syncOperations),
        syncAnalysisMode(syncAnalysisMode){};

  ~RemoveRedundantSync() = default;

  /// Plan entrance, remove redundant sync.
  void Plan();

private:
  /// Save the Global syncIR.
  SyncIRs &syncIR;

  /// Save the Global Sync Memory.
  SyncOperations &syncOperations;

  SyncAnalysisMode syncAnalysisMode{SyncAnalysisMode::NORMALSYNC};

private:
  /// Check if there is the same synchronization.
  bool CheckAllSync(SyncOperation *setFlag, SyncOperation *waitFlag);

  /// Check for repeat synchronization within the synchronized lifecycle.
  bool CheckRepeatSync(unsigned int begin, unsigned int end,
                       SmallVector<bool> &syncFinder, SyncOperation *setFlag);

  /// Check if duplicate synchronization matches both if and else.
  bool CheckBranchBetween(BranchInstanceElement *branchElement,
                          SmallVector<bool> syncFinder, SyncOperation *setFlag,
                          unsigned endId, unsigned &i);

  /// Check if duplicate synchronization matches both if and else.
  bool CheckLoopBetween(const LoopInstanceElement *loopElement,
                        SyncOperation *setFlag, unsigned &i);

  /// Check if duplicate synchronization is matched.
  bool CanMatchedSync(SmallVector<bool> &syncFinder, SyncOperation *relatedSync,
                      SyncOperation *setFlag);
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_REMOVEREDUNDANTSYNC_H