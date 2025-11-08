//===------------ RemoveRedundantSync.h ----Remove redundant sync ---------===//
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
  bool CheckLoopBetween(LoopInstanceElement *loopElement,
                        SyncOperation *setFlag, unsigned &i);

  /// Check if duplicate synchronization is matched.
  bool CanMatchedSync(SmallVector<bool> &syncFinder, SyncOperation *relatedSync,
                      SyncOperation *setFlag);
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_REMOVEREDUNDANTSYNC_H