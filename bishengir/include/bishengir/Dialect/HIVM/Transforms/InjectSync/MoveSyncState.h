//===---------- MoveSyncState.h ----Move out sync for for anda if ---------===//
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
#ifndef BISHENGIR_MOVESYNCSTATE_H
#define BISHENGIR_MOVESYNCSTATE_H

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"

namespace mlir {
namespace hivm {

class MoveSyncState {
public:
  MoveSyncState(SyncIRs &syncIR, SyncOperations &syncOperations)
      : syncIR(syncIR), syncOperations(syncOperations){};

  ~MoveSyncState() = default;

  /// StateOptimize entrance, move out.
  void StateOptimize();

private:
  /// Save the Global syncIR.
  SyncIRs &syncIR;

  /// Save the Global Sync Memory.
  SyncOperations &syncOperations;

private:
  /// Move out sync outside to ifOp.
  void MoveOutBranchSync();

  /// Move out set or wait outside to ifOp.
  void PlanMoveOutBranchSync(InstanceElement *e,
                             std::pair<unsigned int, unsigned int> pair,
                             std::pair<unsigned int, unsigned int> bound);

  /// Move out wait sync outside to ifOp.
  void PlanMoveOutIfWaitSync(SyncOps &newPipeBefore, SyncOperation *s,
                             std::pair<unsigned int, unsigned int> pair,
                             std::pair<unsigned int, unsigned int> bound);

  /// Move out set sync outside to ifOp.
  void PlanMoveOutIfSetSync(SyncOps &newPipeAfter, SyncOperation *s,
                            std::pair<unsigned int, unsigned int> pair,
                            std::pair<unsigned int, unsigned int> bound);

  /// Move out sync outside to forOp.
  void MoveForSync();

  /// Move out set or wait outside to forOp.
  void MoveOutSync(InstanceElement *e,
                   std::pair<unsigned int, unsigned int> pair);

  /// Move out wait sync outside to forOp.
  void PlanMoveOutWaitSync(SyncOps &newPipeBefore, SyncOperation *s,
                           std::pair<unsigned int, unsigned int> pair);

  /// Move out set sync outside to forOp.
  void PlanMoveOutSetSync(SyncOps &newPipeAfter, SyncOperation *s,
                          const std::pair<unsigned int, unsigned int> pair);
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_MOVESYNCSTATE_H
