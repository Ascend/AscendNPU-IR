//===---------- MoveSyncState.h ----Move out sync for for anda if ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
