//===------------- InjectSync.h ----Auto Inject Sync ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_INJECT_SYNC_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_INJECT_SYNC_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/IRTranslator.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/MoveSyncState.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/RemoveRedundantSync.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncAnalysis.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCodegen.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncDebug.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncEventIdAllocation.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include <list>

namespace mlir {
namespace hivm {

class InjectSyncAnalysis {
public:
  InjectSyncAnalysis(func::FuncOp func) : func_(func) {}

  /// Inject PIPE_ALL.
  void InjectSyncAll();

  /// Inject set_flag/wait_flag operations before and after all MmadL1
  /// operations. Is needed by reg-based template implementation.
  void InjectSetWaitPipeMPipeMTE1ForAllMmadL1();

  /// Inject auto sync.
  void AutoInjectSync(bool enableUnitFlag, bool assumeAliveLoops);

private:
  func::FuncOp func_;

  void plan();
};

} // namespace hivm
} // namespace mlir

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_INJEC_SYNC_H
