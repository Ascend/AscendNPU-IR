//===------------- InjectSync.h ----Auto Inject Sync ----------------------===//
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

  /// Inject auto sync.
  void AutoInjectSync(bool enableUnitFlag, bool assumeAliveLoops);

private:
  func::FuncOp func_;

  void plan();
};

} // namespace hivm
} // namespace mlir

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_INJEC_SYNC_H
