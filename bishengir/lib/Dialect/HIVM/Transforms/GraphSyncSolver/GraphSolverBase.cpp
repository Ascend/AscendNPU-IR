//===----------- GraphSolverBase.cpp ---- Graph Sync Solver ---------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/GraphSolver.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-gss-graph-solver"

using namespace mlir;
using namespace hivm::syncsolver;

void GraphSolverBase::clearBarrierIndexes() {
  barrierAllIndexes.clear();
  barrierIndexes.clear();
}

void GraphSolverBase::addConflictPair(ConflictPair *conflictPair, bool isTemp) {
  assert(conflictPair != nullptr);
  LLVM_DEBUG({
    llvm::dbgs() << "add-conflict-pair:\n";
    llvm::dbgs() << conflictPair->str() << '\n';
  });
  if (isTemp) {
    assert(!conflictPair->isBarrier());
    addPair(conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo,
            conflictPair, /*isTemp=*/true);
    return;
  }
  if (conflictPair->isBarrier() &&
      conflictPair->setCorePipeInfo.pipe == hivm::PIPE::PIPE_ALL) {
    assert(conflictPair->startIndex == conflictPair->endIndex);
    barrierAllIndexes.push_back(conflictPair->endIndex);
    return;
  }
  if (conflictPair->isBarrier()) {
    assert(conflictPair->startIndex == conflictPair->endIndex);
    barrierIndexes[conflictPair->setCorePipeInfo].push_back(
        conflictPair->endIndex);
  }
  addPair(conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo,
          conflictPair);
}

bool GraphSolverBase::checkAnyBarrierAllBetween(int startIndex, int endIndex) {
  for (auto barrierIndex : barrierAllIndexes) {
    if (startIndex <= barrierIndex && barrierIndex <= endIndex) {
      return true;
    }
  }
  return false;
}

bool GraphSolverBase::checkAnyBarrierBetween(CorePipeInfo corePipe,
                                             int startIndex, int endIndex) {
  for (auto barrierIndex : barrierIndexes[corePipe]) {
    if (startIndex <= barrierIndex && barrierIndex <= endIndex) {
      return true;
    }
  }
  return false;
}
