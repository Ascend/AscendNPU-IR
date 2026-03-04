//===---------- SyncSolverTest.cpp ---- Graph Sync Solver------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverTest.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <climits>
#include <numeric>
#include <utility>

#define DEBUG_TYPE "hivm-gss-solver-test"

using namespace mlir;
using namespace hivm::syncsolver;

// Lightweight memory-conflict checker used by the test harness (integer ptr
// model).
bool SolverTest::checkTestRWMemoryConflicts(
    const llvm::SmallVector<llvm::SmallVector<int>> &memValsList1,
    const llvm::SmallVector<llvm::SmallVector<int>> &memValsList2) {
  for (auto &ptr1 : memValsList1) {
    for (auto &ptr2 : memValsList2) {
      for (auto val1 : ptr1) {
        for (auto val2 : ptr2) {
          if (val1 == val2) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Compute candidate pipe pairs for memory conflicts between two RW operations
// using the test-model memory lists.
std::vector<std::pair<hivm::PIPE, hivm::PIPE>>
SolverTest::checkTestMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  auto [it, isInserted] =
      checkTestMemoryConflictsMem.insert({{rwOp1, rwOp2}, {}});
  if (!isInserted) {
    return it->second;
  }
  llvm::SetVector<std::pair<hivm::PIPE, hivm::PIPE>> collectedConflictsSet;
  if (checkTestRWMemoryConflicts(rwOp1->testReadMemVals,
                                 rwOp2->testWriteMemVals)) {
    collectedConflictsSet.insert({rwOp1->pipeRead, rwOp2->pipeWrite});
  }
  if (checkTestRWMemoryConflicts(rwOp1->testWriteMemVals,
                                 rwOp2->testReadMemVals)) {
    collectedConflictsSet.insert({rwOp1->pipeWrite, rwOp2->pipeRead});
  }
  if (checkTestRWMemoryConflicts(rwOp1->testWriteMemVals,
                                 rwOp2->testWriteMemVals)) {
    collectedConflictsSet.insert({rwOp1->pipeWrite, rwOp2->pipeWrite});
  }
  std::vector<std::pair<hivm::PIPE, hivm::PIPE>> collectedConflicts(
      collectedConflictsSet.begin(), collectedConflictsSet.end());
  return it->second = collectedConflicts;
}

// Validate whether a chosen eventIdNum avoids conflicts across repeating
// memory access patterns (LCM coverage check).
bool SolverTest::checkEventIdNum(
    const llvm::SmallVector<llvm::SmallVector<int>> &memValsList1,
    const llvm::SmallVector<llvm::SmallVector<int>> &memValsList2, int lcmLen,
    int eventIdNum) {
  for (auto &ptr1 : memValsList1) {
    for (auto &ptr2 : memValsList2) {
      auto sz1 = static_cast<int>(ptr1.size());
      auto sz2 = static_cast<int>(ptr2.size());
      for (int i = 0; i < lcmLen; i++) {
        for (int j = 0; j < lcmLen; j++) {
          if (i % eventIdNum != j % eventIdNum) {
            auto val1 = ptr1[i % sz1];
            auto val2 = ptr2[j % sz2];
            if (val1 == val2) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

// Find maximum safe number of event ids for two RW test ops based on shapes.
int64_t SolverTest::getTestEventIdNum(RWOperation *rwOp1, RWOperation *rwOp2) {
  int lcm = 1;
  int minWriteSize = INT_MAX;

  for (auto &ptr : rwOp1->testReadMemVals) {
    if (auto sz = static_cast<int>(ptr.size()); sz)
      lcm = (lcm * sz) / std::gcd(lcm, sz);
  }

  for (auto &ptr : rwOp1->testWriteMemVals) {
    if (auto sz = static_cast<int>(ptr.size()); sz) {
      minWriteSize = std::min(minWriteSize, sz);
      lcm = (lcm * sz) / std::gcd(lcm, sz);
    }
  }

  for (auto &ptr : rwOp2->testReadMemVals) {
    if (auto sz = static_cast<int>(ptr.size()); sz)
      lcm = (lcm * sz) / std::gcd(lcm, sz);
  }

  for (auto &ptr : rwOp2->testWriteMemVals) {
    if (auto sz = static_cast<int>(ptr.size()); sz) {
      minWriteSize = std::min(minWriteSize, sz);
      lcm = (lcm * sz) / std::gcd(lcm, sz);
    }
  }

  // In case no write sizes were positive.
  if (minWriteSize == INT_MAX)
    minWriteSize = 1;

  int eventIdNum = minWriteSize;
  while (eventIdNum > 1) {
    int lcmLen = (lcm * eventIdNum) / std::gcd(lcm, eventIdNum);

    bool okRW = checkEventIdNum(rwOp1->testReadMemVals, rwOp2->testWriteMemVals,
                                lcmLen, eventIdNum);

    bool okWR = checkEventIdNum(rwOp1->testWriteMemVals, rwOp2->testReadMemVals,
                                lcmLen, eventIdNum);

    bool okWW = checkEventIdNum(rwOp1->testWriteMemVals,
                                rwOp2->testWriteMemVals, lcmLen, eventIdNum);

    if (okRW && okWR && okWW)
      break;

    eventIdNum--;
  }

  return eventIdNum;
}

// Test-mode variant of getTestEventIdNum that uses occurrences (wraps ops).
int64_t SolverTest::getTestEventIdNum(Occurrence *occ1, Occurrence *occ2,
                                      hivm::PIPE setPipe, hivm::PIPE waitPipe) {
  assert(occ1 != nullptr && occ2 != nullptr);
  assert(occ1->op != nullptr && occ2->op != nullptr);

  if (!isBackwardSync(occ1, occ2)) {
    return 1;
  }

  auto *parLoop1 = OperationBase::getParentloop(occ1->op);
  auto *parLoop2 = OperationBase::getParentloop(occ2->op);

  if (parLoop1 == nullptr || parLoop2 == nullptr) {
    return 1;
  }
  if (parLoop1 != parLoop2) {
    return 1;
  }

  auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
  assert(setOcc->op != nullptr);
  assert(waitOcc->op != nullptr);
  if (!parLoop1->isProperAncestor(setOcc->op) ||
      !parLoop1->isProperAncestor(waitOcc->op)) {
    return 1;
  }

  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 && rwOp2);

  return getTestEventIdNum(rwOp1, rwOp2);
}

void SolverTest::processConflict(Occurrence *occ1, Occurrence *occ2,
                                 RWOperation *rwOp1, RWOperation *rwOp2,
                                 bool isUseless) {
  for (auto [setPipe, waitPipe] : checkTestMemoryConflicts(rwOp1, rwOp2)) {
    auto corePipeSrc = CorePipeInfo(rwOp1->coreType, setPipe);
    auto corePipeDst = CorePipeInfo(rwOp2->coreType, waitPipe);
    auto eventIdNum = getTestEventIdNum(occ1, occ2, setPipe, waitPipe);
    handleConflict(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst,
                   isUseless, eventIdNum,
                   /*multibufferLoopPar=*/nullptr);
  }
}
