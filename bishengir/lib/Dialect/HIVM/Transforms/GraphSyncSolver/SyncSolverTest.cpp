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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverTester.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <climits>
#include <memory>
#include <utility>

#define DEBUG_TYPE "hivm-graph-sync-solver"

using namespace mlir;
using namespace hivm::syncsolver;

// Lightweight memory-conflict checker used by the test harness (integer ptr model).
bool Solver::checkTestRWMemoryConflicts(
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
Solver::checkTestMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  auto [it, inserted] =
      checkTestMemoryConflictsMem.insert({{rwOp1, rwOp2}, {}});
  if (!inserted) {
    return it->second;
  }
  std::vector<std::pair<hivm::PIPE, hivm::PIPE>> collectedConflicts;
  if (checkTestRWMemoryConflicts(rwOp1->testReadMemVals,
                                 rwOp2->testWriteMemVals)) {
    collectedConflicts.emplace_back(rwOp1->pipeRead, rwOp2->pipeWrite);
  }
  if (checkTestRWMemoryConflicts(rwOp1->testWriteMemVals,
                                 rwOp2->testReadMemVals)) {
    collectedConflicts.emplace_back(rwOp1->pipeWrite, rwOp2->pipeRead);
  }
  if (checkTestRWMemoryConflicts(rwOp1->testWriteMemVals,
                                 rwOp2->testWriteMemVals)) {
    collectedConflicts.emplace_back(rwOp1->pipeWrite, rwOp2->pipeWrite);
  }
  return it->second = collectedConflicts;
}

// Validate whether a chosen eventIdNum avoids conflicts across repeating
// memory access patterns (LCM coverage check).
bool Solver::checkEventIdNum(
    const llvm::SmallVector<llvm::SmallVector<int>> &memValsList1,
    const llvm::SmallVector<llvm::SmallVector<int>> &memValsList2, int lcmLen,
    int eventIdNum) {
  for (auto &ptr1 : memValsList1) {
    for (auto &ptr2 : memValsList2) {
      size_t sz1 = ptr1.size();
      size_t sz2 = ptr2.size();
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
uint32_t Solver::getTestEventIdNum(RWOperation *rwOp1, RWOperation *rwOp2) {
  int lcm = 1;
  int minWriteSize = INT_MAX;

  for (auto ptr : rwOp1->testReadMemVals) {
    if (auto sz = static_cast<int>(ptr.size()); sz)
      lcm = (lcm * sz) / std::gcd(lcm, sz);
  }

  for (auto ptr : rwOp1->testWriteMemVals) {
    if (auto sz = static_cast<int>(ptr.size()); sz) {
      minWriteSize = std::min(minWriteSize, sz);
      lcm = (lcm * sz) / std::gcd(lcm, sz);
    }
  }

  for (auto ptr : rwOp2->testReadMemVals) {
    if (auto sz = static_cast<int>(ptr.size()); sz)
      lcm = (lcm * sz) / std::gcd(lcm, sz);
  }

  for (auto ptr : rwOp2->testWriteMemVals) {
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

    bool okRW = checkEventIdNum(rwOp1->testReadMemVals,
                                rwOp2->testWriteMemVals,
                                lcmLen, eventIdNum);
    
    bool okWR = checkEventIdNum(rwOp1->testWriteMemVals,
                                rwOp2->testReadMemVals,
                                lcmLen, eventIdNum);
    
    bool okWW = checkEventIdNum(rwOp1->testWriteMemVals,
                                rwOp2->testWriteMemVals,
                                lcmLen, eventIdNum);

    if (okRW && okWR && okWW)
      break;

    eventIdNum--;
  }

  return eventIdNum;
}

// Test-mode variant of getTestEventIdNum that uses occurrences (wraps ops).
uint32_t Solver::getTestEventIdNum(Occurrence *occ1, Occurrence *occ2,
                                   hivm::PIPE setPipe, hivm::PIPE waitPipe) {
  assert(occ1 && occ2);
  assert(occ1->op && occ2->op);

  if (barrierAllPairs.contains({setPipe, waitPipe}))
    return 1;

  if (!isBackwardSync(occ1, occ2))
    return 1;

  auto parLoop1 = OperationBase::getParentloop(occ1->op);
  auto parLoop2 = OperationBase::getParentloop(occ2->op);

  if (parLoop1 == nullptr || parLoop2 == nullptr || parLoop1 != parLoop2)
    return 1;

  auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
  if (isa<Ghost>(setOcc->op) || isa<Ghost>(waitOcc->op))
    return 1;

  assert(setOcc->op && waitOcc->op);

  if (!parLoop1->isProperAncestor(setOcc->op) ||
      !parLoop1->isProperAncestor(waitOcc->op))
    return 1;
  
  auto rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 && rwOp2);

  return getTestEventIdNum(rwOp1, rwOp2);
}

// Process the processing orders in test mode, discover conflicts and call the handler.
void Solver::processOrdersTest() {
  for (auto &[curOcc, start, end, reverseOrder, isUseless, skip] :
       processingOrders) {
    assert(start <= end + 1);

    if (start > end)
      continue;

    if (skip) {
      for (int i = start; i <= end; ++i)
        skipOcc.insert(syncIr[i].get());
      continue;
    }

    if (checkSkippable(curOcc))
      continue;

    assert(llvm::isa_and_present<RWOperation>(curOcc->op));

    int iStart = reverseOrder ? end : start;
    int iEnd = reverseOrder ? start - 1 : end + 1;
    int iStep = reverseOrder ? -1 : 1;

    for (int i = iStart; i != iEnd; i += iStep) {
      auto op = syncIr[i]->op;
      if (!llvm::isa_and_present<RWOperation>(op))
        continue;

      Occurrence *occ1 = reverseOrder ? syncIr[i].get() : curOcc;
      Occurrence *occ2 = reverseOrder ? curOcc : syncIr[i].get();

      if (checkSkippable(occ1) || checkVisited(occ1, occ2) ||
          checkImpossibleOccPair(occ1, occ2) || checkAlreadySynced(occ1, occ2))
        continue;

      auto rwOp1 = dyn_cast<RWOperation>(occ1->op);
      auto rwOp2 = dyn_cast<RWOperation>(occ2->op);
      assert(rwOp1 && rwOp2);

      LLVM_DEBUG({
        llvm::dbgs() << "checking: " << (isUseless ? "is-useless\n" : "\n");
        llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->op->str(0, false)
                     << '\n';
        llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->op->str(0, false)
                     << '\n';
        llvm::dbgs() << "memConflictsNum: "
                     << checkTestMemoryConflicts(rwOp1, rwOp2).size() << '\n';
      });
      for (auto [setPipe, waitPipe] : checkTestMemoryConflicts(rwOp1, rwOp2)) {
        auto eventIdNum = getTestEventIdNum(occ1, occ2, setPipe, waitPipe);
        if (checkGraphConflict(occ1, occ2, setPipe, waitPipe, eventIdNum)) {
          handleConflict(occ1, occ2, setPipe, waitPipe, isUseless, eventIdNum,
                         nullptr);
        }
      }
    }
  }
}

// Orchestrate iterative test-mode solving passes and optional merging behavior.
void Solver::solveTest(int runNum) {
  LLVM_DEBUG(llvm::dbgs() << "runNum: " << runNum << '\n');
  processOrdersTest();
  if (reuseSyncPairToSaveEventIds && !barrierAllPairs.empty()) {
    reusePairs.clear();
    for (auto [pipeSrc, pipeDst] : barrierAllPairs) {
      reusePairs[{pipeSrc, pipeDst}] = 1;
    }
    barrierAllPairs.clear();
    reset();
    insertedBarrierAllBefore.clear();
    processOrdersTest();
  }
  if (disableMultiEventIdForBarrierAllPairs && !barrierAllPairs.empty()) {
    reset();
    insertedBarrierAllBefore.clear();
    processOrdersTest();
  }
  if (considerMergedBackwardSyncEventIds) {
    getBeforeAfterSyncMaps();
    backwardSyncEventsAfterMerge = backwardSyncEvents;
    reset();
    insertedBarrierAllBefore.clear();
    processOrdersTest();
  }
  if (!insertedBarrierAllBefore.empty() && runNum < 99) {
    reset();
    pickAndInsertABarrierAll();
    insertedBarrierAllBefore.clear();
    backwardSyncEventsAfterMerge.clear();
    barrierAllPairs.clear();
    solveTest(runNum + 1);
  }
}

// If environment indicates tester mode, parse env vars and run SyncTester.
bool SyncTester::runTestMode() {
  const char *testModeEnv = std::getenv("BISHENGIR_GSS_TESTER");
  bool testMode = false;

  if (testModeEnv != nullptr)
    testMode = std::stoull(std::string(testModeEnv)) != 0;

  if (!testMode)
    return false;

  // Read env vars exactly once.
  const char *seedEnv = std::getenv("BISHENGIR_GSS_TESTER_SEED");
  const char *numOpsEnv = std::getenv("BISHENGIR_GSS_TESTER_NUM_OPS");
  const char *numPtrsEnv = std::getenv("BISHENGIR_GSS_TESTER_NUM_PTRS");
  const char *multiBufEnv = std::getenv("BISHENGIR_GSS_TESTER_ENABLE_MULTIBUFFER");

  // Copy them into std::string.
  std::optional<uint64_t> seed;
  if (seedEnv != nullptr)
    seed = std::stoull(std::string(seedEnv));

  int numOperations = 40;
  if (numOpsEnv != nullptr)
    numOperations = static_cast<int>(std::stoull(std::string(numOpsEnv)));

  int numPointers = 20;
  if (numPtrsEnv != nullptr)
    numPointers = static_cast<int>(std::stoull(std::string(numPtrsEnv)));

  bool enableMultiBuffer = false;
  if (multiBufEnv != nullptr)
    enableMultiBuffer = std::stoull(std::string(multiBufEnv)) != 0;

  unsigned usedPipesMask = 0u;
  for (auto pipe : { hivm::PIPE::PIPE_MTE1,
                     hivm::PIPE::PIPE_MTE2,
                     hivm::PIPE::PIPE_MTE3}) {
    usedPipesMask |= (1u << static_cast<unsigned>(pipe));
  }

  // Construct tester with fully validated params.
  SyncTester tester(numOperations, numPointers, usedPipesMask,
                    enableMultiBuffer, seed);
  
  auto status = tester.test();
  llvm::outs() << (llvm::succeeded(status) ? "succeeded" : "failed") << "\n";
  return true;
}
