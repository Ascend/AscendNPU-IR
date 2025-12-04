//===--------- SyncSolver.cpp ------- Graph Sync Solver -------------------===//
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
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/GraphSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <climits>
#include <memory>
#include <utility>

#define DEBUG_TYPE "hivm-graph-sync-solver"

using namespace mlir;
using namespace hivm::syncsolver;

// Reset per-pass bookkeeping to start fresh.
void Solver::reset() {
  skipOcc.clear();
  syncedPairs.clear();
  processedOccPairs.clear();
  chosenConflictedPairs.clear();
  scopeOpChosenConflicts.clear();
  scopeOpPairChosenConflicts.clear();
  scopeOccChosenConflicts.clear();
  scopeOccPairChosenConflicts.clear();
  backwardSyncEvents.clear();
  replacedWithReusableSyncedPairs.clear();
}

// Return true if two operations cannot be synchronized due to being in
// different branches of an if (if-else mutual exclusive) under the same
// condition.
bool Solver::checkImpossibleOpPair(OperationBase *op1, OperationBase *op2) {
  assert(op1 != nullptr && op2 != nullptr);
  if (op1->op == op2->op) {
    return false;
  }
  auto [parOp1, parOp2] = OperationBase::getLCAPair(op1, op2);
  assert(parOp1 != nullptr && parOp2 != nullptr);
  bool isIfElseSituation = parOp1->parentOp != nullptr &&
                           parOp1->parentOp == parOp2->parentOp &&
                           llvm::isa_and_present<Condition>(parOp1->parentOp);
  return isIfElseSituation;
}

// Check whether occurrences belong to impossible (if-else) pairing.
bool Solver::checkImpossibleOccPair(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (occ1->op == occ2->op) {
    return false;
  }
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  assert(parOcc1 != nullptr && parOcc2 != nullptr);
  bool isIfElseSituation =
      parOcc1->parentOcc != nullptr &&
      parOcc1->parentOcc == parOcc2->parentOcc &&
      llvm::isa_and_present<Condition>(parOcc1->parentOcc->op);
  return isIfElseSituation;
}

// Detect whether occ1 and occ2 have already been covered by an earlier sync.
bool Solver::checkAlreadySynced(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  assert(occ1->op != nullptr && occ2->op != nullptr);
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  assert(parOcc1->parentOcc != nullptr && parOcc2->parentOcc != nullptr);
  auto [parOp1, parOp2] = OperationBase::getLCAPair(occ1->op, occ2->op);
  assert(parOp1 != nullptr && parOp2 != nullptr);
  assert(parOp1->parentOp != nullptr && parOp2->parentOp != nullptr);
  return OperationBase::getParentloop(parOcc1->op) !=
         OperationBase::getParentloop(parOp1);
}

// Unit-flag reuse check between two RWOperations.
bool Solver::checkAlreadySyncedWithUnitFlag(RWOperation *rwOp1,
                                            RWOperation *rwOp2) {
  if (!enableUnitFlagFeature) {
    return false;
  }
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!rwOp1->hasUnitFlagFeat || !rwOp2->hasUnitFlagFeat) {
    return false;
  }
  RWOperation *curRwOp = rwOp1->linkedUnitFlagOpAsSet;
  while (curRwOp != nullptr) {
    if (curRwOp == rwOp2) {
      return true;
    }
    curRwOp = rwOp1->linkedUnitFlagOpAsSet;
  }
  return false;
}

// Check pointer-cast based buffer overlap conservatively when addresses are
// known. Used for memref pointer-cast conflict detection.
bool Solver::checkPointerCastMemConflict(hivm::PointerCastOp pointerCastOp1,
                                         hivm::PointerCastOp pointerCastOp2) {
  auto spaceAttr1 = GetBufferSpaceAttr(pointerCastOp1.getResult());
  auto spaceAttr2 = GetBufferSpaceAttr(pointerCastOp2.getResult());
  if (!spaceAttr1.has_value() || !spaceAttr2.has_value()) {
    return false;
  }
  auto memSpace1 = spaceAttr1.value().getAddressSpace();
  auto memSpace2 = spaceAttr2.value().getAddressSpace();
  if (memSpace1 != memSpace2) {
    return false;
  }
  auto bufferSize1 = GetBufferSize(pointerCastOp1.getResult());
  auto bufferSize2 = GetBufferSize(pointerCastOp2.getResult());
  assert(bufferSize1.has_value() && bufferSize2.has_value());
  for (auto addr1 : pointerCastOp1.getAddrs()) {
    for (auto addr2 : pointerCastOp2.getAddrs()) {
      auto constOp1 =
          llvm::dyn_cast_if_present<arith::ConstantOp>(addr1.getDefiningOp());
      auto constOp2 =
          llvm::dyn_cast_if_present<arith::ConstantOp>(addr2.getDefiningOp());
      if (constOp1 == nullptr || constOp2 == nullptr) {
        return true;
      }
      uint64_t baseAddr1 = static_cast<uint64_t>(
          cast<IntegerAttr>(constOp1.getValue()).getInt());
      uint64_t baseAddr2 = static_cast<uint64_t>(
          cast<IntegerAttr>(constOp2.getValue()).getInt());
      uint64_t l1 = baseAddr1;
      uint64_t r1 = baseAddr1 + std::max((uint32_t)1, bufferSize1.value());
      uint64_t l2 = baseAddr2;
      uint64_t r2 = baseAddr2 + std::max((uint32_t)1, bufferSize2.value());
      // !(r2 <= l1 || r1 <= l2)
      if (r2 > l1 && r1 > l2) {
        return true;
      }
    }
  }
  return false;
}

// General RW memory-conflict check between lists of Values (handles
// pointer-casts).
bool Solver::checkRWMemoryConflicts(
    const llvm::SmallVector<Value> &memValsList1,
    const llvm::SmallVector<Value> &memValsList2) {
  for (auto val1 : memValsList1) {
    for (auto val2 : memValsList2) {
      if (val1 == val2) {
        return true;
      }
      if (auto pointerCastOp1 =
              dyn_cast_if_present<hivm::PointerCastOp>(val1.getDefiningOp())) {
        if (auto pointerCastOp2 = dyn_cast_if_present<hivm::PointerCastOp>(
                val2.getDefiningOp())) {
          if (checkPointerCastMemConflict(pointerCastOp1, pointerCastOp2)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

// High-level wrapper computing pipe pairs that represent memory conflicts
// between two RW ops.
std::vector<std::pair<hivm::PIPE, hivm::PIPE>>
Solver::checkMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  auto [it, inserted] = checkMemoryConflictsMem.insert({{rwOp1, rwOp2}, {}});
  if (!inserted) {
    return it->second;
  }
  std::vector<std::pair<hivm::PIPE, hivm::PIPE>> collectedConflicts;
  if (checkRWMemoryConflicts(rwOp1->readMemVals, rwOp2->writeMemVals)) {
    collectedConflicts.emplace_back(rwOp1->pipeRead, rwOp2->pipeWrite);
  }
  if (checkRWMemoryConflicts(rwOp1->writeMemVals, rwOp2->readMemVals)) {
    collectedConflicts.emplace_back(rwOp1->pipeWrite, rwOp2->pipeRead);
  }
  if (checkRWMemoryConflicts(rwOp1->writeMemVals, rwOp2->writeMemVals)) {
    collectedConflicts.emplace_back(rwOp1->pipeWrite, rwOp2->pipeWrite);
  }
  return it->second = collectedConflicts;
}

// Helpers that determine whether multi-buffer double-event-id is possible by
// exploring pointer-cast patterns.
std::optional<LoopLikeOpInterface>
Solver::checkDoubleMultiBufferEventId(hivm::PointerCastOp pointerCastOp1,
                                      hivm::PointerCastOp pointerCastOp2) {
  auto loopPar1 = pointerCastOp1->getParentOfType<LoopLikeOpInterface>();
  auto loopPar2 = pointerCastOp2->getParentOfType<LoopLikeOpInterface>();
  if (loopPar1 == nullptr || loopPar2 == nullptr) {
    return {};
  }
  if (loopPar1 != loopPar2) {
    return {};
  }
  auto bufferSize1 = GetBufferSize(pointerCastOp1.getResult());
  auto bufferSize2 = GetBufferSize(pointerCastOp2.getResult());
  assert(bufferSize1.has_value() && bufferSize2.has_value());
  auto addrs1 = pointerCastOp1.getAddrs();
  auto addrs2 = pointerCastOp2.getAddrs();
  int sz1 = addrs1.size();
  int sz2 = addrs2.size();
  assert(sz1 <= 2 && sz2 <= 2);
  const int eventIdNum = 2;
  int lcmLen = sz1 * sz2 / std::__gcd(sz1, sz2);
  lcmLen = (lcmLen * eventIdNum) / std::__gcd(lcmLen, eventIdNum);
  for (int i = 0; i < lcmLen; i++) {
    for (int j = 0; j < lcmLen; j++) {
      if (i % eventIdNum != j % eventIdNum) {
        auto addr1 = addrs1[i % sz1];
        auto addr2 = addrs2[j % sz2];
        auto constOp1 =
            llvm::dyn_cast_if_present<arith::ConstantOp>(addr1.getDefiningOp());
        auto constOp2 =
            llvm::dyn_cast_if_present<arith::ConstantOp>(addr2.getDefiningOp());
        if (constOp1 == nullptr || constOp2 == nullptr) {
          return {};
        }
        uint64_t baseAddr1 = static_cast<uint64_t>(
            cast<IntegerAttr>(constOp1.getValue()).getInt());
        uint64_t baseAddr2 = static_cast<uint64_t>(
            cast<IntegerAttr>(constOp2.getValue()).getInt());
        uint64_t l1 = baseAddr1;
        uint64_t r1 = baseAddr1 + std::max((uint32_t)1, bufferSize1.value());
        uint64_t l2 = baseAddr2;
        uint64_t r2 = baseAddr2 + std::max((uint32_t)1, bufferSize2.value());
        // !(r2 <= l1 || r1 <= l2)
        if (r2 > l1 && r1 > l2) {
          return {};
        }
      }
    }
  }
  return loopPar1;
}

std::optional<LoopLikeOpInterface> Solver::checkDoubleMultiBufferEventId(
    const llvm::SmallVector<Value> &memValsList1,
    const llvm::SmallVector<Value> &memValsList2) {
  LoopLikeOpInterface loopPar = nullptr;
  for (auto &val1 : memValsList1) {
    for (auto &val2 : memValsList2) {
      if (auto pointerCastOp1 =
              dyn_cast_if_present<hivm::PointerCastOp>(val1.getDefiningOp())) {
        if (auto pointerCastOp2 = dyn_cast_if_present<hivm::PointerCastOp>(
                val2.getDefiningOp())) {
          if (!checkPointerCastMemConflict(pointerCastOp1, pointerCastOp2)) {
            continue;
          }
          auto curLoopParOpt =
              checkDoubleMultiBufferEventId(pointerCastOp1, pointerCastOp2);
          if (!curLoopParOpt.has_value()) {
            return {};
          }
          if (loopPar != nullptr && loopPar != curLoopParOpt.value()) {
            return {};
          }
          loopPar = curLoopParOpt.value();
        }
      } else if (val1 == val2) {
        return {};
      }
    }
  }
  if (loopPar == nullptr) {
    return {};
  }
  return loopPar;
}

std::optional<LoopLikeOpInterface>
Solver::checkDoubleMultiBufferEventId(RWOperation *rwOp1, RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  LoopLikeOpInterface loopPar = nullptr;
  if (checkRWMemoryConflicts(rwOp1->readMemVals, rwOp2->writeMemVals)) {
    auto curLoopParOpt =
        checkDoubleMultiBufferEventId(rwOp1->readMemVals, rwOp2->writeMemVals);
    if (!curLoopParOpt.has_value()) {
      return {};
    }
    if (loopPar != nullptr && loopPar != curLoopParOpt.value()) {
      return {};
    }
    loopPar = curLoopParOpt.value();
  }
  if (checkRWMemoryConflicts(rwOp1->writeMemVals, rwOp2->readMemVals)) {
    auto curLoopParOpt =
        checkDoubleMultiBufferEventId(rwOp1->writeMemVals, rwOp2->readMemVals);
    if (!curLoopParOpt.has_value()) {
      return {};
    }
    if (loopPar != nullptr && loopPar != curLoopParOpt.value()) {
      return {};
    }
    loopPar = curLoopParOpt.value();
  }
  if (checkRWMemoryConflicts(rwOp1->writeMemVals, rwOp2->writeMemVals)) {
    auto curLoopParOpt =
        checkDoubleMultiBufferEventId(rwOp1->writeMemVals, rwOp2->writeMemVals);
    if (!curLoopParOpt.has_value()) {
      return {};
    }
    if (loopPar != nullptr && loopPar != curLoopParOpt.value()) {
      return {};
    }
    loopPar = curLoopParOpt.value();
  }
  if (loopPar == nullptr) {
    return {};
  }
  return loopPar;
}

// Determine required event id count and optional multibuffer loop parent for
// occurrences.
std::pair<uint32_t, LoopLikeOpInterface>
Solver::getEventIdNum(Occurrence *occ1, Occurrence *occ2, hivm::PIPE setPipe,
                      hivm::PIPE waitPipe) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (barrierAllPairs.contains({setPipe, waitPipe})) {
    return {1, nullptr};
  }
  assert(occ1->op != nullptr && occ2->op != nullptr);
  if (!isBackwardSync(occ1, occ2)) {
    return {1, nullptr};
  }
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  assert(!checkMemoryConflicts(rwOp1, rwOp2).empty());
  auto loopParOpt = checkDoubleMultiBufferEventId(rwOp1, rwOp2);
  if (!loopParOpt.has_value()) {
    return {1, nullptr};
  }
  auto loopPar = loopParOpt.value();
  assert(loopPar != nullptr);
  auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
  if (isa<Ghost>(setOcc->op) || isa<Ghost>(waitOcc->op)) {
    return {1, nullptr};
  }
  assert(setOcc->op->op != nullptr);
  assert(waitOcc->op->op != nullptr);
  if (!loopPar->isProperAncestor(setOcc->op->op)) {
    return {1, nullptr};
  }
  if (!loopPar->isProperAncestor(waitOcc->op->op)) {
    return {1, nullptr};
  }
  if (!checkAllLoopParentsAreForLoops(loopPar->getParentOp())) {
    return {1, nullptr};
  }
  return {2, loopPar};
}

// Graph-based check to determine if adding a sync between occ1 and occ2 would
// block progress. Uses GraphSolver (Dijkstra) to estimate minimal reachable
// index.
bool Solver::checkGraphConflict(Occurrence *occ1, Occurrence *occ2,
                                hivm::PIPE startPipe, hivm::PIPE endPipe,
                                uint32_t eventIdNum) {
  assert(occ1 != nullptr && occ2 != nullptr);
  GraphSolver graphSolver;
  llvm::DenseSet<ConflictPair *> visited;

  auto handleConflictPair = [&](ConflictPair *conflictPair) {
    if (conflictPair->couldNotRun) {
      return;
    }
    if (conflictPair->replacedWithUnitFlag) {
      if (conflictPair->setPipe == startPipe ||
          conflictPair->waitPipe == endPipe) {
        return;
      }
    }
    if (conflictPair->isInnerBackward &&
        conflictPair->eventIds.size() > eventIdNum) {
      return;
    }
    auto [it, inserted] = visited.insert(conflictPair);
    if (!inserted) {
      return;
    }
    graphSolver.addConflictPair(conflictPair);
  };

  for (auto *parOp : occ1->op->getAllParents()) {
    if (scopeOpChosenConflicts.contains(parOp)) {
      for (auto *conflictPair : scopeOpChosenConflicts[parOp]) {
        handleConflictPair(conflictPair);
      }
    }
  }
  for (auto *parOp : occ2->op->getAllParents()) {
    if (scopeOpChosenConflicts.contains(parOp)) {
      for (auto *conflictPair : scopeOpChosenConflicts[parOp]) {
        handleConflictPair(conflictPair);
      }
    }
  }
  for (auto &[scopeOpPair, chosenConflicts] : scopeOpPairChosenConflicts) {
    auto [scopeOp1, scopeOp2] = scopeOpPair;
    if (scopeOp1->isProperAncestor(occ1->op) &&
        scopeOp2->isProperAncestor(occ2->op)) {
      for (auto *conflictPair : chosenConflicts) {
        handleConflictPair(conflictPair);
      }
    }
  }

  for (auto *parOcc : occ1->getAllParents()) {
    if (scopeOccChosenConflicts.contains(parOcc)) {
      for (auto *conflictPair : scopeOccChosenConflicts[parOcc]) {
        handleConflictPair(conflictPair);
      }
    }
  }
  for (auto *parOcc : occ2->getAllParents()) {
    if (scopeOccChosenConflicts.contains(parOcc)) {
      for (auto *conflictPair : scopeOccChosenConflicts[parOcc]) {
        handleConflictPair(conflictPair);
      }
    }
  }
  for (auto &[scopeOccPair, chosenConflicts] : scopeOccPairChosenConflicts) {
    auto [scopeOcc1, scopeOcc2] = scopeOccPair;
    if (scopeOcc1->isProperAncestor(occ1) &&
        scopeOcc2->isProperAncestor(occ2)) {
      for (auto *conflictPair : chosenConflicts) {
        handleConflictPair(conflictPair);
      }
    }
  }

  for (auto *parOcc : occ1->getAllParents()) {
    if (persistentScopeOccChosenConflicts.contains(parOcc)) {
      for (auto *conflictPair : persistentScopeOccChosenConflicts[parOcc]) {
        handleConflictPair(conflictPair);
      }
    }
  }
  for (auto *parOcc : occ2->getAllParents()) {
    if (persistentScopeOccChosenConflicts.contains(parOcc)) {
      for (auto *conflictPair : persistentScopeOccChosenConflicts[parOcc]) {
        handleConflictPair(conflictPair);
      }
    }
  }
  for (auto &[scopeOccPair, chosenConflicts] :
       persistentScopeOccPairChosenConflicts) {
    auto [scopeOcc1, scopeOcc2] = scopeOccPair;
    if (scopeOcc1->isProperAncestor(occ1) &&
        scopeOcc2->isProperAncestor(occ2)) {
      for (auto *conflictPair : chosenConflicts) {
        handleConflictPair(conflictPair);
      }
    }
  }

  auto mnDistance = graphSolver.runDijkstra(startPipe, endPipe, occ1->endIndex,
                                            occ2->startIndex);
  return !mnDistance.has_value() || mnDistance.value() > occ2->startIndex;
}

// Obtain available event ids while accounting for already chosen conflicts.
SmallVector<hivm::EVENT>
Solver::getAvailableEventIds(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  if (conflictPair->isBarrier()) {
    return {};
  }
  llvm::DenseSet<hivm::EVENT> visitedEventIds;
  for (auto &curConflictPair : chosenConflictedPairs) {
    if (checkIntersect(conflictPair, curConflictPair.get())) {
      for (auto eventId : curConflictPair->eventIds) {
        visitedEventIds.insert(eventId);
      }
    }
  }
  for (auto &curConflictPair : persistentChosenConflictedPairs) {
    if (checkIntersect(conflictPair, curConflictPair.get())) {
      for (auto eventId : curConflictPair->eventIds) {
        visitedEventIds.insert(eventId);
      }
    }
  }
  SmallVector<hivm::EVENT> availableEventIds;
  for (auto eventId :
       getHWAvailableEventIds(conflictPair->setPipe, conflictPair->waitPipe)) {
    if (!visitedEventIds.contains(eventId)) {
      availableEventIds.push_back(eventId);
    }
  }
  return availableEventIds;
}

// Processed-pair tracking helpers.
bool Solver::checkVisited(Occurrence *occ1, Occurrence *occ2) {
  auto [it, inserted] = processedOccPairs.insert(std::make_pair(occ1, occ2));
  return !inserted;
}

bool Solver::checkSkippable(Occurrence *occ) { return skipOcc.contains(occ); }

// Synced-pair memoization helpers.
std::optional<llvm::SmallVector<hivm::EVENT>>
Solver::getOldEventIdIfExists(OperationBase *scopeOp, Occurrence *occ1,
                              Occurrence *occ2, ConflictPair *conflictPair) {
  auto it = syncedPairs.find({scopeOp, occ1->op, occ2->op,
                              conflictPair->setPipe, conflictPair->waitPipe});
  if (it == syncedPairs.end()) {
    return {};
  }
  return it->second->eventIds;
}

void Solver::memorizeSyncedPair(OperationBase *scopeOp,
                                ConflictPair *conflictPair) {
  assert(scopeOp != nullptr && conflictPair != nullptr);
  syncedPairs[{scopeOp, conflictPair->op1, conflictPair->op2,
               conflictPair->setPipe, conflictPair->waitPipe}] = conflictPair;
}

void Solver::forgetSyncedPair(OperationBase *scopeOp,
                              ConflictPair *conflictPair) {
  assert(scopeOp != nullptr && conflictPair != nullptr);
  syncedPairs[{scopeOp, conflictPair->op1, conflictPair->op2,
               conflictPair->setPipe, conflictPair->waitPipe}] = conflictPair;
}

void Solver::memorizeReusedSyncedPair(OperationBase *scopeOp,
                                      ConflictPair *conflictPair,
                                      ConflictPair *reusedConflictPair) {
  assert(scopeOp != nullptr && conflictPair != nullptr);
  replacedWithReusableSyncedPairs[{
      scopeOp, conflictPair->op1, conflictPair->op2, conflictPair->setPipe,
      conflictPair->waitPipe}] = reusedConflictPair;
}

// Select an available event id (or multiple) with optional reversed priority.
llvm::SmallVector<hivm::EVENT>
Solver::getAnyAvailableEventId(ConflictPair *conflictPair, uint32_t count,
                               bool reversedPriority) {
  assert(conflictPair != nullptr);
  auto availableEventIds = getAvailableEventIds(conflictPair);
  if (reversedPriority) {
    std::reverse(availableEventIds.begin(), availableEventIds.end());
  }
  if (availableEventIds.size() > count) {
    availableEventIds.resize(count);
  }
  return availableEventIds;
}

llvm::SmallVector<hivm::EVENT> Solver::getAnyAvailableMultiBufferEventIds(
    ConflictPair *conflictPair, uint32_t count, bool reversedPriority) {
  assert(conflictPair != nullptr);
  auto availableEventIds = getAvailableEventIds(conflictPair);
  assert(conflictPair->isInnerBackward);
  if (reversedPriority) {
    std::reverse(availableEventIds.begin(), availableEventIds.end());
  }
  if (availableEventIds.size() > count) {
    availableEventIds.resize(count);
  }
  return availableEventIds;
}

bool Solver::skipMMad1DecomposedLoopOpt(Occurrence *occ1, Occurrence *occ2) {
  auto *parentLoopOp1 = OperationBase::getParentloop(occ1->op);
  auto *parentLoopOp2 = OperationBase::getParentloop(occ2->op);
  if (parentLoopOp1 != nullptr && parentLoopOp2 != nullptr) {
    if (parentLoopOp1 != parentLoopOp2) {
      if (isa<MmadL1LoopOp>(parentLoopOp1) &&
          isa<MmadL1LoopOp>(parentLoopOp2)) {
        return true;
      }
    }
  }
  return false;
}

std::pair<Occurrence *, Occurrence *>
Solver::checkAndApplyMmadl0LoopOpt(ConflictPair *conflictPair, Occurrence *occ1,
                                   Occurrence *occ2, Occurrence *parOcc1,
                                   Occurrence *parOcc2) {
  if (occ1->parentOcc != nullptr && occ1->parentOcc->parentOcc == parOcc1 &&
      llvm::isa_and_present<syncsolver::LoadL0AOp, syncsolver::LoadL0BOp>(
          occ1->op) &&
      llvm::isa_and_present<syncsolver::MmadL1LoopOp>(parOcc1->op)) {
    conflictPair->setOnLastIterOnly = true;
    return std::make_pair(occ1, parOcc2);
  }
  if (!conflictPair->isInnerBackward && occ2->parentOcc != nullptr &&
      occ2->parentOcc->parentOcc == parOcc2 &&
      llvm::isa_and_present<syncsolver::LoadL0AOp, syncsolver::LoadL0BOp>(
          occ2->op) &&
      llvm::isa_and_present<syncsolver::MmadL1LoopOp>(parOcc2->op)) {
    conflictPair->waitOnFirstIterOnly = true;
    return std::make_pair(parOcc1, occ2);
  }
  return std::make_pair(parOcc1, parOcc2);
}

std::optional<std::pair<hivm::UNIT_FLAG, hivm::UNIT_FLAG>>
Solver::checkUnitFlagPatterns(ConflictPair *conflictPair, Occurrence *occ1,
                              Occurrence *occ2, Occurrence *parentLCALoopOcc) {
  if (!enableUnitFlagFeature) {
    return {};
  }
  if (conflictPair->isBarrier()) {
    return {};
  }
  auto *rwOp1 = dyn_cast<RWOperation>(occ1->op);
  auto *rwOp2 = dyn_cast<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!rwOp1->hasUnitFlagFeat || !rwOp2->hasUnitFlagFeat) {
    return {};
  }
  if (rwOp1->unitFlagModeAsSet != UNIT_FLAG::DISABLED ||
      rwOp2->unitFlagModeAsWait != UNIT_FLAG::DISABLED) {
    return {};
  }
  if (conflictPair->isInnerBackward) {
    assert(parentLCALoopOcc != nullptr);
    assert(parentLCALoopOcc->op != nullptr);
    assert(rwOp1->op != nullptr && rwOp2->op != nullptr);
    if (rwOp1->op->getParentOp() != parentLCALoopOcc->op->op ||
        rwOp2->op->getParentOp() != parentLCALoopOcc->op->op) {
      return {};
    }
  }
  if (auto unitFlagMode = checkMmadl1FixpipeUnitFlagPattern(
          rwOp1, rwOp2, conflictPair->setPipe, conflictPair->waitPipe)) {
    return unitFlagMode;
  }
  if (auto unitFlagMode = checkMmadl1FixpipeSingleForLoopUnitFlagPattern(
          rwOp1, rwOp2, conflictPair->setPipe, conflictPair->waitPipe,
          /*rw1IsFrontOcc=*/true)) {
    return unitFlagMode;
  }
  if (rwOp1->unitFlagModeAsWait == UNIT_FLAG::ENABLED_WITH_UPDATE ||
      rwOp1->unitFlagModeAsWait == UNIT_FLAG::ENABLED_ONLY_FIRST_ITER ||
      rwOp1->unitFlagModeAsWait ==
          UNIT_FLAG::ENABLED_ONLY_FIRST_AND_LAST_ITERS) {
    // fixpipe expect unit-flag to be enabled in a mmadl1 operation before it,
    // so by this condition, we are making sure to not link a fixpipe
    // operation with an operation after it if it was not linked with an
    // operation before it
    if (auto unitFlagMode = checkMmadl1FixpipeUnitFlagPattern(
            rwOp2, rwOp1, conflictPair->waitPipe, conflictPair->setPipe)) {
      return unitFlagMode;
    }
    if (auto unitFlagMode = checkMmadl1FixpipeSingleForLoopUnitFlagPattern(
            rwOp2, rwOp1, conflictPair->waitPipe, conflictPair->setPipe,
            /*rw1IsFrontOcc=*/false)) {
      return unitFlagMode;
    }
  }
  return {};
}

std::optional<std::pair<hivm::UNIT_FLAG, hivm::UNIT_FLAG>>
Solver::checkMmadl1FixpipeUnitFlagPattern(RWOperation *rwOp1,
                                          RWOperation *rwOp2, hivm::PIPE pipe1,
                                          hivm::PIPE pipe2) {
  auto mmadl1Op = dyn_cast<hivm::MmadL1Op>(rwOp1->op);
  auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(rwOp2->op);
  if (fixpipeOp == nullptr || mmadl1Op == nullptr) {
    return {};
  }
  if (pipe1 != PIPE::PIPE_M || pipe2 != PIPE::PIPE_FIX) {
    return {};
  }
  if (fixpipeOp.getSrc() != mmadl1Op.getC()) {
    return {};
  }
  if (fixpipeOp->getParentRegion() != mmadl1Op->getParentRegion()) {
    return {};
  }
  return std::make_pair(UNIT_FLAG::ENABLED_WITH_UPDATE,
                        UNIT_FLAG::ENABLED_WITH_UPDATE);
}

std::optional<std::pair<hivm::UNIT_FLAG, hivm::UNIT_FLAG>>
Solver::checkMmadl1FixpipeSingleForLoopUnitFlagPattern(RWOperation *rwOp1,
                                                       RWOperation *rwOp2,
                                                       hivm::PIPE pipe1,
                                                       hivm::PIPE pipe2,
                                                       bool op1IsFrontOcc) {
  auto mmadl1Op = dyn_cast<hivm::MmadL1Op>(rwOp1->op);
  auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(rwOp2->op);
  if (!fixpipeOp || !mmadl1Op) {
    return {};
  }
  if (pipe1 != PIPE::PIPE_M || pipe2 != PIPE::PIPE_FIX) {
    return {};
  }
  if (fixpipeOp.getSrc() != mmadl1Op.getC()) {
    return {};
  }
  if (fixpipeOp->getParentRegion() == mmadl1Op->getParentRegion()) {
    return {};
  }
  auto mmadl1ForOp = dyn_cast<scf::ForOp>(mmadl1Op->getParentOp());
  auto fixpipeOpForOp = dyn_cast<scf::ForOp>(fixpipeOp->getParentOp());
  if (mmadl1ForOp && fixpipeOpForOp) {
    if (mmadl1ForOp->getParentRegion() == fixpipeOpForOp->getParentRegion()) {
      return std::make_pair(UNIT_FLAG::ENABLED_ONLY_LAST_ITER,
                            UNIT_FLAG::ENABLED_ONLY_FIRST_ITER);
    }
  } else if (mmadl1ForOp) {
    if (mmadl1ForOp->getParentRegion() == fixpipeOp->getParentRegion()) {
      return op1IsFrontOcc ? std::make_pair(UNIT_FLAG::ENABLED_ONLY_LAST_ITER,
                                            UNIT_FLAG::ENABLED_WITH_UPDATE)
                           : std::make_pair(UNIT_FLAG::ENABLED_WITH_UPDATE,
                                            UNIT_FLAG::ENABLED_ONLY_FIRST_ITER);
    }
  } else if (fixpipeOpForOp) {
    if (fixpipeOpForOp->getParentRegion() == mmadl1Op->getParentRegion()) {
      return op1IsFrontOcc ? std::make_pair(UNIT_FLAG::ENABLED_WITH_UPDATE,
                                            UNIT_FLAG::ENABLED_ONLY_FIRST_ITER)
                           : std::make_pair(UNIT_FLAG::ENABLED_ONLY_LAST_ITER,
                                            UNIT_FLAG::ENABLED_WITH_UPDATE);
    }
  }
  return {};
}

std::pair<Occurrence *, Occurrence *> Solver::getSetWaitOcc(Occurrence *occ1,
                                                            Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto [parOp1, parOp2] = OperationBase::getLCAPair(occ1->op, occ2->op);
  assert(parOp1 != nullptr && parOp2 != nullptr);
  assert(parOp1->parentOp != nullptr && parOp2->parentOp != nullptr);
  assert(parOp1->parentOp == parOp2->parentOp);
  auto *parOcc1 = occ1->getParentWithOp(parOp1->parentOp);
  auto *parOcc2 = occ2->getParentWithOp(parOp2->parentOp);
  assert(parOcc1 != nullptr && parOcc2 != nullptr);
  assert(parOcc1 != occ1 && parOcc2 != occ2);
  auto *setOcc = occ1->getNthParent(occ1->depth - parOcc1->depth - 1);
  auto *waitOcc = occ2->getNthParent(occ2->depth - parOcc2->depth - 1);
  // if (setOcc->op != waitOcc->op) {
  //   // llvm::dbgs() << setOcc->op->str(0, false) << '\n';
  //   // llvm::dbgs() << waitOcc->op->str(0, false) << '\n';
  //   if (isa<Condition>(setOcc->parentOcc->op) && !isa<Ghost>(waitOcc->op)) {
  //     assert(setOcc->syncIrEndIndex != -1);
  //     assert(setOcc->syncIrEndIndex < static_cast<int>(syncIr.size()));
  //     auto *ghostOcc = syncIr[setOcc->syncIrEndIndex - 1].get();
  //     assert(isa<Ghost>(ghostOcc->op));
  //     return getSetWaitOcc(occ1, ghostOcc);
  //   }
  // }
  assert(setOcc != nullptr && waitOcc != nullptr && setOcc->parentOcc != nullptr && waitOcc->parentOcc != nullptr);
  if (setOcc->op != waitOcc->op) {
    if (auto *parLoopOp =
            llvm::dyn_cast_if_present<Loop>(setOcc->parentOcc->op)) {
      if (parLoopOp->body.size() > 1 && !isa<Ghost>(waitOcc->op)) {
        assert(setOcc->syncIrEndIndex != -1);
        assert(setOcc->syncIrEndIndex < static_cast<int>(syncIr.size()));
        auto *ghostOcc = syncIr[setOcc->syncIrEndIndex - 1].get();
        assert(isa<Ghost>(ghostOcc->op));
        return getSetWaitOcc(occ1, ghostOcc);
      }
    }
  }

  if (setOcc->parentOcc != nullptr) {
    if (llvm::isa_and_present<Condition>(setOcc->parentOcc->op)) {
      setOcc = setOcc->parentOcc;
    }
  }
  if (waitOcc->parentOcc != nullptr) {
    if (llvm::isa_and_present<Condition>(waitOcc->parentOcc->op)) {
      waitOcc = waitOcc->parentOcc;
    }
  }
  return {setOcc, waitOcc};
}

void Solver::insertBarrierAllBefore(Occurrence *occ, bool isUseless,
                                    bool isPersistent) {
  auto conflictPair = std::make_unique<ConflictPair>(
      nullptr, nullptr, occ->op, occ->op, hivm::PIPE::PIPE_ALL,
      hivm::PIPE::PIPE_ALL, occ->startIndex, occ->startIndex);
  isUseless |= llvm::any_of(
      persistentChosenConflictedPairs, [occ](const auto &conflictPair) {
        return !conflictPair->isUseless &&
               conflictPair->setPipe == hivm::PIPE::PIPE_ALL &&
               conflictPair->opSet == occ->op;
      });
  conflictPair->isUseless = isUseless;
  auto *normScopeOcc = occ->parentOcc;
  assert(normScopeOcc != nullptr);
  LLVM_DEBUG(llvm::dbgs() << occ->op->str(0, false) << ' '
                          << conflictPair->str() << '\n';);
  if (isPersistent) {
    persistentScopeOccChosenConflicts[normScopeOcc].insert(conflictPair.get());
    persistentChosenConflictedPairs.push_back(std::move(conflictPair));
  } else {
    insertedBarrierAllBefore[occ->op].insert({occ, isUseless});
    scopeOccChosenConflicts[normScopeOcc].insert(conflictPair.get());
    chosenConflictedPairs.push_back(std::move(conflictPair));
  }
}

bool Solver::isBackwardSync(Occurrence *occ1, Occurrence *occ2) {
  if (occ1->op->id >= occ2->op->id) {
    return true;
  }
  assert(occ1 != nullptr && occ2 != nullptr);
  assert(occ1->op != nullptr && occ2->op != nullptr);
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  auto [parOp1, parOp2] = OperationBase::getLCAPair(occ1->op, occ2->op);
  return parOcc1->parentOcc->op != parOp1->parentOp;
}

ConflictPair *Solver::getReusableConflictPair(
    ConflictPair *conflictPair,
    const llvm::DenseSet<ConflictPair *> &conflictPairsSet) {
  assert(conflictPair != nullptr);
  ConflictPair *ret = nullptr;
  for (auto *curConflictPair : conflictPairsSet) {
    if (curConflictPair->opSet == nullptr ||
        curConflictPair->opWait == nullptr) {
      continue;
    }
    assert(conflictPair->opSet->parentOp == curConflictPair->opSet->parentOp);
    assert(conflictPair->opWait->parentOp == curConflictPair->opWait->parentOp);
    if (!checkIntersect(conflictPair, curConflictPair)) {
      continue;
    }
    assert(curConflictPair->endIndex <= conflictPair->endIndex);
    assert(curConflictPair->startIndex < conflictPair->startIndex);
    if (ret == nullptr || ret->startIndex < curConflictPair->startIndex) {
      ret = curConflictPair;
    }
  }
  return ret;
}

bool Solver::reuseConflictPair(ConflictPair *conflictPair,
                               Occurrence *scopeOcc1, Occurrence *scopeOcc2) {
  if (conflictPair->isBarrier()) {
    return false;
  }

  auto setPipe = conflictPair->setPipe;
  auto waitPipe = conflictPair->waitPipe;

  ConflictPair *oldReusedConflictPair = nullptr;
  if (conflictPair->isUseless) {
    auto it = replacedWithReusableSyncedPairs.find(
        {scopeOcc1->op, conflictPair->op1, conflictPair->op2,
         conflictPair->setPipe, conflictPair->waitPipe});
    if (it != replacedWithReusableSyncedPairs.end()) {
      oldReusedConflictPair = it->second;
    }
  }

  if (oldReusedConflictPair == nullptr && reusePairs[{setPipe, waitPipe}] < 1) {
    return false;
  }

  ConflictPair *opt1 = nullptr;
  ConflictPair *opt2 = nullptr;
  ConflictPair *opt3 = nullptr;

  auto it1 = scopeOccChosenConflicts.find(scopeOcc1);
  auto it2 = scopeOccChosenConflicts.find(scopeOcc2);
  auto it3 = scopeOccPairChosenConflicts.find({scopeOcc1, scopeOcc2});

  if (it1 != scopeOccChosenConflicts.end()) {
    opt1 = getReusableConflictPair(conflictPair, it1->second);
  }
  if (it2 != scopeOccChosenConflicts.end()) {
    opt2 = getReusableConflictPair(conflictPair, it2->second);
  }
  if (it3 != scopeOccPairChosenConflicts.end()) {
    opt3 = getReusableConflictPair(conflictPair, it3->second);
  }

  ConflictPair *reusableConflictPair = nullptr;
  for (auto *opt : {opt1, opt2, opt3}) {
    if (reusableConflictPair == nullptr ||
        reusableConflictPair->startIndex < opt->startIndex) {
      reusableConflictPair = opt;
    }
  }

  if (reusableConflictPair == nullptr) {
    return false;
  }

  assert(reusableConflictPair->startIndex < conflictPair->startIndex);
  assert(reusableConflictPair->endIndex <= conflictPair->endIndex);
  forgetSyncedPair(scopeOcc1->op, reusableConflictPair);
  reusableConflictPair->op1 = conflictPair->op1;
  reusableConflictPair->opSet = conflictPair->opSet;
  reusableConflictPair->startIndex = conflictPair->startIndex;
  memorizeSyncedPair(scopeOcc1->op, reusableConflictPair);

  if (!conflictPair->isUseless) {
    memorizeReusedSyncedPair(scopeOcc1->op, conflictPair, reusableConflictPair);
  }

  if (oldReusedConflictPair != nullptr) {
    assert(oldReusedConflictPair->op1 == reusableConflictPair->op1);
    assert(oldReusedConflictPair->op2 == reusableConflictPair->op2);
    assert(oldReusedConflictPair->opSet == reusableConflictPair->opSet);
    assert(oldReusedConflictPair->opWait == reusableConflictPair->opWait);
  }

  reusePairs[{setPipe, waitPipe}] -= 1;
  return true;
}

// Core handler that records a discovered conflict, chooses event ids (or
// converts to barrier-all), and records necessary bookkeeping structures.
void Solver::handleConflict(Occurrence *occ1, Occurrence *occ2,
                            hivm::PIPE setPipe, hivm::PIPE waitPipe,
                            bool isUseless, uint32_t eventIdNum,
                            LoopLikeOpInterface multibufferLoopPar) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);

  LLVM_DEBUG({
    llvm::dbgs() << "conflict found: eventIdNum(" << eventIdNum << ")\n";
    llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                 << occ1->endIndex << ' ' << rwOp1->str(0, false) << '\n';
    llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                 << occ2->endIndex << ' ' << rwOp2->str(0, false) << '\n';
  });

  Occurrence *parentLCALoopOcc{nullptr};
  OperationBase *parentLCALoopOp{nullptr};
  [[maybe_unused]] Scope *parentLCALoopScopeOp{nullptr};
  std::unique_ptr<ConflictPair> conflictPair;

  auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
  auto *normScopeOcc1 = setOcc->parentOcc;
  auto *normScopeOcc2 = waitOcc->parentOcc;
  assert(normScopeOcc1->op == normScopeOcc2->op);
  auto *normScopeOp = llvm::dyn_cast_if_present<Scope>(normScopeOcc1->op);
  assert(normScopeOp != nullptr);

  conflictPair = std::make_unique<ConflictPair>(
      rwOp1, rwOp2, setOcc->op, waitOcc->op, setPipe, waitPipe,
      setOcc->endIndex, waitOcc->startIndex);
  conflictPair->isUseless = isUseless;
  assert(conflictPair->startIndex <= conflictPair->endIndex);

  if (reuseConflictPair(conflictPair.get(), normScopeOcc1, normScopeOcc2)) {
    return;
  }

  if (conflictPair->isBarrier() &&
      conflictPair->setPipe == hivm::PIPE::PIPE_S) {
    return;
  }

  if (conflictPair->isBarrier() &&
      conflictPair->setPipe == hivm::PIPE::PIPE_M) {
    conflictPair->isUseless = isUseless = true;
  }

  if (isBackwardSync(occ1, occ2)) {
    auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
    assert(parOcc1 != nullptr && parOcc2 != nullptr);
    assert(parOcc1->parentOcc == parOcc2->parentOcc);
    assert(parOcc1->parentOcc != nullptr);
    parentLCALoopOcc = Occurrence::getParentloop(parOcc1);
    assert(parentLCALoopOcc != nullptr);
    assert(parentLCALoopOcc->op != nullptr);
    // llvm::dbgs() << parentLCALoopOcc->op->str(0, false) << '\n';
    parentLCALoopOp = llvm::dyn_cast_if_present<Loop>(parentLCALoopOcc->op);
    assert(parentLCALoopOp != nullptr);
    parentLCALoopScopeOp =
        llvm::dyn_cast_if_present<Scope>(parentLCALoopOcc->op);
    assert(parentLCALoopScopeOp != nullptr);
  }

  if (!conflictPair->isBarrier()) {
    if (isBackwardSync(setOcc, waitOcc)) {
      assert(parentLCALoopScopeOp != nullptr);
      conflictPair->isInnerBackward = true;
    }
  }

  if (!conflictPair->isBarrier()) {
    auto newParOccs = checkAndApplyMmadl0LoopOpt(conflictPair.get(), occ1, occ2,
                                                 setOcc, waitOcc);
    setOcc = newParOccs.first;
    waitOcc = newParOccs.second;
    conflictPair->updateSetWaitOps(setOcc, waitOcc);
  }

  if (auto unitFlagMode = checkUnitFlagPatterns(conflictPair.get(), occ1, occ2,
                                                parentLCALoopOcc)) {
    setOcc = occ1;
    waitOcc = occ2;
    conflictPair->updateSetWaitOps(setOcc, waitOcc);
    conflictPair->isUseless = true;
    conflictPair->replacedWithUnitFlag = true;
    if (!isUseless) {
      rwOp1->unitFlagModeAsSet = unitFlagMode->first;
      rwOp2->unitFlagModeAsWait = unitFlagMode->second;
      rwOp1->linkedUnitFlagOpAsSet = rwOp2;
      rwOp2->linkedUnitFlagOpAsWait = rwOp1;
    }
    assert(rwOp1->unitFlagModeAsSet == unitFlagMode->first);
    assert(rwOp2->unitFlagModeAsWait == unitFlagMode->second);
    assert(rwOp1->linkedUnitFlagOpAsSet == rwOp2);
    assert(rwOp2->linkedUnitFlagOpAsWait == rwOp1);
  }

  if (!conflictPair->isBarrier() && !conflictPair->replacedWithUnitFlag) {
    llvm::SmallVector<hivm::EVENT> eventIds;
    if (auto oldEventIds = getOldEventIdIfExists(normScopeOp, occ1, occ2,
                                                 conflictPair.get())) {
      eventIds = oldEventIds.value();
    } else {
      bool reversedPriority = false;
      if (conflictPair->isInnerBackward) {
        if (normScopeOcc1->parentOcc != nullptr) {
          if (OperationBase::getParentloop(occ1->op) ==
                  normScopeOcc1->parentOcc->op &&
              OperationBase::getParentloop(occ2->op) ==
                  normScopeOcc1->parentOcc->op) {
            reversedPriority = true;
          }
        }
        auto clonedConflictPair = conflictPair->clone(
            parentLCALoopOcc->startIndex, parentLCALoopOcc->endIndex);
        eventIds = getAnyAvailableMultiBufferEventIds(
            clonedConflictPair.get(), eventIdNum, reversedPriority);
      }
      if (eventIdNum < 2 || eventIds.size() < 2) {
        eventIds =
            getAnyAvailableEventId(conflictPair.get(), 1, reversedPriority);
      }
    }
    if (eventIds.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "will-be-converted-to-barrier-all "
                              << conflictPair->str() << '\n';);
      insertBarrierAllBefore(occ2, conflictPair->isUseless);
      barrierAllPairs.insert({conflictPair->setPipe, conflictPair->waitPipe});
      return;
    }
    if (!eventIds.empty()) {
      conflictPair->eventIds = eventIds;
    }
    if (multibufferLoopPar != nullptr) {
      if (eventIdNum > 1 && conflictPair->eventIds.size() > 1) {
        conflictPair->multibufferLoopPar = multibufferLoopPar;
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << conflictPair->str() << '\n';
    if (parentLCALoopOcc != nullptr) {
      llvm::dbgs() << parentLCALoopOcc->op->str(0, false) << '\n';
    }
  });

  // insert header/footer useless conflictPairs to reserve the eventIds.
  if (conflictPair->isInnerBackward && !conflictPair->eventIds.empty()) {
    auto *loopOpOcc1 = getFirstIterOcc(waitOcc, normScopeOcc1);
    auto *loopOpOcc2 = getLastIterOcc(setOcc, normScopeOcc2);
    assert(loopOpOcc1 != nullptr && loopOpOcc2 != nullptr);
    auto extraConflictPair1 = std::make_unique<ConflictPair>(
        nullptr, nullptr, nullptr, nullptr, setPipe, waitPipe,
        parentLCALoopOcc->startIndex, loopOpOcc1->startIndex);
    auto extraConflictPair2 = std::make_unique<ConflictPair>(
        nullptr, nullptr, nullptr, nullptr, setPipe, waitPipe,
        loopOpOcc2->endIndex, parentLCALoopOcc->endIndex);
    extraConflictPair1->isUseless = true;
    extraConflictPair2->isUseless = true;
    extraConflictPair1->couldNotRun = true;
    extraConflictPair2->couldNotRun = true;
    extraConflictPair1->eventIds = conflictPair->eventIds;
    extraConflictPair2->eventIds = conflictPair->eventIds;
    scopeOccChosenConflicts[parentLCALoopOcc].insert(extraConflictPair1.get());
    scopeOccChosenConflicts[parentLCALoopOcc].insert(extraConflictPair2.get());
    chosenConflictedPairs.push_back(std::move(extraConflictPair1));
    chosenConflictedPairs.push_back(std::move(extraConflictPair2));
  }

  // backward sync are not inserted as a conflictPairs, they are recorded in
  // backwardSyncEvents instead.
  if (conflictPair->isInnerBackward && !conflictPair->eventIds.empty()) {
    for (auto eventId : conflictPair->eventIds) {
      backwardSyncEvents[parentLCALoopOp]
                        [{conflictPair->setPipe, conflictPair->waitPipe}]
                            .insert(eventId);
    }
  }

  // insert useless conflictPair to cover the whole loop when having
  // multi-eventid backward sync to reserve the eventIds.
  if (conflictPair->isInnerBackward && !conflictPair->eventIds.empty()) {
    if (multibufferLoopPar != nullptr) {
      auto extraConflictPair3 = std::make_unique<ConflictPair>(
          nullptr, nullptr, nullptr, nullptr, setPipe, waitPipe,
          parentLCALoopOcc->startIndex, parentLCALoopOcc->endIndex);
      extraConflictPair3->isUseless = true;
      extraConflictPair3->couldNotRun = true;
      extraConflictPair3->eventIds = conflictPair->eventIds;
      assert(parentLCALoopOcc->parentOcc != nullptr);
      scopeOccChosenConflicts[parentLCALoopOcc->parentOcc].insert(
          extraConflictPair3.get());
      chosenConflictedPairs.push_back(std::move(extraConflictPair3));
    }
  }

  // backwardSyncEventsAfterMerge opt
  if (conflictPair->isInnerBackward && !conflictPair->eventIds.empty()) {
    if (backwardSyncEventsAfterMerge[parentLCALoopOp].contains(
            {conflictPair->setPipe, conflictPair->waitPipe})) {
      llvm::SmallVector<hivm::EVENT> eventIdsAfterMerge;
      auto &curBackwardSyncEventsAfterMerge =
          backwardSyncEventsAfterMerge[parentLCALoopOp][{
              conflictPair->setPipe, conflictPair->waitPipe}];
      for (auto eventId : conflictPair->eventIds) {
        if (curBackwardSyncEventsAfterMerge.contains(eventId)) {
          eventIdsAfterMerge.push_back(eventId);
        }
      }
      if (!eventIdsAfterMerge.empty()) {
        auto extraConflictPair4 = std::make_unique<ConflictPair>(
            nullptr, nullptr, nullptr, nullptr, setPipe, waitPipe,
            parentLCALoopOcc->startIndex, parentLCALoopOcc->endIndex);
        extraConflictPair4->isUseless = true;
        extraConflictPair4->couldNotRun = false; // notice this
        extraConflictPair4->eventIds = eventIdsAfterMerge;
        assert(parentLCALoopOcc->parentOcc != nullptr);
        scopeOccChosenConflicts[parentLCALoopOcc->parentOcc].insert(
            extraConflictPair4.get());
        chosenConflictedPairs.push_back(std::move(extraConflictPair4));
      }
    }
  }

  bool dontInsert = false;
  if (parentLCALoopOcc != nullptr && normScopeOcc1 != normScopeOcc2) {
    auto *parCond = OperationBase::getParentCondition(conflictPair->opSet);
    if (auto *conditionOp = llvm::dyn_cast_if_present<Condition>(parCond)) {
      if (parentLCALoopOcc->op->isProperAncestor(conditionOp)) {
        scopeOccPairChosenConflicts[{normScopeOcc1, normScopeOcc2}].insert(
            conflictPair.get());
        dontInsert = true;
      }
    }
  }
  if (!dontInsert) {
    assert(parentLCALoopOcc != nullptr || normScopeOcc1 == normScopeOcc2);
    scopeOccChosenConflicts[normScopeOcc1].insert(conflictPair.get());
    scopeOccChosenConflicts[normScopeOcc2].insert(conflictPair.get());
  }

  memorizeSyncedPair(normScopeOp, conflictPair.get());
  chosenConflictedPairs.push_back(std::move(conflictPair));
}

// Main processing loop that iterates processingOrders and attempts to discover
// and record conflicts.
void Solver::processOrders() {
  for (auto &[curOcc, start, end, reverseOrder, isUseless, skip] :
       processingOrders) {
    assert(start <= end + 1);
    if (start > end) {
      continue;
    }
    if (skip) {
      for (int i = start; i <= end; i++) {
        skipOcc.insert(syncIr[i].get());
      }
      continue;
    }
    if (checkSkippable(curOcc)) {
      continue;
    }
    assert(llvm::isa_and_present<RWOperation>(curOcc->op));
    int iStart, iEnd, iStep;
    if (!reverseOrder) {
      iStart = start;
      iEnd = end + 1;
      iStep = +1;
    } else {
      iStart = end;
      iEnd = start - 1;
      iStep = -1;
    }
    for (int i = iStart; i != iEnd; i += iStep) {
      if (llvm::isa_and_present<RWOperation>(syncIr[i]->op)) {
        Occurrence *occ1, *occ2;
        if (!reverseOrder) {
          occ1 = curOcc;
          occ2 = syncIr[i].get();
        } else {
          occ1 = syncIr[i].get();
          occ2 = curOcc;
        }
        if (checkSkippable(occ1)) {
          continue;
        }
        if (checkVisited(occ1, occ2)) {
          continue;
        }
        if (checkImpossibleOccPair(occ1, occ2)) {
          continue;
        }
        if (checkAlreadySynced(occ1, occ2)) {
          continue;
        }
        if (skipMMad1DecomposedLoopOpt(occ1, occ2)) {
          continue;
        }
        auto *rwOp1 = dyn_cast<RWOperation>(occ1->op);
        auto *rwOp2 = dyn_cast<RWOperation>(occ2->op);
        assert(rwOp1 != nullptr && rwOp2 != nullptr);
        if (checkAlreadySyncedWithUnitFlag(rwOp1, rwOp2)) {
          continue;
        }
        for (auto [setPipe, waitPipe] : checkMemoryConflicts(rwOp1, rwOp2)) {
          auto [eventIdNum, loopPar] =
              getEventIdNum(occ1, occ2, setPipe, waitPipe);
          if (checkGraphConflict(occ1, occ2, setPipe, waitPipe, eventIdNum)) {
            handleConflict(occ1, occ2, setPipe, waitPipe, isUseless, eventIdNum,
                           loopPar);
          }
        }
      }
    }
  }
}

// When barrier-all markers need to be chosen, insert them before all
// occurrences for the chosen op.
void Solver::pickAndInsertABarrierAll() {
  assert(!insertedBarrierAllBefore.empty());
  OperationBase *chosenOp = nullptr;
  for (auto &[op, vec] : insertedBarrierAllBefore) {
    if (vec.empty()) {
      continue;
    }
    if (chosenOp == nullptr || chosenOp->id > op->id) {
      chosenOp = op;
    }
  }
  assert(chosenOp != nullptr);
  for (auto *occ : opAllOccurrences[chosenOp]) {
    insertBarrierAllBefore(occ, false, /*isPersistent=*/true);
  }
  return;
}

// High-level solve orchestration with multiple passes and optional merging
// iterations.
void Solver::solve(int runNum) {
  LLVM_DEBUG(llvm::dbgs() << "runNum: " << runNum << '\n');
  processOrders();
  if (disableMultiEventIdForBarrierAllPairs) {
    if (!barrierAllPairs.empty()) {
      reset();
      insertedBarrierAllBefore.clear();
      processOrders();
    }
  }
  if (considerMergedBackwardSyncEventIds) {
    getBeforeAfterSyncMaps();
    backwardSyncEventsAfterMerge = backwardSyncEvents;
    reset();
    insertedBarrierAllBefore.clear();
    processOrders();
  }
  if (!insertedBarrierAllBefore.empty() && runNum < 99) {
    reset();
    pickAndInsertABarrierAll();
    insertedBarrierAllBefore.clear();
    backwardSyncEventsAfterMerge.clear();
    barrierAllPairs.clear();
    solve(runNum + 1);
  }
}
