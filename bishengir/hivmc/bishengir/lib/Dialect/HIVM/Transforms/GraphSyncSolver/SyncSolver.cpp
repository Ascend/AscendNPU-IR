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
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include <tuple>
#include <utility>

#define DEBUG_TYPE "hivm-gss-solver"

using namespace mlir;
using namespace hivm::syncsolver;

// Reset per-pass bookkeeping to start fresh.
void Solver::reset() {
  skipOcc.clear();
  syncedPairs.clear();
  processedOccPairs.clear();
  chosenConflictedPairs.clear();
  scopeOccChosenConflicts.clear();
  scopeOccPairChosenConflicts.clear();
  backwardSyncEvents.clear();
  replacedWithReusableSyncedPairs.clear();
  reusedPairs.clear();
  barrierAllPairs.clear();
  insertedBarrierAllBefore.clear();
  eventIdSolver.clear();
  resetUnitFlag();
}

void Solver::resetUnitFlag() {
  for (auto *rwOp : unitFlagFeaturedOps) {
    rwOp->mergedUnitFlagInfo.reset();
    for (auto *occ : opAllOccurrences[rwOp]) {
      occ->unitFlagInfo.reset();
    }
  }
}

// Helpers to find first/last iteration occurrences relative to parent
// occurrences.
Occurrence *Solver::getFirstIterOcc(Occurrence *occ, Occurrence *parOcc) {
  assert(occ != nullptr && parOcc != nullptr);
  if (parOcc->depth + 1 < occ->depth) {
    auto *newParOcc = getFirstIterOcc(
        occ->getNthParent(occ->depth - parOcc->depth - 1), parOcc);
    return getFirstIterOcc(occ, newParOcc);
  }
  auto *it =
      std::find_if(occChildrenMem[parOcc].begin(), occChildrenMem[parOcc].end(),
                   [occ](Occurrence *curOcc) { return occ->op == curOcc->op; });
  assert(it != occChildrenMem[parOcc].end());
  return *it;
}

Occurrence *Solver::getLastIterOcc(Occurrence *occ, Occurrence *parOcc) {
  assert(occ != nullptr && parOcc != nullptr);
  if (parOcc->depth + 1 < occ->depth) {
    auto *newParOcc = getLastIterOcc(
        occ->getNthParent(occ->depth - parOcc->depth - 1), parOcc);
    return getLastIterOcc(occ, newParOcc);
  }
  auto it = std::find_if(
      occChildrenMem[parOcc].rbegin(), occChildrenMem[parOcc].rend(),
      [occ](Occurrence *curOcc) { return occ->op == curOcc->op; });
  assert(it != occChildrenMem[parOcc].rend());
  return *it;
}

bool Solver::checkSkipCrossCorePair(Occurrence *occ1, Occurrence *occ2) {
  if (!isCrossCoreMode()) {
    return false;
  }
  auto *rwOp1 = dyn_cast<RWOperation>(occ1->op);
  auto *rwOp2 = dyn_cast<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  assert(rwOp1->coreType != hivm::TCoreType::CUBE_OR_VECTOR);
  assert(rwOp2->coreType != hivm::TCoreType::CUBE_OR_VECTOR);
  if (rwOp1->coreType == rwOp2->coreType) {
    return true;
  }
  if (rwOp1->coreType == hivm::TCoreType::CUBE_AND_VECTOR) {
    return true;
  }
  return false;
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
bool Solver::checkAlreadySyncedWithUnitFlag(Occurrence *occ1,
                                            Occurrence *occ2) {
  if (!enableUnitFlagFeature) {
    return false;
  }
  assert(occ1 != nullptr && occ2 != nullptr);
  if (!occ1->hasUnitFlagFeat || !occ2->hasUnitFlagFeat) {
    return false;
  }
  llvm::DenseSet<Occurrence *> visited;
  Occurrence *curOcc = occ1->unitFlagInfo.linkedElementAsSet;
  while (curOcc != nullptr) {
    auto [it, inserted] = visited.insert(curOcc);
    if (inserted) {
      break;
    }
    if (curOcc == occ2) {
      return true;
    }
    curOcc = curOcc->unitFlagInfo.linkedElementAsSet;
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
      int64_t baseAddr1 =
          static_cast<int64_t>(cast<IntegerAttr>(constOp1.getValue()).getInt());
      int64_t baseAddr2 =
          static_cast<int64_t>(cast<IntegerAttr>(constOp2.getValue()).getInt());
      int64_t l1 = baseAddr1;
      int64_t r1 = baseAddr1 + std::max((uint32_t)1, bufferSize1.value());
      int64_t l2 = baseAddr2;
      int64_t r2 = baseAddr2 + std::max((uint32_t)1, bufferSize2.value());
      // !(r2 <= l1 || r1 <= l2)
      if (r2 > l1 && r1 > l2) {
        return true;
      }
    }
  }
  return false;
}

bool Solver::checkAllocWorkSpaceMemConflict(
    bishengir::memref_ext::AllocWorkspaceOp allocWorkSpaceOp1,
    bishengir::memref_ext::AllocWorkspaceOp allocWorkSpaceOp2) {
  auto bufferSize1 = GetBufferSize(allocWorkSpaceOp1.getResult());
  auto bufferSize2 = GetBufferSize(allocWorkSpaceOp2.getResult());
  assert(bufferSize1.has_value() && bufferSize2.has_value());
  for (auto addr1 : allocWorkSpaceOp1.getOffset()) {
    for (auto addr2 : allocWorkSpaceOp2.getOffset()) {
      auto constOp1 =
          llvm::dyn_cast_if_present<arith::ConstantOp>(addr1.getDefiningOp());
      auto constOp2 =
          llvm::dyn_cast_if_present<arith::ConstantOp>(addr2.getDefiningOp());
      if (constOp1 == nullptr || constOp2 == nullptr) {
        return true;
      }
      int64_t baseAddr1 =
          static_cast<int64_t>(cast<IntegerAttr>(constOp1.getValue()).getInt());
      int64_t baseAddr2 =
          static_cast<int64_t>(cast<IntegerAttr>(constOp2.getValue()).getInt());
      int64_t l1 = baseAddr1;
      int64_t r1 = baseAddr1 + std::max((uint32_t)1, bufferSize1.value());
      int64_t l2 = baseAddr2;
      int64_t r2 = baseAddr2 + std::max((uint32_t)1, bufferSize2.value());
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
      auto pointerCastOp1 =
          dyn_cast_if_present<hivm::PointerCastOp>(val1.getDefiningOp());
      auto pointerCastOp2 =
          dyn_cast_if_present<hivm::PointerCastOp>(val2.getDefiningOp());
      if (pointerCastOp1 && pointerCastOp2) {
        if (checkPointerCastMemConflict(pointerCastOp1, pointerCastOp2)) {
          return true;
        }
      }
      auto allocWorkSpaceOp1 =
          dyn_cast_if_present<bishengir::memref_ext::AllocWorkspaceOp>(
              val1.getDefiningOp());
      auto allocWorkSpaceOp2 =
          dyn_cast_if_present<bishengir::memref_ext::AllocWorkspaceOp>(
              val2.getDefiningOp());
      if (allocWorkSpaceOp1 && allocWorkSpaceOp2) {
        if (checkAllocWorkSpaceMemConflict(allocWorkSpaceOp1,
                                           allocWorkSpaceOp2)) {
          return true;
        }
      }
    }
  }
  return false;
}

// High-level wrapper computing pipe pairs that represent memory conflicts
// between two RW ops.
llvm::SmallVector<std::tuple<CorePipeInfo, CorePipeInfo>>
Solver::checkMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  auto [it, inserted] = checkMemoryConflictsMem.insert({{rwOp1, rwOp2}, {}});
  if (!inserted) {
    return it->second;
  }
  auto coreSrc = rwOp1->coreType;
  auto coreDst = rwOp2->coreType;
  if (isCrossCoreMode()) {
    if (coreDst == hivm::TCoreType::CUBE_AND_VECTOR) {
      coreDst = (coreSrc == hivm::TCoreType::VECTOR) ? hivm::TCoreType::CUBE
                                                     : hivm::TCoreType::VECTOR;
    }
    assert(coreSrc == hivm::TCoreType::VECTOR ||
           coreSrc == hivm::TCoreType::CUBE);
    assert(coreDst == hivm::TCoreType::VECTOR ||
           coreDst == hivm::TCoreType::CUBE);
  }
  llvm::SetVector<std::tuple<CorePipeInfo, CorePipeInfo>> collectedConflictsSet;
  if (checkRWMemoryConflicts(rwOp1->readMemVals, rwOp2->writeMemVals)) {
    collectedConflictsSet.insert({CorePipeInfo(coreSrc, rwOp1->pipeRead),
                                  CorePipeInfo(coreDst, rwOp2->pipeWrite)});
  }
  if (checkRWMemoryConflicts(rwOp1->writeMemVals, rwOp2->readMemVals)) {
    collectedConflictsSet.insert({CorePipeInfo(coreSrc, rwOp1->pipeWrite),
                                  CorePipeInfo(coreDst, rwOp2->pipeRead)});
  }
  if (checkRWMemoryConflicts(rwOp1->writeMemVals, rwOp2->writeMemVals)) {
    collectedConflictsSet.insert({CorePipeInfo(coreSrc, rwOp1->pipeWrite),
                                  CorePipeInfo(coreDst, rwOp2->pipeWrite)});
  }
  llvm::SmallVector<std::tuple<CorePipeInfo, CorePipeInfo>> collectedConflicts(
      collectedConflictsSet.begin(), collectedConflictsSet.end());
  return it->second = collectedConflicts;
}

bool Solver::checkMemoryConflictBetweenOccExclusive(Occurrence *occ1,
                                                    Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  for (int i = occ1->syncIrEndIndex; i < occ2->syncIrIndex; i++) {
    if (auto *otherOp = dyn_cast<RWOperation>(syncIr[i]->op)) {
      if (!checkMemoryConflicts(rwOp1, otherOp).empty()) {
        return true;
      }
      if (!checkMemoryConflicts(rwOp2, otherOp).empty()) {
        return true;
      }
    }
  }
  return false;
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
  auto sz1 = static_cast<int>(addrs1.size());
  auto sz2 = static_cast<int>(addrs2.size());
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
        int64_t baseAddr1 = static_cast<int64_t>(
            cast<IntegerAttr>(constOp1.getValue()).getInt());
        int64_t baseAddr2 = static_cast<int64_t>(
            cast<IntegerAttr>(constOp2.getValue()).getInt());
        int64_t l1 = baseAddr1;
        int64_t r1 = baseAddr1 + std::max((uint32_t)1, bufferSize1.value());
        int64_t l2 = baseAddr2;
        int64_t r2 = baseAddr2 + std::max((uint32_t)1, bufferSize2.value());
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
    bishengir::memref_ext::AllocWorkspaceOp allocWorkSpaceOp1,
    bishengir::memref_ext::AllocWorkspaceOp allocWorkSpaceOp2) {
  auto loopPar1 = allocWorkSpaceOp1->getParentOfType<LoopLikeOpInterface>();
  auto loopPar2 = allocWorkSpaceOp2->getParentOfType<LoopLikeOpInterface>();
  if (loopPar1 == nullptr || loopPar2 == nullptr) {
    return {};
  }
  if (loopPar1 != loopPar2) {
    return {};
  }
  auto bufferSize1 = GetBufferSize(allocWorkSpaceOp1.getResult());
  auto bufferSize2 = GetBufferSize(allocWorkSpaceOp2.getResult());
  assert(bufferSize1.has_value() && bufferSize2.has_value());
  auto addrs1 = allocWorkSpaceOp1.getOffset();
  auto addrs2 = allocWorkSpaceOp2.getOffset();
  int sz1 = static_cast<int>(addrs1.size());
  int sz2 = static_cast<int>(addrs2.size());
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
        int64_t baseAddr1 = static_cast<int64_t>(
            cast<IntegerAttr>(constOp1.getValue()).getInt());
        int64_t baseAddr2 = static_cast<int64_t>(
            cast<IntegerAttr>(constOp2.getValue()).getInt());
        int64_t l1 = baseAddr1;
        int64_t r1 = baseAddr1 + std::max((uint32_t)1, bufferSize1.value());
        int64_t l2 = baseAddr2;
        int64_t r2 = baseAddr2 + std::max((uint32_t)1, bufferSize2.value());
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
      auto pointerCastOp1 =
          dyn_cast_if_present<hivm::PointerCastOp>(val1.getDefiningOp());
      auto pointerCastOp2 =
          dyn_cast_if_present<hivm::PointerCastOp>(val2.getDefiningOp());
      if (pointerCastOp1 && pointerCastOp2) {
        if (checkPointerCastMemConflict(pointerCastOp1, pointerCastOp2)) {
          auto curLoopParOpt =
              checkDoubleMultiBufferEventId(pointerCastOp1, pointerCastOp2);
          if (!curLoopParOpt.has_value() ||
              (loopPar != nullptr && loopPar != curLoopParOpt.value())) {
            return {};
          }
          loopPar = curLoopParOpt.value();
        }
        continue;
      }
      auto allocWorkSpaceOp1 =
          dyn_cast_if_present<bishengir::memref_ext::AllocWorkspaceOp>(
              val1.getDefiningOp());
      auto allocWorkSpaceOp2 =
          dyn_cast_if_present<bishengir::memref_ext::AllocWorkspaceOp>(
              val2.getDefiningOp());
      if (allocWorkSpaceOp1 && allocWorkSpaceOp2) {
        if (checkAllocWorkSpaceMemConflict(allocWorkSpaceOp1,
                                           allocWorkSpaceOp2)) {
          auto curLoopParOpt = checkDoubleMultiBufferEventId(allocWorkSpaceOp1,
                                                             allocWorkSpaceOp2);
          if (!curLoopParOpt.has_value() ||
              (loopPar != nullptr && loopPar != curLoopParOpt.value())) {
            return {};
          }
          loopPar = curLoopParOpt.value();
        }
        continue;
      }
      if (val1 == val2) {
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

std::optional<int64_t> Solver::checkCVMultiBufferEventId(RWOperation *rwOp1,
                                                         RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  assert(rwOp1->op != nullptr && rwOp2->op != nullptr);
  auto nowParentLoop = rwOp1->op->getParentOfType<LoopLikeOpInterface>();
  auto frontParentLoop = rwOp2->op->getParentOfType<LoopLikeOpInterface>();
  if (!nowParentLoop.getOperation() || !frontParentLoop.getOperation()) {
    return {};
  }
  auto getBlockSyncOpEventIdNum =
      [](LoopLikeOpInterface loopOp1,
         LoopLikeOpInterface loopOp2) -> std::optional<int> {
    auto multibufferAttr1 = loopOp1.getOperation()->getAttrOfType<IntegerAttr>(
        kMultibufferUnrollAttrName);
    auto multibufferAttr2 = loopOp2.getOperation()->getAttrOfType<IntegerAttr>(
        kMultibufferUnrollAttrName);
    if (multibufferAttr1 && multibufferAttr2) {
      assert(multibufferAttr1.getInt() == multibufferAttr2.getInt());
      return multibufferAttr2.getInt();
    }
    return {};
  };
  if (auto eventIdNumOpt =
          getBlockSyncOpEventIdNum(nowParentLoop, frontParentLoop);
      eventIdNumOpt.has_value()) {
    return eventIdNumOpt.value();
  }
  auto nowGrandParentLoop =
      nowParentLoop->getParentOfType<LoopLikeOpInterface>();
  auto frontGrandParentLoop =
      frontParentLoop->getParentOfType<LoopLikeOpInterface>();
  if (nowGrandParentLoop.getOperation()) {
    if (auto eventIdNumOpt =
            getBlockSyncOpEventIdNum(nowGrandParentLoop, frontParentLoop);
        eventIdNumOpt.has_value()) {
      return eventIdNumOpt.value();
    }
  }
  if (frontGrandParentLoop.getOperation()) {
    if (auto eventIdNumOpt =
            getBlockSyncOpEventIdNum(nowParentLoop, frontGrandParentLoop);
        eventIdNumOpt.has_value()) {
      return eventIdNumOpt.value();
    }
  }
  if (nowGrandParentLoop.getOperation() &&
      frontGrandParentLoop.getOperation()) {
    if (auto eventIdNumOpt =
            getBlockSyncOpEventIdNum(nowGrandParentLoop, frontGrandParentLoop);
        eventIdNumOpt.has_value()) {
      return eventIdNumOpt.value();
    }
  }
  return {};
}

// Determine required event id count and optional multibuffer loop parent for
// occurrences.
std::pair<int64_t, LoopLikeOpInterface>
Solver::getEventIdNum(Occurrence *occ1, Occurrence *occ2,
                      CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst) {
  assert(occ1 != nullptr && occ2 != nullptr);
  assert(occ1->op != nullptr && occ2->op != nullptr);
  std::pair<int64_t, LoopLikeOpInterface> singleEventId = {1, nullptr};
  if (disabledMultiEventIdPairs.contains({corePipeSrc, corePipeDst})) {
    return singleEventId;
  }
  if (!isBackwardSync(occ1, occ2)) {
    return singleEventId;
  }
  if (!checkAllParentLoopsAreForLoops(occ1->op->op) ||
      !checkAllParentLoopsAreForLoops(occ2->op->op)) {
    return singleEventId;
  }
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  assert(!checkMemoryConflicts(rwOp1, rwOp2).empty());
  if (isCrossCoreMode()) {
    if (auto eventIdNumOpt = checkCVMultiBufferEventId(rwOp1, rwOp2);
        eventIdNumOpt.has_value()) {
      return {eventIdNumOpt.value(), nullptr};
    }
  }
  auto loopParOpt = checkDoubleMultiBufferEventId(rwOp1, rwOp2);
  if (!loopParOpt.has_value()) {
    return singleEventId;
  }
  auto loopPar = loopParOpt.value();
  assert(loopPar != nullptr);
  auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
  if (setOcc->getParentWithOp(loopPar, /*assertExists=*/false) == nullptr ||
      waitOcc->getParentWithOp(loopPar, /*assertExists=*/false) == nullptr) {
    return singleEventId;
  }
  return {2, loopPar};
}

// Graph-based check to determine if adding a sync between occ1 and occ2 would
// block progress. Uses GraphSolver (Dijkstra) to estimate minimal reachable
// index.
bool Solver::checkGraphConflict(
    Occurrence *occ1, Occurrence *occ2, CorePipeInfo corePipeSrc,
    CorePipeInfo corePipeDst, int64_t eventIdNum, std::optional<int> startIndex,
    std::optional<int> endIndex,
    const llvm::SmallVector<ConflictPair *> &extraConflictPairs,
    const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (!startIndex.has_value()) {
    startIndex = occ1->endIndex;
  }
  if (!endIndex.has_value()) {
    endIndex = occ2->startIndex;
  }
  GraphSolver graphSolver;
  llvm::DenseSet<ConflictPair *> visited;
  auto handleConflictPair = [&](ConflictPair *conflictPair) {
    if (conflictPair->couldNotRun) {
      return;
    }
    if (conflictPair->replacedWithUnitFlag) {
      if (conflictPair->setCorePipeInfo.pipe == corePipeSrc.pipe ||
          conflictPair->waitCorePipeInfo.pipe == corePipeDst.pipe) {
        return;
      }
    }
    if (conflictPair->endIndex < startIndex.value() ||
        conflictPair->startIndex > endIndex.value()) {
      return;
    }
    if (conflictPair->isInnerBackward) {
      if (conflictPair->eventIdNode != nullptr &&
          conflictPair->eventIdNode->eventIdNum > eventIdNum) {
        return;
      }
    }
    if (llvm::find(ignoreConflictPairs, conflictPair) !=
        ignoreConflictPairs.end()) {
      return;
    }
    auto [it, inserted] = visited.insert(conflictPair);
    if (!inserted) {
      return;
    }
    DEBUG_WITH_TYPE("gss-sync-solver-check-graph-conflict", {
      llvm::dbgs() << "add-conflict-pair: " << conflictPair->str() << '\n';
    });
    graphSolver.addConflictPair(conflictPair);
  };

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
  for (auto *conflictPair : extraConflictPairs) {
    handleConflictPair(conflictPair);
  }
  auto mnDistance = graphSolver.runDijkstra(
      corePipeSrc, corePipeDst, startIndex.value(), endIndex.value());
  return !mnDistance.has_value() || mnDistance.value() > endIndex.value();
}

bool Solver::checkSyncOpsConflicts(ConflictPair *conflictPair1,
                                   ConflictPair *conflictPair2,
                                   int64_t eventIdNum) {
  if (conflictPair1->isBarrier() || conflictPair2->isBarrier()) {
    return false;
  }
  if (conflictPair1->startIndex > conflictPair2->startIndex) {
    std::swap(conflictPair1, conflictPair2);
  }
  if (conflictPair1->endIndex >= conflictPair2->endIndex) {
    return true;
  }
  bool result = false;
  if (conflictPair1->setCorePipeInfo != conflictPair2->setCorePipeInfo) {
    auto corePipeSrc = conflictPair1->setCorePipeInfo;
    auto corePipeDst = conflictPair2->setCorePipeInfo;
    Occurrence *occ1 = conflictPair1->occSet;
    Occurrence *occ2 = conflictPair2->occSet;
    auto startIndex = conflictPair1->startIndex + 1;
    auto endIndex = conflictPair2->startIndex;
    conflictPair1->startIndex += 1;
    assert(occ1 != nullptr && occ2 != nullptr);
    result = result || checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                          eventIdNum, startIndex, endIndex,
                                          {conflictPair1}, {conflictPair2});
    conflictPair1->startIndex -= 1;
  }
  if (conflictPair1->waitCorePipeInfo != conflictPair2->waitCorePipeInfo) {
    auto corePipeSrc = conflictPair1->waitCorePipeInfo;
    auto corePipeDst = conflictPair2->waitCorePipeInfo;
    Occurrence *occ1 = conflictPair1->occWait;
    Occurrence *occ2 = conflictPair2->occWait;
    auto startIndex = conflictPair1->endIndex;
    auto endIndex = conflictPair2->endIndex - 1;
    conflictPair2->endIndex -= 1;
    assert(occ1 != nullptr && occ2 != nullptr);
    result = result || checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                          eventIdNum, startIndex, endIndex,
                                          {conflictPair1}, {conflictPair2});
    conflictPair2->endIndex += 1;
  }
  LLVM_DEBUG({
    if (result) {
      llvm::dbgs() << "sync-ops-conflict-found: " << "\n";
      llvm::dbgs() << " " << conflictPair1->str() << '\n';
      llvm::dbgs() << " " << conflictPair2->str() << '\n';
    }
  });
  return result;
}

// Check whether two ConflictPair entries conflict in pipe and time ranges.
bool Solver::checkIntersect(ConflictPair *conflictPair1,
                            ConflictPair *conflictPair2, int64_t eventIdNum) {
  assert(conflictPair1 != nullptr && conflictPair2 != nullptr);
  if (conflictPair1 == conflictPair2) {
    return false;
  }
  if (isIntraCoreMode()) {
    if (conflictPair1->setCorePipeInfo != conflictPair2->setCorePipeInfo ||
        conflictPair1->waitCorePipeInfo != conflictPair2->waitCorePipeInfo) {
      return false;
    }
  }
  for (auto [l1, r1] : getRanges(conflictPair1)) {
    for (auto [l2, r2] : getRanges(conflictPair2)) {
      if (checkRangesIntersect(l1, r1 + 1, l2, r2 + 1)) {
        return true;
      }
    }
  }
  if (isCrossCoreMode() && this->isRegBasedArch) {
    if (checkSyncOpsConflicts(conflictPair1, conflictPair2, eventIdNum)) {
      return true;
    }
  }
  return false;
}

// Obtain available event ids while accounting for already chosen conflicts.
std::vector<ConflictPair *>
Solver::getIntersectingConflictPairs(ConflictPair *conflictPair,
                                     int64_t eventIdNum) {
  assert(conflictPair != nullptr);
  if (conflictPair->isBarrier()) {
    return {};
  }
  std::vector<ConflictPair *> intersectingConflictPairs;
  for (auto &curConflictPair : chosenConflictedPairs) {
    if (checkIntersect(conflictPair, curConflictPair.get(), eventIdNum)) {
      intersectingConflictPairs.push_back(curConflictPair.get());
    }
  }
  for (auto &curConflictPair : persistentChosenConflictedPairs) {
    if (checkIntersect(conflictPair, curConflictPair.get(), eventIdNum)) {
      intersectingConflictPairs.push_back(curConflictPair.get());
    }
  }
  return intersectingConflictPairs;
}

// Processed-pair tracking helpers.
bool Solver::checkVisited(Occurrence *occ1, Occurrence *occ2) {
  auto [it, inserted] = processedOccPairs.insert(std::make_pair(occ1, occ2));
  return !inserted;
}

bool Solver::checkSkippable(bool reverseOrder, Occurrence *occ) {
  return skipOcc[reverseOrder].contains(occ);
}

// Synced-pair memoization helpers.
EventIdNode *Solver::getOldEventIdNodeIfExists(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  auto oldConflictPairs = getMemorizedSyncedPairs(conflictPair);
  if (oldConflictPairs.empty()) {
    return {};
  }
  ConflictPair *oldConflictPair = *oldConflictPairs.begin();
  assert(oldConflictPair != nullptr && oldConflictPair->eventIdNode != nullptr);
  return oldConflictPair->eventIdNode;
}

llvm::DenseSet<ConflictPair *>
Solver::getMemorizedSyncedPairs(ConflictPair *conflictPair) {
  auto key = std::make_tuple(conflictPair->backwardSyncLoop, conflictPair->op1,
                             conflictPair->op2, conflictPair->setCorePipeInfo,
                             conflictPair->waitCorePipeInfo);
  return syncedPairs[key];
}

void Solver::memorizeSyncedPair(ConflictPair *conflictPair) {
  auto key = std::make_tuple(conflictPair->backwardSyncLoop, conflictPair->op1,
                             conflictPair->op2, conflictPair->setCorePipeInfo,
                             conflictPair->waitCorePipeInfo);
  syncedPairs[key].insert(conflictPair);
  for (auto *oldConflictPair : syncedPairs[key]) {
    assert(oldConflictPair->eventIdNode == conflictPair->eventIdNode);
  }
}

void Solver::forgetSyncedPair(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  auto key = std::make_tuple(conflictPair->backwardSyncLoop, conflictPair->op1,
                             conflictPair->op2, conflictPair->setCorePipeInfo,
                             conflictPair->waitCorePipeInfo);
  syncedPairs[key].erase(conflictPair);
}

void Solver::memorizeReusedSyncedPair(ConflictPair *conflictPair,
                                      ConflictPair *reusedConflictPair) {
  assert(conflictPair != nullptr);
  replacedWithReusableSyncedPairs[{
      conflictPair->backwardSyncLoop, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo}] =
      reusedConflictPair;
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
  if (occ1->parentOcc != nullptr && occ1->parentOcc->parentOcc != nullptr &&
      occ1->parentOcc->parentOcc->parentOcc == parOcc1 &&
      llvm::isa_and_present<syncsolver::LoadL0AOp, syncsolver::LoadL0BOp>(
          occ1->op) &&
      llvm::isa_and_present<syncsolver::MmadL1LoopOp>(
          occ1->parentOcc->parentOcc->op)) {
    conflictPair->setOnLastIterOnly = true;
    return std::make_pair(occ1, parOcc2);
  }
  if (!conflictPair->isInnerBackward && occ2->parentOcc != nullptr &&
      occ2->parentOcc->parentOcc != nullptr &&
      occ2->parentOcc->parentOcc->parentOcc == parOcc2 &&
      llvm::isa_and_present<syncsolver::LoadL0AOp, syncsolver::LoadL0BOp>(
          occ2->op) &&
      llvm::isa_and_present<syncsolver::MmadL1LoopOp>(
          occ2->parentOcc->parentOcc->op)) {
    conflictPair->waitOnFirstIterOnly = true;
    return std::make_pair(parOcc1, occ2);
  }
  return std::make_pair(parOcc1, parOcc2);
}

std::optional<UnitFlagInfo>
Solver::checkUnitFlagPatterns(ConflictPair *conflictPair, Occurrence *occ1,
                              Occurrence *occ2, Occurrence *parentLCALoopOcc) {
  if (!enableUnitFlagFeature) {
    return {};
  }
  if (conflictPair->isBarrier()) {
    return {};
  }
  assert(occ1 != nullptr && occ1->op != nullptr);
  assert(occ2 != nullptr && occ2->op != nullptr);
  auto *rwOp1 = dyn_cast<RWOperation>(occ1->op);
  auto *rwOp2 = dyn_cast<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!rwOp1->hasUnitFlagFeat || !rwOp2->hasUnitFlagFeat) {
    return {};
  }
  if (!occ1->unitFlagInfo.disabledAsSet() ||
      !occ2->unitFlagInfo.disabledAsWait()) {
    return {};
  }
  scf::ForOp backwardSyncLoop;
  if (conflictPair->isInnerBackward) {
    assert(parentLCALoopOcc != nullptr);
    assert(parentLCALoopOcc->op != nullptr);
    assert(rwOp1->op != nullptr && rwOp2->op != nullptr);
    if (!(backwardSyncLoop = dyn_cast<scf::ForOp>(parentLCALoopOcc->op->op))) {
      return {};
    }
    if (rwOp1->op->getParentOp() != parentLCALoopOcc->op->op ||
        rwOp2->op->getParentOp() != parentLCALoopOcc->op->op) {
      return {};
    }
  }
  if (checkMemoryConflictBetweenOccExclusive(occ1, occ2)) {
    return {};
  }
  if (!occ1->unitFlagInfo.disabledAsSet() ||
      !occ2->unitFlagInfo.disabledAsWait()) {
    return {};
  }
  if (auto unitFlagInfo = checkUnitFlagSameBlockPattern(
          occ1->op->op, occ2->op->op, rwOp1->mergedUnitFlagInfo,
          rwOp2->mergedUnitFlagInfo, backwardSyncLoop)) {
    return std::optional<UnitFlagInfo>(unitFlagInfo);
  }
  if (auto unitFlagInfo = checkUnitFlagOpLoopOpPattern(
          occ1->op->op, occ2->op->op, rwOp1->mergedUnitFlagInfo,
          rwOp2->mergedUnitFlagInfo, backwardSyncLoop)) {
    return std::optional<UnitFlagInfo>(unitFlagInfo);
  }
  return {};
}

Occurrence *Solver::getBeforePlaceHolderOcc(Occurrence *occ) {
  assert(occ != nullptr);
  assert(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrIndex - 1;
  assert(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  assert(placeHolderOp != nullptr);
  assert(placeHolderOp->beforeOp == occ->op);
  return placeHolderOcc;
}

Occurrence *Solver::getAfterPlaceHolderOcc(Occurrence *occ) {
  assert(occ != nullptr);
  assert(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrEndIndex;
  assert(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  assert(placeHolderOp != nullptr);
  assert(placeHolderOp->afterOp == occ->op);
  return placeHolderOcc;
}

Occurrence *Solver::getScopeBeginPlaceHolderOcc(Occurrence *occ) {
  assert(occ != nullptr);
  assert(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrIndex + 1;
  assert(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  assert(placeHolderOp != nullptr);
  assert(placeHolderOp->scopeBegin == occ->op);
  return placeHolderOcc;
}

Occurrence *Solver::getScopeEndPlaceHolderOcc(Occurrence *occ) {
  assert(occ != nullptr);
  assert(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrEndIndex - 1;
  assert(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  assert(placeHolderOp != nullptr);
  assert(placeHolderOp->scopeEnd == occ->op);
  return placeHolderOcc;
}

std::pair<Occurrence *, Occurrence *>
Solver::getPlaceHolderSetWaitOcc(Occurrence *setOcc, Occurrence *waitOcc) {
  assert(setOcc != nullptr && waitOcc != nullptr);
  if (llvm::isa_and_present<Loop>(setOcc->op)) {
    setOcc = getAfterPlaceHolderOcc(setOcc);
  }
  if (llvm::isa_and_present<Loop>(waitOcc->op)) {
    waitOcc = getBeforePlaceHolderOcc(waitOcc);
  }
  return std::make_pair(setOcc, waitOcc);
}

void Solver::inplaceFixPlaceHolderSetWaitOcc(Occurrence *&setOcc,
                                             Occurrence *&waitOcc) {
  auto [newSetOcc, newWaitOcc] = getPlaceHolderSetWaitOcc(setOcc, waitOcc);
  assert(newSetOcc != nullptr && newWaitOcc != nullptr);
  setOcc = newSetOcc;
  waitOcc = newWaitOcc;
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
  assert(setOcc != nullptr && waitOcc != nullptr &&
         setOcc->parentOcc != nullptr && waitOcc->parentOcc != nullptr);
  if (setOcc->op != waitOcc->op) {
    if (auto *parLoopOp =
            llvm::dyn_cast_if_present<Loop>(setOcc->parentOcc->op)) {
      if (parLoopOp->body.size() > 1 && !isa<PlaceHolder>(waitOcc->op)) {
        auto *placeHolderOcc = getScopeEndPlaceHolderOcc(setOcc);
        return getSetWaitOcc(occ1, placeHolderOcc);
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
  if (isCrossCoreMode()) {
    assert(setOcc->op != nullptr && waitOcc->op != nullptr);
    auto forOp1 = llvm::dyn_cast_if_present<scf::ForOp>(setOcc->op->op);
    auto forOp2 = llvm::dyn_cast_if_present<scf::ForOp>(waitOcc->op->op);
    if (forOp1 != nullptr && forOp2 != nullptr) {
      if (forOp1->hasAttr(kMultibufferUnrollAttrName) &&
          forOp2->hasAttr(kMultibufferUnrollAttrName)) {
        setOcc = occ1->getNthParent(occ1->depth - setOcc->depth - 2);
        waitOcc = occ2->getNthParent(occ2->depth - waitOcc->depth - 2);
      }
    }
  }
  inplaceFixPlaceHolderSetWaitOcc(setOcc, waitOcc);
  return {setOcc, waitOcc};
}
std::optional<std::pair<Occurrence *, Occurrence *>>
Solver::getUnlikelyCondSetWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (isCrossCoreMode() && isBackwardSync(occ1, occ2)) {
    return {};
  }
  if (auto *unlikelyParCondOcc1 =
          Occurrence::getUnlikelyParentCondition(occ1)) {
    if (!unlikelyParCondOcc1->isProperAncestor(occ2)) {
      auto *parentLoopOcc = Occurrence::getParentloop(unlikelyParCondOcc1);
      if (parentLoopOcc == nullptr || parentLoopOcc->isProperAncestor(occ2)) {
        auto *placeHolderOcc = getScopeEndPlaceHolderOcc(
            occ1->getNthParent(occ1->depth - unlikelyParCondOcc1->depth - 1));
        return getSetWaitOcc(occ1, placeHolderOcc);
      }
    }
  }
  if (auto *unlikelyParCondOcc2 =
          Occurrence::getUnlikelyParentCondition(occ2)) {
    if (!unlikelyParCondOcc2->isProperAncestor(occ1)) {
      auto *parentLoopOcc = Occurrence::getParentloop(unlikelyParCondOcc2);
      if (parentLoopOcc == nullptr || parentLoopOcc->isProperAncestor(occ1)) {
        auto *placeHolderOcc = getScopeBeginPlaceHolderOcc(
            occ2->getNthParent(occ2->depth - unlikelyParCondOcc2->depth - 1));
        return getSetWaitOcc(placeHolderOcc, occ2);
      }
    }
  }
  return {};
}

void Solver::insertBarrierAllBeforeOcc(Occurrence *occ, bool isUseless,
                                       bool isPersistent) {
  assert(occ != nullptr);
  auto *rwOp = llvm::dyn_cast_if_present<RWOperation>(occ->op);
  assert(rwOp != nullptr);
  auto conflictPair = std::make_unique<ConflictPair>(
      nullptr, nullptr, rwOp, rwOp, occ, occ,
      CorePipeInfo(rwOp->coreType, hivm::PIPE::PIPE_ALL),
      CorePipeInfo(rwOp->coreType, hivm::PIPE::PIPE_ALL), occ->startIndex,
      occ->startIndex);
  conflictPair->isUseless = isUseless;
  auto *normScopeOcc = occ->parentOcc;
  assert(normScopeOcc != nullptr);
  LLVM_DEBUG(llvm::dbgs() << (isPersistent ? "is-persistent " : "")
                          << occ->op->str(0, false) << ' '
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

void Solver::insertBarrierAllBeforeOp(OperationBase *op, bool isUseless,
                                      bool isPersistent) {
  assert(op != nullptr);
  for (auto *occ : opAllOccurrences[op]) {
    insertBarrierAllBeforeOcc(occ, isUseless, isPersistent);
    isUseless = true;
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
  insertBarrierAllBeforeOp(chosenOp, /*isUseless=*/false,
                           /*isPersistent=*/true);
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

bool Solver::reuseCmp(ConflictPair *conflictPair1,
                      ConflictPair *conflictPair2) {
  assert(conflictPair1 != nullptr && conflictPair2 != nullptr);
  assert(conflictPair1->op1 != nullptr && conflictPair1->op2 != nullptr);
  assert(conflictPair2->op1 != nullptr && conflictPair2->op2 != nullptr);
  if (conflictPair1->startIndex != conflictPair2->startIndex) {
    return conflictPair1->startIndex < conflictPair2->startIndex;
  }
  if (conflictPair1->endIndex != conflictPair2->endIndex) {
    return conflictPair1->endIndex > conflictPair2->endIndex;
  }
  if (conflictPair1->op1 != conflictPair2->op1) {
    return conflictPair1->op1->id > conflictPair2->op1->id;
  }
  if (conflictPair1->op2 != conflictPair2->op2) {
    return conflictPair1->op2->id > conflictPair2->op2->id;
  }
  return false;
}

ConflictPair *Solver::getReusableConflictPair(
    ConflictPair *conflictPair,
    const llvm::DenseSet<ConflictPair *> &conflictPairsSet) {
  assert(conflictPair != nullptr);
  ConflictPair *ret = nullptr;
  for (auto *curConflictPair : conflictPairsSet) {
    if (curConflictPair->isBarrier() || curConflictPair->dontReuse) {
      continue;
    }
    if (!checkIntersect(conflictPair, curConflictPair)) {
      continue;
    }
    if (curConflictPair->startIndex >= conflictPair->startIndex) {
      continue;
    }
    assert(conflictPair->startIndex <= curConflictPair->endIndex);
    assert(curConflictPair->endIndex <= conflictPair->endIndex);
    if (ret == nullptr || reuseCmp(ret, curConflictPair)) {
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
  if (scopeOcc1->op != scopeOcc2->op) {
    return false;
  }
  if (!barrierAllPairs.empty()) {
    return false;
  }

  ConflictPair *oldReusedConflictPair = nullptr;
  if (conflictPair->isUseless) {
    auto it = replacedWithReusableSyncedPairs.find(
        {conflictPair->backwardSyncLoop, conflictPair->op1, conflictPair->op2,
         conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo});
    if (it != replacedWithReusableSyncedPairs.end()) {
      oldReusedConflictPair = it->second;
    }
  }

  if (!conflictPair->isUseless) {
    auto it = replacedWithReusableSyncedPairs.find(
        {conflictPair->backwardSyncLoop, conflictPair->op1, conflictPair->op2,
         conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo});
    assert(it == replacedWithReusableSyncedPairs.end());
  }

  if (conflictPair->isUseless && oldReusedConflictPair == nullptr) {
    return false;
  }

  auto corePipeSrc = conflictPair->setCorePipeInfo;
  auto corePipeDst = conflictPair->waitCorePipeInfo;

  if (oldReusedConflictPair == nullptr) {
    if (!reusePairs.contains({corePipeSrc, corePipeDst}) ||
        reusePairs[{corePipeSrc, corePipeDst}] <=
            reusedPairs[{corePipeSrc, corePipeDst}]) {
      return false;
    }
  }

  assert(reusePairs.contains(std::make_tuple(corePipeSrc, corePipeDst)));
  assert(reusePairs[std::make_tuple(corePipeSrc, corePipeDst)] >=
         reusedPairs[std::make_tuple(corePipeSrc, corePipeDst)]);

  ConflictPair *opt1 = nullptr;
  ConflictPair *opt2 = nullptr;
  ConflictPair *opt3 = nullptr;
  ConflictPair *opt4 = nullptr;
  ConflictPair *opt5 = nullptr;

  auto it1 = scopeOccChosenConflicts.find(scopeOcc1);
  auto it2 = scopeOccChosenConflicts.find(scopeOcc2);
  auto it3 = scopeOccPairChosenConflicts.find({scopeOcc1, scopeOcc2});
  auto it4 = persistentScopeOccChosenConflicts.find(scopeOcc1);
  auto it5 = persistentScopeOccChosenConflicts.find(scopeOcc2);

  if (it1 != scopeOccChosenConflicts.end()) {
    opt1 = getReusableConflictPair(conflictPair, it1->second);
  }
  if (it2 != scopeOccChosenConflicts.end()) {
    opt2 = getReusableConflictPair(conflictPair, it2->second);
  }
  if (it3 != scopeOccPairChosenConflicts.end()) {
    opt3 = getReusableConflictPair(conflictPair, it3->second);
  }
  if (it4 != persistentScopeOccChosenConflicts.end()) {
    opt4 = getReusableConflictPair(conflictPair, it4->second);
  }
  if (it5 != persistentScopeOccChosenConflicts.end()) {
    opt5 = getReusableConflictPair(conflictPair, it5->second);
  }

  ConflictPair *reusableConflictPair = nullptr;
  for (auto *opt : {opt1, opt2, opt3, opt4, opt5}) {
    if (opt != nullptr) {
      if (reusableConflictPair == nullptr ||
          reuseCmp(reusableConflictPair, opt)) {
        reusableConflictPair = opt;
      }
    }
  }

  if (reusableConflictPair == nullptr) {
    return false;
  }

  DEBUG_WITH_TYPE("gss-sync-solver-reuse", {
    llvm::dbgs() << "reuse: " << conflictPair->str() << '\n';
    llvm::dbgs() << "with: " << reusableConflictPair->str() << '\n';
  });

  assert(reusableConflictPair->startIndex < conflictPair->startIndex);
  assert(reusableConflictPair->endIndex <= conflictPair->endIndex);
  reusableConflictPair->opSet = conflictPair->opSet;
  reusableConflictPair->occSet = conflictPair->occSet;
  reusableConflictPair->startIndex = conflictPair->startIndex;

  if (!conflictPair->isUseless) {
    memorizeReusedSyncedPair(conflictPair, reusableConflictPair);
  }

  DEBUG_WITH_TYPE("gss-sync-solver-reuse", {
    if (oldReusedConflictPair != nullptr) {
      llvm::dbgs() << "old-reuse: " << oldReusedConflictPair->str() << '\n';
    }
  });

  if (oldReusedConflictPair != nullptr) {
    assert(oldReusedConflictPair->op1 == reusableConflictPair->op1);
    assert(oldReusedConflictPair->op2 == reusableConflictPair->op2);
    assert(oldReusedConflictPair->opWait == reusableConflictPair->opWait);
  }

  if (!conflictPair->isUseless) {
    reusedPairs[{corePipeSrc, corePipeDst}] += 1;
  }

  return true;
}

std::unique_ptr<EventIdSolver> &
Solver::getEventIdSolverRef(hivm::PIPE pipeSrc, hivm::PIPE pipeDst) {
  if (isCrossCoreMode()) {
    pipeSrc = hivm::PIPE::PIPE_UNASSIGNED;
    pipeDst = hivm::PIPE::PIPE_UNASSIGNED;
  }
  auto key = std::make_tuple(pipeSrc, pipeDst);
  if (!eventIdSolver.contains(key)) {
    int64_t eventIdNumMax =
        getHWAvailableEventIdNum(syncMode, pipeSrc, pipeDst);
    eventIdSolver[key] = std::make_unique<EventIdSolver>(eventIdNumMax);
  }
  return eventIdSolver[key];
}

// Core handler that records a discovered conflict, chooses event ids (or
// converts to barrier-all), and records necessary bookkeeping structures.
void Solver::handleConflict(Occurrence *occ1, Occurrence *occ2,
                            CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                            bool isUseless, int64_t eventIdNum,
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

  Occurrence *setOcc{nullptr};
  Occurrence *waitOcc{nullptr};
  Occurrence *parentLCALoopOcc{nullptr};
  Occurrence *parentLCALoopBeforePHOcc{nullptr};
  Occurrence *parentLCALoopAfterPHOcc{nullptr};
  Loop *parentLCALoopOp{nullptr};
  Scope *parentLCALoopScopeOp{nullptr};
  std::unique_ptr<ConflictPair> conflictPair;

  if (auto unlikelyOpt = getUnlikelyCondSetWaitOcc(occ1, occ2)) {
    auto [curSetOcc, curWaitOcc] = unlikelyOpt.value();
    setOcc = curSetOcc;
    waitOcc = curWaitOcc;
  } else {
    auto [curSetOcc, curWaitOcc] = getSetWaitOcc(occ1, occ2);
    setOcc = curSetOcc;
    waitOcc = curWaitOcc;
  }

  bool isBackwardSyncPair = isBackwardSync(setOcc, waitOcc);
  if (!isBackwardSyncPair) {
    eventIdNum = 1;
    multibufferLoopPar = nullptr;
  }

  auto [lcaSetOp, lcaWaitOp] =
      OperationBase::getLCAPair(setOcc->op, waitOcc->op);
  auto *normScopeOcc1 = setOcc->getParentWithOp(lcaSetOp->parentOp);
  auto *normScopeOcc2 = waitOcc->getParentWithOp(lcaWaitOp->parentOp);
  assert(normScopeOcc1->op == normScopeOcc2->op);

  conflictPair = std::make_unique<ConflictPair>(
      rwOp1, rwOp2, setOcc->op, waitOcc->op, setOcc, waitOcc, corePipeSrc,
      corePipeDst, setOcc->endIndex, waitOcc->startIndex);
  conflictPair->isUseless = isUseless;
  assert(conflictPair->startIndex <= conflictPair->endIndex);

  if (conflictPair->isBarrier() &&
      conflictPair->setCorePipeInfo.pipe == hivm::PIPE::PIPE_S) {
    return;
  }
  if (isRegBasedArch && conflictPair->isBarrier() &&
      conflictPair->setCorePipeInfo.pipe == hivm::PIPE::PIPE_V) {
    return;
  }
  if (conflictPair->isBarrier() &&
      conflictPair->setCorePipeInfo.pipe == hivm::PIPE::PIPE_M) {
    conflictPair->isUseless = isUseless = true;
  }

  if (isBackwardSyncPair) {
    auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
    assert(parOcc1 != nullptr && parOcc2 != nullptr);
    parentLCALoopOcc = Occurrence::getParentloop(parOcc1);
    assert(parentLCALoopOcc != nullptr);
    parentLCALoopScopeOp =
        llvm::dyn_cast_if_present<Scope>(parentLCALoopOcc->op);
    assert(parentLCALoopScopeOp != nullptr);
    parentLCALoopOp = llvm::dyn_cast_if_present<Loop>(parentLCALoopOcc->op);
    assert(parentLCALoopOp != nullptr);
    parentLCALoopBeforePHOcc = getBeforePlaceHolderOcc(parentLCALoopOcc);
    parentLCALoopAfterPHOcc = getAfterPlaceHolderOcc(parentLCALoopOcc);
    assert(parentLCALoopBeforePHOcc != nullptr ||
           parentLCALoopAfterPHOcc != nullptr);
    conflictPair->backwardSyncLoop = parentLCALoopOp;
  }

  if (!conflictPair->isBarrier()) {
    if (isBackwardSyncPair) {
      assert(parentLCALoopScopeOp != nullptr);
      conflictPair->isInnerBackward = true;
    }
  }

  if (!conflictPair->isBarrier()) {
    auto newParOccs = checkAndApplyMmadl0LoopOpt(conflictPair.get(), occ1, occ2,
                                                 setOcc, waitOcc);
    setOcc = newParOccs.first;
    waitOcc = newParOccs.second;
    conflictPair->updateSetWaitOccs(setOcc, waitOcc);
  }

  if (auto unitFlagInfoOpt = checkUnitFlagPatterns(conflictPair.get(), occ1,
                                                   occ2, parentLCALoopOcc)) {
    DEBUG_WITH_TYPE("gss-sync-solver-unit-flag", {
      llvm::dbgs() << "replaced-with-unit-flag: " << conflictPair->str()
                   << '\n';
    });
    auto unitFlagInfo = unitFlagInfoOpt.value();
    conflictPair->updateSetWaitOccs(occ1, occ2);
    conflictPair->isUseless = true;
    conflictPair->dontReuse = true;
    conflictPair->replacedWithUnitFlag = true;
    if (!isUseless) {
      occ1->unitFlagInfo.merge(unitFlagInfoOpt.value(), occ1, occ2,
                               /*asSet=*/true, /*asWait=*/false);
      occ2->unitFlagInfo.merge(unitFlagInfoOpt.value(), occ1, occ2,
                               /*asSet=*/false, /*asWait=*/true);
      rwOp1->mergedUnitFlagInfo.merge(unitFlagInfo, /*asSet=*/true,
                                      /*asWait=*/false);
      rwOp2->mergedUnitFlagInfo.merge(unitFlagInfo, /*asSet=*/false,
                                      /*asWait=*/true);
    }
  }

  auto &curEventIdSolver = getEventIdSolverRef(
      conflictPair->setCorePipeInfo.pipe, conflictPair->waitCorePipeInfo.pipe);
  curEventIdSolver->pushActionNone();

  auto checkColorable = [&]() -> bool {
    if (curEventIdSolver->isColorable()) {
      return true;
    }
    LLVM_DEBUG(llvm::dbgs() << "will-be-converted-to-barrier-all "
                            << conflictPair->str() << '\n';);
    insertBarrierAllBeforeOp(occ2->op, conflictPair->isUseless,
                             /*isPersistent=*/false);
    barrierAllPairs.insert({corePipeSrc, corePipeDst});
    curEventIdSolver->undoActions();
    return false;
  };

  if (!conflictPair->isBarrier() && !conflictPair->replacedWithUnitFlag) {
    if (auto *oldEventIdNode = getOldEventIdNodeIfExists(conflictPair.get())) {
      conflictPair->eventIdNode = oldEventIdNode;
      curEventIdSolver->insertConflictPair(oldEventIdNode, conflictPair.get());
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
      }
      conflictPair->eventIdNode = curEventIdSolver->createNode(
          conflictPair.get(), eventIdNum, reversedPriority);
    }
    auto intersectingConflictPairs =
        getIntersectingConflictPairs(conflictPair.get(), eventIdNum);
    curEventIdSolver->addConflicts(conflictPair.get(),
                                   intersectingConflictPairs);
    if (!checkColorable()) {
      return;
    }
    if (multibufferLoopPar != nullptr) {
      if (eventIdNum > 1) {
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

  auto insertExtraConflictPair = [&](Occurrence *setOcc, Occurrence *waitOcc,
                                     Occurrence *parentScope) -> bool {
    assert(setOcc != nullptr && waitOcc != nullptr && parentScope != nullptr);
    auto extraConflictPair = conflictPair->clone(setOcc, waitOcc);
    extraConflictPair->isUseless = true;
    extraConflictPair->dontReuse = true;
    extraConflictPair->couldNotRun = true;
    LLVM_DEBUG({
      llvm::dbgs() << "extra-conflict-pair: " << extraConflictPair->str()
                   << "\n";
    });
    curEventIdSolver->insertConflictPair(conflictPair->eventIdNode,
                                         extraConflictPair.get());
    auto intersectingConflictPairs =
        getIntersectingConflictPairs(extraConflictPair.get(), eventIdNum);
    curEventIdSolver->addConflicts(extraConflictPair.get(),
                                   intersectingConflictPairs);
    if (!checkColorable()) {
      return false;
    }
    scopeOccChosenConflicts[parentScope].insert(extraConflictPair.get());
    chosenConflictedPairs.push_back(std::move(extraConflictPair));
    return true;
  };

  if (conflictPair->isInnerBackward && conflictPair->eventIdNode != nullptr) {
    if (conflictPair->eventIdNode->eventIdNum > 1) {
      // insert useless conflictPair to cover the whole loop when having
      // multi-eventid backward sync to reserve the eventIds.
      if (!insertExtraConflictPair(parentLCALoopBeforePHOcc,
                                   parentLCALoopAfterPHOcc,
                                   parentLCALoopOcc->parentOcc)) {
        return;
      }
    } else {
      // insert header/footer useless conflictPairs to reserve the eventIds.
      auto *loopOpOcc1 = getFirstIterOcc(waitOcc, normScopeOcc1);
      auto *loopOpOcc2 = getLastIterOcc(setOcc, normScopeOcc2);
      if (!insertExtraConflictPair(parentLCALoopBeforePHOcc, loopOpOcc1,
                                   parentLCALoopOcc)) {
        return;
      }
      if (!insertExtraConflictPair(loopOpOcc2, parentLCALoopAfterPHOcc,
                                   parentLCALoopOcc)) {
        return;
      }
    }
  }

  bool dontInsert = false;
  if (isBackwardSyncPair && normScopeOcc1 != normScopeOcc2) {
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

  if (!conflictPair->replacedWithUnitFlag) {
    memorizeSyncedPair(conflictPair.get());
  }
  chosenConflictedPairs.push_back(std::move(conflictPair));
  curEventIdSolver->clearActionStack();
}

void Solver::calcAllEventIds() {
  for (auto &[pipes, eventIdSolver] : eventIdSolver) {
    assert(eventIdSolver != nullptr);
    eventIdSolver->calcEventIds(/*forceConsecutiveIds=*/true);
    assert(eventIdSolver->isColorable());
  }
}

void Solver::collectBackwardSyncEventIds() {
  LLVM_DEBUG(llvm::dbgs() << "collectBackwardSyncEventIds\n";);
  for (auto &conflictPair : chosenConflictedPairs) {
    if (!conflictPair->isUseless && conflictPair->isInnerBackward &&
        conflictPair->eventIdNode != nullptr) {
      LLVM_DEBUG(llvm::dbgs() << "  " << conflictPair->str() << "\n";);
      for (auto eventId : conflictPair->eventIdNode->getEventIds()) {
        backwardSyncEvents[conflictPair->backwardSyncLoop]
                          [{conflictPair->setCorePipeInfo,
                            conflictPair->waitCorePipeInfo}]
                              .insert(eventId);
      }
    }
  }
}

void Solver::resetAndBuildSetWaitOpIndex(const SyncMap &syncMapBefore,
                                         const SyncMap &syncMapAfter) {
  globalSetWaitIndex = 0;
  setWaitStartIndex.clear();
  setWaitEndIndex.clear();
  setWaitStartIndexInclusive.clear();
  setWaitEndIndexInclusive.clear();
  setWaitFlagOpsIndex.clear();
  collectSetWaitOpsIndexes(funcIr.get(), syncMapBefore, syncMapAfter);
}

std::set<std::pair<int64_t, SetWaitOp *>> &
Solver::getSetWaitOpsIndexRef(hivm::PIPE pipeSrc, hivm::PIPE pipeDst,
                              int64_t eventId) {
  auto key = std::make_tuple(pipeSrc, pipeDst, eventId);
  return setWaitFlagOpsIndex[key];
}

// Collect indices for all Set/Wait ops to facilitate merging decisions.
void Solver::collectSetWaitOpsIndexes(OperationBase *op,
                                      const SyncMap &syncMapBefore,
                                      const SyncMap &syncMapAfter) {
  assert(op != nullptr);
  setWaitStartIndexInclusive[op] = globalSetWaitIndex++;
  if (syncMapBefore.count(op)) {
    auto *it = syncMapBefore.find(op);
    assert(it != syncMapBefore.end());
    for (auto &syncOp : it->second) {
      if (auto *setWaitOp = dyn_cast<SetWaitOp>(syncOp.get())) {
        for (auto eventId : setWaitOp->eventIds) {
          auto &index = getSetWaitOpsIndexRef(setWaitOp->pipeSrc,
                                              setWaitOp->pipeDst, eventId);
          index.insert({globalSetWaitIndex++, setWaitOp});
        }
      }
    }
  }
  setWaitStartIndex[op] = globalSetWaitIndex++;
  if (auto *scopeOp = dyn_cast<Scope>(op)) {
    for (auto &childOp : scopeOp->body) {
      collectSetWaitOpsIndexes(childOp.get(), syncMapBefore, syncMapAfter);
    }
  }
  setWaitEndIndex[op] = globalSetWaitIndex++;
  if (syncMapAfter.count(op)) {
    auto *it = syncMapAfter.find(op);
    assert(it != syncMapAfter.end());
    for (auto &syncOp : it->second) {
      if (auto *setWaitOp = dyn_cast<SetWaitOp>(syncOp.get())) {
        for (auto eventId : setWaitOp->eventIds) {
          auto &index = getSetWaitOpsIndexRef(setWaitOp->pipeSrc,
                                              setWaitOp->pipeDst, eventId);
          index.insert({globalSetWaitIndex++, setWaitOp});
        }
      }
    }
  }
  setWaitEndIndexInclusive[op] = globalSetWaitIndex++;
}

bool Solver::checkBackwardSyncEventsContains(OperationBase *op,
                                             CorePipeInfo corePipeSrc,
                                             CorePipeInfo corePipeDst,
                                             int64_t eventId, bool afterMerge) {
  auto &mp = afterMerge ? backwardSyncEventsAfterMerge : backwardSyncEvents;
  auto *it1 = mp.find(op);
  if (it1 == mp.end()) {
    return false;
  }
  auto it2 = it1->second.find({corePipeSrc, corePipeDst});
  if (it2 == it1->second.end()) {
    return false;
  }
  return it2->second.contains(eventId);
}

// Check whether a backward-sync event id can be merged at scope level.
bool Solver::checkMergeable(Scope *scopeOp, CorePipeInfo corePipeSrc,
                            CorePipeInfo corePipeDst, int64_t eventId,
                            bool shouldBeUsedAtleastOnce) {
  auto &index =
      getSetWaitOpsIndexRef(corePipeSrc.pipe, corePipeDst.pipe, eventId);
  if (shouldBeUsedAtleastOnce) {
    auto it = index.lower_bound({setWaitStartIndexInclusive[scopeOp], nullptr});
    bool usedAtleastOnce =
        it != index.end() && it->first < setWaitEndIndexInclusive[scopeOp];
    if (!usedAtleastOnce) {
      return false;
    }
  }
  {
    auto it1 =
        index.lower_bound({setWaitStartIndexInclusive[scopeOp], nullptr});
    auto it2 = index.lower_bound({setWaitEndIndex[scopeOp], nullptr});
    bool usedBefore =
        it1 != index.end() && it1->first < setWaitStartIndex[scopeOp];
    bool usedAfter =
        it2 != index.end() && it2->first < setWaitEndIndexInclusive[scopeOp];
    if (usedBefore || usedAfter) {
      return false;
    }
  }
  if (auto *conditionOp = dyn_cast<Condition>(scopeOp)) {
    return checkMergeable(conditionOp->getTrueScope(), corePipeSrc, corePipeDst,
                          eventId, true) &&
           checkMergeable(conditionOp->getFalseScope(), corePipeSrc,
                          corePipeDst, eventId, true);
  }
  if (auto *loopOp = dyn_cast<Loop>(scopeOp)) {
    for (auto &childOp : loopOp->body) {
      if (auto *childScopeOp = dyn_cast<Scope>(childOp.get())) {
        if (!checkMergeable(childScopeOp, corePipeSrc, corePipeDst, eventId,
                            false)) {
          return false;
        }
      }
    }
    for (auto &childOp : loopOp->body) {
      if (auto *childScopeOp = dyn_cast<Scope>(childOp.get())) {
        if (checkMergeable(childScopeOp, corePipeSrc, corePipeDst, eventId,
                           true)) {
          return true;
        }
      }
    }
    return false;
  }
  for (auto &childOp : scopeOp->body) {
    auto it1 =
        index.lower_bound({setWaitStartIndexInclusive[childOp.get()], nullptr});
    auto it2 = index.lower_bound({setWaitEndIndex[childOp.get()], nullptr});
    bool usedAtleastOnce = it1 != index.end() &&
                           it1->first < setWaitEndIndexInclusive[childOp.get()];
    if (!usedAtleastOnce) {
      continue;
    }
    bool before =
        it1 != index.end() && it1->first < setWaitStartIndex[childOp.get()];
    bool after = it2 != index.end() &&
                 it2->first < setWaitEndIndexInclusive[childOp.get()];
    if (before || after) {
      return false;
    }
    if (!checkBackwardSyncEventsContains(childOp.get(), corePipeSrc,
                                         corePipeDst, eventId)) {
      return false;
    }
    if (checkBackwardSyncEventsContains(childOp.get(), corePipeSrc, corePipeDst,
                                        eventId, /*afterMerge=*/true)) {
      return false;
    }
  }
  return true;
}

// Attempt to merge backward sync events across children and prune duplicates.
void Solver::mergeBackwardSyncEventIds(OperationBase *op) {
  auto *scopeOp = llvm::dyn_cast_if_present<Scope>(op);
  if (scopeOp == nullptr) {
    return;
  }
  for (auto &op : scopeOp->body) {
    mergeBackwardSyncEventIds(op.get());
  }

  if (llvm::isa_and_present<Condition, Loop>(op->parentOp)) {
    return;
  }

  auto *conditionOp = dyn_cast<Condition>(op);
  if (conditionOp != nullptr) {
    if (!conditionOp->hasFalseScope()) {
      return;
    }
  }

  llvm::DenseSet<std::tuple<CorePipeInfo, CorePipeInfo, int64_t>> toBeErased;

  llvm::SmallVector<hivm::TCoreType> coreTypes;
  if (isCrossCoreMode()) {
    coreTypes = {hivm::TCoreType::VECTOR, hivm::TCoreType::CUBE};
  } else {
    coreTypes = {hivm::TCoreType::CUBE_OR_VECTOR};
  }
  size_t pipeNumMax = static_cast<size_t>(hivm::PIPE::PIPE_NUM);
  size_t eventIdMax = static_cast<size_t>(getHWAvailableEventIdNum(syncMode));

  for (size_t eventId = 0; eventId < eventIdMax; eventId++) {
    for (auto coreSrc : coreTypes) {
      for (auto coreDst : coreTypes) {
        for (size_t pipeSrcInt = 0; pipeSrcInt < pipeNumMax; pipeSrcInt++) {
          for (size_t pipeDstInt = 0; pipeDstInt < pipeNumMax; pipeDstInt++) {
            auto pipeSrc = static_cast<hivm::PIPE>(pipeSrcInt);
            auto pipeDst = static_cast<hivm::PIPE>(pipeDstInt);
            auto corePipeSrc = CorePipeInfo(coreSrc, pipeSrc);
            auto corePipeDst = CorePipeInfo(coreDst, pipeDst);
            if (checkBackwardSyncEventsContains(scopeOp, corePipeSrc,
                                                corePipeDst, eventId)) {
              continue;
            }
            if (checkMergeable(scopeOp, corePipeSrc, corePipeDst, eventId)) {
              toBeErased.insert({corePipeSrc, corePipeDst, eventId});
              backwardSyncEvents[scopeOp][{corePipeSrc, corePipeDst}].insert(
                  eventId);
            }
          }
        }
      }
    }
  }

  if (isa<Condition, Loop>(scopeOp)) {
    for (auto &op : scopeOp->body) {
      if (auto *block = llvm::dyn_cast<Scope>(op.get())) {
        for (auto &childOp : block->body) {
          if (auto *childScopeOp = dyn_cast<Scope>(childOp.get())) {
            for (auto [corePipeSrc, corePipeDst, eventId] : toBeErased) {
              if (checkBackwardSyncEventsContains(childScopeOp, corePipeSrc,
                                                  corePipeDst, eventId)) {
                backwardSyncEvents[childScopeOp][{corePipeSrc, corePipeDst}]
                    .erase(eventId);
              }
            }
          }
        }
      }
    }
  } else {
    for (auto &childOp : scopeOp->body) {
      if (auto *childScopeOp = dyn_cast<Scope>(childOp.get())) {
        for (auto [corePipeSrc, corePipeDst, eventId] : toBeErased) {
          if (checkBackwardSyncEventsContains(childScopeOp, corePipeSrc,
                                              corePipeDst, eventId)) {
            backwardSyncEvents[childScopeOp][{corePipeSrc, corePipeDst}].erase(
                eventId);
          }
        }
      }
    }
  }
}

SyncBeforeAfterMap Solver::getBeforeAfterSyncMaps() {
  calcAllEventIds();
  collectBackwardSyncEventIds();

  SyncMap syncMapBefore, syncMapAfter;
  std::vector<ConflictPair *> conflictPairs;
  for (auto &conflictPair : chosenConflictedPairs) {
    conflictPairs.push_back(conflictPair.get());
  }
  for (auto &conflictPair : persistentChosenConflictedPairs) {
    conflictPairs.push_back(conflictPair.get());
  }

  for (auto *conflictPair : conflictPairs) {
    if (conflictPair->isUseless) {
      continue;
    }
    if (conflictPair->replacedWithUnitFlag) {
      continue;
    }
    assert(conflictPair->opSet != nullptr && conflictPair->opWait != nullptr);
    if (conflictPair->isBarrier()) {
      auto barrierOp = std::make_unique<BarrierOp>(
          nullptr, nullptr, conflictPair->waitCorePipeInfo.pipe);
      LLVM_DEBUG(barrierOp->debugId = conflictPair->id);
      syncMapBefore[conflictPair->opWait].push_back(std::move(barrierOp));
    } else {
      assert(conflictPair->eventIdNode != nullptr);
      auto setOp = std::make_unique<SetFlagOp>(
          conflictPair->opSet->op, conflictPair->opSet->parentOp,
          conflictPair->eventIdNode->getEventIds(),
          conflictPair->setCorePipeInfo.pipe,
          conflictPair->waitCorePipeInfo.pipe);
      auto waitOp = std::make_unique<WaitFlagOp>(
          conflictPair->opWait->op, conflictPair->opWait->parentOp,
          conflictPair->eventIdNode->getEventIds(),
          conflictPair->setCorePipeInfo.pipe,
          conflictPair->waitCorePipeInfo.pipe);
      if (isCrossCoreMode()) {
        setOp->coreType = conflictPair->setCorePipeInfo.coreType;
        waitOp->coreType = conflictPair->waitCorePipeInfo.coreType;
      }
      if (conflictPair->multibufferLoopPar != nullptr) {
        setOp->multibufferLoopPar = conflictPair->multibufferLoopPar;
        waitOp->multibufferLoopPar = conflictPair->multibufferLoopPar;
      }
      if (conflictPair->setOnLastIterOnly) {
        setOp->checkLastIter = true;
      }
      if (conflictPair->waitOnFirstIterOnly) {
        waitOp->checkFirstIter = true;
      }
      LLVM_DEBUG({
        setOp->debugId = conflictPair->id;
        waitOp->debugId = conflictPair->id;
      });
      assert(setOp != nullptr && waitOp != nullptr);
      syncMapAfter[conflictPair->opSet].push_back(std::move(setOp));
      syncMapBefore[conflictPair->opWait].push_front(std::move(waitOp));
    }
  }

  if (!isCrossCoreMode()) {
    resetAndBuildSetWaitOpIndex(syncMapBefore, syncMapAfter);
    auto *scopeOp = dyn_cast<Scope>(funcIr.get());
    assert(scopeOp != nullptr && scopeOp->body.front() != nullptr);
    mergeBackwardSyncEventIds(scopeOp->body.front().get());
  }

  for (auto &[op, mp] : backwardSyncEvents) {
    if (mp.empty()) {
      continue;
    }
    auto *scopeOp = dyn_cast<Scope>(op);
    assert(scopeOp != nullptr);
    for (auto [setWaitCorePipes, eventIdsSet] : mp) {
      if (eventIdsSet.empty()) {
        continue;
      }
      llvm::SmallVector<int64_t> eventIds(eventIdsSet.begin(),
                                          eventIdsSet.end());
      auto [corePipeSrc, corePipeDst] = setWaitCorePipes;
      auto setOp =
          std::make_unique<SetFlagOp>(scopeOp->op, scopeOp->parentOp, eventIds,
                                      corePipeSrc.pipe, corePipeDst.pipe);
      auto waitOp =
          std::make_unique<WaitFlagOp>(scopeOp->op, scopeOp->parentOp, eventIds,
                                       corePipeSrc.pipe, corePipeDst.pipe);
      setOp->allAtOnce = true;
      waitOp->allAtOnce = true;
      if (isCrossCoreMode()) {
        setOp->coreType = corePipeSrc.coreType;
        waitOp->coreType = corePipeDst.coreType;
      }
      assert(setOp != nullptr && waitOp != nullptr);
      syncMapBefore[scopeOp].push_back(std::move(setOp));
      syncMapAfter[scopeOp].push_front(std::move(waitOp));
    }
  }
  return std::make_pair(std::move(syncMapBefore), std::move(syncMapAfter));
}

// Main processing loop that iterates processingOrders and attempts to
// discover and record conflicts.
void Solver::processOrders() {
  for (auto &[curOcc, start, end, reverseOrder, isUseless, skip] :
       processingOrders) {
    assert(start <= end + 1);
    if (start > end) {
      continue;
    }
    if (skip) {
      for (int i = start; i <= end; i++) {
        skipOcc[reverseOrder].insert(syncIr[i].get());
      }
      continue;
    }
    assert(llvm::isa_and_present<RWOperation>(curOcc->op));
    int iStart = reverseOrder ? end : start;
    int iEnd = reverseOrder ? start - 1 : end + 1;
    int iStep = reverseOrder ? -1 : 1;
    for (int i = iStart; i != iEnd; i += iStep) {
      if (llvm::isa_and_present<RWOperation>(syncIr[i]->op)) {
        if (checkSkippable(reverseOrder, syncIr[i].get())) {
          continue;
        }
        Occurrence *occ1 = reverseOrder ? syncIr[i].get() : curOcc;
        Occurrence *occ2 = reverseOrder ? curOcc : syncIr[i].get();
        if (checkVisited(occ1, occ2) || checkImpossibleOccPair(occ1, occ2) ||
            checkAlreadySynced(occ1, occ2) ||
            checkSkipCrossCorePair(occ1, occ2)) {
          continue;
        }
        if (skipMMad1DecomposedLoopOpt(occ1, occ2) ||
            checkAlreadySyncedWithUnitFlag(occ1, occ2)) {
          continue;
        }
        auto *rwOp1 = dyn_cast<RWOperation>(occ1->op);
        auto *rwOp2 = dyn_cast<RWOperation>(occ2->op);
        assert(rwOp1 != nullptr && rwOp2 != nullptr);
        DEBUG_WITH_TYPE("gss-sync-solver-checking", {
          llvm::dbgs() << "checking: " << (isUseless ? "is-useless\n" : "\n");
          llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                       << occ1->endIndex << ' ' << occ1->op->str(0, false)
                       << '\n';
          llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                       << occ2->endIndex << ' ' << occ2->op->str(0, false)
                       << '\n';
        });
        for (auto [corePipeSrc, corePipeDst] :
             checkMemoryConflicts(rwOp1, rwOp2)) {
          if (this->alwaysUsePipeSAsWaitingPipe) {
            corePipeDst.pipe = hivm::PIPE::PIPE_S;
          }
          auto [eventIdNum, multibufferLoopPar] =
              getEventIdNum(occ1, occ2, corePipeSrc, corePipeDst);
          if (checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                 eventIdNum)) {
            handleConflict(occ1, occ2, corePipeSrc, corePipeDst, isUseless,
                           eventIdNum, multibufferLoopPar);
          }
        }
      }
    }
  }
}

// High-level solve orchestration with multiple passes and optional merging
// iterations.
void Solver::solve(int runNum) {
  LLVM_DEBUG(llvm::dbgs() << "runNum: " << runNum << '\n');
  processOrders();
  if (considerMergedBackwardSyncEventIds) {
    getBeforeAfterSyncMaps();
    backwardSyncEventsAfterMerge = backwardSyncEvents;
    reset();
    processOrders();
  }
  if (reuseSyncPairToSaveEventIds) {
    if (!barrierAllPairs.empty()) {
      bool limitReached = true;
      for (auto [corePipeSrc, corePipeDst] : barrierAllPairs) {
        if (reusePairs[{corePipeSrc, corePipeDst}] < maxReuseNum) {
          if (reusePairs[{corePipeSrc, corePipeDst}] <=
              reusedPairs[{corePipeSrc, corePipeDst}]) {
            reusePairs[{corePipeSrc, corePipeDst}] += 1;
            limitReached = false;
          }
        }
      }
      DEBUG_WITH_TYPE("gss-sync-solver-reuse", {
        llvm::dbgs() << "reusePairs: \n";
        for (auto [pipeCorePairs, cnt] : reusePairs) {
          llvm::dbgs() << get<0>(pipeCorePairs).pipe << ' '
                       << get<1>(pipeCorePairs).pipe << ' ' << cnt << '\n';
        }
      });
      if (!limitReached) {
        reset();
        disabledMultiEventIdPairs.clear();
        backwardSyncEventsAfterMerge.clear();
        solve(runNum + 1);
        return;
      }
    }
  }
  if (disableMultiEventIdForBarrierAllPairs) {
    if (!barrierAllPairs.empty()) {
      disabledMultiEventIdPairs = barrierAllPairs;
      reset();
      backwardSyncEventsAfterMerge.clear();
      processOrders();
    }
  }
  if (considerMergedBackwardSyncEventIds) {
    getBeforeAfterSyncMaps();
    backwardSyncEventsAfterMerge = backwardSyncEvents;
    reset();
    processOrders();
  }
  if (!barrierAllPairs.empty() && runNum <= maxRunNum) {
    pickAndInsertABarrierAll();
    reset();
    reusePairs.clear();
    disabledMultiEventIdPairs.clear();
    backwardSyncEventsAfterMerge.clear();
    solve(runNum + 1);
  }
}
