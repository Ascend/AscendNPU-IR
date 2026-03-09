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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "hivm-gss-solver"

using namespace mlir;
using namespace hivm::syncsolver;

// Reset per-pass bookkeeping to start fresh.
void Solver::reset(bool resetEventIdRanOutOpts) {
  if (resetEventIdRanOutOpts) {
    reusePairs.clear();
    disabledMultiEventIdPairs.clear();
    backwardSyncEventsAfterMerge.clear();
  }
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
      std::find_if(parOcc->childOccs.begin(), parOcc->childOccs.end(),
                   [occ](Occurrence *curOcc) { return occ->op == curOcc->op; });
  assert(it != parOcc->childOccs.end());
  return *it;
}

Occurrence *Solver::getLastIterOcc(Occurrence *occ, Occurrence *parOcc) {
  assert(occ != nullptr && parOcc != nullptr);
  if (parOcc->depth + 1 < occ->depth) {
    auto *newParOcc = getLastIterOcc(
        occ->getNthParent(occ->depth - parOcc->depth - 1), parOcc);
    return getLastIterOcc(occ, newParOcc);
  }
  auto it =
      std::find_if(parOcc->childOccs.rbegin(), parOcc->childOccs.rend(),
                   [occ](Occurrence *curOcc) { return occ->op == curOcc->op; });
  assert(it != parOcc->childOccs.rend());
  return *it;
}

bool Solver::checkSkipCrossCorePair(Occurrence *occ1, Occurrence *occ2) {
  if (!options.isCrossCoreMode()) {
    return false;
  }
  auto *rwOp1 = llvm::dyn_cast<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast<RWOperation>(occ2->op);
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

bool Solver::checkSkipParallelLoop(Occurrence *occ1, Occurrence *occ2) {
  if (!isBackwardSync(occ1, occ2)) {
    return false;
  }
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  assert(parOcc1 != nullptr && parOcc2 != nullptr);
  auto *parentLCALoopOcc = Occurrence::getParentloop(parOcc1);
  assert(parentLCALoopOcc != nullptr);
  auto *parentLCALoopOp = llvm::cast<Loop>(parentLCALoopOcc->op);
  return parentLCALoopOp->isParallel;
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
  assert(parOcc1 != nullptr && parOcc2 != nullptr);
  assert(parOcc1->parentOcc != nullptr && parOcc2->parentOcc != nullptr);

  auto [parOp1, parOp2] = OperationBase::getLCAPair(occ1->op, occ2->op);
  assert(parOp1 != nullptr && parOp2 != nullptr);
  assert(parOp1->parentOp != nullptr && parOp2->parentOp != nullptr);

  auto *parentLoop = OperationBase::getParentloop(parOcc1->op);
  auto *curLoop = OperationBase::getParentloop(parOp1);
  if (parentLoop == nullptr || parentLoop == curLoop) {
    return false;
  }

  assert(curLoop != nullptr);
  assert(parentLoop->isProperAncestor(curLoop));
  while (curLoop != parentLoop) {
    if (!llvm::cast<Loop>(curLoop)->isParallel) {
      return true;
    }
    curLoop = OperationBase::getParentloop(curLoop);
    assert(curLoop != nullptr);
  }
  return false;
}

// Unit-flag reuse check between two RWOperations.
bool Solver::checkAlreadySyncedWithUnitFlag(Occurrence *occ1,
                                            Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (!options.enableUnitFlagFeature) {
    return false;
  }
  if (!occ1->hasUnitFlagFeat || !occ2->hasUnitFlagFeat) {
    return false;
  }
  llvm::DenseSet<Occurrence *> visited;
  DEBUG_WITH_TYPE("gss-sync-solver-check-unit-flag", {
    llvm::dbgs() << "unit-flag-step: " << occ1->syncIrIndex << ' '
                 << occ1->op->str(0, false) << "\n";
  });
  Occurrence *curOcc = occ1->unitFlagInfo.linkedElementAsSet;
  while (curOcc != nullptr) {
    DEBUG_WITH_TYPE("gss-sync-solver-check-unit-flag", {
      llvm::dbgs() << "unit-flag-step: " << curOcc->syncIrIndex << ' '
                   << curOcc->op->str(0, false) << "\n";
    });
    auto [it, isInserted] = visited.insert(curOcc);
    if (!isInserted) {
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
  auto bufferSize1 = GetBufferBitSize(pointerCastOp1.getResult());
  auto bufferSize2 = GetBufferBitSize(pointerCastOp2.getResult());
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
      int64_t baseAddrInBits1 = baseAddr1 * utils::kBitsToByte;
      int64_t baseAddrInBits2 = baseAddr2 * utils::kBitsToByte;
      int64_t l1 = baseAddrInBits1;
      int64_t r1 = baseAddrInBits1 + bufferSize1.value();
      int64_t l2 = baseAddrInBits2;
      int64_t r2 = baseAddrInBits2 + bufferSize2.value();
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
  auto bufferSize1 = GetBufferBitSize(allocWorkSpaceOp1.getResult());
  auto bufferSize2 = GetBufferBitSize(allocWorkSpaceOp2.getResult());
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
      int64_t baseAddrInBits1 = baseAddr1 * utils::kBitsToByte;
      int64_t baseAddrInBits2 = baseAddr2 * utils::kBitsToByte;
      int64_t l1 = baseAddrInBits1;
      int64_t r1 = baseAddrInBits1 + std::max((int64_t)1, bufferSize1.value());
      int64_t l2 = baseAddrInBits2;
      int64_t r2 = baseAddrInBits2 + std::max((int64_t)1, bufferSize2.value());
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
          llvm::dyn_cast_if_present<hivm::PointerCastOp>(val1.getDefiningOp());
      auto pointerCastOp2 =
          llvm::dyn_cast_if_present<hivm::PointerCastOp>(val2.getDefiningOp());
      if (pointerCastOp1 && pointerCastOp2) {
        if (checkPointerCastMemConflict(pointerCastOp1, pointerCastOp2)) {
          return true;
        }
      }
      auto allocWorkSpaceOp1 =
          llvm::dyn_cast_if_present<bishengir::memref_ext::AllocWorkspaceOp>(
              val1.getDefiningOp());
      auto allocWorkSpaceOp2 =
          llvm::dyn_cast_if_present<bishengir::memref_ext::AllocWorkspaceOp>(
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
  auto [it, isInserted] = checkMemoryConflictsMem.insert({{rwOp1, rwOp2}, {}});
  if (!isInserted) {
    return it->second;
  }
  auto coreSrc = rwOp1->coreType;
  auto coreDst = rwOp2->coreType;
  if (options.isCrossCoreMode()) {
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
    if (auto *otherOp = llvm::dyn_cast_if_present<RWOperation>(syncIr[i]->op)) {
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
  auto bufferSize1 = GetBufferBitSize(pointerCastOp1.getResult());
  auto bufferSize2 = GetBufferBitSize(pointerCastOp2.getResult());
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
        int64_t baseAddrInBits1 = baseAddr1 * utils::kBitsToByte;
        int64_t baseAddrInBits2 = baseAddr2 * utils::kBitsToByte;
        int64_t l1 = baseAddrInBits1;
        int64_t r1 = baseAddrInBits1 + bufferSize1.value();
        int64_t l2 = baseAddrInBits2;
        int64_t r2 = baseAddrInBits2 + bufferSize2.value();
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
  auto bufferSize1 = GetBufferBitSize(allocWorkSpaceOp1.getResult());
  auto bufferSize2 = GetBufferBitSize(allocWorkSpaceOp2.getResult());
  assert(bufferSize1.has_value() && bufferSize2.has_value());
  auto addrs1 = allocWorkSpaceOp1.getOffset();
  auto addrs2 = allocWorkSpaceOp2.getOffset();
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
        int64_t baseAddr1 = static_cast<int64_t>(
            cast<IntegerAttr>(constOp1.getValue()).getInt());
        int64_t baseAddr2 = static_cast<int64_t>(
            cast<IntegerAttr>(constOp2.getValue()).getInt());
        int64_t baseAddrInBits1 = baseAddr1 * utils::kBitsToByte;
        int64_t baseAddrInBits2 = baseAddr2 * utils::kBitsToByte;
        int64_t l1 = baseAddrInBits1;
        int64_t r1 =
            baseAddrInBits1 + std::max((int64_t)1, bufferSize1.value());
        int64_t l2 = baseAddrInBits2;
        int64_t r2 =
            baseAddrInBits2 + std::max((int64_t)1, bufferSize2.value());
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
          llvm::dyn_cast_if_present<hivm::PointerCastOp>(val1.getDefiningOp());
      auto pointerCastOp2 =
          llvm::dyn_cast_if_present<hivm::PointerCastOp>(val2.getDefiningOp());
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
          llvm::dyn_cast_if_present<bishengir::memref_ext::AllocWorkspaceOp>(
              val1.getDefiningOp());
      auto allocWorkSpaceOp2 =
          llvm::dyn_cast_if_present<bishengir::memref_ext::AllocWorkspaceOp>(
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

std::optional<EventIdInfo>
Solver::checkCVMultiBufferEventIdInfo(RWOperation *rwOp1, RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!options.isCrossCoreMode()) {
    return {};
  }
  auto *parentLoop1 = rwOp1->getParentOfType<Loop>();
  auto *parentLoop2 = rwOp2->getParentOfType<Loop>();
  while (parentLoop1 != nullptr && !parentLoop1->multibufferUnrollNum) {
    parentLoop1 = parentLoop1->getParentOfType<Loop>();
  }
  while (parentLoop2 != nullptr && !parentLoop2->multibufferUnrollNum) {
    parentLoop2 = parentLoop2->getParentOfType<Loop>();
  }
  if (!parentLoop1 || !parentLoop2) {
    return {};
  }
  if (auto *parCond1 = rwOp1->getParentOfType<Condition>()) {
    if (!parCond1->isProperAncestor(rwOp2)) {
      return {};
    }
  }
  if (auto *parCond2 = rwOp2->getParentOfType<Condition>()) {
    if (!parCond2->isProperAncestor(rwOp1)) {
      return {};
    }
  }
  assert(parentLoop1->multibufferUnrollNum.value() ==
         parentLoop2->multibufferUnrollNum.value());
  EventIdInfo eventIdInfo;
  eventIdInfo.eventIdNum = parentLoop1->multibufferUnrollNum.value();
  eventIdInfo.multibufferUnrollLoop1 =
      cast<LoopLikeOpInterface>(parentLoop1->op);
  eventIdInfo.multibufferUnrollLoop2 =
      cast<LoopLikeOpInterface>(parentLoop2->op);
  return eventIdInfo;
}

std::optional<EventIdInfo>
Solver::checkMultiBufferEventIdInfo(Occurrence *occ1, Occurrence *occ2,
                                    RWOperation *rwOp1, RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!checkAllParentLoopsAreForLoops(rwOp1->op) ||
      !checkAllParentLoopsAreForLoops(rwOp2->op)) {
    return {};
  }
  if (auto loopParOpt = checkDoubleMultiBufferEventId(rwOp1, rwOp2)) {
    auto loopPar = loopParOpt.value();
    assert(loopPar != nullptr);
    auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
    if (setOcc->getParentWithOp(loopPar, /*assertExists=*/false) &&
        waitOcc->getParentWithOp(loopPar, /*assertExists=*/false)) {
      EventIdInfo eventIdInfo(2);
      eventIdInfo.multibufferLoop = loopPar;
      return eventIdInfo;
    }
  }
  return {};
}

// Determine required event id count and optional multibuffer loop parent for
// occurrences.
EventIdInfo Solver::getEventIdInfo(Occurrence *occ1, Occurrence *occ2,
                                   RWOperation *rwOp1, RWOperation *rwOp2,
                                   CorePipeInfo corePipeSrc,
                                   CorePipeInfo corePipeDst) {
  assert(occ1 != nullptr && occ2 != nullptr);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  EventIdInfo singleEventId(1);
  if (!isBackwardSync(occ1, occ2)) {
    return singleEventId;
  }
  if (auto eventIdInfo = checkCVMultiBufferEventIdInfo(rwOp1, rwOp2)) {
    return eventIdInfo.value();
  }
  if (auto eventIdInfo =
          checkMultiBufferEventIdInfo(occ1, occ2, rwOp1, rwOp2)) {
    return eventIdInfo.value();
  }
  return singleEventId;
}

// Graph-based check to determine if adding a sync between occ1 and occ2 would
// block progress. Uses GraphSolver (Dijkstra) to estimate minimal reachable
// index.
bool Solver::checkGraphConflict(
    Occurrence *occ1, Occurrence *occ2, CorePipeInfo corePipeSrc,
    CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
    std::optional<int> startIndex, std::optional<int> endIndex,
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
    if (conflictPair->endIndex < startIndex.value() ||
        conflictPair->startIndex > endIndex.value()) {
      return;
    }
    if (conflictPair->isInnerBackward) {
      if ((eventIdInfo.eventIdNum * eventIdInfo.eventIdRepeatNum) <
          (conflictPair->eventIdInfo.eventIdNum *
           conflictPair->eventIdInfo.eventIdRepeatNum)) {
        return;
      }
    }
    if (llvm::find(ignoreConflictPairs, conflictPair) !=
        ignoreConflictPairs.end()) {
      return;
    }
    auto [it, isInserted] = visited.insert(conflictPair);
    if (!isInserted) {
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
  std::optional<int> mnDistance;
  if (options.enableUnitFlagFeature) {
    mnDistance = graphSolver.runDijkstraUnitFlagEnabled(
        occ1, occ2, corePipeSrc, corePipeDst, startIndex.value(),
        endIndex.value());
  } else {
    mnDistance = graphSolver.runDijkstra(corePipeSrc, corePipeDst,
                                         startIndex.value(), endIndex.value());
  }
  return !mnDistance.has_value() || mnDistance.value() > endIndex.value();
}

bool Solver::checkSyncOpsConflicts(ConflictPair *conflictPair1,
                                   ConflictPair *conflictPair2) {
  if (conflictPair1->isBarrier() || conflictPair2->isBarrier()) {
    return false;
  }
  if (conflictPair1->startIndex > conflictPair2->startIndex) {
    std::swap(conflictPair1, conflictPair2);
  }
  if (conflictPair1->startIndex >= conflictPair2->startIndex ||
      conflictPair1->endIndex >= conflictPair2->endIndex) {
    return true;
  }
  bool result = false;
  if (conflictPair1->setCorePipeInfo != conflictPair2->setCorePipeInfo) {
    auto corePipeSrc = conflictPair1->setCorePipeInfo;
    auto corePipeDst = conflictPair2->setCorePipeInfo;
    Occurrence *occ1 = conflictPair1->setOcc;
    Occurrence *occ2 = conflictPair2->setOcc;
    auto startIndex = conflictPair1->startIndex + 1;
    auto endIndex = conflictPair2->startIndex;
    conflictPair1->startIndex += 1;
    assert(occ1 != nullptr && occ2 != nullptr);
    result = result ||
             checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                conflictPair1->eventIdInfo, startIndex,
                                endIndex, {conflictPair1}, {conflictPair2});
    conflictPair1->startIndex -= 1;
  }
  if (conflictPair1->waitCorePipeInfo != conflictPair2->waitCorePipeInfo) {
    auto corePipeSrc = conflictPair1->waitCorePipeInfo;
    auto corePipeDst = conflictPair2->waitCorePipeInfo;
    Occurrence *occ1 = conflictPair1->waitOcc;
    Occurrence *occ2 = conflictPair2->waitOcc;
    auto startIndex = conflictPair1->endIndex;
    auto endIndex = conflictPair2->endIndex - 1;
    conflictPair2->endIndex -= 1;
    assert(occ1 != nullptr && occ2 != nullptr);
    result = result ||
             checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                conflictPair1->eventIdInfo, startIndex,
                                endIndex, {conflictPair1}, {conflictPair2});
    conflictPair2->endIndex += 1;
  }
  DEBUG_WITH_TYPE("gss-check-sync-ops-conflicts", {
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
                            ConflictPair *conflictPair2) {
  assert(conflictPair1 != nullptr && conflictPair2 != nullptr);
  if (conflictPair1 == conflictPair2) {
    return false;
  }
  if (conflictPair1->isBarrier() || conflictPair2->isBarrier()) {
    return false;
  }
  if (conflictPair1->dontCheckForConflict ||
      conflictPair2->dontCheckForConflict) {
    return false;
  }
  if (options.isCrossCoreMode()) {
    return checkSyncOpsConflicts(conflictPair1, conflictPair2);
  }
  if (conflictPair1->setCorePipeInfo != conflictPair2->setCorePipeInfo ||
      conflictPair1->waitCorePipeInfo != conflictPair2->waitCorePipeInfo) {
    return false;
  }
  for (auto [l1, r1] : getRanges(conflictPair1)) {
    for (auto [l2, r2] : getRanges(conflictPair2)) {
      if (checkRangesIntersect(l1, r1 + 1, l2, r2 + 1)) {
        return true;
      }
    }
  }
  return false;
}

// Obtain available event ids while accounting for already chosen conflicts.
std::vector<ConflictPair *>
Solver::getIntersectingConflictPairs(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  if (conflictPair->isBarrier()) {
    return {};
  }
  if (conflictPair->dontCheckForConflict) {
    return {};
  }
  std::vector<ConflictPair *> intersectingConflictPairs;
  for (auto &curConflictPair : chosenConflictedPairs) {
    if (checkIntersect(conflictPair, curConflictPair.get())) {
      intersectingConflictPairs.push_back(curConflictPair.get());
    }
  }
  for (auto &curConflictPair : persistentChosenConflictedPairs) {
    if (checkIntersect(conflictPair, curConflictPair.get())) {
      intersectingConflictPairs.push_back(curConflictPair.get());
    }
  }
  return intersectingConflictPairs;
}

// Processed-pair tracking helpers.
bool Solver::checkVisited(Occurrence *occ1, Occurrence *occ2) {
  auto [it, isInserted] = processedOccPairs.insert(std::make_pair(occ1, occ2));
  return !isInserted;
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
  auto key = std::make_tuple(
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo);
  return syncedPairs[key];
}

void Solver::memorizeSyncedPair(ConflictPair *conflictPair) {
  auto key = std::make_tuple(
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo);
  syncedPairs[key].insert(conflictPair);
  for (auto *oldConflictPair : syncedPairs[key]) {
    assert(oldConflictPair->eventIdNode == conflictPair->eventIdNode);
  }
}

void Solver::forgetSyncedPair(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  auto key = std::make_tuple(
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo);
  syncedPairs[key].erase(conflictPair);
}

void Solver::memorizeReusedSyncedPair(ConflictPair *conflictPair,
                                      ConflictPair *reusedConflictPair) {
  assert(conflictPair != nullptr);
  replacedWithReusableSyncedPairs[{
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
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

std::optional<std::pair<Occurrence *, Occurrence *>>
Solver::checkAndApplyMmadl0LoopOpt(ConflictPair *conflictPair, Occurrence *occ1,
                                   Occurrence *occ2, Occurrence *parOcc1,
                                   Occurrence *parOcc2) {
  if (!options.decomposeMmadl1Op) {
    return {};
  }
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
  return {};
}

std::optional<UnitFlagInfo> Solver::checkUnitFlagPatterns(Occurrence *occ1,
                                                          Occurrence *occ2) {
  if (!options.enableUnitFlagFeature) {
    return {};
  }
  assert(occ1 != nullptr && occ1->op != nullptr);
  assert(occ2 != nullptr && occ2->op != nullptr);
  auto *rwOp1 = llvm::dyn_cast<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!rwOp1->hasUnitFlagFeat || !rwOp2->hasUnitFlagFeat) {
    return {};
  }
  if (!occ1->unitFlagInfo.disabledAsSet() ||
      !occ2->unitFlagInfo.disabledAsWait()) {
    return {};
  }
  if (checkMemoryConflictBetweenOccExclusive(occ1, occ2)) {
    return {};
  }
  scf::ForOp backwardSyncLoop;
  if (isBackwardSync(occ1, occ2)) {
    auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
    assert(parOcc1 != nullptr && parOcc2 != nullptr);
    auto *parentLCALoopOcc = Occurrence::getParentloop(parOcc1);
    assert(parentLCALoopOcc != nullptr);
    assert(rwOp1->op != nullptr && rwOp2->op != nullptr);
    if (!(backwardSyncLoop =
              llvm::dyn_cast<scf::ForOp>(parentLCALoopOcc->op->op))) {
      return {};
    }
    if (rwOp1->op->getParentOp() != parentLCALoopOcc->op->op ||
        rwOp2->op->getParentOp() != parentLCALoopOcc->op->op) {
      return {};
    }
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
Solver::getSetWaitLCAPairOcc(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);

  auto [grandParOcc1, grandParOcc2] = Occurrence::getLCAPair(occ1, occ2);
  assert(grandParOcc1 != nullptr && grandParOcc2 != nullptr);
  assert(grandParOcc1->parentOcc != nullptr &&
         grandParOcc2->parentOcc != nullptr);

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
  assert(setOcc != nullptr && waitOcc != nullptr);
  assert(parOcc1->isProperAncestor(setOcc));
  assert(parOcc2->isProperAncestor(waitOcc));

  auto *parLoop = Occurrence::getParentloop(setOcc);
  while (parLoop != nullptr && grandParOcc1->isProperAncestor(parLoop)) {
    setOcc = parLoop;
    waitOcc = Occurrence::getParentloop(waitOcc);
    parLoop = Occurrence::getParentloop(setOcc);
  }
  return std::make_pair(setOcc, waitOcc);
}

std::pair<Occurrence *, Occurrence *>
Solver::getFixedSetWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  // - get setOcc waitOcc where:
  // setOcc->op->parent = waitOcc->op->parent = lca(occ1, occ2)->op
  auto [setOcc, waitOcc] = getSetWaitLCAPairOcc(occ1, occ2);

  // - check if it's the case of while loop:
  // while{
  //  before{
  //    occ1
  //  }
  //  setOcc;
  //  waitOcc;
  //  after{
  //    occ2
  //  }
  // }
  // - and fix it to be:
  // while{
  //  before{
  //    occ1
  //    setOcc;
  //    ...
  //    waitOcc;
  //    placeHolder
  //  }
  //  after{
  //    occ2
  //  }
  // }
  if (setOcc->op != waitOcc->op) {
    if (auto *parLoopOp =
            llvm::dyn_cast_if_present<Loop>(setOcc->parentOcc->op)) {
      if (parLoopOp->body.size() > 1 && !isa<PlaceHolder>(waitOcc->op)) {
        auto *placeHolderOcc = getScopeEndPlaceHolderOcc(setOcc);
        std::tie(setOcc, waitOcc) = getSetWaitLCAPairOcc(occ1, placeHolderOcc);
      }
    }
  }

  // - check if it's the case of:
  // loop(iter-1){
  //   condition{
  //     true-scope{}
  //     setOcc()
  //     false-scope{}
  //   }
  // }
  // loop(iter-2){
  //   condition{
  //     true-scope{}
  //     waitOcc()
  //     false-scope{}
  //   }
  // }
  // - and fix it to be:
  // loop(iter-1){
  //   condition{
  //     true-scope{}
  //     false-scope{}
  //   }
  //   setOcc()
  // }
  // loop(iter-2){
  //   waitOcc()
  //   condition{
  //     true-scope{}
  //     false-scope{}
  //   }
  // }
  if (isBackwardSync(occ1, occ2)) {
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
  }

  // - for the case of cv-pipelining:
  // loop(){
  //   op1
  // } {unroll=x}
  // setOcc
  // waitOcc
  // loop(){
  //   op2
  // } {unroll=x}
  // - and fix it to be:
  // loop(){
  //   op1
  //   setOcc
  // } {unroll=x}
  // loop(){
  //   waitOcc
  //   op2
  // } {unroll=x}
  if (options.isCrossCoreMode()) {
    assert(setOcc->op != nullptr && waitOcc->op != nullptr);
    auto *forOp1 = llvm::dyn_cast<Loop>(setOcc->op);
    auto *forOp2 = llvm::dyn_cast<Loop>(waitOcc->op);
    if (forOp1 != nullptr && forOp2 != nullptr) {
      if (forOp1->multibufferUnrollNum && forOp2->multibufferUnrollNum) {
        setOcc = occ1->getNthParent(occ1->depth - setOcc->depth - 2);
        waitOcc = occ2->getNthParent(occ2->depth - waitOcc->depth - 2);
      }
    }
  }

  // - check if it's the case of:
  // {
  //   op1
  //   setOcc
  //   ...
  //   waitOcc
  //   loop(){}
  //   setOcc
  //   ...
  //   waitOcc
  //   op2
  // }
  // - and fix it to be:
  // {
  //   op1
  //   setOcc
  //   ...
  //   waitOcc
  //   placeHolder
  //   loop(){}
  //   placeHolder
  //   setOcc
  //   ...
  //   waitOcc
  //   op2
  // }
  if (llvm::isa_and_present<Loop>(setOcc->op)) {
    setOcc = getAfterPlaceHolderOcc(setOcc);
  }
  if (llvm::isa_and_present<Loop>(waitOcc->op)) {
    waitOcc = getBeforePlaceHolderOcc(waitOcc);
  }

  return std::make_pair(setOcc, waitOcc);
}

std::optional<std::pair<Occurrence *, Occurrence *>>
Solver::getFunctionBlockSetWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *parFunctionBlock1 = occ1->getParentOfType<FunctionBlock>();
  auto *parFunctionBlock2 = occ2->getParentOfType<FunctionBlock>();
  if (parFunctionBlock1 == parFunctionBlock2) {
    return {};
  }
  auto *placeHolderOcc = getScopeBeginPlaceHolderOcc(parFunctionBlock2);
  return std::make_pair(placeHolderOcc, occ2);
}

std::optional<std::pair<Occurrence *, Occurrence *>>
Solver::getUnlikelyCondSetWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (options.isCrossCoreMode() && isBackwardSync(occ1, occ2)) {
    return {};
  }
  if (auto *unlikelyParCondOcc1 =
          Occurrence::getUnlikelyParentCondition(occ1)) {
    if (!unlikelyParCondOcc1->isProperAncestor(occ2)) {
      auto *parentLoopOcc = Occurrence::getParentloop(unlikelyParCondOcc1);
      if (parentLoopOcc == nullptr || parentLoopOcc->isProperAncestor(occ2)) {
        auto *placeHolderOcc = getScopeEndPlaceHolderOcc(
            occ1->getNthParent(occ1->depth - unlikelyParCondOcc1->depth - 1));
        return std::make_pair(occ1, placeHolderOcc);
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
        return std::make_pair(placeHolderOcc, occ2);
      }
    }
  }
  return {};
}

std::pair<Occurrence *, Occurrence *> Solver::getSetWaitOcc(Occurrence *occ1,
                                                            Occurrence *occ2) {
  if (auto functionBlockOpt = getFunctionBlockSetWaitOcc(occ1, occ2)) {
    std::tie(occ1, occ2) = functionBlockOpt.value();
  }
  if (auto unlikelyOpt = getUnlikelyCondSetWaitOcc(occ1, occ2)) {
    std::tie(occ1, occ2) = unlikelyOpt.value();
  }
  return getFixedSetWaitOcc(occ1, occ2);
}

Occurrence *Solver::getBarrierWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
  if (!waitOcc->isProperAncestor(occ2)) {
    return waitOcc;
  }
  auto allParents = occ2->getAllParents();
  while (!allParents.empty() && allParents.back()->isProperAncestor(waitOcc)) {
    allParents.pop_back();
  }
  while (allParents.size() >= 2 &&
         llvm::isa_and_present<Condition>(allParents.back()->op)) {
    allParents.pop_back();
    assert(llvm::isa_and_present<Scope>(allParents.back()->op));
    allParents.pop_back();
  }
  waitOcc = !allParents.empty() ? allParents.back() : occ2;
  return waitOcc;
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
    if (conflictPair->eventIdNode->eventIdNum <
        curConflictPair->eventIdNode->eventIdNum) {
      continue;
    }
    assert(conflictPair->eventIdNode != nullptr);
    assert(curConflictPair->eventIdNode != nullptr);
    if (conflictPair->eventIdNode->eventIdNum >
        curConflictPair->eventIdNode->eventIdNum) {
      if (conflictPair->eventIdNode->eventIdNum %
          curConflictPair->eventIdNode->eventIdNum) {
        continue;
      }
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
        {conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
         conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo});
    if (it != replacedWithReusableSyncedPairs.end()) {
      oldReusedConflictPair = it->second;
    }
  }

  if (!conflictPair->isUseless) {
    auto it = replacedWithReusableSyncedPairs.find(
        {conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
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
  reusableConflictPair->setOp = conflictPair->setOp;
  reusableConflictPair->setOcc = conflictPair->setOcc;
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
    assert(oldReusedConflictPair->waitOp == reusableConflictPair->waitOp);
  }

  if (!conflictPair->isUseless) {
    reusedPairs[{corePipeSrc, corePipeDst}] += 1;
  }

  return true;
}

std::unique_ptr<EventIdSolver> &
Solver::getEventIdSolverRef(hivm::PIPE pipeSrc, hivm::PIPE pipeDst) {
  if (options.isCrossCoreMode()) {
    pipeSrc = hivm::PIPE::PIPE_UNASSIGNED;
    pipeDst = hivm::PIPE::PIPE_UNASSIGNED;
  }
  auto key = std::make_tuple(pipeSrc, pipeDst);
  if (!eventIdSolver.contains(key)) {
    int64_t eventIdNumMax =
        getHWAvailableEventIdNum(options.syncMode, pipeSrc, pipeDst);
    eventIdSolver[key] = std::make_unique<EventIdSolver>(eventIdNumMax);
  }
  return eventIdSolver[key];
}

bool Solver::checkReuseMultiBufferFlagId(ConflictPair *conflictPair) {
  if (options.useDifferentMultiBufferFlagIds) {
    return false;
  }
  if (!conflictPair->isInnerBackward ||
      conflictPair->eventIdInfo.eventIdNum <= 1) {
    return false;
  }
  auto [setOcc, waitOcc] =
      std::tie(conflictPair->setOcc, conflictPair->waitOcc);
  auto *backwardSyncLoopOcc = conflictPair->backwardSyncLoopOcc;
  assert(backwardSyncLoopOcc != nullptr);
  if (auto *parCondOcc1 = setOcc->getParentOfType<Condition>()) {
    if (!parCondOcc1->isProperAncestor(backwardSyncLoopOcc)) {
      return false;
    }
  }
  if (auto *parCondOcc2 = waitOcc->getParentOfType<Condition>()) {
    if (!parCondOcc2->isProperAncestor(backwardSyncLoopOcc)) {
      return false;
    }
  }
  return true;
}

void Solver::handleSetWaitConflict(Occurrence *occ1, Occurrence *occ2,
                                   CorePipeInfo corePipeSrc,
                                   CorePipeInfo corePipeDst,
                                   EventIdInfo eventIdInfo, bool isUseless) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  assert(corePipeSrc != corePipeDst);

  Loop *parentLCALoopOp{nullptr};
  Occurrence *parentLCALoopOcc{nullptr};
  Occurrence *parentLCALoopBeforePHOcc{nullptr};
  Occurrence *parentLCALoopAfterPHOcc{nullptr};
  auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);

  auto [lcaSetOp, lcaWaitOp] =
      OperationBase::getLCAPair(setOcc->op, waitOcc->op);
  auto *normScopeOcc1 = setOcc->getParentWithOp(lcaSetOp->parentOp);
  auto *normScopeOcc2 = waitOcc->getParentWithOp(lcaWaitOp->parentOp);
  assert(normScopeOcc1->op == normScopeOcc2->op);
  auto *normScopeOp = normScopeOcc1->op;
  assert(normScopeOp != nullptr);
  assert(normScopeOp->parentOp != nullptr);

  auto conflictPair = std::make_unique<ConflictPair>(
      rwOp1, rwOp2, setOcc->op, waitOcc->op, setOcc, waitOcc, corePipeSrc,
      corePipeDst, setOcc->endIndex, waitOcc->startIndex);
  assert(conflictPair->startIndex <= conflictPair->endIndex);

  conflictPair->isUseless = isUseless;
  conflictPair->isInnerBackward = isBackwardSync(setOcc, waitOcc);
  conflictPair->eventIdInfo = eventIdInfo;

  if (conflictPair->isInnerBackward) {
    auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
    assert(parOcc1 != nullptr && parOcc2 != nullptr);

    parentLCALoopOcc = Occurrence::getParentloop(parOcc1);
    assert(parentLCALoopOcc != nullptr);
    conflictPair->backwardSyncLoopOcc = parentLCALoopOcc;

    parentLCALoopOp = llvm::dyn_cast<Loop>(parentLCALoopOcc->op);
    assert(parentLCALoopOp != nullptr);
    conflictPair->backwardSyncLoopOp = parentLCALoopOp;

    parentLCALoopBeforePHOcc = getBeforePlaceHolderOcc(parentLCALoopOcc);
    assert(parentLCALoopBeforePHOcc != nullptr);
    parentLCALoopAfterPHOcc = getAfterPlaceHolderOcc(parentLCALoopOcc);
    assert(parentLCALoopAfterPHOcc != nullptr);
  }

  if (auto setWaitOccs = checkAndApplyMmadl0LoopOpt(conflictPair.get(), occ1,
                                                    occ2, setOcc, waitOcc)) {
    std::tie(setOcc, waitOcc) = setWaitOccs.value();
    conflictPair->updateSetWaitOccs(setOcc, waitOcc);
  }

  if (!conflictPair->isInnerBackward ||
      disabledMultiEventIdPairs.contains({corePipeSrc, corePipeDst})) {
    conflictPair->eventIdInfo = EventIdInfo(1);
  }
  if (checkReuseMultiBufferFlagId(conflictPair.get())) {
    conflictPair->eventIdInfo.eventIdRepeatNum =
        conflictPair->eventIdInfo.eventIdNum;
    conflictPair->eventIdInfo.eventIdNum = 1;
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

  if (auto *oldEventIdNode = getOldEventIdNodeIfExists(conflictPair.get())) {
    conflictPair->eventIdNode = oldEventIdNode;
    curEventIdSolver->insertConflictPair(oldEventIdNode, conflictPair.get());
  } else {
    bool reversedPriority = false;
    if (conflictPair->isInnerBackward) {
      if (OperationBase::getParentloop(occ1->op) == normScopeOp->parentOp &&
          OperationBase::getParentloop(occ2->op) == normScopeOp->parentOp) {
        reversedPriority = true;
      }
    }
    conflictPair->eventIdNode = curEventIdSolver->createNode(
        conflictPair.get(), conflictPair->eventIdInfo.eventIdNum,
        reversedPriority);
  }

  if (options.reuseSyncPairToSaveEventIds) {
    if (reuseConflictPair(conflictPair.get(), normScopeOcc1, normScopeOcc2)) {
      curEventIdSolver->undoActions();
      return;
    }
  }

  auto intersectingConflictPairs =
      getIntersectingConflictPairs(conflictPair.get());
  curEventIdSolver->addConflicts(conflictPair.get(), intersectingConflictPairs);
  if (!checkColorable()) {
    return;
  }

  LLVM_DEBUG({
    llvm::dbgs() << conflictPair->str() << '\n';
    if (parentLCALoopOcc != nullptr) {
      llvm::dbgs() << parentLCALoopOcc->op->str(0, false) << '\n';
    }
  });

  llvm::SmallVector<std::pair<std::unique_ptr<ConflictPair>, Occurrence *>>
      extraConflictPairs;

  auto insertExtraConflictPair = [&](Occurrence *setOcc, Occurrence *waitOcc,
                                     Occurrence *parentScope,
                                     bool couldNotRun = false) -> bool {
    assert(setOcc != nullptr && waitOcc != nullptr && parentScope != nullptr);
    auto extraConflictPair = conflictPair->clone(setOcc, waitOcc);
    extraConflictPair->isUseless = true;
    extraConflictPair->dontReuse = true;
    if (couldNotRun || options.moveOutAndMergeBackwardSyncPairs) {
      extraConflictPair->couldNotRun = true;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "extra-conflict-pair: " << extraConflictPair->str()
                   << "\n";
    });
    curEventIdSolver->insertConflictPair(conflictPair->eventIdNode,
                                         extraConflictPair.get());
    auto intersectingConflictPairs =
        getIntersectingConflictPairs(extraConflictPair.get());
    curEventIdSolver->addConflicts(extraConflictPair.get(),
                                   intersectingConflictPairs);
    if (!checkColorable()) {
      return false;
    }
    extraConflictPairs.push_back(
        std::make_pair(std::move(extraConflictPair), parentScope));
    return true;
  };

  if (conflictPair->isInnerBackward && conflictPair->eventIdNode != nullptr) {
    bool insertOuterBwdConflictPair = false;
    if ((conflictPair->eventIdInfo.eventIdNum *
         conflictPair->eventIdInfo.eventIdRepeatNum) > 1) {
      insertOuterBwdConflictPair = true;
    } else if (options.isCrossCoreMode()) {
      if (setOcc->parentOcc == nullptr ||
          setOcc->parentOcc->parentOcc == nullptr ||
          setOcc->parentOcc->parentOcc->op != parentLCALoopOp) {
        insertOuterBwdConflictPair = true;
      } else if (waitOcc->parentOcc == nullptr ||
                 waitOcc->parentOcc->parentOcc == nullptr ||
                 waitOcc->parentOcc->parentOcc->op != parentLCALoopOp) {
        insertOuterBwdConflictPair = true;
      }
    }
    if (insertOuterBwdConflictPair) {
      // insert useless conflictPair to cover the whole loop when having
      // multi-eventid backward sync to reserve the eventIds.
      if (!insertExtraConflictPair(parentLCALoopBeforePHOcc,
                                   parentLCALoopAfterPHOcc,
                                   parentLCALoopOcc->parentOcc)) {
        return;
      }
    }
  }

  if (conflictPair->isInnerBackward && conflictPair->eventIdNode != nullptr) {
    // insert header/footer useless conflictPairs to reserve the eventIds.
    auto *loopOpOcc1 = getFirstIterOcc(waitOcc, normScopeOcc1);
    auto *loopOpOcc2 = getLastIterOcc(setOcc, normScopeOcc2);
    if (!insertExtraConflictPair(parentLCALoopBeforePHOcc, loopOpOcc1,
                                 parentLCALoopOcc, /*couldNotRun=*/true)) {
      return;
    }
    if (!insertExtraConflictPair(loopOpOcc2, parentLCALoopAfterPHOcc,
                                 parentLCALoopOcc, /*couldNotRun=*/true)) {
      return;
    }
  }

  bool dontInsert = false;
  if (conflictPair->isInnerBackward && normScopeOcc1 != normScopeOcc2) {
    auto *parCond = OperationBase::getParentCondition(conflictPair->setOp);
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

  memorizeSyncedPair(conflictPair.get());
  chosenConflictedPairs.push_back(std::move(conflictPair));

  for (auto &[extraConflictPair, parentScope] : extraConflictPairs) {
    scopeOccChosenConflicts[parentScope].insert(extraConflictPair.get());
    chosenConflictedPairs.push_back(std::move(extraConflictPair));
  }

  curEventIdSolver->clearActionStack();
}

void Solver::handleBarrierConflict(Occurrence *occ1, Occurrence *occ2,
                                   CorePipeInfo corePipeSrc,
                                   CorePipeInfo corePipeDst, bool isUseless) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);

  assert(corePipeSrc == corePipeDst);
  if (corePipeSrc.pipe == hivm::PIPE::PIPE_S) {
    return;
  }
  if (options.isRegBasedArch) {
    if (corePipeSrc.pipe == hivm::PIPE::PIPE_V) {
      return;
    }
  }
  if (corePipeSrc.pipe == hivm::PIPE::PIPE_M) {
    isUseless = true;
  }

  auto *waitOcc = getBarrierWaitOcc(occ1, occ2);

  auto conflictPair = std::make_unique<ConflictPair>(
      rwOp1, rwOp2, waitOcc->op, waitOcc->op, waitOcc, waitOcc, corePipeSrc,
      corePipeDst, waitOcc->startIndex, waitOcc->startIndex);
  conflictPair->isUseless = isUseless;
  assert(conflictPair->startIndex <= conflictPair->endIndex);

  LLVM_DEBUG({ llvm::dbgs() << conflictPair->str() << '\n'; });

  auto *normScopeOcc = waitOcc->parentOcc;
  scopeOccChosenConflicts[normScopeOcc].insert(conflictPair.get());
  chosenConflictedPairs.push_back(std::move(conflictPair));
}

void Solver::handleUnitFlagConflict(Occurrence *occ1, Occurrence *occ2,
                                    CorePipeInfo corePipeSrc,
                                    CorePipeInfo corePipeDst,
                                    UnitFlagInfo unitFlagInfo, bool isUseless) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  assert(corePipeSrc != corePipeDst);

  auto *setOcc = occ1;
  auto *waitOcc = occ2;
  auto *normScopeOcc1 = setOcc->parentOcc;
  auto *normScopeOcc2 = waitOcc->parentOcc;

  auto conflictPair = std::make_unique<ConflictPair>(
      rwOp1, rwOp2, setOcc->op, waitOcc->op, setOcc, waitOcc, corePipeSrc,
      corePipeDst, setOcc->endIndex, waitOcc->startIndex);
  assert(conflictPair->startIndex <= conflictPair->endIndex);

  conflictPair->isUseless = true;
  conflictPair->dontReuse = true;
  conflictPair->replacedWithUnitFlag = true;
  conflictPair->dontCheckForConflict = true;
  conflictPair->isInnerBackward = isBackwardSync(setOcc, waitOcc);

  Occurrence *parentLCALoopOcc{nullptr};
  if (conflictPair->isInnerBackward) {
    auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
    assert(parOcc1 != nullptr && parOcc2 != nullptr);
    parentLCALoopOcc = Occurrence::getParentloop(parOcc1);
    assert(parentLCALoopOcc != nullptr);
  }

  LLVM_DEBUG({
    llvm::dbgs() << conflictPair->str() << '\n';
    if (parentLCALoopOcc != nullptr) {
      llvm::dbgs() << parentLCALoopOcc->op->str(0, false) << '\n';
    }
  });

  occ1->unitFlagInfo.merge(unitFlagInfo, occ1, occ2,
                           /*asSet=*/true, /*asWait=*/false);
  occ2->unitFlagInfo.merge(unitFlagInfo, occ1, occ2,
                           /*asSet=*/false, /*asWait=*/true);
  if (!isUseless) {
    rwOp1->mergedUnitFlagInfo.merge(unitFlagInfo, /*asSet=*/true,
                                    /*asWait=*/false);
    rwOp2->mergedUnitFlagInfo.merge(unitFlagInfo, /*asSet=*/false,
                                    /*asWait=*/true);
  }

  scopeOccPairChosenConflicts[{normScopeOcc1, normScopeOcc2}].insert(
      conflictPair.get());
  chosenConflictedPairs.push_back(std::move(conflictPair));
}

void Solver::handleConflict(Occurrence *occ1, Occurrence *occ2,
                            RWOperation *rwOp1, RWOperation *rwOp2,
                            CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                            EventIdInfo eventIdInfo, bool isUseless) {
  if (!checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst, eventIdInfo)) {
    return;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "conflict found: " << "eventIdNum("
                 << eventIdInfo.eventIdNum << ")\n";
    llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                 << occ1->endIndex << ' ' << rwOp1->str(0, false) << '\n';
    llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                 << occ2->endIndex << ' ' << rwOp2->str(0, false) << '\n';
  });
  if (corePipeSrc == corePipeDst) {
    handleBarrierConflict(occ1, occ2, corePipeSrc, corePipeDst, isUseless);
  } else if (auto unitFlagInfo = checkUnitFlagPatterns(occ1, occ2)) {
    handleUnitFlagConflict(occ1, occ2, corePipeSrc, corePipeDst,
                           unitFlagInfo.value(), isUseless);
  } else {
    handleSetWaitConflict(occ1, occ2, corePipeSrc, corePipeDst, eventIdInfo,
                          isUseless);
  }
}

void Solver::calcAllEventIds() {
  for (auto &[pipes, eventIdSolver] : eventIdSolver) {
    assert(eventIdSolver != nullptr);
    auto result = eventIdSolver->shrinkEventIdMaxToEventIdNum();
    assert(llvm::succeeded(result));
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
        auto &e = backwardSyncEvents[conflictPair->backwardSyncLoopOp]
                                    [{conflictPair->setCorePipeInfo,
                                      conflictPair->waitCorePipeInfo}][eventId];
        e = std::max(e, conflictPair->eventIdInfo.eventIdRepeatNum);
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
      if (auto *setWaitOp = llvm::dyn_cast<SetWaitOp>(syncOp.get())) {
        for (auto eventId : setWaitOp->eventIds) {
          auto &index = getSetWaitOpsIndexRef(setWaitOp->pipeSrc,
                                              setWaitOp->pipeDst, eventId);
          index.insert({globalSetWaitIndex++, setWaitOp});
        }
      }
    }
  }
  setWaitStartIndex[op] = globalSetWaitIndex++;
  if (auto *scopeOp = llvm::dyn_cast<Scope>(op)) {
    for (auto &childOp : scopeOp->body) {
      collectSetWaitOpsIndexes(childOp.get(), syncMapBefore, syncMapAfter);
    }
  }
  setWaitEndIndex[op] = globalSetWaitIndex++;
  if (syncMapAfter.count(op)) {
    auto *it = syncMapAfter.find(op);
    assert(it != syncMapAfter.end());
    for (auto &syncOp : it->second) {
      if (auto *setWaitOp = llvm::dyn_cast<SetWaitOp>(syncOp.get())) {
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
                                             int64_t eventId) {
  auto *it1 = backwardSyncEvents.find(op);
  if (it1 == backwardSyncEvents.end()) {
    return false;
  }
  auto it2 = it1->second.find({corePipeSrc, corePipeDst});
  if (it2 == it1->second.end()) {
    return false;
  }
  return it2->second.contains(eventId);
}

bool Solver::checkBackwardSyncEventsContainsAfterMerge(
    OperationBase *op, CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst) {
  auto *it1 = backwardSyncEventsAfterMerge.find(op);
  if (it1 == backwardSyncEventsAfterMerge.end()) {
    return false;
  }
  return it1->second.contains({corePipeSrc, corePipeDst});
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
  if (auto *conditionOp = llvm::dyn_cast<Condition>(scopeOp)) {
    return checkMergeable(conditionOp->getTrueScope(), corePipeSrc, corePipeDst,
                          eventId, true) &&
           checkMergeable(conditionOp->getFalseScope(), corePipeSrc,
                          corePipeDst, eventId, true);
  }
  if (auto *loopOp = llvm::dyn_cast<Loop>(scopeOp)) {
    for (auto &childOp : loopOp->body) {
      if (auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get())) {
        if (!checkMergeable(childScopeOp, corePipeSrc, corePipeDst, eventId,
                            false)) {
          return false;
        }
      }
    }
    for (auto &childOp : loopOp->body) {
      if (auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get())) {
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
    if (checkBackwardSyncEventsContainsAfterMerge(childOp.get(), corePipeSrc,
                                                  corePipeDst)) {
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

  if (llvm::isa_and_present<FunctionBlock>(op)) {
    return;
  }
  if (llvm::isa_and_present<Condition, Loop>(op->parentOp)) {
    return;
  }

  auto *conditionOp = llvm::dyn_cast<Condition>(op);
  if (conditionOp != nullptr) {
    if (!conditionOp->hasFalseScope()) {
      return;
    }
  }

  llvm::DenseSet<std::tuple<CorePipeInfo, CorePipeInfo, int64_t>> toBeErased;

  llvm::SmallVector<hivm::TCoreType> coreTypes;
  if (options.isCrossCoreMode()) {
    coreTypes = {hivm::TCoreType::VECTOR, hivm::TCoreType::CUBE};
  } else {
    coreTypes = {hivm::TCoreType::CUBE_OR_VECTOR};
  }
  size_t pipeNumMax = static_cast<size_t>(hivm::PIPE::PIPE_NUM);
  size_t eventIdMax = getHWAvailableEventIdNum(options.syncMode);

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
                  {eventId, 1});
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
          if (auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get())) {
            for (auto [corePipeSrc, corePipeDst, eventId] : toBeErased) {
              if (checkBackwardSyncEventsContains(childScopeOp, corePipeSrc,
                                                  corePipeDst, eventId)) {
                auto key = std::make_tuple(corePipeSrc, corePipeDst);
                backwardSyncEvents[childScopeOp][key].erase(eventId);
                if (backwardSyncEvents[childScopeOp][key].empty()) {
                  backwardSyncEvents[childScopeOp].erase(key);
                }
              }
            }
          }
        }
      }
    }
  } else {
    for (auto &childOp : scopeOp->body) {
      if (auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get())) {
        for (auto [corePipeSrc, corePipeDst, eventId] : toBeErased) {
          if (checkBackwardSyncEventsContains(childScopeOp, corePipeSrc,
                                              corePipeDst, eventId)) {
            auto key = std::make_tuple(corePipeSrc, corePipeDst);
            backwardSyncEvents[childScopeOp][key].erase(eventId);
            if (backwardSyncEvents[childScopeOp][key].empty()) {
              backwardSyncEvents[childScopeOp].erase(key);
            }
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
    assert(conflictPair->setOp != nullptr && conflictPair->waitOp != nullptr);
    if (conflictPair->isBarrier()) {
      auto barrierOp = std::make_unique<BarrierOp>(
          conflictPair->waitOp->op, conflictPair->waitOp->parentOp,
          conflictPair->waitCorePipeInfo.pipe);
      LLVM_DEBUG(barrierOp->debugId = conflictPair->id);
      syncMapBefore[conflictPair->waitOp].push_back(std::move(barrierOp));
    } else {
      assert(conflictPair->eventIdNode != nullptr);
      auto setOp = std::make_unique<SetFlagOp>(
          conflictPair->setOp->op, conflictPair->setOp->parentOp,
          conflictPair->eventIdNode->getEventIds(),
          conflictPair->setCorePipeInfo.pipe,
          conflictPair->waitCorePipeInfo.pipe);
      auto waitOp = std::make_unique<WaitFlagOp>(
          conflictPair->waitOp->op, conflictPair->waitOp->parentOp,
          conflictPair->eventIdNode->getEventIds(),
          conflictPair->setCorePipeInfo.pipe,
          conflictPair->waitCorePipeInfo.pipe);
      if (options.isCrossCoreMode()) {
        setOp->coreType = conflictPair->setCorePipeInfo.coreType;
        waitOp->coreType = conflictPair->waitCorePipeInfo.coreType;
      }
      setOp->eventIdInfo = conflictPair->eventIdInfo;
      waitOp->eventIdInfo = conflictPair->eventIdInfo;
      setOp->checkLastIter = conflictPair->setOnLastIterOnly;
      waitOp->checkFirstIter = conflictPair->waitOnFirstIterOnly;
      LLVM_DEBUG({
        setOp->debugId = conflictPair->id;
        waitOp->debugId = conflictPair->id;
      });
      assert(setOp != nullptr && waitOp != nullptr);
      syncMapAfter[conflictPair->setOp].push_back(std::move(setOp));
      syncMapBefore[conflictPair->waitOp].push_front(std::move(waitOp));
    }
  }

  if (options.moveOutAndMergeBackwardSyncPairs) {
    resetAndBuildSetWaitOpIndex(syncMapBefore, syncMapAfter);
    auto *scopeOp = llvm::dyn_cast<Scope>(funcIr.get());
    assert(scopeOp != nullptr && scopeOp->body.front() != nullptr);
    mergeBackwardSyncEventIds(scopeOp->body.front().get());
  }

  for (auto &[op, mp] : backwardSyncEvents) {
    if (mp.empty()) {
      continue;
    }
    auto *scopeOp = llvm::dyn_cast<Scope>(op);
    assert(scopeOp != nullptr);
    for (auto [setWaitCorePipes, eventIdsMp] : mp) {
      if (eventIdsMp.empty()) {
        continue;
      }
      llvm::SmallVector<int64_t> eventIds;
      for (auto [eventId, repeatNum] : eventIdsMp) {
        llvm::SmallVector<int64_t> curEventIds(repeatNum, eventId);
        llvm::append_range(eventIds, curEventIds);
      }
      llvm::sort(eventIds);
      auto [corePipeSrc, corePipeDst] = setWaitCorePipes;
      auto setOp =
          std::make_unique<SetFlagOp>(scopeOp->op, scopeOp->parentOp, eventIds,
                                      corePipeSrc.pipe, corePipeDst.pipe);
      auto waitOp =
          std::make_unique<WaitFlagOp>(scopeOp->op, scopeOp->parentOp, eventIds,
                                       corePipeSrc.pipe, corePipeDst.pipe);
      setOp->allAtOnce = true;
      waitOp->allAtOnce = true;
      if (options.isCrossCoreMode()) {
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

void Solver::processConflict(Occurrence *occ1, Occurrence *occ2,
                             RWOperation *rwOp1, RWOperation *rwOp2,
                             bool isUseless) {
  for (auto [corePipeSrc, corePipeDst] : checkMemoryConflicts(rwOp1, rwOp2)) {
    if (options.alwaysUsePipeSAsWaitingPipe) {
      corePipeDst.pipe = hivm::PIPE::PIPE_S;
    }
    auto eventIdInfo =
        getEventIdInfo(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst);
    handleConflict(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst,
                   eventIdInfo, isUseless);
  }
}

// Main processing loop that iterates processingOrders and attempts to
// discover and record conflicts.
void Solver::processOrders() {
  for (auto &[occ1, occ2, rwOp1, rwOp2, isUseless] : processingOrders) {
    assert(occ1 != occ2);
    assert(occ1->syncIrIndex < occ2->syncIrIndex);
    if (checkVisited(occ1, occ2)) {
      assert(false && "expected to not check a pair more than once.");
      continue;
    }
    if (checkImpossibleOccPair(occ1, occ2) || checkAlreadySynced(occ1, occ2) ||
        skipMMad1DecomposedLoopOpt(occ1, occ2) ||
        checkSkipParallelLoop(occ1, occ2) ||
        checkSkipCrossCorePair(occ1, occ2)) {
      continue;
    }
    DEBUG_WITH_TYPE("gss-sync-solver-checking", {
      llvm::dbgs() << "checking: " << (isUseless ? "is-useless\n" : "\n");
      llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                   << occ1->endIndex << ' ' << occ1->op->str(0, false) << '\n';
      llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                   << occ2->endIndex << ' ' << occ2->op->str(0, false) << '\n';
    });
    if (checkAlreadySyncedWithUnitFlag(occ1, occ2)) {
      continue;
    }
    processConflict(occ1, occ2, rwOp1, rwOp2, isUseless);
  }
}

void Solver::insertMergedBackwardSyncPairs() {
  for (auto &[scopeOp, st] : backwardSyncEventsAfterMerge) {
    for (auto &corePipeInfoPair : st) {
      auto [corePipeSrc, corePipeDst] = corePipeInfoPair;
      for (auto *scopeOcc : opAllOccurrences[scopeOp]) {
        auto *parentScopeOcc = scopeOcc->parentOcc;
        assert(parentScopeOcc != nullptr);
        Occurrence *setOcc = nullptr;
        Occurrence *waitOcc = nullptr;
        auto startIndex = scopeOcc->startIndex;
        auto endIndex = scopeOcc->endIndex;
        if (isa<Loop>(scopeOp)) {
          setOcc = getBeforePlaceHolderOcc(scopeOcc);
          waitOcc = getAfterPlaceHolderOcc(scopeOcc);
          startIndex = setOcc->endIndex;
          endIndex = waitOcc->startIndex;
        }
        auto conflictPair = std::make_unique<ConflictPair>(
            nullptr, nullptr, nullptr, nullptr, setOcc, waitOcc, corePipeSrc,
            corePipeDst, startIndex, endIndex);
        assert(conflictPair->startIndex <= conflictPair->endIndex);
        conflictPair->isUseless = true;
        conflictPair->dontReuse = true;
        conflictPair->dontCheckForConflict = true;
        conflictPair->couldNotRun = false; // notice this
        LLVM_DEBUG({
          llvm::dbgs() << "consider-merged-backward-pair: "
                       << scopeOp->str(0, false) << ' ' << conflictPair->str()
                       << "\n";
        });
        scopeOccChosenConflicts[parentScopeOcc].insert(conflictPair.get());
        chosenConflictedPairs.push_back(std::move(conflictPair));
      }
    }
  }
}

llvm::LogicalResult Solver::considerOuterBackwardSyncPairs() {
  if (!options.considerOuterBackwardSyncPairs) {
    return llvm::failure();
  }
  bool backwardPairsPositionChanged = false;
  for (auto &[scopeOp, st] : backwardSyncEventsAfterMerge) {
    SmallVector<std::tuple<CorePipeInfo, CorePipeInfo>> toBeErased;
    for (auto &corePipeInfoPair : st) {
      if (!backwardSyncEvents.contains(scopeOp) ||
          !backwardSyncEvents[scopeOp].contains(corePipeInfoPair)) {
        toBeErased.push_back(corePipeInfoPair);
      }
    }
    if (!toBeErased.empty()) {
      backwardPairsPositionChanged = true;
      for (auto &corePipeInfoPair : toBeErased) {
        st.erase(corePipeInfoPair);
      }
    }
  }
  int chosenOpsDepth = -1;
  SmallVector<OperationBase *> chosenOps;
  for (auto &[scopeOp, mp] : backwardSyncEvents) {
    if (backwardSyncEventsAfterMerge.contains(scopeOp)) {
      continue;
    }
    int scopeOpDepth = scopeOp->getDepth();
    if (chosenOpsDepth == scopeOpDepth) {
      chosenOps.push_back(scopeOp);
    } else if (chosenOpsDepth == -1 || chosenOpsDepth < scopeOpDepth) {
      chosenOps.clear();
      chosenOps.push_back(scopeOp);
      chosenOpsDepth = scopeOpDepth;
    }
  }
  if (chosenOps.empty()) {
    return llvm::failure();
  }
  bool newPairIsInserted = false;
  for (auto *chosenOp : chosenOps) {
    for (auto &[corePipeInfoPair, eventIdsMp] : backwardSyncEvents[chosenOp]) {
      assert(!eventIdsMp.empty());
      if (!eventIdsMp.empty()) {
        auto [it, isInserted] =
            backwardSyncEventsAfterMerge[chosenOp].insert(corePipeInfoPair);
        newPairIsInserted |= isInserted;
      }
    }
  }
  return llvm::success(backwardPairsPositionChanged || newPairIsInserted);
}

llvm::LogicalResult Solver::reuseSyncPairToSaveEventIds() {
  if (!options.reuseSyncPairToSaveEventIds || barrierAllPairs.empty()) {
    return llvm::failure();
  }
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
  return llvm::success(!limitReached);
}

llvm::LogicalResult Solver::disableMultiEventIdForBarrierAllPairs() {
  if (!options.disableMultiEventIdForBarrierAllPairs ||
      barrierAllPairs.empty()) {
    return llvm::failure();
  }
  bool newPairIsInserted = false;
  for (auto corePipeInfoPair : barrierAllPairs) {
    auto [it, isInserted] = disabledMultiEventIdPairs.insert(corePipeInfoPair);
    newPairIsInserted |= isInserted;
  }
  LLVM_DEBUG({
    if (newPairIsInserted) {
      llvm::dbgs() << "disabled-multi-event-id-pairs: \n";
      for (auto &[corePipeSrc, corePipeDst] : disabledMultiEventIdPairs) {
        llvm::dbgs() << corePipeSrc.coreType << ' ' << corePipeSrc.pipe << ' '
                     << corePipeDst.coreType << ' ' << corePipeDst.pipe << '\n';
      }
    }
  });
  return llvm::success(newPairIsInserted);
}

// High-level solve orchestration with multiple passes and optional merging
// iterations.
llvm::LogicalResult Solver::runSolver(bool enableOpts1, bool enableOpts2) {
  reset(/*resetEventIdRanOutOpts=*/true);

  int64_t runNum = 0;
  while (runNum++ < maxRunNum) {
    LLVM_DEBUG(llvm::dbgs() << "runNum: " << runNum << '\n');

    reset();
    insertMergedBackwardSyncPairs();
    processOrders();

    if (enableOpts1) {
      if (options.considerOuterBackwardSyncPairs) {
        getBeforeAfterSyncMaps();
        if (llvm::succeeded(considerOuterBackwardSyncPairs())) {
          continue;
        }
        if (!barrierAllPairs.empty()) {
          backwardSyncEventsAfterMerge.clear();
        }
      }
    }
    if (enableOpts2) {
      if (!barrierAllPairs.empty()) {
        if (llvm::succeeded(reuseSyncPairToSaveEventIds())) {
          continue;
        }
        if (llvm::succeeded(disableMultiEventIdForBarrierAllPairs())) {
          continue;
        }
      }
    }

    if (!barrierAllPairs.empty()) {
      pickAndInsertABarrierAll();
      reset(/*resetEventIdRanOutOpts=*/true);
      continue;
    }
    break;
  }

  reset();
  insertMergedBackwardSyncPairs();
  processOrders();

  return llvm::success(runNum < maxRunNum);
}

void Solver::solve() {
  if (llvm::succeeded(runSolver())) {
    return;
  }
  if (!options.isTestMode()) {
    if (llvm::succeeded(runSolver(/*enableOpts1=*/false))) {
      return;
    }
    if (llvm::succeeded(
            runSolver(/*enableOpts1=*/false, /*enableOpts2=*/false))) {
      return;
    }
  }
  llvm_unreachable("GSS: runSolver() failed.");
}
