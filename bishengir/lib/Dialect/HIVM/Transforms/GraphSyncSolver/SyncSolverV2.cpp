#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/CorePipeInfo.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include <queue>
#include <tuple>

#define DEBUG_TYPE "hivm-gss-solver"

using namespace mlir;
using namespace hivm::syncsolver;

void SyncSolverV2::reset(bool resetEventIdRanOutOpts) {
  SyncSolverBase::reset(resetEventIdRanOutOpts);
  insertedConflictPairs.clear();
  erasedConflictPairs.clear();
  erasedPersistentConflictPairs.clear();
  graphSolverMap.clear();
}

bool SyncSolverV2::insertConflictPair(
    std::unique_ptr<ConflictPair> conflictPair, Occurrence *parOcc) {
  Occurrence *parOcc1 = parOcc;
  Occurrence *parOcc2 = parOcc;
  if (parOcc == nullptr) {
    assert(conflictPair->setOcc != nullptr);
    assert(conflictPair->waitOcc != nullptr);
    parOcc1 = conflictPair->parOcc1 != nullptr
                  ? conflictPair->parOcc1
                  : conflictPair->setOcc->parentOcc;
    parOcc2 = conflictPair->parOcc2 != nullptr
                  ? conflictPair->parOcc2
                  : conflictPair->waitOcc->parentOcc;
  }
  conflictPair->parOcc1 = parOcc1;
  conflictPair->parOcc2 = parOcc2;

  if (conflictPair->isPersistent) {
    if (parOcc1 != nullptr && parOcc2 != nullptr) {
      if (parOcc1 == parOcc2) {
        persistentScopeOccChosenConflicts[parOcc1].insert(conflictPair.get());
      } else {
        persistentScopeOccPairChosenConflicts[{parOcc1, parOcc2}].insert(
            conflictPair.get());
      }
      insertedPersistentConflictPairs.emplace_back(parOcc1, parOcc2,
                                                   conflictPair.get());
      persistentChosenConflictedPairs.push_back(std::move(conflictPair));
      return true;
    }
    persistentChosenConflictedPairs.push_back(std::move(conflictPair));
    return false;
  } else {
    if (parOcc1 != nullptr && parOcc2 != nullptr) {
      if (parOcc1 == parOcc2) {
        scopeOccChosenConflicts[parOcc1].insert(conflictPair.get());
      } else {
        scopeOccPairChosenConflicts[{parOcc1, parOcc2}].insert(
            conflictPair.get());
      }
      insertedConflictPairs.emplace_back(parOcc1, parOcc2, conflictPair.get());
      chosenConflictedPairs.push_back(std::move(conflictPair));
      return true;
    }
    chosenConflictedPairs.push_back(std::move(conflictPair));
    return false;
  }
}

bool SyncSolverV2::eraseConflictPair(ConflictPair *conflictPair) {
  Occurrence *parOcc1 = conflictPair->parOcc1;
  Occurrence *parOcc2 = conflictPair->parOcc2;
  assert(!conflictPair->isPersistent);
  conflictPair->isErased = true;
  if (conflictPair->isPersistent) {
    if (parOcc1 != nullptr) {
      assert(parOcc2 != nullptr);
      if (parOcc1 == parOcc2) {
        persistentScopeOccChosenConflicts[parOcc1].erase(conflictPair);
      } else {
        persistentScopeOccPairChosenConflicts[{parOcc1, parOcc2}].erase(
            conflictPair);
      }
      erasedPersistentConflictPairs.emplace_back(parOcc1, parOcc2,
                                                 conflictPair);
    }
    auto it = findUniquePtr(persistentChosenConflictedPairs, conflictPair);
    assert(it != persistentChosenConflictedPairs.end());
    erasedPersistentChosenConflictedPairs.push_back(std::move(*it));
    persistentChosenConflictedPairs.erase(it);
  } else {
    if (parOcc1 != nullptr) {
      assert(parOcc2 != nullptr);
      if (parOcc1 == parOcc2) {
        scopeOccChosenConflicts[parOcc1].erase(conflictPair);
      } else {
        scopeOccPairChosenConflicts[{parOcc1, parOcc2}].erase(conflictPair);
      }
      erasedConflictPairs.emplace_back(parOcc1, parOcc2, conflictPair);
    }
    auto it = findUniquePtr(chosenConflictedPairs, conflictPair);
    assert(it != chosenConflictedPairs.end());
    erasedChosenConflictedPairs.push_back(std::move(*it));
    chosenConflictedPairs.erase(it);
  }
  return true;
}

bool SyncSolverV2::insertTempConflictPair(ConflictPair *conflictPair,
                                          Occurrence *parOcc) {
  Occurrence *parOcc1 = parOcc;
  Occurrence *parOcc2 = parOcc;
  if (parOcc == nullptr) {
    assert(conflictPair->setOcc != nullptr);
    assert(conflictPair->waitOcc != nullptr);
    parOcc1 = conflictPair->parOcc1 != nullptr
                  ? conflictPair->parOcc1
                  : conflictPair->setOcc->parentOcc;
    parOcc2 = conflictPair->parOcc2 != nullptr
                  ? conflictPair->parOcc2
                  : conflictPair->waitOcc->parentOcc;
  }
  conflictPair->parOcc1 = parOcc1;
  conflictPair->parOcc2 = parOcc2;
  if (parOcc1 != nullptr && parOcc2 != nullptr) {
    tempInsertedConflictPairs.emplace_back(parOcc1, parOcc2, conflictPair);
    return true;
  }
  return false;
}

std::unique_ptr<GraphSolverBase> &
SyncSolverV2::getGraphSolverRef(Occurrence *occ1, Occurrence *occ2,
                                const EventIdInfo &eventIdInfo) {
  int64_t eventIdNum = eventIdInfo.getEventIdNum();
  int16_t isCVPreloading = eventIdInfo.cvPreloadingInfo.has_value();
  auto key = std::make_tuple(occ1, occ2, eventIdNum, isCVPreloading);
  if (!graphSolverMap.contains(key)) {
    GraphSolverInfo graphSolverInfo;
    if (options.enableUnitFlagFeature) {
      graphSolverInfo.graphSolver =
          std::make_unique<GraphSolverUnitFlag>(options);
    } else {
      graphSolverInfo.graphSolver = std::make_unique<GraphSolver>(options);
    }
    graphSolverMap[key] = std::move(graphSolverInfo);
  }

  auto &graphSolverInfo = graphSolverMap[key];
  auto handleConflictPair = [&](Occurrence *parOcc1, Occurrence *parOcc2,
                                ConflictPair *conflictPair, bool isTemp = false,
                                bool isErase = false) {
    assert(!isErase || conflictPair->isErased);
    if (conflictPair->couldNotRun) {
      return;
    }
    if (eventIdNum < conflictPair->eventIdInfo.getEventIdNum()) {
      return;
    }
    if (isCVPreloading) {
      if (conflictPair->isBackwardPair) {
        if (!conflictPair->setWaitPairInfo.has_value() ||
            !conflictPair->setWaitPairInfo->isCVPreloading) {
          return;
        }
      }
    }
    if (conflictPair->setWaitPairInfo.has_value() &&
        conflictPair->setWaitPairInfo->isCVPreloading) {
      if (!isCVPreloading) {
        return;
      }
    }
    if (parOcc1 == parOcc2) {
      if (!parOcc1->isAncestor(occ1) && !parOcc1->isAncestor(occ2)) {
        return;
      }
    } else if ((!parOcc1->isAncestor(occ1) || !parOcc2->isAncestor(occ2)) &&
               (!parOcc1->isAncestor(occ2) || !parOcc2->isAncestor(occ1))) {
      return;
    }
    if (isErase) {
      graphSolverInfo.graphSolver->eraseConflictPair(conflictPair, isTemp);
    } else {
      graphSolverInfo.graphSolver->insertConflictPair(conflictPair, isTemp);
    }
  };

  graphSolverInfo.graphSolver->clearAdjList(/*isTemp=*/true);
  for (auto [parOcc1, parOcc2, conflictPair] : tempInsertedConflictPairs) {
    handleConflictPair(parOcc1, parOcc2, conflictPair, /*isTemp=*/true);
  }
  for (size_t i = graphSolverInfo.insertedConflictPairsIndex;
       i < insertedConflictPairs.size(); ++i) {
    auto [parOcc1, parOcc2, conflictPair] = insertedConflictPairs[i];
    handleConflictPair(parOcc1, parOcc2, conflictPair);
  }
  for (size_t i = graphSolverInfo.insertedPersistentConflictPairsIndex;
       i < insertedPersistentConflictPairs.size(); ++i) {
    auto [parOcc1, parOcc2, conflictPair] = insertedPersistentConflictPairs[i];
    handleConflictPair(parOcc1, parOcc2, conflictPair);
  }
  for (size_t i = graphSolverInfo.erasedConflictPairsIndex;
       i < erasedConflictPairs.size(); ++i) {
    auto [parOcc1, parOcc2, conflictPair] = erasedConflictPairs[i];
    handleConflictPair(parOcc1, parOcc2, conflictPair, /*isTemp=*/false,
                       /*isErase=*/true);
  }
  for (size_t i = graphSolverInfo.erasedPersistentConflictPairsIndex;
       i < erasedPersistentConflictPairs.size(); ++i) {
    auto [parOcc1, parOcc2, conflictPair] = erasedPersistentConflictPairs[i];
    handleConflictPair(parOcc1, parOcc2, conflictPair, /*isTemp=*/false,
                       /*isErase=*/true);
  }

  graphSolverInfo.insertedConflictPairsIndex = insertedConflictPairs.size();
  graphSolverInfo.insertedPersistentConflictPairsIndex =
      insertedPersistentConflictPairs.size();
  graphSolverInfo.erasedConflictPairsIndex = erasedConflictPairs.size();
  graphSolverInfo.erasedPersistentConflictPairsIndex =
      erasedPersistentConflictPairs.size();
  return graphSolverInfo.graphSolver;
}

bool SyncSolverV2::checkGraphConflict(
    Occurrence *occ1, Occurrence *occ2, CorePipeInfo corePipeSrc,
    CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
    std::optional<int64_t> startIndex, std::optional<int64_t> endIndex,
    const llvm::SmallVector<ConflictPair *> &,
    const llvm::SmallVector<ConflictPair *> &) {
  assert(occ1 != nullptr && occ2 != nullptr);
  this->perfInfo.graphConflictPairsCheckedNum += 1;

  if (!startIndex.has_value()) {
    startIndex = occ1->endIndex;
  }
  if (!endIndex.has_value()) {
    endIndex = occ2->startIndex;
  }

  auto &graphSolver =
      getGraphSolverRef(occ1->parentOcc, occ2->parentOcc, eventIdInfo);

  if (graphSolver->checkAnyBarrierAllBetween(startIndex.value(),
                                             endIndex.value())) {
    this->perfInfo.checkGraphConflictSkipDijNum += 1;
    return false;
  } else if (corePipeSrc == corePipeDst &&
             graphSolver->checkAnyBarrierBetween(
                 corePipeSrc, startIndex.value(), endIndex.value())) {
    this->perfInfo.checkGraphConflictSkipDijNum += 1;
    return false;
  } else {
    auto minDistance =
        graphSolver->runDijkstra(corePipeSrc, corePipeDst, startIndex.value(),
                                 endIndex.value(), occ1, occ2);
    return !minDistance.has_value() || minDistance.value() > endIndex.value();
  }
}

bool SyncSolverV2::checkCrossCoreIntersect(ConflictPair *conflictPair1,
                                           ConflictPair *conflictPair2) {
  if (conflictPair1->isBarrier() || conflictPair2->isBarrier()) {
    return false;
  }
  if (conflictPair1->startIndex > conflictPair2->startIndex) {
    std::swap(conflictPair1, conflictPair2);
  }
  if (conflictPair1->setCorePipeInfo.coreType !=
      conflictPair2->setCorePipeInfo.coreType) {
    return false;
  }
  if (conflictPair1->startIndex >= conflictPair2->startIndex ||
      conflictPair1->endIndex >= conflictPair2->endIndex) {
    return true;
  }

  auto *setOcc1 = conflictPair1->setOcc;
  auto *waitOcc1 = conflictPair1->waitOcc;
  auto *setOcc2 = conflictPair2->setOcc;
  auto *waitOcc2 = conflictPair2->waitOcc;

  bool checkSamePipeSetSet = false;
  if (conflictPair1->setCorePipeInfo == conflictPair2->setCorePipeInfo) {
    auto *parentLoopOp1 = setOcc1->op->getParentOfType<Loop>();
    auto *parentLoopOp2 = setOcc2->op->getParentOfType<Loop>();
    if (parentLoopOp1 && !parentLoopOp1->isProperAncestor(waitOcc1->op)) {
      if (parentLoopOp1->isProperAncestor(setOcc2->op)) {
        checkSamePipeSetSet = true;
      }
    }
    if (parentLoopOp2 && !parentLoopOp2->isProperAncestor(waitOcc2->op)) {
      if (parentLoopOp2->isProperAncestor(setOcc1->op)) {
        checkSamePipeSetSet = true;
      }
    }
  }

  bool checkSamePipeWaitWait = false;
  if (conflictPair1->waitCorePipeInfo == conflictPair2->waitCorePipeInfo) {
    auto *parentLoopOp1 = waitOcc1->op->getParentOfType<Loop>();
    auto *parentLoopOp2 = waitOcc2->op->getParentOfType<Loop>();
    if (parentLoopOp1 && !parentLoopOp1->isProperAncestor(setOcc1->op)) {
      if (parentLoopOp1->isProperAncestor(waitOcc2->op)) {
        checkSamePipeWaitWait = true;
      }
    }
    if (parentLoopOp2 && !parentLoopOp2->isProperAncestor(setOcc2->op)) {
      if (parentLoopOp2->isProperAncestor(waitOcc1->op)) {
        checkSamePipeWaitWait = true;
      }
    }
  }

  bool result = false;
  if (checkSamePipeSetSet ||
      conflictPair1->setCorePipeInfo != conflictPair2->setCorePipeInfo) {
    auto corePipeSrc = conflictPair1->setCorePipeInfo;
    auto corePipeDst = conflictPair2->setCorePipeInfo;
    Occurrence *occ1 = conflictPair1->setOcc;
    Occurrence *occ2 = conflictPair2->setOcc;
    auto startIndex = conflictPair1->startIndex + 1;
    auto endIndex = conflictPair2->startIndex;

    auto clonedConflictPair = conflictPair1->clone();
    clonedConflictPair->startIndex += 1;
    bool insertedTempConflictPair =
        insertTempConflictPair(clonedConflictPair.get());
    clonedConflictPair->isUseless = true;

    assert(occ1 != nullptr && occ2 != nullptr);
    result = result || checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                          conflictPair1->eventIdInfo,
                                          startIndex, endIndex);

    if (insertedTempConflictPair) {
      tempInsertedConflictPairs.pop_back();
    }
  }
  if (checkSamePipeWaitWait ||
      conflictPair1->waitCorePipeInfo != conflictPair2->waitCorePipeInfo) {
    auto corePipeSrc = conflictPair1->waitCorePipeInfo;
    auto corePipeDst = conflictPair2->waitCorePipeInfo;
    Occurrence *occ1 = conflictPair1->waitOcc;
    Occurrence *occ2 = conflictPair2->waitOcc;
    auto startIndex = conflictPair1->endIndex;
    auto endIndex = conflictPair2->endIndex - 1;
    assert(occ1 != nullptr && occ2 != nullptr);
    result = result || checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                          conflictPair1->eventIdInfo,
                                          startIndex, endIndex);
  }
  DEBUG_WITH_TYPE("gss-check-sync-ops-conflicts", {
    if (result) {
      llvm::dbgs() << "sync-ops-conflict-found: \n";
      llvm::dbgs() << " " << conflictPair1->str() << '\n';
      llvm::dbgs() << " " << conflictPair2->str() << '\n';
    }
  });
  return result;
}

llvm::SmallVector<EventIdNode *>
SyncSolverV2::getIntersectingEventIdNodes(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  if (conflictPair->isBarrier()) {
    return {};
  }
  if (conflictPair->dontCheckForConflict) {
    return {};
  }
  llvm::SetVector<EventIdNode *> intersectingNodes;
  for (auto &curConflictPair : chosenConflictedPairs) {
    if (!intersectingNodes.contains(curConflictPair->eventIdNode)) {
      if (checkIntersect(conflictPair, curConflictPair.get())) {
        intersectingNodes.insert(curConflictPair->eventIdNode);
      }
    }
  }
  for (auto &curConflictPair : persistentChosenConflictedPairs) {
    if (!intersectingNodes.contains(curConflictPair->eventIdNode)) {
      if (checkIntersect(conflictPair, curConflictPair.get())) {
        intersectingNodes.insert(curConflictPair->eventIdNode);
      }
    }
  }
  return intersectingNodes.takeVector();
}

ConflictPair *SyncSolverV2::handleConflict(
    Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1, RWOperation *rwOp2,
    CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
    SetWaitPairInfo setWaitPairInfo, bool isUseless) {
  this->perfInfo.handledConflictsNum += 1;
  LLVM_DEBUG({
    llvm::dbgs() << "conflict found: "
                 << "isUseless: " << isUseless
                 << " eventIdNum: " << eventIdInfo.eventIdNum << "\n";
    llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                 << occ1->endIndex << ' ' << rwOp1->str(0, false) << '\n';
    llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                 << occ2->endIndex << ' ' << rwOp2->str(0, false) << '\n';
  });
  if (corePipeSrc == corePipeDst) {
    return handleBarrierConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                 eventIdInfo, isUseless);
  } else if (auto unitFlagInfo = checkUnitFlagPatterns(occ1, occ2)) {
    return handleUnitFlagConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                  unitFlagInfo.value(), isUseless);
  } else {
    return handleSetWaitConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                 eventIdInfo, setWaitPairInfo, isUseless);
  }
}

void SyncSolverV2::processOrder(Occurrence *occ1, Occurrence *occ2,
                                RWOperation *rwOp1, RWOperation *rwOp2,
                                bool isUseless) {
  assert(occ1 != occ2);
  assert(occ1->syncIrIndex < occ2->syncIrIndex);
  this->perfInfo.ordersCheckedNum += 1;

  DEBUG_WITH_TYPE("gss-sync-solver-checking", {
    llvm::dbgs() << "checking: " << (isUseless ? "is-useless\n" : "\n");
    llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                 << occ1->endIndex << ' ' << occ1->op->str(0, false) << '\n';
    llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                 << occ2->endIndex << ' ' << occ2->op->str(0, false) << '\n';
  });
  if (checkImpossibleOccPair(occ1, occ2) || checkAlreadySynced(occ1, occ2) ||
      skipMMad1DecomposedLoopOpt(occ1, occ2) ||
      checkSkipParallelLoop(occ1, occ2) || checkSkipCrossCorePair(occ1, occ2)) {
    this->perfInfo.failedInitialChecksNum += 1;
    return;
  }
  if (checkAlreadySyncedWithUnitFlag(occ1, occ2)) {
    this->perfInfo.failedInitialChecksNum += 1;
    return;
  }

  this->perfInfo.conflictsProcessedNum += 1;
  for (auto [corePipeSrc, corePipeDst] : getMemoryConflicts(rwOp1, rwOp2)) {
    this->perfInfo.memoryConflictsFoundNum += 1;
    auto [corePipeInfo1, corePipeInfo2] =
        getFixedCorePipeInfoPair(corePipeSrc, corePipeDst);
    auto [eventIdInfo, setWaitPairInfo] = getEventIdSetWaitPairInfo(
        occ1, occ2, rwOp1, rwOp2, corePipeInfo1, corePipeInfo2);
    if (checkGraphConflict(occ1, occ2, corePipeInfo1, corePipeInfo2,
                           eventIdInfo)) {
      handleConflict(occ1, occ2, rwOp1, rwOp2, corePipeInfo1, corePipeInfo2,
                     eventIdInfo, setWaitPairInfo, isUseless);
    }
  }
}

bool SyncSolverV2::processOrder(ProcessingOrderV2 processingOrder) {
  auto [lcaOcc1, lcaOcc2, occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst,
        isUseless] = processingOrder;

  assert(occ1 != occ2);
  assert(occ1->syncIrIndex < occ2->syncIrIndex);
  assert(lcaOcc1 != lcaOcc2);
  assert(!lcaOcc2->isProperAncestor(lcaOcc1));
  this->perfInfo.ordersCheckedNum += 1;

  DEBUG_WITH_TYPE("gss-sync-solver-checking", {
    llvm::dbgs() << "checking: " << (isUseless ? "is-useless\n" : "\n");
    llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                 << occ1->endIndex << ' ' << occ1->op->str(0, false) << '\n';
    llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                 << occ2->endIndex << ' ' << occ2->op->str(0, false) << '\n';
  });
  if (checkImpossibleOccPair(occ1, occ2) || checkAlreadySynced(occ1, occ2) ||
      skipMMad1DecomposedLoopOpt(occ1, occ2) ||
      checkSkipParallelLoop(occ1, occ2)) {
    this->perfInfo.failedInitialChecksNum += 1;
    return false;
  }
  if (checkAlreadySyncedWithUnitFlag(occ1, occ2)) {
    this->perfInfo.failedInitialChecksNum += 1;
    return false;
  }

  bool lcaOcc1IsAncestorLcaOcc2 = lcaOcc1->isProperAncestor(lcaOcc2);
  bool lcaOcc2IsAncestorLcaOcc1 = lcaOcc2->isProperAncestor(lcaOcc1);
  assert(!lcaOcc1IsAncestorLcaOcc2 || !lcaOcc2IsAncestorLcaOcc1);

  Occurrence *parOcc1 = lcaOcc1;
  Occurrence *parOcc2 = lcaOcc2;
  if (lcaOcc1IsAncestorLcaOcc2) {
    assert(parOcc1->isProperAncestor(occ1));
    parOcc1 = occ1->getNthParent(occ1->depth - lcaOcc1->depth - 1);
    assert(parOcc1 != nullptr);
    assert(parOcc1->parentOcc == lcaOcc1);
  } else if (lcaOcc2IsAncestorLcaOcc1) {
    assert(parOcc2->isProperAncestor(occ2));
    parOcc2 = occ2->getNthParent(occ2->depth - lcaOcc2->depth - 1);
    assert(parOcc2 != nullptr);
    assert(parOcc2->parentOcc == lcaOcc2);
  }

  this->perfInfo.conflictsProcessedNum += 1;
  this->perfInfo.memoryConflictsFoundNum += 1;
  auto [eventIdInfo, setWaitPairInfo] = getEventIdSetWaitPairInfo(
      occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst);
  if (!checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst, eventIdInfo)) {
    if (!checkGraphConflict(parOcc1, parOcc2, corePipeSrc, corePipeDst,
                            EventIdInfo(1))) {
      return true;
    }
    return false;
  }

  auto *conflictPair =
      handleConflict(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst,
                     eventIdInfo, setWaitPairInfo, isUseless);
  if (conflictPair == nullptr || conflictPair->couldNotRun ||
      conflictPair->setOnLastIterOnly || conflictPair->waitOnFirstIterOnly) {
    return false;
  }
  if (conflictPair->isBarrier()) {
    return !parOcc1->isProperAncestor(conflictPair->waitOcc) &&
           !parOcc2->isProperAncestor(conflictPair->waitOcc);
  }
  if (conflictPair->eventIdInfo.getEventIdNum() <= 1) {
    return (!parOcc1->isProperAncestor(conflictPair->setOcc) &&
            !parOcc2->isProperAncestor(conflictPair->setOcc)) ||
           (!parOcc1->isProperAncestor(conflictPair->waitOcc) &&
            !parOcc2->isProperAncestor(conflictPair->waitOcc));
  }
  return false;
}

void SyncSolverV2::generateProcessingOrders(
    Occurrence *lcaOcc1, Occurrence *lcaOcc2,
    const llvm::ArrayRef<Occurrence *> &occs1,
    const llvm::ArrayRef<Occurrence *> &occs2, bool isUseless,
    bool occs1IsLcaOcc1, bool occs2IsLcaOcc2) {
  if (occs1.empty() || occs2.empty()) {
    return;
  }
  if (occs1.size() == 1) {
    lcaOcc1 = occs1.front();
    occs1IsLcaOcc1 = true;
  }
  if (occs2.size() == 1) {
    lcaOcc2 = occs2.front();
    occs2IsLcaOcc2 = true;
  }

  int64_t lIndex1 = occs1.front()->syncIrIndex;
  int64_t rIndex1 = occs1.back()->syncIrEndIndex;
  int64_t lIndex2 = occs2.front()->syncIrIndex;
  int64_t rIndex2 = occs2.back()->syncIrEndIndex;

  DEBUG_WITH_TYPE("hivm-gss-orders", {
    auto printRange = [&](int64_t l, int64_t r) {
      int64_t rEnd = r - 1;
      llvm::dbgs() << '(' << l;
      if (l != rEnd) {
        llvm::dbgs() << ", " << rEnd;
      }
      llvm::dbgs() << ')';
    };
    llvm::dbgs() << "handlingOccPair: ";
    llvm::dbgs() << lcaOcc1->op->str(0, false);
    printRange(lIndex1, rIndex1);
    llvm::dbgs() << " - ";
    llvm::dbgs() << lcaOcc2->op->str(0, false);
    printRange(lIndex2, rIndex2);
    llvm::dbgs() << " # " << occs1IsLcaOcc1 << ' ' << occs2IsLcaOcc2 << '\n';
  });

  struct QueueElement {
    MemInfoNode *node1{nullptr};
    MemInfoNode *node2{nullptr};
    CorePipeInfo corePipeInfo1;
    CorePipeInfo corePipeInfo2;
    MemInfoOccElement occElement1;
    MemInfoOccElement occElement2;
    MemInfoOccElementList::iterator occElementIt1;
    MemInfoOccElementList::iterator occElementIt2;
    bool runSecondPath{true};

    QueueElement(MemInfoNode *node1, MemInfoNode *node2,
                 CorePipeInfo corePipeInfo1, CorePipeInfo corePipeInfo2,
                 MemInfoOccElement occElement1, MemInfoOccElement occElement2,
                 MemInfoOccElementList::iterator occElementIt1,
                 MemInfoOccElementList::iterator occElementIt2)
        : node1(node1), node2(node2), corePipeInfo1(corePipeInfo1),
          corePipeInfo2(corePipeInfo2), occElement1(occElement1),
          occElement2(occElement2), occElementIt1(occElementIt1),
          occElementIt2(occElementIt2) {}
  };

  auto queueElementCmp = [&](const QueueElement &a,
                             const QueueElement &b) -> bool {
    if (!occs2IsLcaOcc2 &&
        a.occElement2.parentOccIndex != b.occElement2.parentOccIndex) {
      return a.occElement2.parentOccIndex > b.occElement2.parentOccIndex;
    }
    if (!occs1IsLcaOcc1 &&
        a.occElement1.parentOccIndex != b.occElement1.parentOccIndex) {
      return a.occElement1.parentOccIndex < b.occElement1.parentOccIndex;
    }
    if (occs1IsLcaOcc1 || !occs2IsLcaOcc2) {
      if (a.occElement2.occIndex != b.occElement2.occIndex) {
        return a.occElement2.occIndex > b.occElement2.occIndex;
      }
      if (a.occElement1.occIndex != b.occElement1.occIndex) {
        return a.occElement1.occIndex < b.occElement1.occIndex;
      }
    } else {
      if (a.occElement1.occIndex != b.occElement1.occIndex) {
        return a.occElement1.occIndex < b.occElement1.occIndex;
      }
      if (a.occElement2.occIndex != b.occElement2.occIndex) {
        return a.occElement2.occIndex > b.occElement2.occIndex;
      }
    }
    return false;
  };

  std::priority_queue<QueueElement, std::vector<QueueElement>,
                      decltype(queueElementCmp)>
      queue(queueElementCmp);

  for (auto &[memoryEffect1, map1] : lcaOcc1->memInfoTree1.nodeListMap) {
    for (auto &[corePipeSrc, nodeList1] : map1) {
      for (auto &[memoryEffect2, map2] : lcaOcc2->memInfoTree2.nodeListMap) {
        if (memoryEffect1 == MemoryEffect::READ &&
            memoryEffect2 == MemoryEffect::READ) {
          continue;
        }
        for (auto &[corePipeDst, nodeList2] : map2) {
          if (checkSkipIntraCorePair(corePipeSrc.pipe, corePipeDst.pipe) ||
              checkSkipCrossCorePair(corePipeSrc.coreType,
                                     corePipeDst.coreType)) {
            continue;
          }
          auto [corePipeInfo1, corePipeInfo2] =
              getFixedCorePipeInfoPair(corePipeSrc, corePipeDst);

          for (auto &node1 : nodeList1) {
            auto *it1 = node1.lower_bound(
                MemInfoOccElement(nullptr, nullptr, rIndex1, -1));
            if (it1 == node1.occElements.begin()) {
              continue;
            }
            it1 = std::prev(it1);
            if (it1->occIndex < lIndex1) {
              continue;
            }

            for (auto &node2 : nodeList2) {
              if (!checkMemInfoConflict(/*rwOp1=*/nullptr, /*rwOp2=*/nullptr,
                                        node1.rootMemInfo, node2.rootMemInfo)) {
                continue;
              }

              auto *it2 = node2.lower_bound(
                  MemInfoOccElement(nullptr, nullptr, lIndex2, -1));
              if (it2 == node2.occElements.end() || it2->occIndex >= rIndex2) {
                continue;
              }
              perfInfo.priorityQueuePushNum += 1;
              queue.emplace(&node1, &node2, corePipeInfo1, corePipeInfo2, *it1,
                            *it2, it1, it2);
            }
          }
        }
      }
    }
  }

  auto handle = [&](const QueueElement &queueElement) -> bool {
    auto corePipeSrc = queueElement.corePipeInfo1;
    auto *occ1 = queueElement.occElement1.occ;
    auto *rwOp1 = dyn_cast<RWOperation>(occ1->op);
    assert(rwOp1 != nullptr);

    auto corePipeDst = queueElement.corePipeInfo2;
    auto *occ2 = queueElement.occElement2.occ;
    auto *rwOp2 = dyn_cast<RWOperation>(occ2->op);
    assert(rwOp2 != nullptr);

    DEBUG_WITH_TYPE("hivm-gss-orders", {
      llvm::dbgs() << "handling: " << queueElement.occElement1.occIndex << ' '
                   << queueElement.occElement2.occIndex << '\n'
                   << "  " << queueElement.node1->rootMemInfo.str() << ' '
                   << queueElement.node2->rootMemInfo.str() << '\n';
    });

    ProcessingOrderV2 processingOrder(lcaOcc1, lcaOcc2, occ1, occ2, rwOp1,
                                      rwOp2, corePipeSrc, corePipeDst,
                                      isUseless);
    if (!processOrder(processingOrder)) {
      return false;
    }

    DEBUG_WITH_TYPE("hivm-gss-orders", {
      llvm::dbgs() << "skipped: "
                   << "[<" << stringifyTCoreType(corePipeSrc.coreType).str()
                   << ">, <" << stringifyPIPE(corePipeSrc.pipe).str() << ">] "
                   << "[<" << stringifyTCoreType(corePipeDst.coreType).str()
                   << ">, <" << stringifyPIPE(corePipeDst.pipe).str() << ">]\n";
    });
    return true;
  };

  llvm::DenseSet<std::tuple<CorePipeInfo, CorePipeInfo>> handledCorePipePairs;
  while (!queue.empty()) {
    auto current = queue.top();
    queue.pop();

    auto key = std::make_tuple(current.corePipeInfo1, current.corePipeInfo2);
    if (handledCorePipePairs.contains(key)) {
      continue;
    }

    if (handle(current)) {
      perfInfo.solverSkipNum += 1;
      handledCorePipePairs.insert(key);
      continue;
    }

    if (current.occElementIt1 != current.node1->occElements.begin()) {
      auto *nextIt1 = std::prev(current.occElementIt1);
      if (nextIt1->occIndex >= lIndex1) {
        auto next = current;
        next.occElement1 = *nextIt1;
        next.occElementIt1 = nextIt1;
        next.runSecondPath = false;
        perfInfo.priorityQueuePushNum += 1;
        queue.push(next);
      }
    }

    if (current.runSecondPath) {
      auto *nextIt2 = std::next(current.occElementIt2);
      if (nextIt2 != current.node2->occElements.end() &&
          nextIt2->occIndex < rIndex2) {
        auto next = current;
        next.occElement2 = *nextIt2;
        next.occElementIt2 = nextIt2;
        perfInfo.priorityQueuePushNum += 1;
        queue.push(next);
      }
    }
  }
}

void SyncSolverV2::generateProcessingOrders(Occurrence *occ1, Occurrence *occ2,
                                            bool isUseless) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = dyn_cast<RWOperation>(occ1->op);
  auto *rwOp2 = dyn_cast<RWOperation>(occ2->op);
  if (rwOp1 && rwOp2) {
    processOrder(occ1, occ2, rwOp1, rwOp2, isUseless);
    return;
  }

  ArrayRef<Occurrence *> occs1;
  if (isa<Loop>(occ1->op)) {
    occs1 = occ1->getLoopSecondIterOccs();
  } else if (isa<Scope>(occ1->op)) {
    occs1 = ArrayRef(occ1->childOccs);
  } else if (isa<RWOperation>(occ1->op)) {
    occs1 = occ1;
  }

  ArrayRef<Occurrence *> occs2;
  if (isa<Loop>(occ2->op)) {
    occs2 = occ2->getLoopFirstIterOccs();
  } else if (isa<Scope>(occ2->op)) {
    occs2 = ArrayRef(occ2->childOccs);
  } else if (isa<RWOperation>(occ2->op)) {
    occs2 = occ2;
  }

  assert(!occs1.empty() || isa<PlaceHolder>(occ1->op));
  assert(!occs2.empty() || isa<PlaceHolder>(occ2->op));
  if (occs1.empty() || occs2.empty()) {
    return;
  }

  generateProcessingOrders(occ1, occ2, occs1, occs2, isUseless,
                           /*occs1IsLcaOcc1=*/true,
                           /*occs2IsLcaOcc2=*/true);
}

void SyncSolverV2::generateProcessingOrders(
    Occurrence *lcaOcc, const llvm::ArrayRef<Occurrence *> &occs,
    bool isUseless) {
  for (auto [i, occ2] : llvm::enumerate(occs)) {
    auto occs1 = occs.slice(0, i);
    generateProcessingOrders(lcaOcc, occ2, occs1, occ2, isUseless,
                             /*occs1IsLcaOcc1=*/false,
                             /*occs2IsLcaOcc2=*/true);
  }
}

void SyncSolverV2::generateProcessingOrders(Scope *scopeOp, Occurrence *occ,
                                            bool isUseless) {
  assert(scopeOp != nullptr && occ != nullptr);
  generateProcessingOrders(occ, occ->childOccs, isUseless);
}

void SyncSolverV2::generateProcessingOrders(Loop *loopOp, Occurrence *occ,
                                            bool isUseless) {
  assert(loopOp != nullptr && occ != nullptr);
  auto firstLoopIteration = occ->getLoopFirstIterOccs();
  auto secondLoopIteration = occ->getLoopSecondIterOccs();

  for (auto [i, occ2] : llvm::enumerate(firstLoopIteration)) {
    for (auto *occ1 : firstLoopIteration.slice(0, i)) {
      generateProcessingOrders(occ1, occ2, isUseless);
    }
  }
  for (auto [i, occ2] : llvm::enumerate(secondLoopIteration)) {
    for (auto *occ1 : secondLoopIteration.slice(0, i)) {
      generateProcessingOrders(occ1, occ2, true);
    }
  }
  for (auto *scopeOcc2 : secondLoopIteration) {
    for (auto *scopeOcc1 : llvm::reverse(firstLoopIteration)) {
      if (scopeOcc1->op != scopeOcc2->op) {
        generateProcessingOrders(scopeOcc1, scopeOcc2, isUseless);
        continue;
      }

      ArrayRef<Occurrence *> occs1(scopeOcc1->childOccs);
      ArrayRef<Occurrence *> occs2(scopeOcc2->childOccs);
      assert(occs1.size() == occs2.size());
      for (auto [i, occ2] : llvm::enumerate(occs2)) {
        auto *occ1 = occs1[i];
        auto preOccs = occs1.slice(0, i);
        auto sufOccs = occs1.slice(i + 1);
        generateProcessingOrders(scopeOcc1, occ2, sufOccs, occ2, isUseless,
                                 /*occs1IsLcaOcc1=*/false,
                                 /*occs2IsLcaOcc2=*/true);
        generateProcessingOrders(occ1, occ2, isUseless);
        generateProcessingOrders(scopeOcc1, occ2, preOccs, occ2, isUseless,
                                 /*occs1IsLcaOcc1=*/false,
                                 /*occs2IsLcaOcc2=*/true);
      }
    }
  }
}

void SyncSolverV2::collectProcessingOrders(Occurrence *occ, bool isUseless) {
  assert(occ != nullptr);
  if (auto *loopOp = dyn_cast<Loop>(occ->op)) {
    for (auto *scopeOcc : occ->getLoopFirstIterOccs()) {
      collectProcessingOrders(scopeOcc, isUseless);
    }
    for (auto *scopeOcc : occ->getLoopSecondIterOccs()) {
      collectProcessingOrders(scopeOcc, true);
    }
    generateProcessingOrders(loopOp, occ, isUseless);
  } else if (auto *scopeOp = dyn_cast<Scope>(occ->op)) {
    for (auto *childOcc : occ->childOccs) {
      collectProcessingOrders(childOcc, isUseless);
    }
    generateProcessingOrders(scopeOp, occ, isUseless);
  }
}

void SyncSolverV2::processOrders() {
  if (!syncIr.empty()) {
    collectProcessingOrders(syncIr.front().get());
  }
}
