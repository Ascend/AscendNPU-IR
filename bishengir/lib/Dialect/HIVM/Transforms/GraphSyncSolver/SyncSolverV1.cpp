#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "hivm-gss-solver"

using namespace mlir;
using namespace hivm::syncsolver;

bool SyncSolverV1::insertConflictPair(
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
    if (parOcc1 == parOcc2) {
      persistentScopeOccChosenConflicts[parOcc1].insert(conflictPair.get());
    } else {
      persistentScopeOccPairChosenConflicts[{parOcc1, parOcc2}].insert(
          conflictPair.get());
    }
    persistentChosenConflictedPairs.push_back(std::move(conflictPair));
    return true;
  } else {
    if (parOcc1 == parOcc2) {
      scopeOccChosenConflicts[parOcc1].insert(conflictPair.get());
    } else {
      scopeOccPairChosenConflicts[{parOcc1, parOcc2}].insert(
          conflictPair.get());
    }
    chosenConflictedPairs.push_back(std::move(conflictPair));
    return true;
  }
}

bool SyncSolverV1::eraseConflictPair(ConflictPair *conflictPair) {
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
    }
    auto it = findUniquePtr(chosenConflictedPairs, conflictPair);
    assert(it != chosenConflictedPairs.end());
    erasedChosenConflictedPairs.push_back(std::move(*it));
    chosenConflictedPairs.erase(it);
  }
  return true;
}

bool SyncSolverV1::insertTempConflictPair(ConflictPair *conflictPair,
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

bool SyncSolverV1::checkGraphConflict(
    Occurrence *occ1, Occurrence *occ2, CorePipeInfo corePipeSrc,
    CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
    std::optional<int64_t> startIndex, std::optional<int64_t> endIndex,
    const llvm::SmallVector<ConflictPair *> &extraConflictPairs,
    const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (!startIndex.has_value()) {
    startIndex = occ1->endIndex;
  }
  if (!endIndex.has_value()) {
    endIndex = occ2->startIndex;
  }

  std::unique_ptr<GraphSolverBase> graphSolver;
  if (options.enableUnitFlagFeature) {
    graphSolver = std::make_unique<GraphSolverUnitFlag>(options);
  } else {
    graphSolver = std::make_unique<GraphSolver>(options);
  }

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
      if (eventIdInfo.getEventIdNum() <
          conflictPair->eventIdInfo.getEventIdNum()) {
        return;
      }
    }
    if (eventIdInfo.cvPreloadingInfo) {
      if (conflictPair->isBackwardPair) {
        if (!conflictPair->setWaitPairInfo.has_value() ||
            !conflictPair->setWaitPairInfo->isCVPreloading) {
          return;
        }
      }
    }
    if (conflictPair->setWaitPairInfo.has_value() &&
        conflictPair->setWaitPairInfo->isCVPreloading) {
      if (!eventIdInfo.cvPreloadingInfo) {
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
    graphSolver->insertConflictPair(conflictPair);
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
  for (auto [parOcc1, parOcc2, conflictPair] : tempInsertedConflictPairs) {
    if (parOcc1 == parOcc2) {
      if (parOcc1->isAncestor(occ1) || parOcc1->isAncestor(occ2)) {
        handleConflictPair(conflictPair);
      }
      continue;
    }
    if ((parOcc1->isAncestor(occ1) && parOcc2->isAncestor(occ2)) ||
        (parOcc1->isAncestor(occ2) && parOcc2->isAncestor(occ1))) {
      handleConflictPair(conflictPair);
    }
  }
  for (auto *conflictPair : extraConflictPairs) {
    handleConflictPair(conflictPair);
  }

  this->perfInfo.graphConflictPairsCheckedNum += 1;
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

bool SyncSolverV1::checkCrossCoreIntersect(ConflictPair *conflictPair1,
                                           ConflictPair *conflictPair2) {
  if (conflictPair1->isBarrier() || conflictPair2->isBarrier()) {
    return false;
  }
  if (conflictPair1->setCorePipeInfo.coreType !=
      conflictPair2->setCorePipeInfo.coreType) {
    return false;
  }
  if (conflictPair1->startIndex > conflictPair2->startIndex) {
    std::swap(conflictPair1, conflictPair2);
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
    conflictPair1->startIndex += 1;
    assert(occ1 != nullptr && occ2 != nullptr);
    result = result ||
             checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                conflictPair1->eventIdInfo, startIndex,
                                endIndex, {conflictPair1}, {conflictPair2});
    conflictPair1->startIndex -= 1;
  }
  if (checkSamePipeWaitWait ||
      conflictPair1->waitCorePipeInfo != conflictPair2->waitCorePipeInfo) {
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
      llvm::dbgs() << "sync-ops-conflict-found: \n";
      llvm::dbgs() << " " << conflictPair1->str() << '\n';
      llvm::dbgs() << " " << conflictPair2->str() << '\n';
    }
  });
  return result;
}

llvm::SmallVector<EventIdNode *>
SyncSolverV1::getIntersectingEventIdNodes(ConflictPair *conflictPair) {
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

bool SyncSolverV1::skipLaterIterations(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (occ1->parentOcc != nullptr) {
    if (isa<Loop>(occ1->parentOcc->op)) {
      if (occ1->syncIrIndex < occ1->parentOcc->loopSplitIndex &&
          occ1->parentOcc->loopSplitIndex <= occ2->syncIrIndex) {
        return true;
      }
    }
  }
  if (occ2->parentOcc != nullptr) {
    if (isa<Loop>(occ2->parentOcc->op)) {
      if (occ1->syncIrIndex < occ2->parentOcc->loopSplitIndex &&
          occ2->parentOcc->loopSplitIndex <= occ2->syncIrIndex) {
        return true;
      }
    }
  }
  return false;
}

void SyncSolverV1::generateProcessingOrders(Occurrence *occ1, Occurrence *occ2,
                                            bool isUseless) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (skipLaterIterations(occ1, occ2)) {
    return;
  }
  if (isa<Scope>(occ1->op) && isa<Scope>(occ2->op)) {
    generateProcessingOrders(occ1->childOccs, occ2->childOccs, isUseless);
  }
  if (isa<RWOperation>(occ1->op) && isa<Scope>(occ2->op)) {
    generateProcessingOrders({occ1}, occ2->childOccs, isUseless);
  }
  if (isa<Scope>(occ1->op) && isa<RWOperation>(occ2->op)) {
    generateProcessingOrders(occ1->childOccs, {occ2}, isUseless);
  }
  if (auto *rwOp1 = dyn_cast<RWOperation>(occ1->op)) {
    if (auto *rwOp2 = dyn_cast<RWOperation>(occ2->op)) {
      generateProcessingOrders(rwOp1, rwOp2, occ1, occ2, isUseless);
    }
  }
}

void SyncSolverV1::generateProcessingOrders(
    const llvm::SmallVector<Occurrence *> &occs, bool isUseless) {
  int64_t occsNum = static_cast<int64_t>(occs.size());
  for (int64_t i = 0; i < occsNum; i++) {
    for (int64_t j = i - 1; j >= 0; j--) {
      generateProcessingOrders(occs[j], occs[i], isUseless);
    }
  }
}

void SyncSolverV1::generateProcessingOrders(
    const llvm::SmallVector<Occurrence *> &occs1,
    const llvm::SmallVector<Occurrence *> &occs2, bool isUseless) {
  for (auto *occ2 : occs2) {
    for (auto *occ1 : llvm::reverse(occs1)) {
      generateProcessingOrders(occ1, occ2, isUseless);
    }
  }
}

void SyncSolverV1::generateProcessingOrders(Scope *scopeOp, Occurrence *occ,
                                            bool isUseless) {
  assert(scopeOp != nullptr && occ != nullptr);
  assert(occ->op == scopeOp);
  generateProcessingOrders(occ->childOccs, isUseless);
}

void SyncSolverV1::generateProcessingOrders(Loop *loopOp, Occurrence *occ,
                                            bool isUseless) {
  assert(loopOp != nullptr && occ != nullptr);
  assert(occ->op == loopOp);
  assert(occ->loopSplitIndex != -1);
  int64_t childNum = static_cast<int64_t>(occ->childOccs.size());
  assert(childNum % 2 == 0);
  assert(childNum == 2 || childNum == 4);
  llvm::SmallVector<Occurrence *> firstLoopIteration(
      occ->childOccs.begin(), occ->childOccs.begin() + childNum / 2);
  llvm::SmallVector<Occurrence *> secondLoopIteration(
      occ->childOccs.begin() + childNum / 2, occ->childOccs.end());
  generateProcessingOrders(firstLoopIteration, isUseless);
  generateProcessingOrders(secondLoopIteration, true);
  for (auto *scopeOcc2 : secondLoopIteration) {
    for (auto *scopeOcc1 : llvm::reverse(firstLoopIteration)) {
      generateProcessingOrders(scopeOcc1->childOccs, scopeOcc2->childOccs,
                               isUseless);
    }
  }
}

void SyncSolverV1::generateProcessingOrders(RWOperation *rwOp1,
                                            RWOperation *rwOp2,
                                            Occurrence *occ1, Occurrence *occ2,
                                            bool isUseless) {
  assert(rwOp1 != nullptr && occ1 != nullptr);
  assert(rwOp2 != nullptr && occ2 != nullptr);
  assert(occ1->op == rwOp1);
  assert(occ2->op == rwOp2);
  processingOrders.push_back(
      ProcessingOrderV1(occ1, occ2, rwOp1, rwOp2, isUseless));
}

void SyncSolverV1::buildOfflineProcessingOrders(Occurrence *occ,
                                                bool isUseless) {
  assert(occ != nullptr);
  if (auto *loopOp = dyn_cast<Loop>(occ->op)) {
    int64_t childNum = static_cast<int64_t>(occ->childOccs.size());
    assert(childNum % 2 == 0);
    for (int64_t i = 0; i < childNum / 2; ++i) {
      buildOfflineProcessingOrders(occ->childOccs[i], isUseless);
    }
    for (int64_t i = childNum / 2; i < childNum; ++i) {
      buildOfflineProcessingOrders(occ->childOccs[i], true);
    }
    generateProcessingOrders(loopOp, occ, isUseless);
  } else if (auto *scopeOp = dyn_cast<Scope>(occ->op)) {
    for (auto *childOcc : occ->childOccs) {
      buildOfflineProcessingOrders(childOcc, isUseless);
    }
    generateProcessingOrders(scopeOp, occ, isUseless);
  } else {
    for (auto *childOcc : occ->childOccs) {
      buildOfflineProcessingOrders(childOcc, isUseless);
    }
  }
}

ConflictPair *SyncSolverV1::handleConflict(
    Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1, RWOperation *rwOp2,
    CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst, bool isUseless) {
  this->perfInfo.handledConflictsNum += 1;
  bool isBarrier = corePipeSrc == corePipeDst;
  auto unitFlagInfo =
      isBarrier ? std::nullopt : checkUnitFlagPatterns(occ1, occ2);
  auto eventIdInfo =
      getEventIdInfo(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst);
  if (!checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst, eventIdInfo)) {
    return nullptr;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "conflict found: "
                 << "isUseless: " << isUseless
                 << " eventIdNum: " << eventIdInfo.eventIdNum << "\n";
    llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                 << occ1->endIndex << ' ' << rwOp1->str(0, false) << '\n';
    llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                 << occ2->endIndex << ' ' << rwOp2->str(0, false) << '\n';
  });
  if (isBarrier) {
    eventIdInfo.setEventIdNum(1);
    return handleBarrierConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                 eventIdInfo, isUseless);
  } else if (unitFlagInfo) {
    return handleUnitFlagConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                  unitFlagInfo.value(), isUseless);
  } else {
    return handleSetWaitConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                 eventIdInfo, isUseless);
  }
}

void SyncSolverV1::processConflict(Occurrence *occ1, Occurrence *occ2,
                                   RWOperation *rwOp1, RWOperation *rwOp2,
                                   bool isUseless) {
  this->perfInfo.conflictsProcessedNum += 1;
  for (auto [corePipeSrc, corePipeDst] : getMemoryConflicts(rwOp1, rwOp2)) {
    this->perfInfo.memoryConflictsFoundNum += 1;
    if (options.alwaysUsePipeSAsWaitingPipe) {
      corePipeDst.pipe = hivm::PIPE::PIPE_S;
    }
    handleConflict(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst,
                   isUseless);
  }
}

void SyncSolverV1::processOrder(ProcessingOrderV1 processingOrder) {
  auto [occ1, occ2, rwOp1, rwOp2, isUseless] = processingOrder;
  this->perfInfo.ordersCheckedNum += 1;
  assert(occ1 != occ2);
  assert(occ1->syncIrIndex < occ2->syncIrIndex);
  DEBUG_WITH_TYPE("gss-sync-solver-checking", {
    llvm::dbgs() << "checking: "
                 << "isUseless: " << isUseless << '\n';
    llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                 << occ1->endIndex << ' ' << occ1->op->str(0, false) << '\n';
    llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                 << occ2->endIndex << ' ' << occ2->op->str(0, false) << '\n';
  });
  if (checkVisited(occ1, occ2)) {
    assert(false && "expected to not check a pair more than once.");
    return;
  }
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
  processConflict(occ1, occ2, rwOp1, rwOp2, isUseless);
}

void SyncSolverV1::processOrders() {
  processingOrders.clear();
  if (!syncIr.empty()) {
    buildOfflineProcessingOrders(syncIr.front().get());
  }
  for (auto &processingOrder : processingOrders) {
    processOrder(processingOrder);
  }
}
