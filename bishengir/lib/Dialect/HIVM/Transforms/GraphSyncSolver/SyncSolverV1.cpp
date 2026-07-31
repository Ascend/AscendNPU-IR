#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "hivm-gss-solver"

using namespace mlir;
using namespace hivm::syncsolver;

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
