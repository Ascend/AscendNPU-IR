#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <queue>

#define DEBUG_TYPE "hivm-gss-solver"

using namespace mlir;
using namespace hivm::syncsolver;

void SyncSolverV2::processOrder(Occurrence *occ1, Occurrence *occ2,
                                RWOperation *rwOp1, RWOperation *rwOp2,
                                bool isUseless) {
  assert(occ1 != occ2);
  assert(occ1->syncIrIndex < occ2->syncIrIndex);

  if (checkVisited(occ1, occ2)) {
    assert(false && "expected to not check a pair more than once.");
    return;
  }
  if (checkImpossibleOccPair(occ1, occ2) || checkAlreadySynced(occ1, occ2) ||
      skipMMad1DecomposedLoopOpt(occ1, occ2) ||
      checkSkipParallelLoop(occ1, occ2) || checkSkipCrossCorePair(occ1, occ2) ||
      checkAlreadySyncedWithUnitFlag(occ1, occ2)) {
    return;
  }
  processConflict(occ1, occ2, rwOp1, rwOp2, isUseless);
}

bool SyncSolverV2::processOrder(ProcessingOrderV2 processingOrder) {
  auto [lcaOcc1, lcaOcc2, occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst,
        isUseless] = processingOrder;

  assert(occ1 != occ2);
  assert(occ1->syncIrIndex < occ2->syncIrIndex);
  assert(lcaOcc1 != lcaOcc2);
  assert(!lcaOcc2->isProperAncestor(lcaOcc1));

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

  if (checkImpossibleOccPair(occ1, occ2) || checkAlreadySynced(occ1, occ2) ||
      skipMMad1DecomposedLoopOpt(occ1, occ2) ||
      checkSkipParallelLoop(occ1, occ2) || checkSkipCrossCorePair(occ1, occ2)) {
    return false;
  }

  DEBUG_WITH_TYPE("gss-sync-solver-checking", {
    llvm::dbgs() << "checking: " << (isUseless ? "is-useless\n" : "\n");
    llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                 << occ1->endIndex << ' ' << occ1->op->str(0, false) << '\n';
    llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                 << occ2->endIndex << ' ' << occ2->op->str(0, false) << '\n';
  });

  if (checkAlreadySyncedWithUnitFlag(occ1, occ2)) {
    return false;
  }

  if (options.alwaysUsePipeSAsWaitingPipe) {
    corePipeDst.pipe = hivm::PIPE::PIPE_S;
  }
  auto eventIdInfo =
      getEventIdInfo(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst);
  if (!checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst, eventIdInfo)) {
    if (!checkGraphConflict(parOcc1, parOcc2, corePipeSrc, corePipeDst,
                            EventIdInfo(1))) {
      return true;
    }
    return false;
  }

  auto *conflictPair = handleConflict(occ1, occ2, rwOp1, rwOp2, corePipeSrc,
                                      corePipeDst, eventIdInfo, isUseless);
  if (conflictPair == nullptr || conflictPair->couldNotRun) {
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
    MemInfoOccElement occElement1;
    MemInfoOccElement occElement2;
    MemInfoOccElementList::iterator occElementIt1;
    MemInfoOccElementList::iterator occElementIt2;

    QueueElement(MemInfoNode *node1, MemInfoNode *node2,
                 MemInfoOccElement occElement1, MemInfoOccElement occElement2,
                 MemInfoOccElementList::iterator occElementIt1,
                 MemInfoOccElementList::iterator occElementIt2)
        : node1(node1), node2(node2), occElement1(occElement1),
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
      for (auto &node1 : nodeList1) {
        auto it1 =
            node1.lower_bound(MemInfoOccElement(nullptr, nullptr, rIndex1, -1));
        if (it1 == node1.occElements.begin()) {
          continue;
        }
        it1 = std::prev(it1);
        if (it1->occIndex < lIndex1) {
          continue;
        }

        for (auto &[memoryEffect2, map2] : lcaOcc2->memInfoTree2.nodeListMap) {
          if (memoryEffect1 == MemoryEffect::READ &&
              memoryEffect2 == MemoryEffect::READ) {
            continue;
          }
          for (auto &[corePipeDst, nodeList2] : map2) {
            if (checkSkipCrossCorePair(corePipeSrc.coreType,
                                       corePipeDst.coreType)) {
              continue;
            }
            for (auto &node2 : nodeList2) {
              auto it2 = node2.lower_bound(
                  MemInfoOccElement(nullptr, nullptr, lIndex2, -1));
              if (it2 == node2.occElements.end() || it2->occIndex >= rIndex2) {
                continue;
              }
              queue.emplace(&node1, &node2, *it1, *it2, it1, it2);
            }
          }
        }
      }
    }
  }

  llvm::DenseSet<std::tuple<CorePipeInfo, CorePipeInfo>> handledCorePipePairs;
  auto handle = [&](const QueueElement &queueElement) -> bool {
    auto &memInfo1 = queueElement.node1->rootMemInfo;
    auto corePipeSrc = queueElement.node1->corePipeInfo;
    auto *occ1 = queueElement.occElement1.occ;
    auto *rwOp1 = dyn_cast<RWOperation>(occ1->op);
    assert(rwOp1 != nullptr);

    auto &memInfo2 = queueElement.node2->rootMemInfo;
    auto corePipeDst = queueElement.node2->corePipeInfo;
    auto *occ2 = queueElement.occElement2.occ;
    auto *rwOp2 = dyn_cast<RWOperation>(occ2->op);
    assert(rwOp2 != nullptr);

    if (!checkMemInfoConflict(/*rwOp1=*/nullptr, /*rwOp2=*/nullptr, memInfo1,
                              memInfo2)) {
      return false;
    }

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
    handledCorePipePairs.insert(std::make_tuple(corePipeSrc, corePipeDst));
    return true;
  };

  while (!queue.empty()) {
    auto current = queue.top();
    queue.pop();

    auto key = std::make_tuple(current.node1->corePipeInfo,
                               current.node2->corePipeInfo);
    if (handledCorePipePairs.contains(key)) {
      continue;
    }
    if (handle(current)) {
      continue;
    }

    if (current.occElementIt1 != current.node1->occElements.begin()) {
      auto nextIt1 = std::prev(current.occElementIt1);
      if (nextIt1->occIndex >= lIndex1) {
        auto next = current;
        next.occElement1 = *nextIt1;
        next.occElementIt1 = nextIt1;
        queue.push(next);
      }
    }

    auto nextIt2 = std::next(current.occElementIt2);
    if (nextIt2 != current.node2->occElements.end() &&
        nextIt2->occIndex < rIndex2) {
      auto next = current;
      next.occElement2 = *nextIt2;
      next.occElementIt2 = nextIt2;
      queue.push(next);
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

void SyncSolverV2::processConflict(Occurrence *occ1, Occurrence *occ2,
                                   RWOperation *rwOp1, RWOperation *rwOp2,
                                   bool isUseless) {
  for (auto [corePipeSrc, corePipeDst] : getMemoryConflicts(rwOp1, rwOp2)) {
    if (options.alwaysUsePipeSAsWaitingPipe) {
      corePipeDst.pipe = hivm::PIPE::PIPE_S;
    }
    auto eventIdInfo =
        getEventIdInfo(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst);
    handleConflict(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst,
                   eventIdInfo, isUseless);
  }
}

ConflictPair *
SyncSolverV2::handleConflict(Occurrence *occ1, Occurrence *occ2,
                             RWOperation *rwOp1, RWOperation *rwOp2,
                             CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                             EventIdInfo eventIdInfo, bool isUseless) {
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
    eventIdInfo.setEventIdNum(1);
    return handleBarrierConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                 eventIdInfo, isUseless);
  } else if (auto unitFlagInfo = checkUnitFlagPatterns(occ1, occ2)) {
    return handleUnitFlagConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                  unitFlagInfo.value(), isUseless);
  } else {
    return handleSetWaitConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                 eventIdInfo, isUseless);
  }
}
