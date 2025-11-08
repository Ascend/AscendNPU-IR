//===---------- RemoveRedundantSync.cpp ----Remove redundant sync ---------===//
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

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/RemoveRedundantSync.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"

#define DEBUG_TYPE "hivm-inject-sync"

using namespace mlir;
using namespace mlir::hivm;

void RemoveRedundantSync::Plan() {

  std::vector<std::pair<SyncOperation *, SyncOperation *>> syncOps;
  for (auto &syncPair : syncOperations) {
    if (syncPair.size() == 2) {
      auto *setFlag = syncPair[0].get();
      auto *waitFlag = syncPair[1].get();
      syncOps.push_back(std::make_pair(setFlag, waitFlag));
    }
  }

  sort(syncOps.begin(), syncOps.end(),
       [](std::pair<SyncOperation *, SyncOperation *> syncPair1,
          std::pair<SyncOperation *, SyncOperation *> syncPair2) {
         auto *syncOp1 = syncPair1.first;
         auto *syncOp2 = syncPair2.first;
         if (syncOp1->GetForEndIndex().has_value() &&
             syncOp2->GetForEndIndex().has_value()) {
           if (syncOp1->GetForEndIndex().value() !=
               syncOp2->GetForEndIndex().value()) {
             return syncOp1->GetForEndIndex().value() >
                    syncOp2->GetForEndIndex().value();
           } else {
             return syncOp1->GetSyncIndex() > syncOp2->GetSyncIndex();
           }
         }
         if (syncOp1->GetForEndIndex().has_value() ||
             syncOp2->GetForEndIndex().has_value()) {
           return syncOp1->GetForEndIndex().has_value() >
                  syncOp2->GetForEndIndex().has_value();
         }
         return syncOp1->GetSyncIndex() > syncOp2->GetSyncIndex();
       });

  for (auto [setFlag, waitFlag] : syncOps) {
    bool useless = CheckAllSync(setFlag, waitFlag);
    if (useless) {
      auto it0 = std::find(syncIR[setFlag->GetSyncIRIndex()]->pipeAfter.begin(),
                           syncIR[setFlag->GetSyncIRIndex()]->pipeAfter.end(),
                           setFlag);
      if (it0 != syncIR[setFlag->GetSyncIRIndex()]->pipeAfter.end()) {
        syncIR[setFlag->GetSyncIRIndex()]->pipeAfter.erase(it0);
      }
      auto it1 = std::find(
          syncIR[waitFlag->GetSyncIRIndex()]->pipeBefore.begin(),
          syncIR[waitFlag->GetSyncIRIndex()]->pipeBefore.end(), waitFlag);
      if (it1 != syncIR[waitFlag->GetSyncIRIndex()]->pipeBefore.end()) {
        syncIR[waitFlag->GetSyncIRIndex()]->pipeBefore.erase(it1);
      }
    }
  }
}

bool RemoveRedundantSync::CheckAllSync(SyncOperation *setFlag,
                                       SyncOperation *waitFlag) {
  SmallVector<bool> syncFinder(syncOperations.size(), false);
  unsigned int begin = setFlag->GetSyncIRIndex();
  unsigned int end = waitFlag->GetSyncIRIndex();
  auto forEndIndex = setFlag->GetForEndIndex();
  if (begin < end) {
    return CheckRepeatSync(begin, end, syncFinder, setFlag);
  } else {
    checkCondition(forEndIndex.has_value(),
                   "setFlag expected to have forEndIndex");
    auto *ptr =
        dyn_cast<LoopInstanceElement>(syncIR[forEndIndex.value()].get());
    checkCondition(ptr != nullptr, "");
    return CheckRepeatSync(begin, ptr->endId, syncFinder, setFlag) ||
           CheckRepeatSync(ptr->beginId, end, syncFinder, setFlag);
  }
}

bool RemoveRedundantSync::CheckRepeatSync(unsigned int begin, unsigned int end,
                                          SmallVector<bool> &syncFinder,
                                          SyncOperation *setFlag) {
  checkCondition(begin <= end, "expected begin <= end");
  checkSyncIRIndex(syncIR, end);
  bool res{false};
  for (auto &relatedSync : syncIR[begin]->pipeAfter) {
    res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
  }

  for (unsigned i = begin + 1; i <= end - 1; i++) {
    checkSyncIRIndex(syncIR, i);
    for (auto &relatedSync : syncIR[i]->pipeBefore) {
      res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
    }
    if (auto *branchElement =
            dyn_cast<BranchInstanceElement>(syncIR[i].get())) {
      if (CheckBranchBetween(branchElement, syncFinder, setFlag, end, i)) {
        return true;
      }
    }
    if (auto *forElement = dyn_cast<LoopInstanceElement>(syncIR[i].get())) {
      if (CheckLoopBetween(forElement, setFlag, i)) {
        return true;
      }
    }
    for (auto &relatedSync : syncIR[i]->pipeAfter) {
      res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
    }
  }
  for (auto &relatedSync : syncIR[end]->pipeBefore) {
    res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
  }
  return res;
}

bool RemoveRedundantSync::CheckBranchBetween(
    BranchInstanceElement *branchElement, SmallVector<bool> syncFinder,
    SyncOperation *setFlag, unsigned endId, unsigned &i) {
  if (branchElement->getBranchKind() != KindOfBranch::IF_BEGIN) {
    i = branchElement->endId;
    return false;
  }
  bool hasElseBranch = branchElement->branchId < branchElement->endId;
  bool endIsInsideThenBranch =
      (!hasElseBranch && endId < branchElement->endId) ||
      (hasElseBranch && endId < branchElement->branchId);
  if (endIsInsideThenBranch) {
    return false;
  }
  bool endIsInsideElseBranch = hasElseBranch &&
                               endId >= branchElement->branchId &&
                               endId < branchElement->endId;
  if (endIsInsideElseBranch) {
    i = branchElement->branchId;
    return false;
  }
  if (hasElseBranch) {
    if (CheckRepeatSync(branchElement->beginId, branchElement->branchId,
                        syncFinder, setFlag) &&
        CheckRepeatSync(branchElement->branchId, branchElement->endId,
                        syncFinder, setFlag)) {
      return true;
    }
  }
  i = branchElement->endId;
  return false;
}

bool RemoveRedundantSync::CheckLoopBetween(LoopInstanceElement *loopElement,
                                           SyncOperation *setFlag,
                                           unsigned &i) {
  i = loopElement->endId;
  return false;
}

bool RemoveRedundantSync::CanMatchedSync(SmallVector<bool> &syncFinder,
                                         SyncOperation *relatedSync,
                                         SyncOperation *setFlag) {
  bool unrelatedSync =
      (syncAnalysisMode == SyncAnalysisMode::NORMALSYNC &&
       relatedSync->GetType() != SyncOperation::TYPE::WAIT_EVENT &&
       relatedSync->GetType() != SyncOperation::TYPE::SET_EVENT) ||
      (syncAnalysisMode == SyncAnalysisMode::BLOCKSYNC &&
       relatedSync->GetType() != SyncOperation::TYPE::SYNC_BLOCK_WAIT &&
       relatedSync->GetType() != SyncOperation::TYPE::SYNC_BLOCK_SET) ||
      relatedSync->GetSyncIndex() == setFlag->GetSyncIndex() ||
      relatedSync->GetDstPipe() != setFlag->GetDstPipe() ||
      relatedSync->GetSrcPipe() != setFlag->GetSrcPipe() ||
      relatedSync->eventIdNum > setFlag->eventIdNum;
  if (unrelatedSync) {
    return false;
  }
  checkCondition(relatedSync->GetSyncIndex() < syncFinder.size(),
                 "sync operation has index larger than syncFinder size");
  // Process the following scenarios:
  // compound
  // setFlag(srcPipe,dstPipe,-1)  --> current sync
  // ...
  // setFlag(srcPipe,dstPipe,-1)  --> related sync
  // waitFlag(srcPipe,dstPipe,-1) --> related sync
  // ...
  // waitFlag(srcPipe,dstPipe,-1) --> current sync
  // compound
  // change to :
  // compound
  // ...
  // setFlag(srcPipe,dstPipe,-1)  --> related sync
  // waitFlag(srcPipe,dstPipe,-1) --> related sync
  // ...
  // compound
  if (syncFinder[relatedSync->GetSyncIndex()] &&
      ((syncAnalysisMode == SyncAnalysisMode::NORMALSYNC &&
        relatedSync->GetType() == SyncOperation::TYPE::WAIT_EVENT) ||
       (syncAnalysisMode == SyncAnalysisMode::BLOCKSYNC &&
        relatedSync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT))) {
    return true;
  }
  if ((syncAnalysisMode == SyncAnalysisMode::NORMALSYNC &&
       relatedSync->GetType() == SyncOperation::TYPE::SET_EVENT) ||
      (syncAnalysisMode == SyncAnalysisMode::BLOCKSYNC &&
       relatedSync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET)) {
    syncFinder[relatedSync->GetSyncIndex()] = true;
  }
  return false;
}
