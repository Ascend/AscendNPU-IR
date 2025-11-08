//===---------- MoveSyncState.cpp ----Move out sync for forOp and ifOp ----===//
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

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/MoveSyncState.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "hivm-inject-sync"

using namespace mlir;
using namespace mlir::hivm;

void MoveSyncState::StateOptimize() {
  MoveOutBranchSync();
  MoveForSync();
}

void MoveSyncState::MoveOutBranchSync() {
  for (auto &e : syncIR) {
    if (auto *branchElement = dyn_cast<BranchInstanceElement>(e.get())) {
      if (branchElement->getBranchKind() == KindOfBranch::IF_BEGIN) {
        std::pair<unsigned, unsigned> bound = {branchElement->beginId,
                                               branchElement->endId};
        for (unsigned i = branchElement->beginId + 1;
             i < branchElement->branchId; i++) {
          PlanMoveOutBranchSync(
              syncIR[i].get(),
              {branchElement->beginId, branchElement->branchId}, bound);
        }
        if (branchElement->endId == branchElement->branchId) {
          continue;
        }
        for (unsigned i = branchElement->branchId + 1; i < branchElement->endId;
             i++) {
          PlanMoveOutBranchSync(syncIR[i].get(),
                                {branchElement->branchId, branchElement->endId},
                                bound);
        }
      }
    }
  }
}

void MoveSyncState::PlanMoveOutBranchSync(
    InstanceElement *e, std::pair<unsigned int, unsigned int> pair,
    std::pair<unsigned int, unsigned int> bound) {
  checkCondition(
      pair.first < e->GetIndex() && e->GetIndex() < pair.second,
      "PlanMoveOutBranchSync expected element to be within pair bounds");
  SyncOps newPipeBefore;
  for (auto &s : e->pipeBefore) {
    PlanMoveOutIfWaitSync(newPipeBefore, s, pair, bound);
  }

  SyncOps newPipeAfter;
  for (auto &s : llvm::reverse(e->pipeAfter)) {
    PlanMoveOutIfSetSync(newPipeAfter, s, pair, bound);
  }
  e->pipeAfter = newPipeAfter;
  e->pipeBefore = newPipeBefore;
}

void MoveSyncState::PlanMoveOutIfWaitSync(
    SyncOps &newPipeBefore, SyncOperation *s,
    std::pair<unsigned int, unsigned int> pair,
    std::pair<unsigned int, unsigned int> bound) {
  if (s->GetType() != SyncOperation::TYPE::WAIT_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    newPipeBefore.push_back(s);
    return;
  }
  auto &syncPair = syncOperations[s->GetSyncIndex()];
  checkCondition(!syncPair.empty(), "expected syncPair not to be empty");
  auto *setSync = syncPair[0].get();
  if ((setSync->GetSyncIRIndex() >= pair.second) ||
      (setSync->GetSyncIRIndex() <= pair.first)) {
    // Process the following scenarios:
    // setFlag
    // if():
    //    waitFlag
    //    compound
    // change to :
    // setFlag
    // waitFlag
    // if():
    //    compound
    checkSyncIRIndex(syncIR, bound.first);
    syncIR[bound.first]->pipeBefore.push_back(s);
    s->SetSyncIRIndex(bound.first);
  } else {
    newPipeBefore.push_back(s);
  }
}

void MoveSyncState::PlanMoveOutIfSetSync(
    SyncOps &newPipeAfter, SyncOperation *s,
    std::pair<unsigned int, unsigned int> pair,
    std::pair<unsigned int, unsigned int> bound) {
  if (s->GetType() != SyncOperation::TYPE::SET_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_SET) {
    newPipeAfter.push_back(s);
    return;
  }
  auto &syncPair = syncOperations[s->GetSyncIndex()];
  checkCondition(syncPair.size() > 1, "expected syncPair size > 1");
  auto *waitSync = syncPair[1].get();
  if ((waitSync->GetSyncIRIndex() >= pair.second) ||
      (waitSync->GetSyncIRIndex() <= pair.first)) {
    // Process the following scenarios:
    // if():
    //    compound
    //    setFlag
    // waitFlag
    // change to :
    // if():
    //    compound
    // setFlag
    // waitFlag
    checkSyncIRIndex(syncIR, bound.second);
    syncIR[bound.second]->pipeAfter.push_front(s);
    s->SetSyncIRIndex(bound.second);
  } else {
    newPipeAfter.push_back(s);
  }
}

void MoveSyncState::MoveForSync() {
  for (auto &e : syncIR) {
    if (auto *forCompound = dyn_cast<LoopInstanceElement>(e.get())) {
      if (forCompound->getLoopKind() == KindOfLoop::LOOP_END) {
        if (forCompound->ignore_block_sync_move_out) {
          continue;
        }
        for (unsigned i = forCompound->beginId + 1; i < forCompound->endId; i++)
          MoveOutSync(syncIR[i].get(),
                      {forCompound->beginId, forCompound->endId});
      }
    }
  }
}

void MoveSyncState::MoveOutSync(InstanceElement *e,
                                std::pair<unsigned int, unsigned int> pair) {
  checkCondition(pair.first < e->GetIndex() && e->GetIndex() < pair.second,
                 "MoveOutSync expected element to be within pair bounds");
  SyncOps newPipeBefore;
  for (auto &s : e->pipeBefore) {
    PlanMoveOutWaitSync(newPipeBefore, s, pair);
  }

  SyncOps newPipeAfter;
  for (auto &s : llvm::reverse(e->pipeAfter)) {
    PlanMoveOutSetSync(newPipeAfter, s, pair);
  }
  e->pipeAfter = newPipeAfter;
  e->pipeBefore = newPipeBefore;
}

void MoveSyncState::PlanMoveOutWaitSync(
    SyncOps &newPipeBefore, SyncOperation *s,
    std::pair<unsigned int, unsigned int> pair) {
  if (s->GetType() != SyncOperation::TYPE::WAIT_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    newPipeBefore.push_back(s);
    return;
  }
  auto &syncPair = syncOperations[s->GetSyncIndex()];
  checkCondition(!syncPair.empty(), "expected syncPair not to be empty");
  auto *setSync = syncPair[0].get();
  if ((setSync->GetSyncIRIndex() > pair.second) ||
      (setSync->GetSyncIRIndex() < pair.first)) {
    // Process the following scenarios:
    // setFlag
    // for():
    //    waitFlag
    //    compound
    // change to :
    // setFlag
    // waitFlag
    // for():
    //    compound
    checkSyncIRIndex(syncIR, pair.first);
    syncIR[pair.first]->pipeBefore.push_back(s);
    s->SetSyncIRIndex(pair.first);
    return;
  }
  newPipeBefore.push_back(s);
}

void MoveSyncState::PlanMoveOutSetSync(
    SyncOps &newPipeAfter, SyncOperation *s,
    const std::pair<unsigned int, unsigned int> pair) {
  if (s->GetType() != SyncOperation::TYPE::SET_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_SET) {
    newPipeAfter.push_back(s);
    return;
  }
  auto &syncPair = syncOperations[s->GetSyncIndex()];
  checkCondition(syncPair.size() > 1, "expected syncPair size > 1");
  auto *waitSync = syncPair[1].get();
  if ((waitSync->GetSyncIRIndex() > pair.second) ||
      (waitSync->GetSyncIRIndex() < pair.first)) {
    // Process the following scenarios:
    // for():
    //    compound
    //    setFlag
    // waitFlag
    // change to :
    // for():
    //    compound
    // setFlag
    // waitFlag
    checkSyncIRIndex(syncIR, pair.second);
    syncIR[pair.second]->pipeAfter.push_front(s);
    s->SetSyncIRIndex(pair.second);
    return;
  }
  newPipeAfter.push_back(s);
}