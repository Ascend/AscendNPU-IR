//=======- SyncAnalysis.cpp --Dependency analysis and insert sync  ------=====//
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

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncAnalysis.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <utility>

#define DEBUG_TYPE "hivm-inject-sync"

using namespace mlir;
using namespace mlir::hivm;

void SyncAnalyzer::Plan(bool insertBarAllAtLast) {
  syncIndex = syncOperations.size();
  for (auto &nowElement : syncIR) {
    if (auto *nowCompound =
            dyn_cast<CompoundInstanceElement>(nowElement.get())) {
      DealWithCompoundSync(nowCompound);
    } else if (auto *loopElement =
                   dyn_cast<LoopInstanceElement>(nowElement.get())) {
      DealWithLoopSync(loopElement);
    } else if (auto *branchElement =
                   dyn_cast<BranchInstanceElement>(nowElement.get())) {
      continue;
    }
  }

  if (syncAnalysisMode == SyncAnalysisMode::NORMALSYNC) {
    insertMmadL1BackPipeMPipeMTE1Sync();
  }
  if (insertBarAllAtLast) {
    InsertLastPipeAll();
  }
}

void SyncAnalyzer::insertMmadL1BackPipeMPipeMTE1Sync() {
  for (auto &e : syncIR) {
    if (auto *compound = dyn_cast<CompoundInstanceElement>(e.get())) {
      if (!isa<hivm::MmadL1Op>(compound->elementOp)) {
        continue;
      }

      // There must be a for loop on the outer layer of MmadL1Op.
      LoopLikeOpInterface parentLoop =
          compound->elementOp->getParentOfType<LoopLikeOpInterface>();
      if (!parentLoop) {
        continue;
      }

      // for each mmadl1 operation, there is 2 in the sync-ir, for the first one
      // (that has an empty useVec) we will use the id 0, and for the other one
      // we will use id 1.
      int backPipeMEventID = compound->useVec.empty() ? 0 : 1;
      if (this->PipeMTE1ToPipeMSync[backPipeMEventID] != nullptr) {
        compound->PipeMTE1ToPipeMSync =
            this->PipeMTE1ToPipeMSync[backPipeMEventID];
        continue;
      }

      unsigned insertWaitId = syncIR.size() - 1;
      unsigned insertSetId = 0;
      std::unique_ptr<SyncOperation, std::default_delete<SyncOperation>>
          setSyncOp = std::make_unique<SyncOperation>(SyncOperation{
              SyncOperation::TYPE::SET_EVENT, hivm::PIPE::PIPE_M,
              hivm::PIPE::PIPE_MTE1, syncIndex, insertSetId, std::nullopt});
      assert(setSyncOp != nullptr);
      auto waitSyncOp = setSyncOp->GetMatchSync(insertWaitId);
      this->PipeMTE1ToPipeMSync[backPipeMEventID] = setSyncOp.get();
      compound->PipeMTE1ToPipeMSync =
          this->PipeMTE1ToPipeMSync[backPipeMEventID];
      syncIR[insertSetId]->pipeBefore.push_back(setSyncOp.get());
      syncIR[insertWaitId]->pipeAfter.push_back(waitSyncOp.get());
      SmallVector<std::unique_ptr<SyncOperation>> newSync;
      newSync.emplace_back(std::move(setSyncOp));
      newSync.emplace_back(std::move(waitSyncOp));
      syncOperations.emplace_back(std::move(newSync));
      syncIndex++;
      assert(syncOperations.size() == syncIndex);
    }
  }
}

void SyncAnalyzer::InsertLastPipeAll() {
  if (!hacc::utils::isDeviceEntry(func_)) {
    return;
  }
  // insert last barrier all.
  assert(!syncIR.empty());
  auto *nowCompound = syncIR[syncIR.size() - 1].get();
  auto syncFront = std::make_unique<SyncOperation>(SyncOperation{
      SyncOperation::TYPE::PIPE_BARRIER, hivm::PIPE::PIPE_ALL,
      hivm::PIPE::PIPE_ALL, syncIndex, nowCompound->GetIndex(), std::nullopt});
  assert(syncFront != nullptr && "get unique_ptr fail");
  syncIR[syncIR.size() - 1]->pipeAfter.push_back(syncFront.get());
  SmallVector<std::unique_ptr<SyncOperation>> newSync;
  newSync.emplace_back(std::move(syncFront));
  syncOperations.emplace_back(std::move(newSync));
  syncIndex++;
  assert(syncOperations.size() == syncIndex);
}

bool SyncAnalyzer::isParallelLoop(scf::ForOp forOp) const {
  return forOp->hasAttrOfType<UnitAttr>(hivm::ParallelLoopAttr::name);
}

bool SyncAnalyzer::checkUnderParallelLoop(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound) const {
  auto loopOp1 = nowCompound->elementOp->getParentOfType<LoopLikeOpInterface>();
  auto loopOp2 =
      frontCompound->elementOp->getParentOfType<LoopLikeOpInterface>();
  auto forOp1 = llvm::dyn_cast_if_present<scf::ForOp>(loopOp1.getOperation());
  auto forOp2 = llvm::dyn_cast_if_present<scf::ForOp>(loopOp2.getOperation());
  if (!forOp1 || !forOp2 || forOp1 != forOp2) {
    return false;
  }
  return isParallelLoop(forOp1);
}

void SyncAnalyzer::DealWithLoopSync(LoopInstanceElement *nowElement) {
  // insert backward sync:
  // Copy the original command to the back.
  //  * In loop J :
  //  *  A [j]       original cmd
  //  *  B [j]       original cmd
  //  *  A [j + 1]   copy cmd
  //  *  B [j + 1]   copy cmd
  // Then call InsertSeqSync func to insert backward sync as sequential sync.
  if (nowElement->getLoopKind() == KindOfLoop::LOOP_END) {
    SyncIRs backSyncIr;
    assert(syncIR.size() >= nowElement->endId);
    for (unsigned i = nowElement->beginId; i < nowElement->endId; i++) {
      if (auto *compound = dyn_cast<CompoundInstanceElement>(syncIR[i].get())) {
        InsertBackForSync(compound, backSyncIr, nowElement);
      } else if (auto *loopElement =
                     dyn_cast<LoopInstanceElement>(syncIR[i].get())) {
        auto loopKind = loopElement->getLoopKind();
        auto newForPtr = loopElement->CloneFor(loopKind);
        backSyncIr.emplace_back(std::move(newForPtr));
      } else if (auto *branchElement =
                     dyn_cast<BranchInstanceElement>(syncIR[i].get())) {
        auto newBranchPtr =
            branchElement->CloneBranch(branchElement->getBranchKind());
        backSyncIr.emplace_back(std::move(newBranchPtr));
      } else if (auto *placeHolderElement =
                     dyn_cast<PlaceHolderInstanceElement>(syncIR[i].get())) {
        auto newPlaceHolderElement = placeHolderElement->Clone();
        backSyncIr.emplace_back(std::move(newPlaceHolderElement));
      }
    }
  }
}

void SyncAnalyzer::InsertBackForSync(CompoundInstanceElement *nowCompound,
                                     SyncIRs &backSyncIr,
                                     const LoopInstanceElement *loopElement) {
  // Create sync record to transfer the synchronization relationship.
  SyncRecordList syncRecordList;
  auto backCompound =
      std::make_unique<CompoundInstanceElement>(CompoundInstanceElement{
          nowCompound->GetIndex(), nowCompound->defVec, nowCompound->useVec,
          nowCompound->kPipeValue, nowCompound->opName});
  backCompound->compoundCoreType = nowCompound->compoundCoreType;
  backCompound->elementOp = nowCompound->elementOp;
  auto *backCompoundPtr = backCompound.get();
  backSyncIr.emplace_back(std::move(backCompound));
  // Insert sync between the new generated Compound Element.
  //  *  A [j + 1]   copy cmd
  //  *  Set / Wait / Barrier
  //  *  B [j + 1]   copy cmd
  InsertSeqSync(backCompoundPtr, backSyncIr, 0,
                static_cast<int>(backSyncIr.size()) - 1, syncRecordList,
                loopElement->endId);

  // Insert sync between the new generated and
  // original Compound Element and later will substitute iv in inner
  // loops.
  //  *  A [ai+bj1]       original cmd
  //  *  B [ai+bj1]       original cmd
  //  *  setFlag
  //  *  waitFlag / barrier
  //  *  A [a(i+1) + bj2]   copy cmd
  InsertSeqSync(nowCompound, syncIR, nowCompound->GetIndex(),
                loopElement->endId, syncRecordList, loopElement->endId);
}

void SyncAnalyzer::DealWithCompoundSync(CompoundInstanceElement *nowCompound) {
  // Create sync record to transfer the synchronization relationship.
  SyncRecordList syncRecordList;
  InsertSeqSync(nowCompound, syncIR, 0, nowCompound->GetIndex(), syncRecordList,
                std::nullopt);
}

void SyncAnalyzer::InsertSeqSync(CompoundInstanceElement *nowCompound,
                                 SyncIRs &syncElement, int begin, int end,
                                 SyncRecordList &syncRecordList,
                                 const std::optional<unsigned> &forEndIndex) {
  const hivm::PIPE nowPipeValue = nowCompound->kPipeValue;
  checkSyncIRIndex(syncElement, begin);
  checkSyncIRIndex(syncElement, end);
  unsigned syncIRIndex = syncElement[end]->GetIndex();
  UpdateAlreadySync(syncIR[syncIRIndex]->pipeBefore, syncRecordList,
                    nowPipeValue);

  for (int i = end - 1; i >= begin; i--) {
    auto &frontPtr = syncElement[i];
    unsigned frontIndex = frontPtr->GetIndex();
    assert(frontIndex < syncIR.size());
    assert(syncIR[frontIndex] != nullptr);
    // Update sync record
    if (auto *frontCompound =
            dyn_cast<CompoundInstanceElement>(frontPtr.get())) {
      // Insert synchronization between the nowCompound and front
      //  *  front (xBuf.w)
      //  *  setFlag
      //  *  ...
      //  *  waitFlag / barrier
      //  *  nowCompound  (xBuf.r)
      UpdateAlreadySync(syncIR[frontIndex]->pipeAfter, syncRecordList,
                        nowPipeValue);
      InsertSync(nowCompound, frontCompound, syncRecordList, forEndIndex);
      UpdateAlreadySync(syncIR[frontIndex]->pipeBefore, syncRecordList,
                        nowPipeValue);
    } else if (auto *forInstance =
                   dyn_cast<LoopInstanceElement>(frontPtr.get())) {
      assert(syncIR[frontIndex]->pipeAfter.empty());
      int skipLoop = 
          static_cast<int>(InsertLoopSync(i, nowCompound, begin, forInstance, syncElement,
                           syncRecordList, forEndIndex));
      assert(syncIR[frontIndex]->pipeBefore.empty());
      i -= skipLoop;
    } else if (auto *branchElement =
                   dyn_cast<BranchInstanceElement>(frontPtr.get())) {
      assert(syncIR[frontIndex]->pipeBefore.empty());
      assert(syncIR[frontIndex]->pipeAfter.empty());
      int skipBranch =
          static_cast<int>(InsertBranchSync(i, nowCompound, begin, branchElement, syncElement,
                           syncRecordList, forEndIndex));
      i -= skipBranch;
    }
  }
}

void SyncAnalyzer::UpdateAlreadySync(const SyncOps &syncVector,
                                     SyncRecordList &syncRecordList,
                                     const hivm::PIPE nowPipeValue) {
  for (auto &sync : syncVector) {
    for (size_t i = 0; i < syncRecordList.size(); i++) {
      if (i == 0 && sync->eventIdNum > 1 &&
          sync->GetForEndIndex().has_value()) {
        continue;
      }
      UpdateSyncRecord(sync, syncRecordList[i], nowPipeValue);
    }
  }
}

void SyncAnalyzer::UpdateSyncRecord(const SyncOperation *sync,
                                    SyncRecord &syncRecord,
                                    hivm::PIPE nowPipeValue) {
  auto &[recordAlready, recordFinder] = syncRecord;
  const hivm::PIPE setPipeValue = sync->GetSrcPipe();
  hivm::PIPE waitPipeValue = sync->GetDstPipe();
  if (syncAnalysisMode == SyncAnalysisMode::BLOCKSYNC) {
    // Unlike normal-sync mode, in block-sync mode, there is no wait-pipe (or
    // dst-pipe), all further instructions after the syncBlockWait instruction
    // will be blocked, as if it is blocking the pipe-s pipeline.
    nowPipeValue = hivm::PIPE::PIPE_S;
    waitPipeValue = hivm::PIPE::PIPE_S;
  }

  bool barrierFinder = (nowPipeValue == waitPipeValue) &&
                       (sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER);
  if (barrierFinder) {
    recordAlready[static_cast<unsigned int>(nowPipeValue)] = true;
  } else if (recordAlready[static_cast<unsigned int>(waitPipeValue)] ||
             (nowPipeValue == waitPipeValue)) {
    if (recordFinder[sync->GetSyncIndex()] &&
        (sync->GetType() == SyncOperation::TYPE::SET_EVENT ||
         sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET)) {
      recordAlready[static_cast<unsigned int>(setPipeValue)] = true;
    }
    if (sync->GetType() == SyncOperation::TYPE::WAIT_EVENT ||
        sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
      recordFinder[sync->GetSyncIndex()] = true;
    }
  }
}

unsigned SyncAnalyzer::InsertBranchSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    BranchInstanceElement *branchElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  if (branchElement->getBranchKind() == KindOfBranch::IF_END) {
    // Insert synchronization for BranchInstanceElement
    //  *  if (x) :             |    <= after i
    //  *    front0 (xBuf.w)   |-> Call InsertSeqSync
    //  *    setFlag0         /
    //  *  else :
    //  *    front1 (xBuf.w)   |
    //  *    setFlag1          |-> Call InsertSeqSync
    //  *  endif               /     <= now i
    //  *  ...
    //  *  waitFlag0 / barrier
    //  *  waitFlag1 / barrier
    //  *  nowCompound  (x_buf.r)
    SyncRecordList syncRecordIfList = syncRecordList;
    unsigned branchIf = index - (branchElement->endId - branchElement->beginId);
    unsigned branchElse =
        index - (branchElement->endId - branchElement->branchId);
    unsigned branchEnd = index;
    InsertSeqSync(nowCompound, syncElement, branchIf, branchElse,
                  syncRecordIfList, forEndIndex);
    if (branchElement->branchId != branchElement->endId) {
      SyncRecordList syncRecordElseList = syncRecordList;
      InsertSeqSync(nowCompound, syncElement, branchElse, branchEnd,
                    syncRecordElseList, forEndIndex);
      MergeAlreadySync(syncRecordList, syncRecordIfList, syncRecordElseList);
    } else {
      for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size();
           bufferIdx++) {
        syncRecordList[bufferIdx].syncFinder =
            syncRecordIfList[bufferIdx].syncFinder;
      }
    }
    return (branchElement->endId - branchElement->beginId);
  } else if (branchElement->getBranchKind() == KindOfBranch::ELSE_BEGIN &&
             index != begin) {
    // Insert synchronization for BranchInstanceElement
    // the current element must be in the else branch, because the entire
    // if-else branch is traversed to IF_END and then skipped
    //  *  front0 (xBuf.w)
    //  *  setFlag0
    //  *  if (x) :               <= after i
    //  *    front1 (xBuf.w)
    //  *  else :                 <= now i
    //  *    waitFlag0 / barrier
    //  *    nowCompound  (x_buf.r)
    assert(nowCompound->GetIndex() > branchElement->branchId);
    return (branchElement->branchId - branchElement->beginId);
  }
  return 0;
}

unsigned SyncAnalyzer::InsertLoopSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    LoopInstanceElement *loopElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList, const std::optional<int> &forEndIndex) {
  if (loopElement->getLoopKind() == KindOfLoop::LOOP_END) {
    SyncRecordList syncRecordForList = syncRecordList;
    unsigned newBegin =
        std::max(begin, index - (loopElement->endId - loopElement->beginId));
    unsigned newEnd = index;
    InsertSeqSync(nowCompound, syncElement, newBegin, newEnd, syncRecordForList,
                  forEndIndex);
    if (assumeAliveLoops) {
      syncRecordList = std::move(syncRecordForList);
    }
    return loopElement->endId - loopElement->beginId;
  }
  return 0;
}

void SyncAnalyzer::MergeAlreadySync(SyncRecordList &syncRecordList,
                                    const SyncRecordList &syncRecordIfList,
                                    const SyncRecordList &syncRecordElseList) {
  for (size_t i = 0; i < syncRecordList.size(); i++) {
    for (size_t j = 0; j < getPipeNum(); j++) {
      if (syncRecordIfList[i].alreadySync[j] &&
          syncRecordElseList[i].alreadySync[j]) {
        syncRecordList[i].alreadySync[j] = true;
      }
    }
  }
}

bool SyncAnalyzer::isBlockCubeCube(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound) const {
  if (syncAnalysisMode != SyncAnalysisMode::BLOCKSYNC) {
    // only enable in block-sync mode.
    return false;
  }
  if (nowCompound->compoundCoreType != TCoreType::CUBE ||
      frontCompound->compoundCoreType != TCoreType::CUBE) {
    // only enable when both operations are cube-core.
    return false;
  }
  // for now, only process fixpipe/load pairs of cube operations.
  if (!isa<hivm::FixpipeOp>(frontCompound->elementOp) ||
      !isa<hivm::LoadOp>(nowCompound->elementOp)) {
    return false;
  }
  return true;
}

bool SyncAnalyzer::IsNoNeedToInsertSync(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound, bool isBackwardDep) const {
  const hivm::PIPE frontPipeValue = frontCompound->kPipeValue;
  const hivm::PIPE nowPipeValue = nowCompound->kPipeValue;
  if (frontPipeValue == nowPipeValue && frontPipeValue == hivm::PIPE::PIPE_S) {
    // no need to insert sync
    return true;
  }
  if (nowCompound->elementOp == frontCompound->elementOp && !isBackwardDep) {
    // Two same compounds assigned by MacroOp do not require insertion
    // of forward synchronization, only need to be judged in reverse.
    return true;
  }
  if (isBackwardDep) {
    if (checkUnderParallelLoop(nowCompound, frontCompound)) {
      return true;
    }
  }
  if (syncAnalysisMode == SyncAnalysisMode::BLOCKSYNC) {
    // will be handled by normal sync analysis.
    if (frontCompound->compoundCoreType == TCoreType::CUBE_AND_VECTOR) {
      return true;
    }
    // TODO: support vector-vector
    // currently we only support cube-cube (vv unsupported)
    if (nowCompound->compoundCoreType == frontCompound->compoundCoreType) {
      return !isBlockCubeCube(nowCompound, frontCompound);
    }
  }
  if ((nowPipeValue == hivm::PIPE::PIPE_M &&
       frontPipeValue == hivm::PIPE::PIPE_M)) {
    // library will do pipe_barrier(PIPE_M) according to the shape size.
    // no need to insert pipe_barrier(PIPE_M) for inject sync pass.
    return true;
  }
  return false;
}

void SyncAnalyzer::InsertSync(CompoundInstanceElement *nowCompound,
                              CompoundInstanceElement *frontCompound,
                              SyncRecordList &syncRecordList,
                              const std::optional<unsigned> &forEndIndex) {
  if (IsNoNeedToInsertSync(nowCompound, frontCompound,
                           forEndIndex.has_value())) {
    return;
  }
  MemAnalyze(nowCompound, frontCompound, syncRecordList, forEndIndex);
}

bool SyncAnalyzer::isAlreadySync(CompoundInstanceElement *nowCompound,
                                 CompoundInstanceElement *frontCompound,
                                 SyncRecordList &syncRecordList,
                                 unsigned recordListIndex) {
  if (isBlockCubeCube(nowCompound, frontCompound)) {
    // TODO: refactor syncRecordList to add core-type and differentiate between
    // cube-vector and cube-cube/vecto-vector sync operations. Now it might
    // insert redundant barrier.all instructions.
    return false;
  }
  const hivm::PIPE frontPipeValue = frontCompound->kPipeValue;
  if (syncRecordList[recordListIndex]
          .alreadySync[static_cast<unsigned int>(frontPipeValue)]) {
    return true;
  } else if (frontPipeValue == PIPE::PIPE_MTE2 &&
             !isa<hivm::MmadL1Op>(nowCompound->elementOp)) {
    if (syncRecordList[recordListIndex].alreadySync[static_cast<unsigned int>(
            PIPE::VIRTUAL_PIPE_MTE2_L1A)]) {
      return true;
    }
    if (syncRecordList[recordListIndex].alreadySync[static_cast<unsigned int>(
            PIPE::VIRTUAL_PIPE_MTE2_L1B)]) {
      return true;
    }
  }
  return false;
}

bool SyncAnalyzer::IsMemInfoHasDependency(
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  bool hasDependency = memAnalyzer.DepBetween(
      nowCompound->defVec, frontCompound->defVec, depBaseMemInfosVec);
  hasDependency =
      memAnalyzer.DepBetween(nowCompound->useVec, frontCompound->defVec,
                             depBaseMemInfosVec) ||
      hasDependency;
  hasDependency =
      memAnalyzer.DepBetween(nowCompound->defVec, frontCompound->useVec,
                             depBaseMemInfosVec) ||
      hasDependency;
  return hasDependency;
}

void SyncAnalyzer::MemAnalyze(CompoundInstanceElement *nowCompound,
                              CompoundInstanceElement *frontCompound,
                              SyncRecordList &syncRecordList,
                              const std::optional<unsigned> &forEndIndex) {
  if (isAlreadySync(nowCompound, frontCompound, syncRecordList, 0)) {
    // already sync by checking single buffer records, no need to insert sync
    // any more
    return;
  }
  DepBaseMemInfoPairVec depBaseMemInfosVec;
  if (!IsMemInfoHasDependency(nowCompound, frontCompound, depBaseMemInfosVec)) {
    //  no need to insert sync if no dependency.
    return;
  }
  if (forEndIndex.has_value()) {
    int eventIdNum = GetEventIdNum(depBaseMemInfosVec);
    for (int i = 1; i < eventIdNum; i++) {
      if (isAlreadySync(nowCompound, frontCompound, syncRecordList, i)) {
        // already sync by checking corresponding multi buffer records, no need
        // to insert sync any more
        return;
      }
    }
  }
  if (syncAnalysisMode == SyncAnalysisMode::BLOCKSYNC) {
    InsertBlockSyncOperation(nowCompound, frontCompound, depBaseMemInfosVec,
                             forEndIndex);
  } else {
    assert(syncAnalysisMode == SyncAnalysisMode::NORMALSYNC);
    if (auto unitFlagMode =
            checkUnitFlagPatterns(nowCompound, frontCompound, forEndIndex)) {
      insertUnitFlagEnabledSyncOperations(nowCompound, frontCompound,
                                          forEndIndex, unitFlagMode->first,
                                          unitFlagMode->second);
    } else {
      InsertSyncOperation(nowCompound, frontCompound, depBaseMemInfosVec,
                          forEndIndex);
    }
  }
  UpdateSyncRecordInfo(frontCompound, syncRecordList);
}

void SyncAnalyzer::UpdateSyncRecordInfo(CompoundInstanceElement *frontCompound,
                                        SyncRecordList &syncRecordList) {
  for (size_t i = 0; i < syncRecordList.size(); i++) {
    assert(!syncOperations.empty());
    auto &syncPair = syncOperations.back();

    auto *newSync = syncPair[0].get();
    if (i == 0 && newSync->eventIdNum > 1) {
      continue;
    }
    syncRecordList[i]
        .alreadySync[static_cast<unsigned int>(newSync->GetSrcPipe())] = true;
  }
}

std::optional<std::pair<LoopLikeOpInterface, LoopLikeOpInterface>>
SyncAnalyzer::getBlockSyncMultibufferEnabledLoops(
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound) {
  LoopLikeOpInterface nowParentLoop =
      nowCompound->elementOp->getParentOfType<LoopLikeOpInterface>();
  LoopLikeOpInterface frontParentLoop =
      frontCompound->elementOp->getParentOfType<LoopLikeOpInterface>();
  if (!nowParentLoop.getOperation() || !frontParentLoop.getOperation()) {
    return {};
  }
  if (getBlockSyncOpEventIdNum(nowParentLoop, frontParentLoop).has_value()) {
    return std::make_pair(nowParentLoop, frontParentLoop);
  }
  LoopLikeOpInterface nowGrandParentLoop =
      nowParentLoop->getParentOfType<LoopLikeOpInterface>();
  LoopLikeOpInterface frontGrandParentLoop =
      frontParentLoop->getParentOfType<LoopLikeOpInterface>();
  if (nowGrandParentLoop.getOperation()) {
    if (getBlockSyncOpEventIdNum(nowGrandParentLoop, frontParentLoop)
            .has_value()) {
      return std::make_pair(nowGrandParentLoop, frontParentLoop);
    }
  }
  if (frontGrandParentLoop.getOperation()) {
    if (getBlockSyncOpEventIdNum(nowParentLoop, frontGrandParentLoop)
            .has_value()) {
      return std::make_pair(nowParentLoop, frontGrandParentLoop);
    }
  }
  if (nowGrandParentLoop.getOperation() &&
      frontGrandParentLoop.getOperation()) {
    if (getBlockSyncOpEventIdNum(nowGrandParentLoop, frontGrandParentLoop)
            .has_value()) {
      return std::make_pair(nowGrandParentLoop, frontGrandParentLoop);
    }
  }
  return {};
}

std::optional<int> SyncAnalyzer::getBlockSyncOpEventIdNum(
    LoopLikeOpInterface &nowParentLoop,
    LoopLikeOpInterface &frontParentLoop) const {
  auto multibufferAttr1 =
      nowParentLoop.getOperation()->getAttrOfType<IntegerAttr>(
          kMultibufferUnrollAttrName);
  auto multibufferAttr2 =
      frontParentLoop.getOperation()->getAttrOfType<IntegerAttr>(
          kMultibufferUnrollAttrName);
  if (multibufferAttr1 && multibufferAttr2) {
    assert(multibufferAttr1.getInt() == multibufferAttr2.getInt());
    return multibufferAttr2.getInt();
  }
  return {};
}

void SyncAnalyzer::InsertBlockSyncOperation(
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec,
    const std::optional<unsigned> &forEndIndex) {
  unsigned insertWaitId = nowCompound->GetIndex();
  unsigned insertSetId = frontCompound->GetIndex();
  assert(nowCompound->compoundCoreType != TCoreType::CUBE_OR_VECTOR);
  assert(frontCompound->compoundCoreType != TCoreType::CUBE_OR_VECTOR);
  if (nowCompound->compoundCoreType == frontCompound->compoundCoreType) {
    if (nowCompound->compoundCoreType == TCoreType::CUBE) {
      std::unique_ptr<SyncOperation, std::default_delete<SyncOperation>>
          barrierSyncOp = std::make_unique<SyncOperation>(
              SyncOperation{SyncOperation::TYPE::PIPE_BARRIER_CUBE,
                            hivm::PIPE::PIPE_ALL, hivm::PIPE::PIPE_ALL,
                            syncIndex, nowCompound->GetIndex(), forEndIndex});
      barrierSyncOp->SetDepSyncIRIndex(frontCompound->GetIndex());
      syncIR[insertWaitId]->pipeBefore.push_back(barrierSyncOp.get());
      barrierSyncOp->SetSyncIRIndex(insertWaitId);
      SmallVector<std::unique_ptr<SyncOperation>> newSync;
      newSync.emplace_back(std::move(barrierSyncOp));
      syncOperations.emplace_back(std::move(newSync));
    } else {
      // currently it is only expected to have cube-cube pipe-barrier-cube
      // sync-operations. as vecto-vector is not supported yet.
      llvm_unreachable("unsupported vector-vector sync mode");
    }
  } else {
    std::unique_ptr<SyncOperation, std::default_delete<SyncOperation>>
        syncBlockSetOp = std::make_unique<SyncOperation>(SyncOperation{
            SyncOperation::TYPE::SYNC_BLOCK_SET, frontCompound->kPipeValue,
            hivm::PIPE::PIPE_S, syncIndex, insertSetId, forEndIndex});
    auto syncBlockWaitOp = syncBlockSetOp->GetMatchSync(insertWaitId);

    auto parentLoops =
        getBlockSyncMultibufferEnabledLoops(nowCompound, frontCompound);
    if (parentLoops.has_value()) {
      auto [nowParentLoop, frontParentLoop] = parentLoops.value();
      std::optional<int> eventIdNumOpt =
          getBlockSyncOpEventIdNum(nowParentLoop, frontParentLoop);
      assert(eventIdNumOpt.has_value());
      int eventIdNum = eventIdNumOpt.value();
      syncBlockSetOp->eventIdNum = eventIdNum;
      syncBlockWaitOp->eventIdNum = eventIdNum;
      syncBlockSetOp->block_sync_event_value =
          frontParentLoop.getSingleInductionVar().value();
      syncBlockWaitOp->block_sync_event_value =
          nowParentLoop.getSingleInductionVar().value();
    } else {
      UpdateBackSyncMultiBufferInfo(syncBlockSetOp.get(), syncBlockWaitOp.get(),
                                    depBaseMemInfosVec, forEndIndex);
    }
    // set the core-type info.
    syncBlockSetOp->syncCoreType = frontCompound->compoundCoreType;
    syncBlockWaitOp->syncCoreType = nowCompound->compoundCoreType;
    // only wait/core-type can be CUBE_AND_VECTOR, set/core-type cannot.
    assert(syncBlockSetOp->syncCoreType != TCoreType::CUBE_AND_VECTOR);
    // if wait/core-type is CUBE_AND_VECTOR, change it to the other type.
    if (syncBlockWaitOp->syncCoreType == TCoreType::CUBE_AND_VECTOR) {
      if (syncBlockSetOp->syncCoreType == TCoreType::CUBE) {
        syncBlockWaitOp->syncCoreType = TCoreType::VECTOR;
      } else if (syncBlockSetOp->syncCoreType == TCoreType::VECTOR) {
        syncBlockWaitOp->syncCoreType = TCoreType::CUBE;
      }
    }
    syncIR[insertSetId]->pipeAfter.push_back(syncBlockSetOp.get());
    syncIR[insertWaitId]->pipeBefore.push_back(syncBlockWaitOp.get());
    SmallVector<std::unique_ptr<SyncOperation>> newSync;
    newSync.emplace_back(std::move(syncBlockSetOp));
    newSync.emplace_back(std::move(syncBlockWaitOp));
    syncOperations.emplace_back(std::move(newSync));
  }
  syncIndex++;
  assert(syncOperations.size() == syncIndex);
}

void SyncAnalyzer::ChangeToVirtualMTE2IfNeed(
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound, hivm::PIPE &nowPipe,
    hivm::PIPE &frontPipe, DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  auto frontMmadL1Op = dyn_cast<hivm::MmadL1Op>(frontCompound->elementOp);
  auto nowMmadL1Op = dyn_cast<hivm::MmadL1Op>(nowCompound->elementOp);
  if (nowPipe == hivm::PIPE::PIPE_MTE2 && frontPipe == hivm::PIPE::PIPE_MTE1 &&
      frontMmadL1Op) {
    ChangeNowPipeToVirtualMTE2(nowPipe, depBaseMemInfosVec, frontMmadL1Op);
  }
  if (nowPipe == hivm::PIPE::PIPE_MTE1 && frontPipe == hivm::PIPE::PIPE_MTE2 &&
      nowMmadL1Op) {
    ChangeFrontPipeToVirtualMTE2(frontPipe, depBaseMemInfosVec, nowMmadL1Op);
  }
}

void SyncAnalyzer::ChangeNowPipeToVirtualMTE2(
    hivm::PIPE &nowPipe, DepBaseMemInfoPairVec &depBaseMemInfosVec,
    hivm::MmadL1Op mmadL1Op) const {
  Value L1A = mmadL1Op.getA();
  Value L1B = mmadL1Op.getB();
  for (auto &depValue : depBaseMemInfosVec) {
    if (depValue.second->baseBuffer == L1A) {
      nowPipe = hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A;
    } else if (depValue.second->baseBuffer == L1B) {
      nowPipe = hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B;
    }
  }
}

void SyncAnalyzer::ChangeFrontPipeToVirtualMTE2(
    hivm::PIPE &frontPipe, DepBaseMemInfoPairVec &depBaseMemInfosVec,
    hivm::MmadL1Op mmadL1Op) const {
  Value L1A = mmadL1Op.getA();
  Value L1B = mmadL1Op.getB();
  for (auto depValue : depBaseMemInfosVec) {
    if (depValue.first->baseBuffer == L1A) {
      frontPipe = hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A;
    } else if (depValue.first->baseBuffer == L1B) {
      frontPipe = hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B;
    }
  }
}

InstanceElement *
SyncAnalyzer::getParentScope(InstanceElement *instanceElement) {
  for (int i = static_cast<int>(instanceElement->GetIndex()); i >= 0; i--) {
    if (auto *loopOp = dyn_cast<LoopInstanceElement>(syncIR[i].get())) {
      auto loopKind = loopOp->getLoopKind();
      if (loopKind == KindOfLoop::LOOP_BEGIN) {
        return loopOp;
      }
      i = static_cast<int>(loopOp->beginId);
    } else if (auto *branchOp =
                   dyn_cast<BranchInstanceElement>(syncIR[i].get())) {
      auto branchKind = branchOp->getBranchKind();
      if (branchKind == KindOfBranch::IF_BEGIN ||
          branchKind == KindOfBranch::ELSE_BEGIN) {
        return branchOp;
      }
      i = static_cast<int>(branchOp->beginId);
    }
  }
  return nullptr;
}

PlaceHolderInstanceElement *
SyncAnalyzer::checkUnlikelyScope(CompoundInstanceElement *nowCompound,
                                 CompoundInstanceElement *frontCompound) {
  auto *parentScope = getParentScope(frontCompound);
  if (parentScope == nullptr) {
    return nullptr;
  }
  if (parentScope->elementOp == nullptr) {
    return nullptr;
  }
  if (auto *branchOp = dyn_cast<BranchInstanceElement>(parentScope)) {
    if (auto ifOp = dyn_cast<scf::IfOp>(parentScope->elementOp)) {
      if (ifOp->hasAttrOfType<UnitAttr>(hivm::UnlikelyConditionAttr::name)) {
        auto *placeHolder = dyn_cast<PlaceHolderInstanceElement>(
            syncIR[branchOp->endId - 1].get());
        assert(placeHolder != nullptr);
        return placeHolder;
      }
    }
  }
  return nullptr;
}

void SyncAnalyzer::insertUnlikelySyncOperation(
    PlaceHolderInstanceElement *placeHolder,
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec,
    const std::optional<unsigned> &forEndIndex) {
  auto nowPipe = nowCompound->kPipeValue;
  auto frontPipe = frontCompound->kPipeValue;
  ChangeToVirtualMTE2IfNeed(nowCompound, frontCompound, nowPipe, frontPipe,
                            depBaseMemInfosVec);
  if (nowPipe == frontPipe) {
    unsigned insertBarrierId = placeHolder->GetIndex();
    auto barrierSyncOp = std::make_unique<SyncOperation>(
        SyncOperation{SyncOperation::TYPE::PIPE_BARRIER, frontPipe, nowPipe,
                      syncIndex, nowCompound->GetIndex(), forEndIndex});
    barrierSyncOp->SetDepSyncIRIndex(frontCompound->GetIndex());
    syncIR[insertBarrierId]->pipeBefore.push_back(barrierSyncOp.get());
    barrierSyncOp->SetSyncIRIndex(insertBarrierId);
    SmallVector<std::unique_ptr<SyncOperation>> newSync;
    newSync.emplace_back(std::move(barrierSyncOp));
    syncOperations.emplace_back(std::move(newSync));
  } else {
    unsigned insertWaitId = placeHolder->GetIndex();
    unsigned insertSetId = frontCompound->GetIndex();
    auto setFlag = std::make_unique<SyncOperation>(
        SyncOperation{SyncOperation::TYPE::SET_EVENT, frontPipe, nowPipe,
                      syncIndex, insertSetId, forEndIndex});
    auto waitFlag = setFlag->GetMatchSync(insertWaitId);
    UpdateBackSyncMultiBufferInfo(setFlag.get(), waitFlag.get(),
                                  depBaseMemInfosVec, forEndIndex);
    syncIR[insertSetId]->pipeAfter.push_back(setFlag.get());
    syncIR[insertWaitId]->pipeBefore.push_back(waitFlag.get());
    SmallVector<std::unique_ptr<SyncOperation>> newSync;
    newSync.emplace_back(std::move(setFlag));
    newSync.emplace_back(std::move(waitFlag));
    syncOperations.emplace_back(std::move(newSync));
  }
  syncIndex++;
  assert(syncOperations.size() == syncIndex);
}

void SyncAnalyzer::InsertSyncOperation(
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec,
    const std::optional<unsigned> &forEndIndex) {
  if (auto *placeHolder = checkUnlikelyScope(nowCompound, frontCompound)) {
    insertUnlikelySyncOperation(placeHolder, nowCompound, frontCompound,
                                depBaseMemInfosVec, forEndIndex);
    return;
  }

  auto nowPipe = nowCompound->kPipeValue;
  auto frontPipe = frontCompound->kPipeValue;
  ChangeToVirtualMTE2IfNeed(nowCompound, frontCompound, nowPipe, frontPipe,
                            depBaseMemInfosVec);
  if (nowPipe == frontPipe) {
    unsigned insertBarrierId = nowCompound->GetIndex();
    auto barrierSyncOp = std::make_unique<SyncOperation>(
        SyncOperation{SyncOperation::TYPE::PIPE_BARRIER, frontPipe, nowPipe,
                      syncIndex, nowCompound->GetIndex(), forEndIndex});
    barrierSyncOp->SetDepSyncIRIndex(frontCompound->GetIndex());
    syncIR[insertBarrierId]->pipeBefore.push_back(barrierSyncOp.get());
    barrierSyncOp->SetSyncIRIndex(insertBarrierId);
    SmallVector<std::unique_ptr<SyncOperation>> newSync;
    newSync.emplace_back(std::move(barrierSyncOp));
    syncOperations.emplace_back(std::move(newSync));
  } else {
    unsigned insertWaitId = nowCompound->GetIndex();
    unsigned insertSetId = frontCompound->GetIndex();
    auto setFlag = std::make_unique<SyncOperation>(
        SyncOperation{SyncOperation::TYPE::SET_EVENT, frontPipe, nowPipe,
                      syncIndex, insertSetId, forEndIndex});
    auto waitFlag = setFlag->GetMatchSync(insertWaitId);
    UpdateBackSyncMultiBufferInfo(setFlag.get(), waitFlag.get(),
                                  depBaseMemInfosVec, forEndIndex);
    syncIR[insertSetId]->pipeAfter.push_back(setFlag.get());
    syncIR[insertWaitId]->pipeBefore.push_back(waitFlag.get());
    SmallVector<std::unique_ptr<SyncOperation>> newSync;
    newSync.emplace_back(std::move(setFlag));
    newSync.emplace_back(std::move(waitFlag));
    syncOperations.emplace_back(std::move(newSync));
  }
  syncIndex++;
  assert(syncOperations.size() == syncIndex);
}

void SyncAnalyzer::insertUnitFlagEnabledSyncOperations(
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound,
    const std::optional<unsigned> &forEndIndex, UNIT_FLAG frontUnitFlagMode,
    UNIT_FLAG nowUnitFlagMode) {
  auto nowPipe = nowCompound->kPipeValue;
  auto frontPipe = frontCompound->kPipeValue;
  unsigned insertWaitId = nowCompound->GetIndex();
  unsigned insertSetId = frontCompound->GetIndex();

  auto setFlag = std::make_unique<SyncOperation>(
      SyncOperation{SyncOperation::TYPE::SET_EVENT, frontPipe, nowPipe,
                    syncIndex, insertSetId, forEndIndex});
  auto waitFlag = setFlag->GetMatchSync(insertWaitId);

  if (forEndIndex.has_value()) {
    auto *setCompound =
        dyn_cast<CompoundInstanceElement>(syncIR[insertSetId].get());
    auto *waitCompound =
        dyn_cast<CompoundInstanceElement>(syncIR[insertWaitId].get());
    if (!setCompound || !waitCompound ||
        setCompound->elementOp != frontCompound->elementOp ||
        waitCompound->elementOp != nowCompound->elementOp) {
      llvm_unreachable("now/front compounds indexes don't match");
    }
    setCompound->unitFlagModeAsSet = frontUnitFlagMode;
    waitCompound->unitFlagModeAsWait = nowUnitFlagMode;
    setCompound->linkedUnitFlagCompAsSet = waitCompound;
    waitCompound->linkedUnitFlagCompAsWait = setCompound;
  }
  frontCompound->unitFlagModeAsSet = frontUnitFlagMode;
  nowCompound->unitFlagModeAsWait = nowUnitFlagMode;
  frontCompound->linkedUnitFlagCompAsSet = nowCompound;
  nowCompound->linkedUnitFlagCompAsWait = frontCompound;

  setFlag->uselessSync = true;
  waitFlag->uselessSync = true;

  syncIR[insertSetId]->pipeAfter.push_back(setFlag.get());
  syncIR[insertWaitId]->pipeBefore.push_back(waitFlag.get());
  SmallVector<std::unique_ptr<SyncOperation>> newSync;
  newSync.emplace_back(std::move(setFlag));
  newSync.emplace_back(std::move(waitFlag));
  syncOperations.emplace_back(std::move(newSync));
  syncIndex++;
  assert(syncOperations.size() == syncIndex);
}

std::optional<std::pair<UNIT_FLAG, UNIT_FLAG>>
SyncAnalyzer::checkUnitFlagPatterns(
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound,
    const std::optional<unsigned> &forEndIndex) {
  if (!enableUnitFlag) {
    return {};
  }
  if (frontCompound->unitFlagModeAsSet != UNIT_FLAG::DISABLED ||
      nowCompound->unitFlagModeAsWait != UNIT_FLAG::DISABLED) {
    return {};
  }
  if (forEndIndex.has_value()) {
    if (nowCompound->elementOp->getParentOp() !=
        syncIR[forEndIndex.value()]->elementOp) {
      return {};
    }
  }
  if (checkMemoryConflictBetweenExclusive(nowCompound, frontCompound,
                                          forEndIndex)) {
    return {};
  }
  if (frontCompound->unitFlagModeAsWait == UNIT_FLAG::ENABLED_WITH_UPDATE ||
      frontCompound->unitFlagModeAsWait == UNIT_FLAG::ENABLED_ONLY_FIRST_ITER ||
      frontCompound->unitFlagModeAsWait ==
          UNIT_FLAG::ENABLED_ONLY_FIRST_AND_LAST_ITERS) {
    // fixpipe expect unit-flag to be enabled in a mmadl1 operation before it,
    // so by this condition, we are making sure to not link a fixpipe
    // operation with an operation after it if it was not linked with an
    // operation before it
    if (auto unitFlagMode =
            checkMmadl1FixpipeUnitFlagPattern(nowCompound, frontCompound)) {
      return unitFlagMode;
    }
    if (auto unitFlagMode = checkMmadl1FixpipeSingleForLoopUnitFlagPattern(
            nowCompound, frontCompound, /*op1IsFrontCompound=*/false)) {
      return unitFlagMode;
    }
  }
  if (auto unitFlagMode =
          checkMmadl1FixpipeUnitFlagPattern(frontCompound, nowCompound)) {
    return unitFlagMode;
  }
  if (auto unitFlagMode = checkMmadl1FixpipeSingleForLoopUnitFlagPattern(
          frontCompound, nowCompound, /*op1IsFrontCompound=*/true)) {
    return unitFlagMode;
  }
  return {};
}

std::optional<std::pair<UNIT_FLAG, UNIT_FLAG>>
SyncAnalyzer::checkMmadl1FixpipeUnitFlagPattern(
    CompoundInstanceElement *op1, CompoundInstanceElement *op2) const {
  auto mmadl1Op = dyn_cast<hivm::MmadL1Op>(op1->elementOp);
  auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(op2->elementOp);
  if (!fixpipeOp || !mmadl1Op) {
    return {};
  }
  if (op1->kPipeValue != PIPE::PIPE_M) {
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

std::optional<std::pair<UNIT_FLAG, UNIT_FLAG>>
SyncAnalyzer::checkMmadl1FixpipeSingleForLoopUnitFlagPattern(
    CompoundInstanceElement *op1, CompoundInstanceElement *op2,
    bool op1IsFrontCompound) const {
  auto mmadl1Op = dyn_cast<hivm::MmadL1Op>(op1->elementOp);
  auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(op2->elementOp);
  if (!fixpipeOp || !mmadl1Op) {
    return {};
  }
  if (op1->kPipeValue != PIPE::PIPE_M) {
    return {};
  }
  if (fixpipeOp.getSrc() != mmadl1Op.getC()) {
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
      return op1IsFrontCompound
                 ? std::make_pair(UNIT_FLAG::ENABLED_ONLY_LAST_ITER,
                                  UNIT_FLAG::ENABLED_WITH_UPDATE)
                 : std::make_pair(UNIT_FLAG::ENABLED_WITH_UPDATE,
                                  UNIT_FLAG::ENABLED_ONLY_FIRST_ITER);
    }
  } else if (fixpipeOpForOp) {
    if (fixpipeOpForOp->getParentRegion() == mmadl1Op->getParentRegion()) {
      return op1IsFrontCompound
                 ? std::make_pair(UNIT_FLAG::ENABLED_WITH_UPDATE,
                                  UNIT_FLAG::ENABLED_ONLY_FIRST_ITER)
                 : std::make_pair(UNIT_FLAG::ENABLED_ONLY_LAST_ITER,
                                  UNIT_FLAG::ENABLED_WITH_UPDATE);
    }
  }
  return {};
}

bool SyncAnalyzer::checkMemoryConflictBetweenExclusive(
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound,
    const std::optional<unsigned> &forEndIndex) {
  auto checkMemoryConflictRangeExclusive =
      [this, nowCompound, frontCompound](unsigned l, unsigned r) {
        if (l > r) {
          llvm_unreachable("expected l <= r");
        }
        for (unsigned i = l + 1; i < r; i++) {
          if (auto *currentCompound =
                  dyn_cast<CompoundInstanceElement>(syncIR[i].get())) {
            DepBaseMemInfoPairVec depBaseMemInfosVec;
            if (IsMemInfoHasDependency(currentCompound, nowCompound,
                                       depBaseMemInfosVec) ||
                IsMemInfoHasDependency(currentCompound, frontCompound,
                                       depBaseMemInfosVec)) {
              return true;
            }
          }
        }
        return false;
      };

  unsigned nowIndex = nowCompound->GetIndex();
  unsigned frontIndex = frontCompound->GetIndex();
  if (frontIndex <= nowIndex) {
    return checkMemoryConflictRangeExclusive(frontIndex, nowIndex);
  } else {
    if (!forEndIndex.has_value()) {
      llvm_unreachable("expected backward-sync mode.");
    }
    auto *forInst =
        dyn_cast<LoopInstanceElement>(syncIR[forEndIndex.value()].get());
    if (!forInst || !(forInst->beginId < nowIndex && nowIndex < frontIndex &&
                      frontIndex < forInst->endId)) {
      llvm_unreachable("expected indexs to be within a for-op region.");
    }
    if (checkMemoryConflictRangeExclusive(forInst->beginId, nowIndex) ||
        checkMemoryConflictRangeExclusive(frontIndex, forInst->endId)) {
      return true;
    }
  }
  return false;
}

void SyncAnalyzer::UpdateBackSyncMultiBufferInfo(
    SyncOperation *setFlag, SyncOperation *waitFlag,
    DepBaseMemInfoPairVec &depBaseMemInfosVec,
    const std::optional<unsigned> &forEndIndex) {
  if (!forEndIndex.has_value())
    return;
  // For backward dependency, it is necessary to handle multiBuffer.
  // note: always treat as single buffer by default for forward dependency.
  int touchEventIdNum = GetEventIdNum(depBaseMemInfosVec);
  auto commonParentLoop =
      GetCommonParentLoop(depBaseMemInfosVec, touchEventIdNum);
  if (touchEventIdNum > 1 && commonParentLoop.has_value()) {
    setFlag->eventIdNum = touchEventIdNum;
    waitFlag->eventIdNum = touchEventIdNum;
    setFlag->lowestCommonAncestorBuffer = commonParentLoop->first;
    waitFlag->lowestCommonAncestorBuffer = commonParentLoop->first;
  } else {
    // TODO:
    // Currently not supporting PointerCastOp in different loops under db.
    setFlag->eventIdNum = 1;
    waitFlag->eventIdNum = 1;
  }
}

int SyncAnalyzer::GetEventIdNum(
    const DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  int singleBufferNum = 1;
  int doubleBufferNum = 2;
  SmallVector<int> multiBufferNums;
  for (auto &depBaseMemInfos : depBaseMemInfosVec) {
    assert(depBaseMemInfos.first != nullptr &&
           depBaseMemInfos.second != nullptr);
    int aBaseAddressesSize = static_cast<int>(depBaseMemInfos.first->baseAddresses.size());
    int bBaseAddressesSize = static_cast<int>(depBaseMemInfos.second->baseAddresses.size());
    if ((aBaseAddressesSize == bBaseAddressesSize && aBaseAddressesSize == 1) ||
        aBaseAddressesSize != bBaseAddressesSize) {
      return singleBufferNum;
    }

    for (int i = 0; i < aBaseAddressesSize; i++) {
      for (int j = 0; j < bBaseAddressesSize; j++) {
        bool overlap = memAnalyzer.isBufferOverlap(
            depBaseMemInfos.first, depBaseMemInfos.second, i, j);
        if ((i == j && !overlap) || (i != j && overlap)) {
          return singleBufferNum;
        }
      }
    }
    multiBufferNums.push_back(aBaseAddressesSize);
  }

  for (auto num : multiBufferNums) {
    if (num != 2) {
      // TODO:: Currently only processing 2buffer
      return singleBufferNum;
    }
  }
  return doubleBufferNum;
}

Value SyncAnalyzer::GetLowestCommonAncestorBuffer(
    const DepBaseMemInfoPairVec &depBaseMemInfosVec, int eventIdNum) {
  if (eventIdNum != 2) {
    // Not a 2buffer, just return directly.
    return nullptr;
  }
  // All touch dep buffers.
  auto touchedBuffer = GetMemInfoBuffers(depBaseMemInfosVec);
  if (touchedBuffer.empty()) {
    return nullptr;
  }
  Value selectedBuffer = touchedBuffer.front();
  if (touchedBuffer.size() == 1) {
    return selectedBuffer;
  }
  // Get the buffer defined on the outermost loop.
  for (auto *it = touchedBuffer.begin() + 1; it != touchedBuffer.end(); ++it) {
    auto *firstMemoryOp = selectedBuffer.getDefiningOp();
    assert(firstMemoryOp != nullptr);
    assert(IsMemAllocOp(firstMemoryOp));
    LoopLikeOpInterface firstParentLoop =
        firstMemoryOp->getParentOfType<LoopLikeOpInterface>();
    auto *secondMemoryOp = (*it).getDefiningOp();
    assert(secondMemoryOp != nullptr);
    assert(IsMemAllocOp(secondMemoryOp));
    LoopLikeOpInterface secondParentLoop =
        secondMemoryOp->getParentOfType<LoopLikeOpInterface>();
    if (firstParentLoop == nullptr || secondParentLoop == nullptr) {
      return nullptr;
    }
    if (secondParentLoop->isAncestor(firstParentLoop)) {
      selectedBuffer = *it;
    } else if (firstParentLoop->isAncestor(secondParentLoop)) {
      continue;
    } else {
      return nullptr;
    }
  }
  return selectedBuffer;
}

std::optional<std::pair<Value, scf::ForOp>> SyncAnalyzer::GetCommonParentLoop(
    const DepBaseMemInfoPairVec &depBaseMemInfosVec, int eventIdNum) {
  if (eventIdNum != 2) {
    // Not a 2buffer, just return directly.
    return {};
  }
  // All touch dep buffers
  auto touchedBuffer = GetMemInfoBuffers(depBaseMemInfosVec);
  if (touchedBuffer.empty()) {
    return {};
  }
  Value retBuffer = touchedBuffer.front();
  auto *allocOp = retBuffer.getDefiningOp();
  if (allocOp == nullptr) {
    return {};
  }
  scf::ForOp retParentLoop = allocOp->getParentOfType<scf::ForOp>();
  if (!retParentLoop) {
    return {};
  }
  for (auto &buffer : touchedBuffer) {
    auto *curAllocOp = buffer.getDefiningOp();
    assert(curAllocOp != nullptr);
    assert(IsMemAllocOp(curAllocOp));
    auto curParentLoop = curAllocOp->getParentOfType<scf::ForOp>();
    if (curParentLoop == nullptr || retParentLoop != curParentLoop) {
      return {};
    }
  }
  return std::make_pair(retBuffer, retParentLoop);
}

bool SyncAnalyzer::IsMemAllocOp(Operation *op) const {
  return isa<PointerCastOp, bishengir::memref_ext::AllocWorkspaceOp>(op);
}

SmallVector<Value> SyncAnalyzer::GetMemInfoBuffers(
    const DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  // All touch dep buffers.
  DenseSet<Value> touchedBuffer;
  for (auto &depBaseMemInfos : depBaseMemInfosVec) {
    assert(depBaseMemInfos.first != nullptr &&
           depBaseMemInfos.second != nullptr);
    // Update the most original buffer in alias.
    Value originalFirstBuffer;
    Value originalSecondBuffer;
    if (syncAnalysisMode == SyncAnalysisMode::BLOCKSYNC) {
      assert(depBaseMemInfos.first->allocWorkspaceOp.has_value());
      assert(depBaseMemInfos.second->allocWorkspaceOp.has_value());
      originalFirstBuffer =
          depBaseMemInfos.first->allocWorkspaceOp.value()->getResult(0);
      originalSecondBuffer =
          depBaseMemInfos.second->allocWorkspaceOp.value()->getResult(0);
    } else {
      originalFirstBuffer = depBaseMemInfos.first->rootBuffer;
      originalSecondBuffer = depBaseMemInfos.second->rootBuffer;
    }
    auto iterFirst = buffer2ParentAliasBuffer.find(originalFirstBuffer);
    if (iterFirst != buffer2ParentAliasBuffer.end()) {
      originalFirstBuffer = iterFirst->second;
    }
    touchedBuffer.insert(originalFirstBuffer);
    auto iterSecond = buffer2ParentAliasBuffer.find(originalSecondBuffer);
    if (iterSecond != buffer2ParentAliasBuffer.end()) {
      originalSecondBuffer = iterSecond->second;
    }
    touchedBuffer.insert(originalSecondBuffer);
  }
  SmallVector<Value> result(touchedBuffer.begin(), touchedBuffer.end());
  return result;
}
