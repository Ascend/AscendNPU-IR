//===- SyncAnalysis.h ----Dependency analysis and insert sync -------------===//
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
#ifndef BISHENGIR_SYNCANALYSIS_H
#define BISHENGIR_SYNCANALYSIS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/MemoryDependentAnalyzer.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace hivm {

/// Records data during insertion synchronization,
/// which is used to transfer dependency elimination
/// and path collection.
struct SyncRecord {
  /// Record the sync pipe that have already been waited for.
  std::array<bool, getPipeNum()> alreadySync{false};

  /// Record the pairing status of setwait.
  DenseMap<int, bool> syncFinder;
};

using SyncRecordList = std::array<SyncRecord, MAX_MULTI_BUFFER_NUM>;

class SyncAnalyzer {
public:
  SyncAnalyzer(SyncIRs &syncIR, MemoryDependentAnalyzer &memDepAnalyzer,
               SyncOperations &syncOperations, func::FuncOp func,
               SyncAnalysisMode syncAnalysisMode, bool enableUnitFlag,
               bool assumeAliveLoops)
      : syncIR(syncIR), memAnalyzer(memDepAnalyzer),
        syncOperations(syncOperations), func_(func),
        syncAnalysisMode(syncAnalysisMode), enableUnitFlag(enableUnitFlag),
        assumeAliveLoops(assumeAliveLoops) {}

  ~SyncAnalyzer() = default;

  /// Plan entrance, inject sync in the syncIR.
  void Plan(bool insertBarAllAtLast = true);

  /// Set Buffer2InplaceBuffer.
  void SetBuffer2ParentAliasBuffer(
      const DenseMap<Value, Value> &buf2ParentAliasBuffer) {
    buffer2ParentAliasBuffer = buf2ParentAliasBuffer;
  }

  // Shared sync-operation object used to make mmadl1 operations share the same
  // BackPipeMPipeMTE1DBEvent event-ids. Other wise, different mmadl1 operations
  // could run at the same time using different event-ids and cause r/w
  // conflicts.
  DenseMap<int, SyncOperation *> BwdPipeMPipeMTE1SyncPtr;

private:
  /// Save the Global syncIR.
  SyncIRs &syncIR;

  /// Save the baseMemInfo entity and determines memory conflicts.
  MemoryDependentAnalyzer &memAnalyzer;

  /// Save the Global Sync Memory.
  SyncOperations &syncOperations;

  func::FuncOp func_;

  SyncAnalysisMode syncAnalysisMode;

  /// Global id for sync id.
  unsigned syncIndex{0};

  /// Record the relationship between buffer and alias by buffer.
  DenseMap<Value, Value> buffer2ParentAliasBuffer;

  bool enableUnitFlag{false};

  bool assumeAliveLoops{false};

private:
  /// Inset backward sync with LoopInstanceElement's end.
  void DealWithLoopSync(LoopInstanceElement *nowElement);

  /// Inset basic sync with CompoundInstanceElement.
  void DealWithCompoundSync(CompoundInstanceElement *nowCompound);

  /// Inset basic sync in between loop.
  void InsertBackForSync(CompoundInstanceElement *nowCompound,
                         SyncIRs &backSyncIr,
                         const LoopInstanceElement *loopElement);

  /// Process synchronization insert of CompoundInstanceElement and
  /// BranchInstanceElement nodes.
  void InsertSeqSync(CompoundInstanceElement *nowCompound, SyncIRs &syncElement,
                     int begin, int end, SyncRecordList &syncRecordList,
                     const std::optional<unsigned> &forEndIndex);

  /// Check sync between nowCompound and frontCompound, add sync to nowCompound
  /// and front element.
  void InsertSync(CompoundInstanceElement *nowCompound,
                  CompoundInstanceElement *frontCompound,
                  SyncRecordList &syncRecordList,
                  const std::optional<unsigned> &forEndIndex);

  /// Handle the case when it's a block-sync run and the input is a cube-cube
  /// kernal by inserting barrier.all between fixpipe/load pairs.
  bool isBlockCubeCube(const CompoundInstanceElement *nowCompound,
                       const CompoundInstanceElement *frontCompound) const;

  /// No need to insert synchronization scene.
  bool IsNoNeedToInsertSync(const CompoundInstanceElement *nowCompound,
                            const CompoundInstanceElement *frontCompound,
                            bool isBackwardDep) const;

  /// mem dependency analysis and insertion synchronization.
  void MemAnalyze(CompoundInstanceElement *nowCompound,
                  CompoundInstanceElement *frontCompound,
                  SyncRecordList &syncRecordList,
                  const std::optional<unsigned> &forEndIndex);

  /// update syncRecord inform.
  void UpdateAlreadySync(const SyncOps &syncVector,
                         SyncRecordList &syncRecordList,
                         hivm::PIPE nowPipeValue);

  /// update sync Record.
  void UpdateSyncRecord(const SyncOperation *sync, SyncRecord &syncRecord,
                        hivm::PIPE nowPipeValue);

  /// Processing scenarios for sync various inserts of BranchInstanceElement.
  unsigned
  InsertBranchSync(unsigned index, CompoundInstanceElement *nowCompound,
                   unsigned begin, BranchInstanceElement *branchElement,
                   SyncIRs &syncElement, SyncRecordList &syncRecordList,
                   const std::optional<unsigned> &forEndIndex);

  /// Processing scenarios for sync various inserts of LoopInstanceElement.
  unsigned InsertLoopSync(unsigned index, CompoundInstanceElement *nowCompound,
                          unsigned begin, LoopInstanceElement *loopElement,
                          SyncIRs &syncElement, SyncRecordList &syncRecordList,
                          const std::optional<int> &forEndIndex);

  /// Merge SyncRecord.
  void MergeAlreadySync(SyncRecordList &syncRecordList,
                        const SyncRecordList &syncRecordIfList,
                        const SyncRecordList &syncRecordElseList);

  /// Insert synchronization instructions for dependent elements.
  void InsertSyncOperation(CompoundInstanceElement *nowCompound,
                           CompoundInstanceElement *frontCompound,
                           DepBaseMemInfoPairVec &depBaseMemInfosVec,
                           const std::optional<unsigned> &forEndIndex);

  InstanceElement *getParentScope(InstanceElement *instanceElement);

  PlaceHolderInstanceElement *
  CheckUnlikelyScope(CompoundInstanceElement *nowCompound,
                     CompoundInstanceElement *frontCompound);

  /// Insert synchronization instructions for dependent elements when the front
  /// compound is inside an "unlikely" scope.
  void InsertUnlikelySyncOperation(PlaceHolderInstanceElement *placeHolder,
                                   CompoundInstanceElement *nowCompound,
                                   CompoundInstanceElement *frontCompound,
                                   DepBaseMemInfoPairVec &depBaseMemInfosVec,
                                   const std::optional<unsigned> &forEndIndex);

  std::optional<std::pair<LoopLikeOpInterface, LoopLikeOpInterface>>
  getBlockSyncMultibufferEnabledLoops(CompoundInstanceElement *nowCompound,
                                      CompoundInstanceElement *frontCompound);

  /// Get event id number and handle multibuffer-unroll case.
  std::optional<int>
  getBlockSyncOpEventIdNum(LoopLikeOpInterface &nowParentLoop,
                           LoopLikeOpInterface &frontParentLoop) const;

  /// Insert block synchronization instructions for dependent elements.
  void InsertBlockSyncOperation(CompoundInstanceElement *nowCompound,
                                CompoundInstanceElement *frontCompound,
                                DepBaseMemInfoPairVec &depBaseMemInfosVec,
                                const std::optional<unsigned> &forEndIndex);

  /// Is there already sync on the link that does not require further insertion.
  bool isAlreadySync(CompoundInstanceElement *nowCompound,
                     CompoundInstanceElement *frontCompound,
                     SyncRecordList &syncRecordList, unsigned recordListIndex);

  /// Determine whether there is a dependency relationship in the current
  /// memInfos.
  bool IsMemInfoHasDependency(CompoundInstanceElement *nowCompound,
                              CompoundInstanceElement *frontCompound,
                              DepBaseMemInfoPairVec &depBaseMemInfosVec);

  /// After synchronous insertion, update SyncRecordList.
  void UpdateSyncRecordInfo(CompoundInstanceElement *frontCompound,
                            SyncRecordList &syncRecordList);

  /// Get the current event id number of dependent baseMemInfo.
  int GetEventIdNum(const DepBaseMemInfoPairVec &depBaseMemInfosVec);

  // Get all buffers mentioned in mem info pairs.
  SmallVector<Value>
  GetMemInfoBuffers(const DepBaseMemInfoPairVec &depBaseMemInfosVec);

  /// Get the lowest common ancestor buffer of dependent baseMemInfo.
  Value
  GetLowestCommonAncestorBuffer(const DepBaseMemInfoPairVec &depBaseMemInfosVec,
                                int eventIdNum);

  /// When given multiple pointer_cast operations, get their parent loop if it
  /// exists and they all under it's scope.
  std::optional<std::pair<Value, scf::ForOp>>
  GetCommonParentLoop(const DepBaseMemInfoPairVec &depBaseMemInfosVec,
                      int eventIdNum);

  /// insert last barrier all.
  void InsertLastPipeAll();

  /// Update multi buffer info for eventIdNum and lowestCommonAncestorBuffer.
  void
  UpdateBackSyncMultiBufferInfo(SyncOperation *setFlag, SyncOperation *waitFlag,
                                DepBaseMemInfoPairVec &depBaseMemInfosVec,
                                const std::optional<unsigned> &forEndIndex);
  /// is PointerCastOp or AllocWorkspaceOp.
  bool IsMemAllocOp(Operation *op) const;

  /// Change PIPE_MTE2 to VIRTUAL_PIPE_MTE2_L1A and VIRTUAL_PIPE_MTE2_L1B.
  void ChangeToVirtualMTE2IfNeed(CompoundInstanceElement *nowCompound,
                                 CompoundInstanceElement *frontCompound,
                                 hivm::PIPE &nowPipe, hivm::PIPE &frontPipe,
                                 DepBaseMemInfoPairVec &depBaseMemInfosVec);

  /// Change nowPipe MTE2 to VIRTUAL_PIPE_MTE2_L1A and VIRTUAL_PIPE_MTE2_L1B.
  void ChangeNowPipeToVirtualMTE2(hivm::PIPE &nowPipe,
                                  DepBaseMemInfoPairVec &depBaseMemInfosVec,
                                  hivm::MmadL1Op mmadL1Op) const;

  /// Change front MTE2 to VIRTUAL_PIPE_MTE2_L1A and VIRTUAL_PIPE_MTE2_L1B.
  void ChangeFrontPipeToVirtualMTE2(hivm::PIPE &frontPipe,
                                    DepBaseMemInfoPairVec &depBaseMemInfosVec,
                                    hivm::MmadL1Op mmadL1Op) const;

  /// Insert set_flag(PIPE_M, PIPE_MTE1) and wait_flag(PIPE_M, PIPE_MTE1) for
  /// Template.
  void insertMmadL1BackPipeMPipeMTE1Sync();

  /// Enable unit-flag sync mode and insert useless set/wait pair for
  /// compatibility reasons.
  void insertUnitFlagEnabledSyncOperations(
      CompoundInstanceElement *nowCompound,
      CompoundInstanceElement *frontCompound,
      const std::optional<unsigned> &forEndIndex, UNIT_FLAG frontUnitFlagMode,
      UNIT_FLAG nowUnitFlagMode);

  /// Check all patterns where unit-flag is supported.
  std::optional<std::pair<UNIT_FLAG, UNIT_FLAG>>
  checkUnitFlagPatterns(CompoundInstanceElement *nowCompound,
                        CompoundInstanceElement *frontCompound,
                        const std::optional<unsigned> &forEndIndex);

  // checkUnitFlag helper function. Check (mmadl1/fixpipe) pair.
  std::optional<std::pair<UNIT_FLAG, UNIT_FLAG>>
  checkMmadl1FixpipeUnitFlagPattern(CompoundInstanceElement *op1,
                                    CompoundInstanceElement *op2) const;

  std::optional<std::pair<UNIT_FLAG, UNIT_FLAG>>
  checkMmadl1FixpipeSingleForLoopUnitFlagPattern(CompoundInstanceElement *op1,
                                                 CompoundInstanceElement *op2,
                                                 bool op1IsFrontCompound) const;

  bool checkMemoryConflictBetweenExclusive(
      CompoundInstanceElement *nowCompound,
      CompoundInstanceElement *frontCompound,
      const std::optional<unsigned> &forEndIndex);

  bool isParallelLoop(scf::ForOp forOp) const;

  bool
  checkUnderParallelLoop(const CompoundInstanceElement *nowCompound,
                         const CompoundInstanceElement *frontCompound) const;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_SYNCANALYSIS_H
