//===------------- SyncCodegen.h ----Sync information collection ---------===//
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
#ifndef BISHENGIR_SYNCCODEGEN_H
#define BISHENGIR_SYNCCODEGEN_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hivm {

/// Record the synchronization of Operation before and after.
struct SyncPipeBuild {
  SyncOps pipeBefore;
  SyncOps pipeAfter;
};

/// Sync and Template Interaction.
struct SyncTemplateInter {
  SyncTemplateInter() = default;
  SyncTemplateInter(Value MmadL1WaitL1AEvent, Value MmadL1WaitL1BEvent,
                    Value L1AWaitMmadL1Event, Value L1B2WaitMmadL1Event,
                    Value KLoopDBCond, Value BackPipeMPipeMTE1DBEvent0,
                    Value BackPipeMPipeMTE1DBEvent1)
      : MmadL1WaitL1AEvent(MmadL1WaitL1AEvent),
        MmadL1WaitL1BEvent(MmadL1WaitL1BEvent),
        L1AWaitMmadL1Event(L1AWaitMmadL1Event),
        L1B2WaitMmadL1Event(L1B2WaitMmadL1Event), KLoopDBCond(KLoopDBCond),
        BackPipeMPipeMTE1DBEvent0(BackPipeMPipeMTE1DBEvent0),
        BackPipeMPipeMTE1DBEvent1(BackPipeMPipeMTE1DBEvent1) {}

  Value MmadL1WaitL1AEvent;
  Value MmadL1WaitL1BEvent;
  Value L1AWaitMmadL1Event;
  Value L1B2WaitMmadL1Event;
  Value KLoopDBCond;
  Value BackPipeMPipeMTE1DBEvent0;
  Value BackPipeMPipeMTE1DBEvent1;
};

class SyncCodegen {
public:
  SyncCodegen(SyncIRs &syncIR, func::FuncOp func,
              SyncAnalysisMode syncAnalysisMode)
      : syncIR(syncIR), func_(func), syncAnalysisMode(syncAnalysisMode){};

  ~SyncCodegen() = default;

  /// Build entrance, inject sync in the func.
  void Build();

private:
  /// Insert the synchronization instruction into the corresponding position.
  void SyncInsert(IRRewriter &rewriter, Operation *op, SyncOperation *sync,
                  bool beforeInsert);

  /// Insert the synchronization instruction into the corresponding position.
  void preSyncInsert(IRRewriter &rewriter, Operation *op, SyncOperation *sync);

  /// Update the synchronization required for each node to be inserted.
  void UpdateOpInsertSync(IRRewriter &rewriter);

  /// Update the synchronization required for compound element to be inserted.
  void UpdateCompoundOpInsertSync(CompoundInstanceElement *nowCompound);

  /// Update the synchronization required for for element to be inserted.
  void UpdateLoopOpInsertSync(LoopInstanceElement *nowElement);

  /// Update the synchronization required for branch element to be inserted.
  void UpdateBranchOpInsertSync(BranchInstanceElement *nowElement);

  /// Create pipe barrier sync op.
  void CreateBarrierOp(IRRewriter &rewriter, Operation *op, SyncOperation *sync,
                       bool beforeInsert);

  /// Create set_flag or wait_flag sync op for single buffer.
  void CreateSetWaitOpForSingleBuffer(IRRewriter &rewriter, Operation *op,
                                      SyncOperation *sync, bool beforeInsert);

  /// Create set_flag or wait_flag sync op for multi buffer.
  void CreateSetWaitOpForMultiBuffer(IRRewriter &rewriter, Operation *op,
                                     SyncOperation *sync, bool beforeInsert);

  /// Create sync_block_set or sync_block_wait sync op for single buffer.
  void CreateSetWaitBlockOpForSingleBuffer(IRRewriter &rewriter, Operation *op,
                                           SyncOperation *sync,
                                           bool beforeInsert);

  /// Create sync_block_set or sync_block_wait sync op for multi buffer.
  void CreateSetWaitBlockOpForMultiBuffer(IRRewriter &rewriter, Operation *op,
                                          SyncOperation *sync,
                                          bool beforeInsert);

  void addCounterToParentLoop(IRRewriter &rewriter, Operation *op, int offset);

  /// Create sync_block_all_vector or sync_block_all_cube sync op.
  void CreateBlockSyncAllOp(IRRewriter &rewriter, Operation *op,
                            SyncOperation *sync, bool beforeInsert);

  /// Create sync_block_barrier_cube or sync_block_barrier_vector sync op.
  void CreateBlockSyncBarrierOp(IRRewriter &rewriter, Operation *op,
                                const SyncOperation *sync, bool beforeInsert);

  /// Get event id select buffer.
  Value GetBufferSelected(IRRewriter &rewriter, Operation *op,
                          SyncOperation *sync);

  /// Lower synchronization instructions into the library.
  bool NeedLowerSyncToTemplate(IRRewriter &rewriter, Operation *op,
                               SyncOperation *sync, Value eventId = nullptr);

  /// Lower synchronization instructions into the library.
  bool IsNeedLowerSyncToTemplate(Operation *op,
                                 const SyncOperation *sync) const;

  /// Determine whether to synchronize instructions to lower into the library.
  void UpdateSyncTemplateInterForBackPipeMPipeMTE1DB(
      CompoundInstanceElement *nowCompound);

  /// SyncTemplateInter for initialization and library interaction.
  void InitDefaultSyncTemplateInterForMmadL1Op(IRRewriter &rewriter,
                                               hivm::MmadL1Op mmadL1Op);

  /// Update the SyncTemplateInter information for the interaction between
  /// MmadL1 and the library.
  void UpdateMmadL1SyncTemplateInter();

  void handleEnableUnitFlag(IRRewriter &rewriter,
                            CompoundInstanceElement *nowCompound) const;

private:
  /// Save the Global syncIR.
  SyncIRs &syncIR;

  func::FuncOp func_;

  SyncAnalysisMode syncAnalysisMode;

  /// The synchronization that needs to be inserted up and down corresponding to
  /// Operation.
  DenseMap<const Operation *, SyncPipeBuild> op2InsertSync;

  /// The attr type ID corresponding to the int type ID.
  DenseMap<int, hivm::EVENT> eventIdMap{
      {0, hivm::EVENT::EVENT_ID0}, {1, hivm::EVENT::EVENT_ID1},
      {2, hivm::EVENT::EVENT_ID2}, {3, hivm::EVENT::EVENT_ID3},
      {4, hivm::EVENT::EVENT_ID4}, {5, hivm::EVENT::EVENT_ID5},
      {6, hivm::EVENT::EVENT_ID6}, {7, hivm::EVENT::EVENT_ID7}};

  /// Record the loop and corresponding counter buffer.
  DenseMap<LoopLikeOpInterface, Value> loop2BufferCounter;

  /// Collect sync index and corresponding event id expressions.
  DenseMap<unsigned, Value> SyncIndex2SelectBuffer;

  /// Collect sync index and corresponding event id expressions.
  DenseMap<unsigned, Value> SyncIndex2EventID;

  /// Collect MmadL1Op SyncTemplateInter info.
  DenseMap<hivm::MmadL1Op, SyncTemplateInter> mmadL12SyncTemplateInter;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_SYNCCODEGEN_H
